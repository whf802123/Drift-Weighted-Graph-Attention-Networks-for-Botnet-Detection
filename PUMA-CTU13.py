import warnings
warnings.filterwarnings("ignore")

import time
import random
import ipaddress
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'
WINDOW_SIZE = 1000
BATCH_SIZE = 100
EPOCHS_FIRST = 10
EPOCHS_INC = 2
ATTN_HEADS = 4
HIDDEN_CHANNELS = 8
CORR_THRESHOLD = 0.1
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
LR = 5e-3
WEIGHT_DECAY = 0.0
SEED = 42
ENABLE_WINDOW_ANALYSIS = True
WINDOW_METRIC_AVERAGE = "weighted"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
if "Label" not in df.columns:
    raise KeyError("Column 'Label' was not found. Please confirm that the CTU13 CSV label column is named Label.")

feature_cols = [c for c in df.columns if c not in ("num", "Label")]
if len(feature_cols) == 0:
    raise RuntimeError("No usable feature columns were found.")

labels = df["Label"].astype(int).to_numpy(dtype=np.int64)
unique_labels = np.unique(labels)
if len(unique_labels) != 2 or set(unique_labels.tolist()) != {0, 1}:
    raise RuntimeError(f"This code expects binary CTU13 labels 0/1, but found {unique_labels.tolist()}.")

label_to_id = {"Normal": 0, "Botnet": 1}
id_to_label = {0: "Normal", 1: "Botnet"}

feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

N_total = len(df)
all_idx = np.arange(N_total)

train_idx, holdout_idx = train_test_split(
    all_idx,
    test_size=VALIDATION_RATIO + TEST_RATIO,
    stratify=labels,
    random_state=SEED,
    shuffle=True,
)
val_idx, test_idx = train_test_split(
    holdout_idx,
    test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
    stratify=labels[holdout_idx],
    random_state=SEED,
    shuffle=True,
)

medians = feat_df.iloc[train_idx].median(numeric_only=True)
feat_df = feat_df.fillna(medians).fillna(0.0)

scaler = StandardScaler(with_mean=True, with_std=True)
features = np.empty_like(feat_df.values, dtype=np.float64)
features[train_idx] = scaler.fit_transform(feat_df.iloc[train_idx].values.astype(float))
features[val_idx] = scaler.transform(feat_df.iloc[val_idx].values.astype(float))
features[test_idx] = scaler.transform(feat_df.iloc[test_idx].values.astype(float))

N = len(features)
is_train = np.zeros(N, dtype=bool)
is_val = np.zeros(N, dtype=bool)
is_test = np.zeros(N, dtype=bool)
is_train[train_idx] = True
is_val[val_idx] = True
is_test[test_idx] = True

NUM_CLASSES = len(np.unique(labels[train_idx]))
if NUM_CLASSES != 2:
    raise RuntimeError("The training split does not contain both CTU13 classes.")

train_labels_only = labels[train_idx]
class_counts = np.bincount(train_labels_only, minlength=NUM_CLASSES).astype(np.float64)
class_weights_np = class_counts.sum() / np.maximum(class_counts, 1.0)
class_weights_np = class_weights_np / class_weights_np.mean()

print("CTU13 label mapping:", id_to_label)
print(
    f"Data split: train={len(train_idx)} ({TRAIN_RATIO:.0%}), "
    f"validation={len(val_idx)} ({VALIDATION_RATIO:.0%}), "
    f"test={len(test_idx)} ({TEST_RATIO:.0%})"
)
print(f"Samples: total={N_total}, train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
print(f"Features: {len(feature_cols)}")


def _safe_row_corrcoef(x_np):
    if x_np.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if x_np.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(x_np)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

def build_graph(x_np, allow_cross=True, current_count=None):
    n = x_np.shape[0]
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    corr = _safe_row_corrcoef(x_np)
    adj = np.abs(corr) >= CORR_THRESHOLD
    np.fill_diagonal(adj, False)
    if not allow_cross and current_count is not None and current_count < n:
        adj[:current_count, current_count:] = False
        adj[current_count:, :current_count] = False
        adj[current_count:, current_count:] = False
    src, dst = np.where(adj)
    self_nodes = np.arange(n, dtype=np.int64)
    src = np.concatenate([src.astype(np.int64), self_nodes])
    dst = np.concatenate([dst.astype(np.int64), self_nodes])
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=DEVICE)

def build_inductive_eval_graph(x_train_np, x_test_np):
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]
    if n_test == 0:
        return None, None

    x_eval_np = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = _safe_row_corrcoef(x_eval_np)
    src_list = []
    dst_list = []

    if n_train > 0:
        corr_tt = corr[:n_train, :n_train]
        adj_tt = np.abs(corr_tt) >= CORR_THRESHOLD
        np.fill_diagonal(adj_tt, False)
        src_tt, dst_tt = np.where(adj_tt)
        src_list.append(src_tt.astype(np.int64))
        dst_list.append(dst_tt.astype(np.int64))

        corr_train_test = corr[:n_train, n_train:]
        train_src, test_col = np.where(np.abs(corr_train_test) >= CORR_THRESHOLD)
        if train_src.size > 0:
            src_list.append(train_src.astype(np.int64))
            dst_list.append((n_train + test_col).astype(np.int64))

    self_nodes = np.arange(n_train + n_test, dtype=np.int64)
    src_list.append(self_nodes)
    dst_list.append(self_nodes)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=DEVICE)
    x_eval_tensor = torch.tensor(x_eval_np, dtype=torch.float, device=DEVICE)
    return x_eval_tensor, edge_index

class WeightedGATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_classes = num_classes
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            add_self_loops=False,
        )
        self.gcn2 = GCNConv(
            hidden_channels * heads,
            hidden_channels,
            add_self_loops=False,
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.cached_att = None

    def reset_all_parameters(self):
        self.gat1.reset_parameters()
        self.gcn2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x1, (ei_used, att_heads) = self.gat1(
            x,
            edge_index,
            return_attention_weights=True,
        )
        x1 = F.elu(x1)
        edge_weight = att_heads.mean(dim=1)
        self.cached_att = (ei_used, att_heads)
        x2 = self.gcn2(x1, ei_used, edge_weight=edge_weight)
        logits = self.classifier(x2)
        return logits, x2

def attention_heads_node_mean(ei_used, att_heads, num_nodes, heads, reference_nodes=None):
    dst = ei_used[1]
    in_sum = torch.zeros(num_nodes, heads, device=att_heads.device)
    in_cnt = torch.zeros(num_nodes, 1, device=att_heads.device)
    for h in range(heads):
        in_sum[:, h].index_add_(0, dst, att_heads[:, h])
    ones_e = torch.ones(dst.numel(), device=att_heads.device)
    in_cnt.index_add_(0, dst, ones_e.unsqueeze(-1))
    in_mean = in_sum / in_cnt.clamp(min=1.0)
    ref = in_mean[:reference_nodes] if reference_nodes is not None and reference_nodes > 0 else in_mean
    ref_mean = ref.mean(dim=0, keepdim=True)
    ref_std = ref.std(dim=0, keepdim=True)
    return (in_mean - ref_mean) / (ref_std + 1e-6)

def make_model(in_channels):
    model = WeightedGATClassifier(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        heads=ATTN_HEADS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, optimizer, criterion


METHOD_NAME = "PUMA"
PUMA_MEMORY_SIZE = 400
PUMA_MEMORY_PER_UPDATE = 40
PUMA_CLUSTER_STEPS = 3
PUMA_PROPAGATION_ALPHA = 0.5
PUMA_RETRAIN_FROM_SCRATCH = True

def _budget_by_class(y_np, total_budget, balanced=True):
    classes = np.unique(y_np)
    if len(classes) == 0 or total_budget <= 0:
        return {}
    total_budget = int(min(total_budget, len(y_np)))
    if balanced:
        base = total_budget // len(classes)
        rem = total_budget % len(classes)
        out = {int(c): min(int((y_np == c).sum()), base) for c in classes}
        remaining = total_budget - sum(out.values())
        order = sorted(classes, key=lambda c: int((y_np == c).sum()) - out[int(c)], reverse=True)
        ptr = 0
        while remaining > 0 and len(order) > 0:
            c = int(order[ptr % len(order)])
            if out[c] < int((y_np == c).sum()):
                out[c] += 1
                remaining -= 1
            ptr += 1
            if ptr > total_budget * max(1, len(order)) * 2:
                break
        return out
    counts = {int(c): int((y_np == c).sum()) for c in classes}
    n = sum(counts.values())
    out = {}
    for c in classes:
        q = max(1, int(round(total_budget * counts[int(c)] / max(n, 1))))
        out[int(c)] = min(counts[int(c)], q)
    while sum(out.values()) > total_budget:
        c = max(out, key=out.get)
        if out[c] > 1:
            out[c] -= 1
        else:
            break
    while sum(out.values()) < total_budget:
        candidates = [c for c in out if out[c] < counts[c]]
        if not candidates:
            break
        c = max(candidates, key=lambda cc: counts[cc] - out[cc])
        out[c] += 1
    return out

def puma_one_time_propagation(x_np):
    n = len(x_np)
    if n <= 1:
        return x_np.astype(np.float64, copy=True)
    corr = _safe_row_corrcoef(x_np)
    adj = (np.abs(corr) >= CORR_THRESHOLD).astype(np.float64)
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1, keepdims=True)
    neigh = (adj @ x_np) / np.maximum(deg, 1.0)
    return (
        PUMA_PROPAGATION_ALPHA * x_np
        + (1.0 - PUMA_PROPAGATION_ALPHA) * neigh
    )

def _farthest_init(z, k, rng):
    n = len(z)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    center = z.mean(axis=0, keepdims=True)
    d0 = ((z - center) ** 2).sum(axis=1)
    chosen = [int(np.argmax(d0))]
    min_dist = ((z - z[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < k:
        idx = int(np.argmax(min_dist))
        if idx in chosen:
            remaining = [i for i in range(n) if i not in chosen]
            idx = int(rng.choice(remaining))
        chosen.append(idx)
        d = ((z - z[idx]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.asarray(chosen, dtype=np.int64)

def _condense_class(x_cls, z_cls, k, rng):
    n = len(x_cls)
    k = int(min(max(k, 1), n))
    if k == n:
        return x_cls.copy()
    init_idx = _farthest_init(z_cls, k, rng)
    centers = z_cls[init_idx].copy()
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(PUMA_CLUSTER_STEPS):
        dist = ((z_cls[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = z_cls[mask].mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-6, rtol=1e-5):
            centers = new_centers
            break
        centers = new_centers

    prototypes = []
    for j in range(k):
        mask = assign == j
        if mask.any():
            prototypes.append(x_cls[mask].mean(axis=0))
        else:
            nearest = np.argmin(((z_cls - centers[j]) ** 2).sum(axis=1))
            prototypes.append(x_cls[nearest])
    return np.asarray(prototypes, dtype=np.float64)

def puma_condense(x_np, y_np, total_budget, propagate=True):
    if len(x_np) == 0 or total_budget <= 0:
        return (
            np.empty((0, x_np.shape[1] if x_np.ndim == 2 else features.shape[1]), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    total_budget = min(int(total_budget), len(x_np))
    z_np = puma_one_time_propagation(x_np) if propagate else x_np
    budgets = _budget_by_class(y_np, total_budget, balanced=True)
    rng = np.random.default_rng(SEED + int(len(x_np)) + int(total_budget))

    out_x = []
    out_y = []
    for cls in sorted(budgets):
        idx = np.where(y_np == cls)[0]
        if len(idx) == 0 or budgets[cls] <= 0:
            continue
        proto = _condense_class(
            x_np[idx],
            z_np[idx],
            budgets[cls],
            rng,
        )
        out_x.append(proto)
        out_y.append(np.full(len(proto), cls, dtype=np.int64))

    if not out_x:
        return (
            np.empty((0, x_np.shape[1]), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    return np.vstack(out_x), np.concatenate(out_y)

def puma_update_memory(memory_x, memory_y, x_new, y_new):
    if len(x_new) == 0:
        return memory_x, memory_y

    update_budget = min(PUMA_MEMORY_PER_UPDATE, len(x_new))
    new_mem_x, new_mem_y = puma_condense(
        x_new,
        y_new,
        total_budget=update_budget,
        propagate=True,
    )

    if memory_x is None or len(memory_x) == 0:
        cand_x = new_mem_x
        cand_y = new_mem_y
    else:
        cand_x = np.concatenate([memory_x, new_mem_x], axis=0)
        cand_y = np.concatenate([memory_y, new_mem_y], axis=0)

    if len(cand_x) <= PUMA_MEMORY_SIZE:
        return cand_x, cand_y

    return puma_condense(
        cand_x,
        cand_y,
        total_budget=PUMA_MEMORY_SIZE,
        propagate=False,
    )


features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

model = None
optimizer = None
criterion = None
memory_x = None
memory_y = None

y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []
window_metrics = []

global_idx = 0
stream_window_id = 0
memory_update_time = 0.0
training_time = 0.0
run_start = time.perf_counter()

for start in tqdm(range(0, N, BATCH_SIZE), desc=f"{METHOD_NAME} streaming"):
    batch_feats = features[start:start + BATCH_SIZE]
    batch_labels = labels[start:start + BATCH_SIZE]
    bsz = len(batch_feats)

    for i in range(bsz):
        features_window.append(batch_feats[i])
        labels_window.append(int(batch_labels[i]))
        index_window.append(global_idx)
        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    first_window = model is None

    x_win_np = np.asarray(features_window, dtype=np.float64)
    y_win_np = np.asarray(labels_window, dtype=np.int64)
    idx_win_np = np.asarray(index_window, dtype=np.int64)

    new_mask_full = np.zeros(WINDOW_SIZE, dtype=bool)
    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[idx_win_np]
    test_mask_full = is_test[idx_win_np]

    x_train_context_np = x_win_np[train_mask_full]
    y_train_context_np = y_win_np[train_mask_full]
    new_train_mask_local_np = new_mask_full[train_mask_full]

    if first_window:
        x_current_np = x_train_context_np
        y_current_np = y_train_context_np
    else:
        x_current_np = x_train_context_np[new_train_mask_local_np]
        y_current_np = y_train_context_np[new_train_mask_local_np]

    if len(x_current_np) > 0:
        if first_window or (PUMA_RETRAIN_FROM_SCRATCH and not first_window):
            model, optimizer, criterion = make_model(features.shape[1])

        if memory_x is None or len(memory_x) == 0:
            x_fit_np = x_current_np
            y_fit_np = y_current_np
            current_count = len(x_current_np)
        else:
            x_fit_np = np.concatenate([x_current_np, memory_x], axis=0)
            y_fit_np = np.concatenate([y_current_np, memory_y], axis=0)
            current_count = len(x_current_np)

        x_fit = torch.tensor(x_fit_np, dtype=torch.float, device=DEVICE)
        y_fit = torch.tensor(y_fit_np, dtype=torch.long, device=DEVICE)

        train_edge_index = build_graph(
            x_fit_np,
            allow_cross=False,
            current_count=current_count,
        )

        epochs_now = EPOCHS_FIRST if first_window else EPOCHS_INC
        t0 = time.perf_counter()
        for _ in range(epochs_now):
            model.train()
            logits, _ = model(x_fit, train_edge_index)
            loss = criterion(logits, y_fit)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        training_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        memory_x, memory_y = puma_update_memory(
            memory_x,
            memory_y,
            x_current_np,
            y_current_np,
        )
        memory_update_time += time.perf_counter() - t0

    if model is None or len(x_train_context_np) == 0:
        stream_window_id += 1
        continue

    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = new_mask_full & test_mask_full

    if eval_test_mask_full.any():
        x_new_test_np = x_win_np[eval_test_mask_full]
        y_new_test_np = y_win_np[eval_test_mask_full]

        x_eval_tensor, eval_edge_index = build_inductive_eval_graph(
            x_train_context_np,
            x_new_test_np,
        )

        model.eval()
        with torch.no_grad():
            logits_eval, hidden_eval = model(x_eval_tensor, eval_edge_index)
            probs_eval = torch.softmax(logits_eval, dim=1)
            ei_used, att_heads = model.cached_att
            head_eval = attention_heads_node_mean(
                ei_used,
                att_heads,
                num_nodes=x_eval_tensor.size(0),
                heads=ATTN_HEADS,
                reference_nodes=x_train_context_np.shape[0],
            )
            mixed_eval = torch.cat([hidden_eval, head_eval], dim=1)

            n_train_ctx = x_train_context_np.shape[0]
            test_slice = slice(n_train_ctx, n_train_ctx + len(x_new_test_np))
            logits_test = logits_eval[test_slice]
            probs_test = probs_eval[test_slice]
            mixed_test = mixed_eval[test_slice]
            window_pred = torch.argmax(logits_test, dim=1).cpu().numpy()

        y_true_test.extend(y_new_test_np.tolist())
        y_pred_test.extend(window_pred.tolist())
        y_prob_test_all.append(probs_test.cpu().numpy())
        hidden_test.extend(mixed_test.cpu().numpy().tolist())

    if ENABLE_WINDOW_ANALYSIS and test_mask_full.any():
        x_window_test_np = x_win_np[test_mask_full]
        y_window_test_np = y_win_np[test_mask_full]
        x_window_eval_tensor, window_eval_edge_index = build_inductive_eval_graph(
            x_train_context_np,
            x_window_test_np,
        )

        model.eval()
        with torch.no_grad():
            window_logits_eval, _ = model(
                x_window_eval_tensor,
                window_eval_edge_index,
            )
            n_train_ctx = x_train_context_np.shape[0]
            window_logits_test = window_logits_eval[
                n_train_ctx:n_train_ctx + len(x_window_test_np)
            ]
            window_pred_full = torch.argmax(
                window_logits_test,
                dim=1,
            ).cpu().numpy()

        window_metrics.append({
            "window": stream_window_id,
            "stream_start": int(idx_win_np[0]),
            "stream_end": int(idx_win_np[-1]),
            "n_test": int(len(y_window_test_np)),
            "accuracy": float(accuracy_score(y_window_test_np, window_pred_full)),
            "precision": float(precision_score(
                y_window_test_np,
                window_pred_full,
                average=WINDOW_METRIC_AVERAGE,
                zero_division=0,
            )),
            "recall": float(recall_score(
                y_window_test_np,
                window_pred_full,
                average=WINDOW_METRIC_AVERAGE,
                zero_division=0,
            )),
            "f1": float(f1_score(
                y_window_test_np,
                window_pred_full,
                average=WINDOW_METRIC_AVERAGE,
                zero_division=0,
            )),
        })

    stream_window_id += 1
    model.cached_att = None

print(f"\n{METHOD_NAME} final condensed-memory size: {0 if memory_x is None else len(memory_x)}")
print(f"{METHOD_NAME} memory-update time: {memory_update_time:.2f} s")
print(f"{METHOD_NAME} training time: {training_time:.2f} s")


y_true_test = np.asarray(y_true_test, dtype=np.int64)
y_pred_test = np.asarray(y_pred_test, dtype=np.int64)
if len(y_prob_test_all) > 0:
    y_prob_test_all = np.vstack(y_prob_test_all)
else:
    y_prob_test_all = np.empty((0, NUM_CLASSES), dtype=np.float64)

print(f"\n=== {METHOD_NAME} Evaluation on Random Test Split (15%) ===")
print(f"Expected hold-out samples: {len(test_idx)}")
print(f"Actually evaluated samples: {len(y_true_test)}")
print(f"Total runtime: {time.perf_counter() - run_start:.2f} s")

if len(y_true_test) == 0:
    print("The test set is empty. Check the split or window settings.")
else:
    unique_ids = np.arange(NUM_CLASSES)
    target_names = [id_to_label[i] for i in unique_ids]
    report = classification_report(
        y_true_test,
        y_pred_test,
        output_dict=True,
        digits=4,
        labels=unique_ids,
        target_names=target_names,
        zero_division=0,
    )

    print("Classification Report:")
    for lab in target_names:
        m = report[lab]
        print(
            f"{lab:<32} precision: {m['precision'] * 100:.2f}%  "
            f"recall: {m['recall'] * 100:.2f}%  "
            f"f1-score: {m['f1-score'] * 100:.2f}%"
        )
    print(f"{'accuracy':<32}: {report['accuracy'] * 100:.2f}%")
    for k in ["macro avg", "weighted avg"]:
        m = report[k]
        print(
            f"{k:<32} precision: {m['precision'] * 100:.2f}%  "
            f"recall: {m['recall'] * 100:.2f}%  "
            f"f1-score: {m['f1-score'] * 100:.2f}%"
        )

    try:
        cm = confusion_matrix(y_true_test, y_pred_test, labels=unique_ids)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=target_names,
        )
        width = max(5.6, min(12.0, 0.75 * NUM_CLASSES + 3.0))
        height = max(5.0, min(10.0, 0.65 * NUM_CLASSES + 3.0))
        fig_cm, ax_cm = plt.subplots(figsize=(width, height))
        disp.plot(values_format="d", cmap="Blues", colorbar=False, ax=ax_cm)
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig_cm.tight_layout()
        plt.show()
        plt.close(fig_cm)
    except Exception as e:
        print("Confusion-matrix plotting failed:", e)

    try:
        if NUM_CLASSES == 2 and y_prob_test_all.shape[0] == len(y_true_test):
            fpr, tpr, _ = roc_curve(y_true_test, y_prob_test_all[:, 1])
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(6.8, 5.2))
            ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            fig_roc.tight_layout()
            plt.show()
            plt.close(fig_roc)
        elif NUM_CLASSES > 2 and y_prob_test_all.shape[0] == len(y_true_test):
            classes_sorted = unique_ids.tolist()
            y_true_bin = label_binarize(y_true_test, classes=classes_sorted)
            fig_roc, ax_roc = plt.subplots(figsize=(7.2, 5.4))
            plotted = False
            for cls_id in classes_sorted:
                col = classes_sorted.index(cls_id)
                y_true_c = y_true_bin[:, col]
                if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
                    continue
                fpr, tpr, _ = roc_curve(y_true_c, y_prob_test_all[:, cls_id])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"{id_to_label[cls_id]} ({roc_auc:.3f})")
                plotted = True
            if plotted:
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend(fontsize=7, loc="lower right")
                fig_roc.tight_layout()
                plt.show()
            plt.close(fig_roc)
    except Exception as e:
        print("ROC plotting failed:", e)

if ENABLE_WINDOW_ANALYSIS and len(window_metrics) > 0:
    window_df = pd.DataFrame(window_metrics).sort_values("window").reset_index(drop=True)
    print("\n=== Window-level Performance ===")
    print(f"Evaluated windows: {len(window_df)}")
    print(f"Mean test samples/window: {window_df['n_test'].mean():.2f}")
    print(f"Mean window Accuracy: {window_df['accuracy'].mean() * 100:.2f}%")
    print(f"Mean window F1 ({WINDOW_METRIC_AVERAGE}): {window_df['f1'].mean() * 100:.2f}%")

    fig_wp, ax_wp = plt.subplots(figsize=(7.2, 4.6))
    ax_wp.plot(window_df["window"], window_df["accuracy"] * 100.0, linewidth=1.5, label="Accuracy")
    ax_wp.plot(window_df["window"], window_df["f1"] * 100.0, linewidth=1.5, label="F1-score")
    ax_wp.set_xlabel("Window Index")
    ax_wp.set_ylabel("Performance (%)")
    ax_wp.set_ylim(0.0, 101.0)
    ax_wp.grid(True, alpha=0.3)
    ax_wp.legend()
    fig_wp.tight_layout()
    plt.show()
    plt.close(fig_wp)

TSNE_MAX_PER_CLASS = 1000
try:
    if len(hidden_test) > 10:
        hidden_test_np = np.asarray(hidden_test, dtype=np.float64)
        labels_tsne = np.asarray(y_true_test[:len(hidden_test_np)], dtype=np.int64)
        finite_mask = np.isfinite(hidden_test_np).all(axis=1)
        hidden_test_np = hidden_test_np[finite_mask]
        labels_tsne = labels_tsne[finite_mask]

        if len(hidden_test_np) > 10:
            dim_std = hidden_test_np.std(axis=0)
            useful_dims = dim_std > 1e-10
            if useful_dims.sum() >= 2:
                x_tsne = hidden_test_np[:, useful_dims]
                x_tsne = StandardScaler().fit_transform(x_tsne)
                x_tsne = np.nan_to_num(x_tsne, nan=0.0, posinf=10.0, neginf=-10.0)
                x_tsne = np.clip(x_tsne, -10.0, 10.0)

                rng = np.random.default_rng(SEED)
                selected = []
                for cls_id in np.unique(labels_tsne):
                    cls_idx = np.where(labels_tsne == cls_id)[0]
                    if len(cls_idx) > TSNE_MAX_PER_CLASS:
                        cls_idx = rng.choice(cls_idx, size=TSNE_MAX_PER_CLASS, replace=False)
                    selected.append(np.asarray(cls_idx, dtype=np.int64))
                selected = np.concatenate(selected)
                selected.sort()
                x_tsne = x_tsne[selected]
                labels_plot = labels_tsne[selected]

                n_tsne = len(x_tsne)
                perplexity = min(30.0, max(5.0, (n_tsne - 1) / 3.0))
                perplexity = min(perplexity, float(n_tsne - 1))
                emb_2d = TSNE(
                    n_components=2,
                    random_state=SEED,
                    perplexity=perplexity,
                    init="pca",
                    learning_rate="auto",
                ).fit_transform(x_tsne)

                fig_tsne, ax_tsne = plt.subplots(figsize=(7.0, 6.0))
                for cls_id in np.unique(labels_plot):
                    mask = labels_plot == cls_id
                    ax_tsne.scatter(
                        emb_2d[mask, 0],
                        emb_2d[mask, 1],
                        s=12,
                        alpha=0.75,
                        label=id_to_label.get(int(cls_id), str(cls_id)),
                    )
                ax_tsne.set_xlabel("t-SNE Dim 1")
                ax_tsne.set_ylabel("t-SNE Dim 2")
                ax_tsne.set_title(f"{METHOD_NAME} Test Embeddings")
                ax_tsne.legend(fontsize=7, markerscale=1.5, loc="best")
                fig_tsne.tight_layout()
                plt.show()
                plt.close(fig_tsne)
except Exception as e:
    print("t-SNE visualization failed:", e)
