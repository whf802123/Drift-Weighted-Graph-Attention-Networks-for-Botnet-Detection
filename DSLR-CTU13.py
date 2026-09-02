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


from torch_geometric.utils import negative_sampling, coalesce

METHOD_NAME = "DSLR"
DSLR_MEMORY_SIZE = 400
DSLR_COVERAGE_DISTANCE = 1.0
DSLR_LP_EPOCHS = 5
DSLR_LP_LR = 1e-2
DSLR_LP_MAX_POS_EDGES = 2000
DSLR_CANDIDATES = 20
DSLR_KNN = 5
DSLR_LINK_THRESHOLD = 0.5

def _budget_by_class_dslr(y_np, total_budget):
    classes = np.unique(y_np)
    if len(classes) == 0 or total_budget <= 0:
        return {}
    total_budget = min(int(total_budget), len(y_np))
    base = total_budget // len(classes)
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

def coverage_diversity_indices(emb_np, y_np, total_budget):
    if len(emb_np) == 0 or total_budget <= 0:
        return np.empty((0,), dtype=np.int64)

    budgets = _budget_by_class_dslr(y_np, total_budget)
    selected_global = []

    for cls in sorted(budgets):
        cls_global = np.where(y_np == cls)[0]
        k = min(budgets[cls], len(cls_global))
        if k <= 0:
            continue
        if k >= len(cls_global):
            selected_global.extend(cls_global.tolist())
            continue

        z = torch.tensor(emb_np[cls_global], dtype=torch.float)
        dist_matrix = torch.cdist(z, z, p=2).cpu().numpy()
        mean_dist = float(dist_matrix.mean())
        radius = max(mean_dist * DSLR_COVERAGE_DISTANCE, 1e-12)
        dist_bin = (dist_matrix < radius).astype(np.int8)

        temp = dist_bin.copy()
        chosen_local = []
        covered = set()

        for _ in range(k):
            scores = temp.sum(axis=0)
            idx = int(np.argmax(scores))

            if idx in chosen_local:
                remaining = [j for j in range(len(cls_global)) if j not in chosen_local]
                if not remaining:
                    break
                idx = remaining[0]

            chosen_local.append(idx)
            new_cover = np.where(temp[idx] == 1)[0].tolist()
            covered.update(new_cover)

            if new_cover:
                temp[new_cover, :] = 0
                temp[:, new_cover] = 0

            if len(covered) >= 0.9 * len(cls_global):
                temp = dist_bin.copy()
                temp[chosen_local, :] = 0
                temp[:, chosen_local] = 0
                covered = set(chosen_local)

        selected_global.extend(cls_global[np.asarray(chosen_local, dtype=np.int64)].tolist())

    return np.asarray(selected_global, dtype=np.int64)

class DSLRLinkPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def score(self, z, edge_index):
        h = self.proj(z)
        src, dst = edge_index
        return (h[src] * h[dst]).sum(dim=1) / np.sqrt(h.size(1))

def train_link_predictor(z, edge_index):
    n = z.size(0)
    if n <= 1 or edge_index.size(1) == 0:
        return None

    nonself = edge_index[0] != edge_index[1]
    pos_edge = edge_index[:, nonself]
    if pos_edge.size(1) == 0:
        return None

    if pos_edge.size(1) > DSLR_LP_MAX_POS_EDGES:
        perm = torch.randperm(pos_edge.size(1), device=DEVICE)[:DSLR_LP_MAX_POS_EDGES]
        pos_edge = pos_edge[:, perm]

    predictor = DSLRLinkPredictor(z.size(1)).to(DEVICE)
    opt = torch.optim.Adam(predictor.parameters(), lr=DSLR_LP_LR)

    z_det = z.detach()
    for _ in range(DSLR_LP_EPOCHS):
        neg_edge = negative_sampling(
            edge_index=edge_index,
            num_nodes=n,
            num_neg_samples=pos_edge.size(1),
            method="sparse",
        )

        pos_score = predictor.score(z_det, pos_edge)
        neg_score = predictor.score(z_det, neg_edge)

        score = torch.cat([pos_score, neg_score], dim=0)
        target = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score),
        ], dim=0)

        loss = F.binary_cross_entropy_with_logits(score, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return predictor

def dslr_refine_structure(model, x_tensor, base_edge_index, current_count):
    n = x_tensor.size(0)
    replay_count = n - current_count
    if replay_count <= 0 or current_count <= 0:
        return base_edge_index

    model.eval()
    with torch.no_grad():
        _, z = model(x_tensor, base_edge_index)

    predictor = train_link_predictor(z, base_edge_index)
    if predictor is None:
        return base_edge_index

    z_norm = F.normalize(z.detach(), dim=1)
    replay_nodes = torch.arange(current_count, n, device=DEVICE)

    src_keep = []
    dst_keep = []

    base_src, base_dst = base_edge_index
    current_edge_mask = (base_src < current_count) & (base_dst < current_count)
    src_keep.append(base_src[current_edge_mask])
    dst_keep.append(base_dst[current_edge_mask])

    self_nodes = torch.arange(n, device=DEVICE)
    src_keep.append(self_nodes)
    dst_keep.append(self_nodes)

    predictor.eval()
    with torch.no_grad():
        for r in replay_nodes.tolist():
            sim = torch.matmul(z_norm[:current_count], z_norm[r])
            k_cand = min(DSLR_CANDIDATES, current_count)
            candidate = torch.topk(sim, k=k_cand, largest=True).indices

            cand_edges = torch.stack([
                torch.full_like(candidate, r),
                candidate,
            ], dim=0)
            probs = torch.sigmoid(predictor.score(z, cand_edges))

            valid = torch.where(probs >= DSLR_LINK_THRESHOLD)[0]
            if valid.numel() == 0:
                valid = torch.topk(
                    probs,
                    k=min(DSLR_KNN, len(probs)),
                    largest=True,
                ).indices
            elif valid.numel() > DSLR_KNN:
                local_top = torch.topk(
                    probs[valid],
                    k=DSLR_KNN,
                    largest=True,
                ).indices
                valid = valid[local_top]

            nbr = candidate[valid]
            if nbr.numel() > 0:
                r_vec = torch.full_like(nbr, r)
                src_keep.append(r_vec)
                dst_keep.append(nbr)
                src_keep.append(nbr)
                dst_keep.append(r_vec)

    src = torch.cat(src_keep)
    dst = torch.cat(dst_keep)
    return coalesce(torch.stack([src, dst], dim=0), num_nodes=n)

def dslr_update_memory(memory_x, memory_y, candidate_x, candidate_y, candidate_emb):
    total_budget = min(DSLR_MEMORY_SIZE, len(candidate_x))
    selected = coverage_diversity_indices(
        candidate_emb,
        candidate_y,
        total_budget,
    )
    return candidate_x[selected].copy(), candidate_y[selected].copy()


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
structure_time = 0.0
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

    if model is None and len(x_current_np) > 0:
        model, optimizer, criterion = make_model(features.shape[1])

    if model is not None and len(x_current_np) > 0:
        if memory_x is None or len(memory_x) == 0:
            x_fit_np = x_current_np
            y_fit_np = y_current_np
            current_count = len(x_current_np)
            replay_count = 0
        else:
            x_fit_np = np.concatenate([x_current_np, memory_x], axis=0)
            y_fit_np = np.concatenate([y_current_np, memory_y], axis=0)
            current_count = len(x_current_np)
            replay_count = len(memory_x)

        x_fit = torch.tensor(x_fit_np, dtype=torch.float, device=DEVICE)
        y_fit = torch.tensor(y_fit_np, dtype=torch.long, device=DEVICE)
        base_edge_index = build_graph(x_fit_np, allow_cross=True)

        t0 = time.perf_counter()
        train_edge_index = dslr_refine_structure(
            model,
            x_fit,
            base_edge_index,
            current_count=current_count,
        )
        structure_time += time.perf_counter() - t0

        current_idx_fit = torch.arange(current_count, device=DEVICE)
        replay_idx_fit = torch.arange(current_count, len(x_fit_np), device=DEVICE)

        epochs_now = EPOCHS_FIRST if first_window else EPOCHS_INC
        t0 = time.perf_counter()
        for _ in range(epochs_now):
            model.train()
            logits, _ = model(x_fit, train_edge_index)

            current_loss = criterion(
                logits[current_idx_fit],
                y_fit[current_idx_fit],
            )

            if replay_count > 0:
                replay_loss = criterion(
                    logits[replay_idx_fit],
                    y_fit[replay_idx_fit],
                )
                beta = replay_count / max(current_count + replay_count, 1)
                loss = (1.0 - beta) * current_loss + beta * replay_loss
            else:
                loss = current_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        training_time += time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            _, candidate_emb_t = model(x_fit, train_edge_index)
        candidate_emb = candidate_emb_t.cpu().numpy()

        t0 = time.perf_counter()
        memory_x, memory_y = dslr_update_memory(
            memory_x,
            memory_y,
            x_fit_np,
            y_fit_np,
            candidate_emb,
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

print(f"\n{METHOD_NAME} final replay-memory size: {0 if memory_x is None else len(memory_x)}")
print(f"{METHOD_NAME} structure-learning time: {structure_time:.2f} s")
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
