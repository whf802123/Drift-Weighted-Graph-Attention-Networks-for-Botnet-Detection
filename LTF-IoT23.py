
import warnings
warnings.filterwarnings("ignore")

from collections import deque
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt
from tqdm import tqdm

CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\iot23_combined_new.csv'

SAMPLE_FRAC = 0.05
MANUAL_MINORITY_LABELS = {"C&C", "C&C-HeartBeat"}
RARE_LABELS_TO_DROP = {
    "C&C-FileDownload",
    "C&C-Torii",
    "FileDownload",
    "C&C-HeartBeat-FileDownload",
    "Okiru-Attack",
    "C&C-Mirai",
    "Attack",
}


WINDOW_SIZE = 1000
BATCH_SIZE = 100
EPOCHS_FIRST = 5
EPOCHS_INC = 1

CORR_THRESHOLD = 0.4
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
SEED = 42

HIDDEN_DIM = 32
LR = 5e-3
WEIGHT_DECAY = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import ipaddress

def ip_to_int(v):
    try:
        s = str(v).strip()
        if s in ("", "-", "nan"):
            return np.nan
        return int(ipaddress.ip_address(s))
    except Exception:
        return np.nan

def to_float(v):
    try:
        s = str(v).strip()
        if s in ("", "-", "nan", "None", "NaN"):
            return np.nan
        return float(s)
    except Exception:
        return np.nan

df = pd.read_csv(CSV_PATH)

if len(df.columns) > 0 and (
        str(df.columns[0]).lower().startswith("unnamed")
        or df.columns[0] == ""
):
    df.drop(df.columns[0], axis=1, inplace=True)

if "label" not in df.columns:
    raise KeyError("IoT23 label column 'label' was not found.")

df["_stream_order_"] = np.arange(len(df), dtype=np.int64)

df = df[
    ~df["label"].isin(RARE_LABELS_TO_DROP)
].copy()

def keep_or_sample(g):
    if str(g.name) in MANUAL_MINORITY_LABELS:
        return g
    return g.sample(frac=SAMPLE_FRAC, random_state=SEED)

df = (
    df.groupby("label", group_keys=False)
    .apply(keep_or_sample)
    .sort_values("_stream_order_")
    .reset_index(drop=True)
)

df.drop(columns=["_stream_order_"], inplace=True)

counts = df["label"].astype(str).value_counts()
too_small = set(counts[counts < 2].index.tolist())
if too_small:
    df = df[
        ~df["label"].isin(too_small)
    ].reset_index(drop=True)

ip_cols = [
    c for c in ["id.orig_h", "id.resp_h"]
    if c in df.columns
]
num_cols = [
    c for c in [
        "ts", "id.orig_p", "id.resp_p", "duration",
        "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes",
        "resp_pkts", "resp_ip_bytes",
    ]
    if c in df.columns
]
cat_cols = [
    c for c in [
        "proto", "service", "conn_state",
        "history", "local_orig", "local_resp",
    ]
    if c in df.columns
]

for c in ip_cols:
    df[c] = df[c].map(ip_to_int)
for c in num_cols:
    df[c] = df[c].map(to_float)
for c in cat_cols:
    df[c] = (
        df[c].astype(str)
        .fillna("-")
        .replace({"nan": "-"})
    )

label_text = df["label"].astype(str).values
unique_text = pd.unique(label_text)
label_to_id = {lab: i for i, lab in enumerate(unique_text)}
id_to_label = {i: lab for lab, i in label_to_id.items()}
labels = np.asarray(
    [label_to_id[x] for x in label_text],
    dtype=np.int64,
)

num_df = (
    df[num_cols + ip_cols].copy()
    if (num_cols or ip_cols)
    else pd.DataFrame(index=df.index)
)
cat_df = (
    pd.get_dummies(df[cat_cols], prefix=cat_cols)
    if cat_cols
    else pd.DataFrame(index=df.index)
)
feat_df = pd.concat([num_df, cat_df], axis=1)

all_idx = np.arange(len(df), dtype=np.int64)
# Stratified 70/15/15 train/validation/test split.
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

scaler = StandardScaler()
features = np.empty_like(feat_df.values, dtype=np.float64)
features[train_idx] = scaler.fit_transform(
    feat_df.iloc[train_idx].values.astype(float)
)
features[test_idx] = scaler.transform(
    feat_df.iloc[test_idx].values.astype(float)
)
features[val_idx] = scaler.transform(
    feat_df.iloc[val_idx].values.astype(float)
)

N = len(features)
is_train = np.zeros(N, dtype=bool)
is_train[train_idx] = True
is_test = np.zeros(N, dtype=bool)
is_test[test_idx] = True
is_val = np.zeros(N, dtype=bool)
is_val[val_idx] = True
print(
    f"Data split: train={len(train_idx)} ({TRAIN_RATIO:.0%}), "
    f"validation={len(val_idx)} ({VALIDATION_RATIO:.0%}), "
    f"test={len(test_idx)} ({TEST_RATIO:.0%})"
)

NUM_CLASSES = len(np.unique(labels))
if len(np.unique(labels[train_idx])) != NUM_CLASSES:
    raise RuntimeError("IoT23 training split does not cover all retained classes.")

print("IoT23 label mapping:", id_to_label)
print(
    f"IoT23: total={N}, train={len(train_idx)}, "
    f"test={len(test_idx)}, classes={NUM_CLASSES}, "
    f"features={features.shape[1]}"
)


# ============================================================
# 2. Graph construction
# ============================================================
def safe_corrcoef(x_np):
    if len(x_np) == 0:
        return np.empty((0, 0), dtype=np.float64)
    if len(x_np) == 1:
        return np.ones((1, 1), dtype=np.float64)
    c = np.corrcoef(x_np)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def build_train_graph(x_np):
    n = len(x_np)
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    corr = safe_corrcoef(x_np)
    adj = np.abs(corr) >= CORR_THRESHOLD
    np.fill_diagonal(adj, False)
    src, dst = np.where(adj)

    self_nodes = np.arange(n, dtype=np.int64)
    src = np.concatenate([src.astype(np.int64), self_nodes])
    dst = np.concatenate([dst.astype(np.int64), self_nodes])

    return torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
        device=DEVICE,
    )


def build_eval_graph(x_train_np, x_test_np):
    n_train = len(x_train_np)
    n_test = len(x_test_np)
    n_total = n_train + n_test

    x_all = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = safe_corrcoef(x_all)

    src_parts, dst_parts = [], []

    if n_train > 0:
        # train -> train
        tt = np.abs(corr[:n_train, :n_train]) >= CORR_THRESHOLD
        np.fill_diagonal(tt, False)
        s, d = np.where(tt)
        if len(s):
            src_parts.append(s.astype(np.int64))
            dst_parts.append(d.astype(np.int64))

        # train -> test only
        s, test_col = np.where(
            np.abs(corr[:n_train, n_train:]) >= CORR_THRESHOLD
        )
        if len(s):
            src_parts.append(s.astype(np.int64))
            dst_parts.append((n_train + test_col).astype(np.int64))

    self_nodes = np.arange(n_total, dtype=np.int64)
    src_parts.append(self_nodes)
    dst_parts.append(self_nodes)

    edge_index = torch.tensor(
        np.vstack([
            np.concatenate(src_parts),
            np.concatenate(dst_parts),
        ]),
        dtype=torch.long,
        device=DEVICE,
    )

    return (
        torch.tensor(x_all, dtype=torch.float32, device=DEVICE),
        edge_index,
    )

# ============================================================
# 3. Lightweight GCN backbone
# ============================================================
class LiteGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(
            in_dim, hidden_dim,
            add_self_loops=False,
            normalize=True,
        )
        self.conv2 = GCNConv(
            hidden_dim, hidden_dim,
            add_self_loops=False,
            normalize=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, return_embedding=False):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits

# ============================================================
# 4. Method-specific helpers
# ============================================================

# Lightweight LTF settings.
LTF_MEMORY_PER_CLASS = 12
LTF_SIM_PER_CLASS = 12
LTF_CANDIDATE_CAP = 160
LTF_ALPHA = 0.5
LTF_MMD_WEIGHT = 0.10

def _normalize01(v):
    if v.numel() == 0:
        return v
    lo = v.min()
    hi = v.max()
    return (v - lo) / (hi - lo + 1e-12)

@torch.no_grad()
def _greedy_mean_match(emb, candidate_idx, target_mean, budget):
    """
    Linear-kernel MMD herding:
    greedily keeps the selected-set mean close to the target mean.
    """
    if candidate_idx.numel() == 0 or budget <= 0:
        return torch.empty(0, dtype=torch.long, device=DEVICE)

    remaining = candidate_idx.clone()
    selected = []
    running_sum = torch.zeros_like(target_mean)

    for k in range(min(budget, remaining.numel())):
        cand_emb = emb[remaining]
        new_mean = (
                           running_sum.unsqueeze(0) + cand_emb
                   ) / float(k + 1)
        score = ((new_mean - target_mean.unsqueeze(0)) ** 2).mean(dim=1)
        j = int(torch.argmin(score).item())
        chosen = remaining[j]
        selected.append(chosen)
        running_sum = running_sum + emb[chosen]
        remaining = torch.cat([remaining[:j], remaining[ j +1:]])
        if remaining.numel() == 0:
            break

    return torch.stack(selected) if selected else torch.empty(
        0, dtype=torch.long, device=DEVICE
    )

@torch.no_grad()
def select_ltf_subsets(model, x, edge_index, y, new_mask, window_id):
    """
    Fixed-label streaming adaptation of LTF:
    - G_sub: low previous-model CE + low distribution discrepancy.
    - G_sim: distribution-only representative set.
    """
    model.eval()
    logits, emb = model(x, edge_index, return_embedding=True)
    per_node_ce = F.cross_entropy(logits, y, reduction="none")

    old_idx = torch.where(~new_mask)[0]
    sub_parts, sim_parts = [], []

    rng = np.random.default_rng(SEED + window_id)

    for cls in range(NUM_CLASSES):
        cls_old = old_idx[y[old_idx] == cls]
        if cls_old.numel() == 0:
            continue

        # Lightweight candidate partitioning/capping.
        if cls_old.numel() > LTF_CANDIDATE_CAP:
            chosen_np = rng.choice(
                cls_old.detach().cpu().numpy(),
                size=LTF_CANDIDATE_CAP,
                replace=False,
            )
            cls_old = torch.tensor(
                chosen_np, dtype=torch.long, device=DEVICE
            )

        target_mean = emb[cls_old].mean(dim=0)

        # G_sim: distribution discrepancy only.
        sim_idx = _greedy_mean_match(
            emb, cls_old, target_mean, LTF_SIM_PER_CLASS
        )
        if sim_idx.numel():
            sim_parts.append(sim_idx)

        # G_sub: greedy objective = alpha*CE + mean-matching discrepancy.
        remaining = cls_old.clone()
        selected = []
        running_sum = torch.zeros_like(target_mean)
        ce_norm_all = _normalize01(per_node_ce[cls_old])
        ce_map = {
            int(i.item()): ce_norm_all[j]
            for j, i in enumerate(cls_old)
        }

        for k in range(min(LTF_MEMORY_PER_CLASS, remaining.numel())):
            cand_emb = emb[remaining]
            mean_after = (
                                 running_sum.unsqueeze(0) + cand_emb
                         ) / float(k + 1)
            mmd_score = ((mean_after - target_mean.unsqueeze(0)) ** 2).mean(dim=1)
            mmd_score = _normalize01(mmd_score)

            ce_score = torch.stack([
                ce_map[int(i.item())]
                for i in remaining
            ])

            score = LTF_ALPHA * ce_score + mmd_score
            j = int(torch.argmin(score).item())
            chosen = remaining[j]
            selected.append(chosen)
            running_sum = running_sum + emb[chosen]
            remaining = torch.cat([remaining[:j], remaining[ j +1:]])
            if remaining.numel() == 0:
                break

        if selected:
            sub_parts.append(torch.stack(selected))

    sub_idx = (
        torch.unique(torch.cat(sub_parts))
        if sub_parts
        else torch.empty(0, dtype=torch.long, device=DEVICE)
    )
    sim_idx = (
        torch.unique(torch.cat(sim_parts))
        if sim_parts
        else torch.empty(0, dtype=torch.long, device=DEVICE)
    )
    return sub_idx, sim_idx

def ltf_distribution_loss(emb, y, sub_idx, sim_idx):
    if sub_idx.numel() == 0 or sim_idx.numel() == 0:
        return torch.zeros((), device=DEVICE)

    loss = torch.zeros((), device=DEVICE)
    used = 0
    for cls in range(NUM_CLASSES):
        a = sub_idx[y[sub_idx] == cls]
        b = sim_idx[y[sim_idx] == cls]
        if a.numel() == 0 or b.numel() == 0:
            continue

        # Linear MMD / mean alignment; target branch is stop-gradient.
        mu_sub = emb[a].mean(dim=0)
        mu_sim = emb[b].detach().mean(dim=0)
        loss = loss + F.mse_loss(mu_sub, mu_sim)
        used += 1

    return loss / max(used, 1)


# ============================================================
# 5. Streaming state
# ============================================================
features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

model = None
optimizer = None

# Match the earlier CTU13 setup: binary class weighting only.
if NUM_CLASSES == 2:
    train_labels_only = labels[train_idx]
    pos_ratio = (train_labels_only == 1).mean() + 1e-8
    w_neg = 1.0 / max(1e-8, 1.0 - pos_ratio)
    w_pos = 1.0 / max(1e-8, pos_ratio)
    class_weights = torch.tensor(
        [w_neg, w_pos],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

global_idx = 0
window_id = 0

y_true_test = []
y_pred_test = []
y_prob_test = []

window_metrics = []
selection_sizes = []

# ============================================================
# 6. Streaming training + evaluation
# ============================================================
print("\n=== Streaming LTF-Lite ===")

for start in tqdm(range(0, N, BATCH_SIZE), desc="LTF-Lite"):
    bx = features[start:start + BATCH_SIZE]
    by = labels[start:start + BATCH_SIZE]
    bsz = len(bx)

    for i in range(bsz):
        features_window.append(bx[i])
        labels_window.append(int(by[i]))
        index_window.append(global_idx)
        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    first_window = model is None
    window_id += 1

    x_win = np.asarray(features_window, dtype=np.float64)
    y_win = np.asarray(labels_window, dtype=np.int64)
    idx_win = np.asarray(index_window, dtype=np.int64)

    new_mask_full = np.zeros(len(x_win), dtype=bool)
    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[idx_win]
    test_mask_full = is_test[idx_win]

    x_train_np = x_win[train_mask_full]
    y_train_np = y_win[train_mask_full]
    new_train_mask_np = new_mask_full[train_mask_full]

    if len(x_train_np) == 0:
        continue

    x_train = torch.tensor(
        x_train_np, dtype=torch.float32, device=DEVICE
    )
    y_train = torch.tensor(
        y_train_np, dtype=torch.long, device=DEVICE
    )
    edge_train = build_train_graph(x_train_np)

    if model is None:
        model = LiteGCN(
            in_dim=x_train.size(1),
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    # Method-specific selection/setup once per window.

    new_train_mask = torch.tensor(
        new_train_mask_np,
        dtype=torch.bool,
        device=DEVICE,
    )

    if first_window:
        # No old subset exists yet.
        ltf_sub_idx = torch.empty(
            0, dtype=torch.long, device=DEVICE
        )
        ltf_sim_idx = torch.empty(
            0, dtype=torch.long, device=DEVICE
        )
        used_idx = torch.arange(
            len(x_train_np), device=DEVICE
        )
    else:
        ltf_sub_idx, ltf_sim_idx = select_ltf_subsets(
            model, x_train, edge_train, y_train,
            new_train_mask, window_id
        )
        new_idx = torch.where(new_train_mask)[0]
        used_idx = torch.unique(
            torch.cat([new_idx, ltf_sub_idx])
        )
        if used_idx.numel() == 0:
            used_idx = new_idx

    selection_sizes.append(int(used_idx.numel()))


    epochs_now = EPOCHS_FIRST if first_window else EPOCHS_INC

    for _ in range(epochs_now):
        model.train()
        logits, emb = model(
            x_train, edge_train, return_embedding=True
        )


        ce = criterion(logits[used_idx], y_train[used_idx])
        dist = ltf_distribution_loss(
            emb, y_train, ltf_sub_idx, ltf_sim_idx
        )
        loss = ce + LTF_MMD_WEIGHT * dist


        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --------------------------------------------------------
    # Progressive test collection: every hold-out sample once.
    # --------------------------------------------------------
    eval_test_mask = (
        test_mask_full
        if first_window
        else (new_mask_full & test_mask_full)
    )

    if eval_test_mask.any():
        x_test_np = x_win[eval_test_mask]
        y_test_np = y_win[eval_test_mask]

        x_eval, edge_eval = build_eval_graph(
            x_train_np, x_test_np
        )

        model.eval()
        with torch.no_grad():
            logits_eval = model(x_eval, edge_eval)
            logits_t = logits_eval[len(x_train_np):]
            probs_t = torch.softmax(logits_t, dim=1)
            pred_t = torch.argmax(logits_t, dim=1)

        y_true_test.extend(y_test_np.tolist())
        y_pred_test.extend(pred_t.cpu().numpy().tolist())
        y_prob_test.append(probs_t.cpu().numpy())

    # --------------------------------------------------------
    # Whole-current-window diagnostics.
    # --------------------------------------------------------
    if test_mask_full.any():
        x_wtest_np = x_win[test_mask_full]
        y_wtest_np = y_win[test_mask_full]

        x_eval, edge_eval = build_eval_graph(
            x_train_np, x_wtest_np
        )

        model.eval()
        with torch.no_grad():
            logits_eval = model(x_eval, edge_eval)
            pred = torch.argmax(
                logits_eval[len(x_train_np):],
                dim=1
            ).cpu().numpy()

        window_metrics.append({
            "window": window_id,
            "accuracy": accuracy_score(y_wtest_np, pred),
            "precision": precision_score(
                y_wtest_np, pred,
                average="macro",
                labels=np.arange(NUM_CLASSES),
                zero_division=0,
            ),
            "recall": recall_score(
                y_wtest_np, pred,
                average="macro",
                labels=np.arange(NUM_CLASSES),
                zero_division=0,
            ),
            "f1": f1_score(
                y_wtest_np, pred,
                average="macro",
                labels=np.arange(NUM_CLASSES),
                zero_division=0,
            ),
        })

# ============================================================
# 7. Final evaluation
# ============================================================
y_true_test = np.asarray(y_true_test, dtype=np.int64)
y_pred_test = np.asarray(y_pred_test, dtype=np.int64)
y_prob_test = (
    np.vstack(y_prob_test)
    if len(y_prob_test) > 0
    else np.empty((0, NUM_CLASSES), dtype=np.float64)
)

print("\n=== Final Hold-out Evaluation ===")
print("Expected test samples:", len(test_idx))
print("Actually evaluated:", len(y_true_test))

if len(y_true_test) == 0:
    raise RuntimeError("No test samples were evaluated.")

ids = np.arange(NUM_CLASSES)
names = [id_to_label[int(i)] for i in ids]

report = classification_report(
    y_true_test,
    y_pred_test,
    labels=ids,
    target_names=names,
    output_dict=True,
    zero_division=0,
)

for name in names:
    m = report[name]
    print(
        f"{name:<28} "
        f"P={m['precision' ] *100:.2f}%  "
        f"R={m['recall' ] *100:.2f}%  "
        f"F1={m['f1-score' ] *100:.2f}%"
    )

print(f"{'accuracy':<28} {report['accuracy' ] *100:.2f}%")
for k in ["macro avg", "weighted avg"]:
    m = report[k]
    print(
        f"{k:<28} "
        f"P={m['precision' ] *100:.2f}%  "
        f"R={m['recall' ] *100:.2f}%  "
        f"F1={m['f1-score' ] *100:.2f}%"
    )

print(
    f"Mean selected/replayed nodes per window: "
    f"{np.mean(selection_sizes) if selection_sizes else 0:.2f}"
)

# Confusion matrix
try:
    cm = confusion_matrix(
        y_true_test, y_pred_test, labels=ids
    )
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ConfusionMatrixDisplay(
        cm, display_labels=names
    ).plot(
        values_format="d",
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)
except Exception as e:
    print("Confusion matrix failed:", e)

# Multiclass/binary ROC.
try:
    if y_prob_test.shape[0] == len(y_true_test):
        if NUM_CLASSES == 2:
            if len(np.unique(y_true_test)) == 2:
                fpr, tpr, _ = roc_curve(
                    y_true_test, y_prob_test[:, 1]
                )
                print(f"ROC-AUC: {auc(fpr, tpr ) *100:.2f}%")
        else:
            y_bin = label_binarize(y_true_test, classes=ids)
            aucs = []
            for c in ids:
                yc = y_bin[:, int(c)]
                if yc.sum() == 0 or yc.sum() == len(yc):
                    continue
                fpr, tpr, _ = roc_curve(
                    yc, y_prob_test[:, int(c)]
                )
                aucs.append(auc(fpr, tpr))
            if aucs:
                print(
                    f"Macro OvR ROC-AUC: "
                    f"{np.mean(aucs ) *100:.2f}%"
                )
except Exception as e:
    print("ROC failed:", e)

# Window-level curve.
if window_metrics:
    wdf = pd.DataFrame(window_metrics)
    print(
        "Mean window metrics: "
        f"Acc={wdf['accuracy'].mean( ) *100:.2f}%, "
        f"MacroP={wdf['precision'].mean( ) *100:.2f}%, "
        f"MacroR={wdf['recall'].mean( ) *100:.2f}%, "
        f"MacroF1={wdf['f1'].mean( ) *100:.2f}%"
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(wdf["window"], wdf["accuracy" ] *100, label="Accuracy")
    ax.plot(wdf["window"], wdf["f1" ] *100, label="Macro F1")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("LTF-Lite Window-level Performance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)
