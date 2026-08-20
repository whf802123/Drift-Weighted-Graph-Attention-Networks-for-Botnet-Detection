
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

        tt = np.abs(corr[:n_train, :n_train]) >= CORR_THRESHOLD
        np.fill_diagonal(tt, False)
        s, d = np.where(tt)
        if len(s):
            src_parts.append(s.astype(np.int64))
            dst_parts.append(d.astype(np.int64))


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

GPC_PER_CLASS = 10
ANCHOR_PER_CLASS = 4
CANDIDATE_CAP_PER_CLASS = 160
SKETCH_RANK = 12
FEATURE_SKETCH_DIM = 8
STRUCT_WEIGHT = 0.20
FIRST_WINDOW_WARMUP_EPOCHS = 2

def _row_normalized_dense_adj(edge_index, n):
    A = torch.zeros((n, n), dtype=torch.float32, device=DEVICE)
    A[edge_index[1], edge_index[0]] = 1.0
    deg = A.sum(dim=1, keepdim=True).clamp_min(1.0)
    return A / deg

@torch.no_grad()
def light_spectral_sketch(x, edge_index, window_id):
    n, d = x.shape
    A = _row_normalized_dense_adj(edge_index, n)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + 1000 + window_id)

    rank = min(SKETCH_RANK, n)
    perm = torch.randperm(n, generator=g, device=DEVICE)[:rank]
    struct = A[:, perm]


    gp = torch.Generator(device=DEVICE)
    gp.manual_seed(SEED + 777)
    R = torch.randn(
        (d, FEATURE_SKETCH_DIM),
        generator=gp,
        device=DEVICE,
    ) / np.sqrt(max(d, 1))
    feat = x @ R

    z = torch.cat([struct, feat], dim=1)
    return F.normalize(z, dim=1, eps=1e-12)

@torch.no_grad()
def classifier_gradient_proxy(model, logits, emb, y):
    p = torch.softmax(logits, dim=1)
    onehot = F.one_hot(y, num_classes=NUM_CLASSES).float()
    residual = p - onehot
    grad_w = torch.einsum("nc,nh->nch", residual, emb)
    grad_b = residual
    g = torch.cat(
        [grad_w.reshape(len(y), -1), grad_b],
        dim=1,
    )
    return F.normalize(g, dim=1, eps=1e-12)

@torch.no_grad()
def select_lightgntk_nodes(
        model, x, edge_index, y, new_mask, window_id
):
    model.eval()
    logits, emb = model(x, edge_index, return_embedding=True)

    grad_proxy = classifier_gradient_proxy(
        model, logits, emb, y
    )
    spectral = light_spectral_sketch(
        x, edge_index, window_id
    )

    rng = np.random.default_rng(SEED + 5000 + window_id)
    selected_parts = []


    old_idx = torch.where(~new_mask)[0]

    for cls in range(NUM_CLASSES):
        cls_old = old_idx[y[old_idx] == cls]
        if cls_old.numel() == 0:
            continue

        if cls_old.numel() > CANDIDATE_CAP_PER_CLASS:
            ids_np = rng.choice(
                cls_old.cpu().numpy(),
                size=CANDIDATE_CAP_PER_CLASS,
                replace=False,
            )
            cls_old = torch.tensor(
                ids_np, dtype=torch.long, device=DEVICE
            )

        n_anchor = min(ANCHOR_PER_CLASS, cls_old.numel())
        anchor_np = rng.choice(
            cls_old.cpu().numpy(),
            size=n_anchor,
            replace=False,
        )
        anchor = torch.tensor(
            anchor_np, dtype=torch.long, device=DEVICE
        )

        center_g = grad_proxy[anchor].mean(dim=0)
        center_s = spectral[anchor].mean(dim=0)

        gd = ((grad_proxy[cls_old] - center_g) ** 2).mean(dim=1)
        sd = ((spectral[cls_old] - center_s) ** 2).mean(dim=1)

        gd = (gd - gd.min()) / (gd.max() - gd.min() + 1e-12)
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-12)

        score = gd + STRUCT_WEIGHT * sd
        k = min(GPC_PER_CLASS, cls_old.numel())
        chosen = cls_old[torch.argsort(score)[:k]]
        selected_parts.append(chosen)

    if selected_parts:
        return torch.unique(torch.cat(selected_parts))
    return torch.empty(0, dtype=torch.long, device=DEVICE)

features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

model = None
optimizer = None


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

print("\n=== Streaming LightGNTK-Lite ===")

for start in tqdm(range(0, N, BATCH_SIZE), desc="LightGNTK-Lite"):
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

    new_train_mask = torch.tensor(
        new_train_mask_np,
        dtype=torch.bool,
        device=DEVICE,
    )

    if first_window:

        for _warm in range(FIRST_WINDOW_WARMUP_EPOCHS):
            model.train()
            logits_w = model(x_train, edge_train)
            loss_w = criterion(logits_w, y_train)
            optimizer.zero_grad(set_to_none=True)
            loss_w.backward()
            optimizer.step()

        distilled_idx = torch.arange(
            len(x_train_np), device=DEVICE
        )
        used_idx = distilled_idx
    else:
        distilled_idx = select_lightgntk_nodes(
            model, x_train, edge_train, y_train,
            new_train_mask, window_id
        )
        new_idx = torch.where(new_train_mask)[0]
        used_idx = torch.unique(
            torch.cat([new_idx, distilled_idx])
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

        loss = criterion(
            logits[used_idx],
            y_train[used_idx],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


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
    ax.set_title("LightGNTK-Lite Window-level Performance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)
