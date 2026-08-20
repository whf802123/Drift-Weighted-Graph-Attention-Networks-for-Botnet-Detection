
from pathlib import Path
import ast


import warnings
warnings.filterwarnings("ignore")

from collections import deque
import ipaddress

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from tqdm import tqdm

CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\iot23_combined_new.csv'

WINDOW_SIZE = 1000
BATCH_SIZE = 100


EPOCHS_FIRST = 10
EPOCHS_INC = 2

CORR_THRESHOLD = 0.4
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
SEED = 42

SAMPLE_FRAC = 0.05
MANUAL_MINORITY_LABELS = {'C&C', 'C&C-HeartBeat'}

RARE_LABELS_TO_DROP = {
    'C&C-FileDownload',
    'C&C-Torii',
    'FileDownload',
    'C&C-HeartBeat-FileDownload',
    'Okiru-Attack',
    'C&C-Mirai',
    'Attack',
}


PDGNN_HIDDEN = 256
PDGNN_L = 2
LR = 5e-3
WEIGHT_DECAY = 5e-4
MLP_DROPOUT = 0.0
LINEAR_BIAS = False


TEM_STORE_RATIO = 0.10
TEM_CAPACITY = 1000
TEM_SAMPLER = "degree"

TSNE_MAX_PER_CLASS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def is_unnamed(colname):
    return (
        colname == ''
        or pd.isna(colname)
        or str(colname).lower().startswith('unnamed')
    )


def ip_to_int(val):
    try:
        s = str(val).strip()
        if s == '-' or s == '' or s.lower() == 'nan':
            return np.nan
        return int(ipaddress.ip_address(s))
    except Exception:
        return np.nan


def to_float(val):
    try:
        s = str(val).strip()
        if s in ('', '-', 'nan', 'None', 'NaN'):
            return np.nan
        return float(s)
    except Exception:
        return np.nan


df = pd.read_csv(CSV_PATH)


if len(df.columns) > 0 and is_unnamed(df.columns[0]):
    df.drop(df.columns[0], axis=1, inplace=True)


if 'num' not in df.columns:
    df.insert(0, 'num', range(len(df)))

if 'label' not in df.columns:
    raise KeyError(
        "Column 'label' was not found. Ensure that the label column in the IoT23 CSV is named 'label' in lowercase."
    )


before = len(df)
df = df[~df['label'].isin(RARE_LABELS_TO_DROP)].reset_index(drop=True)
after = len(df)
print(
    f"Dropped rare classes {sorted(list(RARE_LABELS_TO_DROP))}. "
    f"Rows: {before} -> {after}"
)

print(f"Minority (kept intact): {sorted(MANUAL_MINORITY_LABELS)}")
print(f"Sampling {SAMPLE_FRAC * 100:.1f}% for ALL other classes ...")


def keep_or_sample(group: pd.DataFrame) -> pd.DataFrame:
    lab = str(group.name)
    if lab in MANUAL_MINORITY_LABELS:
        return group
    return group.sample(frac=SAMPLE_FRAC, random_state=SEED)

df = (
    df.groupby('label', group_keys=False)
      .apply(keep_or_sample)
      .reset_index(drop=True)
)

label_counts_sampled = df['label'].astype(str).value_counts()
print("Label counts after sampling:")
print(label_counts_sampled.to_string())


too_small = set(
    label_counts_sampled[label_counts_sampled < 2].index.tolist()
)
if too_small:
    print(
        f"[NOTICE] Classes with <2 samples after sampling are dropped: "
        f"{sorted(list(too_small))}"
    )
    df = df[~df['label'].isin(too_small)].reset_index(drop=True)


ip_cols = [
    c for c in ['id.orig_h', 'id.resp_h']
    if c in df.columns
]

num_cols_raw = [
    c for c in [
        'ts',
        'id.orig_p',
        'id.resp_p',
        'duration',
        'orig_bytes',
        'resp_bytes',
        'missed_bytes',
        'orig_pkts',
        'orig_ip_bytes',
        'resp_pkts',
        'resp_ip_bytes',
    ]
    if c in df.columns
]

cat_cols = [
    c for c in [
        'proto',
        'service',
        'conn_state',
        'history',
        'local_orig',
        'local_resp',
    ]
    if c in df.columns
]

for c in ip_cols:
    df[c] = df[c].map(ip_to_int)

for c in num_cols_raw:
    df[c] = df[c].map(to_float)

for c in cat_cols:
    df[c] = (
        df[c]
        .astype(str)
        .fillna('-')
        .replace({'nan': '-'})
    )


label_text = df['label'].astype(str).values
unique_labels_text = pd.unique(label_text)
label_to_id = {
    lab: i for i, lab in enumerate(unique_labels_text)
}
id_to_label = {
    v: k for k, v in label_to_id.items()
}
labels = np.asarray(
    [label_to_id[x] for x in label_text],
    dtype=np.int64,
)

print("\nLabel mapping after sampling:", label_to_id)


num_df = (
    df[num_cols_raw + ip_cols].copy()
    if (num_cols_raw or ip_cols)
    else pd.DataFrame(index=df.index)
)

cat_df = (
    pd.get_dummies(
        df[cat_cols],
        prefix=cat_cols,
        dummy_na=False,
    )
    if cat_cols
    else pd.DataFrame(index=df.index)
)

feat_df = pd.concat(
    [num_df, cat_df],
    axis=1,
)

if feat_df.shape[1] == 0:
    raise RuntimeError(
        "No feature columns could be constructed. Check the input CSV."
    )


N_total = len(df)
all_idx = np.arange(N_total, dtype=np.int64)


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


scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

features = np.empty_like(
    feat_df.values,
    dtype=np.float64,
)

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

train_unique_ids = np.unique(labels[train_idx])
NUM_CLASSES = len(train_unique_ids)

if NUM_CLASSES != len(np.unique(labels)):
    raise RuntimeError(
        "The training set does not contain all classes; the current multiclass configuration cannot be used."
    )

print("\nDevice:", DEVICE)
print(
    f"Samples: total={N_total}, "
    f"train={len(train_idx)}, "
    f"test={len(test_idx)}"
)
print(f"Features: {feat_df.shape[1]}")
print(
    f"PDGNN-TEM: L={PDGNN_L}, hidden={PDGNN_HIDDEN}, "
    f"TEM capacity={TEM_CAPACITY}, sampler={TEM_SAMPLER}"
)

def safe_row_corrcoef(x_np: np.ndarray) -> np.ndarray:
    n = x_np.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    if n == 1:
        return np.ones((1, 1), dtype=np.float64)

    corr = np.corrcoef(x_np)

    return np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

def build_train_adjacency(
    x_train_np: np.ndarray,
) -> np.ndarray:
    n = x_train_np.shape[0]

    if n == 0:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    corr = safe_row_corrcoef(x_train_np)

    adj = (
        np.abs(corr) >= CORR_THRESHOLD
    ).astype(np.float32)

    np.fill_diagonal(adj, 1.0)

    return adj

def build_inductive_eval_adjacency(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
):
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]
    n_total = n_train + n_test

    x_all = np.concatenate(
        [x_train_np, x_test_np],
        axis=0,
    )

    corr = safe_row_corrcoef(x_all)

    adj = np.zeros(
        (n_total, n_total),
        dtype=np.float32,
    )

    if n_train > 0:

        corr_tt = corr[:n_train, :n_train]
        adj[:n_train, :n_train] = (
            np.abs(corr_tt) >= CORR_THRESHOLD
        ).astype(np.float32)


        corr_test_train = corr[n_train:, :n_train]
        adj[n_train:, :n_train] = (
            np.abs(corr_test_train) >= CORR_THRESHOLD
        ).astype(np.float32)

    np.fill_diagonal(adj, 1.0)

    return x_all, adj

def normalized_adjacency(
    adj_np: np.ndarray,
) -> torch.Tensor:
    A = torch.tensor(
        adj_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    row_deg = A.sum(dim=1).clamp_min(1.0)
    col_deg = A.sum(dim=0).clamp_min(1.0)

    row_scale = row_deg.pow(-0.5).unsqueeze(1)
    col_scale = col_deg.pow(-0.5).unsqueeze(0)

    return row_scale * A * col_scale

@torch.no_grad()
def topology_aware_embeddings(
    x_np: np.ndarray,
    adj_np: np.ndarray,
    L: int = PDGNN_L,
) -> torch.Tensor:
    x = torch.tensor(
        x_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    A_hat = normalized_adjacency(adj_np)

    e = x

    for _ in range(L):
        e = A_hat @ e

    return e.detach()

class PDGNNClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            in_dim,
            hidden_dim,
            bias=LINEAR_BIAS,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            num_classes,
            bias=LINEAR_BIAS,
        )

    def forward(
        self,
        te: torch.Tensor,
    ):
        h = self.fc1(te)
        h = F.relu(h)

        h = F.dropout(
            h,
            p=MLP_DROPOUT,
            training=self.training,
        )

        logits = self.fc2(h)

        return logits, h

class TEMBuffer:
    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        seed: int,
    ):
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.rng = np.random.default_rng(seed)

        self.vecs = torch.empty(
            (0, feature_dim),
            dtype=torch.float32,
            device=DEVICE,
        )

        self.labels = torch.empty(
            (0,),
            dtype=torch.long,
            device=DEVICE,
        )

        self.n_seen_selected = 0

    def __len__(self):
        return int(self.labels.numel())

    @torch.no_grad()
    def add(
        self,
        vecs: torch.Tensor,
        labels_t: torch.Tensor,
    ):
        if vecs.numel() == 0:
            return

        for i in range(vecs.size(0)):
            self.n_seen_selected += 1

            v = vecs[i:i + 1].detach()
            y = labels_t[i:i + 1].detach()

            if len(self) < self.capacity:
                self.vecs = torch.cat(
                    [self.vecs, v],
                    dim=0,
                )

                self.labels = torch.cat(
                    [self.labels, y],
                    dim=0,
                )

            else:
                j = int(
                    self.rng.integers(
                        0,
                        self.n_seen_selected,
                    )
                )

                if j < self.capacity:
                    self.vecs[j] = v[0]
                    self.labels[j] = y[0]

def exact_two_hop_coverage_counts(
    adj_np: np.ndarray,
) -> np.ndarray:
    reach = adj_np.astype(bool)

    if PDGNN_L <= 1:
        return reach.sum(
            axis=1
        ).astype(np.float64)

    reach_l = reach.copy()
    base = reach.astype(np.uint8)

    for _ in range(1, PDGNN_L):
        reach_l = (
            reach_l.astype(np.uint8) @ base
        ) > 0

    return reach_l.sum(
        axis=1
    ).astype(np.float64)


def select_tem_candidates(
    candidate_local_ids: np.ndarray,
    y_train_np: np.ndarray,
    adj_np: np.ndarray,
    rng: np.random.Generator,
):
    candidate_local_ids = np.asarray(
        candidate_local_ids,
        dtype=np.int64,
    )

    if candidate_local_ids.size == 0:
        return np.empty(
            (0,),
            dtype=np.int64,
        )

    if TEM_SAMPLER == "degree":
        scores_all = adj_np.sum(
            axis=1
        ).astype(np.float64)

    elif TEM_SAMPLER == "exact_coverage":
        scores_all = exact_two_hop_coverage_counts(
            adj_np
        )

    else:
        raise ValueError(
            "TEM_SAMPLER must be "
            "'degree' or 'exact_coverage'."
        )

    selected = []

    current_labels = y_train_np[
        candidate_local_ids
    ]

    for cls_id in np.unique(current_labels):
        cls_ids = candidate_local_ids[
            current_labels == cls_id
        ]

        if cls_ids.size == 0:
            continue

        k = max(
            1,
            int(
                np.ceil(
                    TEM_STORE_RATIO * cls_ids.size
                )
            ),
        )

        k = min(
            k,
            cls_ids.size,
        )

        scores = scores_all[
            cls_ids
        ].astype(np.float64)

        scores = np.maximum(
            scores,
            1e-12,
        )

        probs = scores / scores.sum()

        chosen = rng.choice(
            cls_ids,
            size=k,
            replace=False,
            p=probs,
        )

        selected.extend(
            chosen.tolist()
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )

def class_balanced_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
):
    counts = torch.bincount(
        y,
        minlength=NUM_CLASSES,
    ).float()

    weights = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float32,
        device=DEVICE,
    )

    present = counts > 0

    weights[present] = (
        1.0 / counts[present]
    )

    return F.cross_entropy(
        logits,
        y,
        weight=weights,
    )
features_window = deque(
    maxlen=WINDOW_SIZE
)
labels_window = deque(
    maxlen=WINDOW_SIZE
)
index_window = deque(
    maxlen=WINDOW_SIZE
)

model = None
optimizer = None
tem = None

global_idx = 0
session_idx = 0

y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []

sampling_rng = np.random.default_rng(
    SEED + 2024
)

print("\n=== Streaming PDGNN + TEM Training ===")

for start in tqdm(
    range(0, N, BATCH_SIZE),
    desc="Processing batches",
):
    end = min(
        start + BATCH_SIZE,
        N,
    )

    batch_feats = features[start:end]
    batch_labels = labels[start:end]
    bsz = len(batch_feats)

    for i in range(bsz):
        features_window.append(
            batch_feats[i]
        )

        labels_window.append(
            int(batch_labels[i])
        )

        index_window.append(
            global_idx
        )

        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    first_window = (
        model is None
    )

    x_win_np = np.asarray(
        features_window,
        dtype=np.float64,
    )

    y_win_np = np.asarray(
        labels_window,
        dtype=np.int64,
    )

    idx_win_np = np.asarray(
        index_window,
        dtype=np.int64,
    )

    new_mask_full = np.zeros(
        len(x_win_np),
        dtype=bool,
    )

    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[
        idx_win_np
    ]

    test_mask_full = is_test[
        idx_win_np
    ]

    x_train_np = x_win_np[
        train_mask_full
    ]

    y_train_np = y_win_np[
        train_mask_full
    ]

    new_train_mask_local_np = new_mask_full[
        train_mask_full
    ]

    if x_train_np.shape[0] == 0:
        continue

    train_adj_np = build_train_adjacency(
        x_train_np
    )

    topo_train = topology_aware_embeddings(
        x_train_np,
        train_adj_np,
        L=PDGNN_L,
    )

    if model is None:
        model = PDGNNClassifier(
            in_dim=topo_train.size(1),
            hidden_dim=PDGNN_HIDDEN,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        tem = TEMBuffer(
            capacity=TEM_CAPACITY,
            feature_dim=topo_train.size(1),
            seed=SEED + 77,
        )

    if first_window:
        current_ids = np.arange(
            len(x_train_np),
            dtype=np.int64,
        )

        epochs_now = EPOCHS_FIRST

    else:
        current_ids = np.where(
            new_train_mask_local_np
        )[0].astype(np.int64)

        epochs_now = EPOCHS_INC

    if current_ids.size > 0:
        current_ids_t = torch.tensor(
            current_ids,
            dtype=torch.long,
            device=DEVICE,
        )

        current_te = topo_train[
            current_ids_t
        ]

        current_y = torch.tensor(
            y_train_np[current_ids],
            dtype=torch.long,
            device=DEVICE,
        )


        if len(tem) > 0:
            train_te = torch.cat(
                [current_te, tem.vecs],
                dim=0,
            )

            train_y = torch.cat(
                [current_y, tem.labels],
                dim=0,
            )

        else:
            train_te = current_te
            train_y = current_y

        for _ in range(epochs_now):
            model.train()

            logits, _ = model(
                train_te
            )

            loss = class_balanced_ce(
                logits,
                train_y,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

        selected_ids = select_tem_candidates(
            candidate_local_ids=current_ids,
            y_train_np=y_train_np,
            adj_np=train_adj_np,
            rng=sampling_rng,
        )

        if selected_ids.size > 0:
            selected_t = torch.tensor(
                selected_ids,
                dtype=torch.long,
                device=DEVICE,
            )

            selected_y = torch.tensor(
                y_train_np[selected_ids],
                dtype=torch.long,
                device=DEVICE,
            )

            tem.add(
                topo_train[selected_t],
                selected_y,
            )

    session_idx += 1

    if first_window:
        eval_test_mask_full = (
            test_mask_full
        )
    else:
        eval_test_mask_full = (
            new_mask_full
            & test_mask_full
        )

    if eval_test_mask_full.any():
        x_new_test_np = x_win_np[
            eval_test_mask_full
        ]

        y_new_test_np = y_win_np[
            eval_test_mask_full
        ]

        x_eval_np, eval_adj_np = (
            build_inductive_eval_adjacency(
                x_train_np,
                x_new_test_np,
            )
        )

        topo_eval = topology_aware_embeddings(
            x_eval_np,
            eval_adj_np,
            L=PDGNN_L,
        )

        n_train_context = (
            x_train_np.shape[0]
        )

        test_te = topo_eval[
            n_train_context:
        ]

        model.eval()

        with torch.no_grad():
            (
                logits_test,
                hidden_test_batch,
            ) = model(test_te)

            probs_test = torch.softmax(
                logits_test,
                dim=1,
            )

        y_true_test.extend(
            y_new_test_np.tolist()
        )

        y_pred_test.extend(
            torch.argmax(
                logits_test,
                dim=1,
            )
            .cpu()
            .numpy()
            .tolist()
        )

        y_prob_test_all.append(
            probs_test
            .cpu()
            .numpy()
        )

        hidden_test.extend(
            hidden_test_batch
            .cpu()
            .numpy()
            .tolist()
        )

if model is None:
    raise RuntimeError(
        "The model was not initialized. Ensure WINDOW_SIZE "
        "is smaller than the number of valid samples."
    )

print("\n=== PDGNN + TEM Diagnostics ===")
print(f"Streaming sessions: {session_idx}")
print(
    f"TEM current size: "
    f"{len(tem)} / {TEM_CAPACITY}"
)
print(
    f"TEM selected candidates seen: "
    f"{tem.n_seen_selected}"
)

y_true_test = np.asarray(
    y_true_test,
    dtype=np.int64,
)

y_pred_test = np.asarray(
    y_pred_test,
    dtype=np.int64,
)

if len(y_prob_test_all) > 0:
    y_prob_test_all = np.vstack(
        y_prob_test_all
    )
else:
    y_prob_test_all = np.empty(
        (0, NUM_CLASSES),
        dtype=np.float64,
    )

print("\n=== Evaluation on Random Test Split (15%) ===")
print(
    f"Expected hold-out samples: "
    f"{len(test_idx)}"
)
print(
    f"Actually evaluated samples: "
    f"{len(y_true_test)}"
)

if len(y_true_test) == 0:
    raise RuntimeError(
        "The test set is empty or no test samples were evaluated."
    )

unique_ids = np.arange(
    NUM_CLASSES
)

target_names = [
    id_to_label[i]
    for i in unique_ids
]

report = classification_report(
    y_true_test,
    y_pred_test,
    output_dict=True,
    digits=4,
    labels=unique_ids,
    target_names=target_names,
    zero_division=0,
)

print("\nClassification Report:")

for lab in target_names:
    m = report[lab]

    print(
        f"{lab:<28} "
        f"precision: {m['precision'] * 100:.2f}%  "
        f"recall: {m['recall'] * 100:.2f}%  "
        f"f1-score: {m['f1-score'] * 100:.2f}%"
    )

print(
    f"{'accuracy':<28}: "
    f"{report['accuracy'] * 100:.2f}%"
)

for k in [
    'macro avg',
    'weighted avg',
]:
    m = report[k]

    print(
        f"{k:<28} "
        f"precision: {m['precision'] * 100:.2f}%  "
        f"recall: {m['recall'] * 100:.2f}%  "
        f"f1-score: {m['f1-score'] * 100:.2f}%"
    )

try:
    cm = confusion_matrix(
        y_true_test,
        y_pred_test,
        labels=unique_ids,
    )

    fig_cm, ax_cm = plt.subplots(
        figsize=(7.0, 6.0)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    disp.plot(
        values_format='d',
        cmap='Blues',
        colorbar=False,
        ax=ax_cm,
    )

    plt.setp(
        ax_cm.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor',
    )

    ax_cm.set_xlabel(
        'Predicted label'
    )

    ax_cm.set_ylabel(
        'True label'
    )

    fig_cm.tight_layout()
    plt.show()
    plt.close(fig_cm)

except Exception as e:
    print(
        "Error plotting the confusion matrix:",
        e,
    )

try:
    if NUM_CLASSES > 2:
        y_true_bin = label_binarize(
            y_true_test,
            classes=unique_ids.tolist(),
        )

        fig_roc, ax_roc = plt.subplots(
            figsize=(7.2, 5.4)
        )

        plotted = False

        for cls_id in unique_ids:
            col = int(cls_id)

            y_true_c = y_true_bin[:, col]
            y_score_c = y_prob_test_all[:, col]

            if (
                y_true_c.sum() == 0
                or y_true_c.sum() == len(y_true_c)
            ):
                continue

            fpr, tpr, _ = roc_curve(
                y_true_c,
                y_score_c,
            )

            roc_auc = auc(
                fpr,
                tpr,
            )

            ax_roc.plot(
                fpr,
                tpr,
                label=(
                    f"{id_to_label[int(cls_id)]} "
                    f"(AUC={roc_auc:.3f})"
                ),
            )

            plotted = True

        if plotted:
            ax_roc.set_xlabel(
                'False Positive Rate'
            )

            ax_roc.set_ylabel(
                'True Positive Rate'
            )

            ax_roc.legend(
                fontsize=8,
                loc='lower right',
            )

            fig_roc.tight_layout()
            plt.show()

        plt.close(fig_roc)

    elif NUM_CLASSES == 2:
        if len(np.unique(y_true_test)) == 2:
            fpr, tpr, _ = roc_curve(
                y_true_test,
                y_prob_test_all[:, 1],
            )

            roc_auc = auc(
                fpr,
                tpr,
            )

            print(
                f"ROC-AUC: "
                f"{roc_auc * 100:.2f}%"
            )

except Exception as e:
    print(
        "Error computing or plotting ROC curves:",
        e,
    )

try:
    hidden_np = np.asarray(
        hidden_test,
        dtype=np.float64,
    )

    if len(hidden_np) > 10:
        labels_tsne = y_true_test[
            :len(hidden_np)
        ]

        finite_mask = np.isfinite(
            hidden_np
        ).all(axis=1)

        hidden_np = hidden_np[
            finite_mask
        ]

        labels_tsne = labels_tsne[
            finite_mask
        ]

        rng = np.random.default_rng(
            SEED
        )

        selected = []

        for cls_id in np.unique(
            labels_tsne
        ):
            cls_idx = np.where(
                labels_tsne == cls_id
            )[0]

            if len(cls_idx) > TSNE_MAX_PER_CLASS:
                cls_idx = rng.choice(
                    cls_idx,
                    size=TSNE_MAX_PER_CLASS,
                    replace=False,
                )

            selected.append(
                np.asarray(
                    cls_idx,
                    dtype=np.int64,
                )
            )

        selected = np.concatenate(
            selected
        )

        selected.sort()

        x_tsne = hidden_np[
            selected
        ]

        y_tsne = labels_tsne[
            selected
        ]

        x_tsne = StandardScaler().fit_transform(
            x_tsne
        )

        x_tsne = np.nan_to_num(
            x_tsne,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        n_tsne = len(x_tsne)

        perplexity = min(
            30.0,
            max(
                5.0,
                (n_tsne - 1) / 3.0,
            ),
        )

        perplexity = min(
            perplexity,
            float(n_tsne - 1),
        )

        emb_2d = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
        ).fit_transform(
            x_tsne
        )

        fig_tsne, ax_tsne = plt.subplots(
            figsize=(7.0, 6.0)
        )

        for cls_id in np.unique(
            y_tsne
        ):
            mask = (
                y_tsne == cls_id
            )

            ax_tsne.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=12,
                alpha=0.75,
                label=id_to_label.get(
                    int(cls_id),
                    str(cls_id),
                ),
            )

        ax_tsne.set_xlabel(
            't-SNE Dim 1'
        )

        ax_tsne.set_ylabel(
            't-SNE Dim 2'
        )

        ax_tsne.set_title(
            'PDGNN-TEM Test Hidden Embeddings'
        )

        ax_tsne.legend(
            fontsize=7,
            markerscale=1.5,
            loc='best',
        )

        fig_tsne.tight_layout()
        plt.show()
        plt.close(fig_tsne)

    else:
        print(
            "t-SNE: Too few test hidden vectors; "
            "skipping visualization."
        )

except Exception as e:
    print(
        "t-SNE visualization failed:",
        e,
    )
