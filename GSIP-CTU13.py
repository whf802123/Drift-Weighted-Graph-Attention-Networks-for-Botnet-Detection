

import warnings
warnings.filterwarnings("ignore")

from collections import deque
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 0. Configuration
# ============================================================
CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'

WINDOW_SIZE = 1000
BATCH_SIZE = 100

EPOCHS_FIRST = 10
EPOCHS_INC = 2

CORR_THRESHOLD = 0.1
TEST_RATIO = 0.30
SEED = 42

# Official GSIP/ERGNN backbone defaults.
HIDDEN_DIM = 256
LR = 5e-3
WEIGHT_DECAY = 5e-4

# Replay: official ERGNN examples use budget=100 per class/task.
# In the streaming adaptation we keep a bounded 100 samples/class.
BUFFER_PER_CLASS = 100
CM_DISTANCE = 0.5

# GSIP similarity thresholds.
NEIBT_LL = 0.99

# The official CoraFull+GCN command uses neibt1=0.5 and
# w_ll=50, w_lg=0.05, w_h=10. CTU13 is not an official GSIP dataset,
# so these are starting values and should ideally be tuned on validation data.
NEIBT_H = 0.5
W_LL = 50.0
W_LG = 0.05
W_H = 10.0

GSIP_TEMPERATURE = 1.0

# t-SNE only.
TSNE_MAX_PER_CLASS = 1000

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1. Data loading and preprocessing
# ============================================================
df = pd.read_csv(CSV_PATH)

if 'Label' not in df.columns:
    raise KeyError(
        "未找到列名 'Label'。请确认 CTU13 CSV 中标签列名为 Label。"
    )

feature_cols = [
    c for c in df.columns
    if c not in ('num', 'Label')
]

if len(feature_cols) == 0:
    raise RuntimeError("未找到可用特征列。")

labels = df['Label'].astype(int).to_numpy(
    dtype=np.int64
)

unique_labels = np.unique(labels)

if len(unique_labels) != 2 or set(unique_labels.tolist()) != {0, 1}:
    raise RuntimeError(
        f"当前代码按 CTU13 二分类 0/1 设置，实际标签为 "
        f"{unique_labels.tolist()}。"
    )

id_to_label = {
    0: 'Normal',
    1: 'Botnet',
}

feat_df = df[feature_cols].apply(
    pd.to_numeric,
    errors='coerce',
)

N_total = len(df)
all_idx = np.arange(
    N_total,
    dtype=np.int64,
)

train_idx, test_idx = train_test_split(
    all_idx,
    test_size=TEST_RATIO,
    stratify=labels,
    random_state=SEED,
    shuffle=True,
)

# Train-only missing-value statistics.
medians = feat_df.iloc[
    train_idx
].median(numeric_only=True)

feat_df = (
    feat_df
    .fillna(medians)
    .fillna(0.0)
)

# Train-only normalization.
scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

features = np.empty_like(
    feat_df.values,
    dtype=np.float64,
)

features[train_idx] = scaler.fit_transform(
    feat_df.iloc[
        train_idx
    ].values.astype(float)
)

features[test_idx] = scaler.transform(
    feat_df.iloc[
        test_idx
    ].values.astype(float)
)

N = len(features)

is_train = np.zeros(
    N,
    dtype=bool,
)
is_train[train_idx] = True
is_test = ~is_train

NUM_CLASSES = len(
    np.unique(labels[train_idx])
)

if NUM_CLASSES != 2:
    raise RuntimeError(
        "训练集未同时包含 Normal 和 Botnet 两类。"
    )

print("Device:", DEVICE)
print("CTU13 label mapping:", id_to_label)
print(
    f"Samples: total={N_total}, "
    f"train={len(train_idx)}, "
    f"test={len(test_idx)}"
)
print(f"Features: {len(feature_cols)}")
print(
    "GSIP settings: "
    f"buffer/class={BUFFER_PER_CLASS}, "
    f"neibt_ll={NEIBT_LL}, neibt_h={NEIBT_H}, "
    f"w_ll={W_LL}, w_lg={W_LG}, w_h={W_H}"
)


# ============================================================
# 2. Pearson graph construction
# ============================================================
def safe_row_corrcoef(
    x_np: np.ndarray,
) -> np.ndarray:
    if x_np.shape[0] == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    if x_np.shape[0] == 1:
        return np.ones(
            (1, 1),
            dtype=np.float64,
        )

    corr = np.corrcoef(x_np)

    return np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_train_graph(
    x_train_np: np.ndarray,
):
    """
    Graph contains only current-window TRAIN nodes.

    Pearson determines connectivity.
    GCNConv adds self-loops internally.
    """
    n = x_train_np.shape[0]

    if n <= 1:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=DEVICE,
        )

    corr = safe_row_corrcoef(
        x_train_np
    )

    adj = (
        np.abs(corr)
        >= CORR_THRESHOLD
    )

    np.fill_diagonal(
        adj,
        False,
    )

    src, dst = np.where(adj)

    return torch.tensor(
        np.vstack([
            src.astype(np.int64),
            dst.astype(np.int64),
        ]),
        dtype=torch.long,
        device=DEVICE,
    )


def build_inductive_eval_graph(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
):
    """
    Node order:
      [current-window training context, test nodes]

    Allowed:
      train -> train
      train -> test

    Self-loops are added internally by GCNConv.

    Forbidden:
      test -> train
      test -> test
    """
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]

    x_eval_np = np.concatenate(
        [
            x_train_np,
            x_test_np,
        ],
        axis=0,
    )

    corr = safe_row_corrcoef(
        x_eval_np
    )

    src_list = []
    dst_list = []

    if n_train > 0:
        corr_tt = corr[
            :n_train,
            :n_train,
        ]

        adj_tt = (
            np.abs(corr_tt)
            >= CORR_THRESHOLD
        )

        np.fill_diagonal(
            adj_tt,
            False,
        )

        src_tt, dst_tt = np.where(
            adj_tt
        )

        if src_tt.size > 0:
            src_list.append(
                src_tt.astype(np.int64)
            )
            dst_list.append(
                dst_tt.astype(np.int64)
            )

        # train -> test.
        # corr[:n_train, n_train:] rows=train, cols=test.
        corr_train_test = corr[
            :n_train,
            n_train:
        ]

        train_src, test_col = np.where(
            np.abs(corr_train_test)
            >= CORR_THRESHOLD
        )

        if train_src.size > 0:
            src_list.append(
                train_src.astype(np.int64)
            )
            dst_list.append(
                (
                    n_train
                    + test_col
                ).astype(np.int64)
            )

    if len(src_list) == 0:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=DEVICE,
        )
    else:
        edge_index = torch.tensor(
            np.vstack([
                np.concatenate(src_list),
                np.concatenate(dst_list),
            ]),
            dtype=torch.long,
            device=DEVICE,
        )

    x_eval_tensor = torch.tensor(
        x_eval_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    return (
        x_eval_tensor,
        edge_index,
    )


# ============================================================
# 3. GCN backbone
# ============================================================
class GSIPGCN(nn.Module):
    """
    Two-layer GCN with 256-d hidden representation.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
    ):
        super().__init__()

        self.gcn1 = GCNConv(
            in_channels,
            hidden_channels,
            add_self_loops=True,
            normalize=True,
        )

        self.gcn2 = GCNConv(
            hidden_channels,
            num_classes,
            add_self_loops=True,
            normalize=True,
        )

    def forward(
        self,
        x,
        edge_index,
    ):
        hidden = self.gcn1(
            x,
            edge_index,
        )

        hidden = F.relu(
            hidden
        )

        logits = self.gcn2(
            hidden,
            edge_index,
        )

        return logits, hidden


# ============================================================
# 4. ERGNN-style CM replay buffer
# ============================================================
class ClassBalancedCMBuffer:
    """
    Bounded class-wise replay memory.

    CM follows the official ERGNN sampler logic:
    for samples of one class, count how many samples of other classes lie
    within distance d and rank samples by that count in ascending order.

    Because the original GSIP protocol receives discrete class tasks and
    extends memory per task, while CTU13 receives endless windows with fixed
    classes, this adaptation re-selects a fixed budget per class from:
        previous buffer + newly arrived training samples.
    """
    def __init__(
        self,
        per_class_budget,
        d,
        seed,
    ):
        self.per_class_budget = int(
            per_class_budget
        )
        self.d = float(d)
        self.rng = random.Random(
            seed
        )
        self.ids = []

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return list(self.ids)

    def update(
        self,
        new_global_ids,
        features_all,
        labels_all,
    ):
        candidate = sorted(
            set(
                self.ids
                + [
                    int(x)
                    for x in new_global_ids
                ]
            )
        )

        if len(candidate) == 0:
            return

        candidate_np = np.asarray(
            candidate,
            dtype=np.int64,
        )

        x = torch.tensor(
            features_all[candidate_np],
            dtype=torch.float32,
            device=DEVICE,
        )

        y = labels_all[
            candidate_np
        ]

        selected_global_ids = []

        for cls_id in range(NUM_CLASSES):
            cls_local = np.where(
                y == cls_id
            )[0]

            if len(cls_local) == 0:
                continue

            other_local = np.where(
                y != cls_id
            )[0]

            k = min(
                self.per_class_budget,
                len(cls_local),
            )

            if len(other_local) == 0:
                chosen_local = cls_local[
                    :k
                ]
            else:
                x_cls = x[
                    torch.tensor(
                        cls_local,
                        dtype=torch.long,
                        device=DEVICE,
                    )
                ]

                x_other = x[
                    torch.tensor(
                        other_local,
                        dtype=torch.long,
                        device=DEVICE,
                    )
                ]

                dist = torch.cdist(
                    x_cls,
                    x_other,
                )

                counts = (
                    dist < self.d
                ).sum(
                    dim=1
                )

                order = torch.argsort(
                    counts,
                    descending=False,
                )

                chosen_local = cls_local[
                    order[
                        :k
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                ]

            selected_global_ids.extend(
                candidate_np[
                    chosen_local
                ].tolist()
            )

        self.ids = [
            int(x)
            for x in selected_global_ids
        ]


# ============================================================
# 5. Loss helpers
# ============================================================
def balanced_ce(
    logits,
    y,
):
    """
    Per-batch inverse-frequency class weighting, following the official
    GSIP/ERGNN implementation's class balancing.
    """
    if y.numel() == 0:
        return None

    counts = torch.bincount(
        y,
        minlength=NUM_CLASSES,
    ).float()

    weights = torch.ones(
        NUM_CLASSES,
        dtype=torch.float32,
        device=DEVICE,
    )

    present = counts > 0

    weights[present] = (
        1.0
        / counts[present]
    )

    return F.cross_entropy(
        logits,
        y,
        weight=weights,
    )


def gsip_losses(
    new_logits,
    old_logits,
    replay_labels,
):
    """
    Returns:
      loss_ll : low-frequency local preservation
      loss_lg : low-frequency global preservation
      loss_h  : high-frequency preservation

    This follows the official ERGNN+GSIP code structure.
    """
    device = new_logits.device
    zero = torch.zeros(
        (),
        dtype=new_logits.dtype,
        device=device,
    )

    if new_logits.size(0) == 0:
        return zero, zero, zero

    old_detached = old_logits.detach()

    # Old-model pair similarity.
    sim_old = F.cosine_similarity(
        old_detached.unsqueeze(1),
        old_detached.unsqueeze(0),
        dim=2,
    )

    # --------------------------------------------------------
    # Low-frequency local preservation
    # --------------------------------------------------------
    same_label = torch.eq(
        replay_labels.unsqueeze(1),
        replay_labels.unsqueeze(0),
    )

    local_mask = (
        sim_old > NEIBT_LL
    ) & same_label

    local_pairs = torch.nonzero(
        local_mask,
        as_tuple=False,
    )

    if (
        W_LL != 0.0
        and local_pairs.numel() > 0
    ):
        src_old = local_pairs[:, 0]
        dst_new = local_pairs[:, 1]

        loss_ll = F.mse_loss(
            new_logits[
                dst_new
            ],
            old_detached[
                src_old
            ],
        )
    else:
        loss_ll = zero

    # --------------------------------------------------------
    # Low-frequency global preservation
    # --------------------------------------------------------
    if W_LG != 0.0:
        loss_lg = F.mse_loss(
            new_logits.mean(
                dim=0
            ),
            old_detached.mean(
                dim=0
            ),
        )
    else:
        loss_lg = zero

    # --------------------------------------------------------
    # High-frequency preservation
    # --------------------------------------------------------
    high_mask = (
        sim_old > NEIBT_H
    )

    high_pairs = torch.nonzero(
        high_mask,
        as_tuple=False,
    )

    if (
        W_H != 0.0
        and high_pairs.numel() > 0
    ):
        i = high_pairs[:, 0]
        j = high_pairs[:, 1]

        diff_new = torch.abs(
            new_logits[i]
            - new_logits[j]
        )

        diff_old = torch.abs(
            old_detached[i]
            - old_detached[j]
        )

        loss_h = F.kl_div(
            F.log_softmax(
                diff_new
                / GSIP_TEMPERATURE,
                dim=-1,
            ),
            F.softmax(
                diff_old
                / GSIP_TEMPERATURE,
                dim=-1,
            ),
            reduction='batchmean',
        )
    else:
        loss_h = zero

    return (
        loss_ll,
        loss_lg,
        loss_h,
    )


# ============================================================
# 6. Streaming state
# ============================================================
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

replay_buffer = ClassBalancedCMBuffer(
    per_class_budget=BUFFER_PER_CLASS,
    d=CM_DISTANCE,
    seed=SEED + 31,
)

global_idx = 0
session_idx = 0

# Final progressive hold-out collection.
y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []

# Window-level performance curve.
window_perf_ids = []
window_perf_n_test = []
window_perf_accuracy = []
window_perf_precision = []
window_perf_recall = []
window_perf_f1 = []

# GSIP diagnostics.
gsip_ll_values = []
gsip_lg_values = []
gsip_h_values = []


# ============================================================
# 7. Main streaming loop
# ============================================================
print("\n=== Streaming GSIP Training ===")

for start in tqdm(
    range(
        0,
        N,
        BATCH_SIZE,
    ),
    desc='Processing batches',
):
    batch_feats = features[
        start:
        start + BATCH_SIZE
    ]

    batch_labels = labels[
        start:
        start + BATCH_SIZE
    ]

    bsz = len(
        batch_feats
    )

    for i in range(bsz):
        features_window.append(
            batch_feats[i]
        )
        labels_window.append(
            int(
                batch_labels[i]
            )
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

    session_idx += 1

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

    # Newly arrived positions.
    new_mask_full = np.zeros(
        len(x_win_np),
        dtype=bool,
    )

    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[
            -bsz:
        ] = True

    train_mask_full = is_train[
        idx_win_np
    ]

    test_mask_full = is_test[
        idx_win_np
    ]

    # --------------------------------------------------------
    # 7.1 Current training graph
    # --------------------------------------------------------
    x_train_np = x_win_np[
        train_mask_full
    ]

    y_train_np = y_win_np[
        train_mask_full
    ]

    train_global_ids = idx_win_np[
        train_mask_full
    ]

    new_train_mask_local_np = new_mask_full[
        train_mask_full
    ]

    if x_train_np.shape[0] == 0:
        continue

    x_train_tensor = torch.tensor(
        x_train_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_train_tensor = torch.tensor(
        y_train_np,
        dtype=torch.long,
        device=DEVICE,
    )

    train_edge_index = build_train_graph(
        x_train_np
    )

    if model is None:
        model = GSIPGCN(
            in_channels=x_train_tensor.size(1),
            hidden_channels=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        used_idx = torch.arange(
            x_train_tensor.size(0),
            dtype=torch.long,
            device=DEVICE,
        )

        epochs_now = EPOCHS_FIRST

        prev_model = None

    else:
        # Freeze the previous-session model before the current update.
        prev_model = copy.deepcopy(
            model
        ).to(DEVICE)

        prev_model.eval()

        for p in prev_model.parameters():
            p.requires_grad_(False)

        used_idx_np = np.where(
            new_train_mask_local_np
        )[0].astype(np.int64)

        if len(used_idx_np) == 0:
            used_idx = None
        else:
            used_idx = torch.tensor(
                used_idx_np,
                dtype=torch.long,
                device=DEVICE,
            )

        epochs_now = EPOCHS_INC

    # Historical replay graph is fixed during this session.
    old_buffer_ids = replay_buffer.get_ids()

    if len(old_buffer_ids) > 0:
        old_buffer_ids_np = np.asarray(
            old_buffer_ids,
            dtype=np.int64,
        )

        x_replay_np = features[
            old_buffer_ids_np
        ]

        y_replay_np = labels[
            old_buffer_ids_np
        ]

        x_replay_tensor = torch.tensor(
            x_replay_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        y_replay_tensor = torch.tensor(
            y_replay_np,
            dtype=torch.long,
            device=DEVICE,
        )

        replay_edge_index = build_train_graph(
            x_replay_np
        )

        if prev_model is not None:
            with torch.no_grad():
                old_replay_logits, _ = prev_model(
                    x_replay_tensor,
                    replay_edge_index,
                )
        else:
            old_replay_logits = None

    else:
        x_replay_tensor = None
        y_replay_tensor = None
        replay_edge_index = None
        old_replay_logits = None

    # --------------------------------------------------------
    # 7.2 Current session optimization
    # --------------------------------------------------------
    for _ in range(epochs_now):
        if used_idx is None:
            break

        model.train()

        current_logits, _ = model(
            x_train_tensor,
            train_edge_index,
        )

        loss_current = balanced_ce(
            current_logits[
                used_idx
            ],
            y_train_tensor[
                used_idx
            ],
        )

        if loss_current is None:
            continue

        total_loss = loss_current

        if (
            prev_model is not None
            and x_replay_tensor is not None
            and x_replay_tensor.size(0) > 0
        ):
            new_replay_logits, _ = model(
                x_replay_tensor,
                replay_edge_index,
            )

            loss_replay = balanced_ce(
                new_replay_logits,
                y_replay_tensor,
            )

            (
                loss_ll,
                loss_lg,
                loss_h,
            ) = gsip_losses(
                new_logits=new_replay_logits,
                old_logits=old_replay_logits,
                replay_labels=y_replay_tensor,
            )

            n_current = int(
                used_idx.numel()
            )
            n_buffer = int(
                y_replay_tensor.numel()
            )

            beta = (
                n_buffer
                / max(
                    1,
                    n_buffer
                    + n_current,
                )
            )

            total_loss = (
                beta
                * loss_current
                + (1.0 - beta)
                * loss_replay
                + W_LL
                * loss_ll
                + W_LG
                * loss_lg
                + W_H
                * loss_h
            )

            gsip_ll_values.append(
                float(
                    loss_ll
                    .detach()
                    .cpu()
                )
            )
            gsip_lg_values.append(
                float(
                    loss_lg
                    .detach()
                    .cpu()
                )
            )
            gsip_h_values.append(
                float(
                    loss_h
                    .detach()
                    .cpu()
                )
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()
        optimizer.step()

    # --------------------------------------------------------
    # 7.3 Progressive hold-out evaluation
    # --------------------------------------------------------
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

        (
            x_eval_tensor,
            eval_edge_index,
        ) = build_inductive_eval_graph(
            x_train_np,
            x_new_test_np,
        )

        model.eval()

        with torch.no_grad():
            (
                logits_eval,
                hidden_eval,
            ) = model(
                x_eval_tensor,
                eval_edge_index,
            )

            n_train_context = (
                x_train_np.shape[0]
            )

            logits_test = logits_eval[
                n_train_context:
            ]

            hidden_test_batch = hidden_eval[
                n_train_context:
            ]

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

    # --------------------------------------------------------
    # 7.4 Whole-current-window performance curve
    # --------------------------------------------------------
    if test_mask_full.any():
        x_curve_test_np = x_win_np[
            test_mask_full
        ]

        y_curve_test_np = y_win_np[
            test_mask_full
        ]

        (
            x_curve_tensor,
            curve_edge_index,
        ) = build_inductive_eval_graph(
            x_train_np,
            x_curve_test_np,
        )

        model.eval()

        with torch.no_grad():
            logits_curve, _ = model(
                x_curve_tensor,
                curve_edge_index,
            )

            n_train_context = (
                x_train_np.shape[0]
            )

            pred_curve = torch.argmax(
                logits_curve[
                    n_train_context:
                ],
                dim=1,
            ).cpu().numpy()

        window_perf_ids.append(
            session_idx
        )

        window_perf_n_test.append(
            len(
                y_curve_test_np
            )
        )

        window_perf_accuracy.append(
            accuracy_score(
                y_curve_test_np,
                pred_curve,
            )
        )

        window_perf_precision.append(
            precision_score(
                y_curve_test_np,
                pred_curve,
                average='macro',
                labels=np.arange(
                    NUM_CLASSES
                ),
                zero_division=0,
            )
        )

        window_perf_recall.append(
            recall_score(
                y_curve_test_np,
                pred_curve,
                average='macro',
                labels=np.arange(
                    NUM_CLASSES
                ),
                zero_division=0,
            )
        )

        window_perf_f1.append(
            f1_score(
                y_curve_test_np,
                pred_curve,
                average='macro',
                labels=np.arange(
                    NUM_CLASSES
                ),
                zero_division=0,
            )
        )

    # --------------------------------------------------------
    # 7.5 Update replay memory AFTER the current session
    # --------------------------------------------------------
    if first_window:
        new_train_global_ids = (
            train_global_ids
        )
    else:
        new_train_global_ids = (
            train_global_ids[
                new_train_mask_local_np
            ]
        )

    replay_buffer.update(
        new_global_ids=new_train_global_ids,
        features_all=features,
        labels_all=labels,
    )


if model is None:
    raise RuntimeError(
        "模型未初始化。请检查 WINDOW_SIZE 是否小于有效数据量。"
    )


# ============================================================
# 8. GSIP diagnostics
# ============================================================
print("\n=== GSIP Diagnostics ===")
print(
    f"Continual sessions: "
    f"{session_idx}"
)
print(
    f"Replay buffer size: "
    f"{len(replay_buffer)}"
)

if gsip_ll_values:
    print(
        "Mean unweighted GSIP losses: "
        f"LL={np.mean(gsip_ll_values):.6f}, "
        f"LG={np.mean(gsip_lg_values):.6f}, "
        f"H={np.mean(gsip_h_values):.6f}"
    )


# ============================================================
# 9. Window-level Performance Curve
# ============================================================
if len(window_perf_ids) > 0:
    print(
        "\n=== Window-level Performance Diagnostics ==="
    )

    print(
        f"Evaluated windows: "
        f"{len(window_perf_ids)}"
    )

    print(
        "Mean window-level metrics: "
        f"Accuracy={np.mean(window_perf_accuracy) * 100:.2f}%, "
        f"Macro Precision={np.mean(window_perf_precision) * 100:.2f}%, "
        f"Macro Recall={np.mean(window_perf_recall) * 100:.2f}%, "
        f"Macro F1={np.mean(window_perf_f1) * 100:.2f}%"
    )

    fig_window, ax_window = plt.subplots(
        figsize=(8.0, 5.2)
    )

    ax_window.plot(
        window_perf_ids,
        np.asarray(
            window_perf_accuracy
        ) * 100.0,
        label='Accuracy',
        linewidth=1.5,
    )

    ax_window.plot(
        window_perf_ids,
        np.asarray(
            window_perf_precision
        ) * 100.0,
        label='Precision',
        linewidth=1.5,
    )

    ax_window.plot(
        window_perf_ids,
        np.asarray(
            window_perf_recall
        ) * 100.0,
        label='Recall',
        linewidth=1.5,
    )

    ax_window.plot(
        window_perf_ids,
        np.asarray(
            window_perf_f1
        ) * 100.0,
        label='F1-score',
        linewidth=1.5,
    )

    ax_window.set_xlabel(
        'Window Index'
    )
    ax_window.set_ylabel(
        'Performance (%)'
    )
    ax_window.set_title(
        'GSIP Window-level Performance Curve'
    )
    ax_window.set_ylim(
        0.0,
        100.0,
    )
    ax_window.grid(
        True,
        alpha=0.25,
    )
    ax_window.legend(
        loc='best'
    )

    fig_window.tight_layout()
    plt.show()
    plt.close(
        fig_window
    )


# ============================================================
# 10. Final progressive hold-out evaluation
# ============================================================
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

print(
    "\n=== Evaluation on Random Hold-out (30%) ==="
)

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
        "测试集为空或没有测试样本被评估。"
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

print(
    "\nClassification Report:"
)

for lab in target_names:
    m = report[
        lab
    ]

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
    m = report[
        k
    ]

    print(
        f"{k:<28} "
        f"precision: {m['precision'] * 100:.2f}%  "
        f"recall: {m['recall'] * 100:.2f}%  "
        f"f1-score: {m['f1-score'] * 100:.2f}%"
    )


# ============================================================
# 11. Confusion matrix
# ============================================================
try:
    cm = confusion_matrix(
        y_true_test,
        y_pred_test,
        labels=unique_ids,
    )

    fig_cm, ax_cm = plt.subplots(
        figsize=(5.6, 5.0)
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

    ax_cm.set_xlabel(
        'Predicted label'
    )

    ax_cm.set_ylabel(
        'True label'
    )

    fig_cm.tight_layout()
    plt.show()
    plt.close(
        fig_cm
    )

except Exception as e:
    print(
        "绘制混淆矩阵出错：",
        e,
    )


# ============================================================
# 12. ROC
# ============================================================
try:
    if (
        NUM_CLASSES == 2
        and y_prob_test_all.shape[0]
        == len(y_true_test)
        and len(
            np.unique(
                y_true_test
            )
        ) == 2
    ):
        fpr, tpr, _ = roc_curve(
            y_true_test,
            y_prob_test_all[:, 1],
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        fig_roc, ax_roc = plt.subplots(
            figsize=(6.0, 5.0)
        )

        ax_roc.plot(
            fpr,
            tpr,
            label=f"AUC={roc_auc:.4f}",
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            linestyle='--',
        )

        ax_roc.set_xlabel(
            'False Positive Rate'
        )

        ax_roc.set_ylabel(
            'True Positive Rate'
        )

        ax_roc.legend(
            loc='lower right'
        )

        fig_roc.tight_layout()
        plt.show()
        plt.close(
            fig_roc
        )

        print(
            f"ROC-AUC: "
            f"{roc_auc * 100:.2f}%"
        )

except Exception as e:
    print(
        "ROC 计算/绘制出错：",
        e,
    )


# ============================================================
# 13. t-SNE of test hidden representations
# ============================================================
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
        ).all(
            axis=1
        )

        hidden_np = hidden_np[
            finite_mask
        ]

        labels_tsne = labels_tsne[
            finite_mask
        ]

        if len(hidden_np) > 10:
            rng = np.random.default_rng(
                SEED
            )

            selected = []

            for cls_id in np.unique(
                labels_tsne
            ):
                cls_idx = np.where(
                    labels_tsne
                    == cls_id
                )[0]

                if (
                    len(cls_idx)
                    > TSNE_MAX_PER_CLASS
                ):
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

            n_tsne = len(
                x_tsne
            )

            perplexity = min(
                30.0,
                max(
                    5.0,
                    (
                        n_tsne
                        - 1
                    ) / 3.0,
                ),
            )

            perplexity = min(
                perplexity,
                float(
                    n_tsne
                    - 1
                ),
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
                    y_tsne
                    == cls_id
                )

                ax_tsne.scatter(
                    emb_2d[
                        mask,
                        0
                    ],
                    emb_2d[
                        mask,
                        1
                    ],
                    s=12,
                    alpha=0.75,
                    label=id_to_label.get(
                        int(
                            cls_id
                        ),
                        str(
                            cls_id
                        ),
                    ),
                )

            ax_tsne.set_xlabel(
                't-SNE Dim 1'
            )

            ax_tsne.set_ylabel(
                't-SNE Dim 2'
            )

            ax_tsne.set_title(
                'GSIP Test Hidden Embeddings'
            )

            ax_tsne.legend()

            fig_tsne.tight_layout()
            plt.show()
            plt.close(
                fig_tsne
            )

    else:
        print(
            "t-SNE: 测试隐藏向量过少，跳过可视化。"
        )

except Exception as e:
    print(
        "t-SNE 可视化失败：",
        e,
    )
