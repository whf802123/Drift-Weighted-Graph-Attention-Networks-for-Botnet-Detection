


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

MANUAL_MINORITY_LABELS = {
    "C&C",
    "C&C-HeartBeat",
}

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
SOURCE_EPOCHS = 5
ADAPT_EPOCHS = 1

CORR_THRESHOLD = 0.4
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
SEED = 42

HIDDEN_DIM = 32
LR_MODEL = 1e-3
WD_MODEL = 0.0

# GCAL-style information maximization / replay.
ENTROPY_WEIGHT = 1.0
REPLAY_WEIGHT = 0.5
EMA_MOMENTUM = 0.90

# Lightweight variational memory.
SYN_RATIO = 0.05
MEMORY_INTERVAL = 5
MAX_MEMORY_DOMAINS = 3
MEMORY_INNER_STEPS = 1
LR_MEMORY = 1e-3
WD_MEMORY = 0.0

KL_WEIGHT = 1e-4
MMD_WEIGHT = 1.0
GRAD_MATCH_WEIGHT = 1.0

# Synthetic edge sparsification.
SYN_EDGE_THRESHOLD = 0.60
SYN_PROJ_DIM = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1. Dataset preprocessing
# ============================================================

import ipaddress


def ip_to_int(v):
    try:
        s = str(
            v
        ).strip()

        if s in (
            "",
            "-",
            "nan",
        ):
            return np.nan

        return int(
            ipaddress.ip_address(
                s
            )
        )

    except Exception:
        return np.nan


def to_float(v):
    try:
        s = str(
            v
        ).strip()

        if s in (
            "",
            "-",
            "nan",
            "None",
            "NaN",
        ):
            return np.nan

        return float(
            s
        )

    except Exception:
        return np.nan


df = pd.read_csv(
    CSV_PATH
)

if (
    len(
        df.columns
    ) > 0
    and (
        str(
            df.columns[
                0
            ]
        ).lower().startswith(
            "unnamed"
        )
        or df.columns[
            0
        ] == ""
    )
):
    df.drop(
        df.columns[
            0
        ],
        axis=1,
        inplace=True,
    )

if "label" not in df.columns:
    raise KeyError(
        "IoT23 label column 'label' was not found."
    )

df[
    "_stream_order_"
] = np.arange(
    len(
        df
    ),
    dtype=np.int64,
)

df = df[
    ~df[
        "label"
    ].isin(
        RARE_LABELS_TO_DROP
    )
].copy()


def keep_or_sample(g):
    if str(
        g.name
    ) in MANUAL_MINORITY_LABELS:
        return g

    return g.sample(
        frac=SAMPLE_FRAC,
        random_state=SEED,
    )


df = (
    df.groupby(
        "label",
        group_keys=False,
    )
    .apply(
        keep_or_sample
    )
    .sort_values(
        "_stream_order_"
    )
    .reset_index(
        drop=True
    )
)

df.drop(
    columns=[
        "_stream_order_"
    ],
    inplace=True,
)

counts = (
    df[
        "label"
    ]
    .astype(
        str
    )
    .value_counts()
)

too_small = set(
    counts[
        counts < 2
    ].index.tolist()
)

if too_small:
    df = df[
        ~df[
            "label"
        ].isin(
            too_small
        )
    ].reset_index(
        drop=True
    )

ip_cols = [
    c
    for c in [
        "id.orig_h",
        "id.resp_h",
    ]
    if c in df.columns
]

num_cols = [
    c
    for c in [
        "ts",
        "id.orig_p",
        "id.resp_p",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
    ]
    if c in df.columns
]

cat_cols = [
    c
    for c in [
        "proto",
        "service",
        "conn_state",
        "history",
        "local_orig",
        "local_resp",
    ]
    if c in df.columns
]

for c in ip_cols:
    df[
        c
    ] = df[
        c
    ].map(
        ip_to_int
    )

for c in num_cols:
    df[
        c
    ] = df[
        c
    ].map(
        to_float
    )

for c in cat_cols:
    df[
        c
    ] = (
        df[
            c
        ]
        .astype(
            str
        )
        .fillna(
            "-"
        )
        .replace({
            "nan": "-"
        })
    )

label_text = (
    df[
        "label"
    ]
    .astype(
        str
    )
    .values
)

unique_text = pd.unique(
    label_text
)

label_to_id = {
    lab: i
    for i, lab in enumerate(
        unique_text
    )
}

id_to_label = {
    i: lab
    for lab, i in label_to_id.items()
}

labels = np.asarray(
    [
        label_to_id[
            x
        ]
        for x in label_text
    ],
    dtype=np.int64,
)

num_df = (
    df[
        num_cols
        + ip_cols
    ].copy()
    if (
        num_cols
        or ip_cols
    )
    else pd.DataFrame(
        index=df.index
    )
)

cat_df = (
    pd.get_dummies(
        df[
            cat_cols
        ],
        prefix=cat_cols,
    )
    if cat_cols
    else pd.DataFrame(
        index=df.index
    )
)

feat_df = pd.concat(
    [
        num_df,
        cat_df,
    ],
    axis=1,
)

all_idx = np.arange(
    len(
        df
    ),
    dtype=np.int64,
)

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

medians = feat_df.iloc[
    train_idx
].median(
    numeric_only=True
)

feat_df = (
    feat_df
    .fillna(
        medians
    )
    .fillna(
        0.0
    )
)

scaler = StandardScaler()

features = np.empty_like(
    feat_df.values,
    dtype=np.float64,
)

features[
    train_idx
] = scaler.fit_transform(
    feat_df.iloc[
        train_idx
    ].values.astype(
        float
    )
)

features[
    test_idx
] = scaler.transform(
    feat_df.iloc[
        test_idx
    ].values.astype(
        float
    )
)
features[
    val_idx
] = scaler.transform(
    feat_df.iloc[
        val_idx
    ].values.astype(
        float
    )
)

N = len(
    features
)

is_train = np.zeros(
    N,
    dtype=bool,
)

is_train[
    train_idx
] = True

is_test = np.zeros(N, dtype=bool)
is_test[test_idx] = True
is_val = np.zeros(N, dtype=bool)
is_val[val_idx] = True
print(
    f"Data split: train={len(train_idx)} ({TRAIN_RATIO:.0%}), "
    f"validation={len(val_idx)} ({VALIDATION_RATIO:.0%}), "
    f"test={len(test_idx)} ({TEST_RATIO:.0%})"
)

NUM_CLASSES = len(
    np.unique(
        labels
    )
)

if len(
    np.unique(
        labels[
            train_idx
        ]
    )
) != NUM_CLASSES:
    raise RuntimeError(
        "IoT23 training split does not cover all retained classes."
    )

print(
    "IoT23 label mapping:",
    id_to_label,
)

print(
    f"IoT23: total={N}, "
    f"train={len(train_idx)}, "
    f"test={len(test_idx)}, "
    f"classes={NUM_CLASSES}, "
    f"features={features.shape[1]}"
)



# ============================================================
# 2. Pearson graph construction
# ============================================================
def safe_corrcoef(x_np):
    if len(x_np) == 0:
        return np.empty((0, 0), dtype=np.float64)

    if len(x_np) == 1:
        return np.ones((1, 1), dtype=np.float64)

    corr = np.corrcoef(x_np)

    return np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_train_graph(x_np):
    n = len(x_np)

    if n == 0:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=DEVICE,
        )

    corr = safe_corrcoef(x_np)

    adj = (
        np.abs(corr)
        >= CORR_THRESHOLD
    )

    np.fill_diagonal(
        adj,
        False,
    )

    src, dst = np.where(adj)

    # Explicit self-loops.
    self_nodes = np.arange(
        n,
        dtype=np.int64,
    )

    src = np.concatenate([
        src.astype(np.int64),
        self_nodes,
    ])

    dst = np.concatenate([
        dst.astype(np.int64),
        self_nodes,
    ])

    return torch.tensor(
        np.vstack([
            src,
            dst,
        ]),
        dtype=torch.long,
        device=DEVICE,
    )


def build_eval_graph(
    x_train_np,
    x_test_np,
):
    """
    Node order:
      [train context, test nodes]

    Allowed:
      train -> train
      train -> test
      self-loop

    Forbidden:
      test -> train
      test -> test
    """
    n_train = len(x_train_np)
    n_test = len(x_test_np)
    n_total = n_train + n_test

    x_all = np.concatenate(
        [
            x_train_np,
            x_test_np,
        ],
        axis=0,
    )

    corr = safe_corrcoef(
        x_all
    )

    src_parts = []
    dst_parts = []

    if n_train > 0:
        # train -> train
        tt = (
            np.abs(
                corr[
                    :n_train,
                    :n_train,
                ]
            )
            >= CORR_THRESHOLD
        )

        np.fill_diagonal(
            tt,
            False,
        )

        s, d = np.where(
            tt
        )

        if len(s) > 0:
            src_parts.append(
                s.astype(np.int64)
            )
            dst_parts.append(
                d.astype(np.int64)
            )

        # train -> test only
        s, test_col = np.where(
            np.abs(
                corr[
                    :n_train,
                    n_train:
                ]
            )
            >= CORR_THRESHOLD
        )

        if len(s) > 0:
            src_parts.append(
                s.astype(np.int64)
            )

            dst_parts.append(
                (
                    n_train
                    + test_col
                ).astype(np.int64)
            )

    self_nodes = np.arange(
        n_total,
        dtype=np.int64,
    )

    src_parts.append(
        self_nodes
    )

    dst_parts.append(
        self_nodes
    )

    edge_index = torch.tensor(
        np.vstack([
            np.concatenate(
                src_parts
            ),
            np.concatenate(
                dst_parts
            ),
        ]),
        dtype=torch.long,
        device=DEVICE,
    )

    x_tensor = torch.tensor(
        x_all,
        dtype=torch.float32,
        device=DEVICE,
    )

    return (
        x_tensor,
        edge_index,
    )


# ============================================================
# 3. Base GCN
# ============================================================
class GCALBackbone(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_dim,
            hidden_dim,
            add_self_loops=False,
            normalize=True,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
            add_self_loops=False,
            normalize=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        return_embedding=False,
    ):
        h = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        h = F.relu(
            h
        )

        h = self.conv2(
            h,
            edge_index,
            edge_weight=edge_weight,
        )

        h = F.relu(
            h
        )

        logits = self.classifier(
            h
        )

        if return_embedding:
            return (
                logits,
                h,
            )

        return logits


# ============================================================
# 4. GCAL losses / EMA
# ============================================================
def information_max_loss(
    logits,
):
    """
    GCAL-style information maximization.

    Minimize:
      sample entropy
      + ENTROPY_WEIGHT * sum(p_bar * log p_bar)

    The second term is negative marginal entropy and therefore encourages
    diverse predictions across the current graph domain.
    """
    p = torch.softmax(
        logits,
        dim=1,
    )

    log_p = torch.log_softmax(
        logits,
        dim=1,
    )

    sample_entropy = -(
        p
        * log_p
    ).sum(
        dim=1
    ).mean()

    marginal = p.mean(
        dim=0
    ).clamp_min(
        1e-12
    )

    diversity_term = (
        marginal
        * torch.log(
            marginal
        )
    ).sum()

    return (
        sample_entropy
        + ENTROPY_WEIGHT
        * diversity_term
    )


@torch.no_grad()
def ema_update(
    ema_model,
    model,
):
    for p_ema, p in zip(
        ema_model.parameters(),
        model.parameters(),
    ):
        p_ema.data.mul_(
            EMA_MOMENTUM
        ).add_(
            p.data,
            alpha=(
                1.0
                - EMA_MOMENTUM
            ),
        )


def cosine_gradient_distance(
    grad_a,
    grad_b,
):
    a = torch.cat([
        g.reshape(-1)
        for g in grad_a
    ])

    b = torch.cat([
        g.reshape(-1)
        for g in grad_b
    ])

    return (
        1.0
        - F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).mean()
    )


# ============================================================
# 5. Lightweight variational memory generator
# ============================================================
class LiteMemoryGenerator(nn.Module):
    """
    Compact replacement for the full VGAE + PGE generation loop.

    It preserves the GCAL concept:
      real target graph -> variational node distribution -> small memory graph.

    The decoder uses a learnable low-dimensional pairwise similarity instead
    of a deep all-pairs PGE, which is much faster for repeated windows.
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=32,
        proj_dim=SYN_PROJ_DIM,
    ):
        super().__init__()

        self.enc1 = GCNConv(
            in_dim,
            hidden_dim,
            add_self_loops=False,
            normalize=True,
        )

        self.mu_head = nn.Linear(
            hidden_dim,
            in_dim,
        )

        self.logvar_head = nn.Linear(
            hidden_dim,
            in_dim,
        )

        self.score_head = nn.Linear(
            hidden_dim,
            1,
        )

        self.edge_proj = nn.Linear(
            in_dim,
            proj_dim,
            bias=False,
        )

    def encode(
        self,
        x,
        edge_index,
    ):
        h = F.relu(
            self.enc1(
                x,
                edge_index,
            )
        )

        mu = self.mu_head(
            h
        )

        logvar = self.logvar_head(
            h
        ).clamp(
            -6.0,
            6.0,
        )

        score = self.score_head(
            h
        ).squeeze(-1)

        return (
            mu,
            logvar,
            score,
        )

    def choose_nodes(
        self,
        score,
    ):
        n = score.numel()

        k = max(
            2,
            int(
                round(
                    SYN_RATIO
                    * n
                )
            ),
        )

        k = min(
            k,
            n,
        )

        return torch.topk(
            score,
            k=k,
            largest=True,
        ).indices

    def reparameterize(
        self,
        mu,
        logvar,
    ):
        std = torch.exp(
            0.5
            * logvar
        )

        eps = torch.randn_like(
            std
        )

        return (
            mu
            + eps
            * std
        )

    def decode_graph(
        self,
        z,
        hard=False,
    ):
        q = F.normalize(
            self.edge_proj(
                z
            ),
            dim=1,
            eps=1e-8,
        )

        prob = torch.sigmoid(
            q
            @ q.T
        )

        n = len(z)

        if hard:
            mask = (
                prob
                >= SYN_EDGE_THRESHOLD
            )
        else:
            # Straight-through style hard sparsification with differentiable
            # retained weights.
            mask = (
                prob.detach()
                >= SYN_EDGE_THRESHOLD
            )

        # Always keep synthetic self-loops.
        diag = torch.arange(
            n,
            device=z.device,
        )

        mask[
            diag,
            diag
        ] = True

        dst, src = torch.where(
            mask
        )

        edge_index = torch.stack(
            [
                src,
                dst,
            ],
            dim=0,
        )

        edge_weight = prob[
            dst,
            src
        ]

        return (
            edge_index,
            edge_weight,
            prob,
        )

    def forward(
        self,
        x,
        edge_index,
    ):
        mu_all, logvar_all, score = self.encode(
            x,
            edge_index,
        )

        selected = self.choose_nodes(
            score
        )

        mu = mu_all[
            selected
        ]

        logvar = logvar_all[
            selected
        ]

        z = self.reparameterize(
            mu,
            logvar,
        )

        syn_edge_index, syn_edge_weight, prob = self.decode_graph(
            z,
            hard=False,
        )

        return (
            z,
            syn_edge_index,
            syn_edge_weight,
            prob,
            mu,
            logvar,
            selected,
        )

    @torch.no_grad()
    def materialize(
        self,
        x,
        edge_index,
    ):
        mu_all, logvar_all, score = self.encode(
            x,
            edge_index,
        )

        selected = self.choose_nodes(
            score
        )

        # Deterministic memory uses mu.
        z = mu_all[
            selected
        ]

        syn_edge_index, syn_edge_weight, _ = self.decode_graph(
            z,
            hard=True,
        )

        return (
            z.detach().cpu(),
            syn_edge_index.detach().cpu(),
            syn_edge_weight.detach().cpu(),
        )


class MemoryGraph:
    def __init__(
        self,
        x,
        edge_index,
        edge_weight,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def to_device(self):
        return (
            self.x.to(
                DEVICE
            ),
            self.edge_index.to(
                DEVICE
            ),
            self.edge_weight.to(
                DEVICE
            ),
        )


def fit_and_store_memory(
    generator,
    generator_optimizer,
    model,
    x,
    edge_index,
):
    """
    One lightweight inner-loop update:
      classifier-gradient matching
      + embedding mean alignment (MMD)
      + variational KL.

    Then materialize a compact synthetic graph.
    """
    for _ in range(
        MEMORY_INNER_STEPS
    ):
        model.eval()

        real_logits, real_h = model(
            x,
            edge_index,
            return_embedding=True,
        )

        real_loss = information_max_loss(
            real_logits
        )

        classifier_params = [
            model.classifier.weight,
            model.classifier.bias,
        ]

        grad_real = torch.autograd.grad(
            real_loss,
            classifier_params,
            retain_graph=False,
            create_graph=False,
        )

        grad_real = [
            g.detach()
            for g in grad_real
        ]

        (
            syn_x,
            syn_edge_index,
            syn_edge_weight,
            edge_prob,
            mu,
            logvar,
            _selected,
        ) = generator(
            x,
            edge_index,
        )

        syn_logits, syn_h = model(
            syn_x,
            syn_edge_index,
            edge_weight=syn_edge_weight,
            return_embedding=True,
        )

        syn_loss = information_max_loss(
            syn_logits
        )

        grad_syn = torch.autograd.grad(
            syn_loss,
            classifier_params,
            create_graph=True,
            retain_graph=True,
        )

        loss_grad = cosine_gradient_distance(
            grad_syn,
            grad_real,
        )

        # Mean-embedding MMD.
        loss_mmd = F.mse_loss(
            syn_h.mean(
                dim=0
            ),
            real_h.detach().mean(
                dim=0
            ),
        )

        # Variational information-bottleneck term.
        loss_kl = -0.5 * torch.mean(
            1.0
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

        gen_loss = (
            GRAD_MATCH_WEIGHT
            * loss_grad
            + MMD_WEIGHT
            * loss_mmd
            + KL_WEIGHT
            * loss_kl
        )

        generator_optimizer.zero_grad(
            set_to_none=True
        )

        gen_loss.backward()
        generator_optimizer.step()

    (
        mem_x,
        mem_edge,
        mem_weight,
    ) = generator.materialize(
        x,
        edge_index,
    )

    return MemoryGraph(
        mem_x,
        mem_edge,
        mem_weight,
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
ema_model = None
optimizer = None
memory_generator = None
memory_optimizer = None

memory_bank = []

global_idx = 0
window_id = 0

y_true_test = []
y_pred_test = []
y_prob_test = []

window_metrics = []
memory_sizes = []


# ============================================================
# 7. Streaming training / GCAL adaptation
# ============================================================
print(
    "\n=== Streaming GCAL-Lite ==="
)

for start in tqdm(
    range(
        0,
        N,
        BATCH_SIZE,
    ),
    desc="GCAL-Lite",
):
    batch_x = features[
        start:
        start + BATCH_SIZE
    ]

    batch_y = labels[
        start:
        start + BATCH_SIZE
    ]

    bsz = len(
        batch_x
    )

    for i in range(
        bsz
    ):
        features_window.append(
            batch_x[
                i
            ]
        )

        labels_window.append(
            int(
                batch_y[
                    i
                ]
            )
        )

        index_window.append(
            global_idx
        )

        global_idx += 1

    if len(
        features_window
    ) < WINDOW_SIZE:
        continue

    window_id += 1

    first_window = (
        model is None
    )

    x_win = np.asarray(
        features_window,
        dtype=np.float64,
    )

    y_win = np.asarray(
        labels_window,
        dtype=np.int64,
    )

    idx_win = np.asarray(
        index_window,
        dtype=np.int64,
    )

    new_mask_full = np.zeros(
        len(
            x_win
        ),
        dtype=bool,
    )

    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[
            -bsz:
        ] = True

    train_mask_full = is_train[
        idx_win
    ]

    test_mask_full = is_test[
        idx_win
    ]

    x_train_np = x_win[
        train_mask_full
    ]

    y_train_np = y_win[
        train_mask_full
    ]

    if len(
        x_train_np
    ) == 0:
        continue

    x_train = torch.tensor(
        x_train_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_train = torch.tensor(
        y_train_np,
        dtype=torch.long,
        device=DEVICE,
    )

    edge_train = build_train_graph(
        x_train_np
    )

    if model is None:
        model = GCALBackbone(
            in_dim=x_train.size(1),
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
        ).to(
            DEVICE
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR_MODEL,
            weight_decay=WD_MODEL,
        )

        memory_generator = LiteMemoryGenerator(
            in_dim=x_train.size(1),
            hidden_dim=HIDDEN_DIM,
        ).to(
            DEVICE
        )

        memory_optimizer = torch.optim.Adam(
            memory_generator.parameters(),
            lr=LR_MEMORY,
            weight_decay=WD_MEMORY,
        )

        # ----------------------------------------------------
        # Source initialization: labels are used only here.
        # ----------------------------------------------------
        for _ in range(
            SOURCE_EPOCHS
        ):
            model.train()

            logits = model(
                x_train,
                edge_train,
            )

            source_loss = F.cross_entropy(
                logits,
                y_train,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            source_loss.backward()
            optimizer.step()

        ema_model = copy.deepcopy(
            model
        ).to(
            DEVICE
        )

        ema_model.eval()

        for p in ema_model.parameters():
            p.requires_grad_(
                False
            )

    else:
        # ----------------------------------------------------
        # Unlabeled target adaptation.
        # ----------------------------------------------------
        for _ in range(
            ADAPT_EPOCHS
        ):
            model.train()

            logits = model(
                x_train,
                edge_train,
            )

            adapt_loss = information_max_loss(
                logits
            )

            # Re-adapt historical synthetic memories.
            for memory in memory_bank:
                (
                    mx,
                    me,
                    mw,
                ) = memory.to_device()

                mem_logits = model(
                    mx,
                    me,
                    edge_weight=mw,
                )

                adapt_loss = (
                    adapt_loss
                    + REPLAY_WEIGHT
                    * information_max_loss(
                        mem_logits
                    )
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            adapt_loss.backward()
            optimizer.step()

            ema_update(
                ema_model,
                model,
            )

        # GCAL loads its EMA state back after target adaptation.
        model.load_state_dict(
            copy.deepcopy(
                ema_model.state_dict()
            )
        )

    # --------------------------------------------------------
    # Generate compact memory sparsely, not every overlapping window.
    # --------------------------------------------------------
    should_generate_memory = (
        first_window
        or window_id % MEMORY_INTERVAL == 0
    )

    if should_generate_memory:
        memory = fit_and_store_memory(
            generator=memory_generator,
            generator_optimizer=memory_optimizer,
            model=model,
            x=x_train,
            edge_index=edge_train,
        )

        memory_bank.append(
            memory
        )

        if len(
            memory_bank
        ) > MAX_MEMORY_DOMAINS:
            memory_bank.pop(
                0
            )

        memory_sizes.append(
            int(
                memory.x.shape[0]
            )
        )

    # --------------------------------------------------------
    # Progressive hold-out evaluation: each test node once.
    # --------------------------------------------------------
    eval_test_mask = (
        test_mask_full
        if first_window
        else (
            new_mask_full
            & test_mask_full
        )
    )

    if eval_test_mask.any():
        x_test_np = x_win[
            eval_test_mask
        ]

        y_test_np = y_win[
            eval_test_mask
        ]

        (
            x_eval,
            edge_eval,
        ) = build_eval_graph(
            x_train_np,
            x_test_np,
        )

        model.eval()

        with torch.no_grad():
            logits_eval = model(
                x_eval,
                edge_eval,
            )

            logits_test = logits_eval[
                len(
                    x_train_np
                ):
            ]

            probs_test = torch.softmax(
                logits_test,
                dim=1,
            )

            pred_test = torch.argmax(
                logits_test,
                dim=1,
            )

        y_true_test.extend(
            y_test_np.tolist()
        )

        y_pred_test.extend(
            pred_test.cpu()
            .numpy()
            .tolist()
        )

        y_prob_test.append(
            probs_test.cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Whole-current-window diagnostics.
    # --------------------------------------------------------
    if test_mask_full.any():
        x_window_test_np = x_win[
            test_mask_full
        ]

        y_window_test_np = y_win[
            test_mask_full
        ]

        (
            x_eval,
            edge_eval,
        ) = build_eval_graph(
            x_train_np,
            x_window_test_np,
        )

        model.eval()

        with torch.no_grad():
            logits_eval = model(
                x_eval,
                edge_eval,
            )

            pred = torch.argmax(
                logits_eval[
                    len(
                        x_train_np
                    ):
                ],
                dim=1,
            ).cpu().numpy()

        window_metrics.append({
            "window": window_id,
            "accuracy": accuracy_score(
                y_window_test_np,
                pred,
            ),
            "precision": precision_score(
                y_window_test_np,
                pred,
                average="macro",
                labels=np.arange(
                    NUM_CLASSES
                ),
                zero_division=0,
            ),
            "recall": recall_score(
                y_window_test_np,
                pred,
                average="macro",
                labels=np.arange(
                    NUM_CLASSES
                ),
                zero_division=0,
            ),
            "f1": f1_score(
                y_window_test_np,
                pred,
                average="macro",
                labels=np.arange(
                    NUM_CLASSES
                ),
                zero_division=0,
            ),
        })


if model is None:
    raise RuntimeError(
        "Model was never initialized. "
        "Check WINDOW_SIZE and dataset size."
    )


# ============================================================
# 8. Final evaluation
# ============================================================
y_true_test = np.asarray(
    y_true_test,
    dtype=np.int64,
)

y_pred_test = np.asarray(
    y_pred_test,
    dtype=np.int64,
)

y_prob_test = (
    np.vstack(
        y_prob_test
    )
    if len(
        y_prob_test
    ) > 0
    else np.empty(
        (
            0,
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )
)

print(
    "\n=== GCAL-Lite Final Hold-out Evaluation ==="
)

print(
    "Expected test samples:",
    len(
        test_idx
    ),
)

print(
    "Actually evaluated:",
    len(
        y_true_test
    ),
)

print(
    "Stored memory domains:",
    len(
        memory_bank
    ),
)

if memory_sizes:
    print(
        "Mean generated memory nodes:",
        f"{np.mean(memory_sizes):.2f}",
    )

if len(
    y_true_test
) == 0:
    raise RuntimeError(
        "No test samples were evaluated."
    )

ids = np.arange(
    NUM_CLASSES
)

names = [
    id_to_label[
        int(i)
    ]
    for i in ids
]

report = classification_report(
    y_true_test,
    y_pred_test,
    labels=ids,
    target_names=names,
    output_dict=True,
    zero_division=0,
)

for name in names:
    m = report[
        name
    ]

    print(
        f"{name:<28} "
        f"P={m['precision']*100:.2f}%  "
        f"R={m['recall']*100:.2f}%  "
        f"F1={m['f1-score']*100:.2f}%"
    )

print(
    f"{'accuracy':<28} "
    f"{report['accuracy']*100:.2f}%"
)

for k in [
    "macro avg",
    "weighted avg",
]:
    m = report[
        k
    ]

    print(
        f"{k:<28} "
        f"P={m['precision']*100:.2f}%  "
        f"R={m['recall']*100:.2f}%  "
        f"F1={m['f1-score']*100:.2f}%"
    )


# ============================================================
# 9. Confusion matrix
# ============================================================
try:
    cm = confusion_matrix(
        y_true_test,
        y_pred_test,
        labels=ids,
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 5.2)
    )

    ConfusionMatrixDisplay(
        cm,
        display_labels=names,
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
    plt.close(
        fig
    )

except Exception as e:
    print(
        "Confusion matrix failed:",
        e,
    )


# ============================================================
# 10. ROC
# ============================================================
try:
    if (
        y_prob_test.shape[0]
        == len(
            y_true_test
        )
    ):
        if NUM_CLASSES == 2:
            if len(
                np.unique(
                    y_true_test
                )
            ) == 2:
                fpr, tpr, _ = roc_curve(
                    y_true_test,
                    y_prob_test[
                        :,
                        1
                    ],
                )

                print(
                    "ROC-AUC: "
                    f"{auc(fpr, tpr)*100:.2f}%"
                )

        else:
            y_bin = label_binarize(
                y_true_test,
                classes=ids,
            )

            aucs = []

            for c in ids:
                yc = y_bin[
                    :,
                    int(c)
                ]

                if (
                    yc.sum() == 0
                    or yc.sum()
                    == len(
                        yc
                    )
                ):
                    continue

                fpr, tpr, _ = roc_curve(
                    yc,
                    y_prob_test[
                        :,
                        int(c)
                    ],
                )

                aucs.append(
                    auc(
                        fpr,
                        tpr,
                    )
                )

            if aucs:
                print(
                    "Macro OvR ROC-AUC: "
                    f"{np.mean(aucs)*100:.2f}%"
                )

except Exception as e:
    print(
        "ROC failed:",
        e,
    )


# ============================================================
# 11. Window-level curve
# ============================================================
if window_metrics:
    window_df = pd.DataFrame(
        window_metrics
    )

    print(
        "\nMean window metrics: "
        f"Acc={window_df['accuracy'].mean()*100:.2f}%, "
        f"MacroP={window_df['precision'].mean()*100:.2f}%, "
        f"MacroR={window_df['recall'].mean()*100:.2f}%, "
        f"MacroF1={window_df['f1'].mean()*100:.2f}%"
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    ax.plot(
        window_df[
            "window"
        ],
        window_df[
            "accuracy"
        ] * 100,
        label="Accuracy",
    )

    ax.plot(
        window_df[
            "window"
        ],
        window_df[
            "f1"
        ] * 100,
        label="Macro F1",
    )

    ax.set_xlabel(
        "Window Index"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        "GCAL-Lite Window-level Performance"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()
    plt.show()
    plt.close(
        fig
    )

