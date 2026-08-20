
import warnings
warnings.filterwarnings("ignore")

from collections import deque
import random
import ipaddress

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


HIDDEN_DIM = 48
LR = 1e-2
WEIGHT_DECAY = 0.0
REDUCTION_RATE = 0.50
BUFFER_SIZE = 200

EPS = 1e-12
TSNE_MAX_PER_CLASS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
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
        "未找到列名 'label'。请确认 IoT23 CSV 中标签列名为小写 label。"
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
    label_counts_sampled[
        label_counts_sampled < 2
    ].index.tolist()
)
if too_small:
    print(
        "[NOTICE] Classes with <2 samples after sampling are dropped: "
        f"{sorted(list(too_small))}"
    )
    df = df[
        ~df['label'].isin(too_small)
    ].reset_index(drop=True)


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
    lab: i
    for i, lab in enumerate(unique_labels_text)
}
id_to_label = {
    v: k
    for k, v in label_to_id.items()
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
        "未能构建任何特征列，请检查输入 CSV。"
    )

N_total = len(df)
all_idx = np.arange(
    N_total,
    dtype=np.int64,
)



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
].median(numeric_only=True)

feat_df = (
    feat_df
    .fillna(medians)
    .fillna(0.0)
)


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
features[val_idx] = scaler.transform(
    feat_df.iloc[
        val_idx
    ].values.astype(float)
)

N = len(features)

is_train = np.zeros(
    N,
    dtype=bool,
)
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

train_unique_ids = np.unique(
    labels[train_idx]
)
NUM_CLASSES = len(train_unique_ids)

if NUM_CLASSES != len(np.unique(labels)):
    raise RuntimeError(
        "训练集未包含全部类别，无法进行当前多分类设置。"
    )

print("\nDevice:", DEVICE)
print(
    f"Samples: total={N_total}, "
    f"train={len(train_idx)}, "
    f"test={len(test_idx)}"
)
print(f"Features: {feat_df.shape[1]}")
print(
    f"TACO: classes={NUM_CLASSES}, "
    f"hidden={HIDDEN_DIM}, "
    f"reduction={REDUCTION_RATE}, "
    f"reservoir={BUFFER_SIZE}"
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


def raw_pearson_edges(x_np: np.ndarray):
    n = x_np.shape[0]

    if n <= 1:
        return np.empty((2, 0), dtype=np.int64)

    corr = safe_row_corrcoef(x_np)
    adj = np.abs(corr) >= CORR_THRESHOLD
    np.fill_diagonal(adj, False)

    src, dst = np.where(adj)

    return np.vstack([
        src.astype(np.int64),
        dst.astype(np.int64),
    ])


def edge_dict_to_tensors(edge_dict):
    if len(edge_dict) == 0:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=DEVICE,
        )
        edge_weight = torch.empty(
            (0,),
            dtype=torch.float32,
            device=DEVICE,
        )
        return edge_index, edge_weight

    keys = list(edge_dict.keys())
    vals = [edge_dict[k] for k in keys]

    edge_index = torch.tensor(
        np.asarray(keys, dtype=np.int64).T,
        dtype=torch.long,
        device=DEVICE,
    )

    edge_weight = torch.tensor(
        np.asarray(vals, dtype=np.float32),
        dtype=torch.float32,
        device=DEVICE,
    )

    return edge_index, edge_weight


def tensors_to_edge_dict(edge_index, edge_weight):
    out = {}

    if edge_index.numel() == 0:
        return out

    ei = edge_index.detach().cpu().numpy()
    ew = edge_weight.detach().cpu().numpy()

    for k in range(ei.shape[1]):
        u = int(ei[0, k])
        v = int(ei[1, k])
        w = float(ew[k])

        if u == v:
            continue




        old = out.get((u, v))
        if old is None or w > old:
            out[(u, v)] = w

    return out





class TACO_GCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
            add_self_loops=True,
            normalize=True,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            output_dim,
            add_self_loops=True,
            normalize=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        return_hidden=False,
    ):
        h = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        h = F.relu(h)

        if return_hidden:
            return h

        logits = self.conv2(
            h,
            edge_index,
            edge_weight=edge_weight,
        )

        return logits





class GraphState:
    def __init__(
        self,
        x,
        y_soft,
        edge_index,
        edge_weight,
        clusters,
    ):
        self.x = x
        self.y_soft = y_soft
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.clusters = clusters
        self.rebuild_n2c()

    def rebuild_n2c(self):
        self.n2c = {}

        for coarse_id, members in enumerate(self.clusters):
            for gid in members:
                self.n2c[int(gid)] = int(coarse_id)

    @property
    def y(self):
        return torch.argmax(
            self.y_soft,
            dim=1,
        )

    @property
    def num_nodes(self):
        return int(self.x.size(0))


def singleton_state(
    raw_global_ids,
    raw_x_np,
    raw_y_np,
    raw_edge_local,
):
    n = len(raw_global_ids)

    x = torch.tensor(
        raw_x_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_t = torch.tensor(
        raw_y_np,
        dtype=torch.long,
        device=DEVICE,
    )

    y_soft = F.one_hot(
        y_t,
        num_classes=NUM_CLASSES,
    ).float()

    clusters = [
        {int(gid)}
        for gid in raw_global_ids
    ]

    edge_dict = {}

    for k in range(raw_edge_local.shape[1]):
        u = int(raw_edge_local[0, k])
        v = int(raw_edge_local[1, k])

        if u != v:
            edge_dict[(u, v)] = 1.0

    edge_index, edge_weight = edge_dict_to_tensors(
        edge_dict
    )

    return GraphState(
        x=x,
        y_soft=y_soft,
        edge_index=edge_index,
        edge_weight=edge_weight,
        clusters=clusters,
    )


def expand_and_align_state(
    prev_state: GraphState,
    current_train_global_ids,
    current_train_x_np,
    current_train_y_np,
    current_raw_edges_local,
):
    current_train_global_ids = np.asarray(
        current_train_global_ids,
        dtype=np.int64,
    )


    x_parts = [prev_state.x]
    y_parts = [prev_state.y_soft]
    clusters = [
        set(members)
        for members in prev_state.clusters
    ]

    n2c = dict(prev_state.n2c)


    new_local_positions = []

    for local_pos, gid in enumerate(current_train_global_ids):
        gid = int(gid)

        if gid not in n2c:
            new_local_positions.append(local_pos)

            new_idx = len(clusters)
            n2c[gid] = new_idx
            clusters.append({gid})

    if len(new_local_positions) > 0:
        new_x_np = current_train_x_np[
            new_local_positions
        ]

        new_y_np = current_train_y_np[
            new_local_positions
        ]

        new_x = torch.tensor(
            new_x_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        new_y = torch.tensor(
            new_y_np,
            dtype=torch.long,
            device=DEVICE,
        )

        new_y_soft = F.one_hot(
            new_y,
            num_classes=NUM_CLASSES,
        ).float()

        x_parts.append(new_x)
        y_parts.append(new_y_soft)

    x = torch.cat(
        x_parts,
        dim=0,
    )

    y_soft = torch.cat(
        y_parts,
        dim=0,
    )


    edge_dict = tensors_to_edge_dict(
        prev_state.edge_index,
        prev_state.edge_weight,
    )


    for k in range(current_raw_edges_local.shape[1]):
        lu = int(current_raw_edges_local[0, k])
        lv = int(current_raw_edges_local[1, k])

        gid_u = int(current_train_global_ids[lu])
        gid_v = int(current_train_global_ids[lv])

        cu = int(n2c[gid_u])
        cv = int(n2c[gid_v])

        if cu == cv:
            continue


        old = edge_dict.get((cu, cv))
        if old is None or 1.0 > old:
            edge_dict[(cu, cv)] = 1.0

    edge_index, edge_weight = edge_dict_to_tensors(
        edge_dict
    )

    state = GraphState(
        x=x,
        y_soft=y_soft,
        edge_index=edge_index,
        edge_weight=edge_weight,
        clusters=clusters,
    )

    return state





class FidelityReservoir:
    def __init__(
        self,
        capacity,
        seed,
    ):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.buffer = []
        self.num_seen = 0

    def update(self, global_ids):
        for gid in global_ids:
            gid = int(gid)

            self.num_seen += 1

            if len(self.buffer) < self.capacity:
                self.buffer.append(gid)
            else:
                j = self.rng.randint(
                    0,
                    self.num_seen - 1,
                )

                if j < self.capacity:
                    self.buffer[j] = gid

    def protected_coarse_nodes(
        self,
        n2c,
    ):
        protected = set()

        for gid in self.buffer:
            if gid in n2c:
                protected.add(
                    int(n2c[gid])
                )

        return protected





def cosine_similarity_np(a, b):
    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if abs(denom) < 1e-8:
        denom = 1e-8

    return float(
        np.dot(a, b) / denom
    )


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.count = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[
                self.parent[x]
            ]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        self.parent[ra] = rb
        self.count -= 1
        return True


def unique_undirected_edges(
    edge_index,
    edge_weight,
):
    edge_map = {}

    if edge_index.numel() == 0:
        return []

    ei = edge_index.detach().cpu().numpy()
    ew = edge_weight.detach().cpu().numpy()

    for k in range(ei.shape[1]):
        u = int(ei[0, k])
        v = int(ei[1, k])

        if u == v:
            continue

        a, b = (
            (u, v)
            if u < v
            else (v, u)
        )

        w = float(ew[k])

        old = edge_map.get((a, b))
        if old is None or w > old:
            edge_map[(a, b)] = w

    return [
        (u, v, w)
        for (u, v), w in edge_map.items()
    ]


@torch.no_grad()
def taco_coarsen(
    state: GraphState,
    hidden,
    protected_nodes,
    reduction_rate,
):
    n = state.num_nodes

    if n <= 2:
        return state

    target_n = int(
        np.ceil(
            (1.0 - reduction_rate)
            * n
        )
    )
    target_n = max(
        1,
        min(target_n, n),
    )

    if target_n >= n:
        return state

    hidden_np = (
        hidden
        .detach()
        .cpu()
        .numpy()
    )

    candidate_edges = unique_undirected_edges(
        state.edge_index,
        state.edge_weight,
    )

    scored_edges = []

    for u, v, _edge_w in candidate_edges:
        sim = cosine_similarity_np(
            hidden_np[u],
            hidden_np[v],
        )

        if (
            u in protected_nodes
            or v in protected_nodes
        ):
            priority = sim - 100.0
        else:
            priority = sim

        scored_edges.append(
            (
                priority,
                u,
                v,
            )
        )

    scored_edges.sort(
        key=lambda z: z[0],
        reverse=True,
    )

    uf = UnionFind(n)

    for _priority, u, v in scored_edges:
        uf.union(u, v)

        if uf.count <= target_n:
            break



    groups_dict = {}

    for i in range(n):
        root = uf.find(i)

        groups_dict.setdefault(
            root,
            [],
        ).append(i)

    groups = list(
        groups_dict.values()
    )

    old_to_new = {}

    for new_id, members in enumerate(groups):
        for old_id in members:
            old_to_new[old_id] = new_id


    degrees = np.zeros(
        n,
        dtype=np.float64,
    )

    if state.edge_index.numel() > 0:
        ei_np = (
            state.edge_index
            .detach()
            .cpu()
            .numpy()
        )

        ew_np = (
            state.edge_weight
            .detach()
            .cpu()
            .numpy()
        )

        for k in range(ei_np.shape[1]):
            u = int(ei_np[0, k])
            w = float(ew_np[k])
            degrees[u] += max(w, 0.0)

    degrees = np.maximum(
        degrees,
        1.0,
    )

    new_x = []
    new_y_soft = []
    new_clusters = []


    coeff_by_old = np.zeros(
        n,
        dtype=np.float64,
    )

    for members in groups:
        member_arr = np.asarray(
            members,
            dtype=np.int64,
        )

        d = degrees[
            member_arr
        ]

        total_d = float(
            d.sum()
        )

        if total_d <= EPS:
            coeff = np.full(
                len(members),
                1.0 / np.sqrt(
                    max(1, len(members))
                ),
                dtype=np.float64,
            )
        else:
            coeff = np.sqrt(
                d / total_d
            )

        coeff_t = torch.tensor(
            coeff,
            dtype=torch.float32,
            device=DEVICE,
        )

        member_t = torch.tensor(
            member_arr,
            dtype=torch.long,
            device=DEVICE,
        )

        x_c = (
            state.x[member_t]
            * coeff_t.unsqueeze(1)
        ).sum(dim=0)

        y_c = (
            state.y_soft[member_t]
            * coeff_t.unsqueeze(1)
        ).sum(dim=0)

        new_x.append(x_c)
        new_y_soft.append(y_c)

        merged_members = set()

        for old_id, c in zip(
            members,
            coeff.tolist(),
        ):
            coeff_by_old[
                old_id
            ] = c

            merged_members.update(
                state.clusters[old_id]
            )

        new_clusters.append(
            merged_members
        )

    new_x = torch.stack(
        new_x,
        dim=0,
    )

    new_y_soft = torch.stack(
        new_y_soft,
        dim=0,
    )



    new_edge_dict = {}

    if state.edge_index.numel() > 0:
        ei_np = (
            state.edge_index
            .detach()
            .cpu()
            .numpy()
        )

        ew_np = (
            state.edge_weight
            .detach()
            .cpu()
            .numpy()
        )

        for k in range(ei_np.shape[1]):
            u = int(ei_np[0, k])
            v = int(ei_np[1, k])

            cu = int(
                old_to_new[u]
            )
            cv = int(
                old_to_new[v]
            )

            if cu == cv:
                continue

            w = (
                float(ew_np[k])
                * float(coeff_by_old[u])
                * float(coeff_by_old[v])
            )

            new_edge_dict[
                (cu, cv)
            ] = (
                new_edge_dict.get(
                    (cu, cv),
                    0.0,
                )
                + w
            )

    new_edge_index, new_edge_weight = edge_dict_to_tensors(
        new_edge_dict
    )

    return GraphState(
        x=new_x,
        y_soft=new_y_soft,
        edge_index=new_edge_index,
        edge_weight=new_edge_weight,
        clusters=new_clusters,
    )





def build_eval_graph(
    state: GraphState,
    current_train_global_ids,
    current_train_x_np,
    test_x_np,
):
    n_train_mem = state.num_nodes
    n_test = test_x_np.shape[0]

    if n_test == 0:
        return None

    x_test = torch.tensor(
        test_x_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    x_eval = torch.cat(
        [
            state.x,
            x_test,
        ],
        dim=0,
    )

    edge_dict = tensors_to_edge_dict(
        state.edge_index,
        state.edge_weight,
    )

    current_train_global_ids = np.asarray(
        current_train_global_ids,
        dtype=np.int64,
    )



    raw_eval_np = np.concatenate(
        [
            current_train_x_np,
            test_x_np,
        ],
        axis=0,
    )

    corr = safe_row_corrcoef(
        raw_eval_np
    )

    n_raw_train = current_train_x_np.shape[0]

    corr_test_train = corr[
        n_raw_train:,
        :n_raw_train,
    ]

    test_row, train_col = np.where(
        np.abs(
            corr_test_train
        ) >= CORR_THRESHOLD
    )

    for t_local, tr_local in zip(
        test_row.tolist(),
        train_col.tolist(),
    ):
        gid_train = int(
            current_train_global_ids[
                tr_local
            ]
        )

        if gid_train not in state.n2c:
            continue

        coarse_src = int(
            state.n2c[
                gid_train
            ]
        )

        test_dst = (
            n_train_mem
            + int(t_local)
        )


        edge_dict[
            (
                coarse_src,
                test_dst,
            )
        ] = 1.0

    edge_index, edge_weight = edge_dict_to_tensors(
        edge_dict
    )

    return (
        x_eval,
        edge_index,
        edge_weight,
        n_train_mem,
    )





def train_one_period(
    model,
    optimizer,
    state,
    epochs,
):
    model.train()

    y_train = state.y

    for _ in range(epochs):
        logits = model(
            state.x,
            state.edge_index,
            edge_weight=state.edge_weight,
        )

        loss = F.cross_entropy(
            logits,
            y_train,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()


@torch.no_grad()
def hidden_for_coarsening(
    model,
    state,
):
    model.eval()

    return model(
        state.x,
        state.edge_index,
        edge_weight=state.edge_weight,
        return_hidden=True,
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
state = None

reservoir = FidelityReservoir(
    capacity=BUFFER_SIZE,
    seed=SEED + 100,
)

global_idx = 0
period_idx = 0

y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []

print("\n=== Streaming TACO Training ===")

for start in tqdm(
    range(
        0,
        N,
        BATCH_SIZE,
    ),
    desc="Processing batches",
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
        state is None
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
        new_mask_full[
            -bsz:
        ] = True

    train_mask_full = is_train[
        idx_win_np
    ]

    test_mask_full = is_test[
        idx_win_np
    ]


    current_train_global_ids = idx_win_np[
        train_mask_full
    ]

    current_train_x_np = x_win_np[
        train_mask_full
    ]

    current_train_y_np = y_win_np[
        train_mask_full
    ]

    current_new_train_mask = new_mask_full[
        train_mask_full
    ]

    if current_train_x_np.shape[0] == 0:
        continue

    current_raw_edges = raw_pearson_edges(
        current_train_x_np
    )




    if first_window:
        state = singleton_state(
            raw_global_ids=current_train_global_ids,
            raw_x_np=current_train_x_np,
            raw_y_np=current_train_y_np,
            raw_edge_local=current_raw_edges,
        )

        model = TACO_GCN(
            input_dim=state.x.size(1),
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        epochs_now = EPOCHS_FIRST

    else:
        state = expand_and_align_state(
            prev_state=state,
            current_train_global_ids=current_train_global_ids,
            current_train_x_np=current_train_x_np,
            current_train_y_np=current_train_y_np,
            current_raw_edges_local=current_raw_edges,
        )

        epochs_now = EPOCHS_INC

    expanded_nodes = state.num_nodes




    train_one_period(
        model=model,
        optimizer=optimizer,
        state=state,
        epochs=epochs_now,
    )




    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = (
            new_mask_full
            & test_mask_full
        )

    if eval_test_mask_full.any():
        test_x_np = x_win_np[
            eval_test_mask_full
        ]

        test_y_np = y_win_np[
            eval_test_mask_full
        ]

        eval_graph = build_eval_graph(
            state=state,
            current_train_global_ids=current_train_global_ids,
            current_train_x_np=current_train_x_np,
            test_x_np=test_x_np,
        )

        (
            x_eval,
            eval_edge_index,
            eval_edge_weight,
            n_train_mem,
        ) = eval_graph

        model.eval()

        with torch.no_grad():
            logits_eval = model(
                x_eval,
                eval_edge_index,
                edge_weight=eval_edge_weight,
            )

            hidden_eval = model(
                x_eval,
                eval_edge_index,
                edge_weight=eval_edge_weight,
                return_hidden=True,
            )

            logits_test = logits_eval[
                n_train_mem:
            ]

            hidden_test_batch = hidden_eval[
                n_train_mem:
            ]

            probs_test = torch.softmax(
                logits_test,
                dim=1,
            )

        y_true_test.extend(
            test_y_np.tolist()
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




    if first_window:
        new_train_global_ids = (
            current_train_global_ids
        )
    else:
        new_train_global_ids = (
            current_train_global_ids[
                current_new_train_mask
            ]
        )

    reservoir.update(
        new_train_global_ids
    )

    replay_nodes = reservoir.protected_coarse_nodes(
        state.n2c
    )




    hidden = hidden_for_coarsening(
        model,
        state,
    )

    state = taco_coarsen(
        state=state,
        hidden=hidden,
        protected_nodes=replay_nodes,
        reduction_rate=REDUCTION_RATE,
    )

    period_idx += 1

    if (
        period_idx == 1
        or period_idx % 20 == 0
    ):
        print(
            f"\n[TACO] period={period_idx}, "
            f"expanded_nodes={expanded_nodes}, "
            f"reduced_nodes={state.num_nodes}, "
            f"protected={len(replay_nodes)}, "
            f"reservoir={len(reservoir.buffer)}"
        )


if model is None or state is None:
    raise RuntimeError(
        "模型未初始化。请检查 WINDOW_SIZE 是否小于有效数据量。"
    )





print("\n=== TACO Diagnostics ===")
print(f"Continual periods: {period_idx}")
print(
    f"Final reduced graph nodes: "
    f"{state.num_nodes}"
)
print(
    f"Reservoir size: "
    f"{len(reservoir.buffer)} / {BUFFER_SIZE}"
)
print(
    f"Original training IDs represented in final graph: "
    f"{len(state.n2c)}"
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

if len(y_true_test) != len(test_idx):
    print(
        "[WARNING] Evaluated test sample count differs from hold-out size: "
        f"{len(y_true_test)} vs {len(test_idx)}"
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
    "macro avg",
    "weighted avg",
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
        figsize=(5.6, 5.0)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    disp.plot(
        values_format="d",
        cmap="Blues",
        colorbar=False,
        ax=ax_cm,
    )

    ax_cm.set_xlabel(
        "Predicted label"
    )
    ax_cm.set_ylabel(
        "True label"
    )

    fig_cm.tight_layout()
    plt.show()
    plt.close(fig_cm)

except Exception as e:
    print(
        "绘制混淆矩阵出错：",
        e,
    )





try:
    if (
        y_prob_test_all.shape[0] == len(y_true_test)
        and len(y_true_test) > 0
    ):
        if NUM_CLASSES == 2:
            if len(np.unique(y_true_test)) == 2:
                fpr, tpr, _ = roc_curve(
                    y_true_test,
                    y_prob_test_all[:, 1],
                )
                roc_auc = auc(fpr, tpr)

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
                    linestyle="--",
                )
                ax_roc.set_xlabel(
                    "False Positive Rate"
                )
                ax_roc.set_ylabel(
                    "True Positive Rate"
                )
                ax_roc.legend(
                    loc="lower right"
                )
                fig_roc.tight_layout()
                plt.show()
                plt.close(fig_roc)

                print(
                    f"ROC-AUC: "
                    f"{roc_auc * 100:.2f}%"
                )

        else:
            y_true_bin = label_binarize(
                y_true_test,
                classes=np.arange(NUM_CLASSES),
            )

            fig_roc, ax_roc = plt.subplots(
                figsize=(7.2, 5.4)
            )

            plotted = False
            auc_values = []

            for cls_id in range(NUM_CLASSES):
                y_true_c = y_true_bin[:, cls_id]
                y_score_c = y_prob_test_all[:, cls_id]

                if (
                    y_true_c.sum() == 0
                    or y_true_c.sum() == len(y_true_c)
                ):
                    continue

                fpr, tpr, _ = roc_curve(
                    y_true_c,
                    y_score_c,
                )
                roc_auc = auc(fpr, tpr)
                auc_values.append(roc_auc)

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
                ax_roc.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                )
                ax_roc.set_xlabel(
                    "False Positive Rate"
                )
                ax_roc.set_ylabel(
                    "True Positive Rate"
                )
                ax_roc.legend(
                    fontsize=8,
                    loc="lower right",
                )
                fig_roc.tight_layout()
                plt.show()

                if auc_values:
                    print(
                        "Macro OvR ROC-AUC: "
                        f"{np.mean(auc_values) * 100:.2f}%"
                    )

            plt.close(fig_roc)

except Exception as e:
    print(
        "ROC 计算/绘制出错：",
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

        n_tsne = len(
            x_tsne
        )

        perplexity = min(
            30.0,
            max(
                5.0,
                (n_tsne - 1)
                / 3.0,
            ),
        )

        perplexity = min(
            perplexity,
            float(
                n_tsne - 1
            ),
        )

        emb_2d = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
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
            "t-SNE Dim 1"
        )

        ax_tsne.set_ylabel(
            "t-SNE Dim 2"
        )

        ax_tsne.set_title(
            "TACO IoT23 Test Hidden Embeddings"
        )

        ax_tsne.legend()

        fig_tsne.tight_layout()
        plt.show()
        plt.close(fig_tsne)

    else:
        print(
            "t-SNE: 测试隐藏向量过少，跳过可视化。"
        )

except Exception as e:
    print(
        "t-SNE 可视化失败：",
        e,
    )

