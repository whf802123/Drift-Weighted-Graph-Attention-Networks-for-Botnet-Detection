import warnings
warnings.filterwarnings("ignore")

from collections import deque
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from tqdm import tqdm

CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'

WINDOW_SIZE = 1000
BATCH_SIZE = 100
EPOCHS_FIRST = 10
EPOCHS_INC = 2

CORR_THRESHOLD = 0.1
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
SEED = 42

HIDDEN_DIM = 32
ATTN_HEADS = 4
LR = 5e-3
WEIGHT_DECAY = 0.0

PROP_STEPS = 2
CONDENSE_PER_CLASS = 8
CHAIN_SIM_THRESHOLD = 0.75
ATTACH_TOPK = 5

TUCKER_NODE_RANK = 16
TUCKER_FEATURE_RANK = 16
TUCKER_TIME_RANK = 8
HOOI_ITERS = 5

MAX_HISTORY_SNAPSHOTS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)

if 'Label' not in df.columns:
    raise KeyError("Column 'Label' was not found.")

feature_cols = [c for c in df.columns if c not in ('num', 'Label')]
if len(feature_cols) == 0:
    raise RuntimeError("No usable feature columns were found.")

labels = df['Label'].astype(int).to_numpy(dtype=np.int64)
unique_labels = np.unique(labels)

if len(unique_labels) != 2 or set(unique_labels.tolist()) != {0, 1}:
    raise RuntimeError(
        f"This code expects binary CTU13 labels 0/1, but found {unique_labels.tolist()}."
    )

id_to_label = {0: 'Normal', 1: 'Botnet'}

feat_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')

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

train_labels_only = labels[train_idx]
pos_ratio = (train_labels_only == 1).mean() + 1e-8
w_neg = 1.0 / max(1e-8, 1.0 - pos_ratio)
w_pos = 1.0 / max(1e-8, pos_ratio)

print("CTU13 label mapping:", id_to_label)
print(f"Samples: total={N_total}, train={len(train_idx)}, test={len(test_idx)}")
print(f"Features: {len(feature_cols)}")


def safe_corrcoef(x_np):
    n = x_np.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(x_np)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def build_base_graph(x_np):
    n = x_np.shape[0]
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


def normalized_adjacency_numpy(x_np):
    n = x_np.shape[0]
    corr = safe_corrcoef(x_np)
    adj = (np.abs(corr) >= CORR_THRESHOLD).astype(np.float64)
    np.fill_diagonal(adj, 1.0)

    deg = adj.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


def propagated_features(x_np):
    a_hat = normalized_adjacency_numpy(x_np)
    h = x_np.copy()
    for _ in range(PROP_STEPS):
        h = a_hat @ h
    return h


def condense_snapshot(x_np, y_np):
    h = propagated_features(x_np)

    condensed_x = []
    condensed_y = []

    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        if len(idx) == 0:
            continue

        h_cls = h[idx]
        k = min(CONDENSE_PER_CLASS, len(h_cls))

        if k == 1:
            centers = h_cls.mean(axis=0, keepdims=True)
        else:
            km = KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=10,
            )
            km.fit(h_cls)
            centers = km.cluster_centers_

        condensed_x.append(centers)
        condensed_y.extend([int(cls)] * len(centers))

    if not condensed_x:
        return np.empty((0, x_np.shape[1])), np.empty((0,), dtype=np.int64)

    return (
        np.concatenate(condensed_x, axis=0).astype(np.float64),
        np.asarray(condensed_y, dtype=np.int64),
    )


def row_normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def greedy_match(prev_x, curr_x, threshold):
    if len(prev_x) == 0 or len(curr_x) == 0:
        return []

    p = row_normalize(prev_x)
    c = row_normalize(curr_x)
    sim = p @ c.T

    candidates = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            candidates.append((float(sim[i, j]), i, j))

    candidates.sort(key=lambda z: z[0], reverse=True)

    used_prev = set()
    used_curr = set()
    matches = []

    for score, i, j in candidates:
        if score < threshold:
            break
        if i in used_prev or j in used_curr:
            continue
        matches.append((i, j, score))
        used_prev.add(i)
        used_curr.add(j)

    return matches


def build_node_chains(snapshot_features):
    t_count = len(snapshot_features)
    if t_count == 0:
        return []

    chains = [[i] for i in range(len(snapshot_features[0]))]

    for t in range(1, t_count):
        prev_x = snapshot_features[t - 1]
        curr_x = snapshot_features[t]

        prev_node_to_chain = {}

        for chain_id, chain in enumerate(chains):
            prev_node = chain[t - 1] if t - 1 < len(chain) else None
            if prev_node is not None:
                prev_node_to_chain[prev_node] = chain_id

        for chain in chains:
            chain.append(None)

        matches = greedy_match(
            prev_x,
            curr_x,
            CHAIN_SIM_THRESHOLD,
        )

        matched_curr = set()

        for prev_node, curr_node, _ in matches:
            chain_id = prev_node_to_chain.get(prev_node)
            if chain_id is not None:
                chains[chain_id][t] = curr_node
                matched_curr.add(curr_node)

        for curr_node in range(len(curr_x)):
            if curr_node not in matched_curr:
                chain = [None] * t + [curr_node]
                chains.append(chain)

    return chains


def fill_chain_tensor(snapshot_features, chains):
    if not chains:
        return None

    t_count = len(snapshot_features)
    d = snapshot_features[0].shape[1]

    tensor = np.zeros(
        (len(chains), d, t_count),
        dtype=np.float64,
    )

    for c_idx, chain in enumerate(chains):
        observed = []

        for t, node_id in enumerate(chain):
            if node_id is not None and node_id < len(snapshot_features[t]):
                observed.append(t)

        if not observed:
            continue

        for t in range(t_count):
            node_id = chain[t] if t < len(chain) else None

            if node_id is not None and node_id < len(snapshot_features[t]):
                tensor[c_idx, :, t] = snapshot_features[t][node_id]
            else:
                nearest = min(observed, key=lambda z: abs(z - t))
                nearest_node = chain[nearest]
                tensor[c_idx, :, t] = snapshot_features[nearest][nearest_node]

    return tensor


def unfold(x, mode):
    return np.reshape(
        np.moveaxis(x, mode, 0),
        (x.shape[mode], -1),
    )


def mode_product(x, matrix, mode):
    y = np.tensordot(matrix, x, axes=[1, mode])
    return np.moveaxis(y, 0, mode)


def top_left_singular_vectors(matrix, rank):
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank]


def hooi_factors(x, ranks, n_iter=5):
    r1 = min(ranks[0], x.shape[0])
    r2 = min(ranks[1], x.shape[1])
    r3 = min(ranks[2], x.shape[2])

    u = top_left_singular_vectors(unfold(x, 0), r1)
    v = top_left_singular_vectors(unfold(x, 1), r2)
    w = top_left_singular_vectors(unfold(x, 2), r3)

    for _ in range(n_iter):
        z = mode_product(x, v.T, 1)
        z = mode_product(z, w.T, 2)
        u = top_left_singular_vectors(unfold(z, 0), r1)

        z = mode_product(x, u.T, 0)
        z = mode_product(z, w.T, 2)
        v = top_left_singular_vectors(unfold(z, 1), r2)

        z = mode_product(x, u.T, 0)
        z = mode_product(z, v.T, 1)
        w = top_left_singular_vectors(unfold(z, 2), r3)

    return u, v, w


def historical_node_factors(snapshot_features):
    if len(snapshot_features) < 2:
        return None

    chains = build_node_chains(snapshot_features)
    tensor = fill_chain_tensor(snapshot_features, chains)

    if tensor is None or tensor.shape[0] == 0:
        return None

    u, _, _ = hooi_factors(
        tensor,
        (
            TUCKER_NODE_RANK,
            TUCKER_FEATURE_RANK,
            TUCKER_TIME_RANK,
        ),
        n_iter=HOOI_ITERS,
    )

    out = np.zeros(
        (u.shape[0], TUCKER_NODE_RANK),
        dtype=np.float32,
    )
    out[:, :u.shape[1]] = u.astype(np.float32)
    return out


def cosine_topk_edges(source_x, target_x, source_offset, target_offset, k):
    if len(source_x) == 0 or len(target_x) == 0:
        return [], []

    s = row_normalize(source_x)
    t = row_normalize(target_x)
    sim = s @ t.T

    src_edges = []
    dst_edges = []

    k_eff = min(k, len(target_x))

    for i in range(len(source_x)):
        idx = np.argpartition(-sim[i], k_eff - 1)[:k_eff]
        for j in idx:
            src_edges.append(source_offset + i)
            dst_edges.append(target_offset + int(j))

    return src_edges, dst_edges


def build_extended_train_graph(x_train_np, generated_np):
    n_orig = len(x_train_np)
    n_gen = len(generated_np)

    base_edge = build_base_graph(x_train_np).detach().cpu().numpy()

    src = base_edge[0].tolist()
    dst = base_edge[1].tolist()

    if n_gen > 0:
        s1, d1 = cosine_topk_edges(
            generated_np,
            x_train_np,
            n_orig,
            0,
            ATTACH_TOPK,
        )
        src.extend(s1)
        dst.extend(d1)

        src.extend(d1)
        dst.extend(s1)

        for i in range(n_gen):
            src.append(n_orig + i)
            dst.append(n_orig + i)

    return torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
        device=DEVICE,
    )


def build_inductive_eval_graph(context_np, test_np):
    n_ctx = len(context_np)
    n_test = len(test_np)

    if n_test == 0:
        return None

    corr_ctx = safe_corrcoef(context_np)
    adj_ctx = np.abs(corr_ctx) >= CORR_THRESHOLD
    np.fill_diagonal(adj_ctx, False)
    src_ctx, dst_ctx = np.where(adj_ctx)

    src = src_ctx.astype(np.int64).tolist()
    dst = dst_ctx.astype(np.int64).tolist()

    for i in range(n_ctx):
        src.append(i)
        dst.append(i)

    if n_ctx > 0:
        all_x = np.concatenate([context_np, test_np], axis=0)
        corr = safe_corrcoef(all_x)
        corr_ctx_test = corr[:n_ctx, n_ctx:]

        src_ct, test_col = np.where(
            np.abs(corr_ctx_test) >= CORR_THRESHOLD
        )

        src.extend(src_ct.astype(np.int64).tolist())
        dst.extend((n_ctx + test_col).astype(np.int64).tolist())

    for i in range(n_test):
        src.append(n_ctx + i)
        dst.append(n_ctx + i)

    return torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
        device=DEVICE,
    )


class CADGCLModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, num_classes):
        super().__init__()

        self.history_proj = nn.Linear(
            TUCKER_NODE_RANK,
            in_dim,
            bias=False,
        )

        self.gat1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            add_self_loops=False,
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def generate_history_nodes(self, factors):
        if factors is None or factors.numel() == 0:
            return None
        return self.history_proj(factors)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        return self.classifier(h), h


features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

history_snapshots = deque(maxlen=MAX_HISTORY_SNAPSHOTS)

model = None
optimizer = None

class_weights = torch.tensor(
    [w_neg, w_pos],
    dtype=torch.float,
    device=DEVICE,
)
criterion = nn.CrossEntropyLoss(weight=class_weights)

y_true_test = []
y_pred_test = []

global_idx = 0

for start in tqdm(
    range(0, N, BATCH_SIZE),
    desc="Processing CA-DGCL windows",
):
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
        WINDOW_SIZE,
        dtype=bool,
    )

    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[idx_win_np]
    test_mask_full = is_test[idx_win_np]

    x_train_np = x_win_np[train_mask_full]
    y_train_np = y_win_np[train_mask_full]

    new_train_mask_local_np = new_mask_full[train_mask_full]

    if len(x_train_np) == 0:
        continue

    if model is None:
        model = CADGCLModel(
            in_dim=x_train_np.shape[1],
            hidden_dim=HIDDEN_DIM,
            heads=ATTN_HEADS,
            num_classes=2,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        epochs_now = EPOCHS_FIRST
    else:
        epochs_now = EPOCHS_INC

    historical_factors_np = historical_node_factors(
        list(history_snapshots)
    )

    if historical_factors_np is None:
        historical_factors = None
    else:
        historical_factors = torch.tensor(
            historical_factors_np,
            dtype=torch.float,
            device=DEVICE,
        )

    x_train_tensor = torch.tensor(
        x_train_np,
        dtype=torch.float,
        device=DEVICE,
    )

    y_train_tensor = torch.tensor(
        y_train_np,
        dtype=torch.long,
        device=DEVICE,
    )

    if first_window:
        used_idx = torch.arange(
            len(x_train_np),
            device=DEVICE,
        )
    else:
        used_idx = torch.where(
            torch.tensor(
                new_train_mask_local_np,
                dtype=torch.bool,
                device=DEVICE,
            )
        )[0]

        if used_idx.numel() == 0:
            condensed_x, _ = condense_snapshot(
                x_train_np,
                y_train_np,
            )
            history_snapshots.append(condensed_x)
            continue

    for _ in range(epochs_now):
        model.train()

        generated_tensor = model.generate_history_nodes(
            historical_factors
        )

        if generated_tensor is None:
            generated_np = np.empty(
                (0, x_train_np.shape[1]),
                dtype=np.float32,
            )
            x_ext_tensor = x_train_tensor
        else:
            generated_np = (
                generated_tensor.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            x_ext_tensor = torch.cat(
                [x_train_tensor, generated_tensor],
                dim=0,
            )

        edge_ext = build_extended_train_graph(
            x_train_np,
            generated_np,
        )

        logits_ext, _ = model(
            x_ext_tensor,
            edge_ext,
        )

        logits_orig = logits_ext[:len(x_train_np)]

        loss = criterion(
            logits_orig[used_idx],
            y_train_tensor[used_idx],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = (
            new_mask_full & test_mask_full
        )

    if eval_test_mask_full.any():
        x_test_np = x_win_np[
            eval_test_mask_full
        ]
        y_test_np = y_win_np[
            eval_test_mask_full
        ]

        model.eval()

        with torch.no_grad():
            generated_tensor = model.generate_history_nodes(
                historical_factors
            )

            if generated_tensor is None:
                generated_np = np.empty(
                    (0, x_train_np.shape[1]),
                    dtype=np.float64,
                )
            else:
                generated_np = (
                    generated_tensor
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            context_np = np.concatenate(
                [x_train_np, generated_np],
                axis=0,
            )

            x_eval_np = np.concatenate(
                [context_np, x_test_np],
                axis=0,
            )

            x_eval_tensor = torch.tensor(
                x_eval_np,
                dtype=torch.float,
                device=DEVICE,
            )

            eval_edge = build_inductive_eval_graph(
                context_np,
                x_test_np,
            )

            logits_eval, _ = model(
                x_eval_tensor,
                eval_edge,
            )

            n_ctx = len(context_np)

            pred = torch.argmax(
                logits_eval[
                    n_ctx:n_ctx + len(x_test_np)
                ],
                dim=1,
            ).cpu().numpy()

        y_true_test.extend(
            y_test_np.tolist()
        )
        y_pred_test.extend(
            pred.tolist()
        )

    condensed_x, _ = condense_snapshot(
        x_train_np,
        y_train_np,
    )

    history_snapshots.append(
        condensed_x
    )


y_true_test = np.asarray(
    y_true_test,
    dtype=np.int64,
)

y_pred_test = np.asarray(
    y_pred_test,
    dtype=np.int64,
)

print("\n=== CA-DGCL Evaluation on CTU13 ===")
print(f"Expected hold-out samples: {len(test_idx)}")
print(f"Actually evaluated samples: {len(y_true_test)}")

if len(y_true_test) == 0:
    print("The test set is empty.")
else:
    report = classification_report(
        y_true_test,
        y_pred_test,
        labels=[0, 1],
        target_names=[
            id_to_label[0],
            id_to_label[1],
        ],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    for name in [
        'Normal',
        'Botnet',
    ]:
        m = report[name]
        print(
            f"{name:<12} "
            f"precision: {m['precision'] * 100:.2f}%  "
            f"recall: {m['recall'] * 100:.2f}%  "
            f"f1-score: {m['f1-score'] * 100:.2f}%"
        )

    print(
        f"{'accuracy':<12}: "
        f"{report['accuracy'] * 100:.2f}%"
    )

    for key in [
        'macro avg',
        'weighted avg',
    ]:
        m = report[key]
        print(
            f"{key:<12} "
            f"precision: {m['precision'] * 100:.2f}%  "
            f"recall: {m['recall'] * 100:.2f}%  "
            f"f1-score: {m['f1-score'] * 100:.2f}%"
        )

    cm = confusion_matrix(
        y_true_test,
        y_pred_test,
        labels=[0, 1],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            'Normal',
            'Botnet',
        ],
    )

    fig, ax = plt.subplots(
        figsize=(5.6, 5.0)
    )

    disp.plot(
        values_format='d',
        cmap='Blues',
        colorbar=False,
        ax=ax,
    )

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    fig.tight_layout()
    plt.show()
    plt.close(fig)
