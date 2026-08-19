
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

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from tqdm import tqdm


CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'

WINDOW_SIZE = 1000
BATCH_SIZE = 100
FAST_EPOCHS = 2
SLOW_EPOCHS_FIRST = 2
SLOW_EPOCHS_INC = 1

CORR_THRESHOLD = 0.1
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
SEED = 42

# Official TRACE toy-code defaults.
EMB_DIM = 64
LR = 3e-4
WEIGHT_DECAY = 1e-5
FEATURE_MASK_P = 0.10
EDGE_DROP_P = 0.20
INIT_NUM_CLUSTERS = 5
BETA = 0.01
VARIANCE_THRESHOLD = 0.10
COMPACTNESS_THRESHOLD_MODE = "avg"   # "avg" or "28"
MEMORY_RETENTION_THRESHOLD = 0.80

MAX_SPACED_MEMORY_CHECK = 10
MAX_SPACED_REPLAY = 2

MAX_RECURSION_DEPTH = 3
LINEAR_PROBE_MAX_ITER = 1000
TSNE_MAX_PER_CLASS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)
if "Label" not in df.columns:
    raise KeyError("未找到 Label 列。")

feature_cols = [c for c in df.columns if c not in ("num", "Label")]
labels = df["Label"].astype(int).to_numpy(np.int64)

if set(np.unique(labels).tolist()) != {0, 1}:
    raise RuntimeError(f"当前代码按二分类 0/1 设置，实际标签={np.unique(labels)}")

id_to_label = {0: "Normal", 1: "Botnet"}

feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
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
NUM_CLASSES = 2

print("Device:", DEVICE)
print("CTU13 label mapping:", id_to_label)
print(f"Samples: total={N}, train={len(train_idx)}, test={len(test_idx)}")
print(f"Features: {features.shape[1]}")


# ============================================================
# 2. Graph utilities
# ============================================================
def safe_corr(x_np):
    if len(x_np) == 0:
        return np.empty((0, 0), dtype=np.float64)
    if len(x_np) == 1:
        return np.ones((1, 1), dtype=np.float64)
    return np.nan_to_num(np.corrcoef(x_np), nan=0.0, posinf=0.0, neginf=0.0)


def build_train_graph(x_np):
    n = len(x_np)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    adj = np.abs(safe_corr(x_np)) >= CORR_THRESHOLD
    np.fill_diagonal(adj, False)
    src, dst = np.where(adj)
    return torch.tensor(
        np.vstack([src.astype(np.int64), dst.astype(np.int64)]),
        dtype=torch.long, device=DEVICE
    )


def build_eval_graph(x_train_np, x_test_np):
    n_train = len(x_train_np)
    x_all = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = safe_corr(x_all)

    src_parts, dst_parts = [], []

    if n_train > 0:
        # train -> train
        adj_tt = np.abs(corr[:n_train, :n_train]) >= CORR_THRESHOLD
        np.fill_diagonal(adj_tt, False)
        s, d = np.where(adj_tt)
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

    if src_parts:
        edge_index = torch.tensor(
            np.vstack([np.concatenate(src_parts), np.concatenate(dst_parts)]),
            dtype=torch.long, device=DEVICE
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    return (
        torch.tensor(x_all, dtype=torch.float32, device=DEVICE),
        edge_index
    )


# ============================================================
# 3. TRACE augmentation + encoder
# ============================================================
def mask_features(x, p):
    if p <= 0:
        return x
    keep = torch.bernoulli(
        (1.0 - p) * torch.ones((1, x.size(1)), device=x.device)
    )
    return x * keep


def drop_edges(edge_index, p):
    if p <= 0 or edge_index.numel() == 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= p
    return edge_index[:, keep]


def augment(x, edge_index):
    return (
        (mask_features(x, FEATURE_MASK_P), drop_edges(edge_index, EDGE_DROP_P)),
        (mask_features(x, FEATURE_MASK_P), drop_edges(edge_index, EDGE_DROP_P))
    )


class TraceGCN(nn.Module):
    def __init__(self, in_dim, out_dim=EMB_DIM):
        super().__init__()
        self.c1 = GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
        self.c2 = GCNConv(out_dim, out_dim, add_self_loops=True, normalize=True)

    def forward(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        return self.c2(h, edge_index)


def barlow_loss(z_a, z_b):
    eps = 1e-15
    n, d = z_a.shape
    if n <= 1:
        return ((z_a - z_b) ** 2).mean()

    a = (z_a - z_a.mean(0)) / (z_a.std(0, unbiased=False) + eps)
    b = (z_b - z_b.mean(0)) / (z_b.std(0, unbiased=False) + eps)
    c = (a.T @ b) / n

    diag = (1.0 - torch.diagonal(c)).pow(2).sum()
    mask = ~torch.eye(d, dtype=torch.bool, device=c.device)
    off = c[mask].pow(2).sum()
    return diag + (1.0 / d) * off


def cos_matrix(a, b):
    a = F.normalize(a, dim=-1, eps=1e-12)
    b = F.normalize(b, dim=-1, eps=1e-12)
    return a @ b.T


# ============================================================
# 4. Progressive clustering node proxies
# ============================================================
def cluster_once(embs, ids, k):
    ids = np.asarray(ids, dtype=np.int64)
    k = max(1, min(int(k), len(ids)))
    subset = embs[ids]

    if k == 1:
        center = subset.mean(0, keepdims=True)
        dist = np.linalg.norm(subset - center, axis=1)
        rep = int(ids[np.argmin(dist)])
        return [{
            "ids": ids, "count": len(ids),
            "compactness": float(dist.mean()), "rep": rep
        }]

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    pseudo = km.fit_predict(subset)
    out = []

    for cid in range(k):
        local = np.where(pseudo == cid)[0]
        if len(local) == 0:
            continue
        original = ids[local]
        cemb = subset[local]
        dist = np.linalg.norm(cemb - km.cluster_centers_[cid], axis=1)
        out.append({
            "ids": original,
            "count": len(original),
            "compactness": float(dist.mean()),
            "rep": int(original[np.argmin(dist)])
        })
    return out


def progressive_clustering(embs, ids, k, depth=0):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) <= 2 or depth >= MAX_RECURSION_DEPTH:
        return ids.tolist()

    info = cluster_once(embs, ids, k)
    comp = [x["compactness"] for x in info]

    if len(comp) <= 1 or np.var(comp) < VARIANCE_THRESHOLD:
        return [x["rep"] for x in info]

    if COMPACTNESS_THRESHOLD_MODE == "28":
        s = sorted(comp, reverse=True)
        q = max(int(np.ceil(len(s) * 0.2)), 0)
        threshold = (
            min(float(np.mean(s)), float(np.max(s)))
            if q == 0 else s[min(q, len(s) - 1)]
        )
    else:
        threshold = float(np.mean(comp))

    result = []
    total = float(len(ids))

    for item in info:
        if item["compactness"] > threshold and item["count"] >= 3:
            frac = item["count"] / total
            new_k = max(2, int(round(k * frac)))
            if new_k <= 2 and frac < 0.2:
                result.append(item["rep"])
            else:
                result.extend(
                    progressive_clustering(
                        embs, item["ids"], new_k, depth + 1
                    )
                )
        else:
            result.append(item["rep"])

    return list(dict.fromkeys(int(x) for x in result))


# ============================================================
# 5. Proxy memory
# ============================================================
class ProxyMemory:
    def __init__(self, x, edge_index, novelty):
        self.x = x.detach().cpu()
        self.edge_index = edge_index.detach().cpu()
        self.novelty = float(novelty)

    def tensors(self):
        return self.x.to(DEVICE), self.edge_index.to(DEVICE)


def select_current_proxies(x_train_np, fast_emb):
    embs = fast_emb.detach().cpu().numpy()
    ids = progressive_clustering(
        embs, np.arange(len(embs)),
        min(INIT_NUM_CLUSTERS, len(embs))
    )
    if not ids:
        ids = [0]

    ids = np.asarray(ids, dtype=np.int64)
    proxy_np = x_train_np[ids]
    return (
        torch.tensor(proxy_np, dtype=torch.float32, device=DEVICE),
        build_train_graph(proxy_np),
        ids
    )


@torch.no_grad()
def novelty(encoder, x, edge_index):
    z = encoder(x, edge_index)
    return float(cos_matrix(z, z).mean().item())


@torch.no_grad()
def retention(encoder, memory):
    x, e = memory.tensors()
    now = novelty(encoder, x, e)
    return round(now / max(abs(memory.novelty), 1e-8), 2)


# ============================================================
# 6. Fast / slow learning
# ============================================================
def train_fast(x, edge_index, in_dim):
    enc = TraceGCN(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for _ in range(FAST_EPOCHS):
        enc.train()
        (xa, ea), (xb, eb) = augment(x, edge_index)
        loss = barlow_loss(enc(xa, ea), enc(xb, eb))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    enc.eval()
    with torch.no_grad():
        emb = enc(x, edge_index)
    return enc, emb


def relation_loss(current, reference, cur_proxy, old_proxy):
    cx, ce = cur_proxy
    ox, oe = old_proxy

    (cxa, cea), (cxb, ceb) = augment(cx, ce)
    (oxa, oea), (oxb, oeb) = augment(ox, oe)

    with torch.no_grad():
        rca, rcb = reference(cxa, cea), reference(cxb, ceb)
        roa, rob = reference(oxa, oea), reference(oxb, oeb)
        ref_cross_a, ref_cross_b = cos_matrix(rca, roa), cos_matrix(rcb, rob)
        ref_self_a, ref_self_b = cos_matrix(roa, roa), cos_matrix(rob, rob)

    cca, ccb = current(cxa, cea), current(cxb, ceb)
    coa, cob = current(oxa, oea), current(oxb, oeb)

    loss_cross = (
        F.mse_loss(cos_matrix(cca, coa), ref_cross_a)
        + F.mse_loss(cos_matrix(ccb, cob), ref_cross_b)
    ) / 2.0

    loss_self = (
        F.mse_loss(cos_matrix(coa, coa), ref_self_a)
        + F.mse_loss(cos_matrix(cob, cob), ref_self_b)
    ) / 2.0

    return loss_cross + loss_self


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_reference(state, in_dim):
    m = TraceGCN(in_dim).to(DEVICE)
    m.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ============================================================
# 7. Linear probe
# ============================================================
def fit_probe(train_emb, train_y):
    classes = np.unique(train_y)
    if len(classes) < 2:
        return {"constant": int(classes[0])}

    clf = LogisticRegression(
        max_iter=LINEAR_PROBE_MAX_ITER,
        class_weight="balanced",
        random_state=SEED
    )
    clf.fit(train_emb, train_y)
    return clf


def probe_predict(probe, emb):
    if isinstance(probe, dict):
        c = probe["constant"]
        pred = np.full(len(emb), c, dtype=np.int64)
        prob = np.zeros((len(emb), NUM_CLASSES), dtype=np.float64)
        prob[:, c] = 1.0
        return pred, prob

    pred = probe.predict(emb).astype(np.int64)
    raw = probe.predict_proba(emb)
    prob = np.zeros((len(emb), NUM_CLASSES), dtype=np.float64)
    for j, c in enumerate(probe.classes_):
        prob[:, int(c)] = raw[:, j]
    return pred, prob


# ============================================================
# 8. Streaming state
# ============================================================
features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

slow_encoder = None
slow_optimizer = None

proxy_memories = []
historical_states = []
memory_retention = []

global_idx = 0
session_idx = 0

y_true_test, y_pred_test = [], []
y_prob_test_all, hidden_test = [], []

window_ids, window_acc, window_p, window_r, window_f1 = [], [], [], [], []
proxy_counts, spaced_counts = [], []


# ============================================================
# 9. Main loop
# ============================================================
print("\n=== Streaming TRACE Training ===")

for start in tqdm(range(0, N, BATCH_SIZE), desc="Processing batches"):
    batch_x = features[start:start + BATCH_SIZE]
    batch_y = labels[start:start + BATCH_SIZE]
    bsz = len(batch_x)

    for i in range(bsz):
        features_window.append(batch_x[i])
        labels_window.append(int(batch_y[i]))
        index_window.append(global_idx)
        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    session_idx += 1
    first_window = slow_encoder is None

    x_win = np.asarray(features_window, dtype=np.float64)
    y_win = np.asarray(labels_window, dtype=np.int64)
    idx_win = np.asarray(index_window, dtype=np.int64)

    new_mask = np.zeros(len(x_win), dtype=bool)
    if first_window:
        new_mask[:] = True
    else:
        new_mask[-bsz:] = True

    train_mask = is_train[idx_win]
    test_mask = is_test[idx_win]

    x_train = x_win[train_mask]
    y_train = y_win[train_mask]
    if len(x_train) == 0:
        continue

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    train_edges = build_train_graph(x_train)
    in_dim = x_train_t.size(1)

    # ---------- fast learning + proxy selection ----------
    fast_encoder, fast_emb = train_fast(x_train_t, train_edges, in_dim)
    cur_px, cur_pe, cur_proxy_ids = select_current_proxies(x_train, fast_emb)
    proxy_counts.append(len(cur_proxy_ids))

    # ---------- slow encoder ----------
    if first_window:
        slow_encoder = TraceGCN(in_dim).to(DEVICE)
        slow_encoder.load_state_dict(fast_encoder.state_dict())
        slow_optimizer = torch.optim.AdamW(
            slow_encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        previous_encoder = None
        slow_epochs = SLOW_EPOCHS_FIRST
    else:
        previous_encoder = copy.deepcopy(slow_encoder).to(DEVICE)
        previous_encoder.eval()
        for p in previous_encoder.parameters():
            p.requires_grad_(False)
        slow_epochs = SLOW_EPOCHS_INC

    # ---------- adaptive spaced replay trigger ----------
    spaced_ids = []
    if not first_window and len(proxy_memories) >= 2:
        start_j = max(0, len(proxy_memories) - 1 - MAX_SPACED_MEMORY_CHECK)
        for j in range(start_j, len(proxy_memories) - 1):
            rj = retention(slow_encoder, proxy_memories[j])
            memory_retention[j] = rj
            if (
                rj < MEMORY_RETENTION_THRESHOLD
                or rj > 2.0 - MEMORY_RETENTION_THRESHOLD
            ):
                spaced_ids.append(j)

    spaced_counts.append(len(spaced_ids))

    # ---------- slow learning ----------
    for _ in range(slow_epochs):
        slow_encoder.train()

        (xa, ea), (xb, eb) = augment(x_train_t, train_edges)
        loss = barlow_loss(
            slow_encoder(xa, ea),
            slow_encoder(xb, eb)
        )

        # Most recent proxy relation is always preserved.
        if previous_encoder is not None and proxy_memories:
            last_x, last_e = proxy_memories[-1].tensors()
            loss += BETA * relation_loss(
                slow_encoder,
                previous_encoder,
                (cur_px, cur_pe),
                (last_x, last_e)
            )

        # Older proxy memories are replayed only when retention degrades/shifts.
        for j in spaced_ids:
            ref = load_reference(historical_states[j], in_dim)
            old_x, old_e = proxy_memories[j].tensors()

            rel = relation_loss(
                slow_encoder,
                ref,
                (cur_px, cur_pe),
                (old_x, old_e)
            )

            weight = BETA * min(1.0, max(0.0, memory_retention[j]))
            loss += weight * rel
            del ref

        slow_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        slow_optimizer.step()

    # ---------- save proxy memory ----------
    cur_novelty = novelty(slow_encoder, cur_px, cur_pe)
    proxy_memories.append(ProxyMemory(cur_px, cur_pe, cur_novelty))
    historical_states.append(cpu_state(slow_encoder))
    memory_retention.append(1.0)

    # ---------- current-window linear probe ----------
    slow_encoder.eval()
    with torch.no_grad():
        train_emb = slow_encoder(x_train_t, train_edges).cpu().numpy()

    probe = fit_probe(train_emb, y_train)

    # ---------- progressive hold-out evaluation ----------
    eval_test_mask = test_mask if first_window else (new_mask & test_mask)

    if eval_test_mask.any():
        x_test_new = x_win[eval_test_mask]
        y_test_new = y_win[eval_test_mask]

        x_eval_t, eval_edges = build_eval_graph(x_train, x_test_new)

        with torch.no_grad():
            emb_eval = slow_encoder(x_eval_t, eval_edges)

        test_emb = emb_eval[len(x_train):].cpu().numpy()
        pred, prob = probe_predict(probe, test_emb)

        y_true_test.extend(y_test_new.tolist())
        y_pred_test.extend(pred.tolist())
        y_prob_test_all.append(prob)
        hidden_test.extend(test_emb.tolist())

    # ---------- whole-window performance curve ----------
    if test_mask.any():
        x_test_window = x_win[test_mask]
        y_test_window = y_win[test_mask]

        x_eval_t, eval_edges = build_eval_graph(x_train, x_test_window)
        with torch.no_grad():
            emb_eval = slow_encoder(x_eval_t, eval_edges)

        test_emb = emb_eval[len(x_train):].cpu().numpy()
        pred, _ = probe_predict(probe, test_emb)

        window_ids.append(session_idx)
        window_acc.append(accuracy_score(y_test_window, pred))
        window_p.append(precision_score(
            y_test_window, pred, average="macro",
            labels=np.arange(NUM_CLASSES), zero_division=0
        ))
        window_r.append(recall_score(
            y_test_window, pred, average="macro",
            labels=np.arange(NUM_CLASSES), zero_division=0
        ))
        window_f1.append(f1_score(
            y_test_window, pred, average="macro",
            labels=np.arange(NUM_CLASSES), zero_division=0
        ))

    if session_idx == 1 or session_idx % 20 == 0:
        print(
            f"\n[TRACE] session={session_idx}, proxies={proxy_counts[-1]}, "
            f"memories={len(proxy_memories)}, spaced_replay={spaced_counts[-1]}"
        )

    del fast_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if slow_encoder is None:
    raise RuntimeError("模型未初始化，请检查 WINDOW_SIZE。")


# ============================================================
# 10. Diagnostics + Window-level curve
# ============================================================
print("\n=== TRACE Diagnostics ===")
print(f"Sessions: {session_idx}")
print(f"Stored proxy memories: {len(proxy_memories)}")
print(f"Mean proxies/session: {np.mean(proxy_counts):.2f}")
print(f"Mean spaced memories/session: {np.mean(spaced_counts):.2f}")

if window_ids:
    print("\n=== Window-level Performance ===")
    print(
        f"Accuracy={np.mean(window_acc)*100:.2f}%  "
        f"MacroP={np.mean(window_p)*100:.2f}%  "
        f"MacroR={np.mean(window_r)*100:.2f}%  "
        f"MacroF1={np.mean(window_f1)*100:.2f}%"
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(window_ids, np.asarray(window_acc)*100, label="Accuracy")
    ax.plot(window_ids, np.asarray(window_p)*100, label="Precision")
    ax.plot(window_ids, np.asarray(window_r)*100, label="Recall")
    ax.plot(window_ids, np.asarray(window_f1)*100, label="F1-score")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Performance (%)")
    ax.set_title("TRACE Window-level Performance Curve")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ============================================================
# 11. Final hold-out report
# ============================================================
y_true_test = np.asarray(y_true_test, dtype=np.int64)
y_pred_test = np.asarray(y_pred_test, dtype=np.int64)
y_prob_test_all = (
    np.vstack(y_prob_test_all)
    if y_prob_test_all
    else np.empty((0, NUM_CLASSES), dtype=np.float64)
)

print("\n=== Evaluation on Random Test Split (15%) ===")
print(f"Expected hold-out samples: {len(test_idx)}")
print(f"Actually evaluated samples: {len(y_true_test)}")

if len(y_true_test) == 0:
    raise RuntimeError("没有测试样本被评估。")

target_names = [id_to_label[i] for i in range(NUM_CLASSES)]
report = classification_report(
    y_true_test, y_pred_test,
    labels=np.arange(NUM_CLASSES),
    target_names=target_names,
    output_dict=True, zero_division=0
)

for lab in target_names:
    m = report[lab]
    print(
        f"{lab:<20} precision={m['precision']*100:.2f}%  "
        f"recall={m['recall']*100:.2f}%  "
        f"f1={m['f1-score']*100:.2f}%"
    )

print(f"{'accuracy':<20} {report['accuracy']*100:.2f}%")
for k in ["macro avg", "weighted avg"]:
    m = report[k]
    print(
        f"{k:<20} precision={m['precision']*100:.2f}%  "
        f"recall={m['recall']*100:.2f}%  "
        f"f1={m['f1-score']*100:.2f}%"
    )


# ============================================================
# 12. Confusion matrix
# ============================================================
try:
    cm = confusion_matrix(
        y_true_test, y_pred_test, labels=np.arange(NUM_CLASSES)
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ConfusionMatrixDisplay(
        cm, display_labels=target_names
    ).plot(values_format="d", cmap="Blues", colorbar=False, ax=ax)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
except Exception as e:
    print("Confusion matrix failed:", e)


# ============================================================
# 13. ROC
# ============================================================
try:
    if (
        y_prob_test_all.shape[0] == len(y_true_test)
        and len(np.unique(y_true_test)) == 2
    ):
        fpr, tpr, _ = roc_curve(y_true_test, y_prob_test_all[:, 1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"ROC-AUC: {roc_auc*100:.2f}%")
except Exception as e:
    print("ROC failed:", e)


# ============================================================
# 14. t-SNE
# ============================================================
try:
    h = np.asarray(hidden_test, dtype=np.float64)
    if len(h) > 10:
        y = y_true_test[:len(h)]
        good = np.isfinite(h).all(axis=1)
        h, y = h[good], y[good]

        rng = np.random.default_rng(SEED)
        selected = []
        for cls in np.unique(y):
            ids = np.where(y == cls)[0]
            if len(ids) > TSNE_MAX_PER_CLASS:
                ids = rng.choice(ids, TSNE_MAX_PER_CLASS, replace=False)
            selected.append(ids)

        ids = np.concatenate(selected)
        h, y = h[ids], y[ids]
        h = StandardScaler().fit_transform(h)

        perplexity = min(30.0, max(5.0, (len(h)-1)/3.0), float(len(h)-1))
        xy = TSNE(
            n_components=2, random_state=SEED,
            perplexity=perplexity, init="pca", learning_rate="auto"
        ).fit_transform(h)

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        for cls in np.unique(y):
            m = y == cls
            ax.scatter(
                xy[m, 0], xy[m, 1], s=12, alpha=0.75,
                label=id_to_label[int(cls)]
            )
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.set_title("TRACE Test Embeddings")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)
except Exception as e:
    print("t-SNE failed:", e)

