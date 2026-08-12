
import warnings
warnings.filterwarnings("ignore")

from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

CSV_PATH = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'

WINDOW_SIZE = 1000
BATCH_SIZE = 100

# Keep the same incremental schedule as the current CTU13 experiments.
EPOCHS_FIRST = 10
EPOCHS_INC = 1

CORR_THRESHOLD = 0.1
TEST_RATIO = 0.30
SEED = 42

# PDGNN paper / public implementation settings.
PDGNN_HIDDEN = 256
PDGNN_L = 2                    # paper: L=2
LR = 5e-3
WEIGHT_DECAY = 5e-4            # public implementation default
MLP_DROPOUT = 0.0
LINEAR_BIAS = False

TEM_STORE_RATIO = 0.50
TEM_CAPACITY = 1000            # fixed number of topology-aware embeddings
TEM_SAMPLER = "degree"         # "degree" (official code default) or "exact_coverage"

TSNE_MAX_PER_CLASS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV_PATH)

if "Label" not in df.columns:
    raise KeyError("未找到列名 'Label'。请确认 CTU13 CSV 中标签列名为 Label。")

feature_cols = [c for c in df.columns if c not in ("num", "Label")]
if len(feature_cols) == 0:
    raise RuntimeError("未找到可用特征列。")

labels = df["Label"].astype(int).to_numpy(dtype=np.int64)
unique_labels = np.unique(labels)
if len(unique_labels) != 2 or set(unique_labels.tolist()) != {0, 1}:
    raise RuntimeError(
        f"当前代码按 CTU13 二分类 0/1 设置，实际标签为 {unique_labels.tolist()}。"
    )

id_to_label = {0: "Normal", 1: "Botnet"}

feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

N_total = len(df)
all_idx = np.arange(N_total, dtype=np.int64)

train_idx, test_idx = train_test_split(
    all_idx,
    test_size=TEST_RATIO,
    stratify=labels,
    random_state=SEED,
    shuffle=True,
)

# Train-only imputation.
medians = feat_df.iloc[train_idx].median(numeric_only=True)
feat_df = feat_df.fillna(medians).fillna(0.0)

# Train-only standardization.
scaler = StandardScaler(with_mean=True, with_std=True)
features = np.empty_like(feat_df.values, dtype=np.float64)
features[train_idx] = scaler.fit_transform(
    feat_df.iloc[train_idx].values.astype(float)
)
features[test_idx] = scaler.transform(
    feat_df.iloc[test_idx].values.astype(float)
)

N = len(features)
is_train = np.zeros(N, dtype=bool)
is_train[train_idx] = True
is_test = ~is_train

NUM_CLASSES = len(np.unique(labels[train_idx]))
if NUM_CLASSES != 2:
    raise RuntimeError("训练集未同时包含 Normal 和 Botnet 两类。")

print("Device:", DEVICE)
print("CTU13 label mapping:", id_to_label)
print(f"Samples: total={N_total}, train={len(train_idx)}, test={len(test_idx)}")
print(f"Features: {len(feature_cols)}")
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
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def build_train_adjacency(x_train_np: np.ndarray) -> np.ndarray:
    n = x_train_np.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    corr = safe_row_corrcoef(x_train_np)
    adj = (np.abs(corr) >= CORR_THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


def build_inductive_eval_adjacency(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
):
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]
    n_total = n_train + n_test

    x_all = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = safe_row_corrcoef(x_all)

    adj = np.zeros((n_total, n_total), dtype=np.float32)

    if n_train > 0:
        # train -> train
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


def normalized_adjacency(adj_np: np.ndarray) -> torch.Tensor:
    A = torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)

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
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    A_hat = normalized_adjacency(adj_np)

    e = x
    for _ in range(L):
        e = A_hat @ e

    return e.detach()


class PDGNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=LINEAR_BIAS)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=LINEAR_BIAS)

    def forward(self, te: torch.Tensor):
        h = self.fc1(te)
        h = F.relu(h)
        h = F.dropout(h, p=MLP_DROPOUT, training=self.training)
        logits = self.fc2(h)
        return logits, h

class TEMBuffer:
    def __init__(self, capacity: int, feature_dim: int, seed: int):
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

        # Number of already processed selected candidates.
        self.n_seen_selected = 0

    def __len__(self):
        return int(self.labels.numel())

    @torch.no_grad()
    def add(self, vecs: torch.Tensor, labels_t: torch.Tensor):
        if vecs.numel() == 0:
            return

        for i in range(vecs.size(0)):
            self.n_seen_selected += 1
            v = vecs[i:i + 1].detach()
            y = labels_t[i:i + 1].detach()

            if len(self) < self.capacity:
                self.vecs = torch.cat([self.vecs, v], dim=0)
                self.labels = torch.cat([self.labels, y], dim=0)
            else:
                # Reservoir replacement for bounded streaming memory.
                j = int(self.rng.integers(0, self.n_seen_selected))
                if j < self.capacity:
                    self.vecs[j] = v[0]
                    self.labels[j] = y[0]


def exact_two_hop_coverage_counts(adj_np: np.ndarray) -> np.ndarray:
    reach = adj_np.astype(bool)
    if PDGNN_L <= 1:
        return reach.sum(axis=1).astype(np.float64)

    reach_l = reach.copy()
    base = reach.astype(np.uint8)

    # Boolean matrix propagation.
    for _ in range(1, PDGNN_L):
        reach_l = (reach_l.astype(np.uint8) @ base) > 0

    return reach_l.sum(axis=1).astype(np.float64)


def select_tem_candidates(
    candidate_local_ids: np.ndarray,
    y_train_np: np.ndarray,
    adj_np: np.ndarray,
    rng: np.random.Generator,
):
    candidate_local_ids = np.asarray(candidate_local_ids, dtype=np.int64)
    if candidate_local_ids.size == 0:
        return np.empty((0,), dtype=np.int64)

    if TEM_SAMPLER == "degree":
        scores_all = adj_np.sum(axis=1).astype(np.float64)
    elif TEM_SAMPLER == "exact_coverage":
        scores_all = exact_two_hop_coverage_counts(adj_np)
    else:
        raise ValueError(
            "TEM_SAMPLER must be 'degree' or 'exact_coverage'."
        )

    selected = []

    for cls_id in np.unique(y_train_np[candidate_local_ids]):
        cls_ids = candidate_local_ids[
            y_train_np[candidate_local_ids] == cls_id
        ]
        if cls_ids.size == 0:
            continue

        # Streaming adaptation of the memory budget:
        # keep approximately 10% of new samples from each class.
        k = max(1, int(np.ceil(TEM_STORE_RATIO * cls_ids.size)))
        k = min(k, cls_ids.size)

        scores = scores_all[cls_ids].astype(np.float64)
        scores = np.maximum(scores, 1e-12)
        probs = scores / scores.sum()

        chosen = rng.choice(
            cls_ids,
            size=k,
            replace=False,
            p=probs,
        )
        selected.extend(chosen.tolist())

    return np.asarray(selected, dtype=np.int64)


# ============================================================
# 5. Loss
# ============================================================
def class_balanced_ce(logits: torch.Tensor, y: torch.Tensor):
    """
    Class-size reweighting, following the TEM public implementation:
      weight_c = 1 / n_c
    """
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=DEVICE)

    present = counts > 0
    weights[present] = 1.0 / counts[present]

    return F.cross_entropy(logits, y, weight=weights)


# ============================================================
# 6. Streaming training
# ============================================================
features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

model = None
optimizer = None
tem = None

global_idx = 0
session_idx = 0

# Test collections.
y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []

sampling_rng = np.random.default_rng(SEED + 2024)

print("\n=== Streaming PDGNN + TEM Training ===")

for start in tqdm(
    range(0, N, BATCH_SIZE),
    desc="Processing batches",
):
    end = min(start + BATCH_SIZE, N)
    batch_feats = features[start:end]
    batch_labels = labels[start:end]
    bsz = len(batch_feats)

    for i in range(bsz):
        features_window.append(batch_feats[i])
        labels_window.append(int(batch_labels[i]))
        index_window.append(global_idx)
        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    first_window = (model is None)

    x_win_np = np.asarray(features_window, dtype=np.float64)
    y_win_np = np.asarray(labels_window, dtype=np.int64)
    idx_win_np = np.asarray(index_window, dtype=np.int64)

    # New positions in the full sliding window.
    new_mask_full = np.zeros(len(x_win_np), dtype=bool)
    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[idx_win_np]
    test_mask_full = is_test[idx_win_np]

    # --------------------------------------------------------
    # 6.1 Training graph: TRAIN nodes only
    # --------------------------------------------------------
    x_train_np = x_win_np[train_mask_full]
    y_train_np = y_win_np[train_mask_full]
    new_train_mask_local_np = new_mask_full[train_mask_full]

    if x_train_np.shape[0] == 0:
        continue

    train_adj_np = build_train_adjacency(x_train_np)
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

    # Current session examples:
    #   first window -> all current training nodes
    #   later windows -> only newly arrived training nodes
    if first_window:
        current_ids = np.arange(len(x_train_np), dtype=np.int64)
        epochs_now = EPOCHS_FIRST
    else:
        current_ids = np.where(new_train_mask_local_np)[0].astype(np.int64)
        epochs_now = EPOCHS_INC

    if current_ids.size > 0:
        current_ids_t = torch.tensor(
            current_ids,
            dtype=torch.long,
            device=DEVICE,
        )

        current_te = topo_train[current_ids_t]
        current_y = torch.tensor(
            y_train_np[current_ids],
            dtype=torch.long,
            device=DEVICE,
        )

        # IMPORTANT:
        # Replay uses old TEM only. Newly selected TEs are added after this
        # session's optimization and therefore benefit future sessions.
        if len(tem) > 0:
            train_te = torch.cat([current_te, tem.vecs], dim=0)
            train_y = torch.cat([current_y, tem.labels], dim=0)
        else:
            train_te = current_te
            train_y = current_y

        for _ in range(epochs_now):
            model.train()
            logits, _ = model(train_te)
            loss = class_balanced_ce(logits, train_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Populate TEM after learning the current session.
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
            tem.add(topo_train[selected_t], selected_y)

    session_idx += 1

    # --------------------------------------------------------
    # 6.2 Evaluation: TEST nodes never train the model
    # --------------------------------------------------------
    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = new_mask_full & test_mask_full

    if eval_test_mask_full.any():
        x_new_test_np = x_win_np[eval_test_mask_full]
        y_new_test_np = y_win_np[eval_test_mask_full]

        x_eval_np, eval_adj_np = build_inductive_eval_adjacency(
            x_train_np,
            x_new_test_np,
        )

        topo_eval = topology_aware_embeddings(
            x_eval_np,
            eval_adj_np,
            L=PDGNN_L,
        )

        n_train_context = x_train_np.shape[0]
        test_te = topo_eval[n_train_context:]

        model.eval()
        with torch.no_grad():
            logits_test, hidden_test_batch = model(test_te)
            probs_test = torch.softmax(logits_test, dim=1)

        y_true_test.extend(y_new_test_np.tolist())
        y_pred_test.extend(
            torch.argmax(logits_test, dim=1).cpu().numpy().tolist()
        )
        y_prob_test_all.append(probs_test.cpu().numpy())
        hidden_test.extend(hidden_test_batch.cpu().numpy().tolist())


if model is None:
    raise RuntimeError(
        "模型未初始化。请检查 WINDOW_SIZE 是否小于有效数据量。"
    )


# ============================================================
# 7. Diagnostics
# ============================================================
print("\n=== PDGNN + TEM Diagnostics ===")
print(f"Streaming sessions: {session_idx}")
print(f"TEM current size: {len(tem)} / {TEM_CAPACITY}")
print(f"TEM selected candidates seen: {tem.n_seen_selected}")
print("TEM stores fixed-size topology-aware embeddings, not raw ego-subgraphs.")


# ============================================================
# 8. Evaluation
# ============================================================
y_true_test = np.asarray(y_true_test, dtype=np.int64)
y_pred_test = np.asarray(y_pred_test, dtype=np.int64)

if len(y_prob_test_all) > 0:
    y_prob_test_all = np.vstack(y_prob_test_all)
else:
    y_prob_test_all = np.empty((0, NUM_CLASSES), dtype=np.float64)

print("\n=== Evaluation on Random Hold-out (30%) ===")
print(f"Expected hold-out samples: {len(test_idx)}")
print(f"Actually evaluated samples: {len(y_true_test)}")

if len(y_true_test) != len(test_idx):
    print(
        "[WARNING] Evaluated test sample count differs from hold-out size: "
        f"{len(y_true_test)} vs {len(test_idx)}"
    )

if len(y_true_test) == 0:
    raise RuntimeError("测试集为空或没有测试样本被评估。")

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

print("\nClassification Report:")
for lab in target_names:
    m = report[lab]
    print(
        f"{lab:<28} "
        f"precision: {m['precision'] * 100:.2f}%  "
        f"recall: {m['recall'] * 100:.2f}%  "
        f"f1-score: {m['f1-score'] * 100:.2f}%"
    )

print(f"{'accuracy':<28}: {report['accuracy'] * 100:.2f}%")

for k in ["macro avg", "weighted avg"]:
    m = report[k]
    print(
        f"{k:<28} "
        f"precision: {m['precision'] * 100:.2f}%  "
        f"recall: {m['recall'] * 100:.2f}%  "
        f"f1-score: {m['f1-score'] * 100:.2f}%"
    )


# ============================================================
# 9. Confusion matrix
# ============================================================
try:
    cm = confusion_matrix(
        y_true_test,
        y_pred_test,
        labels=unique_ids,
    )

    fig_cm, ax_cm = plt.subplots(figsize=(5.6, 5.0))
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
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    fig_cm.tight_layout()
    plt.show()
    plt.close(fig_cm)
except Exception as e:
    print("绘制混淆矩阵出错：", e)


# ============================================================
# 10. Binary ROC
# ============================================================
try:
    if (
        NUM_CLASSES == 2
        and y_prob_test_all.shape[0] == len(y_true_test)
        and len(np.unique(y_true_test)) == 2
    ):
        fpr, tpr, _ = roc_curve(
            y_true_test,
            y_prob_test_all[:, 1],
        )
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(6.0, 5.0))
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        plt.show()
        plt.close(fig_roc)

        print(f"ROC-AUC: {roc_auc * 100:.2f}%")
except Exception as e:
    print("ROC 计算/绘制出错：", e)


# ============================================================
# 11. t-SNE of PDGNN hidden embeddings
# ============================================================
try:
    hidden_np = np.asarray(hidden_test, dtype=np.float64)

    if len(hidden_np) > 10:
        labels_tsne = y_true_test[:len(hidden_np)]

        finite_mask = np.isfinite(hidden_np).all(axis=1)
        hidden_np = hidden_np[finite_mask]
        labels_tsne = labels_tsne[finite_mask]

        # Stratified display-only sampling.
        rng = np.random.default_rng(SEED)
        selected = []

        for cls_id in np.unique(labels_tsne):
            cls_idx = np.where(labels_tsne == cls_id)[0]
            if len(cls_idx) > TSNE_MAX_PER_CLASS:
                cls_idx = rng.choice(
                    cls_idx,
                    size=TSNE_MAX_PER_CLASS,
                    replace=False,
                )
            selected.append(np.asarray(cls_idx, dtype=np.int64))

        selected = np.concatenate(selected)
        selected.sort()

        x_tsne = hidden_np[selected]
        y_tsne = labels_tsne[selected]

        x_tsne = StandardScaler().fit_transform(x_tsne)
        x_tsne = np.nan_to_num(
            x_tsne,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        n_tsne = len(x_tsne)
        perplexity = min(
            30.0,
            max(5.0, (n_tsne - 1) / 3.0),
        )
        perplexity = min(perplexity, float(n_tsne - 1))

        emb_2d = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(x_tsne)

        fig_tsne, ax_tsne = plt.subplots(figsize=(7.0, 6.0))
        for cls_id in np.unique(y_tsne):
            mask = y_tsne == cls_id
            ax_tsne.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=12,
                alpha=0.75,
                label=id_to_label.get(int(cls_id), str(cls_id)),
            )

        ax_tsne.set_xlabel("t-SNE Dim 1")
        ax_tsne.set_ylabel("t-SNE Dim 2")
        ax_tsne.set_title("PDGNN-TEM Test Hidden Embeddings")
        ax_tsne.legend()
        fig_tsne.tight_layout()
        plt.show()
        plt.close(fig_tsne)
    else:
        print("t-SNE: 测试隐藏向量过少，跳过可视化。")
except Exception as e:
    print("t-SNE 可视化失败：", e)
    
