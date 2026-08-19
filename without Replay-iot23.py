
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import subprocess

import pandas as pd
import numpy as np
from collections import deque
import ipaddress

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
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

SWEEP_CHILD = os.environ.get("CORR_SWEEP_CHILD", "0") == "1"

if not SWEEP_CHILD:
    script_path = os.path.abspath(__file__)
    output_dir = os.path.dirname(script_path)
    thresholds = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]

    results = []
    print("=" * 72)
    print("Starting CORR_THRESHOLD sweep:", thresholds)
    print("=" * 72)

    for threshold in thresholds:
        print(f"\n>>> Running CORR_THRESHOLD = {threshold:.1f}")

        child_env = os.environ.copy()
        child_env["CORR_SWEEP_CHILD"] = "1"
        child_env["CORR_THRESHOLD"] = f"{threshold:.1f}"
        # 防止批量运行时 matplotlib 图窗阻塞子进程。
        child_env["MPLBACKEND"] = "Agg"

        proc = subprocess.run(
            [sys.executable, script_path],
            env=child_env,
            cwd=output_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # 保存每个阈值的完整日志，便于后续追溯。
        log_path = os.path.join(
            output_dir,
            f"corr_threshold_{threshold:.1f}.log",
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n\n===== STDERR =====\n")
                f.write(proc.stderr)

        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(
                f"CORR_THRESHOLD={threshold:.1f} 运行失败，"
                f"请查看日志：{log_path}"
            )

        result_prefix = "__CORR_SWEEP_RESULT__="
        result_lines = [
            line for line in proc.stdout.splitlines()
            if line.startswith(result_prefix)
        ]
        if not result_lines:
            print(proc.stdout)
            raise RuntimeError(
                f"CORR_THRESHOLD={threshold:.1f} 未输出可解析指标，"
                f"请查看日志：{log_path}"
            )

        metrics = json.loads(result_lines[-1][len(result_prefix):])
        results.append(metrics)

        print(
            f"    Accuracy={metrics['Accuracy']:.4f}% | "
            f"Macro Precision={metrics['Macro Precision']:.4f}% | "
            f"Macro Recall={metrics['Macro Recall']:.4f}% | "
            f"Macro F1-score={metrics['Macro F1-score']:.4f}%"
        )

    # ===== 保存结果表 =====
    results_df = pd.DataFrame(results).sort_values("CORR_THRESHOLD")
    csv_out = os.path.join(output_dir, "corr_threshold_sweep_results.csv")
    results_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n=== CORR_THRESHOLD Sweep Results ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults CSV saved to: {csv_out}")

    # ===== 四项指标画在同一张图 =====
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    metric_columns = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-score",
    ]

    for metric in metric_columns:
        ax.plot(
            results_df["CORR_THRESHOLD"],
            results_df[metric],
            marker="o",
            linewidth=2,
            label=metric,
        )

    ax.set_xlabel("CORR_THRESHOLD")
    ax.set_ylabel("Score (%)")
    ax.set_title("Performance vs. CORR_THRESHOLD")
    ax.set_xticks(thresholds)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    # 根据实际结果自动缩放 Y 轴，便于观察小幅变化，同时不越过 0~100%。
    y_min = float(results_df[metric_columns].min().min())
    y_max = float(results_df[metric_columns].max().max())
    y_span = max(y_max - y_min, 1.0)
    margin = max(1.0, 0.12 * y_span)
    ax.set_ylim(max(0.0, y_min - margin), min(100.0, y_max + margin))

    fig.tight_layout()
    fig_out = os.path.join(output_dir, "corr_threshold_metrics.png")
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    print(f"Metric figure saved to: {fig_out}")

    plt.show()
    plt.close(fig)

    # 父进程只负责扫描，不进入下面的单次训练代码。
    sys.exit(0)


# ==================== 0. 配置 ====================
CSV_PATH         = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\iot23_combined_new.csv'
WINDOW_SIZE      = 1000
BATCH_SIZE       = 100
GAT_EPOCHS_FIRST = 10
GAT_EPOCHS_INC   = 2
ATTN_HEADS       = 4
HIDDEN_CHANNELS  = 8
CORR_THRESHOLD   = float(os.environ.get("CORR_THRESHOLD", "0.5"))
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
LR               = 5e-3
WEIGHT_DECAY     = 0.0
SEED             = 42

USE_REPLAY       = True
REPLAY_RATIO     = 0.10

ENTROPY_LAMBDA   = 1e-3

# ==================== IRGD 配置 ====================
# Intrinsic-Relational Gradient Decomposition
# full graph gradient = intrinsic gradient + relational gradient
# 只有两者发生负冲突时，才投影 relational gradient 的冲突分量。
IRGD_ENABLED          = True
IRGD_ON_FIRST_WINDOW  = False   # 首窗口保持原 baseline 初始化
IRGD_REL_WEIGHT       = 1.0     # =1 时，无冲突情况下最终梯度严格等于 baseline full-graph 梯度
IRGD_EPS              = 1e-12
IRGD_GRAD_CLIP        = None    # 例如设为 5.0；None 表示与 baseline 一样不裁剪

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# === 采样策略：仅手动指定少数类全保留，其余一律按比例采样 ===
SAMPLE_FRAC = 0.05
MANUAL_MINORITY_LABELS = {'C&C', 'C&C-HeartBeat'}

print(f"\n[Run config] CORR_THRESHOLD = {CORR_THRESHOLD:.1f}")


# ==================== 1. 数据加载与预处理 ====================
def is_unnamed(colname):
    return (colname == '') or pd.isna(colname) or str(colname).lower().startswith('unnamed')


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

# 丢弃空名/Unnamed 首列
if is_unnamed(df.columns[0]):
    df.drop(df.columns[0], axis=1, inplace=True)

# 插入原始行号
# 注意：后续经过采样/reset_index 后，num 保留的是采样前原始位置，可用于追踪。
df.insert(0, 'num', range(len(df)))

if 'label' not in df.columns:
    raise KeyError("未找到列名 'label'。请确认 CSV 中的标签列名为小写 label。")

# ===== 移除指定的极少数类 =====
rare_labels_to_drop = {
    'C&C-FileDownload', 'C&C-Torii', 'FileDownload',
    'C&C-HeartBeat-FileDownload', 'Okiru-Attack', 'C&C-Mirai', 'Attack'
}
before = len(df)
df = df[~df['label'].isin(rare_labels_to_drop)].reset_index(drop=True)
after = len(df)
print(f"Dropped rare classes {sorted(list(rare_labels_to_drop))}. Rows: {before} -> {after}")

# ===== 手动指定少数类全保留，其余类别统一采样 =====
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
print("Label counts after manual-minority keep & uniform sampling:")
print(label_counts_sampled.to_string())

# stratify 至少要求每类 >= 2
# 这里继续沿用原代码的保护逻辑。
too_small = set(label_counts_sampled[label_counts_sampled < 2].index.tolist())
if too_small:
    print(f"[NOTICE] Classes with <2 samples after sampling are dropped: {sorted(list(too_small))}")
    df = df[~df['label'].isin(too_small)].reset_index(drop=True)

# ====== 列分组 ======
ip_cols = [c for c in ['id.orig_h', 'id.resp_h'] if c in df.columns]
num_cols_raw = [
    c for c in [
        'ts', 'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
        'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
    ] if c in df.columns
]
cat_cols = [
    c for c in ['proto', 'service', 'conn_state', 'history', 'local_orig', 'local_resp']
    if c in df.columns
]

for c in ip_cols:
    df[c] = df[c].map(ip_to_int)

for c in num_cols_raw:
    df[c] = df[c].map(to_float)

for c in cat_cols:
    df[c] = df[c].astype(str).fillna('-').replace({'nan': '-'})

# 标签映射
label_text = df['label'].astype(str).values
unique_labels_text = pd.unique(label_text)
label_to_id = {lab: i for i, lab in enumerate(unique_labels_text)}
id_to_label = {v: k for k, v in label_to_id.items()}
labels = np.array([label_to_id[x] for x in label_text], dtype=int)
print("\nLabel mapping after sampling:", label_to_id)

# 特征矩阵
num_df = (
    df[num_cols_raw + ip_cols].copy()
    if (num_cols_raw or ip_cols)
    else pd.DataFrame(index=df.index)
)
cat_df = (
    pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False)
    if cat_cols
    else pd.DataFrame(index=df.index)
)
feat_df = pd.concat([num_df, cat_df], axis=1)
if feat_df.shape[1] == 0:
    raise RuntimeError("未能构建任何特征列，请检查输入 CSV。")

# ===== 分层随机 hold-out =====
N_total = len(df)
all_idx = np.arange(N_total)
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

# 缺失值填补：只使用训练集统计量
medians = feat_df.iloc[train_idx].median(numeric_only=True)
feat_df = feat_df.fillna(medians)

# 标准化：只在训练集 fit
scaler = StandardScaler(with_mean=True, with_std=True)
features = np.empty_like(feat_df.values, dtype=np.float64)
features[train_idx] = scaler.fit_transform(feat_df.iloc[train_idx].values.astype(float))
features[test_idx] = scaler.transform(feat_df.iloc[test_idx].values.astype(float))
features[val_idx] = scaler.transform(feat_df.iloc[val_idx].values.astype(float))

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

# 类别数只根据训练集确认；stratify 保证正常情况下训练集包含所有类别。
train_unique_ids = np.unique(labels[train_idx])
NUM_CLASSES = len(train_unique_ids)
if NUM_CLASSES != len(np.unique(labels)):
    raise RuntimeError("训练集未包含全部类别，无法进行当前多分类设置。")


# ==================== 2. 图构建 ====================
def _safe_row_corrcoef(x_np: np.ndarray) -> np.ndarray:
    """计算样本-样本相关矩阵，并把常量行造成的 NaN/Inf 安全置零。"""
    if x_np.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if x_np.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(x_np)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def build_train_graph(x_train_np: np.ndarray):
    """
    仅根据当前窗口中的训练节点构建图。

    关键修复：
    1) 测试节点绝不进入训练图；
    2) 每个窗口重新构图，不保存旧窗口的局部节点编号；
    3) np.where 对对称邻接已经产生两个方向，因此不再重复拼接边。
    """
    n = x_train_np.shape[0]
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    corr = _safe_row_corrcoef(x_train_np)
    adj = (np.abs(corr) >= CORR_THRESHOLD)
    np.fill_diagonal(adj, False)

    src, dst = np.where(adj)

    # 显式加入 self-loop，使 GAT 与 GCN 使用完全一致的 edge_index。
    self_nodes = np.arange(n, dtype=np.int64)
    src = np.concatenate([src.astype(np.int64), self_nodes])
    dst = np.concatenate([dst.astype(np.int64), self_nodes])

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
        device=DEVICE,
    )
    return edge_index


def build_inductive_eval_graph(x_train_np: np.ndarray, x_test_np: np.ndarray):
    """
    为当前新测试样本构建严格分离的推理图。

    节点排列：
        [当前窗口训练节点, 当前需要评估的新测试节点]

    允许的边：
        train -> train
        train -> test
        self-loop

    禁止：
        test -> train
        test -> test（不同测试样本之间）

    因此测试节点不会参与参数更新，也不会改变训练节点表示；不同测试样本之间
    也不会通过 message passing 相互泄漏信息。
    """
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]
    n_total = n_train + n_test

    if n_test == 0:
        return None, None

    x_eval_np = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = _safe_row_corrcoef(x_eval_np)

    src_list = []
    dst_list = []

    # train -> train：使用当前训练上下文内部相关图。
    if n_train > 0:
        corr_tt = corr[:n_train, :n_train]
        adj_tt = (np.abs(corr_tt) >= CORR_THRESHOLD)
        np.fill_diagonal(adj_tt, False)
        src_tt, dst_tt = np.where(adj_tt)
        src_list.append(src_tt.astype(np.int64))
        dst_list.append(dst_tt.astype(np.int64))

        # train -> test：测试节点只能接收训练节点信息。
        corr_train_test = corr[:n_train, n_train:]
        train_src, test_col = np.where(np.abs(corr_train_test) >= CORR_THRESHOLD)
        if train_src.size > 0:
            src_list.append(train_src.astype(np.int64))
            dst_list.append((n_train + test_col).astype(np.int64))

    # 所有节点显式 self-loop。
    self_nodes = np.arange(n_total, dtype=np.int64)
    src_list.append(self_nodes)
    dst_list.append(self_nodes)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
        device=DEVICE,
    )
    x_eval_tensor = torch.tensor(x_eval_np, dtype=torch.float, device=DEVICE)
    return x_eval_tensor, edge_index


# ==================== 3. 模型 ====================
class WeightedGATClassifier(nn.Module):
    """
    GAT -> attention-weighted GCN -> classifier

    与原代码相比，不再维护跨窗口 edge_weight_dict。
    当前 forward 内直接把 GAT 的多头平均 attention 作为同一张当前图的 GCN 边权，
    因而不存在旧窗口局部编号错位，也更直接地实现：
        GAT attention -> GCN edge weight
    """
    def __init__(self, in_channels, hidden_channels, heads, num_classes=2):
        super().__init__()
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

    def forward(self, x, edge_index):
        x1, (ei_used, att_heads) = self.gat1(
            x,
            edge_index,
            return_attention_weights=True,
        )
        x1 = F.elu(x1)

        # 多头 GAT attention 直接成为当前 forward 中 GCN 的 edge weight。
        edge_weight = att_heads.mean(dim=1)
        self.cached_att = (ei_used, att_heads)

        x2 = self.gcn2(x1, ei_used, edge_weight=edge_weight)
        logits = self.classifier(x2)
        return logits, x2


# ==================== 4. 滑动窗口、模型与损失 ====================
features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

model = None
optimizer = None
criterion = None

# 二分类 class weight 只根据 TRAIN labels 计算，避免测试标签分布泄漏。
if NUM_CLASSES == 2:
    train_labels_only = labels[train_idx]
    pos_ratio = (train_labels_only == 1).mean() + 1e-8
    w_neg = 1.0 / max(1e-8, 1.0 - pos_ratio)
    w_pos = 1.0 / max(1e-8, pos_ratio)
else:
    w_neg = w_pos = 1.0

# 测试结果收集
# 不再缓存用于 CSV 导出的全量 embedding。
y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []

# 跟踪全局流式位置
global_idx = 0

# IRGD 诊断统计
irgd_steps = 0
irgd_conflicts = 0
irgd_cosines = []


# ==================== 5. 工具函数 ====================
def select_training_indices(
    n_nodes,
    new_train_mask_local,
    first_window=False,
):
    """
    每个 epoch 只采样一次训练节点，full-graph 与 self-only 两个分支共用同一 used_idx，
    保证梯度差异只来自图关系，而不是 replay 随机采样差异。
    """
    if n_nodes == 0:
        return None

    if first_window:
        return torch.arange(n_nodes, device=DEVICE)

    new_idx = torch.where(new_train_mask_local)[0]
    if new_idx.numel() == 0:
        return None

    used_parts = [new_idx]

    if USE_REPLAY:
        old_mask_local = ~new_train_mask_local
        old_idx = torch.where(old_mask_local)[0]
        if old_idx.numel() > 0:
            k = max(1, int(REPLAY_RATIO * old_idx.numel()))
            k = min(k, old_idx.numel())
            perm = torch.randperm(old_idx.numel(), device=DEVICE)[:k]
            used_parts.append(old_idx[perm])

    return torch.cat(used_parts, dim=0)


def build_self_only_graph(n_nodes):
    """
    Counterfactual self-only graph：每个节点只保留 self-loop。

    它使用与 full graph 完全相同的 GAT/GCN 参数，但移除所有邻居消息，
    因此得到的梯度作为 intrinsic feature update。
    """
    nodes = torch.arange(n_nodes, dtype=torch.long, device=DEVICE)
    return torch.stack([nodes, nodes], dim=0)


def _loss_on_used_nodes(logits, y_train_local, used_idx):
    if used_idx is None or used_idx.numel() == 0:
        return None
    return criterion(logits[used_idx], y_train_local[used_idx])


def _zeros_like_param(param):
    return torch.zeros_like(param, memory_format=torch.preserve_format)


def _materialize_grads(grads, params):
    """把 autograd.grad 返回的 None 安全转换为零张量。"""
    out = []
    for g, p in zip(grads, params):
        out.append(_zeros_like_param(p) if g is None else g)
    return out


def apply_irgd_step(
    model,
    optimizer,
    x_train_tensor,
    y_train_tensor,
    train_edge_index,
    self_edge_index,
    used_idx,
):
    """
    IRGD 单步更新。

    1) Full graph 得到 g_graph；
    2) Self-only graph 得到 g_self；
    3) g_rel = g_graph - g_self；
    4) 若 <g_self, g_rel> < 0，仅投影掉 g_rel 与 g_self 冲突的分量；
    5) GAT+GCN 使用 g_self + lambda * g_rel*；classifier 始终使用 g_graph。

    当不存在冲突且 IRGD_REL_WEIGHT=1 时：
        g_self + (g_graph - g_self) = g_graph
    因此更新严格退化为原 baseline。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    encoder_param_ids = {
        id(p) for p in list(model.gat1.parameters()) + list(model.gcn2.parameters())
    }

    # ---------- Full graph ----------
    logits_full, _ = model(x_train_tensor, train_edge_index)
    ce_full = _loss_on_used_nodes(logits_full, y_train_tensor, used_idx)
    if ce_full is None:
        return None

    ei_full, att_full = model.cached_att
    nodes_mask = torch.zeros(
        x_train_tensor.size(0), dtype=torch.bool, device=DEVICE
    )
    nodes_mask[used_idx] = True

    ent_full = sparse_entropy_loss_sum_heads(
        ei_full,
        att_full,
        num_nodes=x_train_tensor.size(0),
        heads=ATTN_HEADS,
        nodes_mask=nodes_mask,
    )
    loss_full = ce_full + ENTROPY_LAMBDA * ent_full

    grads_graph_raw = torch.autograd.grad(
        loss_full,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    grads_graph = _materialize_grads(grads_graph_raw, params)

    # ---------- Self-only counterfactual graph ----------
    logits_self, _ = model(x_train_tensor, self_edge_index)
    ce_self = _loss_on_used_nodes(logits_self, y_train_tensor, used_idx)

    # self-only 图每个节点只有一个 incoming self-loop，attention entropy 为 0，
    # 因而 intrinsic objective 只需要分类损失。
    grads_self_raw = torch.autograd.grad(
        ce_self,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    grads_self = _materialize_grads(grads_self_raw, params)

    # ---------- 仅在 encoder 参数上计算 intrinsic-relational conflict ----------
    dot = torch.zeros((), device=DEVICE)
    self_norm_sq = torch.zeros((), device=DEVICE)
    rel_norm_sq = torch.zeros((), device=DEVICE)

    rel_grads = []
    for p, gg, gs in zip(params, grads_graph, grads_self):
        gr = gg - gs
        rel_grads.append(gr)
        if id(p) in encoder_param_ids:
            dot = dot + torch.sum(gs * gr)
            self_norm_sq = self_norm_sq + torch.sum(gs * gs)
            rel_norm_sq = rel_norm_sq + torch.sum(gr * gr)

    conflict = bool(dot.detach().item() < 0.0)

    # cosine 仅用于实验诊断，不参与优化。
    denom = torch.sqrt(self_norm_sq.clamp_min(IRGD_EPS)) * torch.sqrt(
        rel_norm_sq.clamp_min(IRGD_EPS)
    )
    cosine = (dot / denom.clamp_min(IRGD_EPS)).detach().item()

    if conflict:
        # dot < 0。减去 relational gradient 在 intrinsic gradient 上的负投影，
        # 使投影后 <g_self, g_rel*> ~= 0。
        coeff = dot / self_norm_sq.clamp_min(IRGD_EPS)
    else:
        coeff = torch.zeros((), device=DEVICE)

    optimizer.zero_grad(set_to_none=True)

    for p, gg, gs, gr in zip(params, grads_graph, grads_self, rel_grads):
        if id(p) in encoder_param_ids:
            if conflict:
                gr_safe = gr - coeff * gs
            else:
                gr_safe = gr
            final_grad = gs + IRGD_REL_WEIGHT * gr_safe
        else:
            # 分类器保持 baseline full-graph 梯度，减少对决策头的额外干预。
            final_grad = gg

        p.grad = final_grad.detach()

    if IRGD_GRAD_CLIP is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), IRGD_GRAD_CLIP)

    optimizer.step()

    return {
        'loss_full': float(loss_full.detach().cpu()),
        'ce_full': float(ce_full.detach().cpu()),
        'ce_self': float(ce_self.detach().cpu()),
        'conflict': conflict,
        'dot': float(dot.detach().cpu()),
        'cosine': float(cosine),
    }


def attention_heads_node_mean_from_cached_incoming(
    ei_used,
    att_heads,
    num_nodes,
    heads,
    reference_nodes=None,
):
    """
    把每个节点收到的各 attention head 权重取均值。

    推理阶段可只使用训练上下文节点估计标准化均值/标准差，
    避免不同测试样本通过 embedding 标准化统计量相互影响。
    """
    dst = ei_used[1]
    E = dst.numel()
    device = att_heads.device

    in_sum = torch.zeros(num_nodes, heads, device=device)
    in_cnt = torch.zeros(num_nodes, 1, device=device)

    for h in range(heads):
        in_sum[:, h].index_add_(0, dst, att_heads[:, h])

    ones_e = torch.ones(E, device=device)
    in_cnt.index_add_(0, dst, ones_e.unsqueeze(-1))

    in_mean = in_sum / in_cnt.clamp(min=1.0)

    if reference_nodes is not None and reference_nodes > 0:
        ref = in_mean[:reference_nodes]
    else:
        ref = in_mean

    ref_mean = ref.mean(dim=0, keepdim=True)
    ref_std = ref.std(dim=0, keepdim=True)
    in_mean = (in_mean - ref_mean) / (ref_std + 1e-6)
    return in_mean


def sparse_entropy_loss_sum_heads(
    ei_used,
    att_heads,
    num_nodes,
    heads,
    nodes_mask=None,
    eps=1e-12,
):
    """按节点归一化的多头 incoming-attention entropy。"""
    dst = ei_used[1]
    E = dst.numel()
    device = att_heads.device

    deg_in = torch.zeros(num_nodes, device=device).index_add_(
        0,
        dst,
        torch.ones(E, device=device),
    )
    logZ = torch.log(deg_in.clamp_min(1.0))

    neg_p_logp_sum_heads = torch.zeros(num_nodes, device=device)
    for h in range(heads):
        p = att_heads[:, h].clamp_min(eps)
        tmp = torch.zeros(num_nodes, device=device)
        tmp.index_add_(0, dst, -(p * p.log()))
        neg_p_logp_sum_heads += tmp

    denom = (heads * logZ).clamp_min(1e-6)
    norm_entropy = neg_p_logp_sum_heads / denom
    norm_entropy = torch.where(
        logZ > 1e-6,
        norm_entropy,
        torch.zeros_like(norm_entropy),
    )

    if nodes_mask is not None:
        norm_entropy = norm_entropy[nodes_mask]

    if norm_entropy.numel() == 0:
        return torch.tensor(0.0, device=device)

    return norm_entropy.mean()


# ==================== 6. 主循环 ====================
for start in tqdm(range(0, N, BATCH_SIZE), desc='Processing batches'):
    batch_feats = features[start:start + BATCH_SIZE]
    batch_labels = labels[start:start + BATCH_SIZE]
    bsz = len(batch_feats)

    # 当前批加入滑动窗口
    for i in range(bsz):
        features_window.append(batch_feats[i])
        labels_window.append(int(batch_labels[i]))
        index_window.append(global_idx)
        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    first_window = (model is None)

    # 当前完整窗口
    x_win_np = np.asarray(features_window, dtype=np.float64)
    y_win_np = np.asarray(labels_window, dtype=np.int64)
    idx_win_np = np.asarray(index_window, dtype=np.int64)

    # 当前真正“新进入”的位置。
    # 首窗口中 WINDOW_SIZE 个节点都是首次出现；之后只有当前 batch 的 bsz 个。
    new_mask_full = np.zeros(WINDOW_SIZE, dtype=bool)
    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[idx_win_np]
    test_mask_full = is_test[idx_win_np]

    # ========== 6.1 训练：只使用当前窗口 TRAIN 节点构图 ==========
    x_train_np = x_win_np[train_mask_full]
    y_train_np = y_win_np[train_mask_full]

    # 映射 full-window new mask -> train-local mask
    new_train_mask_local_np = new_mask_full[train_mask_full]

    if x_train_np.shape[0] == 0:
        continue

    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float, device=DEVICE)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long, device=DEVICE)
    new_train_mask_local = torch.tensor(
        new_train_mask_local_np,
        dtype=torch.bool,
        device=DEVICE,
    )

    # 每个窗口重新构建当前训练图，不保留旧局部节点编号/旧边。
    train_edge_index = build_train_graph(x_train_np)

    if model is None:
        model = WeightedGATClassifier(
            in_channels=x_train_tensor.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            heads=ATTN_HEADS,
            num_classes=NUM_CLASSES,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        if NUM_CLASSES == 2:
            class_weights = torch.tensor(
                [w_neg, w_pos],
                dtype=torch.float,
                device=DEVICE,
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        epochs_now = GAT_EPOCHS_FIRST
    else:
        epochs_now = GAT_EPOCHS_INC

    # Self-only graph 在当前窗口节点数固定时只需构建一次。
    self_edge_index = build_self_only_graph(x_train_tensor.size(0))

    # 首窗口：保持原 baseline 初始化；后续窗口默认使用 IRGD。
    for _ in range(epochs_now):
        model.train()

        # 同一个 epoch 的 full/self 两个视图必须使用完全相同的新样本 + replay。
        used_idx = select_training_indices(
            x_train_tensor.size(0),
            new_train_mask_local,
            first_window=first_window,
        )
        if used_idx is None:
            continue

        use_irgd_now = IRGD_ENABLED and (IRGD_ON_FIRST_WINDOW or not first_window)

        if use_irgd_now:
            stats = apply_irgd_step(
                model=model,
                optimizer=optimizer,
                x_train_tensor=x_train_tensor,
                y_train_tensor=y_train_tensor,
                train_edge_index=train_edge_index,
                self_edge_index=self_edge_index,
                used_idx=used_idx,
            )
            if stats is not None:
                irgd_steps += 1
                irgd_conflicts += int(stats['conflict'])
                irgd_cosines.append(stats['cosine'])
        else:
            # 原 baseline 更新：首窗口默认走这里。
            logits_train, _hidden_train = model(x_train_tensor, train_edge_index)
            ce_loss = _loss_on_used_nodes(logits_train, y_train_tensor, used_idx)
            if ce_loss is None:
                continue

            ei_used, att_heads = model.cached_att
            nodes_mask = torch.zeros(
                x_train_tensor.size(0),
                dtype=torch.bool,
                device=DEVICE,
            )
            nodes_mask[used_idx] = True

            ent_loss = sparse_entropy_loss_sum_heads(
                ei_used,
                att_heads,
                num_nodes=x_train_tensor.size(0),
                heads=ATTN_HEADS,
                nodes_mask=nodes_mask,
            )

            loss = ce_loss + ENTROPY_LAMBDA * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if IRGD_GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), IRGD_GRAD_CLIP)
            optimizer.step()

    # ========== 6.2 评估：测试节点完全不参与训练 ==========
    # 首窗口要把此前尚未评估过的全部测试节点一次性评估；
    # 后续只评估当前新进入 batch 中的测试节点。
    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = new_mask_full & test_mask_full

    if eval_test_mask_full.any():
        x_new_test_np = x_win_np[eval_test_mask_full]
        y_new_test_np = y_win_np[eval_test_mask_full]

        # 推理图中：train->train + train->test + self-loop；
        # 不存在 test->train 或 test->test。
        x_eval_tensor, eval_edge_index = build_inductive_eval_graph(
            x_train_np,
            x_new_test_np,
        )

        model.eval()
        with torch.no_grad():
            logits_eval, hidden8_eval = model(x_eval_tensor, eval_edge_index)
            probs_eval = torch.softmax(logits_eval, dim=1)

            ei_used, att_heads = model.cached_att
            head4_eval = attention_heads_node_mean_from_cached_incoming(
                ei_used,
                att_heads,
                num_nodes=x_eval_tensor.size(0),
                heads=ATTN_HEADS,
                reference_nodes=x_train_np.shape[0],
            )
            mixed12_eval = torch.cat([hidden8_eval, head4_eval], dim=1)

            n_train_ctx = x_train_np.shape[0]
            test_slice = slice(n_train_ctx, n_train_ctx + len(x_new_test_np))

            logits_test = logits_eval[test_slice]
            probs_test = probs_eval[test_slice]
            mixed12_test = mixed12_eval[test_slice]

            y_true_test.extend(y_new_test_np.tolist())
            y_pred_test.extend(
                torch.argmax(logits_test, dim=1).cpu().numpy().tolist()
            )
            y_prob_test_all.append(probs_test.cpu().numpy())
            hidden_test.extend(mixed12_test.cpu().numpy().tolist())

    model.cached_att = None



# ==================== IRGD 训练诊断 ====================
if IRGD_ENABLED:
    print("\n=== IRGD Training Diagnostics ===")
    print(f"IRGD update steps: {irgd_steps}")
    if irgd_steps > 0:
        conflict_rate = 100.0 * irgd_conflicts / irgd_steps
        mean_cos = float(np.mean(irgd_cosines)) if irgd_cosines else 0.0
        print(f"Conflict steps: {irgd_conflicts} ({conflict_rate:.2f}%)")
        print(f"Mean cosine(g_self, g_rel): {mean_cos:.4f}")
        print("No-conflict steps are mathematically identical to baseline full-graph updates when IRGD_REL_WEIGHT=1.0.")


# ==================== 7. 评估 ====================
y_true_test = np.asarray(y_true_test, dtype=np.int64)
y_pred_test = np.asarray(y_pred_test, dtype=np.int64)

if len(y_prob_test_all) > 0:
    y_prob_test_all = np.vstack(y_prob_test_all)
else:
    y_prob_test_all = np.empty((0, NUM_CLASSES), dtype=np.float64)

print("\n=== Evaluation on Random Test Split (15%) ===")
print(f"Expected hold-out samples: {len(test_idx)}")
print(f"Actually evaluated samples: {len(y_true_test)}")

if len(y_true_test) == 0:
    print("测试集为空：检查划分或窗口设置。")
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

    print("Classification Report (Hold-out Test):")
    for lab in target_names:
        m = report[lab]
        print(
            f"{lab:<28} precision: {m['precision'] * 100:.2f}%  "
            f"recall: {m['recall'] * 100:.2f}%  "
            f"f1-score: {m['f1-score'] * 100:.2f}%"
        )

    print(f"{'accuracy':<28}: {report['accuracy'] * 100:.2f}%")
    for k in ['macro avg', 'weighted avg']:
        m = report[k]
        print(
            f"{k:<28} precision: {m['precision'] * 100:.2f}%  "
            f"recall: {m['recall'] * 100:.2f}%  "
            f"f1-score: {m['f1-score'] * 100:.2f}%"
        )

    # ===== 阈值扫描需要的四项汇总指标 =====
    sweep_metrics = {
        "CORR_THRESHOLD": float(CORR_THRESHOLD),
        "Accuracy": float(report["accuracy"] * 100.0),
        "Macro Precision": float(report["macro avg"]["precision"] * 100.0),
        "Macro Recall": float(report["macro avg"]["recall"] * 100.0),
        "Macro F1-score": float(report["macro avg"]["f1-score"] * 100.0),
    }
    print(
        "__CORR_SWEEP_RESULT__="
        + json.dumps(sweep_metrics, ensure_ascii=False)
    )

    # 批量扫描子进程在拿到四项指标后立即退出：
    # 不再绘制混淆矩阵 / ROC / t-SNE，避免十轮扫描产生大量图并浪费时间。
    if SWEEP_CHILD:
        sys.exit(0)

    # —— 混淆矩阵 ——
    try:
        labels_order = unique_ids.tolist()
        display_names = [id_to_label[i] for i in labels_order]
        cm = confusion_matrix(
            y_true_test,
            y_pred_test,
            labels=labels_order,
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_names,
        )
        # 显式复用同一组 axes，避免 ConfusionMatrixDisplay.plot() 再创建一张
        # Figure，导致额外出现空白图。
        fig_cm, ax_cm = plt.subplots(figsize=(5.6, 5.0))
        disp.plot(values_format='d', cmap='Blues', colorbar=False, ax=ax_cm)
        ax_cm.set_xlabel('Predicted label')
        ax_cm.set_ylabel('True label')
        fig_cm.tight_layout()
        plt.show()
        plt.close(fig_cm)
    except Exception as e:
        print("绘制混淆矩阵出错：", e)

    # —— 多分类 ROC：One-vs-Rest ——
    try:
        if NUM_CLASSES > 2:
            if y_prob_test_all.shape[0] == 0:
                print("[ROC] 没有收集到测试概率，跳过绘图。")
            else:
                classes_sorted = unique_ids.tolist()
                y_true_bin = label_binarize(
                    y_true_test,
                    classes=classes_sorted,
                )

                plt.figure(figsize=(7.2, 5.4))
                plotted = False

                for cls_id in classes_sorted:
                    col = classes_sorted.index(cls_id)
                    y_true_c = y_true_bin[:, col]
                    y_score_c = y_prob_test_all[:, cls_id]

                    if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
                        continue

                    fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(
                        fpr,
                        tpr,
                        label=f"{id_to_label[cls_id]} (AUC={roc_auc:.3f})",
                    )
                    plotted = True

                if plotted:
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend(fontsize=8, loc='lower right')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("[ROC] 各类别在测试集中都缺少正/负样本，无法绘制多分类 ROC。")
        else:
            print("[ROC] 当前为二分类，已跳过多分类 ROC。")
    except Exception as e:
        print("ROC 计算/绘制出错：", e)


# ==================== 8. t-SNE（测试样本 12 维混合向量） ====================
# 仅修改可视化，不影响训练、预测或评估指标。
# 主要修复：
#   1) 去除 NaN / Inf 样本；
#   2) 删除零方差维度；
#   3) t-SNE 前对 12D mixed embedding 再标准化，避免 hidden8 与 head4
#      数值尺度差异导致二维布局退化；
#   4) 类别分层限量抽样，避免测试集过大时 t-SNE 极慢；
#   5) 每个类别单独 scatter，保证离散标签清楚可见。
TSNE_MAX_PER_CLASS = 1000

try:
    if len(hidden_test) > 10:
        hidden_test_np = np.asarray(hidden_test, dtype=np.float64)
        labels_tsne = np.asarray(y_true_test[:len(hidden_test_np)], dtype=np.int64)

        print("\n=== t-SNE Diagnostics ===")
        print(f"Raw embedding shape: {hidden_test_np.shape}")

        # 1. 过滤非有限 embedding。
        finite_mask = np.isfinite(hidden_test_np).all(axis=1)
        if not finite_mask.all():
            bad = int((~finite_mask).sum())
            print(f"[t-SNE] Dropped {bad} samples containing NaN/Inf.")
            hidden_test_np = hidden_test_np[finite_mask]
            labels_tsne = labels_tsne[finite_mask]

        if len(hidden_test_np) <= 10:
            print("t-SNE: 有效测试隐藏向量过少，跳过可视化。")
        else:
            # 2. 删除完全/近似不变的维度。
            dim_std = hidden_test_np.std(axis=0)
            useful_dims = dim_std > 1e-10
            if useful_dims.sum() < 2:
                print("t-SNE: embedding 有效变化维度少于 2，无法可靠可视化。")
            else:
                if useful_dims.sum() != hidden_test_np.shape[1]:
                    removed = hidden_test_np.shape[1] - int(useful_dims.sum())
                    print(f"[t-SNE] Removed {removed} near-constant dimensions.")
                x_tsne = hidden_test_np[:, useful_dims]

                # 3. 12D mixed embedding 中 hidden8 与 head4 来源不同。
                #    可视化前统一尺度，避免某几维数值幅度支配 t-SNE 距离。
                x_tsne = StandardScaler().fit_transform(x_tsne)
                x_tsne = np.nan_to_num(x_tsne, nan=0.0, posinf=10.0, neginf=-10.0)
                x_tsne = np.clip(x_tsne, -10.0, 10.0)

                # 4. 分类别限量采样：只影响图，不影响前面的测试指标。
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
                x_tsne = x_tsne[selected]
                labels_plot = labels_tsne[selected]

                n_tsne = len(x_tsne)
                perplexity = min(30.0, max(5.0, (n_tsne - 1) / 3.0))
                # sklearn 要求 perplexity < n_samples。
                perplexity = min(perplexity, float(n_tsne - 1))

                print(f"Samples plotted: {n_tsne}")
                print(f"Input dimensions after filtering: {x_tsne.shape[1]}")
                print(f"Perplexity: {perplexity:.2f}")

                emb_2d = TSNE(
                    n_components=2,
                    random_state=SEED,
                    perplexity=perplexity,
                    init='pca',
                    learning_rate='auto',
                ).fit_transform(x_tsne)

                print(
                    "2D range: "
                    f"x=[{emb_2d[:, 0].min():.3f}, {emb_2d[:, 0].max():.3f}], "
                    f"y=[{emb_2d[:, 1].min():.3f}, {emb_2d[:, 1].max():.3f}]"
                )

                # 5. 显式创建并关闭当前 Figure，避免受到前面图形状态影响。
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

                ax_tsne.set_xlabel('t-SNE Dim 1')
                ax_tsne.set_ylabel('t-SNE Dim 2')
                ax_tsne.set_title('t-SNE of Test Mixed Embeddings')
                ax_tsne.legend(fontsize=7, markerscale=1.5, loc='best')
                fig_tsne.tight_layout()
                plt.show()
                plt.close(fig_tsne)
    else:
        print("t-SNE: 测试隐藏向量过少，跳过可视化。")
except Exception as e:
    print("t-SNE 可视化失败：", e)


