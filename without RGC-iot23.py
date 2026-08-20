

import warnings
warnings.filterwarnings("ignore")

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



CSV_PATH         = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\iot23_combined_new.csv'
WINDOW_SIZE      = 1000
BATCH_SIZE       = 100
GAT_EPOCHS_FIRST = 10
GAT_EPOCHS_INC   = 2
ATTN_HEADS       = 4
HIDDEN_CHANNELS  = 8
CORR_THRESHOLD   = 0.6
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
LR               = 5e-3
WEIGHT_DECAY     = 0.0
SEED             = 42

USE_REPLAY       = True
REPLAY_RATIO     = 0.10

ENTROPY_LAMBDA   = 1e-3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


SAMPLE_FRAC = 0.05
MANUAL_MINORITY_LABELS = {'C&C', 'C&C-HeartBeat'}



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


if is_unnamed(df.columns[0]):
    df.drop(df.columns[0], axis=1, inplace=True)



df.insert(0, 'num', range(len(df)))

if 'label' not in df.columns:
    raise KeyError("未找到列名 'label'。请确认 CSV 中的标签列名为小写 label。")


rare_labels_to_drop = {
    'C&C-FileDownload', 'C&C-Torii', 'FileDownload',
    'C&C-HeartBeat-FileDownload', 'Okiru-Attack', 'C&C-Mirai', 'Attack'
}
before = len(df)
df = df[~df['label'].isin(rare_labels_to_drop)].reset_index(drop=True)
after = len(df)
print(f"Dropped rare classes {sorted(list(rare_labels_to_drop))}. Rows: {before} -> {after}")


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



too_small = set(label_counts_sampled[label_counts_sampled < 2].index.tolist())
if too_small:
    print(f"[NOTICE] Classes with <2 samples after sampling are dropped: {sorted(list(too_small))}")
    df = df[~df['label'].isin(too_small)].reset_index(drop=True)


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


label_text = df['label'].astype(str).values
unique_labels_text = pd.unique(label_text)
label_to_id = {lab: i for i, lab in enumerate(unique_labels_text)}
id_to_label = {v: k for k, v in label_to_id.items()}
labels = np.array([label_to_id[x] for x in label_text], dtype=int)
print("\nLabel mapping after sampling:", label_to_id)


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
feat_df = feat_df.fillna(medians)


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


train_unique_ids = np.unique(labels[train_idx])
NUM_CLASSES = len(train_unique_ids)
if NUM_CLASSES != len(np.unique(labels)):
    raise RuntimeError("训练集未包含全部类别，无法进行当前多分类设置。")



def _safe_row_corrcoef(x_np: np.ndarray) -> np.ndarray:
    if x_np.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if x_np.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(x_np)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def build_train_graph(x_train_np: np.ndarray):
    n = x_train_np.shape[0]
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    corr = _safe_row_corrcoef(x_train_np)
    adj = (np.abs(corr) >= CORR_THRESHOLD)
    np.fill_diagonal(adj, False)

    src, dst = np.where(adj)


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
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]
    n_total = n_train + n_test

    if n_test == 0:
        return None, None

    x_eval_np = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = _safe_row_corrcoef(x_eval_np)

    src_list = []
    dst_list = []


    if n_train > 0:
        corr_tt = corr[:n_train, :n_train]
        adj_tt = (np.abs(corr_tt) >= CORR_THRESHOLD)
        np.fill_diagonal(adj_tt, False)
        src_tt, dst_tt = np.where(adj_tt)
        src_list.append(src_tt.astype(np.int64))
        dst_list.append(dst_tt.astype(np.int64))


        corr_train_test = corr[:n_train, n_train:]
        train_src, test_col = np.where(np.abs(corr_train_test) >= CORR_THRESHOLD)
        if train_src.size > 0:
            src_list.append(train_src.astype(np.int64))
            dst_list.append((n_train + test_col).astype(np.int64))


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



class WeightedGATClassifier(nn.Module):
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


        edge_weight = att_heads.mean(dim=1)
        self.cached_att = (ei_used, att_heads)

        x2 = self.gcn2(x1, ei_used, edge_weight=edge_weight)
        logits = self.classifier(x2)
        return logits, x2



features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)

model = None
optimizer = None
criterion = None


if NUM_CLASSES == 2:
    train_labels_only = labels[train_idx]
    pos_ratio = (train_labels_only == 1).mean() + 1e-8
    w_neg = 1.0 / max(1e-8, 1.0 - pos_ratio)
    w_pos = 1.0 / max(1e-8, pos_ratio)
else:
    w_neg = w_pos = 1.0



y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []


global_idx = 0



def compute_training_loss(
    logits,
    y_train_local,
    new_train_mask_local,
    first_window=False,
):
    n = logits.shape[0]
    if n == 0:
        return None, None

    if first_window:
        used_idx = torch.arange(n, device=DEVICE)
        return criterion(logits[used_idx], y_train_local[used_idx]), used_idx

    new_idx = torch.where(new_train_mask_local)[0]
    if new_idx.numel() == 0:

        return None, None

    used_parts = [new_idx]

    if USE_REPLAY:
        old_mask_local = ~new_train_mask_local
        old_idx = torch.where(old_mask_local)[0]
        if old_idx.numel() > 0:
            k = max(1, int(REPLAY_RATIO * old_idx.numel()))
            k = min(k, old_idx.numel())
            perm = torch.randperm(old_idx.numel(), device=DEVICE)[:k]
            replay_idx = old_idx[perm]
            used_parts.append(replay_idx)

    used_idx = torch.cat(used_parts, dim=0)
    return criterion(logits[used_idx], y_train_local[used_idx]), used_idx


def attention_heads_node_mean_from_cached_incoming(
    ei_used,
    att_heads,
    num_nodes,
    heads,
    reference_nodes=None,
):
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



for start in tqdm(range(0, N, BATCH_SIZE), desc='Processing batches'):
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

    first_window = (model is None)


    x_win_np = np.asarray(features_window, dtype=np.float64)
    y_win_np = np.asarray(labels_window, dtype=np.int64)
    idx_win_np = np.asarray(index_window, dtype=np.int64)



    new_mask_full = np.zeros(WINDOW_SIZE, dtype=bool)
    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True

    train_mask_full = is_train[idx_win_np]
    test_mask_full = is_test[idx_win_np]


    x_train_np = x_win_np[train_mask_full]
    y_train_np = y_win_np[train_mask_full]


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



    for _ in range(epochs_now):
        model.train()
        logits_train, _hidden_train = model(x_train_tensor, train_edge_index)

        ce_loss, used_idx = compute_training_loss(
            logits_train,
            y_train_tensor,
            new_train_mask_local,
            first_window=first_window,
        )
        if ce_loss is None:
            continue

        ei_used, att_heads = model.cached_att

        nodes_mask = None
        if used_idx is not None:
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = new_mask_full & test_mask_full

    if eval_test_mask_full.any():
        x_new_test_np = x_win_np[eval_test_mask_full]
        y_new_test_np = y_win_np[eval_test_mask_full]



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
        plt.figure(figsize=(5.6, 5.0))
        disp.plot(values_format='d', cmap='Blues', colorbar=False)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("绘制混淆矩阵出错：", e)


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



try:
    if len(hidden_test) > 10:
        hidden_test_np = np.asarray(hidden_test, dtype=np.float64)

        perplexity = min(30.0, max(5.0, (len(hidden_test_np) - 1) / 3.0))
        emb_2d = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
        ).fit_transform(hidden_test_np)

        plt.figure(figsize=(6, 6))
        plt.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=y_true_test[:len(emb_2d)],
            s=5,
        )
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.tight_layout()
        plt.show()
    else:
        print("t-SNE: 测试隐藏向量过少，跳过可视化。")
except Exception as e:
    print("t-SNE 可视化失败：", e)

