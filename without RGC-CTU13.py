

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from collections import deque
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
CSV_PATH         = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'
WINDOW_SIZE      = 1000
BATCH_SIZE       = 100
GAT_EPOCHS_FIRST = 10
GAT_EPOCHS_INC   = 2
ATTN_HEADS       = 4
HIDDEN_CHANNELS  = 8
CORR_THRESHOLD   = 0.6
TEST_RATIO       = 0.15
VAL_RATIO        = 0.15
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
df = pd.read_csv(CSV_PATH)
if 'Label' not in df.columns:
    raise KeyError("Column 'Label' not found. Please confirm the label column name in CTU13.csv is Label.")
feature_cols = [c for c in df.columns if c not in ('num', 'Label')]
if len(feature_cols) == 0:
    raise RuntimeError("No available feature columns found.")
labels = df['Label'].astype(int).to_numpy(dtype=np.int64)
unique_labels = np.unique(labels)
if not set(unique_labels.tolist()).issubset({0, 1}):
    raise ValueError(f"The current CTU13 code is configured for binary classification. Detected labels: {unique_labels.tolist()}")
id_to_label = {0: 'Normal', 1: 'Botnet'}
label_to_id = {'Normal': 0, 'Botnet': 1}
N_total = len(df)
all_idx = np.arange(N_total)
train_idx, temp_idx = train_test_split(
    all_idx,
    test_size=0.30,
    stratify=labels,
    random_state=SEED,
    shuffle=True,
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=labels[temp_idx],
    random_state=SEED,
    shuffle=True,
)
feat_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
medians = feat_df.iloc[train_idx].median(numeric_only=True)
feat_df = feat_df.fillna(medians)
feat_df = feat_df.fillna(0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
features = np.empty((N_total, len(feature_cols)), dtype=np.float64)
features[train_idx] = scaler.fit_transform(
    feat_df.iloc[train_idx].to_numpy(dtype=np.float64)
)
features[test_idx] = scaler.transform(
    feat_df.iloc[test_idx].to_numpy(dtype=np.float64)
)
N = len(features)
is_train = np.zeros(N, dtype=bool)
is_train[train_idx] = True
is_test = ~is_train
train_unique_ids = np.unique(labels[train_idx])
NUM_CLASSES = len(train_unique_ids)
if NUM_CLASSES != len(unique_labels):
    raise RuntimeError("The training set does not contain all classes.")
if NUM_CLASSES != 2:
    raise RuntimeError(f"The CTU13 ablation version requires binary classification, but the current number of classes is {NUM_CLASSES}。")
print(f"CTU13 samples: {N_total}")
print(f"Features: {len(feature_cols)}")
print(f"Train/Test: {len(train_idx)}/{len(test_idx)}")
print(f"Label counts: {dict(zip(*np.unique(labels, return_counts=True)))}")
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
print("\n=== Evaluation on Random Hold-out (30%) ===")
print(f"Expected hold-out samples: {len(test_idx)}")
print(f"Actually evaluated samples: {len(y_true_test)}")
if len(y_true_test) == 0:
    print("Test set is empty: check split or window settings.")
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
        print("Error plotting confusion matrix:", e)
    try:
        if NUM_CLASSES > 2:
            if y_prob_test_all.shape[0] == 0:
                print("[ROC] 没有收集到test概率，skipped绘图。")
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
                    print("[ROC] All classes lack positive or negative samples in the test set; multiclass ROC cannot be plotted.")
        else:
            print("[ROC] Current task is binary classification; multiclass ROC skipped.")
    except Exception as e:
        print("ROC calculation/plottingfailed：", e)
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
        print("t-SNE: Too few test hidden vectors; visualization skipped.")
except Exception as e:
    print("t-SNE Visualization failed:", e)



