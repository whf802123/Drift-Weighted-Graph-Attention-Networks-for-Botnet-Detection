import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CSV_PATH         = r'C:\Users\whf80\Desktop\ICASSP\CTU13.csv'
WINDOW_SIZE      = 1000  # sliding window size
BATCH_SIZE       = 100   # batch size per update
GAT_EPOCHS_FIRST = 10    # training epochs for the first window
GAT_EPOCHS_INC   = 2     # incremental fine-tuning epochs
ATTN_HEADS       = 4
HIDDEN_CHANNELS  = 8     # GCN Output dim
CORR_THRESHOLD   = 0.6
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15
LR               = 5e-3
WEIGHT_DECAY     = 0.0
SEED             = 42

USE_REPLAY       = True
REPLAY_RATIO     = 0.10

EXPORT_EMB       = True
EMB_CSV_PATH     = 'all_embeddings_12d_cached_att.csv'

ENTROPY_LAMBDA   = 1e-3

EDGE_EW_EMA      = 0.10

ENTROPY_RENORM   = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

df = pd.read_csv(CSV_PATH)
feature_cols = [c for c in df.columns if c not in ('num', 'Label')]
labels = df['Label'].astype(int).values
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

features_raw = df[feature_cols].astype(np.float32).values
scaler = StandardScaler()
features = np.empty_like(features_raw, dtype=np.float32)
features[train_idx] = scaler.fit_transform(features_raw[train_idx]).astype(np.float32)
features[test_idx]  = scaler.transform(features_raw[test_idx]).astype(np.float32)
features[val_idx]  = scaler.transform(features_raw[val_idx]).astype(np.float32)

N = len(features)
is_train = np.zeros(N, dtype=bool); is_train[train_idx] = True
is_test = np.zeros(N, dtype=bool)
is_test[test_idx] = True
is_val = np.zeros(N, dtype=bool)
is_val[val_idx] = True
print(
    f"Data split: train={len(train_idx)} ({TRAIN_RATIO:.0%}), "
    f"validation={len(val_idx)} ({VALIDATION_RATIO:.0%}), "
    f"test={len(test_idx)} ({TEST_RATIO:.0%})"
)

class WeightedGATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, num_classes=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gcn2 = GCNConv(hidden_channels * heads, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.cached_att = None  # (edge_index_used, att_heads)

    def forward(self, x, edge_index, edge_weight):
        x1, (ei_used, att_heads) = self.gat1(
            x, edge_index, return_attention_weights=True
        )
        x1 = F.elu(x1)
        self.cached_att = (ei_used, att_heads)
        x2 = self.gcn2(x1, edge_index, edge_weight=edge_weight)
        logits = self.classifier(x2)
        return logits, x2

features_window = deque(maxlen=WINDOW_SIZE)
labels_window   = deque(maxlen=WINDOW_SIZE)
index_window    = deque(maxlen=WINDOW_SIZE)

edge_weight_dict = {}  # (u,v) -> weight

model = None
optimizer = None
criterion = None

pos_ratio = labels.mean() + 1e-8
w_neg = 1.0 / (1.0 - pos_ratio)
w_pos = 1.0 / pos_ratio

all_embeddings = []
all_labels     = []
all_indices    = []
first_window_committed = False

y_true_test, y_prob_test, y_pred_test = [], [], []
hidden_test = []

global_idx = 0

def build_all_edges_and_weights():
    if not edge_weight_dict:
        return None, None
    all_edges = list(edge_weight_dict.keys())
    src_all, dst_all = zip(*all_edges)
    ei = torch.tensor([src_all, dst_all], dtype=torch.long, device=DEVICE)
    ew = torch.tensor([edge_weight_dict[(u, v)] for (u, v) in all_edges],
                      dtype=torch.float32, device=DEVICE)
    return ei, ew

def compute_loss_on_new_nodes(logits, y_win, idx_np):
    W = logits.shape[0]
    train_mask = torch.tensor(is_train[idx_np], dtype=torch.bool, device=DEVICE)

    take = min(BATCH_SIZE, W)
    new_mask = torch.zeros(W, dtype=torch.bool, device=DEVICE)
    new_mask[-take:] = True

    final_mask = train_mask & new_mask

    if not final_mask.any():
        if train_mask.any():
            return criterion(logits[train_mask], y_win[train_mask]), train_mask
        else:
            return None, None

    if USE_REPLAY:
        old_mask = train_mask & (~new_mask)
        if old_mask.any():
            old_idx = torch.where(old_mask)[0]
            k = max(1, int(REPLAY_RATIO * old_idx.numel()))
            perm = torch.randperm(old_idx.numel(), device=DEVICE)[:k]
            replay_idx = old_idx[perm]
            mix_idx = torch.cat([torch.where(final_mask)[0], replay_idx], dim=0)
            return criterion(logits[mix_idx], y_win[mix_idx]), mix_idx

    return criterion(logits[final_mask], y_win[final_mask]), final_mask

def attention_heads_node_mean_from_cached_incoming(ei_used, att_heads, num_nodes, heads):
    dst = ei_used[1]  # [E]
    E = dst.numel()
    device = att_heads.device

    in_sum = torch.zeros(num_nodes, heads, device=device)  # [N, H]
    in_cnt = torch.zeros(num_nodes, 1, device=device)      # [N, 1]

    for h in range(heads):
        in_sum[:, h].index_add_(0, dst, att_heads[:, h])

    ones_e = torch.ones(E, device=device)
    in_cnt.index_add_(0, dst, ones_e.unsqueeze(-1))

    in_mean = in_sum / in_cnt.clamp(min=1.0)
    in_mean = (in_mean - in_mean.mean(dim=0, keepdim=True)) / (in_mean.std(dim=0, keepdim=True) + 1e-6)
    return in_mean  # [N, heads]

def refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used, att_heads, ema=EDGE_EW_EMA):
    att_mean_edge = att_heads.mean(dim=1).detach().cpu().numpy()  # [E]
    for k, (u, v) in enumerate(ei_used.t().tolist()):
        old = edge_weight_dict.get((u, v), 1.0)
        if ema is None or ema <= 0.0:
            edge_weight_dict[(u, v)] = float(att_mean_edge[k])
        else:
            edge_weight_dict[(u, v)] = float((1.0 - ema) * old + ema * att_mean_edge[k])

def sparse_entropy_loss_sum_heads(ei_used, att_heads, num_nodes, heads, nodes_mask=None, eps=1e-12,
                                  renorm=ENTROPY_RENORM):
    dst = ei_used[1]  # [E]
    E = dst.numel()
    device = att_heads.device

    deg_in = torch.zeros(num_nodes, device=device).index_add_(0, dst, torch.ones(E, device=device))
    logZ = torch.log(deg_in.clamp_min(1.0))  # log(deg_i)

    neg_p_logp_sum_heads = torch.zeros(num_nodes, device=device)  # [N]
    for h in range(heads):
        p = att_heads[:, h].clamp_min(eps)
        tmp = torch.zeros(num_nodes, device=device)
        tmp.index_add_(0, dst, -(p * p.log()))
        neg_p_logp_sum_heads += tmp

    denom = (heads * logZ).clamp_min(1e-6)
    norm_entropy = neg_p_logp_sum_heads / denom
    norm_entropy = torch.where(logZ > 1e-6, norm_entropy, torch.zeros_like(norm_entropy))

    if nodes_mask is not None:
        norm_entropy = norm_entropy[nodes_mask]

    return norm_entropy.mean()

for start in tqdm(range(0, N, BATCH_SIZE), desc='Processing batches'):
    batch_feats  = features[start:start + BATCH_SIZE]
    batch_labels = labels[start:start + BATCH_SIZE]
    bsz = len(batch_feats)

    for i in range(bsz):
        features_window.append(batch_feats[i])
        labels_window.append(int(batch_labels[i]))
        index_window.append(global_idx)
        global_idx += 1

    if len(features_window) < WINDOW_SIZE:
        continue

    x_np = np.array(features_window, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        x_norm = x_tensor / (x_tensor.norm(dim=1, keepdim=True) + 1e-9)  # [W, F]
        sim = torch.matmul(x_norm, x_norm.t())                            # [W, W]
        adj = sim.abs() >= CORR_THRESHOLD
        adj.fill_diagonal_(False)
        rows_t, cols_t = adj.nonzero(as_tuple=True)
        rows = rows_t.detach().cpu().numpy()
        cols = cols_t.detach().cpu().numpy()

    if model is None:
        src = np.concatenate([rows, cols])
        dst = np.concatenate([cols, rows])
        edge_weight_dict.clear()
        for u, v in zip(src.tolist(), dst.tolist()):
            edge_weight_dict[(u, v)] = 1.0

        model = WeightedGATClassifier(
            in_channels=x_tensor.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            heads=ATTN_HEADS,
            num_classes=2
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for _ in range(GAT_EPOCHS_FIRST):
            model.train()
            ei, ew = build_all_edges_and_weights()
            if ei is None:
                continue
            logits, _hidden = model(x_tensor, ei, ew)     # 此处已缓存注意力
            y_win  = torch.tensor(list(labels_window), dtype=torch.long, device=DEVICE)
            idx_np = np.array(index_window)
            ce_loss, used_mask = compute_loss_on_new_nodes(logits, y_win, idx_np)
            if ce_loss is None:
                continue

            ei_used, att_heads = model.cached_att
            nodes_mask = None
            if used_mask is not None:
                bool_mask = torch.zeros(WINDOW_SIZE, dtype=torch.bool, device=DEVICE)
                bool_mask[used_mask] = True
                nodes_mask = bool_mask

            ent_loss = sparse_entropy_loss_sum_heads(
                ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS, nodes_mask=nodes_mask
            )

            loss = ce_loss + ENTROPY_LAMBDA * ent_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            ei, ew = build_all_edges_and_weights()
            _ = model(x_tensor, ei, ew)
            ei_used, att_heads = model.cached_att
            refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used, att_heads, ema=EDGE_EW_EMA)

    else:
        new_start = WINDOW_SIZE - BATCH_SIZE
        mask = ((rows >= new_start) | (cols >= new_start))
        rows_inc, cols_inc = rows[mask], cols[mask]
        if rows_inc.size > 0:
            src_inc = np.concatenate([rows_inc, cols_inc])
            dst_inc = np.concatenate([cols_inc, rows_inc])
            for u, v in zip(src_inc.tolist(), dst_inc.tolist()):
                if (u, v) not in edge_weight_dict:
                    edge_weight_dict[(u, v)] = 1.0

        for _ in range(GAT_EPOCHS_INC):
            model.train()
            ei, ew = build_all_edges_and_weights()
            if ei is None:
                continue
            logits, _hidden = model(x_tensor, ei, ew)
            y_win  = torch.tensor(list(labels_window), dtype=torch.long, device=DEVICE)
            idx_np = np.array(index_window)
            ce_loss, used_mask = compute_loss_on_new_nodes(logits, y_win, idx_np)
            if ce_loss is None:
                continue

            ei_used, att_heads = model.cached_att
            nodes_mask = None
            if used_mask is not None:
                bool_mask = torch.zeros(WINDOW_SIZE, dtype=torch.bool, device=DEVICE)
                bool_mask[used_mask] = True
                nodes_mask = bool_mask

            ent_loss = sparse_entropy_loss_sum_heads(
                ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS, nodes_mask=nodes_mask
            )

            loss = ce_loss + ENTROPY_LAMBDA * ent_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            ei, ew = build_all_edges_and_weights()
            _ = model(x_tensor, ei, ew)
            ei_used, att_heads = model.cached_att
            refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used, att_heads, ema=EDGE_EW_EMA)

    model.eval()
    with torch.no_grad():
        ei, ew = build_all_edges_and_weights()
        if ei is not None:
            logits, hidden8 = model(x_tensor, ei, ew)
            probs = torch.softmax(logits, dim=1)[:, 1]

            ei_used, att_heads = model.cached_att
            head4 = attention_heads_node_mean_from_cached_incoming(
                ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS
            )
            mixed12 = torch.cat([hidden8, head4], dim=1)  # [N, 12]
            model.cached_att = None

            if not first_window_committed:
                all_embeddings.extend(mixed12.detach().cpu().numpy().tolist())
                all_labels.extend(list(labels_window))
                all_indices.extend(list(index_window))
                first_window_committed = True
            else:
                take = min(BATCH_SIZE, WINDOW_SIZE)
                new_slice = slice(WINDOW_SIZE - take, WINDOW_SIZE)
                all_embeddings.extend(mixed12[new_slice].detach().cpu().numpy().tolist())
                all_labels.extend(list(labels_window)[new_slice])
                all_indices.extend(list(index_window)[new_slice])

            W = WINDOW_SIZE
            take = min(BATCH_SIZE, W)
            new_mask = np.zeros(W, dtype=bool); new_mask[-take:] = True
            idx_np = np.array(index_window)
            test_mask = is_test[idx_np]
            final_mask = (new_mask & test_mask)

            if final_mask.any():
                y_true_test.extend(np.array(list(labels_window))[final_mask].tolist())
                y_prob_test.extend(probs[final_mask].detach().cpu().numpy().tolist())
                y_pred_test.extend((probs[final_mask] >= 0.5).long().cpu().numpy().tolist())
                hidden_test.extend(mixed12[final_mask].detach().cpu().numpy().tolist())

y_true_test = np.array(y_true_test)
y_prob_test = np.array(y_prob_test)
y_pred_test = np.array(y_pred_test)

print("\n=== Evaluation on Random Test Split (15%) ===")
if len(y_true_test) == 0:
    print("Test set is empty")
else:
    report = classification_report(
        y_true_test, y_pred_test,
        labels=[0,1], target_names=['Normal','Botnet'], output_dict=True, digits=4
    )
    print("Classification Report (Test Slice):")
    for label in ['Normal','Botnet','accuracy','macro avg','weighted avg']:
        if label == 'accuracy':
            print(f"{label:<12}: {report[label]*100:.2f}%")
        else:
            m = report[label]
            print(
                f"{label:<12} precision: {m['precision']*100:.2f}%  "
                f"recall:    {m['recall']*100:.2f}%  "
                f"f1-score:  {m['f1-score']*100:.2f}%"
            )

    try:
        roc_auc = roc_auc_score(y_true_test, y_prob_test)
        print(f"ROC AUC: {roc_auc*100:.2f}%")
    except ValueError:
        print("ROC AUC error")

    try:
        fpr, tpr, _ = roc_curve(y_true_test, y_prob_test)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("ROC AUC error", e)

    try:
        labels_order = [0, 1]
        display_names = ['Normal', 'Botnet']

        cm = confusion_matrix(y_true_test, y_pred_test, labels=labels_order)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
        plt.figure(figsize=(4.5, 4))
        disp.plot(values_format='d', cmap='Blues', colorbar=False)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Confusion Matrix error", e)

try:
    n = len(hidden_test)
    if n > 2:
        perp = max(5, min(30, (n - 1) // 3))
        perp = max(5, min(perp, n - 2))
        emb_2d = TSNE(n_components=2, random_state=SEED, perplexity=perp).fit_transform(np.array(hidden_test))
        plt.figure(figsize=(6,6))
        plt.scatter(emb_2d[:,0], emb_2d[:,1], c=y_true_test[:len(emb_2d)], s=5)
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.tight_layout()
        plt.show()
    else:
        print("t-SNE error")
except Exception as e:
    print("t-SNE error", e)
