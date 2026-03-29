import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import deque, defaultdict
import ipaddress

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
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

TEST_RATIO       = 0.30
LR               = 5e-3
WEIGHT_DECAY     = 0.0

EXPORT_EMB       = True
EMB_CSV_PATH     = 'all_embeddings_12d_dwgatv2_iot23.csv'

DF_FRAC     = 0.05   # Part
MAX_WINDOWS = None     #

USE_DRIFT_CONTROLLER = True

DRIFT_ALPHA        = 0.05
DRIFT_W_CLIP_MIN   = 0.90
DRIFT_W_CLIP_MAX   = 1.30

TAU0 = 0.60
TAU_ALPHA = 0.05
TAU_MIN, TAU_MAX = 0.40, 0.90

REPLAY0 = 0.10
REPLAY_ALPHA = 0.02
REPLAY_MIN, REPLAY_MAX = 0.02, 0.40

H0 = 0.15
H_ALPHA = 0.05
H_MIN, H_MAX = 0.05, 0.85

ENTROPY_LAMBDA = 1e-3
EDGE_EW_EMA = 0.10

JS_BETA = 0.9
JS_COMPRESS_K = 1.0

LOG_SIGNALS = True
log_mean_js, log_mean_js_attack = [], []
log_w, log_tau, log_r, log_edges, log_ent = [], [], [], [], []

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

SAMPLE_FRAC = DF_FRAC
MANUAL_MINORITY_LABELS = {'C&C', 'C&C-HeartBeat'}

rare_labels_to_drop = {
    'C&C-FileDownload', 'C&C-Torii', 'FileDownload',
    'C&C-HeartBeat-FileDownload', 'Okiru-Attack', 'C&C-Mirai', 'Attack'
}

def is_unnamed(colname):
    return (colname == '') or pd.isna(colname) or str(colname).lower().startswith('unnamed')

def ip_to_int(val):
    try:
        s = str(val).strip()
        if s in ('-', '', 'nan', 'None', 'NaN'):
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

before = len(df)
df = df[~df['label'].isin(rare_labels_to_drop)].reset_index(drop=True)
after = len(df)
print(f"Dropped rare classes {sorted(list(rare_labels_to_drop))}. Rows: {before} -> {after}")

print(f"Minority (kept intact): {sorted(MANUAL_MINORITY_LABELS)}")
print(f"Sampling {SAMPLE_FRAC*100:.1f}% for ALL other classes ...")

def keep_or_sample(group: pd.DataFrame) -> pd.DataFrame:
    lab = str(group.name)
    if lab in MANUAL_MINORITY_LABELS:
        return group
    return group.sample(frac=SAMPLE_FRAC)

df = (df.groupby('label', group_keys=False)
        .apply(keep_or_sample)
        .reset_index(drop=True))

label_counts = df['label'].astype(str).value_counts()
print("Label counts after sampling:")
print(label_counts.to_string())

too_small = set(label_counts[label_counts < 2].index.tolist())
if too_small:
    print(f"[NOTICE] Classes with <2 samples after sampling are dropped: {sorted(list(too_small))}")
    df = df[~df['label'].isin(too_small)].reset_index(drop=True)

ip_cols       = [c for c in ['id.orig_h','id.resp_h'] if c in df.columns]
num_cols_raw  = [c for c in ['ts','id.orig_p','id.resp_p','duration','orig_bytes','resp_bytes',
                             'missed_bytes','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes']
                 if c in df.columns]
cat_cols      = [c for c in ['proto','service','conn_state','history','local_orig','local_resp']
                 if c in df.columns]

for c in ip_cols:
    df[c] = df[c].map(ip_to_int)

for c in num_cols_raw:
    df[c] = df[c].map(to_float)

for c in cat_cols:
    df[c] = df[c].astype(str).fillna('-').replace({'nan':'-'})

label_text = df['label'].astype(str).values
unique_labels_text = pd.unique(label_text)
label_to_id = {lab: i for i, lab in enumerate(unique_labels_text)}
id_to_label = {v: k for k, v in label_to_id.items()}
labels = np.array([label_to_id[x] for x in label_text], dtype=int)
num_classes = len(np.unique(labels))
print("\nLabel mapping:", label_to_id)

majority_label_name = df['label'].astype(str).value_counts().idxmax()
majority_label_id = label_to_id[majority_label_name]
print(f"[INFO] Treat majority class as benign: {majority_label_name} (id={majority_label_id}). Attack = others.")

num_df = df[num_cols_raw + ip_cols].copy() if (num_cols_raw or ip_cols) else pd.DataFrame(index=df.index)
cat_df = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False) if cat_cols else pd.DataFrame(index=df.index)
feat_df = pd.concat([num_df, cat_df], axis=1)

N_total = len(df)
all_idx = np.arange(N_total)
train_idx, test_idx = train_test_split(
    all_idx, test_size=TEST_RATIO, stratify=labels, shuffle=True
)

medians = feat_df.iloc[train_idx].median(numeric_only=True)
feat_df = feat_df.fillna(medians)

scaler = StandardScaler(with_mean=True, with_std=True)
features = np.empty_like(feat_df.values, dtype=np.float64)
features[train_idx] = scaler.fit_transform(feat_df.iloc[train_idx].values.astype(float))
features[test_idx]  = scaler.transform(feat_df.iloc[test_idx].values.astype(float))

N = len(features)
is_train = np.zeros(N, dtype=bool); is_train[train_idx] = True
is_test  = ~is_train

class WeightedGATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, num_classes):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gcn2 = GCNConv(hidden_channels * heads, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.cached_att = None

    def forward(self, x, edge_index, edge_weight):
        x1, (ei_used, att_heads) = self.gat1(x, edge_index, return_attention_weights=True)
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

# embeddings & eval collection
all_embeddings, all_labels, all_indices = [], [], []
first_window_committed = False

y_true_test, y_pred_test = [], []
hidden_test = []
y_prob_test_all = []

global_idx = 0

# ===== Drift v2 caches =====
prev_incoming_att = {}   # node_global -> {neigh_global: prob}
mean_js_prev = 0.0
mean_js_attack_prev = 0.0

def build_all_edges_and_weights():
    if not edge_weight_dict:
        return None, None
    all_edges = list(edge_weight_dict.keys())
    src_all, dst_all = zip(*all_edges)
    ei = torch.tensor([src_all, dst_all], dtype=torch.long, device=DEVICE)
    ew = torch.tensor([edge_weight_dict[(u, v)] for (u, v) in all_edges],
                      dtype=torch.float32, device=DEVICE)
    return ei, ew

def refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used, att_heads, ema=EDGE_EW_EMA):
    att_mean_edge = att_heads.mean(dim=1).detach().cpu().numpy()  # [E]
    for k, (u, v) in enumerate(ei_used.t().tolist()):
        old = edge_weight_dict.get((u, v), 1.0)
        edge_weight_dict[(u, v)] = float((1.0 - ema) * old + ema * att_mean_edge[k])

def attention_heads_node_mean_from_cached_incoming(ei_used, att_heads, num_nodes, heads):
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
    in_mean = (in_mean - in_mean.mean(dim=0, keepdim=True)) / (in_mean.std(dim=0, keepdim=True) + 1e-6)
    return in_mean  # [N, heads]

def sparse_entropy_loss_sum_heads(ei_used, att_heads, num_nodes, heads, nodes_mask=None, eps=1e-12):
    dst = ei_used[1]
    E = dst.numel()
    device = att_heads.device
    deg_in = torch.zeros(num_nodes, device=device).index_add_(0, dst, torch.ones(E, device=device))
    logZ = torch.log(deg_in.clamp_min(1.0))
    neg_p_logp_sum_heads = torch.zeros(num_nodes, device=device)
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

def build_incoming_attention_distribution(ei_used, att_heads, index_window_list, eps=1e-12):
    """
    dict[v_global][u_global] = prob
    """
    att_mean_edge = att_heads.mean(dim=1).clamp_min(eps)
    ei_np = ei_used.detach().cpu().numpy()
    src_loc = ei_np[0]; dst_loc = ei_np[1]
    idx_glob = np.array(index_window_list)

    incoming = defaultdict(lambda: defaultdict(float))
    sum_by_dst = defaultdict(float)
    for k in range(len(src_loc)):
        u_g = int(idx_glob[src_loc[k]])
        v_g = int(idx_glob[dst_loc[k]])
        a = float(att_mean_edge[k].detach().cpu().item())
        incoming[v_g][u_g] += a
        sum_by_dst[v_g] += a

    for v_g, neighs in incoming.items():
        Z = max(sum_by_dst[v_g], eps)
        for u_g in list(neighs.keys()):
            neighs[u_g] = neighs[u_g] / Z
    return dict(incoming)

def js_divergence_dict(p_dict, q_dict, eps=1e-12):
    keys = set(p_dict.keys()) | set(q_dict.keys())
    if not keys:
        return 0.0
    p = np.array([p_dict.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    q = np.array([q_dict.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5*(p+q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5*(kl_pm + kl_qm))

def compute_window_drift_stats(curr_incoming, prev_incoming, index_window_list, y_window_list):
    gid = np.array(index_window_list)
    js_vals = []
    js_attack = []
    for i, g in enumerate(gid):
        p = prev_incoming.get(int(g), {})
        q = curr_incoming.get(int(g), {})
        js = js_divergence_dict(p, q)
        js_vals.append(js)

        if int(y_window_list[i]) != int(majority_label_id):
            js_attack.append(js)

    mean_js = float(np.mean(js_vals)) if len(js_vals) else 0.0
    mean_js_attack = float(np.mean(js_attack)) if len(js_attack) else mean_js
    return mean_js, mean_js_attack

def drift_controller(mean_js, mean_js_attack):
    global mean_js_prev, mean_js_attack_prev

    mean_js_s = JS_BETA * mean_js_prev + (1 - JS_BETA) * mean_js
    mean_js_attack_s = JS_BETA * mean_js_attack_prev + (1 - JS_BETA) * mean_js_attack

    js_eff = np.tanh(JS_COMPRESS_K * mean_js_s)
    js_attack_eff = np.tanh(JS_COMPRESS_K * mean_js_attack_s)

    w = 1.0 + DRIFT_ALPHA * js_eff
    w = float(np.clip(w, DRIFT_W_CLIP_MIN, DRIFT_W_CLIP_MAX))

    tau = TAU0 + TAU_ALPHA * js_eff
    tau = float(np.clip(tau, TAU_MIN, TAU_MAX))

    r = REPLAY0 + REPLAY_ALPHA * js_attack_eff
    r = float(np.clip(r, REPLAY_MIN, REPLAY_MAX))

    H_star = H0 + H_ALPHA * js_eff
    H_star = float(np.clip(H_star, H_MIN, H_MAX))

    return w, tau, r, H_star

def compute_loss_with_replay_and_weight(logits, y_win, idx_np, r_t, w_update):
    W = logits.shape[0]
    train_mask = torch.tensor(is_train[idx_np], dtype=torch.bool, device=DEVICE)

    take = min(BATCH_SIZE, W)
    new_mask = torch.zeros(W, dtype=torch.bool, device=DEVICE)
    new_mask[-take:] = True

    final_mask = train_mask & new_mask

    if not final_mask.any():
        if train_mask.any():
            ce = criterion(logits[train_mask], y_win[train_mask])
            return w_update * ce, train_mask
        return None, None

    idx_used = torch.where(final_mask)[0]

    # replay
    old_mask = train_mask & (~new_mask)
    if old_mask.any():
        old_idx = torch.where(old_mask)[0]
        k = max(1, int(r_t * old_idx.numel()))
        k = min(k, old_idx.numel())
        perm = torch.randperm(old_idx.numel(), device=DEVICE)[:k]
        replay_idx = old_idx[perm]
        idx_used = torch.cat([idx_used, replay_idx], dim=0)

    ce = criterion(logits[idx_used], y_win[idx_used])
    return w_update * ce, idx_used

window_counter = 0

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

    window_counter += 1
    if (MAX_WINDOWS is not None) and (window_counter > int(MAX_WINDOWS)):
        print(f"[INFO] Reached MAX_WINDOWS={MAX_WINDOWS}, stop early.")
        break

    if USE_DRIFT_CONTROLLER:
        _w_tmp, tau_t, r_t, H_star = drift_controller(mean_js_prev, mean_js_attack_prev)
    else:
        tau_t, r_t, H_star = TAU0, REPLAY0, H0

    x_np = np.array(features_window, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    corr = np.corrcoef(x_np)
    adj = (np.abs(corr) >= tau_t)
    np.fill_diagonal(adj, False)
    rows, cols = np.where(adj)

    if LOG_SIGNALS:
        log_tau.append(float(tau_t))
        log_edges.append(int(len(rows)))

    if model is None:
        # init edges with 1.0 (bidirectional)
        src = np.concatenate([rows, cols])
        dst = np.concatenate([cols, rows])
        edge_weight_dict.clear()
        for u, v in zip(src.tolist(), dst.tolist()):
            edge_weight_dict[(u, v)] = 1.0

        model = WeightedGATClassifier(
            in_channels=x_tensor.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            heads=ATTN_HEADS,
            num_classes=num_classes
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        # first window training
        for _ in range(GAT_EPOCHS_FIRST):
            model.train()
            ei, ew = build_all_edges_and_weights()
            if ei is None:
                continue

            logits, _hidden = model(x_tensor, ei, ew)
            y_win  = torch.tensor(list(labels_window), dtype=torch.long, device=DEVICE)
            idx_np = np.array(index_window)

            # attention drift stats (one pass)
            ei_used, att_heads = model.cached_att
            curr_incoming = build_incoming_attention_distribution(ei_used, att_heads, list(index_window))
            mean_js, mean_js_attack = compute_window_drift_stats(
                curr_incoming, prev_incoming_att, list(index_window), list(labels_window)
            )

            if USE_DRIFT_CONTROLLER:
                w_update, _tau_unused, r_t, H_star = drift_controller(mean_js, mean_js_attack)
            else:
                w_update, r_t, H_star = 1.0, REPLAY0, H0

            # CE + replay + weight
            ce_w, used_idx = compute_loss_with_replay_and_weight(logits, y_win, idx_np, r_t, w_update)
            if ce_w is None:
                continue

            # entropy gap regularization: |H - H*|^2 on used nodes
            nodes_mask = torch.zeros(WINDOW_SIZE, dtype=torch.bool, device=DEVICE)
            nodes_mask[used_idx] = True
            ent = sparse_entropy_loss_sum_heads(ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS, nodes_mask=nodes_mask)
            ent_gap = (ent - torch.tensor(H_star, dtype=torch.float32, device=DEVICE)) ** 2

            loss = ce_w + ENTROPY_LAMBDA * ent_gap

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logs
            mean_js_prev = mean_js
            mean_js_attack_prev = mean_js_attack

            if LOG_SIGNALS:
                log_mean_js.append(float(mean_js))
                log_mean_js_attack.append(float(mean_js_attack))
                log_w.append(float(w_update))
                log_r.append(float(r_t))
                log_ent.append(float(ent.detach().cpu().item()))

        # refresh edge weights (EMA)
        model.eval()
        with torch.no_grad():
            ei, ew = build_all_edges_and_weights()
            _ = model(x_tensor, ei, ew)
            ei_used2, att_heads2 = model.cached_att
            refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used2, att_heads2, ema=EDGE_EW_EMA)

        # update prev attention baseline (after training)
        prev_incoming_att = build_incoming_attention_distribution(ei_used2, att_heads2, list(index_window))

    else:
        # incremental: only add new edges
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

            # drift stats
            ei_used, att_heads = model.cached_att
            curr_incoming = build_incoming_attention_distribution(ei_used, att_heads, list(index_window))
            mean_js, mean_js_attack = compute_window_drift_stats(
                curr_incoming, prev_incoming_att, list(index_window), list(labels_window)
            )

            if USE_DRIFT_CONTROLLER:
                w_update, _tau_unused, r_t, H_star = drift_controller(mean_js, mean_js_attack)
            else:
                w_update, r_t, H_star = 1.0, REPLAY0, H0

            ce_w, used_idx = compute_loss_with_replay_and_weight(logits, y_win, idx_np, r_t, w_update)
            if ce_w is None:
                continue

            nodes_mask = torch.zeros(WINDOW_SIZE, dtype=torch.bool, device=DEVICE)
            nodes_mask[used_idx] = True
            ent = sparse_entropy_loss_sum_heads(ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS, nodes_mask=nodes_mask)
            ent_gap = (ent - torch.tensor(H_star, dtype=torch.float32, device=DEVICE)) ** 2

            loss = ce_w + ENTROPY_LAMBDA * ent_gap

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_js_prev = mean_js
            mean_js_attack_prev = mean_js_attack

            if LOG_SIGNALS:
                log_mean_js.append(float(mean_js))
                log_mean_js_attack.append(float(mean_js_attack))
                log_w.append(float(w_update))
                log_r.append(float(r_t))
                log_ent.append(float(ent.detach().cpu().item()))

        # refresh edge weights
        model.eval()
        with torch.no_grad():
            ei, ew = build_all_edges_and_weights()
            _ = model(x_tensor, ei, ew)
            ei_used2, att_heads2 = model.cached_att
            refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used2, att_heads2, ema=EDGE_EW_EMA)

        prev_incoming_att = build_incoming_attention_distribution(ei_used2, att_heads2, list(index_window))

    # ===== inference & embedding export =====
    model.eval()
    with torch.no_grad():
        ei, ew = build_all_edges_and_weights()
        if ei is None:
            continue

        logits, hidden8 = model(x_tensor, ei, ew)
        probs_all = torch.softmax(logits, dim=1).detach().cpu().numpy()  # [W, K]

        ei_used, att_heads = model.cached_att
        head4 = attention_heads_node_mean_from_cached_incoming(
            ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS
        )
        mixed12 = torch.cat([hidden8, head4], dim=1)  # 12-d
        model.cached_att = None

        if EXPORT_EMB:
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

        # eval slice: only new entering test nodes
        take = min(BATCH_SIZE, WINDOW_SIZE)
        new_mask = np.zeros(WINDOW_SIZE, dtype=bool)
        new_mask[-take:] = True
        idx_np = np.array(index_window)
        test_mask = is_test[idx_np]
        final_mask = (new_mask & test_mask)

        if final_mask.any():
            y_true_test.extend(np.array(list(labels_window))[final_mask].tolist())
            y_pred_test.extend(torch.argmax(logits[final_mask], dim=1).cpu().numpy().tolist())
            hidden_test.extend(mixed12[final_mask].detach().cpu().numpy().tolist())
            y_prob_test_all.append(probs_all[final_mask])

# Evaluation 
y_true_test = np.array(y_true_test)
y_pred_test = np.array(y_pred_test)
if len(y_prob_test_all) > 0:
    y_prob_test_all = np.vstack(y_prob_test_all)  # [N_test_slice, K]
else:
    y_prob_test_all = np.empty((0, num_classes))

print("\n=== Evaluation on Random Hold-out (30%) ===")
if len(y_true_test) == 0:
    print("Test set empty")
else:
    unique_ids = np.unique(labels)
    target_names = [id_to_label[i] for i in unique_ids]

    report = classification_report(
        y_true_test, y_pred_test, output_dict=True, digits=4,
        labels=unique_ids, target_names=target_names
    )
    print("Classification Report (Test Slice):")
    for lab in target_names:
        m = report[lab]
        print(f"{lab:<28} precision: {m['precision']*100:.2f}%  "
              f"recall: {m['recall']*100:.2f}%  f1-score: {m['f1-score']*100:.2f}%")
    print(f"{'accuracy':<28}: {report['accuracy']*100:.2f}%")
    for k in ['macro avg','weighted avg']:
        m = report[k]
        print(f"{k:<28} precision: {m['precision']*100:.2f}%  "
              f"recall: {m['recall']*100:.2f}%  f1-score: {m['f1-score']*100:.2f}%")

    # Confusion Matrix
    try:
        labels_order = unique_ids.tolist()
        display_names = [id_to_label[i] for i in labels_order]
        cm = confusion_matrix(y_true_test, y_pred_test, labels=labels_order)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
        plt.figure(figsize=(6.5, 6.0))
        disp.plot(values_format='d', cmap='Blues', colorbar=False)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Confusion Matrix Error", e)

    # Multi-class ROC (OvR)
    try:
        if num_classes > 2 and y_prob_test_all.shape[0] > 0:
            classes_sorted = np.unique(labels).tolist()
            y_true_bin = label_binarize(y_true_test, classes=classes_sorted)

            plt.figure(figsize=(7.2, 5.4))
            plotted = False
            for cls_id in classes_sorted:
                col = classes_sorted.index(cls_id)
                y_true_c  = y_true_bin[:, col]
                y_score_c = y_prob_test_all[:, cls_id]
                if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
                    continue
                fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{id_to_label[cls_id]} (AUC={roc_auc:.3f})")
                plotted = True

            if plotted:
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(fontsize=8, loc="lower right")
                plt.tight_layout()
                plt.show()
            else:
                print("[ROC] No valid class (missing positive/negative samples).")
        else:
            print("[ROC] Binary classification or empty probability output. Skipped multi-class ROC.")
    except Exception as e:
        print("ROC computation/plot error:", e)


print("\n[DEBUG] Drift logs:")
print("len(log_mean_js)       =", len(log_mean_js))
print("len(log_mean_js_attack)=", len(log_mean_js_attack))
print("len(log_w)             =", len(log_w))
print("len(log_r)             =", len(log_r))
print("len(log_tau)           =", len(log_tau))
print("len(log_edges)         =", len(log_edges))
print("len(log_ent)           =", len(log_ent))

def safe_show_1d(y, label, title):
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        print(f"[WARN] {title}: empty, skip")
        return
    x = np.arange(y.size)
    plt.figure(figsize=(8, 3))
    plt.plot(x, y, label=label)
    plt.xlabel("Window index")
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def safe_show_2d(y1, y2, l1, l2, title):
    m = min(len(y1), len(y2))
    if m == 0:
        print(f"[WARN] {title}: empty, skip")
        return
    y1 = np.asarray(y1[:m], dtype=np.float64)
    y2 = np.asarray(y2[:m], dtype=np.float64)
    x = np.arange(m)
    plt.figure(figsize=(8, 3))
    plt.plot(x, y1, label=l1)
    plt.plot(x, y2, label=l2)
    plt.xlabel("Window index")
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

if LOG_SIGNALS:
    safe_show_1d(log_mean_js, "mean_js", "Drift signal: mean_js (all nodes)")
    safe_show_1d(log_mean_js_attack, "mean_js_attack", "Drift signal: mean_js_attack (non-majority classes)")
    safe_show_1d(log_w, "w(t)", "Controller: learning pressure w(t)")
    safe_show_1d(log_r, "r(t)", "Controller: replay ratio r(t)")
    # safe_show_2d(log_tau, log_edges, "tau(t)", "|E|", "Controller: tau(t) and graph sparsity |E|")

    safe_show_1d(log_tau, "tau(t)", "Controller: adaptive correlation threshold tau(t)")
    safe_show_1d(log_edges, "|E|", "Graph sparsity: number of edges |E|")
    safe_show_1d(log_ent, "H(t)", "Observed attention entropy on used nodes")

try:
    if len(hidden_test) > 20:
        emb_2d = TSNE(n_components=2).fit_transform(np.array(hidden_test))
        plt.figure(figsize=(6,6))
        plt.scatter(emb_2d[:,0], emb_2d[:,1], c=y_true_test[:len(emb_2d)], s=6)
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.tight_layout()
        plt.show()
    else:
        print("t-SNE: Not enough test embeddings, skip visualization.")

except Exception as e:
    print("t-SNE visualization failed:", e)

if EXPORT_EMB and len(all_embeddings) > 0:
    try:
        out = pd.DataFrame(np.array(all_embeddings, dtype=np.float32))
        out.columns = [f"dim_{i}" for i in range(out.shape[1])]
        out["Label"] = np.array(all_labels, dtype=np.int32)
        out["Index"] = np.array(all_indices, dtype=np.int64)
        out.to_csv(EMB_CSV_PATH, index=False)

        print(f"\n Exported embeddings to: {EMB_CSV_PATH}")

    except Exception as e:
        print("Embedding export failed:", e)
