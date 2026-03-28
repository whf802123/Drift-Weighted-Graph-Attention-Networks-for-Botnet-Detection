import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import deque, defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==================== 0. 配置 ====================
CSV_PATH         = r'C:\Users\whf80\Desktop\DW-GAT\ICASSP\CTU13.csv'
WINDOW_SIZE      = 1000    # sliding window size
BATCH_SIZE       = 100     # batch size per update
GAT_EPOCHS_FIRST = 10      # training epochs for the first window
GAT_EPOCHS_INC   = 1       # incremental fine-tuning epochs

ATTN_HEADS       = 4
HIDDEN_CHANNELS  = 8

TEST_RATIO       = 0.30
LR               = 5e-3
WEIGHT_DECAY     = 0.0
SEED             = None

USE_REPLAY       = True

EXPORT_EMB       = True
EMB_CSV_PATH     = 'embeddings.csv'

ENTROPY_LAMBDA   = 1e-3
EDGE_EW_EMA      = 0.10
ENTROPY_RENORM   = False

USE_DRIFT_CONTROLLER = True

DRIFT_ALPHA        = 0.2
DRIFT_W_CLIP_MIN   = 0.90
DRIFT_W_CLIP_MAX   = 1.30

TAU0 = 0.60
TAU_ALPHA = 0.05
TAU_MIN, TAU_MAX = 0.40, 0.90

REPLAY0 = 0.10
REPLAY_ALPHA = 0.15
REPLAY_MIN, REPLAY_MAX = 0.02, 0.40

H0 = 0.15
H_ALPHA = 0.2
H_MIN, H_MAX = 0.05, 0.85

JS_BETA = 0.8
JS_COMPRESS_K = 3.0

mean_js_prev = 0.0
mean_js_pos_prev = 0.0

PRIORITY_UPDATE      = False
PRIORITY_TOP_RATIO   = 0.30
PRIORITY_ALPHA       = 0.5
PRIORITY_MIN_SAMPLES = 8

LOG_SIGNALS = True
log_mean_js, log_mean_js_pos = [], []
log_w, log_tau, log_r, log_edges, log_ent = [], [], [], [], []

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

df = pd.read_csv(CSV_PATH)

# df = df.sample(frac=0.05, random_state=SEED).reset_index(drop=True) # Part

feature_cols = [c for c in df.columns if c not in ('num', 'Label')]
labels = df['Label'].astype(int).values
N_total = len(df)

all_idx = np.arange(N_total)
train_idx, test_idx = train_test_split(
    all_idx, test_size=TEST_RATIO, stratify=labels, shuffle=True
)

features_raw = df[feature_cols].astype(np.float32).values
scaler = StandardScaler()
features = np.empty_like(features_raw, dtype=np.float32)
features[train_idx] = scaler.fit_transform(features_raw[train_idx]).astype(np.float32)
features[test_idx]  = scaler.transform(features_raw[test_idx]).astype(np.float32)

N = len(features)
is_train = np.zeros(N, dtype=bool); is_train[train_idx] = True
is_test  = ~is_train

class WeightedGATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, num_classes=2, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat_res = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.res_ln  = nn.LayerNorm(hidden_channels * heads)
        self.dropout_p = dropout
        self.gcn2 = GCNConv(hidden_channels * heads, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.cached_att = None  # (edge_index_used, att_heads)

    def forward(self, x, edge_index, edge_weight):
        x1, (ei_used, att_heads) = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        self.cached_att = (ei_used, att_heads)

        res = x1
        x_res = self.gat_res(x1, edge_index)
        x_res = F.elu(x_res)
        x1 = self.res_ln(x_res + res)
        x1 = F.dropout(x1, p=self.dropout_p, training=self.training)

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

prev_incoming_att = {}   # dict: node_global_id -> dict(neigh_global_id -> prob)
prev_prob_dict     = {}  # dict: node_global_id -> p_{t-1}

mean_js_prev = 0.0
mean_js_pos_prev = 0.0

def build_all_edges_and_weights():
    if not edge_weight_dict:
        return None, None
    all_edges = list(edge_weight_dict.keys())
    src_all, dst_all = zip(*all_edges)
    ei = torch.tensor([src_all, dst_all], dtype=torch.long, device=DEVICE)
    ew = torch.tensor([edge_weight_dict[(u, v)] for (u, v) in all_edges],
                      dtype=torch.float32, device=DEVICE)
    return ei, ew

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
    att_mean_edge = att_heads.mean(dim=1).clamp_min(eps)
    ei_np = ei_used.detach().cpu().numpy()
    src_loc = ei_np[0]; dst_loc = ei_np[1]
    idx_glob = np.array(index_window_list)

    incoming = defaultdict(lambda: defaultdict(float))
    sum_by_dst = defaultdict(float)
    for k in range(len(src_loc)):
        u_g = int(idx_glob[src_loc[k]])
        v_g = int(idx_glob[dst_loc[k]])
        a   = float(att_mean_edge[k].detach().cpu().item())
        incoming[v_g][u_g] += a
        sum_by_dst[v_g]    += a

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
    js = 0.5*(kl_pm + kl_qm)
    return float(js)

def compute_window_drift_stats(curr_incoming, prev_incoming, index_window_list, y_window_list=None):
    gid = np.array(index_window_list)
    js_vals = []
    js_pos  = []
    for i, g in enumerate(gid):
        p = prev_incoming.get(int(g), {})
        q = curr_incoming.get(int(g), {})
        js = js_divergence_dict(p, q)
        js_vals.append(js)
        if y_window_list is not None and int(y_window_list[i]) == 1:
            js_pos.append(js)

    mean_js = float(np.mean(js_vals)) if len(js_vals) else 0.0
    mean_js_pos = float(np.mean(js_pos)) if len(js_pos) else mean_js
    return mean_js, mean_js_pos

def drift_controller(mean_js, mean_js_pos):
    global mean_js_prev, mean_js_pos_prev

    mean_js_s = JS_BETA * mean_js_prev + (1 - JS_BETA) * mean_js
    mean_js_pos_s = JS_BETA * mean_js_pos_prev + (1 - JS_BETA) * mean_js_pos

    js_eff = np.tanh(JS_COMPRESS_K * mean_js_s)
    js_pos_eff = np.tanh(JS_COMPRESS_K * mean_js_pos_s)

    w = 1.0 + DRIFT_ALPHA * js_eff
    w = float(np.clip(w, DRIFT_W_CLIP_MIN, DRIFT_W_CLIP_MAX))

    tau = TAU0 + TAU_ALPHA * js_eff
    tau = float(np.clip(tau, TAU_MIN, TAU_MAX))

    r = REPLAY0 + REPLAY_ALPHA * js_pos_eff
    r = float(np.clip(r, REPLAY_MIN, REPLAY_MAX))

    H_star = H0 + H_ALPHA * js_eff
    H_star = float(np.clip(H_star, H_MIN, H_MAX))

    return w, tau, r, H_star

def compute_attention_drift_vector(curr_incoming, prev_incoming, index_window_list):
    W = len(index_window_list)
    drift = np.zeros(W, dtype=np.float32)
    gid = np.array(index_window_list)
    for i in range(W):
        g = int(gid[i])
        p_dict = prev_incoming.get(g, {})
        q_dict = curr_incoming.get(g, {})
        drift[i] = js_divergence_dict(p_dict, q_dict)  # JS ∈ [0, ln2]
    std = drift.std()
    if std > 1e-9:
        drift = (drift - drift.mean()) / (std + 1e-6)
    else:
        drift = drift * 0.0
    return torch.tensor(drift, dtype=torch.float32, device=DEVICE)

def select_priority_subset(mix_idx, probs, idx_np, drift_vec, prev_prob_dict,
                           top_ratio=PRIORITY_TOP_RATIO, alpha=PRIORITY_ALPHA):
    if mix_idx.numel() == 0:
        return mix_idx
    mix_idx_cpu = mix_idx.detach().cpu().numpy()
    gid = idx_np[mix_idx_cpu]
    p_now = probs[mix_idx].detach().cpu().numpy()
    p_prev = np.array([prev_prob_dict.get(int(g), float(p_now[i])) for i, g in enumerate(gid)], dtype=np.float32)
    delta = np.abs(p_now - p_prev).astype(np.float32)

    if delta.std() > 1e-9:
        delta = (delta - delta.mean()) / (delta.std() + 1e-6)
    else:
        delta = delta * 0.0

    drift_score = drift_vec[mix_idx].detach().cpu().numpy()
    score = alpha*drift_score + (1.0 - alpha)*delta
    k = max(PRIORITY_MIN_SAMPLES, int(np.ceil(top_ratio * len(score))))
    k = min(k, len(score))
    topk_idx = np.argsort(-score)[:k]
    return mix_idx[topk_idx]

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

    if USE_DRIFT_CONTROLLER:
        _w_tmp, tau_t, r_t, H_star = drift_controller(mean_js_prev, mean_js_pos_prev)
    else:
        tau_t, r_t, H_star = TAU0, REPLAY0, H0

    x_np = np.array(features_window, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    corr = np.corrcoef(x_np)
    adj = (np.abs(corr) >= tau_t)
    np.fill_diagonal(adj, False)
    rows, cols = np.where(adj)


    if model is None:
        src = np.concatenate([rows, cols]); dst = np.concatenate([cols, rows])
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

        for ep in range(GAT_EPOCHS_FIRST):
            model.train()
            ei, ew = build_all_edges_and_weights()
            if ei is None:
                continue

            logits, _hidden = model(x_tensor, ei, ew)
            y_win  = torch.tensor(list(labels_window), dtype=torch.long, device=DEVICE)
            idx_np = np.array(index_window)

            W = logits.size(0)
            train_mask = torch.tensor(is_train[idx_np], dtype=torch.bool, device=DEVICE)
            take = min(BATCH_SIZE, W)
            new_mask = torch.zeros(W, dtype=torch.bool, device=DEVICE); new_mask[-take:] = True
            final_mask = train_mask & new_mask

            if not final_mask.any():
                if train_mask.any():
                    cand_idx = torch.where(train_mask)[0]
                else:
                    continue
            else:
                cand_idx = torch.where(final_mask)[0]
                if USE_REPLAY:
                    old_mask = train_mask & (~new_mask)
                    if old_mask.any():
                        old_idx = torch.where(old_mask)[0]
                        k = max(1, int(r_t * old_idx.numel()))
                        perm = torch.randperm(old_idx.numel(), device=DEVICE)[:k]
                        replay_idx = old_idx[perm]
                        cand_idx = torch.cat([cand_idx, replay_idx], dim=0)

            ei_used, att_heads = model.cached_att
            idx_win_list = list(index_window)
            curr_incoming = build_incoming_attention_distribution(ei_used, att_heads, idx_win_list)

            mean_js, mean_js_pos = compute_window_drift_stats(
                curr_incoming, prev_incoming_att, idx_win_list, y_window_list=list(labels_window)
            )

            if USE_DRIFT_CONTROLLER:
                w_update, _tau_unused, r_t, H_star = drift_controller(mean_js, mean_js_pos)
            else:
                w_update, r_t, H_star = 1.0, REPLAY0, H0

            if PRIORITY_UPDATE and cand_idx.numel() > 0:
                drift_vec = compute_attention_drift_vector(curr_incoming, prev_incoming_att, idx_win_list)
                probs_now = torch.softmax(logits, dim=1)[:, 1]
                cand_idx = select_priority_subset(cand_idx, probs_now, idx_np, drift_vec, prev_prob_dict)

            if cand_idx.numel() > 0:
                loss_ce = criterion(logits[cand_idx], y_win[cand_idx])

                mask_nodes = torch.zeros(W, dtype=torch.bool, device=DEVICE)
                mask_nodes[cand_idx] = True
                ent_loss = sparse_entropy_loss_sum_heads(
                    ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS, nodes_mask=mask_nodes
                )
                H_star_t = torch.tensor(H_star, dtype=torch.float32, device=DEVICE)
                ent_gap = (ent_loss - H_star_t) ** 2

                loss = w_update * loss_ce + ENTROPY_LAMBDA * ent_gap

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if LOG_SIGNALS:
                    log_ent.append(float(ent_loss.detach().cpu().item()))

            mean_js_prev = mean_js
            mean_js_pos_prev = mean_js_pos
            if LOG_SIGNALS:
                log_mean_js.append(float(mean_js))
                log_mean_js_pos.append(float(mean_js_pos))
                log_w.append(float(w_update))
                log_r.append(float(r_t))
                log_tau.append(float(tau_t))
                log_edges.append(int(len(rows)))
                if len(log_ent) < len(log_mean_js):
                    log_ent.append(float(ent_loss.detach().cpu().item()) if 'ent_loss' in locals() else 0.0)

        model.eval()
        with torch.no_grad():
            ei, ew = build_all_edges_and_weights()
            _ = model(x_tensor, ei, ew)
            ei_used2, att_heads2 = model.cached_att
            refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used2, att_heads2, ema=EDGE_EW_EMA)

        with torch.no_grad():
            probs_now = torch.softmax(model(x_tensor, ei, ew)[0], dim=1)[:, 1]
        prev_incoming_att = build_incoming_attention_distribution(ei_used2, att_heads2, list(index_window))
        for i_loc, g in enumerate(np.array(index_window)):
            prev_prob_dict[int(g)] = float(probs_now[i_loc].detach().cpu().item())

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

        for ep in range(GAT_EPOCHS_INC):
            model.train()
            ei, ew = build_all_edges_and_weights()
            if ei is None:
                continue

            logits, _hidden = model(x_tensor, ei, ew)
            y_win  = torch.tensor(list(labels_window), dtype=torch.long, device=DEVICE)
            idx_np = np.array(index_window)

            W = logits.size(0)
            train_mask = torch.tensor(is_train[idx_np], dtype=torch.bool, device=DEVICE)
            take = min(BATCH_SIZE, W)
            new_mask = torch.zeros(W, dtype=torch.bool, device=DEVICE); new_mask[-take:] = True
            final_mask = train_mask & new_mask

            if not final_mask.any():
                if train_mask.any():
                    cand_idx = torch.where(train_mask)[0]
                else:
                    continue
            else:
                cand_idx = torch.where(final_mask)[0]
                if USE_REPLAY:
                    old_mask = train_mask & (~new_mask)
                    if old_mask.any():
                        old_idx = torch.where(old_mask)[0]
                        k = max(1, int(r_t * old_idx.numel()))
                        perm = torch.randperm(old_idx.numel(), device=DEVICE)[:k]
                        replay_idx = old_idx[perm]
                        cand_idx = torch.cat([cand_idx, replay_idx], dim=0)

            ei_used, att_heads = model.cached_att
            idx_win_list = list(index_window)
            curr_incoming = build_incoming_attention_distribution(ei_used, att_heads, idx_win_list)

            mean_js, mean_js_pos = compute_window_drift_stats(
                curr_incoming, prev_incoming_att, idx_win_list, y_window_list=list(labels_window)
            )

            if USE_DRIFT_CONTROLLER:
                w_update, _tau_unused, r_t, H_star = drift_controller(mean_js, mean_js_pos)
            else:
                w_update, r_t, H_star = 1.0, REPLAY0, H0


            if PRIORITY_UPDATE and cand_idx.numel() > 0:
                drift_vec = compute_attention_drift_vector(curr_incoming, prev_incoming_att, idx_win_list)
                probs_now = torch.softmax(logits, dim=1)[:, 1]
                cand_idx = select_priority_subset(cand_idx, probs_now, idx_np, drift_vec, prev_prob_dict)

            if cand_idx.numel() > 0:
                loss_ce = criterion(logits[cand_idx], y_win[cand_idx])

                mask_nodes = torch.zeros(W, dtype=torch.bool, device=DEVICE)
                mask_nodes[cand_idx] = True
                ent_loss = sparse_entropy_loss_sum_heads(
                    ei_used, att_heads, num_nodes=x_tensor.size(0), heads=ATTN_HEADS, nodes_mask=mask_nodes
                )
                H_star_t = torch.tensor(H_star, dtype=torch.float32, device=DEVICE)
                ent_gap = (ent_loss - H_star_t) ** 2

                loss = w_update * loss_ce + ENTROPY_LAMBDA * ent_gap

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if LOG_SIGNALS:
                    log_ent.append(float(ent_loss.detach().cpu().item()))

            mean_js_prev = mean_js
            mean_js_pos_prev = mean_js_pos
            if LOG_SIGNALS:
                log_mean_js.append(float(mean_js))
                log_mean_js_pos.append(float(mean_js_pos))
                log_w.append(float(w_update))
                log_r.append(float(r_t))
                log_tau.append(float(tau_t))
                log_edges.append(int(len(rows)))
                if len(log_ent) < len(log_mean_js):
                    log_ent.append(float(ent_loss.detach().cpu().item()) if 'ent_loss' in locals() else 0.0)

        model.eval()
        with torch.no_grad():
            ei, ew = build_all_edges_and_weights()
            _ = model(x_tensor, ei, ew)
            ei_used2, att_heads2 = model.cached_att
            refresh_edge_weights_from_cached_att(edge_weight_dict, ei_used2, att_heads2, ema=EDGE_EW_EMA)

        with torch.no_grad():
            probs_now = torch.softmax(model(x_tensor, ei, ew)[0], dim=1)[:, 1]
        prev_incoming_att = build_incoming_attention_distribution(ei_used2, att_heads2, list(index_window))
        for i_loc, g in enumerate(np.array(index_window)):
            prev_prob_dict[int(g)] = float(probs_now[i_loc].detach().cpu().item())

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

            idx_win_list = list(index_window)
            curr_incoming_inf = build_incoming_attention_distribution(ei_used, att_heads, idx_win_list)
            drift_vec = compute_attention_drift_vector(curr_incoming_inf, prev_incoming_att, idx_win_list)

            mixed13 = torch.cat([hidden8, head4, drift_vec.view(-1, 1)], dim=1)  # [N, 13]
            model.cached_att = None

            if EXPORT_EMB:
                if not first_window_committed:
                    all_embeddings.extend(mixed13.detach().cpu().numpy().tolist())
                    all_labels.extend(list(labels_window))
                    all_indices.extend(list(index_window))
                    first_window_committed = True
                else:
                    take = min(BATCH_SIZE, WINDOW_SIZE)
                    new_slice = slice(WINDOW_SIZE - take, WINDOW_SIZE)
                    all_embeddings.extend(mixed13[new_slice].detach().cpu().numpy().tolist())
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
                hidden_test.extend(mixed13[final_mask].detach().cpu().numpy().tolist())

            for i_loc, g in enumerate(idx_np):
                prev_prob_dict[int(g)] = float(probs[i_loc].detach().cpu().item())

y_true_test = np.array(y_true_test)
y_prob_test = np.array(y_prob_test)
y_pred_test = np.array(y_pred_test)

print("\n=== Evaluation on Random Hold-out (30%) ===")
if len(y_true_test) == 0:
    print("Test set empty")
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
        print("ROC AUC Error")

    try:
        fpr, tpr, _ = roc_curve(y_true_test, y_prob_test)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("ROC Error", e)

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
        print("Confusion Matrix Error", e)

print("\n[DEBUG] Drift logs:")
print("len(log_mean_js) =", len(log_mean_js))
print("len(log_w)       =", len(log_w))
print("len(log_r)       =", len(log_r))
print("len(log_tau)     =", len(log_tau))
print("len(log_edges)   =", len(log_edges))
print("len(log_ent)     =", len(log_ent))


def safe_show_1d(y, label, title):
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        print(f"[WARN] {title}: empty, skip")
        return
    x = np.arange(y.size)
    plt.figure(figsize=(8, 3))
    plt.plot(x, y, linewidth=2, label=label)
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
    plt.plot(x, y1, linewidth=2, label=l1)
    plt.plot(x, y2, linewidth=2, label=l2)
    plt.xlabel("Window index")
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

safe_show_1d(log_mean_js, "mean_js", "Drift signal: mean_js")
safe_show_1d(log_w, "w(t)", "Learning pressure w(t)")
safe_show_1d(log_r, "replay r(t)", "Replay ratio r(t)")

safe_show_1d(log_tau, "tau(t)", "Graph gate tau(t)")
safe_show_1d(log_edges, "|E|", "Graph sparsity |E|")
safe_show_1d(log_ent, "H(t)", "Observed attention entropy on used nodes")

# safe_show_2d(log_tau, log_edges, "tau(t)", "|E|", "Graph gate vs sparsity")

try:
    emb = np.array(hidden_test, dtype=np.float32)
    n = len(emb)
    if n >= 3:
        perp = max(5, min(30, n // 10 if n // 10 >= 5 else n - 1))
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(perp, n - 1))
        emb_2d = tsne.fit_transform(emb)

        plt.figure(figsize=(6, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true_test[:n], s=6)
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.tight_layout()
        plt.show()

except Exception as e:
    print("t-SNE Error", e)

if EXPORT_EMB and len(all_embeddings) > 0:
    try:
        out = pd.DataFrame(np.array(all_embeddings, dtype=np.float32))
        out.columns = [f"dim_{i}" for i in range(out.shape[1])]
        out["Label"] = np.array(all_labels, dtype=np.int32)
        out["Index"] = np.array(all_indices, dtype=np.int64)
        out.to_csv(EMB_CSV_PATH, index=False)
        print(f"\n[OK] Exported embeddings to: {EMB_CSV_PATH}")
    except Exception as e:
        print("Output embeddings Error：", e)
