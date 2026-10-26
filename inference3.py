# -*- coding: utf-8 -*-
"""
EoR Final Inference - Corrected for Your Data Structure
参数锁定：Vote Weight = 0.80, Model Weight = 0.20
"""
import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import re
import string
import collections
import warnings

warnings.filterwarnings("ignore")

# ==================== 🔒 最终参数 ====================
HIDDEN_DIM = 512
EMB_DIM = 384
W_VOTE = 0.8
W_MODEL = 0.2
ACCURACY_THRESHOLD = 0.8
FEATURE_GROUPS = ["query_embedding", "base_scores", "stats_group"]


# ==================== 📏 评估工具函数 ====================

def normalize_answer(s):
    """标准化答案：去标点、去冠词、小写化"""
    if not isinstance(s, str): return ""

    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text): return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def load_method_weights(weight_file):
    """从文本文件加载方法权重"""
    if not weight_file or not os.path.exists(weight_file):
        print(f"⚠️ Weight file not found: {weight_file}")
        return {}
    
    weights = {}
    try:
        with open(weight_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 格式: nli: 0.79722 bertscore: 0.44752 ...
            # split by space
            parts = content.split()
            for i in range(0, len(parts), 2):
                key = parts[i].rstrip(':')
                val = float(parts[i+1])
                weights[key] = val
        print(f"⚖️ Loaded {len(weights)} method weights from {weight_file}")
        return weights
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return {}


def compute_f1(prediction, truth):
    """计算单对 F1 Score"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

    # 如果任一为空
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0: return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """针对多个标准答案取最高分"""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0


# ==================== 模型定义 ====================
class ContentFeatureExtractor:
    def __init__(self, encoder_model_path, device='cpu'):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
            self.encoder = AutoModel.from_pretrained(encoder_model_path)
            self.encoder.to(device).eval()
        except:
            self.encoder = None

    def _batch_encode(self, texts, encode_batch_size=64):
        """分批编码文本，避免显存溢出"""
        valid_texts = [t if isinstance(t, str) and len(t.strip()) > 0 else "empty" for t in texts]
        if self.encoder is None: 
            return np.zeros((len(texts), EMB_DIM))
        
        all_embs = []
        for i in range(0, len(valid_texts), encode_batch_size):
            batch_texts = valid_texts[i:i + encode_batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                out = self.encoder(**inputs)
                if hasattr(out, 'last_hidden_state'):
                    embs = out.last_hidden_state[:, 0]
                else:
                    embs = out[0][:, 0]
                embs = F.normalize(embs, p=2, dim=1).cpu().numpy()
                all_embs.append(embs)
            del inputs, out
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        
        return np.concatenate(all_embs, axis=0)

    def extract_features_batch(self, queries, vote_list, ans_list, method_names):
        print("   🔹 Encoding Queries...")
        q_embs = self._batch_encode([f"Represent this sentence for searching relevant passages: {q}" for q in queries])
        content_feats = []
        print("   🔹 Encoding Answers & Computing Similarity...")
        for m_idx, m_name in enumerate(method_names):
            current_method_answers = []
            lengths = []
            for i in range(len(queries)):
                ans = ans_list[i][m_idx]
                current_method_answers.append(ans)
                lengths.append(min(len(str(ans).split()), 512) / 512.0)
            a_embs = self._batch_encode(current_method_answers)
            sims = np.sum(q_embs * a_embs, axis=1)
            feat_block = np.stack([sims, np.array(lengths)], axis=1)
            content_feats.append(feat_block)
        content_feats = np.concatenate(content_feats, axis=1)
        final_feats = []
        for i in range(len(queries)):
            f = []
            f.append(q_embs[i])
            f.append(content_feats[i])
            v_scores = np.array(vote_list[i])
            if len(v_scores) > 0:
                mx = np.max(v_scores)
                f.append(np.array([mx, np.std(v_scores), np.mean(v_scores)], dtype=np.float32))
                f.append((mx - v_scores).astype(np.float32))
                if np.sum(v_scores) > 0:
                    f.append((v_scores / np.sum(v_scores)).astype(np.float32))
                else:
                    f.append(np.zeros_like(v_scores))
            f.append(np.array(vote_list[i], dtype=np.float32))  # base_scores block
            final_feats.append(np.concatenate(f))
        return np.array(final_feats, dtype=np.float32)


class ResidualPolicyNetwork(nn.Module):
    def __init__(self, num_methods, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM), nn.ReLU(), nn.BatchNorm1d(HIDDEN_DIM), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(), nn.BatchNorm1d(HIDDEN_DIM // 2), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM // 2, num_methods)
        )
        self.net_scale = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x, base_scores, temp=1.0):
        residual = self.net(x)
        logits = torch.log(base_scores + 1e-6) + self.net_scale * residual
        return logits, F.softmax(logits / temp, dim=-1)


def run_inference(args):
    print(f"📥 加载模型: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    saved_methods = checkpoint['method_names']
    feature_dim = checkpoint['feature_dim']
    feat_mean = checkpoint['feat_mean']
    feat_std = checkpoint['feat_std']

    print(f"📖 读取数据: {args.test_file}")
    with open(args.test_file, 'r') as f:
        data = json.load(f)

    queries, vote_list, gt_list, ans_list, em_list = [], [], [], [], []
    golden_answers_list = []
    proposed_answers = {}  # {query: answer}  供 entity_fusion 用作 baseline

    # 遍历数据
    for q, info in data.items():
        if 'details' not in info: continue
        queries.append(q)

        # 🔥 修正点：直接读取 'ground_truth_answer'
        # 你的数据里这个字段是个 list，如 ["Lawyer"]
        gt_raw = info.get('ground_truth_answer', [])
        # 确保它是 list，防止数据里偶尔出现 null
        if gt_raw is None: gt_raw = []
        if isinstance(gt_raw, str): gt_raw = [gt_raw]

        golden_answers_list.append(gt_raw)

        v, g, e, a = [], [], [], []
        for m in saved_methods:
            d = info['details'].get(m, {})
            v.append(d.get('vote_score', 0.0))
            g.append(d.get('bem_score', 0.0))
            e.append(d.get('em_score', 0.0))
            a.append(d.get('response', ""))
        vote_list.append(v)
        gt_list.append(g)
        em_list.append(e)
        ans_list.append(a)

    # GPU 选择
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"🖥️ Using GPU: {args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️ Using default GPU")
    else:
        device = torch.device('cpu')
        print(f"🖥️ Using CPU")
    extractor = ContentFeatureExtractor(args.encoder_model, device)

    print("🧩 提取特征...")
    t0 = time.time()
    feat_matrix = extractor.extract_features_batch(queries, vote_list, ans_list, saved_methods)
    t1 = time.time()
    feat_time = t1 - t0

    X = torch.FloatTensor(feat_matrix).to(device)
    VoteS = torch.FloatTensor(np.array(vote_list)).to(device)
    X = (X - feat_mean.to(device)) / feat_std.to(device)

    model = ResidualPolicyNetwork(len(saved_methods), feature_dim).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"🚀 计算全维指标 (EM / F1 / BEM)...")

    metrics = {
        'em': {'base': [], 'final': [], 'rrf': [], 'eor': []},
        'f1': {'base': [], 'final': [], 'rrf': [], 'eor': []},
        'bem_score': {'base': [], 'final': [], 'rrf': [], 'eor': []},
        'bem_acc': {'base': [], 'final': [], 'rrf': [], 'eor': []}
    }

    # 加载权重
    method_weights = load_method_weights(args.weight_file)

    with torch.no_grad():
        t2 = time.time()
        _, all_weights = model(X, VoteS)
        model_weights = all_weights.cpu().numpy()
        t3 = time.time()
        model_time = t3 - t2

        for i in tqdm(range(len(queries))):
            v_scores = np.array(vote_list[i])
            if np.sum(v_scores) > 0: v_scores = v_scores / np.sum(v_scores)

            m_scores = model_weights[i]
            final_scores = (W_VOTE * v_scores) + (W_MODEL * m_scores)

            final_idx = np.argmax(final_scores)
            base_idx = np.argmax(vote_list[i])
            proposed_answers[queries[i]] = str(ans_list[i][final_idx])
            
            # Identify Baseline Method and its Weight
            # EoR Logic: Select the answer with max ( vote_score * weight_from_file )
            eor_best_score = -float('inf')
            eor_winner_idx = base_idx # Default fallback
            
            for idx, m_name in enumerate(saved_methods):
                # Try to find weight. Handle potential naming mismatches if necessary, 
                # but user instruction implies direct match or "names need to match".
                # We try exact match first.
                w = method_weights.get(m_name, 0.0)
                
                # Raw vote score
                v_s = vote_list[i][idx]
                
                score = v_s * w
                if score > eor_best_score:
                    eor_best_score = score
                    eor_winner_idx = idx

            # --- RRF (Reciprocal Rank Fusion) / Consensus Logic ---
            # Formula: Score = sum(1 / (k + rank)). Since each method gives 1 answer, rank=1 always.
            # This effectively becomes Majority Voting.
            rrf_k = 60
            rrf_scores = collections.defaultdict(float)
            # Map normalized string back to original for evaluation
            norm_to_original = {}
            
            curr_answers = ans_list[i]
            for ans in curr_answers:
                # Use normalized answer for grouping
                n_ans = normalize_answer(ans)
                if not n_ans: continue # Skip empty
                rrf_scores[n_ans] += 1.0 / (rrf_k + 1)
                # Keep the first original string encountered for this normalized form
                if n_ans not in norm_to_original:
                    norm_to_original[n_ans] = ans
            
            # Find winner
            if not rrf_scores:
                rrf_winner_txt = ""
                # If no consensus, fallback to baseline idx? Or just empty.
                # Let's use base_idx for fallback or just empty. Using base_idx as soft fallback.
                rrf_winner_idx = base_idx 
            else:
                best_norm = max(rrf_scores, key=rrf_scores.get)
                rrf_winner_txt = norm_to_original[best_norm]
                # We need an index to get EM/BEM scores from the lists. 
                # Problem: 'rrf_winner_txt' might match multiple indices. 
                # Strategy: Pick the index of the first method that produced this answer.
                rrf_winner_idx = -1
                for midx, ans in enumerate(curr_answers):
                    if normalize_answer(ans) == best_norm:
                        rrf_winner_idx = midx
                        break
                if rrf_winner_idx == -1: rrf_winner_idx = base_idx 

            # 1. EM
            base_em = em_list[i][base_idx]
            metrics['em']['base'].append(base_em)
            metrics['em']['final'].append(em_list[i][final_idx])
            metrics['em']['rrf'].append(em_list[i][rrf_winner_idx])
            metrics['em']['eor'].append(em_list[i][eor_winner_idx])

            # 2. BEM Metrics
            base_bem = gt_list[i][base_idx]
            final_bem = gt_list[i][final_idx]
            rrf_bem = gt_list[i][rrf_winner_idx]
            eor_bem = gt_list[i][eor_winner_idx]
            
            metrics['bem_score']['base'].append(base_bem)
            metrics['bem_score']['final'].append(final_bem)
            metrics['bem_score']['rrf'].append(rrf_bem)
            metrics['bem_score']['eor'].append(eor_bem)
            
            metrics['bem_acc']['base'].append(1 if base_bem > ACCURACY_THRESHOLD else 0)
            metrics['bem_acc']['final'].append(1 if final_bem > ACCURACY_THRESHOLD else 0)
            metrics['bem_acc']['rrf'].append(1 if rrf_bem > ACCURACY_THRESHOLD else 0)
            metrics['bem_acc']['eor'].append(1 if eor_bem > ACCURACY_THRESHOLD else 0)

            # 3. 🔥 F1 Calculation
            gts = golden_answers_list[i]
            if len(gts) > 0:
                # Baseline F1
                base_txt = str(ans_list[i][base_idx])
                f1_b = metric_max_over_ground_truths(compute_f1, base_txt, gts)
                metrics['f1']['base'].append(f1_b)


                # EoR F1 (Weighted Selection)
                eor_txt = str(ans_list[i][eor_winner_idx])
                f1_e = metric_max_over_ground_truths(compute_f1, eor_txt, gts)
                metrics['f1']['eor'].append(f1_e)

                # Proposed F1
                final_txt = str(ans_list[i][final_idx])
                f1_f = metric_max_over_ground_truths(compute_f1, final_txt, gts)
                metrics['f1']['final'].append(f1_f)

                # RRF F1
                rrf_txt = str(ans_list[i][rrf_winner_idx])
                f1_r = metric_max_over_ground_truths(compute_f1, rrf_txt, gts)
                metrics['f1']['rrf'].append(f1_r)

    # === 保存 Proposed 答案供 entity_fusion 使用 ===
    if args.save_proposed:
        with open(args.save_proposed, "w", encoding="utf-8") as fp:
            json.dump(proposed_answers, fp, ensure_ascii=False, indent=2)
        print(f"💾 Proposed answers saved to: {args.save_proposed}")

    # === 打印报表 ===
    print("\n" + "=" * 115)
    print(f"📊 Final Paper Metrics (Dataset: WebQ)")
    print("-" * 115)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'EoR (Weighted)':<15} | {'RRF (Consens)':<12} | {'Proposed':<12} | {'Lift (vs Base)':<10}")
    print("-" * 115)

    def print_row(name, key, is_percent=True):
        if len(metrics[key]['base']) == 0:
            print(f"{name:<20} | N/A           | N/A             | N/A            | N/A           | N/A")
            return
        b = np.mean(metrics[key]['base'])
        f = np.mean(metrics[key]['final'])
        r = np.mean(metrics[key]['rrf'])
        e = np.mean(metrics[key]['eor'])
        
        if is_percent:
            b *= 100
            f *= 100
            r *= 100
            e *= 100
            print(f"{name:<20} | {b:.2f}%      | {e:.2f}%           | {r:.2f}%       | {f:.2f}%      | {f - b:+.2f}%")
        else:
            print(f"{name:<20} | {b:.4f}      | {e:.4f}          | {r:.4f}       | {f:.4f}      | {f - b:+.4f}")

    print_row("Exact Match (EM)", 'em')
    print_row("F1 Score", 'f1')
    print("-" * 95)
    print_row("BEM Score (Avg)", 'bem_score', is_percent=False)
    print_row(f"BEM Acc (>{ACCURACY_THRESHOLD})", 'bem_acc')
    print("-" * 115)
    
    # Time Cost Analysis
    total_time_s = feat_time + model_time
    avg_latency_ms = (total_time_s / len(queries)) * 1000
    print(f"⏱️ Time Cost Analysis (Total Queries: {len(queries)})")
    print(f"   - Feature Extraction: {feat_time:.4f}s")
    print(f"   - Neural Network:     {model_time:.4f}s")
    print(f"   - Total Time:         {total_time_s:.4f}s")
    print(f"   - Avg Latency:        {avg_latency_ms:.2f} ms/query")
    print("=" * 95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 记得改成你的 WebQ 测试集路径
    parser.add_argument('--test_file', type=str, default='./output/webq-test/train_result.json')
    parser.add_argument('--model_path', type=str, default='./output/webq-train/best_rl_v5_1.pth')
    parser.add_argument('--encoder_model', type=str, default='./data/model/bge-small-en-v1.5')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use. -1 for auto/default.')
    parser.add_argument('--weight_file', type=str, default='./output/parameter_setting/wq-llama-7b2.txt', help='Path to method weights file for EoR calculation')
    parser.add_argument('--save_proposed', type=str, default=None, help='若指定路径，将 Proposed 答案保存为 json，供 entity_fusion 用作 baseline')
    args = parser.parse_args()
    run_inference(args)
