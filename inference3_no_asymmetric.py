# -*- coding: utf-8 -*-
"""
EoR Final Inference - Corrected for Your Data Structure (Ablation: No Asymmetric Reward)
消融实验：去除非对称奖励，使用标准对称奖励（+1/-1）训练的模型。
参数锁定：Vote Weight = 0.60, Model Weight = 0.40 (WebQ Best)
"""
import argparse
import os
import json
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
W_VOTE = 0.95
W_MODEL = 0.05
ACCURACY_THRESHOLD = 0.8


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


def compute_f1(prediction, truth):
    """计算单对 F1 Score"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

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

    def _batch_encode(self, texts, batch_size=64):
        """分批编码文本，避免 OOM"""
        valid_texts = [t if isinstance(t, str) and len(t.strip()) > 0 else "empty" for t in texts]
        if self.encoder is None: return np.zeros((len(texts), EMB_DIM))

        all_embs = []
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)
            with torch.no_grad():
                out = self.encoder(**inputs)
                if hasattr(out, 'last_hidden_state'):
                    embs = out.last_hidden_state[:, 0]
                else:
                    embs = out[0][:, 0]
                embs = F.normalize(embs, p=2, dim=1).cpu().numpy()
                all_embs.append(embs)
            # 及时释放显存
            del inputs, out
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

    # 遍历数据
    for q, info in data.items():
        if 'details' not in info: continue
        queries.append(q)

        gt_raw = info.get('ground_truth_answer', [])
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

    print("🧩 提取特征 (Ablation: No Asymmetric)...")
    feat_matrix = extractor.extract_features_batch(queries, vote_list, ans_list, saved_methods)

    X = torch.FloatTensor(feat_matrix).to(device)
    VoteS = torch.FloatTensor(np.array(vote_list)).to(device)
    X = (X - feat_mean.to(device)) / feat_std.to(device)

    model = ResidualPolicyNetwork(len(saved_methods), feature_dim).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"🚀 计算全维指标 (EM / F1 / BEM)...")

    metrics = {
        'em': {'base': [], 'final': []},
        'f1': {'base': [], 'final': []},
        'bem_score': {'base': [], 'final': []},
        'bem_acc': {'base': [], 'final': []}
    }

    with torch.no_grad():
        _, all_weights = model(X, VoteS)
        model_weights = all_weights.cpu().numpy()

        for i in tqdm(range(len(queries))):
            v_scores = np.array(vote_list[i])
            if np.sum(v_scores) > 0: v_scores = v_scores / np.sum(v_scores)

            m_scores = model_weights[i]
            final_scores = (W_VOTE * v_scores) + (W_MODEL * m_scores)

            final_idx = np.argmax(final_scores)
            base_idx = np.argmax(vote_list[i])

            # 1. EM
            metrics['em']['base'].append(em_list[i][base_idx])
            metrics['em']['final'].append(em_list[i][final_idx])

            # 2. BEM Metrics
            base_bem = gt_list[i][base_idx]
            final_bem = gt_list[i][final_idx]
            metrics['bem_score']['base'].append(base_bem)
            metrics['bem_score']['final'].append(final_bem)
            metrics['bem_acc']['base'].append(1 if base_bem > ACCURACY_THRESHOLD else 0)
            metrics['bem_acc']['final'].append(1 if final_bem > ACCURACY_THRESHOLD else 0)

            # 3. F1 Calculation
            gts = golden_answers_list[i]
            if len(gts) > 0:
                # Baseline F1
                base_txt = str(ans_list[i][base_idx])
                f1_b = metric_max_over_ground_truths(compute_f1, base_txt, gts)
                metrics['f1']['base'].append(f1_b)

                # Proposed F1
                final_txt = str(ans_list[i][final_idx])
                f1_f = metric_max_over_ground_truths(compute_f1, final_txt, gts)
                metrics['f1']['final'].append(f1_f)

    # === 打印报表 ===
    print("\n" + "=" * 80)
    print(f"📊 Final Paper Metrics (Dataset: WebQ) - Ablation: No Asymmetric Reward")
    print("-" * 80)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'Proposed':<12} | {'Lift':<10}")
    print("-" * 80)

    def print_row(name, key, is_percent=True):
        if len(metrics[key]['base']) == 0:
            print(f"{name:<20} | N/A           | N/A           | N/A")
            return
        b = np.mean(metrics[key]['base'])
        f = np.mean(metrics[key]['final'])
        if is_percent:
            b *= 100
            f *= 100
            print(f"{name:<20} | {b:.2f}%      | {f:.2f}%      | {f - b:+.2f}%")
        else:
            print(f"{name:<20} | {b:.4f}      | {f:.4f}      | {f - b:+.4f}")

    print_row("Exact Match (EM)", 'em')
    print_row("F1 Score", 'f1')
    print("-" * 80)
    print_row("BEM Score (Avg)", 'bem_score', is_percent=False)
    print_row(f"BEM Acc (>{ACCURACY_THRESHOLD})", 'bem_acc')
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./output/tqa-test/train_result.json')
    # 默认加载无非对称奖励模型
    parser.add_argument('--model_path', type=str, default='./output/output_rl_v5_0_no_asymmetric/nq/best_rl_v5_0_no_asymmetric.pth')
    parser.add_argument('--encoder_model', type=str, default='./data/model/bge-small-en-v1.5')
    parser.add_argument('--gpu', type=int, default=2, help='GPU ID to use, -1 for CPU')
    args = parser.parse_args()
    run_inference(args)
