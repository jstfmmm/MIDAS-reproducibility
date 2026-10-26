# -*- coding: utf-8 -*-
"""
EoR BERT Inference
Compatible with run_contextual_rl_training3_bert.py checkpoint.
"""

import argparse
import os
os.environ.setdefault("USE_TORCHVISION", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")
import importlib.util
_orig_find_spec = importlib.util.find_spec

def _find_spec_without_torchvision(name, *args, **kwargs):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _orig_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _find_spec_without_torchvision
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

# ==================== 参数 ====================
HIDDEN_DIM = 512
W_VOTE = 0.8
W_MODEL = 0.2
ACCURACY_THRESHOLD = 0.8


# ==================== 评估工具函数 ====================
def normalize_answer(s):
    if not isinstance(s, str):
        return ""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def load_method_weights(weight_file):
    if not weight_file or not os.path.exists(weight_file):
        print(f"⚠️ Weight file not found: {weight_file}")
        return {}

    weights = {}
    try:
        with open(weight_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            parts = content.split()
            for i in range(0, len(parts), 2):
                key = parts[i].rstrip(':')
                val = float(parts[i + 1])
                weights[key] = val
        print(f"⚖️ Loaded {len(weights)} method weights from {weight_file}")
        return weights
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return {}


def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores = []
    for gt in ground_truths:
        scores.append(metric_fn(prediction, gt))
    return max(scores) if scores else 0


# ==================== BERT 策略模型 ====================
class BertResidualPolicyNetwork(nn.Module):
    def __init__(self, bert_model_name, num_methods, freeze_bert=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.freeze_bert = freeze_bert

        if self.freeze_bert:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + num_methods, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM // 2, num_methods),
        )

        self.net_scale = nn.Parameter(torch.tensor([0.01]))

    def forward(self, input_ids, attention_mask, base_scores, temp=1.0):
        if self.freeze_bert:
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls = out.last_hidden_state[:, 0]
        x = torch.cat([cls, base_scores], dim=-1)
        residual = self.fusion(x)
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
    bert_model = checkpoint.get('bert_model', args.bert_model)
    freeze_bert = checkpoint.get('freeze_bert', True)
    max_length = checkpoint.get('max_length', args.max_length)

    print(f"📖 读取数据: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries, vote_list, gt_list, ans_list, em_list = [], [], [], [], []
    golden_answers_list = []

    for q, info in data.items():
        if 'details' not in info:
            continue
        queries.append(q)

        gt_raw = info.get('ground_truth_answer', [])
        if gt_raw is None:
            gt_raw = []
        if isinstance(gt_raw, str):
            gt_raw = [gt_raw]
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

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"🖥️ Using GPU: {args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("🖥️ Using default GPU")
    else:
        device = torch.device('cpu')
        print("🖥️ Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BertResidualPolicyNetwork(
        bert_model_name=bert_model,
        num_methods=len(saved_methods),
        freeze_bert=freeze_bert,
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("🚀 计算全维指标 (EM / F1 / BEM)...")

    metrics = {
        'em': {'base': [], 'final': [], 'rrf': [], 'eor': []},
        'f1': {'base': [], 'final': [], 'rrf': [], 'eor': []},
        'bem_score': {'base': [], 'final': [], 'rrf': [], 'eor': []},
        'bem_acc': {'base': [], 'final': [], 'rrf': [], 'eor': []}
    }

    method_weights = load_method_weights(args.weight_file)

    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(len(queries))):
            base_scores = np.array(vote_list[i], dtype=np.float32)
            if np.sum(base_scores) > 0:
                base_scores = base_scores / np.sum(base_scores)

            enc = tokenizer(
                [queries[i]],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            vote_tensor = torch.FloatTensor(base_scores).unsqueeze(0).to(device)

            _, weights = model(input_ids, attention_mask, vote_tensor)
            m_scores = weights[0].cpu().numpy()

            final_scores = (W_VOTE * base_scores) + (W_MODEL * m_scores)
            final_idx = np.argmax(final_scores)
            base_idx = np.argmax(vote_list[i])

            eor_best_score = -float('inf')
            eor_winner_idx = base_idx
            for idx, m_name in enumerate(saved_methods):
                w = method_weights.get(m_name, 0.0)
                score = vote_list[i][idx] * w
                if score > eor_best_score:
                    eor_best_score = score
                    eor_winner_idx = idx

            rrf_k = 60
            rrf_scores = collections.defaultdict(float)
            norm_to_original = {}
            curr_answers = ans_list[i]

            for ans in curr_answers:
                n_ans = normalize_answer(ans)
                if not n_ans:
                    continue
                rrf_scores[n_ans] += 1.0 / (rrf_k + 1)
                if n_ans not in norm_to_original:
                    norm_to_original[n_ans] = ans

            if not rrf_scores:
                rrf_winner_idx = base_idx
            else:
                best_norm = max(rrf_scores, key=rrf_scores.get)
                rrf_winner_idx = -1
                for midx, ans in enumerate(curr_answers):
                    if normalize_answer(ans) == best_norm:
                        rrf_winner_idx = midx
                        break
                if rrf_winner_idx == -1:
                    rrf_winner_idx = base_idx

            metrics['em']['base'].append(em_list[i][base_idx])
            metrics['em']['final'].append(em_list[i][final_idx])
            metrics['em']['rrf'].append(em_list[i][rrf_winner_idx])
            metrics['em']['eor'].append(em_list[i][eor_winner_idx])

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

            gts = golden_answers_list[i]
            if len(gts) > 0:
                base_txt = str(ans_list[i][base_idx])
                eor_txt = str(ans_list[i][eor_winner_idx])
                final_txt = str(ans_list[i][final_idx])
                rrf_txt = str(ans_list[i][rrf_winner_idx])

                metrics['f1']['base'].append(metric_max_over_ground_truths(compute_f1, base_txt, gts))
                metrics['f1']['eor'].append(metric_max_over_ground_truths(compute_f1, eor_txt, gts))
                metrics['f1']['final'].append(metric_max_over_ground_truths(compute_f1, final_txt, gts))
                metrics['f1']['rrf'].append(metric_max_over_ground_truths(compute_f1, rrf_txt, gts))

    t1 = time.time()

    print("\n" + "=" * 115)
    print("📊 Final Paper Metrics (BERT Policy)")
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

    total_time_s = t1 - t0
    avg_latency_ms = (total_time_s / max(1, len(queries))) * 1000
    print(f"⏱️ Time Cost Analysis (Total Queries: {len(queries)})")
    print(f"   - Total Time:         {total_time_s:.4f}s")
    print(f"   - Avg Latency:        {avg_latency_ms:.2f} ms/query")
    print("=" * 95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./output/nq-dev/train_result2.json')
    parser.add_argument('--model_path', type=str, default='./output/output_rl_v5_bert/nq/best_rl_v5_bert.pth')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use. -1 for auto/default.')
    parser.add_argument('--weight_file', type=str, default='./output/parameter_setting/nq-llama-7b2.txt', help='Path to method weights file for EoR calculation')
    args = parser.parse_args()
    run_inference(args)