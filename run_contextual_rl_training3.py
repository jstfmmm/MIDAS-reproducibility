# -*- coding: utf-8 -*-
"""
EoR V5.0 - Conservative Corrector
策略调整：
1. 采样比例：Type A (60%) : Type B (40%)。该比例由 WeightedRandomSampler 的组权重控制。
2. 严厉惩罚：Type A 失败惩罚 -5.0。动了不该动的，直接罚出局。
3. 奖励：Type B 成功 +5.0，失败 -0.1；Type A 成功 +0.1，失败 -5.0。
"""

import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import re
import string
import warnings

warnings.filterwarnings("ignore")

# ==================== ⚙️ 参数 ====================
HIDDEN_DIM = 512
EMB_DIM = 384
EXTRA_FEATS = 12
EPOCHS = 150  # 轮数可以少一点，避免过拟合
LEARNING_RATE = 5e-5  # 🔥 LR 再次降低，微操
WEIGHT_DECAY = 1e-4  # 🔥 正则化加强，压制模型乱动的欲望
BATCH_SIZE = 64
BEM_THRESHOLD = 0.8
FEATURE_GROUPS = ["query_embedding", "base_scores", "stats_group"]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== 特征提取 (保持不变) ====================
def normalize_answer(s):
    if not isinstance(s, str): return ""

    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text): return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


class ContentFeatureExtractor:
    def __init__(self, encoder_model_path, device='cpu'):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
            self.encoder = AutoModel.from_pretrained(encoder_model_path)
            self.encoder.to(device).eval()
        except:
            self.encoder = None

    def _batch_encode(self, texts, batch_size=16):
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
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        
        return np.concatenate(all_embs, axis=0)

    def extract_features_batch(self, queries, vote_list, ans_list, method_names, feature_groups=None):
        if feature_groups is None:
            feature_groups = FEATURE_GROUPS
        feature_groups = set(feature_groups)

        q_embs = None
        if "query_embedding" in feature_groups or "stats_group" in feature_groups:
            print("   🔹 Encoding Queries...")
            q_embs = self._batch_encode([f"Represent this sentence for searching relevant passages: {q}" for q in queries])

        content_feats = []
        if "stats_group" in feature_groups:
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
            content_feats = np.concatenate(content_feats, axis=1) if len(content_feats) > 0 else None

        final_feats = []
        for i in range(len(queries)):
            f = []
            if "query_embedding" in feature_groups:
                f.append(q_embs[i])
            if "stats_group" in feature_groups and content_feats is not None:
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
            if "base_scores" in feature_groups:
                f.append(np.array(vote_list[i], dtype=np.float32))
            final_feats.append(np.concatenate(f))
        return np.array(final_feats, dtype=np.float32)


class ThresholdDataset(Dataset):
    def __init__(self, features, method_scores, em_scores, bem_scores, types):
        self.features = torch.FloatTensor(features)
        self.method_scores = torch.FloatTensor(method_scores)
        self.em_scores = torch.FloatTensor(em_scores)
        self.bem_scores = torch.FloatTensor(bem_scores)
        self.types = torch.LongTensor(types)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx): return self.features[idx], self.method_scores[idx], self.em_scores[idx], \
    self.bem_scores[idx], self.types[idx]


class ResidualPolicyNetwork(nn.Module):
    def __init__(self, num_methods, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM), nn.ReLU(), nn.BatchNorm1d(HIDDEN_DIM), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(), nn.BatchNorm1d(HIDDEN_DIM // 2), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM // 2, num_methods)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net_scale = nn.Parameter(torch.tensor([0.01]))  # 🔥 初始 scale 极小，强制从 Baseline 开始微调

    def forward(self, x, base_scores, temp=1.0):
        residual = self.net(x)
        logits = torch.log(base_scores + 1e-6) + self.net_scale * residual
        return logits, F.softmax(logits / temp, dim=-1)


# ==================== 🔥 保守型训练器 🔥 ====================
class ConservativeRLTrainer:
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def train_step(self, features, method_scores, em_scores, bem_scores, types):
        self.policy.train()
        logits, weights = self.policy(features, method_scores)

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        selected_em = em_scores.gather(1, actions.unsqueeze(1)).squeeze(1)
        selected_bem = bem_scores.gather(1, actions.unsqueeze(1)).squeeze(1)
        is_success = (selected_em == 1.0) | (selected_bem > BEM_THRESHOLD)

        rewards = torch.zeros_like(selected_bem)

        for i in range(len(types)):
            success = is_success[i].item()
            is_type_b = (types[i] == 1)

            if is_type_b:
                # Type B (进攻): 成功了大奖，失败了轻罚
                if success:
                    rewards[i] = 5
                else:
                    rewards[i] = -0.1
            else:
                # Type A (防守): 失败了重罚！
                if success:
                    rewards[i] = 0.1  # 守住有奖
                else:
                    rewards[i] = -5.0  # 🔥 绝对高压线！

        loss = -(log_probs * rewards).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return rewards.mean().item()


# ==================== 主流程 ====================
def run_training(args):
    set_global_seed(args.seed)
    print(f"🎲 Random seed: {args.seed}")
    print(f"📖 读取: {args.train_result}")
    with open(args.train_result, 'r') as f:
        data = json.load(f)

    queries, vote_list, em_list, bem_list, ans_list = [], [], [], [], []
    first_key = list(data.keys())[0]
    method_names = sorted(list(data[first_key]['details'].keys()))

    for q, info in data.items():
        if 'details' not in info: continue
        queries.append(q)
        v, e, b, a = [], [], [], []
        for m in method_names:
            d = info['details'].get(m, {})
            v.append(d.get('vote_score', 0.0));
            e.append(d.get('em_score', 0.0));
            b.append(d.get('bem_score', 0.0));
            a.append(d.get('response', ""))
        vote_list.append(v);
        em_list.append(e);
        bem_list.append(b);
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
    print("🧩 提取特征 (V3.0 Content-Aware)...")
    print(f"🧪 Feature groups: {args.feature_groups}")
    feat_matrix = extractor.extract_features_batch(queries, vote_list, ans_list, method_names, feature_groups=args.feature_groups)

    type_a_indices, type_b_indices = [], []
    for i in range(len(queries)):
        v = np.array(vote_list[i])
        e = np.array(em_list[i])
        b = np.array(bem_list[i])
        base_idx = np.argmax(v)
        base_success = (e[base_idx] == 1.0) or (b[base_idx] > BEM_THRESHOLD)
        any_success = np.any((e == 1.0) | (b > BEM_THRESHOLD))
        if base_success:
            type_a_indices.append(i)
        elif any_success:
            type_b_indices.append(i)

    print(f"\n📊 数据分拣:")
    print(f"   - Type A: {len(type_a_indices)}")
    print(f"   - Type B: {len(type_b_indices)}")

    valid_indices = type_a_indices + type_b_indices
    data_types = np.zeros(len(valid_indices), dtype=int)
    data_types[len(type_a_indices):] = 1

    X_final = feat_matrix[valid_indices]
    V_final = np.array(vote_list)[valid_indices]
    E_final = np.array(em_list)[valid_indices]
    B_final = np.array(bem_list)[valid_indices]

    X_mean, X_std = X_final.mean(axis=0), X_final.std(axis=0) + 1e-8
    X_final = (X_final - X_mean) / X_std

    if len(type_b_indices) > 0:
        # 🔥 调整采样比例：60% A : 40% B
        # 每个组内均匀采样，组总权重分别为 0.6 和 0.4。
        # 组总权重为 0.6 和 0.4，采样时按组内样本数均匀分配。
        weight_a = 0.6 / len(type_a_indices)
        weight_b = 0.4 / len(type_b_indices)
        samples_weights = torch.tensor([weight_a if t == 0 else weight_b for t in data_types])
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    else:
        sampler = None

    dataset = ThresholdDataset(X_final, V_final, E_final, B_final, data_types)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    policy = ResidualPolicyNetwork(len(method_names), X_final.shape[1]).to(device)
    trainer = ConservativeRLTrainer(policy)

    print("\n🚀 开始 V5.0 保守型修正训练...")
    pbar = tqdm(range(EPOCHS))
    for ep in pbar:
        total_r = 0
        steps = 0
        for batch_x, batch_v, batch_e, batch_b, batch_t in dataloader:
            batch_x, batch_v = batch_x.to(device), batch_v.to(device)
            batch_e, batch_b = batch_e.to(device), batch_b.to(device)
            batch_t = batch_t.to(device)
            r = trainer.train_step(batch_x, batch_v, batch_e, batch_b, batch_t)
            total_r += r
            steps += 1
        pbar.set_postfix({'Reward': f"{total_r / steps:.4f}"})
        if ep % 10 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'state_dict': policy.state_dict(),
                'feat_mean': torch.FloatTensor(X_mean),
                'feat_std': torch.FloatTensor(X_std),
                'method_names': method_names,
                'feature_dim': X_final.shape[1],
                'feature_groups': list(args.feature_groups)
            }, os.path.join(args.output_dir, "best_rl_v5_1.pth"))

    print(f"\n✅ 训练完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--train_result', type=str, default='./output/webq-train/train_result3.json')
    parser.add_argument('-o', '--output_dir', type=str, default='./output/webq-train/')
    parser.add_argument('--encoder_model', type=str, default='./data/model/bge-small-en-v1.5')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use. -1 for auto/default.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for variance experiments.')
    parser.add_argument('--feature_groups', type=str, nargs='+', default=FEATURE_GROUPS, choices=FEATURE_GROUPS,
                        help='Feature groups used for training: query_embedding, base_scores, stats_group')
    args = parser.parse_args()
    run_training(args)
