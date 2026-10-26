# -*- coding: utf-8 -*-
"""
BERT-based contextual RL trainer (derived from run_contextual_rl_training3.py)
策略改动：
1. 用 BERT 编码 query（CLS）作为策略语义特征
2. 与 base vote scores 融合，输出 residual logits
3. 保留原有保守奖励逻辑（Type A / Type B）
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
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

# ==================== ⚙️ 参数 ====================
HIDDEN_DIM = 512
EPOCHS = 150
LEARNING_RATE_HEAD = 5e-5
LEARNING_RATE_BERT = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
BEM_THRESHOLD = 0.8


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


class BertThresholdDataset(Dataset):
    def __init__(self, queries, method_scores, em_scores, bem_scores, types):
        self.queries = queries
        self.method_scores = torch.FloatTensor(method_scores)
        self.em_scores = torch.FloatTensor(em_scores)
        self.bem_scores = torch.FloatTensor(bem_scores)
        self.types = torch.LongTensor(types)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return (
            self.queries[idx],
            self.method_scores[idx],
            self.em_scores[idx],
            self.bem_scores[idx],
            self.types[idx],
        )


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

        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
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


class ConservativeRLTrainer:
    def __init__(self, policy, freeze_bert=True):
        self.policy = policy

        if freeze_bert:
            self.optimizer = optim.AdamW(
                [p for p in policy.parameters() if p.requires_grad],
                lr=LEARNING_RATE_HEAD,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            head_params = []
            bert_params = []
            for n, p in policy.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("encoder."):
                    bert_params.append(p)
                else:
                    head_params.append(p)
            self.optimizer = optim.AdamW(
                [
                    {"params": bert_params, "lr": LEARNING_RATE_BERT},
                    {"params": head_params, "lr": LEARNING_RATE_HEAD},
                ],
                weight_decay=WEIGHT_DECAY,
            )

    def train_step(self, input_ids, attention_mask, method_scores, em_scores, bem_scores, types):
        self.policy.train()
        logits, _ = self.policy(input_ids, attention_mask, method_scores)

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
                rewards[i] = 5 if success else -0.1
            else:
                rewards[i] = 0.1 if success else -5.0

        loss = -(log_probs * rewards).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return rewards.mean().item()


def build_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        queries = [x[0] for x in batch]
        method_scores = torch.stack([x[1] for x in batch], dim=0)
        em_scores = torch.stack([x[2] for x in batch], dim=0)
        bem_scores = torch.stack([x[3] for x in batch], dim=0)
        types = torch.stack([x[4] for x in batch], dim=0)

        enc = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return enc["input_ids"], enc["attention_mask"], method_scores, em_scores, bem_scores, types

    return collate_fn


def run_training(args):
    set_global_seed(args.seed)
    print(f"🎲 Random seed: {args.seed}")
    print(f"📖 读取: {args.train_result}")

    with open(args.train_result, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries, vote_list, em_list, bem_list = [], [], [], []
    first_key = list(data.keys())[0]
    method_names = sorted(list(data[first_key]["details"].keys()))

    for q, info in data.items():
        if "details" not in info:
            continue

        queries.append(q)
        v, e, b = [], [], []
        for m in method_names:
            d = info["details"].get(m, {})
            v.append(d.get("vote_score", 0.0))
            e.append(d.get("em_score", 0.0))
            b.append(d.get("bem_score", 0.0))

        vote_list.append(v)
        em_list.append(e)
        bem_list.append(b)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"🖥️ Using GPU: {args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🖥️ Using default GPU")
    else:
        device = torch.device("cpu")
        print("🖥️ Using CPU")

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

    print("\n📊 数据分拣:")
    print(f"   - Type A: {len(type_a_indices)}")
    print(f"   - Type B: {len(type_b_indices)}")

    valid_indices = type_a_indices + type_b_indices
    data_types = np.zeros(len(valid_indices), dtype=int)
    data_types[len(type_a_indices):] = 1

    queries_final = [queries[i] for i in valid_indices]
    v_final = np.array(vote_list, dtype=np.float32)[valid_indices]
    e_final = np.array(em_list, dtype=np.float32)[valid_indices]
    b_final = np.array(bem_list, dtype=np.float32)[valid_indices]

    if len(type_b_indices) > 0:
        weight_a = 0.6 / max(1, len(type_a_indices))
        weight_b = 0.4 / max(1, len(type_b_indices))
        samples_weights = torch.tensor([weight_a if t == 0 else weight_b for t in data_types], dtype=torch.float)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    else:
        sampler = None

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = BertThresholdDataset(queries_final, v_final, e_final, b_final, data_types)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=build_collate_fn(tokenizer, args.max_length),
    )

    policy = BertResidualPolicyNetwork(
        bert_model_name=args.bert_model,
        num_methods=len(method_names),
        freeze_bert=args.freeze_bert,
    ).to(device)

    trainer = ConservativeRLTrainer(policy, freeze_bert=args.freeze_bert)

    print("\n🚀 开始 BERT 策略训练...")
    pbar = tqdm(range(args.epochs))

    for ep in pbar:
        total_r = 0.0
        steps = 0

        for input_ids, attention_mask, batch_v, batch_e, batch_b, batch_t in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_v = batch_v.to(device)
            batch_e = batch_e.to(device)
            batch_b = batch_b.to(device)
            batch_t = batch_t.to(device)

            r = trainer.train_step(input_ids, attention_mask, batch_v, batch_e, batch_b, batch_t)
            total_r += r
            steps += 1

        pbar.set_postfix({"Reward": f"{total_r / max(1, steps):.4f}"})

        if ep % 10 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "method_names": method_names,
                    "bert_model": args.bert_model,
                    "freeze_bert": args.freeze_bert,
                    "max_length": args.max_length,
                },
                os.path.join(args.output_dir, "best_rl_v5_bert.pth"),
            )

    print("\n✅ 训练完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_result", type=str, default="./output/webq-train/train_result3.json")
    parser.add_argument("-o", "--output_dir", type=str, default="./output/output_rl_v5_bert/webq")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use. -1 for auto/default.")
    parser.add_argument("--seed", type=int, default=54, help="Random seed for variance experiments.")
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze BERT encoder and only train head.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=128)

    args = parser.parse_args()
    run_training(args)