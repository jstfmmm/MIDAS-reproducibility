# MIDAS / Contextual RL reproducibility repository

This repository contains the cleaned source code for the MIDAS contextual residual policy and the underlying Ensemble-of-Retrievers (EoR) framework. Full datasets, downloaded model weights, retrieval indexes, checkpoints, API credentials, and experiment outputs are excluded. A small web retrieval example is included as described below.

## Repository layout

```text
run_contextual_rl_training3.py   train the contextual residual policy
inference3.py                    run the policy and report EM/F1/BEM metrics
run.py                           original EoR generation entry point
run_evaluate.py                  evaluate an EoR result file
eor/                             EoR framework implementation
config/                          YAML templates for the EoR pipeline
data/datasets/                   dataset conversion and sampling utilities only
```

The following paths are reserved for local files and ignored by Git (the repository does not include full experiment datasets, model weights, indexes, or outputs):

```text
data/datasets/   input datasets and processed JSON/JSONL files
data/model/      Hugging Face, BEM, Contriever, and generator weights
data/index/      local indexes used by the Chinese/Wikipedia retrievers
output/          generated answers, caches, logs, and checkpoints
```

## Environment

Use Python 3.10–3.11 for the full EoR workflow and install a PyTorch build that matches the local CUDA driver (or the CPU build for a smoke test). The contextual RL scripts can also run under Python 3.9. Then install the project dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # choose the suitable PyTorch index
python -m pip install -r requirements.txt
```

`torch`, `faiss`, and `pyserini` are hardware/platform dependent. Install the compatible variants for the target machine. The contextual RL scripts only need PyTorch and Transformers; the full EoR pipeline additionally needs the retriever, generation, evaluation, and optional API dependencies listed in `requirements.txt`.

Copy `.env.example` to `.env` or export the variables in the shell. The code reads credentials from the environment; no key belongs in a YAML file or source file. The web fetcher optionally honors `HTTPS_PROXY` or `HTTP_PROXY` when a proxy is required.

```bash
# Windows PowerShell
$env:SERPER_API_KEY="..."
$env:OPENAI_API_KEY="..."    # only for OpenAI-backed modules

# Linux/macOS
export SERPER_API_KEY=...
export OPENAI_API_KEY=...    # only for OpenAI-backed modules
```

## Contextual RL input format

The training and inference scripts consume a JSON object whose keys are queries. Each item must contain a `details` mapping with the same method names for every example. Training uses `vote_score`, `em_score`, `bem_score`, and `response`; inference also reads `ground_truth_answer` when F1 is required.

```json
{
  "who wrote hamlet": {
    "ground_truth_answer": ["William Shakespeare"],
    "details": {
      "wiki": {
        "vote_score": 0.8,
        "em_score": 1.0,
        "bem_score": 0.95,
        "response": "William Shakespeare"
      },
      "gendoc": {
        "vote_score": 0.2,
        "em_score": 0.0,
        "bem_score": 0.10,
        "response": "Christopher Marlowe"
      }
    }
  }
}
```

All examples used by a checkpoint must have the same method ordering. The policy checkpoint stores `state_dict`, `feat_mean`, `feat_std`, `method_names`, `feature_dim`, and `feature_groups`; `inference3.py` checks these fields when loading a checkpoint.

## Train and infer

The current training script uses the following code-level defaults: 384-dimensional encoder features, a 512-unit policy hidden layer, 150 epochs, batch size 64, learning rate `5e-5`, weight decay `1e-4`, BEM success threshold `0.8`, and seed `42`. Type A/Type B examples are sampled with group weights 0.6/0.4. Override paths and the seed from the command line:

```bash
python run_contextual_rl_training3.py \
  --train_result path/to/train_result.json \
  --output_dir output/webq-train \
  --encoder_model BAAI/bge-small-en-v1.5 \
  --gpu 0 \
  --seed 42
```

The encoder must produce 384-dimensional embeddings, as expected by the current feature extractor. The checked-in script defaults to `./data/model/bge-small-en-v1.5`; you can instead pass the equivalent Hugging Face model ID `BAAI/bge-small-en-v1.5`, which is downloaded to the Hugging Face cache unless a local path is supplied.

Run inference with the checkpoint produced by training:

```bash
python inference3.py \
  --test_file path/to/test_result.json \
  --model_path output/webq-train/best_rl_v5_1.pth \
  --encoder_model BAAI/bge-small-en-v1.5 \
  --gpu 0 \
  --save_proposed output/proposed_answers.json
```

The final policy combines normalized vote scores and policy scores with code-level weights `W_VOTE=0.8` and `W_MODEL=0.2`. The script reports EM, F1, BEM score, BEM accuracy, and timing for the baseline, EoR-weighted, RRF, and proposed selections. A method-weight file is optional; omit `--weight_file` when the EoR-weighted comparison is not needed.

### Ablation and comparison scripts

The repository includes the training and inference entry points used for the paper's ablations and encoder comparison. They are kept alongside the main experiment so that each reported comparison can be reproduced from its corresponding source file.

| Training script | Inference script | Comparison |
|---|---|---|
| `run_contextual_rl_training3_no_asymmetric.py` | `inference3_no_asymmetric.py` | Symmetric reward ablation; removes the Type A/Type B asymmetric reward magnitudes. |
| `run_contextual_rl_training3_no_sampling.py` | `inference3_no_sampling.py` | Sampling ablation; removes the fixed Type A/Type B sampling proportions and follows the available data distribution. |
| `run_contextual_rl_training3_no_semantic.py` | `inference3_no_semantic.py` | Semantic-feature ablation; removes semantic query features and retains the voting/statistical features. |
| `run_contextual_rl_training3_bert.py` | `inference3_bert.py` | BERT policy-encoder comparison. |

The main contextual-RL script currently defaults to 150 epochs. Some legacy ablation scripts in this repository default to 200 epochs, while the BERT training script uses its own encoder and optimization defaults. These values are source-level defaults; use the command-line arguments and configuration in the selected script when reproducing a specific table or ablation. The scripts require the same external datasets, retrieval outputs, encoder checkpoints, and output directories described above; those files remain ignored and are not committed.

### Code-level policy configuration (verified)

The values below are read directly from `run_contextual_rl_training3.py` and `inference3.py` and are the reproducibility values that can be supported by this source tree:

| Component | Setting |
|---|---|
| Query/answer encoder | `BAAI/bge-small-en-v1.5` (384-dimensional CLS embedding, L2-normalized; tokenizer truncation `max_length=128`) |
| Feature blocks | `query_embedding`, `base_scores`, `stats_group`; the resulting feature dimension is checked in the checkpoint |
| Policy MLP | `d → 512 → 256 → K`, ReLU and BatchNorm after each hidden linear layer, Dropout `0.3`, final linear output of `K` methods |
| Residual parameterization | `log(base_scores + 1e-6) + net_scale * residual`; `net_scale` is initialized to `0.01`; final layer weights and bias are initialized to zero; softmax temperature is `1.0` |
| Optimizer | AdamW, learning rate `5e-5`, weight decay `1e-4` |
| Training | `150` epochs, batch size `64`, global seed default `42`, deterministic cuDNN flags when CUDA is available |
| Sampling | WeightedRandomSampler with group totals Type A `0.6` and Type B `0.4` |
| Success rule | `EM == 1.0` or `BEM > 0.8` |
| Rewards | Type A: success `+0.1`, failure `-5.0`; Type B: success `+5.0`, failure `-0.1` |
| Inference fusion | normalized vote score weight `0.8` and policy weight `0.2` |
| Exploration | No ε-greedy or ε parameter is implemented; training samples from the categorical policy distribution induced by the logits |

For the checked-in EoR template `config/llama3_8b_generator_on_7b_cache.yaml`, the source records Wiki `top_k=5`, Contriever `top_k=5` with a 512-token truncation and mean pooling, and one-response generation with Llama-3.1-8B-Instruct, `temperature=0.7`, `top_p=0.9`, `top_k=50`, `do_sample=false`, and `max_new_tokens=150`. Its prompt template IDs are stored in the YAML file. The Wiki implementation uses the Pyserini prebuilt index `wikipedia-dpr-multi-bf` with `facebook/dpr-question_encoder-multiset-base`; the web retriever uses Serper and reads `SERPER_API_KEY` from the environment.

These are source-level settings, not a claim that the original run used every template in this directory. The target experiment must identify the exact YAML file and cache/snapshot used.

## Original EoR pipeline

`run.py` and `run_evaluate.py` are retained for the original EoR workflow. Start from one of the templates in `config/`, update dataset and model paths, and run:

```bash
python run.py --config config/llama7b_optimized.yaml
python run_evaluate.py \
  --result_path output/<run>/result.json \
  --output_path output/<run> \
  --dataset_name NQ \
  --metric em bem f1
```

The templates describe the available modules and their retrieval/generation settings; they do not provide the corresponding data or weights. Web retrieval requires `SERPER_API_KEY`. OpenAI-backed generation requires `OPENAI_API_KEY`. The full EoR Wiki module uses the Pyserini prebuilt `wikipedia-dpr-multi-bf` index and the DPR question encoder `facebook/dpr-question_encoder-multiset-base`; these are downloaded or cached outside this repository. The Chinese Wiki path additionally expects a local `data/index/` snapshot and encoder, which must be supplied by the user.

## External data and models

Download the NQ, TriviaQA, and WebQ source files from the dataset location documented by the upstream EoR project, then process them with the utilities under `data/datasets/`. Keep the resulting files under the ignored `data/datasets/` directory. The BEM evaluator expects the Google Answer Equivalence model (`google/answer_equivalence/bem/1`) and the local vocabulary file already retained under `eor/evaluation/metrics/vocab.txt`; place the downloaded BEM model under `data/model/` or adjust the evaluator path locally. Contriever/WebGLM encoders and causal language models are likewise external dependencies and should be referenced by Hugging Face ID or a local path.

### Example web retrieval snapshot

The `snapshots/web-example/` directory contains the supplied `web_doc.json` and `web_info.json` cache files for three queries. They include retrieved document text and search/fetch metadata, respectively. The files are a partial example, not the full training or test data, and their connection to any reported experiment has not been verified. The retrieval date and exact index or model revisions are not established by these files.

Generated answer files, model directories, checkpoints, `.env`, and logs are not included. Exact numerical reproduction still requires the full dataset and retrieval snapshots, model revisions, retrieval index, CUDA/PyTorch build, and random seeds used for the target experiment.

The following items remain unresolved and must be supplied before claiming a fully closed P0 reproducibility record: the exact target YAML/config revision; dataset and web/Wikipedia snapshot identifiers or hashes; the Pyserini/FAISS and model revisions actually used; exact CUDA, driver, PyTorch and hardware information; the original repository commit or release; and the checkpoint file and metadata used for the reported results. The source tree contains no evidence for a 200-epoch run or an `epoch-191` checkpoint; the checked-in contextual RL script defaults to 150 epochs and writes `best_rl_v5_1.pth`.

## License and upstream attribution

The `eor/` implementation is derived from the upstream Ensemble-of-Retrievers project. Preserve its original license and attribution requirements when publishing this repository. Add the applicable license file before public release if the upstream license requires it.
