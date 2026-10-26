import argparse
import warnings
import torch
import os

from eor.config import Config
from eor.main import run_eor

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 新增：在程序开始时限制线程数
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer多线程
    # os.environ["RAYON_NUM_THREADS"] = "1"  # 限制Rust线程池
    # os.environ["OMP_NUM_THREADS"] = "1"  # 关闭OpenMP并行
    # os.environ["MKL_NUM_THREADS"] = "1"  # 关闭MKL并行
    # torch.set_num_threads(1)  # 限制PyTorch线程

    # CUDA显存优化设置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 调试时才开启，会降低性能

    # DDP分布式训练端口设置（避免端口冲突）
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"  # 改用29501端口，避免与其他进程冲突

    warnings.filterwarnings('ignore')

    # parse args
    parser = argparse.ArgumentParser(description='a project on Ensemble of Retrievers (EoR)')
    parser.add_argument('-c', '--config', type=str, default='config/llama7b_multi_gpu.yaml', help='config file(yaml) path')
    parser.add_argument('-d', '--debug', action='store_true', help='use valid dataset to debug your system')
    args, _ = parser.parse_known_args()

    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 14327))
        debugpy.wait_for_client()

    config = Config(args.config, debug=args.debug)

    if config['ddp']:
        world_size = len(config["device"])
        torch.multiprocessing.spawn(
            run_eor,
            args=(config, args.debug),
            nprocs=world_size,
            join=True
        )
    else:
        run_eor(None, config)