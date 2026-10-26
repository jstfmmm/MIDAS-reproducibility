import os
import glob
from loguru import logger

def get_distributed_cache_paths(base_path, filename, world_size):
    """
    获取分布式环境下的所有缓存文件路径
    
    Args:
        base_path: 基础输出目录路径
        filename: 缓存文件名
        world_size: 分布式环境的进程数量
        
    Returns:
        list: 所有缓存文件的完整路径列表
    """
    all_files = []
    
    # 搜索分布式缓存文件的多种可能路径格式
    search_patterns = [
        # 格式1: base_path/0/filename, base_path/1/filename, ...
        [os.path.join(base_path, str(rank), filename) for rank in range(world_size)],
        # 格式2: base_path/rank0/filename, base_path/rank1/filename, ...
        [os.path.join(base_path, f"rank{rank}", filename) for rank in range(world_size)],
        # 格式3: base_path/*/filename (通配符搜索)
        [os.path.join(base_path, "*", filename)],
    ]
    
    for pattern_list in search_patterns:
        for pattern in pattern_list:
            if "*" in pattern:
                # 使用glob进行通配符匹配
                files = glob.glob(pattern)
                all_files.extend(files)
            else:
                # 直接检查文件是否存在
                if os.path.exists(pattern):
                    all_files.append(pattern)
    
    # 去除重复文件路径
    unique_files = list(set(all_files))
    
    return unique_files

def prepare_distributed_cache_paths(config, module_name):
    """
    准备分布式环境下的缓存文件路径
    
    Args:
        config: 配置字典
        module_name: 模块名称
        
    Returns:
        tuple: (load_result_paths, load_info_paths) 加载结果和信息的路径列表
    """
    # 从config字典中获取模块配置
    active_modules = config.get("ActiveModules", {})
    module_config = active_modules.get(module_name, {})
    world_size = len(config["device"]) if config.get("ddp", False) else 1
    
    load_result_paths = None
    load_info_paths = None
    
    # 检查是否需要加载结果缓存
    if module_config.get("load_result_file") and config.get("cache_dir"):
        # 获取所有分布式缓存文件路径
        load_result_paths = get_distributed_cache_paths(
            config["cache_dir"], 
            module_config["load_result_file"],
            world_size
        )
        if not load_result_paths:
            logger.warning(f"No distributed cache files found for {module_name} results")
            load_result_paths = None
        else:
            logger.info(f"Found {len(load_result_paths)} distributed cache files for {module_name} results")
    
    # 同样处理info缓存
    if module_config.get("load_info_file") and config.get("cache_dir"):
        load_info_paths = get_distributed_cache_paths(
            config["cache_dir"], 
            module_config["load_info_file"],
            world_size
        )
        if not load_info_paths:
            logger.warning(f"No distributed cache files found for {module_name} info")
            load_info_paths = None
        else:
            logger.info(f"Found {len(load_info_paths)} distributed cache files for {module_name} info")
    
    return load_result_paths, load_info_paths