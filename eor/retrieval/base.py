import json
import os

from loguru import logger
from tqdm import tqdm


class BaseRetriever:
    def __init__(self, config):
        self.result_dict = {}
        self.info_dict = {}
                

        self.log_detail = config["verbose"] if config["verbose"] is not None else True       
        self.save_result = config["save_result_path"] is not None
        self.save_info = config["save_info_path"] is not None
        self.save_result_path = config["save_result_path"]
        self.save_info_path = config["save_info_path"]
        
        # 检查是否启用分布式缓存加载
        self.load_distributed_cache = config.get("load_distributed_cache", False)
        
        if config["load_result_path"] is not None:
            if self.log_detail:
                logger.warning(f"[Load docs from {config['load_result_path']}]")

            # 如果启用分布式缓存，尝试加载分布式缓存文件
            if self.load_distributed_cache and hasattr(config, 'distributed_cache_paths'):
                self.load_distributed_result(config['distributed_cache_paths'])
            else:
                self.load_result(config["load_result_path"])
        
        if config["load_info_path"] is not None :
            if self.log_detail:
                logger.warning(f"[Load info from {config['load_info_path']}]")
            
            # 如果启用分布式缓存，尝试加载分布式缓存文件
            if self.load_distributed_cache and hasattr(config, 'distributed_info_paths'):
                self.load_distributed_info(config['distributed_info_paths'])
            else:
                self.load_info(config["load_info_path"])

    def retrieve(self, query, language, device="gpu0"):
        return self.result_dict[query] if query in self.result_dict.keys() else self._retrieve(query, language, device)
    
    def batch_retrieve(self, queries, language_list, device="gpu0", world_size=1):
        return self._batch_retrieve(queries, language_list, device) if len(self.result_dict) == 0 else self._batch_retrieve_use_cache(queries, language_list, device, world_size=world_size)
    
    def load_result(self, path):
        self.result_dict = json.load(open(path, "r", encoding="UTF-8"))

    def load_info(self, path):
        self.info_dict = json.load(open(path, "r", encoding="UTF-8"))
    
    def load_distributed_result(self, paths):
        """加载分布式缓存结果文件"""
        merged_dict = {}
        loaded_count = 0
        
        for path in paths:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="UTF-8") as f:
                        part_result = json.load(f)
                        merged_dict.update(part_result)
                        loaded_count += 1
                    if self.log_detail:
                        logger.info(f"Loaded distributed cache from: {path}")
            except Exception as e:
                if self.log_detail:
                    logger.warning(f"Failed to load cache from {path}: {e}")
        
        self.result_dict = merged_dict
        if self.log_detail:
            logger.info(f"Successfully merged {loaded_count} distributed cache files with {len(merged_dict)} total entries")
    
    def load_distributed_info(self, paths):
        """加载分布式缓存信息文件"""
        merged_dict = {}
        loaded_count = 0
        
        for path in paths:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="UTF-8") as f:
                        part_result = json.load(f)
                        merged_dict.update(part_result)
                        loaded_count += 1
                    if self.log_detail:
                        logger.info(f"Loaded distributed info cache from: {path}")
            except Exception as e:
                if self.log_detail:
                    logger.warning(f"Failed to load info cache from {path}: {e}")
        
        self.info_dict = merged_dict
        if self.log_detail:
            logger.info(f"Successfully merged {loaded_count} distributed info cache files with {len(merged_dict)} total entries")
    
    def save_if_set(self):
        if self.save_result:
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        if self.save_info:
            json.dump(self.info_dict, open(self.save_info_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
    
    def _retrieve(self, query, language, device="gpu0"):
        pass

    def _batch_retrieve(self, queries, language_list, device="gpu0"):
        if not self.log_detail:
            return [self._retrieve(query, language, device) for query, language in zip(queries, language_list)]
        else:
            return [self._retrieve(queries[index], language_list[index], device) for index in tqdm(range(len(queries)))]
    
    def _batch_retrieve_use_cache(self, queries, language_list, device="gpu0", world_size=1):
        cache = set(self.result_dict.keys())
        no_cache_queries = [query for query in queries if query not in cache]
        no_cache_language_list = [language for query, language in zip(queries, language_list) if query not in cache]
        if len(no_cache_queries) > 0:
            self._batch_retrieve(no_cache_queries, no_cache_language_list, device)
        return [self.result_dict[query] for query in queries]