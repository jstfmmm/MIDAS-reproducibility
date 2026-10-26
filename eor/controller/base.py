from itertools import chain
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import json
import os
import re
import time
import datetime
from ..retrieval import WebRetriever, WikiRetriever, GenerationRetriever
from ..retrieval.maud_bm25 import MAUDBm25Retriever
from ..retrieval.sciq_bm25 import SciQBm25Retriever
from ..knowledge import ContrieverConstructor, SummarizeConstructor
from transformers import LlamaTokenizer
import numpy as np

from ..utils import truncate_en_doc, truncate_zh_doc

# 定义不同检索方法所需的模块映射
# 每种检索方法需要使用不同的检索器、筛选器和摘要器
METHOD_MODULE_MAPPING = {
    "google_best": {
        "retriever":["Web"],        # 使用Web检索器获取最佳Google结果
        "summarizer": ["Summarizer"] # 使用摘要器对结果进行摘要
    },
    "wiki": {
        "retriever":["Wiki"],        # 使用Wiki检索器获取百科知识
        "summarizer": ["Summarizer"]
    },
    "gendoc":{
        "retriever":["Gendoc"],      # 使用生成式文档检索器
        "summarizer": ["Summarizer"]
    },
    "google_contrieve":{
        "retriever":["Web"],         # 使用Web检索器获取Google结果
        "contriever": ["Contriever"], # 使用Contriever对结果进行筛选
        "summarizer": ["Summarizer"]
    },
    "google_merge": {
        "retriever":["Web"],         # 合并多个Google结果
        "summarizer": ["Summarizer"]
    },
    "google_merge+wiki": {
        "retriever":["Web", "Wiki"],  # 合并Google和Wiki结果
        "summarizer": ["Summarizer"]
    },
    "google_contrieve+wiki": {
        "retriever":["Web", "Wiki"],  # 使用Google和Wiki结果
        "contriever": ["Contriever"],  # 使用Contriever筛选
        "summarizer": ["Summarizer"]
    },
    "contrieve_all_sources":{
        "retriever":["Web", "Wiki", "Gendoc"], # 使用所有来源
        "contriever": ["Contriever"],          # 使用Contriever筛选所有来源
        "summarizer": ["Summarizer"]
    },
    # MAUD 法律文档检索方法
    "maud_bm25_top1": {
        "retriever": ["MAUDBm25"],
        "summarizer": ["Summarizer"]
    },
    "maud_bm25_top5": {
        "retriever": ["MAUDBm25"],
        "summarizer": ["Summarizer"]
    },
    "maud_bm25_top5_summary": {
        "retriever": ["MAUDBm25"],
        "summarizer": ["Summarizer"]
    },
    # SciQ support corpus BM25 检索方法
    "sciq_bm25_top1": {
        "retriever": ["SciQBm25"],
        "summarizer": ["Summarizer"]
    },
    "sciq_bm25_top5": {
        "retriever": ["SciQBm25"],
        "summarizer": ["Summarizer"]
    },
    "sciq_bm25_top5_summary": {
        "retriever": ["SciQBm25"],
        "summarizer": ["Summarizer"]
    },
}

class ControllerConstructor:
    """
    控制器构造器类：负责协调多种检索方法、处理和构造知识。
    该类是整个检索系统的核心，管理不同检索器、筛选器和摘要器的工作流程。
    """

    def __init__(self, config, device, world_size = 1, rank=None):
        """
        初始化控制器构造器
        
        参数:
            config: 配置对象，包含所有模块和方法的配置信息
            device: 运行设备(cpu或gpu)
            world_size: 分布式训练的世界大小
            rank: 当前进程的rank
        """
        self.verbose = config["log_detail"]
        self.active_methods = config.active_methods   # 激活的检索方法
        self.modules = config.active_modules          # 激活的模块
        self.rank = rank
        self.config=config
        self.device = device
        self.ddp = config["ddp"]                      # 是否启用分布式训练
        self.world_size = world_size

        # 获取所需的各类模块列表
        self.required_retrieval_modules = sorted(list(config.retrieval_modules), reverse=True)
        self.required_contriever_modules = sorted(list(config.contriever_modules))
        self.required_summarizer_modules = sorted(list(config.summarizer_modules))

        self.max_knowledge_length = config["max_knowledge_length"]  # 最大知识长度限制

        # 初始化结果字典
        self.result_dict = dict()
        self.result_dict["active_methods"] = self.active_methods
        self.result_dict["processed_knowledge"] = dict()
        
        # 设置保存路径
        if config["output_dir"] is not None:
            self.save_result = True
            if "$rank$" in config["output_dir"] and rank is not None:
                self.save_result_path = os.path.join(config["output_dir"].replace("$rank$", str(rank)), "controller_result.json")
            else: 
                self.save_result_path = os.path.join(config["output_dir"], "controller_result.json")
        else:
            # 如果没有设置输出目录，禁用保存功能
            self.save_result = False
            self.save_result_path = None

        # 是否从缓存加载结果
        self.from_cache = getattr(config, 'load_controller_from_cache', False)
        if self.from_cache:
            self.load_result_path = os.path.join(config["cache_dir"], "controller_result.json")

        # 设置日志
        log_name = config.get("log_name", config["time"]) + ".log"
        logger.add(os.path.join("log", log_name), level='INFO')
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level='INFO')

        
    def batch_process(self, queries, language_list):
        """
        批处理查询，为每个查询检索和处理知识
        
        参数:
            queries: 查询列表
            language_list: 查询对应的语言列表
            
        返回:
            处理后的知识字典，格式为 {方法名: [处理后的知识列表]}
        """
        if self.active_methods == dict():
            if self.rank == 0 or not self.ddp:
                logger.info("Controller do not retrieve and process any knowledge!")
            return None
        
        # 如果使用缓存，先加载缓存结果
        if self.from_cache:
            result_dict = json.load(open(self.load_result_path, "r", encoding="UTF-8"))
            
            if self.rank == 0 or not self.ddp:
                logger.info(f"load cached data from '{self.load_result_path}'")
            self.result_dict = result_dict
            
            # 找出没有缓存的查询，需要重新处理
            cached = set(result_dict["processed_knowledge"].keys())
            no_cache_queries = [query for query in queries if query not in cached]
            if self.verbose:
                logger.info(f"{self.device}:{len(queries) - len(no_cache_queries)} cached queries found, {len(no_cache_queries)} more queries to be processed.")
            
            no_cache_language_list = [language for query, language in zip(queries, language_list) if query not in cached]
            
            # 处理未缓存的查询
            if len(no_cache_queries) > 0:
                _ = self._batch_process(no_cache_queries, no_cache_language_list)
                
            # 重组结果为每个方法对应一个知识列表
            processed_knowledge = defaultdict(list)
            for q in queries:
                for m_name, m_doc in self.result_dict["processed_knowledge"][q].items():
                    processed_knowledge[m_name].append(m_doc)
            return processed_knowledge
            
        else:
            # 不使用缓存，从头开始处理
            if self.rank == 0 or not self.ddp:
                if self.verbose:
                    logger.info(f"No cached file found for controller. Start to retrieve and process knowledge from scratch.")
            return self._batch_process(queries, language_list)
                
            
    
    def _batch_process(self, queries, language_list):
        """
        实际的批处理实现，包含完整的检索-筛选-摘要流程
        
        参数:
            queries: 查询列表
            language_list: 查询对应的语言列表
            
        返回:
            处理后的知识字典
        """
        query_num = len(queries)
        
        # 开始检索阶段
        if self.rank == 0 or not self.ddp:
            if self.verbose:
                logger.info(f"************Start to retrieval, modules used to retrieve are {'|'.join(self.required_retrieval_modules)}************")
        
        # 调用批量检索方法获取原始文档
        retrieval_doc_list = self._batch_retrieve(queries, language_list)

        # 用于存储各阶段处理的知识
        processed_knowledge = dict()

        # 各种索引和映射，用于跟踪不同方法和模块处理的文档
        retriever_processed = dict()
        retrieval_index = {name: self.build_retrieval_method_index(method) for name, method in self.active_methods.items()}
        contriever_index = dict()
        postprocess_index = dict()
        summarizer_index = dict()
        docs_to_contrieve = defaultdict(set) 
        docs_to_summarize = defaultdict(list)

        # 处理检索结果，并确定哪些文档需要Contriever筛选和摘要
        for name, method in self.active_methods.items():
            
            # 处理检索结果（如果同类检索方法已处理，复用结果）
            if retrieval_index[name] not in retriever_processed.keys():
                retriever_processed[retrieval_index[name]] = self._process_retrieved_docs(method, language_list, retrieval_doc_list)
            
            processed_knowledge[name] = retriever_processed[retrieval_index[name]]
            
            # 如果方法需要Contriever，添加到筛选列表
            if method["contriever"] is not None:
                contriever_index[name] = retrieval_index[name]+"-"+method["contriever"][0]
                docs_to_contrieve[method["contriever"][0]].add(retrieval_index[name])
            
            # 特殊处理google_contrieve+wiki方法
            if method["method_name"] == "google_contrieve+wiki":
                postprocess_index[name] = {
                    "method_name": "google_contrieve+wiki",
                    "contriever": contriever_index[name],
                    "retriever": self.get_module_names(method, "Wiki")
                }
            
            # 如果方法需要摘要，添加到摘要列表
            if method["summarizer"] is not None:
                docs_to_summarize[method["summarizer"][0]].append(name)

        # 将set转换为list以便序列化
        docs_to_contrieve = {k: list(v) for k,v in docs_to_contrieve.items()}

        # 开始Contriever筛选阶段
        if self.rank == 0 or not self.ddp:
            if self.verbose:
                logger.info(f"************Start to contrieve, modules used to contrieve are '{'|'.join(list(self.required_contriever_modules))}'************")
        
        contriever_processed = dict()
        begin_time = time.time()
        
        # 对每个Contriever模块处理相应的文档
        for contriever_name, doc_names in docs_to_contrieve.items():
            Contriever = ContrieverConstructor(self.modules[contriever_name])
            doc_type = len(doc_names)
            # 合并不同检索方法的文档
            docs = list(chain(*[retriever_processed[n] for n in doc_names]))
            qs = queries * doc_type
            ls = language_list * doc_type
            # qs = queries
            # ls = language_list
            # 批量筛选文档
            knowledges = Contriever.batch_construct(qs, ls, docs, device=self.device, top_k=10, world_size=self.world_size)
            Contriever.save_if_set()
            
            # 将结果分配给对应的检索方法
            for idx, r_name in enumerate(doc_names):
                c_idx = r_name + "-" + contriever_name
                assert c_idx not in contriever_processed.keys(), "something is wrong"
                contriever_processed[c_idx] = knowledges[query_num*idx : query_num*(idx+1)]
            del Contriever
        
        # 更新处理后的知识
        for name, c_index in contriever_index.items():
            processed_knowledge[name] = contriever_processed[c_index]
        
        if self.verbose:
            logger.info(f"********************{self.device}: contriever finished, time used {datetime.timedelta(seconds = time.time()-begin_time)}")

        # 后处理阶段 - 特殊处理需要组合多种来源的方法
        for name, p_index in postprocess_index.items():
            contriever_docs = [d.split("\n") for d in contriever_processed[p_index["contriever"]]]
            retrieval_docs = retrieval_doc_list[p_index["retriever"]]
            processed_knowledge[name] = self._postprocess_docs(p_index["method_name"],contriever_docs, retrieval_docs, language_list)
        
        # 截断所有知识到最大长度
        truncate_doc = {
            "en": truncate_en_doc,
            "zh": truncate_zh_doc
        }
        if self.rank == 0 or not self.ddp:
            if self.verbose:
                logger.info(f"************Start to truncate long knowledge to maximum knowledge length {self.max_knowledge_length}************")

        # 获取合适的分词器用于文本长度控制
        if self.config["tokenizer"] == "llama":
            tokenizer = LlamaTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf")
        elif self.config["tokenizer"] is None or self.config["tokenizer"] == "nltk":
            tokenizer = self.config["tokenizer"]
        else:
            raise KeyError("tokenizer should be one of None, llama, nltk.")
        
        # 截断所有知识文档
        truncated_knowledge = dict()
        for n, k in processed_knowledge.items():
            new_k = []
            for d,lang in zip(k, language_list):
                new_k.append(truncate_doc[lang](" ".join(re.sub(r'[\n]+', '\n', d).split()), max_doc_len = self.max_knowledge_length, min_doc_len = 0, tokenizer=tokenizer))
            truncated_knowledge[n] = new_k
        
        # 开始摘要阶段
        if self.verbose:
            logger.info(f"************{self.device}: Start to summarize, modules used to summarize are '{'|'.join(list(self.required_summarizer_modules))}'************")
        
        begin_time = time.time()
        # 对每个摘要模块处理相应的文档
        for summarizer_name, methods_to_sum in docs_to_summarize.items():
            Summarizer = SummarizeConstructor(self.modules[summarizer_name])
            doc_type = len(methods_to_sum)
            docs = list(chain(*[truncated_knowledge[n] for n in methods_to_sum]))
            qs = queries * doc_type
            ls = language_list * doc_type
            assert len(docs) == doc_type * query_num, "something wrong"
            
            # 批量生成摘要
            knowledges = Summarizer.batch_construct(qs, ls, docs, device=self.device, world_size=self.world_size)
            Summarizer.save_if_set()
            
            # 将结果分配给对应的方法
            for idx, m_name in enumerate(methods_to_sum):
                truncated_knowledge[m_name] = knowledges[query_num*idx : query_num*(idx+1)]

            del Summarizer
        if self.verbose:
            logger.info(f"********************{self.device}: summarization finished, time used {datetime.timedelta(seconds = time.time()-begin_time)}")

        # 重组结果为每个查询一个字典
        for idx, query in enumerate(queries):
            collected_k = dict()
            for m_name, ds in truncated_knowledge.items():
                collected_k[m_name] = ds[idx]
            self.result_dict["processed_knowledge"][query] = collected_k

        return truncated_knowledge            

    
    def _batch_retrieve(self, queries, language_list):
        """
        批量检索文档
        
        参数:
            queries: 查询列表
            language_list: 语言列表
            
        返回:
            检索结果字典 {模块名: 文档列表}
        """
        retrieval_doc_list = dict()
        begin_time = time.time()
        
        # 对每个检索模块执行检索
        for r_name in self.required_retrieval_modules:
            # Web检索 - 使用搜索引擎获取结果
            if self.modules[r_name]["type"] == "Web":
                if self.verbose:
                    logger.info(f"************{self.device}: Start Web Retriever: {r_name}************")
                retriever = WebRetriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries,language_list)
                # 过滤掉平均句长太短且总长度较长的文档（可能是格式混乱的内容）
                doc_list = [[d for d in docs if not (np.mean([len(s.split()) for s in re.split(r"([.!?;,])", d)]) < 1.5 and len(d.split()) > 30)] for docs in doc_list]
                if self.verbose:
                    logger.info(f"************{self.device}: Web Retriever Finished************")
                
            # Wiki检索 - 使用百科知识库检索
            elif self.modules[r_name]["type"] == "Wiki":
                if self.verbose:
                    logger.info(f"************{self.device}: Start Wiki Retriever: {r_name}************")
                retriever = WikiRetriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries, language_list)
                if self.verbose:
                    logger.info(f"************{self.device}: Wiki Retriever Finished************")

            # 生成式文档检索 - 使用模型生成相关文档
            elif self.modules[r_name]["type"] == "Gendoc":
                if self.verbose:
                    logger.info(f"************{self.device}: Start Gendoc Retriever: {r_name}************")
                retriever = GenerationRetriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries, language_list, self.device, world_size=self.world_size)
                if isinstance(doc_list[0], list):
                    doc_list = ["\n".join(d) for d in doc_list]
                if self.verbose:
                    logger.info(f"************{self.device}: Gendoc Retriever Finished************")

            # MAUD BM25检索 - 使用BM25索引检索法律文档
            elif self.modules[r_name]["type"] == "MAUDBm25":
                if self.verbose:
                    logger.info(f"************{self.device}: Start MAUD BM25 Retriever: {r_name}************")
                retriever = MAUDBm25Retriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries, language_list)
                if self.verbose:
                    logger.info(f"************{self.device}: MAUD BM25 Retriever Finished************")

            # SciQ BM25检索 - 使用BM25索引检索support语料库
            elif self.modules[r_name]["type"] == "SciQBm25":
                if self.verbose:
                    logger.info(f"************{self.device}: Start SciQ BM25 Retriever: {r_name}************")
                retriever = SciQBm25Retriever(self.modules[r_name])
                doc_list = retriever.batch_retrieve(queries, language_list)
                if self.verbose:
                    logger.info(f"************{self.device}: SciQ BM25 Retriever Finished************")

            else:
                raise TypeError(f"retrieval modules should be one of Web, Wiki, Gendoc, MAUDBm25 or SciQBm25, but '{r_name}' is given")

            # 保存检索结果
            retriever.save_if_set()
            retrieval_doc_list[r_name] = doc_list

            del retriever
        if self.verbose:
            logger.info(f"********************{self.device}: Retrieval finished, time used {datetime.timedelta(seconds = time.time()-begin_time)}********************")

        return retrieval_doc_list
    
    def _process_retrieved_docs(self, config, language_list, retrieved_docs):
        """
        处理检索到的文档，根据不同方法执行不同的处理策略
        
        参数:
            config: 方法配置
            language_list: 语言列表
            retrieved_docs: 检索到的文档
            
        返回:
            处理后的文档列表
        """
        truncate_doc = {
            "en": truncate_en_doc,
            "zh": truncate_zh_doc
        }

        # 处理google_best方法 - 选择最佳搜索结果
        if config["method_name"] == "google_best":
            web_module_name = self.get_module_names(config, "Web")
            # 取每个查询的第一个搜索结果，截断到1000个token
            return [truncate_doc[lang](doc_list[0], 1000, 0) if len(doc_list) > 0 else "" for lang, doc_list in zip(language_list, retrieved_docs[web_module_name])]
        
        # 处理wiki方法 - 使用维基百科检索结果
        elif config["method_name"] == "wiki":
            wiki_module_name = self.get_module_names(config, "Wiki")
            # 取每个查询的前10个Wiki结果并合并
            return ["\n".join(doc_list[:10]) for doc_list in retrieved_docs[wiki_module_name]]
        
        # 处理gendoc方法 - 使用生成式文档
        elif config["method_name"] == "gendoc":
            gen_module_name = self.get_module_names(config, "Gendoc")
            # 直接使用生成的文档
            return retrieved_docs[gen_module_name]
        
        # 处理google_contrieve和google_contrieve+wiki方法 - 合并所有搜索结果供Contriever筛选
        elif config["method_name"] == "google_contrieve" or config["method_name"] == "google_contrieve+wiki":
            web_module_name = self.get_module_names(config, "Web")
            # 合并所有搜索结果
            return ["\n".join(doc_list) for doc_list in retrieved_docs[web_module_name]]
        
        # 处理google_merge方法 - 合并前4个搜索结果，各截断到250个token
        elif config["method_name"] == "google_merge":
            web_module_name = self.get_module_names(config, "Web")
            docs = retrieved_docs[web_module_name]
            new_docs = []
            for doc, lang in zip(docs, language_list):
                new_docs.append("\n".join([truncate_doc[lang](d, 250) for d in doc[:4]]))
            return new_docs
        
        # 处理google_merge+wiki方法 - 合并前2个搜索结果和前5个Wiki结果
        elif config["method_name"] == "google_merge+wiki":
            web_module_name = self.get_module_names(config, "Web")
            wiki_module_name = self.get_module_names(config, "Wiki")
        
            web_docs = retrieved_docs[web_module_name]
            wiki_docs = retrieved_docs[wiki_module_name]
            new_docs = []
            for web, wiki, lang in zip(web_docs, wiki_docs, language_list):
                new_docs.append("\n".join([truncate_doc[lang](d, 250) for d in web[:2]]) + "\n" + "\n".join(wiki[:5]))
            return new_docs
        
        # 处理MAUD BM25方法 - 使用BM25索引检索法律文档
        elif config["method_name"] == "maud_bm25_top1":
            bm25_module_name = self.get_module_names(config, "MAUDBm25")
            # 取第一个检索结果（类似 google_best）
            return [doc_list[0] if len(doc_list) > 0 else "" for doc_list in retrieved_docs[bm25_module_name]]

        elif config["method_name"] == "maud_bm25_top5":
            bm25_module_name = self.get_module_names(config, "MAUDBm25")
            # 合并前5个检索结果（类似 google_merge）
            return ["\n".join(doc_list[:5]) for doc_list in retrieved_docs[bm25_module_name]]

        elif config["method_name"] == "maud_bm25_top5_summary":
            bm25_module_name = self.get_module_names(config, "MAUDBm25")
            # 合并前5个检索结果（summary在后续处理）
            return ["\n".join(doc_list[:5]) for doc_list in retrieved_docs[bm25_module_name]]

        # 处理SciQ support corpus BM25方法
        elif config["method_name"] == "sciq_bm25_top1":
            bm25_module_name = self.get_module_names(config, "SciQBm25")
            return [doc_list[0] if len(doc_list) > 0 else "" for doc_list in retrieved_docs[bm25_module_name]]

        elif config["method_name"] == "sciq_bm25_top5":
            bm25_module_name = self.get_module_names(config, "SciQBm25")
            return ["\n".join(doc_list[:5]) for doc_list in retrieved_docs[bm25_module_name]]

        elif config["method_name"] == "sciq_bm25_top5_summary":
            bm25_module_name = self.get_module_names(config, "SciQBm25")
            return ["\n".join(doc_list[:5]) for doc_list in retrieved_docs[bm25_module_name]]

        # 处理contrieve_all_sources方法 - 合并Web、Wiki和Gendoc的所有结果
        elif config["method_name"] == "contrieve_all_sources":
            web_module_name = self.get_module_names(config, "Web")
            wiki_module_name = self.get_module_names(config, "Wiki")
            gen_module_name = self.get_module_names(config, "Gendoc")
            web_docs = retrieved_docs[web_module_name]
            wiki_docs = retrieved_docs[wiki_module_name]
            gen_docs = retrieved_docs[gen_module_name]
            new_docs = []
            for web, wiki, gen in zip(web_docs, wiki_docs, gen_docs):
                new_docs.append("\n".join(web) + "\n" + "\n".join(wiki) + "\n" + gen)
            return new_docs
        else:
            raise TypeError(f"method {config['method_name']} is not supported.")
    
    def _postprocess_docs(self, method_name, contrieved_docs, retrieved_docs, language_list):
        """
        对Contriever筛选后的文档进行后处理
        
        参数:
            method_name: 方法名称
            contrieved_docs: Contriever筛选后的文档
            retrieved_docs: 原始检索文档
            language_list: 语言列表
            
        返回:
            后处理的文档列表
        """
        truncate_doc = {
            "en": truncate_en_doc,
            "zh": truncate_zh_doc
        }
        # 处理google_contrieve+wiki方法 - 合并前5个Contriever筛选结果和前5个Wiki结果
        if method_name == "google_contrieve+wiki":
            post_docs = []
            for c_d, r_d, lang in zip(contrieved_docs, retrieved_docs, language_list):
                post_docs.append(truncate_doc[lang]("\n".join(c_d[:5]), 500) + "\n" + truncate_doc[lang]("\n".join(r_d[:5]), 500))
            
            return post_docs
        else:
            raise TypeError(f"{method_name} is not supported for postprocessing")

    def build_retrieval_method_index(self, config):
        """
        构建检索方法索引，用于标识处理过的检索文档
        
        参数:
            config: 方法配置
            
        返回:
            索引字符串，格式为 {method_name}-"-".join({used_retriever_module_names})
        """
        if config["method_name"] == "google_contrieve+wiki":
            return "google_contrieve" + "-" + self.get_module_names(config, "Web")
        else:
            return config["method_name"] + "-" + "-".join(sorted(config["retriever"]))
        
    
    def get_module_names(self, method_config, module_type):
        """
        获取给定方法配置和模块类型的模块名称
        
        参数:
            method_config: 方法配置
            module_type: 模块类型(Web, Wiki, Gendoc, Contriever, Summarizer)
            
        返回:
            对应的模块名称
            
        示例:
        config:
        {
        "method_name": "gendoc",
        "retriever": ["Gendoc_1"]
        }
        当给定type="Gendoc"时，将返回"Gendoc_1"
        """
        if module_type in ["Web", "Wiki", "Gendoc"]:
            module_name = [n for n in method_config["retriever"] if self.modules[n]["type"]==module_type]
        elif module_type == "Contriever":
            module_name = [n for n in method_config["contriever"] if self.modules[n]["type"]==module_type]
        elif module_type == "Summarizer":
            module_name = [n for n in method_config["summarizer"] if self.modules[n]["type"]==module_type]
        else:
            raise TypeError(f"given module type '{module_type} is not supported")
        
        return module_name[0]

    def save_if_set(self):
        """
        如果设置了保存路径，将结果保存到文件
        """
        if self.save_result and self.save_result_path is not None:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.save_result_path), exist_ok=True)
            json.dump(self.result_dict, open(self.save_result_path, "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
            if self.verbose:
                logger.info(f"Controller results saved to: {self.save_result_path}")
        elif self.verbose:
            logger.info("Controller results not saved (no output directory configured)")
    
    def load_from_cache(self, queries, result_dict, knowledge_names):
        """
        从缓存加载处理过的知识
        
        参数:
            queries: 查询列表
            result_dict: 缓存的结果字典
            knowledge_names: 知识名称列表
            
        返回:
            处理后的知识字典
        """
        processed_knowledge = defaultdict(list)
        for q in queries:
            for m_name in knowledge_names:
                if m_name == "query_only":
                    continue
                processed_knowledge[m_name].append(result_dict["processed_knowledge"][q][m_name])
        return processed_knowledge





    
    
    