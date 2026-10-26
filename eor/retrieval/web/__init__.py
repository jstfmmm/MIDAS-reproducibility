import json
import time
import math

from tqdm import tqdm

from ..base import BaseRetriever
from .search import serp_api, filter_urls, serper_api
from .fetch import fetch
from .extract import SyntaxExtractor

from loguru import logger


class WebRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)

        self.ssl_verify = config["ssl_verify"]
        self.extractor = SyntaxExtractor(config)
        # 从配置中读取 API key，如果没有则使用默认值
        self.api_key = config.get("api_key", None)
        
        # 打印当前使用的 API key（只显示前8位和后4位）
        if self.api_key:
            masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}"
            logger.info(f"WebRetriever initialized with API key: {masked_key}")
        else:
            logger.info("WebRetriever initialized with default API key")

    def _retrieve(self, query, language, device="gpu0"):
        # 首先检查是否有完整的文档缓存
        if query in self.result_dict:
            if self.log_detail:
                logger.info(f"Loading cached documents for query: '{query}' ({len(self.result_dict[query])} docs)")
            return self.result_dict[query]

        if self.log_detail:
            logger.info(f"begin to search '{query}'.")
            t = time.time()

        url_cache = self.info_dict.get(query, None)
        if url_cache is None:
            self.info_dict[query] = {"search": None, "fetch": None}
            _urls, info = serper_api(query, language, api_key=self.api_key)
            self.info_dict[query]["search"] = info
        else:
            _urls = [item["link"] for item in url_cache["search"]["organic"]]

        urls = filter_urls(_urls)

        if self.log_detail:
            if len(urls) > 0:
                logger.info(f"search {len(urls)} urls. Time cost: {time.time() - t}")
            else:
                logger.warning("return no available url.")
                return []

        # 检查是否有fetch缓存，如果有则直接使用
        if url_cache is not None and url_cache.get("fetch") is not None:
            if self.log_detail:
                logger.info(f"Loading cached extracted documents for query: '{query}'")
            docs = list(url_cache["fetch"].values())
            self.result_dict[query] = docs
            return docs

        if self.log_detail:
            logger.info("begin to fetch.")
            t = time.time()

        fetch_pages = fetch(urls)

        if self.log_detail:
            if len(fetch_pages) > 0:
                logger.info(f"fetch {len(fetch_pages)} pages. Time cost: {time.time() - t}")
            else:
                logger.warning("fetch no available page.")
                return []

        if self.log_detail:
            logger.info("begin to extract.")
            t = time.time()

        docs = []
        self.info_dict[query]["fetch"] = {}
        #self.info_dict[query]["html"] = {}
        for url, text in fetch_pages.items():
            #self.info_dict[query]["html"][url] = text
            doc = self.extractor.extract(language,  text)
            if doc is not None:
                docs.append(doc)
                self.info_dict[query]["fetch"][url] = doc

        if len(docs) == 0:
            for url, text in fetch_pages.items():
                doc = self.extractor.extract(language,  text, loosen=True)
                if doc is not None:
                    docs.append(doc)
                    self.info_dict[query]["fetch"][url] = doc
        
        if self.log_detail:
            if len(docs) > 0:
                logger.info(f"extract {len(docs)} docs. Time cost: {time.time() - t}")
            else:
                logger.warning("extract no available doc.")

        self.result_dict[query] = docs

        return docs
    
    def _batch_retrieve(self, queries, language_list, device="gpu0"):
        # 检查哪些查询已经有缓存
        cached_queries = []
        uncached_queries = []
        uncached_languages = []
        
        for query, language in zip(queries, language_list):
            if query in self.result_dict:
                cached_queries.append(query)
            else:
                uncached_queries.append(query)
                uncached_languages.append(language)
        
        if self.log_detail and len(cached_queries) > 0:
            logger.info(f"Found {len(cached_queries)} cached queries, processing {len(uncached_queries)} new queries")

        # 如果所有查询都有缓存，直接返回
        if len(uncached_queries) == 0:
            return [self.result_dict[query] for query in queries]

        query_urls_dict = {}
        if self.log_detail:
            logger.info(f"begin to search.")
        t = time.time()
        for query, language in tqdm(zip(uncached_queries, uncached_languages)):
            url_cache = self.info_dict.get(query, None)
            if url_cache is None:
                self.info_dict[query] = {"search": None, "fetch": None}
                _urls, info = serper_api(query, language, api_key=self.api_key)
                # _urls, info = serp_api(query, 10)
                time.sleep(2)
                self.info_dict[query]["search"] = info
            else:
                _urls = [item["link"] for item in url_cache["search"]["organic"]]

            urls = filter_urls(_urls)
            # logger.info(f"query: {query} fetch: {len(_urls)}, filtered: {len(urls)}")

            query_urls_dict[query] = urls
            
        total_num = sum([len(urls) for urls in query_urls_dict.values()])
        if self.log_detail:
            if total_num > 0:
                logger.info(f"search {total_num} urls. Time cost: {time.time() - t}")
            else:
                logger.warning("return no available url.")

            logger.info("begin to fetch.")
        t = time.time()

        flat_urls_list = [u for urls in query_urls_dict.values() for u in urls]
        flat_fetch_pages = {}
        
        fetch_part_num = 20
        _num = math.ceil(len(flat_urls_list)/fetch_part_num)
        for i in tqdm(range(_num)):
            flat_fetch_pages.update(fetch(flat_urls_list[i*fetch_part_num:(i+1)*fetch_part_num]))
            time.sleep(2)
            
        if len(flat_fetch_pages) > 0:
            if self.log_detail:
                logger.info(f"fetch {len(flat_fetch_pages)} pages. Time cost: {time.time() - t}")
        else:
            if self.log_detail:
                logger.warning("fetch no available page.")
            # 对于没有获取到页面的情况，返回已缓存的结果加上空列表
            result = []
            for query in queries:
                if query in self.result_dict:
                    result.append(self.result_dict[query])
                else:
                    result.append([])
            return result

        docs_list = []
        if self.log_detail:
            logger.info("begin to extract.")
        t = time.time()
        for query, language in tqdm(zip(uncached_queries, uncached_languages)):
            # 检查是否有fetch缓存
            url_cache = self.info_dict.get(query, None)
            if url_cache is not None and url_cache.get("fetch") is not None:
                docs = list(url_cache["fetch"].values())
                self.result_dict[query] = docs
                docs_list.append(docs)
                continue

            docs = []
            self.info_dict[query]["fetch"] = {}
            self.info_dict[query]["html"] = {}
            fetch_pages = {url: flat_fetch_pages[url] for url in query_urls_dict[query] if url in flat_fetch_pages}
            
            for url, text in fetch_pages.items():
                # self.info_dict[query]["html"][url] = text
                doc = self.extractor.extract(language,  text)
                if doc is not None:
                    docs.append(doc)
                    self.info_dict[query]["fetch"][url] = doc

            if len(docs) == 0:
                for url, text in fetch_pages.items():
                    doc = self.extractor.extract(language,  text, loosen=True)
                    if doc is not None:
                        docs.append(doc)
                        self.info_dict[query]["fetch"][url] = doc

            self.result_dict[query] = docs
            docs_list.append(docs)

        total_num = sum([len(docs) for docs in docs_list])
        if self.log_detail:
            if total_num > 0:
                logger.info(f"extract {total_num} docs. Time cost: {time.time() - t}")
            else:
                logger.warning("extract no available doc.")
        
        # 按原始查询顺序返回结果
        result = []
        for query in queries:
            result.append(self.result_dict[query])
        
        return result