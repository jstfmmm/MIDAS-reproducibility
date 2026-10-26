import json
from loguru import logger
from tqdm import tqdm
from ..utils import PromptTemplate, LLM

zh_relevance_template = {
    1: "问题：{query}\n\n文档：{doc}\n\n请判断上述文档是否包含回答该问题所需的有用信息。请只回答'yes'或'no'。\n"
}

en_relevance_template = {
    1: "Question: {query}\n\nDocument: {doc}\n\nPlease determine if the above document contains useful information needed to answer this question. Please answer only 'yes' or 'no'.\n"
}

zh_system_message = "你是一个文档相关性判断助手。你的任务是根据给定的文档，判断这份文档中是否包含回答给定问题所需的有用信息。"
en_system_message = "You are a document relevance judge. Your task is to determine if the given document contains useful information needed to answer the given question."

class DocumentRelevanceJudger:
    """
    文档相关性判断器，用于判断检索到的文档是否对回答给定问题有用
    """

    def __init__(self, config):
        """
        初始化文档相关性判断器

        Args:
            config: 配置字典，包含以下内容：
                - zh_model_name: 中文模型名称
                - en_model_name: 英文模型名称
                - zh_template_id: 中文模板ID
                - en_template_id: 英文模板ID
                - verbose: 是否显示详细信息
                - device_name: 设备名称（如'gpu0'）
                - generate_kwargs: 生成参数
        """
        self.config = config
        self.verbose = config.get("verbose", True)
        self.device = config.get("device_name", "gpu0")

        # 控制台打印：初始化信息
        if self.verbose:
            logger.info("=" * 80)
            logger.info("DocumentRelevanceJudger 初始化")
            logger.info(f"  模型名称: {config.get('model_name', '未指定')}")
            logger.info(f"  中文模型: {config.get('zh_model_name', '未指定')}")
            logger.info(f"  英文模型: {config.get('en_model_name', '未指定')}")
            logger.info(f"  设备: {self.device}")
            logger.info(f"  详细日志: {self.verbose}")
            logger.info("=" * 80)

        # 构建提示模板
        self.zh_prompt_template, self.en_prompt_template = self._build_prompt_templates(config)

    def _build_prompt_templates(self, config):
        """
        构建中英文提示模板

        Args:
            config: 配置字典

        Returns:
            zh_prompt_template: 中文提示模板
            en_prompt_template: 英文提示模板
        """
        # 获取模板ID，默认为1
        zh_template_id = config.get("zh_relevance_template_id", 1)
        en_template_id = config.get("en_relevance_template_id", 1)

        # 获取模型名称
        zh_model_name = config.get("zh_model_name", config.get("model_name", "gpt-3.5-turbo"))
        en_model_name = config.get("en_model_name", config.get("model_name", "gpt-3.5-turbo"))

        # 创建中文提示模板
        zh_prompt_template = PromptTemplate(
            language="zh",
            model_name=zh_model_name,
            template=zh_relevance_template[zh_template_id],
            system_message=zh_system_message,
            task_name="relevance_judgment",
            template_id=zh_template_id,
            use_system_message=True
        )

        # 创建英文提示模板
        en_prompt_template = PromptTemplate(
            language="en",
            model_name=en_model_name,
            template=en_relevance_template[en_template_id],
            system_message=en_system_message,
            task_name="relevance_judgment",
            template_id=en_template_id,
            use_system_message=True
        )

        return zh_prompt_template, en_prompt_template

    def judge_single(self, query, doc, language="en"):
        """
        判断单个文档是否与问题相关

        Args:
            query: 问题字符串
            doc: 文档字符串
            language: 语言（'zh' 或 'en'）

        Returns:
            bool: True表示文档相关，False表示不相关
        """
        if language == "zh":
            prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "doc": doc})
            model_name = self.config.get("zh_model_name", self.config.get("model_name", "gpt-3.5-turbo"))
        else:
            prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "doc": doc})
            model_name = self.config.get("en_model_name", self.config.get("model_name", "gpt-3.5-turbo"))

        # 准备生成参数
        kwargs = self.config.copy()
        kwargs["prompts"] = [prompt]
        kwargs["model_name"] = model_name
        kwargs["device_name"] = self.device

        # 调用LLM生成判断结果
        judgment = LLM.lm_generate(**kwargs)[0]

        # 解析判断结果
        return self._parse_judgment(judgment, doc)

    def judge_batch(self, queries, doc_lists, language_list):
        """
        批量判断文档相关性

        Args:
            queries: 问题列表
            doc_lists: 文档列表的列表（每个问题对应一个文档列表）
            language_list: 语言列表

        Returns:
            filtered_doc_lists: 过滤后的文档列表（只保留相关的文档）
        """
        if self.verbose:
            logger.info(f"Starting document relevance judgment for {len(queries)} queries")

        # 构建所有待判断的 (query, doc, language) 组合
        all_prompts = []
        all_infos = []
        all_model_names = []
        query_doc_map = []  # 记录每个prompt对应的query索引和doc索引

        for query_idx, (query, docs, language) in enumerate(zip(queries, doc_lists, language_list)):
            for doc_idx, doc in enumerate(docs):
                if language == "zh":
                    prompt, fill_info = self.zh_prompt_template.build_prompt({"query": query, "doc": doc})
                    model_name = self.config.get("zh_model_name", self.config.get("model_name", "gpt-3.5-turbo"))
                else:
                    prompt, fill_info = self.en_prompt_template.build_prompt({"query": query, "doc": doc})
                    model_name = self.config.get("en_model_name", self.config.get("model_name", "gpt-3.5-turbo"))

                all_prompts.append(prompt)
                all_infos.append(fill_info)
                all_model_names.append(model_name)
                query_doc_map.append((query_idx, doc_idx))

        if len(all_prompts) == 0:
            return doc_lists

        # 准备生成参数
        kwargs = self.config.copy()
        kwargs["prompts"] = all_prompts
        kwargs["device_name"] = self.device

        # 需要处理不同的模型（可能需要分批处理）
        # 这里假设所有使用相同的模型，或者LLM.lm_generate能自动处理
        unique_models = list(set(all_model_names))
        if len(unique_models) == 1:
            kwargs["model_name"] = unique_models[0]
            judgments = LLM.lm_generate(**kwargs)
        else:
            # 如果模型不同，需要分批处理
            judgments = []
            for i, (prompt, model_name) in enumerate(zip(all_prompts, all_model_names)):
                kwargs_batch = kwargs.copy()
                kwargs_batch["prompts"] = [prompt]
                kwargs_batch["model_name"] = model_name
                result = LLM.lm_generate(**kwargs_batch)[0]
                judgments.append(result)

        # 解析判断结果
        filtered_doc_lists = [[] for _ in range(len(queries))]
        relevance_info = []

        for (query_idx, doc_idx), doc, judgment in zip(query_doc_map, [d for docs in doc_lists for d in docs], judgments):
            is_relevant = self._parse_judgment(judgment, doc)

            # 记录判断结果
            relevance_info.append({
                "query": queries[query_idx],
                "doc": doc[:200] + "..." if len(doc) > 200 else doc,  # 截断显示
                "judgment": judgment,
                "is_relevant": is_relevant
            })

            # 如果文档相关，则保留
            if is_relevant:
                filtered_doc_lists[query_idx].append(doc)

        # 打印统计信息
        if self.verbose:
            original_count = sum(len(docs) for docs in doc_lists)
            filtered_count = sum(len(docs) for docs in filtered_doc_lists)
            logger.info(f"Document relevance judgment completed:")
            logger.info(f"  Original documents: {original_count}")
            logger.info(f"  Filtered documents: {filtered_count}")
            logger.info(f"  Filtered out: {original_count - filtered_count} documents ({(original_count - filtered_count) / original_count * 100:.1f}%)")

            # 显示前几个示例
            logger.info("\nRelevance judgment examples:")
            for i, info in enumerate(relevance_info[:5]):
                logger.info(f"\n  Example {i+1}:")
                logger.info(f"    Query: {info['query'][:100]}...")
                logger.info(f"    Doc preview: {info['doc'][:100]}...")
                logger.info(f"    Judgment: {info['judgment'].strip()}")
                logger.info(f"    Is relevant: {info['is_relevant']}")

        return filtered_doc_lists

    def _parse_judgment(self, judgment, doc):
        """
        解析LLM的判断结果

        Args:
            judgment: LLM生成的判断字符串
            doc: 原始文档（用于在没有明确结果时做出判断）

        Returns:
            bool: True表示文档相关，False表示不相关
        """
        judgment_lower = judgment.strip().lower()

        # 检查明确的yes/no回答
        if "yes" in judgment_lower and "no" not in judgment_lower:
            return True
        elif "no" in judgment_lower and "yes" not in judgment_lower:
            return False

        # 如果模型没有给出明确的yes/no，我们可以尝试其他启发式方法
        # 这种情况可能发生在模型给出了更详细的解释时

        # 启发式1: 如果模型说"这个文档包含..."，可能是相关的
        if "contains" in judgment_lower or "include" in judgment_lower or "有" in judgment_lower or "包含" in judgment_lower:
            # 如果是中文，检查是否包含"没有"等信息
            if "没有" in judgment_lower or "不包含" in judgment_lower:
                return False
            return True

        # 启发式2: 如果模型说"这个文档没有..."或"不包含..."，可能是不相关的
        irrelevant_patterns = [
            "does not", "doesn't", "not contain", "no information",
            "不包含", "没有信息", "无法回答", "不能回答"
        ]
        for pattern in irrelevant_patterns:
            if pattern in judgment_lower:
                return False

        # 启发式3: 如果模型给了一个具体答案，可能是相关的
        # 但这不太可能发生，因为模板明确要求yes/no

        # 如果实在是无法判断，默认返回True（保留文档，保守策略）
        logger.warning(f"Could not parse judgment: '{judgment}'. Defaulting to True (relevant).")
        return True