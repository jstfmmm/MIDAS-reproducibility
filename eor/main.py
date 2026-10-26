import json
import math
import os

import torch.distributed as dist
import time
from loguru import logger
import datetime
from itertools import chain

from eor.data import load_example
from eor.response import StandardGenerator
from eor.voting import StandardVoter
from eor.scorer import RewardScorer, aggregate_voting_rewarding_scores
from eor.controller import ControllerConstructor
from eor.evaluation import Evaluator 
from .utils import LLM, collect_saved_files


def run_eor(rank, config, debug=False):
    """
    主函数：执行Ensemble of Retrievers(EOR)的完整流程。
    EOR是一个多检索器集成的RAG(检索增强生成)系统，通过多种检索方法获取知识，
    生成多个候选回答，然后通过投票和评分机制选出最佳答案。

    参数:
        rank (int): 当前进程的rank，用于分布式训练
        config (dict): 配置信息，包含模型、数据、检索方法等配置
        debug (bool): 是否开启调试模式
    """
    # 初始化LLM模型配置
    LLM.get_llm_config(config["LLMConfig"])
    LLM.gpu_ids = config["device"]
    LLM.ddp = config["ddp"]
    verbose = config["log_detail"]

    
    if rank is None:
        # 非分布式环境设置
        world_size = 1
        rank = 0 if config["device"][0] == "cpu" else config["device"][0]
    else:
        # 分布式环境设置
        world_size = len(config["device"])
    
    if config["device"][0] == "cpu":
        device = "cpu"
    else:
        device = f"gpu{rank}"
    
    # 初始化分布式环境
    if config["ddp"]:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist.barrier()

        if rank == 0 and verbose:
            logger.info(f"********************DDP connection finished for all rank********************")

    
    # 加载数据集
    logger.info(f"********************{device}: start to load dataset from {config['test_file']}********************")
    begin_time = time.time()
    dataset = load_example(config["test_file"])
    # dataset = dataset.shuffle(seed=42)

    
    if config["ddp"]:
        # 分布式模式下，数据被分割到各个进程
        part_num = math.ceil(len(dataset)/world_size)
        begin, end = part_num * rank, part_num * (rank + 1)

        queries = list(dataset["query"][begin:end])
        language_list = list(dataset["language"][begin:end])
        truthful_answer_list = list(dataset["truthful answer"][begin:end])
        
        dist.barrier()
        
    else:
        # 非分布式模式，使用完整数据集
        queries = list(dataset["query"])
        language_list = list(dataset["language"])
        truthful_answer_list = list(dataset["truthful answer"])
    
    if rank == 0 or not config["ddp"]:
        logger.info(f"********************Dataset Loaded for all ranks, total batch size: {len(dataset)}, batch size for each rank: {len(queries)}********************")
    
    
    load_time_usage = time.time() - begin_time
    
    # 替换保存路径中的rank占位符
    if config["result_path"] is not None and "$rank$" in config["result_path"]:
        config["result_path"] = config["result_path"].replace("$rank$", str(rank))
    
    if config["result_info_path"] is not None and "$rank$" in config["result_info_path"]:
        config["result_info_path"] = config["result_info_path"].replace("$rank$", str(rank))
    
    # 替换各模块保存路径中的rank占位符
    for name, module in config.active_modules.items():
        if module["save_result_path"] is not None and "$rank$" in module["save_result_path"]:
            module["save_result_path"] = module["save_result_path"].replace("$rank$", str(rank))
        if module["save_info_path"] is not None and "$rank$" in module["save_info_path"]:
            module["save_info_path"] = module["save_info_path"].replace("$rank$", str(rank))
    
    # 初始化控制器，负责检索和处理知识
    logger.info(f"{device}: save result to {config['output_dir'].replace('$rank$/', '')}")
    logger.info(f"********************{device}: start to retrieve and process knowledge with controller********************")
    
    begin_time = time.time()

    # 创建Controller并执行批量知识检索和处理
    Controller = ControllerConstructor(config, device, world_size=world_size, rank=rank)
    # 对每个查询执行所有配置的检索方法(Wiki, Web, Gendoc等)
    processed_knowledge = Controller.batch_process(queries, language_list)
    Controller.save_if_set()
     
    if config["ddp"]:
        dist.barrier()
    
    controller_time_usage = time.time() - begin_time
        
    if rank == 0 or not config["ddp"]:
        logger.info(f"********************knowledge processing finished for all rank, time used: {datetime.timedelta(seconds = controller_time_usage)}********************")
        if config["ddp"]:
            # 收集所有分布式进程的保存文件
            collect_saved_files(config["output_dir"], world_size)
    
    if config["ddp"]:
        dist.barrier()
        
    # 开始生成回答
    if config.generator is not None:
        # 初始化生成器
        generator = StandardGenerator(config["ModuleConfig"][config.generator])

        _query_list = []
        _language_list = []
        knowledge_names = []
        _knowledge_list = []
        _responses_list = []

        if processed_knowledge is not None:
            # 准备用于生成的数据
            # 每个查询需要与每个检索方法获取的知识配对生成回答
            knowledge_num = len(processed_knowledge.keys())

            # 将 Column 对象转换为列表
            queries_list = list(queries)
            language_list_values = list(language_list)
            num_queries = len(queries_list)

            # 正确的配对顺序：
            # query:     [q1, q1, q1, q2, q2, q2, ...]  每个查询重复knowledge_num次
            # knowledge: [k1_q1, k2_q1, k3_q1, k1_q2, k2_q2, k3_q2, ...]  每个查询对应所有方法的知识
            
            _query_list = []
            _language_list = []
            _knowledge_list = []
            
            # 获取方法名列表和对应的知识文档列表
            method_names_list = list(processed_knowledge.keys())
            method_docs_list = list(processed_knowledge.values())
            
            # 对每个查询，配对所有方法的知识
            for i in range(num_queries):
                for j, method_name in enumerate(method_names_list):
                    _query_list.append(queries_list[i])
                    _language_list.append(language_list_values[i])
                    _knowledge_list.append(method_docs_list[j][i])
            
            # 保存方法名（只需一次）
            knowledge_names = method_names_list
            # 基于检索知识生成回答
            logger.info(f"********************{device}: start to generate retrieval augmented responses")
            begin_time = time.time()
            _responses_list = generator.batch_response(_query_list, _language_list, _knowledge_list, device, world_size=world_size)
            generator.save_if_set(new_only=True)
            logger.info(f"********************{device}: knowledge groundred responses generation finished, time used {datetime.timedelta(seconds = (time.time() - begin_time))}")
        else:
            begin_time = time.time()
            knowledge_num = 0
            
        # 生成仅基于查询的回答(无检索知识)
        if config["GeneratorConfig"]["dynamic_retrieval"]:
            logger.info(f"********************{device}: start to generate query only responses")
            generator.set_query_only_mode()
            q_only_begin_time = time.time()
            query_only_responses = generator.batch_response(queries, language_list, None, device, world_size=world_size)
            
            # 将query_only插入到正确的位置，保持与其他方法相同的顺序
            # 正确顺序：[q0_m0, q0_m1, ..., q0_qo, q1_m0, q1_m1, ..., q1_qo, ...]
            queries_list = list(queries)
            language_list_values = list(language_list)
            num_queries = len(queries_list)
            
            # 如果有检索方法，需要将query_only插入到每个查询块的末尾
            if knowledge_num > 0:
                new_responses_list = []
                for i in range(num_queries):
                    # 添加该查询的所有检索方法的回答
                    for j in range(knowledge_num):
                        new_responses_list.append(_responses_list[i * knowledge_num + j])
                    # 添加该查询的query_only回答
                    new_responses_list.append(query_only_responses[i])
                    
                    # 同时更新_query_list和_language_list和_knowledge_list
                    _query_list.append(queries_list[i])
                    _language_list.append(language_list_values[i])
                    _knowledge_list.append(None)
                
                _responses_list = new_responses_list
            else:
                # 如果没有检索方法，直接添加query_only
                for i in range(num_queries):
                    _query_list.append(queries_list[i])
                    _language_list.append(language_list_values[i])
                    _knowledge_list.append(None)
                _responses_list = query_only_responses
            
            knowledge_names.append("query_only")
            knowledge_num += 1  # query_only也算一个方法
            logger.info(f"********************{device}: query only responses generation finished, time used {datetime.timedelta(seconds = (time.time() - q_only_begin_time))}")
        
        response_time_usage_for_rank = time.time() - begin_time
        logger.info(f"********************{device}: responses generation finished, time used {datetime.timedelta(seconds = response_time_usage_for_rank)}")
        generator.save_if_set(new_only=True)

        # 重组回答列表，使每个查询对应一个字典，字典中包含不同检索方法生成的回答
        # _responses_list的顺序: [q0_m0, q0_m1, q0_m2, q1_m0, q1_m1, q1_m2, ...]
        responses_list = []
        for query_index in range(len(queries)):
            responses_list.append({k_name: _responses_list[query_index * knowledge_num + k_index] for k_index, k_name in enumerate(knowledge_names)})
        
        # 保存生成结果
        response_to_save = {q: a for q,a in zip(queries, responses_list)}
        logger.info(f"save generator results to {os.path.join(config['output_dir'].replace('$rank$', str(rank)), 'generator_result.json')}")
        json.dump(response_to_save, open(os.path.join(config["output_dir"].replace("$rank$", str(rank)), "generator_result.json"), "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
       
        if config["ddp"]:
            dist.barrier()
         
        if rank == 0 or not config["ddp"]:
            logger.info(f"********************responses generation finished for all rank********************")
            if config["ddp"]:
                collect_saved_files(config["output_dir"], world_size)

    response_time_usage = time.time() - begin_time

    if config["ddp"]:
        dist.barrier()
        
    # 释放所有LLM资源
    LLM.release_all()
    begin_time = time.time()
    
    # 投票阶段 - 比较不同检索方法生成的回答并打分
    if config.voter is not None:
        logger.info(f"********************{device}:Start to Voting")
        
        voter = StandardVoter(config["ModuleConfig"]["Voter"])
        if voter.use_method_weight:
            # 使用方法权重，不同检索方法具有不同权重
            logger.info("method weight is used for voting!!")
            weight = [config.method_weight[k] for k in knowledge_names]
            logger.info(f"method weight used for each method is: {json.dumps({k: w for k, w in zip(knowledge_names, weight)}, indent=4)}")
        else:
            # 不使用方法权重，所有检索方法权重相等
            logger.info("method weight is Not used for voting, each method is treated as same.")
            weight = None
            
        # 重组回答列表，便于投票
        _responses_list = [[responses_list[query_idx][name] for name in knowledge_names] for query_idx in range(len(responses_list))]
        # 执行批量投票
        _vote_scores_list, _responses_list, knowledge_names  = voter.batch_voting(queries, language_list, _responses_list, device, knowledge_names=knowledge_names, voting_weight=weight) 
        voter.save_if_set(new_only=True)
        
        # 重组回答列表，使每个查询对应一个字典
        responses_list = []
        for query_index in range(len(queries)):
            responses_list.append({k_name: _responses_list[query_index][k_index] for k_index, k_name in enumerate(knowledge_names)})

        # 重组投票分数，使每个查询对应一个字典
        vote_scores_list = [{name:vote_score for name, vote_score in zip(knowledge_names, _vote_scores_list[i])} for i in range(len(queries))]    
        
        if config["ddp"]:
            dist.barrier()
        if rank == 0 or not config["ddp"]:
            logger.info(f"********************Voting finished for all rank, time used {datetime.timedelta(seconds = time.time() - begin_time)}********************")
            if config["ddp"]:
                collect_saved_files(config["output_dir"], world_size)
    else:
        # 如果不使用投票，所有回答默认投票分数为1
        logger.info(f"********************{device}: Voter is closed, directly go to Scorer")
        vote_scores_list = [{name: 1.0 for name in knowledge_names} for i in range(len(queries))]
    
    vote_time_usage = time.time() - begin_time

    # 评分阶段 - 使用奖励模型对回答进行评分
    begin_time = time.time()
    
    # 准备用于评分的数据结构
    select_query_list, select_response_list, select_language_list, select_name_list, select_voting_score_list = [], [], [], [], []
    select_index = [0]  # 记录每个查询对应的回答起始索引
    answer_list = [None for _ in range(len(queries))]  # 最终选择的回答
    answer_name_list = ["" for _ in range(len(queries))]  # 最终选择的回答来源
    name_list = []
    
    # 筛选投票后的回答，准备进行评分
    for query_index in range(len(queries)):
        vote_scores = vote_scores_list[query_index]
        # 筛选投票分数大于0的回答
        filter_names = [name for name, score in vote_scores.items() if score > 0.0]
        filter_scores = [vote_scores[name] for name in filter_names]
        
        # 如果只有一个回答通过筛选，直接使用该回答
        if len(filter_names) == 1:            
            answer_list[query_index] = responses_list[query_index][filter_names[0]]
            answer_name_list[query_index] = filter_names[0]
            select_index.append(select_index[-1] + 1)
            select_voting_score_list.append(filter_scores)
            select_query_list.append(queries[query_index])
            select_language_list.append(language_list[query_index])
            select_response_list.append(responses_list[query_index][filter_names[0]])
            select_name_list.append(filter_names[0])
            continue       
        
        # 如果没有回答通过筛选，使用所有回答
        if len(filter_names) == 0:
            filter_names = list(vote_scores.keys())
            filter_scores = list(vote_scores.values())

        # 准备评分所需数据
        for name in filter_names:
            select_query_list.append(queries[query_index])
            select_language_list.append(language_list[query_index])
            select_response_list.append(responses_list[query_index][name])
            select_name_list.append(name)
        
        select_index.append(select_index[-1] + len(filter_names))
        select_voting_score_list.append(filter_scores)

    # 使用奖励模型进行评分
    if config.scorer is not None:
        logger.info(f"********************{device}: Start to Scoring")
        
        scorer = RewardScorer(config["ModuleConfig"]["Scorer"])
        scores_list = scorer.batch_score(select_query_list, select_language_list, select_response_list, device)
        scorer.save_if_set()

        if config["ddp"]:
            dist.barrier()
        if rank == 0 or not config["ddp"]:
            logger.info(f"********************Scoring finished for all rank, time used {datetime.timedelta(seconds = time.time() - begin_time)}********************")
            if config["ddp"]:
                collect_saved_files(config["output_dir"], world_size)
    else:
        # 如果不使用评分，直接使用投票分数
        scores_list = list(chain(*select_voting_score_list))

    
    score_time_usage = time.time() - begin_time
    
    # 释放所有LLM资源
    LLM.release_all()
    
    # 最终选择答案阶段
    reward_scores = []  # 奖励模型评分
    final_scores = []   # 最终综合评分
    
    # 对每个查询，综合投票分数和奖励模型评分选择最终答案
    for query_index in range(len(queries)):
        # 获取当前查询的所有候选回答及其分数
        select_responses = select_response_list[select_index[query_index]:select_index[query_index+1]]
        select_reward_scores = scores_list[select_index[query_index]:select_index[query_index+1]]
        select_queries = select_query_list[select_index[query_index]:select_index[query_index+1]]
        select_names = select_name_list[select_index[query_index]:select_index[query_index+1]]
        select_voting_score = select_voting_score_list[query_index]
        
        # 检查查询一致性    
        for i in range(1, len(select_queries)):
            assert select_queries[i] == select_queries[0] 
        
        # 决定是否忽略投票分数
        if config.voter is None:
            ignore_voting = True
        else: 
            ignore_voting = False
        
        # 综合投票分数和奖励模型评分
        select_scores = aggregate_voting_rewarding_scores(select_voting_score, select_reward_scores, ignore_voting=ignore_voting)

        # 🎯 在最终选择时应用method_weight
        # 这样vote_score保持纯粹的相似性得分（不含权重）
        # 但在选择答案时会考虑学习到的或预设的方法权重
        if hasattr(config, 'method_weight') and config.method_weight is not None:
            # 应用权重：weighted_score = final_score × method_weight
            weighted_scores = []
            for score, name in zip(select_scores, select_names):
                weight = config.method_weight.get(name, 1.0)  # 如果没有该方法的权重，默认为1.0
                weighted_scores.append(score * weight)
            
            # 选择加权分数最高的回答
            best_idx = weighted_scores.index(max(weighted_scores))
            logger.debug(f"Query {query_index}: 使用method_weight选择答案")
            logger.debug(f"  候选方法: {select_names}")
            logger.debug(f"  final_score: {[f'{s:.4f}' for s in select_scores]}")
            logger.debug(f"  method_weight: {[f'{config.method_weight.get(n, 1.0):.4f}' for n in select_names]}")
            logger.debug(f"  weighted_score: {[f'{s:.4f}' for s in weighted_scores]}")
            logger.debug(f"  选中: {select_names[best_idx]}")
        else:
            # 不使用权重，直接选择final_score最高的
            best_idx = select_scores.index(max(select_scores))
            weighted_scores = select_scores  # 用于保存
        
        answer_list[query_index] = select_responses[best_idx]
        answer_name_list[query_index] = select_names[best_idx]
        name_list.append(select_names)
        reward_scores.append({n: s for n, s in zip(select_names, select_reward_scores)})
        final_scores.append({n: s for n, s in zip(select_names, select_scores)})
    
    # 记录各阶段时间使用情况
    time_usage_dict = {
        "load_time_usage": load_time_usage,
        "controller_time_usage": controller_time_usage,
        "response_time_usage": response_time_usage,
        "vote_time_usage": vote_time_usage,
        "score_time_usage": score_time_usage
    }
    
    logger.info(time_usage_dict.__str__())
    if config["result_info_path"] is not None:
        json.dump(time_usage_dict, open(config["result_info_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent=4)

    # 保存最终结果
    result = dict()
    for query_index, query in enumerate(queries):
        # 为每个查询保存系统回答和真实答案
        result[query] = {
            "system_answer": answer_list[query_index],
            "system_answer_name": answer_name_list[query_index]
        }
        
        result[query]["ground_truth_answer"] = truthful_answer_list[query_index]
        result[query]["details"] = dict()
        
        # 保存每个检索方法的详细信息
        for name in knowledge_names:
            knowledge = processed_knowledge[name][query_index] if (name in processed_knowledge and processed_knowledge is not None) else None
            temp = {
                "knowledge": knowledge, 
                "response": responses_list[query_index][name],
                "vote_score": vote_scores_list[query_index][name],
                # 因为奖励模型的评分不是每个回复都有，所以只给出最后答案，不在这里给每个答案的评分。
            }
            if name in name_list[query_index]:
                temp["reward_score"] = reward_scores[query_index][name]
                temp["final_score"] = final_scores[query_index][name]
            else: 
                temp["final_score"] = temp["vote_score"]
            
            # 🎯 添加weighted_score：final_score × method_weight
            if hasattr(config, 'method_weight') and config.method_weight is not None:
                method_weight = config.method_weight.get(name, 1.0)
                temp["method_weight"] = method_weight
                temp["weighted_score"] = temp["final_score"] * method_weight
            else:
                # 如果没有method_weight，weighted_score等于final_score
                temp["method_weight"] = 1.0
                temp["weighted_score"] = temp["final_score"]
            
            result[query]["details"][name] = temp

    # 保存结果到文件
    if config["result_path"] is not None:
        json.dump(result, open(config["result_path"], "w", encoding="UTF-8"), ensure_ascii=False, indent=4)
        logger.info(f"********************{device}: result file saved to {config['result_path']}********************")

    if config["ddp"]:
        dist.barrier()

    # 所有进程完成，开始评估
    if rank == 0 or not config["ddp"]:
        logger.info(f"********************WHOLE PROCESS FINISHED!!!********************")
        if config["ddp"]:
            collect_saved_files(config["output_dir"], world_size)
    
    # 运行评估，计算模型性能指标
    if rank == 0 or not config["ddp"]:
        if config["run_evaluation"]:
            # 初始化评估器，使用EM(Exact Match)和BEM(Binary Exact Match)指标
            evaluator = Evaluator("NQ", metrics=["em","bem"])

            score_column_mapping = {"em": "em_score", "bem": "bem_score"}
            cached_score_path = os.path.join(config["cached_score_path"], "results.xlsx") if config["cached_score_path"] is not None else None
            
            # 执行评估
            evaluator.evaluate(result, cached_score_path=cached_score_path, cached_score_name_mapping=score_column_mapping, system_only=False, verbose=verbose)
            cached_dir = os.path.normpath(config["output_dir"]).replace("$rank$", "")
            evaluator.result.to_excel(cached_dir)
            
            # 获取EM和BEM分数
            em_score = evaluator.result.scores["em_score"]["system_answer"]
            bem_score = evaluator.result.scores["bem_acc"]["system_answer"]
            logger.info("*"*10 + f"[SYSTEM FINAL SCORE] em: {em_score} | bem {bem_score} on dataset [{config['data']} - {config['test_file'].split('/')[-1][:-6]}] "+"*"*10)
            
            # 保存训练用的结果文件（包含vote_score, em_score, bem_score）
            logger.info("********************Saving training result with evaluation scores********************")
            train_result = {}
            for query, query_data in result.items():
                train_result[query] = {
                    "system_answer": query_data.get("system_answer", ""),
                    "system_answer_name": query_data.get("system_answer_name", ""),
                    "ground_truth_answer": query_data.get("ground_truth_answer", []),
                    "details": {}
                }
                
                # 遍历每个方法，添加vote_score和评估分数
                if "details" in query_data:
                    for method_name, method_data in query_data["details"].items():
                        train_result[query]["details"][method_name] = {
                            "response": method_data.get("response", ""),
                            "vote_score": method_data.get("vote_score", 0.0),
                            "final_score": method_data.get("final_score", 0.0),
                        }
                        
                        # 从评估结果中添加em_score和bem_score
                        # 在evaluator.result.data中查找对应的分数
                        try:
                            # 找到对应query和method的评估分数
                            query_mask = evaluator.result.data['questions'] == query
                            method_mask = evaluator.result.data['models'] == method_name
                            combined_mask = query_mask & method_mask
                            
                            if combined_mask.any():
                                em = evaluator.result.data.loc[combined_mask, 'em_score'].values[0]
                                bem = evaluator.result.data.loc[combined_mask, 'bem_score'].values[0]
                                train_result[query]["details"][method_name]["em_score"] = float(em)
                                train_result[query]["details"][method_name]["bem_score"] = float(bem)
                            else:
                                train_result[query]["details"][method_name]["em_score"] = 0.0
                                train_result[query]["details"][method_name]["bem_score"] = 0.0
                        except Exception as e:
                            logger.warning(f"Failed to get evaluation scores for {query} - {method_name}: {e}")
                            train_result[query]["details"][method_name]["em_score"] = 0.0
                            train_result[query]["details"][method_name]["bem_score"] = 0.0
            
            # 保存train_result.json
            train_result_path = os.path.join(cached_dir, "train_result.json")
            with open(train_result_path, 'w', encoding='utf-8') as f:
                json.dump(train_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"********************Training result saved to {train_result_path}********************")
    
