import re
from tqdm import tqdm
from typing import List, Tuple
import math
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, selfask_pred_parse, ircot_pred_parse, search, find_covering_segments
from flashrag.pipeline import BasicPipeline
from flashrag.dataset.utils import get_batch_dataset, merge_batch_dataset
from flashrag.prompt import PromptTemplate


class IterativePipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, iter_num=3, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.iter_num = iter_num
        if generator is None:
            generator = get_generator(config)
        # if retriever is None:
        #     retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever
        

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        trajectories = ['' for _ in range(len(dataset))]
        questions = dataset.question
        if self.config['dataset_name'] == 'arc':
            choices = dataset.choices
        # 处理选择题的选项
        if self.config['dataset_name'] == 'arc':
            # 将每个选项转换为格式化的字符串，例如 "A: option1 B: option2 ..."
            options = [' '.join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(choice)]) for choice in choices]
            # 更新数据集中的选项
            dataset.update_output("options", options)
        # run in batch
        past_generation_result = []  # list of N items
        for iter_idx in range(self.iter_num):
            trajectories_iter = []
            if iter_idx == 0:
                input_query = questions
            else:
                assert len(questions) == len(past_generation_result)
                input_query = [f"{q} {r}" for q, r in zip(questions, past_generation_result)]

            # generation-augmented retrieval
            retrieval_results = [search(query, top_n=4) for query in input_query]
            for i in range(len(retrieval_results)):
                item_retrieval_results = retrieval_results[i]['results']
                for j in range(len(item_retrieval_results)):
                    item_retrieval_results[j]= {'contents': item_retrieval_results[j]}
                retrieval_results[i] = item_retrieval_results
            for i in range(len(retrieval_results)):
                trajectories_iter.append(self.prompt_template.format_reference(retrieval_results[i]))
            dataset.update_output(f"retrieval_result_iter_{iter_idx}", retrieval_results)
            dataset.update_output(f"trajectory_iter_{iter_idx}", trajectories_iter)

            # retrieval-augmented generation
            # input_prompts = self.build_prompt(questions, retrieval_results)
            if self.config['dataset_name'] == 'arc':
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r, options=options[i])
                    for i, (q, r) in enumerate(zip(questions, retrieval_results))
                ]
            else:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(questions, retrieval_results)
                ]
            
            for i in range(len(trajectories)):
                trajectories[i] += f'iter {iter_idx}: \n' + trajectories_iter[i] + '\n\n'
            
            dataset.update_output(f"prompt_iter_{iter_idx}", input_prompts)
            past_generation_result, scores = self.generator.generate(input_prompts, return_scores=True)
            dataset.update_output(f"pred_iter_{iter_idx}", past_generation_result)
            dataset.update_output(f"scores_iter_{iter_idx}", scores)

        # use last retrieval result for evaluation
        # trajectories = [
        #     'sequence O: \n' + dataset.prompt_iter_0[i] + '\niter 0 answer: ' + dataset.pred_iter_0[i] + '\n\n' + 'sequence 1: \n' + dataset.prompt_iter_1[i] + '\niter 1 answer: ' + dataset.pred_iter_1[i] + '\n\n' + 'sequence 2: \n' + dataset.prompt_iter_2[i] + '\niter 2 answer: ' + dataset.pred_iter_2[i] + '\n\n' for i in range(len(dataset))
        # ]
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("trajectory", trajectories)
        dataset.update_output("pred", past_generation_result)
        dataset.update_output("scores", scores)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        if self.config['return_prob']:
            return dataset, scores
        return dataset


class SelfRAGPipeline(BasicPipeline):
    # Source: https://github.com/AkariAsai/self-rag
    # The code is released under MIT license

    rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
    retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
    utility_tokens_names = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    ground_tokens_names = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    other_special_tokens = ["<s>", "</s>", "[PAD]", "<unk>", "<paragraph>", "</paragraph>"]
    control_tokens = [
        "[Fully supported]",
        "[Partially supported]",
        "[No support / Contradictory]",
        "[No Retrieval]",
        "[Retrieval]",
        "[Irrelevant]",
        "[Relevant]",
        "<paragraph>",
        "</paragraph>",
        "[Utility:1]",
        "[Utility:2]",
        "[Utility:3]",
        "[Utility:4]",
        "[Utility:5]",
    ]

    task_inst = {
        "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
        "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
        "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
        "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "arc": "Select the correct answer from the options provided based on the given document to answer the question. Only provide a letter of the correct option (e.g., A, B, C, D) and do not output any other words.",
        # "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
        "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
        "normal_qa": "Answer the following question, give me a short answer.",
        "wikiasp": "Write a detailed and coherent summary to the given entity based on the provided documents. Your response should be a well-structured paragraph that captures the key details and essence of the entity. Only give me the summary and do not output any other words. "
    }

    def __init__(
        self,
        config,
        threshold=0.2,
        max_depth=2,
        beam_width=2,
        w_rel=1.0,
        w_sup=1.0,
        w_use=1.0,
        use_grounding=True,
        use_utility=True,
        use_seqscore=True,
        ignore_cont=True,
        mode="adaptive_retrieval",
        prompt_template=None,
        retriever=None,
        generator=None
    ):

        super().__init__(config, prompt_template)
        if generator is None:
            generator = get_generator(config)
        # if retriever is None:
        #     retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        assert mode in ["adaptive_retrieval", "always_retrieve", "no_retrieval"]

        self.task = config["dataset_name"]
        self.task_instruction = self.task_inst.get(self.task, self.task_inst["normal_qa"])
        if self.task_instruction is not None:
            question_inst = self.task_instruction + "\n\n## Input:\n\n{question}"
        else:
            question_inst = "{question}"
        if self.config['dataset_name'] == 'arc':
            question_inst += "\n\n## Options:\n\n{options}"
        
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                config, user_prompt="### Instruction:\n" + question_inst + "\n\n### Response:\n", enable_chat=False
            )

        self.threshold = threshold
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.w_rel, self.w_sup, self.w_use = w_rel, w_sup, w_use
        self.use_grounding = use_grounding
        self.use_utility = use_utility
        self.use_seqscore = use_seqscore
        self.ignore_cont = ignore_cont
        self.mode = mode
        self.closed = self.task in ["fever", "arc_c"]
        tokenizer = AutoTokenizer.from_pretrained(config["generator_model_path"], padding_side="left")
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = self.load_special_tokens(
            tokenizer, use_grounding=use_grounding, use_utility=use_utility
        )
        self.vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)

    def load_special_tokens(self, tokenizer, use_grounding, use_utility):
        ret_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in self.retrieval_tokens_names}
        rel_tokens = {}
        for token in ["[Irrelevant]", "[Relevant]"]:
            rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        grd_tokens = None
        if use_grounding is True:
            grd_tokens = {}
            for token in self.ground_tokens_names:
                grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        ut_tokens = None
        if use_utility is True:
            ut_tokens = {}
            for token in self.utility_tokens_names:
                ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens

    def judge_retrieve(self, input_prompts):
        """Calculate whether a retrieve is required based on the output probability of
        the special token in the model"""

        if self.mode == "always_retrieve":
            retrieval_flags = [True] * len(input_prompts)

        elif self.mode == "no_retrieval":
            retrieval_flags = [False] * len(input_prompts)

        else:
            # result for total batch
            all_pred_text = []
            all_pred_log_probs = []
            # For vllm, requesting too many logprobes can seriously affect speed
            # 20 probs is enough for calculate
            if self.config['return_prob']:
                preds, scores = self.generator.generate(input_prompts, return_raw_output=True, logprobs=20, max_tokens=1, skip_special_tokens=False, return_scores=True)
            else:
                preds = self.generator.generate(input_prompts, return_raw_output=True, max_tokens=1, skip_special_tokens=False)
            for single_pred in preds:
                pred_text = single_pred.outputs[0].text
                pred_log_probs = single_pred.outputs[0].logprobs
                all_pred_text.append(pred_text)
                all_pred_log_probs.append(pred_log_probs)

            retrieval_flags = []
            for idx, single_pred in enumerate(preds):
                if self.threshold is not None:
                    score_dict = {}
                    for tok, tok_id in self.ret_tokens.items():
                        if tok_id not in all_pred_log_probs[idx][0]:
                            score_dict[tok] = np.exp(-100)
                        else:
                            prob = all_pred_log_probs[idx][0][tok_id].logprob
                            score_dict[tok] = np.exp(prob)
                    do_retrieve = (
                        score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])
                        > self.threshold
                    )
                else:
                    do_retrieve = "[Retrieval]" in all_pred_text[idx]

                retrieval_flags.append(do_retrieve)

        return retrieval_flags

    def critic_preds(self, preds, pred_scores=None):
        """
        评估使用不同检索文档的预测结果。
        
        该函数对模型生成的预测结果进行评分，主要从三个维度进行评估：
        1. 相关性（Relevance）：预测结果与检索文档的相关程度
        2. 事实依据（Groundness）：预测结果是否有事实依据支持
        3. 实用性（Utility）：预测结果的实用价值
        
        参数:
            preds: 模型生成的预测结果列表
            scores: 可选的预测分数
            
        返回:
            results: 包含每个预测及其分数的字典
            final_preds: 处理后的最终预测结果列表
            scores: 每个预测的总体分数列表
            overall_scores: 包含详细评分信息的字典
        """

        # 初始化各评分维度的字典
        relevance_score_dict = {}  # 相关性评分
        grd_score_dict = {}        # 事实依据评分
        ut_score_dict = {}         # 实用性评分
        overall_scores = {}        # 总体评分
        results = {}               # 结果字典
        
        # 遍历每个预测结果
        for p_idx, pred in enumerate(preds):
            # 获取预测的token ID、文本和对数概率
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            # 计算序列平均对数概率（用于评估生成质量）
            seq_score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)
            
            # 初始化当前预测的评分字典
            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            
            # 计算相关性评分 - 检查特殊token的概率
            for tok, id in self.rel_tokens.items():
                prob = pred_log_probs[0][id].logprob if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            # 计算事实依据评分（如果有相关token）
            if self.grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(self.grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in self.grd_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))
            
            # 计算实用性评分（如果有相关token）
            utility_token_appear_indices = []
            if self.ut_tokens is not None:
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(self.ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in self.ut_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            # 计算归一化的相关性分数
            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values()))
            )

            # 计算归一化的事实依据分数（完全支持得1分，部分支持得0.5分）
            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum
                )
            else:
                ground_score = 0.0

            # 计算归一化的实用性分数（使用加权平均）
            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]  # 实用性的权重分数
                utility_score = np.sum(
                    [
                        ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i + 1)] / ut_sum)
                        for i in range(len(ut_scores))
                    ]
                )
            else:
                utility_score = 0.0

            # 计算最终分数，可以选择是否包含序列分数
            if self.use_seqscore is True:
                final_score = (
                    np.exp(seq_score)
                    + self.w_rel * relevance_score
                    + self.w_sup * ground_score
                    + self.w_use * utility_score
                )
            else:
                final_score = self.w_rel * relevance_score + self.w_sup * ground_score + self.w_use * utility_score

            # 保存所有评分信息
            overall_scores[p_idx] = {
                "final_score": final_score,
                "relevance_score": relevance_score,
                "ground_score": ground_score,
                "utility_score": utility_score,
                "relevance_score_dict": relevance_score_dict,
                "grd_score_dict": grd_score_dict,
                "ut_score_dict": utility_score,
            }
            # 保存预测结果和分数
            results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": final_score}

        # 处理长文本生成中的检索决策（是否继续检索）
        final_preds = []
        if "[No Retrieval]" in pred_text:
            # 找出所有"No Retrieval"标记的位置
            ret_token_appear_indices = []
            substrings = pred_text.split("[No Retrieval]")

            for tok_idx, tok in enumerate(pred_token_ids):
                if tok == self.ret_tokens["[No Retrieval]"]:
                    ret_token_appear_indices.append(tok_idx)

            # 计算每个检索决策点的概率
            ret_token_score_dict = {}
            retrieval_remap = {}
            for order, idx in enumerate(ret_token_appear_indices):
                ret_token_score_dict.setdefault(order, {})
                for tok, tok_id in self.ret_tokens.items():
                    prob = pred_log_probs[idx][tok_id].logprob if tok_id in pred_log_probs[idx] else -100
                    ret_token_score_dict[order][tok] = np.exp(prob)
                
                # 根据阈值决定是否进行检索
                if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                    do_retrieve = (
                        ret_token_score_dict[order]["[Retrieval]"]
                        + ret_token_score_dict[order]["[Continue to Use Evidence]"]
                    ) / (
                        ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]
                    ) > self.threshold
                else:
                    do_retrieve = 0.0
                
                # 记录每个决策点的检索决定
                if do_retrieve > self.threshold:
                    retrieval_remap[order] = True
                else:
                    retrieval_remap[order] = False
            
            # 根据检索决策重建预测文本
            processed_pred = ""
            for substr_i, substring in enumerate(substrings):
                if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                    processed_pred += substring + "[Retrieval]"
                else:
                    processed_pred += substring + "[No Retrieval]"
            pred_text = processed_pred
            final_preds.append(pred_text)
        else:
            final_preds.append(pred_text)

        # 提取每个预测的最终分数
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]

        return results, final_preds, scores, overall_scores

    def postprocess_prediction(self, pred, initial_score=None):
        def fix_spacing(input_text):
            # Add a space after periods that lack whitespace
            output_text = re.sub(r"(?<=\w)([.!?])(?=\w)", r"\1 ", input_text)
            return output_text

        # 如果提供了initial_score，我们需要处理它
        if initial_score is not None:
            tokenizer = self.generator.tokenizer
            original_tokens = tokenizer.encode(pred, add_special_tokens=False)
            processed_pred = pred
            score_indices_to_remove = []
            
            # 处理控制标记
            for token in self.control_tokens:
                if token in processed_pred:
                    token_ids = tokenizer.encode(token, add_special_tokens=False)
                    for i in range(len(original_tokens) - len(token_ids) + 1):
                        if original_tokens[i:i+len(token_ids)] == token_ids:
                            score_indices_to_remove.extend(range(i, i+len(token_ids)))
                    processed_pred = processed_pred.replace(token, "")
            
            # 处理特殊标记
            special_tokens = ["</s>", "\n", "<|endoftext|>"]
            for token in special_tokens:
                if token in processed_pred:
                    token_ids = tokenizer.encode(token, add_special_tokens=False)
                    for i in range(len(original_tokens) - len(token_ids) + 1):
                        if original_tokens[i:i+len(token_ids)] == token_ids:
                            score_indices_to_remove.extend(range(i, i+len(token_ids)))
                    processed_pred = processed_pred.replace(token, "")
            
            # 处理前导字符
            processed_pred = processed_pred.strip()
            if len(processed_pred) > 0 and (processed_pred[0] == "#" or processed_pred[0] == ":"):
                first_char_id = tokenizer.encode(processed_pred[0], add_special_tokens=False)[0]
                if original_tokens[0] == first_char_id:
                    score_indices_to_remove.append(0)
                processed_pred = processed_pred[1:]
            
            # 从initial_score中移除对应的分数
            if score_indices_to_remove:
                score_indices_to_remove = sorted(list(set(score_indices_to_remove)))
                initial_score = [s for i, s in enumerate(initial_score) if i not in score_indices_to_remove]
            
            pred = processed_pred
        else:
            # 原始处理逻辑
            for token in self.control_tokens:
                pred = pred.replace(token, "")
            if "</s>" in pred:
                pred = pred.replace("</s>", "")
            if "\n" in pred:
                pred = pred.replace("\n", "")
            if "<|endoftext|>" in pred:
                pred = pred.replace("<|endoftext|>", "")

            pred = pred.strip()
            if type(pred) is str and (pred[0] == "#" or pred[0] == ":"):
                pred = pred[1:]
                
        if len(pred) == 0:
            if initial_score is not None:
                return '', initial_score
            else:
                return ''

        if initial_score is not None:
            return fix_spacing(pred), initial_score
        else:
            return fix_spacing(pred)

    def select_best_prediction(self, results, scores=None):
        answer2score = {}
        best_score = None
        if self.closed is True:
            for i, (key, result) in enumerate(results.items()):
                answer = self.postprocess_prediction(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)
            best_pred = sorted_answers[0][0]
            # 找到best_pred对应的原始结果中的key
            best_key = None
            for key, result in results.items():
                if self.postprocess_prediction(result["pred"]) == best_pred:
                    best_key = key
                    break
            # 使用scores中对应key的分数作为best_score
            best_score = scores[best_key] if best_key is not None and scores is not None else sorted_answers[0][1]
        else:
            path2score = {key: item["score"] for key, item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            best_pred = results[best_path]["pred"]
            # 找到best_pred对应的原始结果中的key
            for idx, (key, result) in enumerate(results.items()):
                if result['pred'] == best_pred:
                    best_score = scores[idx]
                    break
            # 使用scores中对应key的分数作为best_score

        return best_pred, best_score

    def run_single_beam(self, prompt, item_retrieval_result=None):
        curr_depth = 1
        terminated = False
        node_id = 0
        prediction_tree = {}
        levels = {}
        prediction_tree[node_id] = {
            "prompt": prompt,
            "pred": "[Retrieval]",
            "processed_pred": "",
            "score": None,
            "ctx": None,
            "parent": None,
        }
        levels[0] = [0]
        while curr_depth < self.max_depth:
            levels[curr_depth] = []
            if curr_depth - 1 in levels and terminated is False:
                for node in levels[curr_depth - 1]:
                    pred = prediction_tree[node]["pred"]
                    if pred == "</s>":
                        terminated = True
                        continue
                    prompt = prediction_tree[node]["prompt"]
                    prev_generation = prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"]
                    if "[Retrieval]" in pred:
                        retrieval_results = {}

                        if item_retrieval_result is not None:
                            aug_prompts = [
                                prompt
                                + prev_generation
                                + "[Retrieval]"
                                + "<paragraph>{}</paragraph>".format(para["contents"])
                                for para in item_retrieval_result
                            ]
                        else:
                            aug_prompts = [prompt + prev_generation]

                        if self.config['return_prob']:
                            item_pred, item_scores = self.generator.generate(aug_prompts, return_raw_output=True, logprobs=5, return_scores=True)
                        else:
                            item_pred = self.generator.generate(aug_prompts, return_raw_output=True, logprobs=5)
                        _, preds, scores, overall_score_dict = self.critic_preds(item_pred)

                        for i, (pred, p_score) in enumerate(zip(preds, scores)):
                            retrieval_results[i] = {"pred": pred, "score": p_score}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = result["score"] * score if score is not None else result["score"]
                            pred = result["pred"]
                            prediction_tree[node_id] = {
                                "prompt": prompt + prev_generation,
                                "pred": pred,
                                "score": node_score,
                                "ctx": item_retrieval_result[i],
                                "parent": node,
                                "overall_score_dict": overall_score_dict,
                            }

                            if "[Retrieval]" in pred:
                                gen_result_index = pred.index("[Retrieval]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)

                current_rank = levels[curr_depth]
                node2score = {node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[: self.beam_width]
                levels[curr_depth] = [node[0] for node in top_nodes]
                curr_depth += 1
            else:
                break

        final_prediction = ""
        parent = 0
        best_selections = {}

        # Traverse from the bottom
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
        for path_i, node in enumerate(levels[len(levels)]):
            if node == 0:
                break
            best_selections[path_i] = [node]
            current_node = node
            current_level = curr_depth
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i]
                current_node = parent
                current_level += 1

        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items():
            final_prediction[path_i] = " ".join(
                [
                    prediction_tree[node]["processed_pred"]
                    for node in nodes
                    if node is not None
                    and (
                        self.ignore_cont is False
                        or (
                            self.ignore_cont is True
                            and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                        )
                    )
                ]
            )
            splitted_sentences[path_i] = [
                prediction_tree[node]["processed_pred"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]
            original_splitted_sentences[path_i] = [
                prediction_tree[node]["pred"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]
            ctxs[path_i] = [
                prediction_tree[node]["ctx"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]

        result = {
            "final_prediction": final_prediction,
            "splitted_sentences": splitted_sentences,
            "original_splitted_sentences": original_splitted_sentences,
            "best_selections": best_selections,
            "ctxs": ctxs,
            "prediction_tree": prediction_tree,
        }

        return final_prediction[0], result

    def postprocess_long_form(self, pred, intermediate):
        final_output = ""
        docs = []
        prev_gen = []
        if "splitted_sentences" not in intermediate:
            final_output = self.postprocess_prediction(pred)
        else:
            if len(self.postprocess_prediction(pred)) == 0:
                intermediate["splitted_sentences"][0], intermediate["ctxs"][0] = (
                    intermediate["splitted_sentences"][1],
                    intermediate["ctxs"][1],
                )
            for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = self.postprocess_prediction(sent)
                if postprocessed_result in prev_gen:
                    continue
                else:
                    prev_gen.append(postprocessed_result)
                final_output += postprocessed_result[:-1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if len(final_output) == 0:
                final_output = final_output
            if len(final_output) > 0 and final_output[-1] == " ":
                final_output = final_output[:-1]
            final_output = final_output.strip()
            final_output = final_output.replace(".[Continue to Use Evidence]", " [1]. ")
            final_output = final_output.replace(". [1] ", " [1]. ")

        return final_output

    def run_batch_pred_long_form(self, dataset):
        questions = dataset.question
        retrieval_results = []
        for query in questions:
            search_results = search(query, top_n=5)
            if isinstance(search_results, dict):
                search_results = search_results['results']
            processed_results = []
            for result in search_results:
                # print(result)
                # assert False
                split_index = result.find('\n')
                if split_index != -1:
                    title = result[:split_index]
                    contents = result[split_index+1:]
                    processed_results.append({"title": title, "contents": contents})
                else:
                    # 如果没有找到换行符，则整个字符串作为contents
                    processed_results.append({"title": "", "contents": result})
            retrieval_results.append(processed_results)
        dataset.update_output("retrieval_result", retrieval_results)

        # input_prompts = self.build_prompt(questions)
        input_prompts = [self.prompt_template.get_string(question=q) for q in questions]

        # determine whether to retrieve
        retrieval_flags = self.judge_retrieve(input_prompts)
        dataset.update_output("retrieval_flag", retrieval_flags)

        # for long form task, only support single item run
        for item, prompt, retrieval_flag in zip(dataset, input_prompts, retrieval_flags):
            if retrieval_flag:
                pred, intermediate_result = self.run_single_beam(prompt, item_retrieval_result=item.retrieval_result)
                item.update_output("intermediate_result", intermediate_result)

                if self.task == "factscore":
                    pred = self.postprocess_prediction(pred)
                else:
                    assert self.task in ["asqa", "eli5"]
                    pred = self.postprocess_long_form(pred, intermediate_result)
            else:
                prompt += "[No Retrieval]"
                pred = self.generator.generate(prompt)[0]

            item.update_output("pred", pred)

        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None, long_form=False):
        run_func = self.run_batch_pred_long_form if long_form else self.run_batch_pred
        
        # # to avoid oom, split the total dataset into small batches
        # all_dataset_list = []
        # for batch_dataset in tqdm(get_batch_dataset(dataset, batch_size=batch_size), desc="Batch dataset: "):
        #     batch_dataset = run_func(batch_dataset)
        #     all_dataset_list.append(batch_dataset)
        # dataset = merge_batch_dataset(all_dataset_list)

        dataset = run_func(dataset)
        if self.config['return_prob']:
            scores = dataset.scores
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        if self.config['return_prob']:
            return dataset, scores
        return dataset

    def run_batch_pred(self, dataset):
        trajectories = []
        questions = dataset.question
        if self.config['dataset_name'] == 'arc':
            choices = dataset.choices
        # 处理选择题的选项
        if self.config['dataset_name'] == 'arc':
            # 将每个选项转换为格式化的字符串，例如 "A: option1 B: option2 ..."
            options = [' '.join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(choice)]) for choice in choices]
            # 更新数据集中的选项
            dataset.update_output("options", options)
        # retrieval_results = self.retriever.batch_search(questions)
        retrieval_results = []
        for i in tqdm(range(len(questions)), desc="Retrieval: "):
            query = questions[i]
            search_results = search(query, top_n=5)
            if isinstance(search_results, dict):
                search_results = search_results['results']
            processed_results = []
            for result in search_results:
                # print(result)
                # assert False
                split_index = result.find('\n')
                if split_index != -1:
                    title = result[:split_index]
                    contents = result[split_index+1:]
                    processed_results.append({"title": title, "contents": contents})
                else:
                    # 如果没有找到换行符，则整个字符串作为contents
                    processed_results.append({"title": "", "contents": result})
            retrieval_results.append(processed_results)
        dataset.update_output("retrieval_result", retrieval_results)
        # print(dataset.retrieval_results[0])
        # assert False

        # input_prompts = self.build_prompt(questions)
        if self.config['dataset_name'] == 'arc':
            input_prompts = [self.prompt_template.get_string(question=q, options=options[i]) for i, q in enumerate(questions)]
        else:
            input_prompts = [self.prompt_template.get_string(question=q) for q in questions]
        # print(input_prompts[0])
        # assert False
        initial_input_prompts = input_prompts.copy()

        # determine whether to retrieve
        retrieval_flags = self.judge_retrieve(input_prompts)
        dataset.update_output("retrieval_flag", retrieval_flags)

        # process input item based on whether to retrieve
        all_input_list = []
        for idx, (prompt, item) in enumerate(zip(input_prompts, dataset)):
            retrieval_flag = retrieval_flags[idx]

            if retrieval_flag:
                retrieval_result = retrieval_results[idx]
                # for each doc in retrieval result, there is a prompt as input
                prompt_list = [
                    prompt + "[Retrieval]<paragraph>{}</paragraph>".format(para["contents"])
                    for para in retrieval_result
                ]
            else:
                prompt += "[No Retrieval]"
                prompt_list = [prompt]

            item.update_output("prompt", prompt_list)
            all_input_list += prompt_list

        trajectories = all_input_list

        if self.config['return_prob']:
            batch_pred, scores = self.generator.generate(all_input_list, return_raw_output=True, logprobs=5, return_scores=True)
        else:
            batch_pred = self.generator.generate(all_input_list, return_raw_output=True, logprobs=5)

        # parse output based on retrieval flag
        pred_idx = 0
        pred_answer_list = []
        scores_list = []
        trajectories = [
            trajectory + '\ninitial answer: ' + pred.outputs[0].text for trajectory, pred in zip(trajectories, batch_pred)
        ]
        for idx, (retrieval_flag, item) in enumerate(zip(retrieval_flags, dataset)):
            if retrieval_flag:
                # for item that need retrieval, there may have more than one prediction
                item_pred = batch_pred[pred_idx : pred_idx + len(retrieval_results[idx])]
                item_score = scores[pred_idx : pred_idx + len(retrieval_results[idx])]
                pred_idx += len(retrieval_results[idx])
                critic_result, _, _, _ = self.critic_preds(item_pred)
                item.update_output("critic_result", critic_result)

                # select best prediction
                pred, item_score = self.select_best_prediction(critic_result, item_score)

            else:
                item_pred = batch_pred[pred_idx : pred_idx + 1][0]
                item_score = scores[pred_idx : pred_idx + 1][0]
                pred_idx += 1
                pred = item_pred.outputs[0].text
            temp_pred = pred

            pred = self.postprocess_prediction(pred)
            # 如果pred为空，则将item_score赋值为0
            if not pred:
                item_score = 0
            else:
                # print(temp_pred)
                # print(pred)
                # assert False
                output_tokens = self.generator.tokenizer.encode(pred, add_special_tokens=False)
                detokenized_answer = [self.generator.tokenizer.decode(token) for token in output_tokens]
                begin_end = find_covering_segments(temp_pred, detokenized_answer, pred)
                item_score = item_score[begin_end[0]:begin_end[1]+1]
            # print(pred)
            pred_answer_list.append(pred)
            scores_list.append(item_score)
        trajectories = [
            trajectory + '\nfinal answer: ' + pred for trajectory, pred in zip(trajectories, pred_answer_list)
        ]
        # print(trajectories[0])
        # print('==========')
        # print(initial_input_prompts[0])
        # print('==========')
        # print(trajectories[0][len(initial_input_prompts[0]):])
        # assert False
        trajectories = [
            trajectory[trajectory.find('['):] if '[' in trajectory else trajectory for i, trajectory in enumerate(trajectories)
        ]
        # print(pred_answer_list)
        dataset.update_output("pred", pred_answer_list)
        dataset.update_output("scores", scores_list)
        dataset.update_output("trajectory", trajectories)

        return dataset


class FLAREPipeline(BasicPipeline):
    def __init__(
        self,
        config,
        threshold=0.2,
        look_ahead_steps=64,
        max_generation_length=256,
        max_iter_num=5,
        prompt_template=None,
        retriever=None,
        generator=None
    ):
        super().__init__(config, prompt_template)
        if generator is None:
            generator = get_generator(config)
        # if retriever is None:
        #     retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        print(self.prompt_template)

        self.threshold = threshold
        self.max_generation_length = max_generation_length
        self.max_iter_num = max_iter_num
        self.look_ahead_steps = look_ahead_steps
        self.stop_sym = list("!@#$%^&*()\n\n)(*&^%$#@!")

    def get_next_sentence(self, output, scores):
        tokenizer = self.generator.tokenizer
        text_sentences = re.split(r"(?<=[^A-Z].[.?]) +", output)
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
        else:
            token_id_sentences = [tokenizer.encode(s, allowed_special="all") for s in text_sentences]

        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # assert sum([len(s) for s in token_id_sentences]) == len(
        #    output_ids), "token id sentences length not equal to output ids length"

        first_sent_ids = token_id_sentences[0]
        first_sent_score = scores[: len(first_sent_ids)]

        return text_sentences[0], first_sent_score

    def judge_sent_confidence(self, sent, sent_score):
        judge_result = all([score > self.threshold for score in sent_score])
        new_query = None
        if not judge_result:
            tokenizer = self.generator.tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                sent_ids = tokenizer.encode(sent, add_special_tokens=False)
            else:
                sent_ids = tokenizer.encode(sent, allowed_special="all")
            # assert len(sent_ids) == len(sent_score)
            new_query_ids = [i for i, score in zip(sent_ids, sent_score) if score > self.threshold]
            new_query = tokenizer.decode(new_query_ids)
            if len(new_query) == 0:
                judge_result = True
        return judge_result, new_query

    def run_item(self, item):
        question = item.question
        if self.config['dataset_name'] == 'arc':
            choices = item.choices
            options = [' '.join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(choice)]) for choice in choices]
        gen_length = 0
        iter_round = 0
        final_gen_result = ""
        while gen_length < self.max_generation_length and iter_round < self.max_iter_num:
            if self.config['dataset_name'] == 'arc':
                input_prompt = self.prompt_template.get_string(question=question, previous_gen=final_gen_result, options=options)
            else:
                input_prompt = self.prompt_template.get_string(question=question, previous_gen=final_gen_result)

            # input_prompt = self.build_prompt(
            #     question_list=[question], use_reference=False, previous_gen=final_gen_result)[0]
            # scores: token logits of the whole generation seq
            round_gen_output, scores = self.generator.generate(
                input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
            )
            round_gen_output, scores = round_gen_output[0], scores[0]
            # next_sent_scores: token logits of the first sent in generation seq
            next_sent, next_sent_score = self.get_next_sentence(round_gen_output, scores)
            # judge next sentence
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score)
            item.update_output(f"judge_result_iter{iter_round}", judge_result)

            if not judge_result:
                # do retrieval-augmented generation
                # retrieval_result = self.retriever.search(query)
                retrieval_result = [search(query) for query in query]
                item.update_output("retrieval_result", retrieval_result)
                if self.config['dataset_name'] == 'arc':
                    input_prompt = self.prompt_template.get_string(
                        question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result, options=options
                    )
                else:
                    input_prompt = self.prompt_template.get_string(
                        question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result
                    )

                # input_prompt = self.build_prompt(
                #     question_list = [question],
                #     retrieval_results = [retrieval_result],
                #     previous_gen = final_gen_result)[0]
                output, scores = self.generator.generate(
                    input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
                )
                output, scores = output[0], scores[0]
                next_sent, _ = self.get_next_sentence(output, scores)
                item.update_output(f"gen_iter_{iter_round}", next_sent)
                item.update_output("retrieval_result", retrieval_result)

            final_gen_result += next_sent
            gen_length += len(next_sent_score)
            iter_round += 1

        item.update_output("pred", final_gen_result)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class SelfAskPipeline(BasicPipeline):
    FOLLOW_UP_PATTERN = r"Follow up:.*\n"

    def __init__(self, config, prompt_template=None, max_iter=5, single_hop=True, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        from flashrag.prompt.selfask_examplars import SELF_ASK_PROMPT_SINGLE_HOP, SELF_ASK_PROMPT_MULTI_HOP

        if generator is None:
            generator = get_generator(config)
        # if retriever is None:
        #     retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.single_hop = single_hop
        self.max_iter = max_iter
        self.P_INS = SELF_ASK_PROMPT_SINGLE_HOP if self.single_hop else SELF_ASK_PROMPT_MULTI_HOP

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Context{idx+1}: {text}\n"

        return format_reference

    def _remove_duplicate_doc(self, docs):
        assert all(["id" in doc for doc in docs])
        new_doc_list = []
        exist_ids = []
        for doc in docs:
            doc_id = doc["id"]
            if doc_id not in exist_ids:
                exist_ids.append(doc_id)
                new_doc_list.append(doc)
        return new_doc_list

    def run_item(self, item):
        question = item.question
        # retrieval_result = self.retriever.search(question)
        retrieval_result = search(question)

        stop_condition = "Intermediate answer:"
        follow_ups = "No." if self.single_hop else "Yes."
        res = ""
        early_exit = False
        for idx in range(self.max_iter):
            input_prompt = (
                self.P_INS
                + "\n"
                + self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )
            gen_out = self.generator.generate(input_prompt, stop=["Context:", "#", stop_condition])[0]
            item.update_output(f"intermediate_output_iter{idx}", gen_out)

            if stop_condition == "Intermediate answer:":
                res += gen_out.split("Intermediate answer:")[0]
                stop_condition = "Follow up:"

            elif stop_condition == "Follow up:":
                followup_split = re.split(self.FOLLOW_UP_PATTERN, gen_out)
                res += followup_split[0]

                if len(followup_split) > 1:
                    res += re.findall(self.FOLLOW_UP_PATTERN, gen_out)[0]
                stop_condition = "Intermediate answer:"

            # make sure the result does not end in a new line
            if len(res) == 0:
                early_exit = True
                break
            if res[-1] == "\n":
                res = res[:-1]

            if "Follow up: " in gen_out:
                # get the first follow up
                new_query = [l for l in gen_out.split("\n") if "Follow up: " in l][0].split("Follow up: ")[-1]
                # retrieval_result = self.retriever.search(new_query)
                retrieval_result = search(new_query)

            if "So the final answer is: " in gen_out:
                res = (
                    self.format_reference(retrieval_result)
                    + f"\nQuesiton: {question}"
                    + "\nAre follow up questions needed here: "
                    + follow_ups
                    + "\n"
                    + res
                )
                early_exit = True
                # print("Success: early exit!")
                break

        if not early_exit:
            res = (
                self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("pred", res)

    def run(self, dataset, do_eval=True, pred_process_fun=selfask_pred_parse):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    
class IRCOTPipeline(BasicPipeline):
    IRCOT_INSTRUCTION = '''You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. \
This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. \
Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, \
start with "So the answer is:".'''
    IRCOT_EXAMPLE = """Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, \
which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. \
This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\n \
Wikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, \
each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. \
The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\n \
A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments \
that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, \
Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: \
00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi \
and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in \
the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"""

    def __init__(
        self, config, prompt_template=None, max_iter=2, retriever=None, generator=None
    ):
        # if not provide prompt template, use default template provided by IRCOT
        if config['dataset_name'] == 'arc':
            self.IRCOT_INSTRUCTION = self.IRCOT_INSTRUCTION.replace("the answer is:", "the option is:")
            self.IRCOT_INSTRUCTION += " Besides, your answer should only be one of the options (A, B, C, D)."
            self.IRCOT_EXAMPLE = """
Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, \
which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. \
This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\n \
Wikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, \
each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. \
The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\n \
A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments \
that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, \
Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: \
00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi \
and Trojkrsti located in the same country?\nOptions: A: yes B: no\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in \
the country of Republic of Macedonia. Thus, they are not in the same country. So the option is: B\n\n
""" 
        if config['dataset_name'] == 'wikiasp':
            self.IRCOT_INSTRUCTION = """
You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. \
This task is illustrated through demonstrations, each consisting of a document set paired with a relevant entity and its multi-hop reasoning thoughts. \
Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, \
start with "So the summary is:".
"""
            self.IRCOT_EXAMPLE = """
Wikipedia Title: The introduction of Paris\nParis is the capital and largest city of France, located in the northern part of the country along the banks of the Seine River. \
With a population of approximately 2.1 million in the city proper and over 12 million in its metropolitan area as of 2025, it is one of Europe’s most populous urban centers. \
Famous for its rich history, stunning architecture, and cultural landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, \
Paris is a global hub for art, fashion, and gastronomy. The city is divided into 20 arrondissements, each with its own distinct character, \
and is surrounded by notable suburbs like Versailles and Saint-Denis. \n\nEntity: Paris\nThought: Paris is the capital and largest city of France, \
located in the northern part of the country along the banks of the Seine River. With a population of approximately 2.1 million in the city proper \
and over 12 million in its metropolitan area as of 2025, it is one of Europe’s most populous urban centers. Famous for its rich history, \
stunning architecture, and cultural landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, Paris is a global hub for art, \
fashion, and gastronomy. The city is divided into 20 arrondissements, each with its own distinct character, and is surrounded by notable suburbs like Versailles and Saint-Denis. \
So the summary is: Paris, the capital of France, lies along the Seine River in the northern part of the country. \
It has a population of about 2.1 million in the city and over 12 million in its metropolitan area. \
Known for landmarks like the Eiffel Tower, Notre-Dame, and the Louvre, Paris is a global center for art, fashion, and culture, \
with a rich history spanning the French Revolution to modern times.\n\n
"""
        if prompt_template is None:
            temp_user_prompt = "{reference}Question: {question}\nThought:"
            if config['dataset_name'] == 'arc':
                temp_user_prompt = "{reference}Question: {question}\nOptions: {options}\nThought:"
            if config['dataset_name'] == 'wikiasp':
                temp_user_prompt = "{reference}Entity: {question}\nThought:"
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=f"{self.IRCOT_INSTRUCTION}\n\n{self.IRCOT_EXAMPLE}",
                user_prompt=temp_user_prompt,
                reference_template="Wikipedia Title: {title}\n{text}\n\n",
                enable_chat=False,
            )

        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        # self.retriever = get_retriever(config) if retriever is None else retriever

        self.max_iter = max_iter

    def run_batch(self, items):
        # Initialize the necessary data structures
        batch_thoughts = {item_id: [] for item_id in range(len(items))}
        batch_scores = {item_id: [] for item_id in range(len(items))}
        iter_num = 0
        batch_retrieval_results = []
        doc2score_batch = []
        id2doc_batch = []
        trajectories = []

        # Initial retrieval for all items in the batch
        questions = [item.question for item in items]
        if self.config['dataset_name'] == 'arc':
            choices = [item.choices for item in items]
        # 处理选择题的选项
        if self.config['dataset_name'] == 'arc':
            # 将每个选项转换为格式化的字符串，例如 "A: option1 B: option2 ..."
            options = [' '.join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(choice)]) for choice in choices]
            items.update_output("options", options)
        # retrieval_results, scoress = self.retriever.batch_search(questions, return_score=True)
        retrieval_results = []
        scoress = []
        seen_contents = set()  # 用于检测重复文档
        doc_idx = 0  # 初始化文档ID
        for i in tqdm(range(len(questions)), desc="Retrieval: "):
            question = questions[i]
            search_results_ = search(question, top_n=5, return_score=True)
            if isinstance(search_results_, dict):
                search_results = search_results_['results']
                scores = search_results_['scores']
            else:
                search_results, scores = search_results_
            
            processed_results = []
            
            for result in search_results:
                split_index = result.find('\n')
                if split_index != -1:
                    title = result[:split_index]
                    contents = result[split_index+1:]
                    # if len(contents) > 750:
                    #     contents = contents[:750] + "..."
                    
                    # 检查是否为重复文档
                    content_hash = hash(contents)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        processed_results.append({"id": f"doc_{doc_idx}", "title": title, "contents": contents})
                        doc_idx += 1
                else:
                    # 如果没有找到换行符，则整个字符串作为contents
                    content_hash = hash(result)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        processed_results.append({"id": f"doc_{doc_idx}", "title": "", "contents": result})
                        doc_idx += 1
            
            retrieval_results.append(processed_results)
            scoress.append(scores)
        for retrieval_result, scores in zip(retrieval_results,scoress):   
            
            doc2score = {doc_item['id']: score for doc_item, score in zip(retrieval_result, scores)}
            id2doc = {doc_item['id']: doc_item for doc_item in retrieval_result}
            batch_retrieval_results.append(retrieval_result)
            doc2score_batch.append(doc2score)
            id2doc_batch.append(id2doc)

        # Start the iterative process
        active_item_ids = list(range(len(items)))  # Track items that need more iterations
        while iter_num < self.max_iter:
            print('iter_num:', iter_num)
            # print('========= 第一个问题的检索文档数量 =========')
            # print(len(batch_retrieval_results[active_item_ids[0]]))
            # Generate prompts and new thoughts for the active items
            if self.config['dataset_name'] == 'arc':
                input_prompts = [
                    self.prompt_template.get_string(
                        question=items[item_id].question,
                    retrieval_result=batch_retrieval_results[item_id],
                    previous_gen=' '.join(batch_thoughts[item_id]),
                    options=items[item_id].options
                )
                for item_id in active_item_ids
                ]
                # print(input_prompts[0])
            else:
                input_prompts = [
                    self.prompt_template.get_string(
                        question=items[item_id].question,
                        retrieval_result=batch_retrieval_results[item_id],
                        previous_gen=' '.join(batch_thoughts[item_id])
                    )
                    for item_id in active_item_ids
                ]

            # print(input_prompts[0])
            # assert False
            if input_prompts == []:
                break
            # assert False

            # Batch generation for active items
            if self.config['return_prob']:
                new_thoughts_batch, scores = self.generator.generate(input_prompts, stop=['.', '\n'], return_scores=True)
            else:
                new_thoughts_batch = self.generator.generate(input_prompts, stop=['.', '\n'])
            
            # Update thoughts and determine next active items
            new_active_item_ids = []
            for idx, item_id in enumerate(active_item_ids):
                new_thought = new_thoughts_batch[idx]
                batch_thoughts[item_id].append(new_thought)
                batch_scores[item_id] = scores[idx]
                # Check for termination condition
                # Store intermediate outputs
                if "So the answer is:" in new_thought or "So the option is:" in new_thought or "So the summary is:" in new_thought:
                    temp_scores = scores[idx]
                    items[item_id].update_output(
                        f'intermediate_output_iter{iter_num}', 
                        {
                            'input_prompt': input_prompts[idx],
                            'new_thought': new_thought,
                        },
                    )
                    # 把new_thought后面的部分提取出来
                    if "So the answer is:" in new_thought:
                        latter_part = new_thought.split("So the answer is:")[1]
                    elif "So the option is:" in new_thought:
                        latter_part = new_thought.split("So the option is:")[1]
                    elif "So the summary is:" in new_thought:
                        latter_part = new_thought.split("So the summary is:")[1]
                    output_tokens = self.generator.tokenizer.encode(new_thought, add_special_tokens=False)
                    detokenized_answer = [self.generator.tokenizer.decode(token) for token in output_tokens]
                    begin_end = find_covering_segments(new_thought, detokenized_answer, latter_part)
                    batch_scores[item_id] = temp_scores[begin_end[0]:begin_end[1]+1]
                        
                else:
                    new_active_item_ids.append(item_id)

            # Update active item IDs for the next iteration
            active_item_ids = new_active_item_ids

            # Perform batch retrieval for new thoughts of active items
            if active_item_ids:
                new_thoughts_for_retrieval = [batch_thoughts[item_id][-1] for item_id in active_item_ids]
                # new_retrieval_results, new_scoress = self.retriever.batch_search(new_thoughts_for_retrieval, return_score=True)
                new_retrieval_results, new_scoress = [], []
                for new_thought_for_retrieval in new_thoughts_for_retrieval:
                    # results, new_score = search(new_thought_for_retrieval, top_n=5, return_score=True)
                    # new_retrieval_results.append(results)
                    # new_scoress.append(new_score)
                    if new_thought_for_retrieval == '':
                        print(new_thought_for_retrieval)
                        assert False
                    search_results_ = search(new_thought_for_retrieval, top_n=5, return_score=True)
                    if isinstance(search_results_, dict):
                        # print(search_results_)
                        # assert False
                        search_results = search_results_['results']
                        scores = search_results_['scores']
                    else:
                        search_results, scores = search_results_
                    processed_results = []
                    for result in search_results:
                        split_index = result.find('\n')
                        if split_index != -1:
                            title = result[:split_index]
                            contents = result[split_index+1:]
                            # if len(contents) > 750:
                            #     contents = contents[:750] + "..."
                            content_hash = hash(contents)
                            if content_hash not in seen_contents:
                                seen_contents.add(content_hash)
                                processed_results.append({"id": f"doc_{doc_idx}", "title": title, "contents": contents})
                                doc_idx += 1
                        else:
                            content_hash = hash(result)
                            if content_hash not in seen_contents:
                                seen_contents.add(content_hash)
                                processed_results.append({"id": f"doc_{doc_idx}", "title": "", "contents": result})
                                doc_idx += 1
                    new_retrieval_results.append(processed_results)
                    new_scoress.append(scores)

                # 遍历所有活跃的项目ID
                for i, item_id in enumerate(active_item_ids):
                    # 获取当前项目的新检索结果和对应的分数
                    new_retrieval_result, new_scores = new_retrieval_results[i], new_scoress[i]
                    
                    # 更新当前项目的文档到分数映射和ID到文档映射
                    for doc_item, score in zip(new_retrieval_result, new_scores):
                        # 获取文档ID
                        doc_id = doc_item['id']
                        # 将文档ID和文档内容存入映射表
                        id2doc_batch[item_id][doc_id] = doc_item
                        # 如果文档ID已存在于分数映射表中，取最高分数
                        if doc_id in doc2score_batch[item_id]:
                            doc2score_batch[item_id][doc_id] = max(doc2score_batch[item_id][doc_id], score)
                        # 否则，直接添加新的文档ID和分数
                        else:
                            doc2score_batch[item_id][doc_id] = score

                    # 根据分数对文档进行排序并更新检索结果
                    # 按分数从低到高排序文档（reverse=False表示升序）
                    sorted_doc_score = sorted(doc2score_batch[item_id].items(), key=lambda x: x[1], reverse=False)
                    # 提取排序后的文档ID列表
                    sorted_doc_id = [t[0] for t in sorted_doc_score]
                    # 根据排序后的文档ID更新批处理检索结果
                    batch_retrieval_results[item_id] = [id2doc_batch[item_id][id] for id in sorted_doc_id]
                # print('========= 第一个问题的检索文档数量 =========')
                # print(len(batch_retrieval_results[active_item_ids[0]]))

            iter_num += 1

        # Final update for each item in the batch
        for item_id, item in enumerate(items):
            item.update_output('retrieval_result', batch_retrieval_results[item_id])
            item.update_output('pred', ' '.join(batch_thoughts[item_id]))
            item.update_output('scores', batch_scores[item_id])

    def run(self, dataset, do_eval=True, pred_process_fun=ircot_pred_parse):

        self.run_batch(dataset)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

class RQRAGPipeline(BasicPipeline):
    expand_on_tokens = [
        "[S_Rewritten_Query]",
        "[S_Decomposed_Query]",
        "[S_Disambiguated_Query]",
        "[A_Response]"
    ]
    
    system_prompt = {
        "qa": "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response."
    }
    
    response_generation_params = {
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["[EOS]", "</s>"],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": 1,
        "spaces_between_special_tokens": False,
        "max_tokens": 4096
    }
    
    other_generation_params = {
        "temperature": 1,
        "top_p": 0.9,
        "stop": ["[EOS]", "</s>"],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": 1,
        "spaces_between_special_tokens": False,
        "max_tokens": 4096
    }

    def __init__(
        self,
        config: dict,
        prompt_template = None,
        retriever = None,
        generator = None,
        max_depth = 3,
        batch_size = 32
    ):
        super().__init__(config, prompt_template)

        self.generator = generator if generator is not None else get_generator(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["generator_model_path"], padding_side = "left")
        # self.retriever = retriever if retriever is not None else get_retriever(config)
        
        self.max_depth = max_depth
        self.batch_size = batch_size
        
        # Due to the low effiency of original method, it only supports vllm now.
    
    def preprocess_eval_data(self, items: List) -> List[str]:
        eval_examples = []

        for item in items:
            eval_example = f"<s><|system|>\n{self.system_prompt['qa']}" + self.tokenizer.eos_token + "\n<|user|>\n" + item.question + self.tokenizer.eos_token + "\n"
            eval_example += "<|assistant|>\n"
            eval_examples.append(eval_example)

        return eval_examples

    def format_evidences(self, evidences: List[str]):
        format_evidence = ""
        for evidence in evidences:
            title = evidence['contents'].split('\n')[0]
            text = "\n".join(evidence['contents'].split('\n')[1:])
            format_evidence += f"Title: {title}\n"
            format_evidence += f"Text: {text}\n"
        return format_evidence

    def generate_tree_of_thoughts_batch(self, initial_prompts_batch: List[str]):
        paths_batch_dict = {
            idx: [{
                "prompt": initial_prompt,
                "depth": 0,
                "done": False
            }]
            for idx, initial_prompt in enumerate(initial_prompts_batch)
        }
        
        final_outputs_batch = {idx: [] for idx in range(len(initial_prompts_batch))}
        
        while any(paths for paths in paths_batch_dict.values()):
            current_batch = []
            for i, _ in paths_batch_dict.items():
                if paths_batch_dict[i]:
                    current_path = paths_batch_dict[i].pop(0)
                    current_batch.append(current_path)
                else:
                    continue
            
            if not current_batch:
                break
            
            for special_token in self.expand_on_tokens:
                
                if current_batch[0]["depth"] >= self.max_depth and special_token != "[A_Response]":
                    continue
                
                # Prepare for inputs
                input_texts = [path["prompt"] + special_token for path in current_batch]
            
                # Generate outputs
                if special_token != "[A_Response]":
                    init_outputs = self.generator.generate(
                        input_list = input_texts,
                        return_raw_output = True,
                        **self.response_generation_params
                    )
                else:
                    init_outputs = self.generator.generate(
                        input_list = input_texts,
                        return_raw_output = True,
                        **self.other_generation_params
                    )

                # Decode outputs
                decoded_outputs = [output.outputs[0].text for output in init_outputs]
                # Initialize lists to collect queries for batch retrieval
                queries_for_search = []
                
                # Process outputs and prepare for retrieval
                for i, decoded_output in enumerate(decoded_outputs):
                    current_path = current_batch[i]
                    decoded_output = decoded_output.replace("<s> ", "<s>")
                    
                    if special_token == "[A_Response]":
                        pattern = r"(.*?)\[EOS\]"
                        matches = re.findall(pattern, decoded_output, re.DOTALL)
                        result = matches[-1].strip() if matches else "Unable to detect valid answer"
                        token_ids = init_outputs[i].outputs[0].token_ids[1:-1]
                        logprobs = init_outputs[i].outputs[0].logprobs[1:-1]
                        confidence = 0
                        for token_id, logprobs in zip(token_ids, logprobs):
                            logprob = logprobs[token_id].logprob
                            prob = math.exp(logprob)
                            confidence += prob
                        
                        if len(token_ids) > 0:
                            confidence /= len(token_ids)
                        
                        new_path = {
                            "prompt": input_texts[i] + decoded_output,
                            "depth": current_path["depth"],
                            "done": True,
                            "final_answer": result,
                            "confidence": confidence
                        }
                        final_outputs_batch[i].append(new_path)
                    else:
                        # Extract the query
                        pattern = r"(.*?)\[EOS\]"
                        matches = re.findall(pattern, decoded_output, re.DOTALL)
                        query_for_search = matches[-1].strip() if matches else "dummy"
                        queries_for_search.append(query_for_search)
                
                # Perform batch retrieval
                if queries_for_search:
                    # batch_search_results = self.retriever.batch_search(queries_for_search)
                    batch_search_results = [search(query) for query in queries_for_search]
                    
                    for i, decoded_output in enumerate(decoded_outputs):
                        search_results = batch_search_results[i]
                        format_evidence = self.format_evidences(search_results)
                        new_prompt = decoded_output + "[R_Evidences]" + format_evidence + "[/R_Evidences]"
                        new_path = {
                            "prompt": input_texts[i] + new_prompt,
                            "depth": current_path["depth"] + 1,
                            "done": False,
                        }
                        paths_batch_dict[i].append(new_path)

        final_outputs_batch_list = [final_outputs_batch[i] for i in range(len(initial_prompts_batch))]
        
        return final_outputs_batch_list

    def select_best_path_single_turn(self, final_outputs):
        # After generating all paths, we can select the best answer
        # Compute perplexity and confidence for each path
        
        scores = []
        for path in final_outputs:
            confidence = path["confidence"]
            path["confidence"] = confidence
            scores.append((path, confidence))

        # Select the path with the highest confidence
        best_path = max(scores, key = lambda x: x[1])[0]  # x[2] is confidence
        pred = best_path["final_answer"]

        return pred, best_path

    def run(self, dataset, do_eval = True):
        preds = []
        meta_results = []

        for i in tqdm(range(0, len(dataset), self.batch_size), position=0, desc='RQRAG Process'):
            batch_items = dataset[i : i + self.batch_size]
            eval_datas = self.preprocess_eval_data(batch_items)
            paths_batch = self.generate_tree_of_thoughts_batch(initial_prompts_batch = eval_datas)
            for paths in paths_batch:
                pred, best_path = self.select_best_path_single_turn(paths)
                preds.append(pred)
                meta_results.append(best_path)


        dataset.update_output("paths", meta_results)
        dataset.update_output("pred", preds)

        dataset = self.evaluate(dataset, do_eval = do_eval)
        return dataset
    
