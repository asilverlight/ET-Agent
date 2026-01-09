import itertools
from typing import List
import re
from tqdm import tqdm
import numpy as np
from transformers import LogitsProcessorList
from flashrag.utils import get_generator, search, find_covering_segments
from flashrag.pipeline import BasicPipeline
from flashrag.prompt import PromptTemplate


class REPLUGPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        from flashrag.pipeline.replug_utils import load_replug_model

        super().__init__(config, prompt_template)
        # load specify model for REPLUG
        if generator is None:
            model = load_replug_model(config["generator_model_path"])
            generator = get_generator(config, model=model)
        # if retriever is None:
        #     retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever
    def build_single_doc_prompt(self, question: str, doc_list: List[str], options: str=None):
        if options is not None:
            return [self.prompt_template.get_string(question=question, formatted_reference=doc, options=options) for doc in doc_list]
        return [self.prompt_template.get_string(question=question, formatted_reference=doc) for doc in doc_list]

    def format_reference(self, doc_item):
        content = doc_item["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        return f"Document(Title: {title}): {text}"

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        import torch
        from flashrag.pipeline.replug_utils import REPLUGLogitsProcessor

        trajectories = []

        input_query = dataset.question
        if self.config['dataset_name'] == 'arc':
            options = [' '.join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(choice)]) for choice in dataset.choices]
            dataset.update_output("options", options)
        # retrieval_results, doc_scores = self.retriever.batch_search(input_query, return_score=True) 
        retrieval_results, doc_scores = [], []
        for i in tqdm(range(len(input_query)), desc="Retrieval: "):
            query = input_query[i]
            # results, score = search(query, top_n=5, return_score=True)
            search_results = search(query, top_n=5, return_score=True)
            if isinstance(search_results, dict):
                results = search_results['results']
                scores = search_results['scores']
            else:
                results, scores = search_results
            
            # processed_results = []
            # for result in results:
            #     split_index = result.find('\n')
            #     if split_index != -1:
            #         title = result[:split_index]
            #         content = result[split_index+1:]
            #         processed_results.append({"title": title, "contents": content})
            #     else:
            #         processed_results.append({"title": "", "contents": result})
            for i in range(len(results)):
                results[i] = results[i].split('\n')
                results[i] = {
                    "title": results[i][0],
                    "contents": "\n".join(results[i][1:])
                }
            retrieval_results.append(results)
            doc_scores.append(scores)
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("doc_scores", doc_scores)

        pred_answer_list = []
        final_scores = []
        # each doc has a prompt
        for i in tqdm(range(len(dataset)), desc="Inference: "):
            item = dataset[i]
            trajectory = ''
            docs = [self.format_reference(doc_item) for doc_item in item.retrieval_result]
            if self.config['dataset_name'] == 'arc':
                options = item.options
                prompts = self.build_single_doc_prompt(question=item.question, doc_list=docs, options=options)
            else:
                prompts = self.build_single_doc_prompt(question=item.question, doc_list=docs)
            # print(prompts[0])
            trajectory += '\n'.join(docs) + '\n\n'
            scores = torch.tensor(item.doc_scores, dtype=torch.float32).to(self.device)
            if self.config['return_prob']:
                output, log_probs = self.generator.generate(
                    prompts, batch_size=len(docs), logits_processor=LogitsProcessorList([REPLUGLogitsProcessor(scores)]), return_scores=True
                )
            else:
                output = self.generator.generate(
                    prompts, batch_size=len(docs), logits_processor=LogitsProcessorList([REPLUGLogitsProcessor(scores)])
                )
            # trajectory += output[0]
            trajectories.append(trajectory)
            # the output of the batch is same
            output = output[0]
            log_probs = log_probs[0]# 一个list，里面都是数
            pred_answer_list.append(output)
            final_scores.append(log_probs)

        dataset.update_output("pred", pred_answer_list)
        dataset.update_output("scores", final_scores)
        dataset.update_output("trajectory", trajectories)

        # for i in range(len(pred_answer_list)):
        #     output_tokens = self.generator.tokenizer.encode(pred_answer_list[i], add_special_tokens=False)
        #     detokenized_answer = [self.generator.tokenizer.decode(token) for token in output_tokens]
        #     begin_end = find_covering_segments(pred_answer_list[i], detokenized_answer, detokenized_answer)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        if self.config['return_prob']:
            return dataset, final_scores
        return dataset


class SuRePipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        if generator is None:
            generator = get_generator(config)
        # if retriever is None:
        #     retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.load_prompts()

    def load_prompts(self):
        # prompt for candidates generation
        P_CAN_INSTRUCT = (
            "Below are {N} passages related to the question at the end. After reading"
            "the passages, provide two correct candidates for the answer to the"
            "question at the end. Each answer should be in the form: (a) xx, (b)"
            "yy, and should not exceed 3 words for each candidate.\n\n"
            "{reference}"
            "Question: {question}\n"
            "Answer:"
        )

        # prompt for candidate-conditioned summarization
        P_SUM_INSTRUCT = (
            "Reference:\n{reference}\n"
            "Your job is to act as a professional writer. You need to write a"
            "good-quality passage that can support the given prediction about the"
            "question only based on the information in the provided supporting passages.\n"
            "Now, let's start. After you write, please write [DONE] to indicate you"
            "are done. Do not write a prefix (e.g., 'Response:') while writing a passage.\n"
            "Question: {question}\n"
            "Prediction: {pred}\n"
            "Passage:"
        )

        # prompt for instance-wise validation
        P_VAL_INSTRUCT = (
            "Question: {question}\n"
            "Prediction: {pred}\n"
            "Passage: {summary}\n"
            "Does the passage correctly support the prediction? Choices: [True,False].\n"
            "Answer:"
        )

        # prompt for pair-wise ranking
        P_RANK_INSTRUCT = (
            "Question: Given the following passages, determine which one provides a"
            "more informative answer to the subsequent question.\n"
            "Passage 1: {summary1}\n"
            "Passage 2: {summary2}\n"
            "Target Question: {question}\n"
            "Your Task:\n"
            "Identify which passage (Passage 1 or Passage 2) is more relevant and"
            "informative to answer the question at hand. Choices: [Passage 1,Passage 2].\n"
            "Answer:"
        )

        self.P_CAN_TEMPLATE = PromptTemplate(self.config, "", P_CAN_INSTRUCT)
        self.P_SUM_TEMPLATE = PromptTemplate(self.config, "", P_SUM_INSTRUCT)
        self.P_VAL_TEMPLATE = PromptTemplate(self.config, "", P_VAL_INSTRUCT)
        self.P_RANK_TEMPLATE = PromptTemplate(self.config, "", P_RANK_INSTRUCT)

    @staticmethod
    def format_ref(titles, texts):
        formatted_ref = ""
        idx = 1
        for title, text in zip(titles, texts):
            formatted_ref += f"Passage #{idx} Title: {title}\n"
            formatted_ref += f"Passage #{idx} Text: {text}\n"
            formatted_ref += "\n"
            idx += 1
        return formatted_ref

    @staticmethod
    def parse_candidates(model_response):
        """Parse candidates from model response"""
        model_response = model_response.strip("\n").strip()
        # r'\([a-z]\) ([^,]+)'
        candidates = re.findall("\((\w+)\)\s*([^()]+)", model_response)
        candidates = [cand[1].split("\n")[0].strip() for cand in candidates]
        # post-process
        candidates = [cand.replace(",", "").strip() for cand in candidates]
        return candidates

    @staticmethod
    def parse_validation(model_response):
        """Parse model's validation result into score based on the paper formula"""
        model_response = model_response.strip().lower()
        if "true" in model_response:
            return 1
        else:
            return 0

    @staticmethod
    def parse_ranking(model_response):
        """Parse model's pair ranking result into score"""
        model_response = model_response.strip().lower()
        if "passage 1" in model_response:
            score = 1
        elif "passage 2" in model_response:
            score = 0
        else:
            score = 0.5
        return score

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        # retrieval_results, doc_scores = self.retriever.batch_search(input_query, return_score=True)
        retrieval_results, doc_scores = [], []
        for query in input_query:
            result, scores = search(query, return_score=True)
            retrieval_results.append(result)
            doc_scores.append(scores)
        dataset.update_output("retrieval_result", retrieval_results)

        pred_answer_list = []
        trajectories = []
        for item in tqdm(dataset, desc="Pipeline runing: "):
            trajectory = ''
            retrieval_result = item.retrieval_result
            doc_num = len(retrieval_result)
            # format all docs
            temp_retrieval_result = []
            for doc_item in retrieval_result:
                # if "title" not in doc_item or "text" not in doc_item:
                #     doc_item["title"] = doc_item["contents"].split("\n")[0]
                #     doc_item["text"] = "\n".join(doc_item["contents"].split("\n")[1:])
                temp_retrieval_result.append(
                    {
                        'title': doc_item.split("\n")[0],
                        'text': "\n".join(doc_item.split("\n")[1:])
                    }
                )
            formatted_ref = self.format_ref(
                titles=[i["title"] for i in temp_retrieval_result], texts=[i["text"] for i in temp_retrieval_result]
            )
            trajectory += formatted_ref
            # get candidates

            input_prompt = self.P_CAN_TEMPLATE.get_string(
                N=doc_num, formatted_reference=formatted_ref, question=item.question
            )
            trajectory += input_prompt + '\n'
            output = self.generator.generate([input_prompt])[0]
            candidates = self.parse_candidates(output)
            item.update_output("candidates", candidates)
            trajectory += output + '\n'
            if len(candidates) == 0:
                print("No valid predictions!")
                pred = ""
                pred_answer_list.append(pred)
                continue

            # get summarization for each candidate
            input_prompts = [
                self.P_SUM_TEMPLATE.get_string(question=item.question, pred=cand, formatted_reference=formatted_ref)
                for cand in candidates
            ]
            temp_input_prompts1 = [f' {input_prompt}\n' for i, input_prompt in enumerate(input_prompts)]

            all_summary = self.generator.generate(input_prompts)
            temp_all_summary = [f'Summary: {summary}' for i, summary in enumerate(all_summary)]
            item.update_output("all_summary", all_summary)
            # instance-wise validation
            input_prompts = [
                self.P_VAL_TEMPLATE.get_string(question=item.question, pred=cand, summary=summary)
                for cand, summary in zip(candidates, all_summary)
            ]
            temp_input_prompts2 = [f'{input_prompt}\n' for i, input_prompt in enumerate(input_prompts)]
            val_results = self.generator.generate(input_prompts)
            val_scores = [self.parse_validation(res) for res in val_results]
            item.update_output("val_scores", val_scores)

            for i in range(len(temp_input_prompts1)):
                trajectory += temp_input_prompts1[i] + temp_all_summary[i] + temp_input_prompts2[i] + val_results[i] + '\n'

            # pair-wise ranking
            summary_num = len(all_summary)
            score_matrix = np.zeros((summary_num, summary_num))
            # 这行代码创建了一个包含所有可能的两两排列组合的列表
            # itertools.permutations(range(summary_num), 2) 生成所有可能的摘要索引对(i,j)，其中i≠j
            # 例如，如果summary_num=3，则生成 (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
            # 这些索引对将用于后续的摘要两两比较排名
            iter_idxs = list(itertools.permutations(range(summary_num), 2))
            input_prompts = [
                self.P_RANK_TEMPLATE.get_string(
                    question=item.question, summary1=all_summary[idx_tuple[0]], summary2=all_summary[idx_tuple[1]]
                )
                for idx_tuple in iter_idxs
            ]
            ranking_output = self.generator.generate(input_prompts)
            for i in range(len(input_prompts)):
                trajectory += input_prompts[i] + ranking_output[i] + '\n'
            ranking_scores = [self.parse_ranking(res) for res in ranking_output]
            for idx_tuple, score in zip(iter_idxs, ranking_scores):
                score_matrix[idx_tuple[0], idx_tuple[1]] = score
            ranking_scores = score_matrix.sum(axis=1).squeeze().tolist()  # ranking score for each summary
            item.update_output("ranking_scores", ranking_scores)

            # combine two scores as the final score for each summary
            if not isinstance(ranking_scores, list):
                ranking_scores = [ranking_scores]
            if not isinstance(val_scores, list):
                val_scores = [val_scores]
            total_scores = [x + y for x, y in zip(val_scores, ranking_scores)]

            best_idx = np.argmax(total_scores)
            pred = candidates[best_idx]
            pred_answer_list.append(pred)
            trajectories.append(trajectory)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset, trajectories
