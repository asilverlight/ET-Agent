import time
import hashlib
from .utils import *

class SampleProcessor:
    def __init__(
        self,
        prompt_manager,
        tool_executor,
        vllm_pool,
        tokenizer,
        args,
        sample_stat,
        session_id,
    ):
        self.prompt_manager = prompt_manager
        self.tool_executor = tool_executor
        self.vllm_pool = vllm_pool
        self.tokenizer = tokenizer
        self.args = args
        self.sample_stat = sample_stat
        self.question = question = sample_stat.get("question", sample_stat["input"])
        self.choice = sample_stat["choice"]
        self.format = sample_stat["format"]
        system_prompt = self.prompt_manager.get_system_prompt(self.format)

        if not session_id:
            session_content = f"{system_prompt}_{question}"
            session_id = hashlib.md5(session_content.encode()).hexdigest()
        self.session_id = session_id

        self.sample_start_time = None
        self.llm_time = 0
        self.python_time = 0
        self.search_time = 0
        self.total_time = None
        self.python_rounds = 0
        self.search_rounds = 0
        self.in_context = ""
        if self.format == 'Multiple-choice':
            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{self.sample_stat['input']}\n\n{make_choices_format(self.choice)}"},
            ]
        else:
            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.sample_stat['input']},
            ]

    def log_output(self, role: str, content: str):
        self.sample_stat["choice"] = self.choice
        self.sample_stat["format"] = self.format
        if "output" not in self.sample_stat:
            self.sample_stat["output"] = ""
        self.sample_stat["output"] += content
        self.sample_stat["logs"].append(content)
        self.in_context += content

    def process_input(self):
        self.in_context = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

    async def call_local_llm(self, stop) -> str:
        in_context = self.in_context

        all_output = ""
        try_time = 0
        while try_time < 4:
            try_time += 1
            llm_start = time.time()
            result = await self.vllm_pool.generate(
                in_context,
                (
                    self.args.sampling_params
                    if stop is True
                    else self.args.sampling_params_nostop
                ),
                session_id=self.session_id,
            )
            self.llm_time += time.time() - llm_start
            if not result:
                # print("The LLM fails to generate output!\n")
                return 'None' if not all_output else all_output
            output = result.choices[0].text
            output = output.split("<result>")[0]
            if "</search>" in output:
                output = output.split("</search>")[0] + "</search>"
            if "</python>" in output:
                output = output.split("</python>")[0] + "</python>"
            if "</answer>" in output:
                output = output.split("</answer>")[0] + "</answer>"
            all_output += output
            if (
                "</search>" not in all_output
                and "</python>" not in all_output
                and "</answer>" not in all_output
            ):
                in_context += output
            else:
                break
        return all_output

    async def call_llm(self, stop=True):
        output = await self.call_local_llm(stop)
        self.log_output("assistant", output)
        return output

    async def call_python(self, python_code: str):
        python_start = time.time()
        python_result = await self.tool_executor.execute(
            "python", python_code, timeout=120
        )
        self.python_time += time.time() - python_start
        tool_result = f"<result>{python_result}</result>"
        self.log_output("user", tool_result)
        self.python_rounds += 1

    async def call_search(self, search_query: str):
        search_start = time.time()
        if not self.args.compatible_search:
            if self.args.use_local_search:
                search_result = await self.tool_executor.execute(
                    "localsearch",
                    search_query,
                    timeout=120,
                    sample_stat=self.sample_stat,
                )
            else:
                search_result = await self.tool_executor.execute(
                    "websearch",
                    search_query,
                    timeout=120,
                    sample_stat=self.sample_stat,
                )
        else:
            if self.args.dataset_name in ['2wiki', 'bamboogle', 'musique', 'hotpotqa', 'csbench']:
                search_result = await self.tool_executor.execute(
                    "localsearch",
                    search_query,
                    timeout=120,
                    sample_stat=self.sample_stat,
                )
            else:
                search_result = await self.tool_executor.execute(
                    "websearch",
                    search_query,
                    timeout=120,
                    sample_stat=self.sample_stat,
                )
        self.search_time += time.time() - search_start
        if search_query is None or search_result is None:
            tool_result = f"<result></result>"
        else:
            tool_result = f"<result>{search_result}</result>"
        self.log_output("user", tool_result)
        self.search_rounds += 1

    async def run(self):
        """Process one QA pair..."""
        self.sample_start_time = time.time()
        self.process_input()
        while True:
            if self.args.use_log:
                print(f"current_prompt:\n{self.in_context}")
            output = await self.call_llm()
            if self.args.use_log:
                print(f"current_output:\n{output}")
            if not output:
                break
            tool_tag = self.tool_executor.identify_tool(output)
            if tool_tag == "python" and self.python_rounds < self.args.max_python_times:
                python_code = self.tool_executor.extract_content(output, "python")
                await self.call_python(python_code)
            elif (
                tool_tag == "search" and self.search_rounds < self.args.max_search_times
            ):
                search_query = self.tool_executor.extract_content(output, "search")
                await self.call_search(search_query)
            else:
                if not output.strip().endswith("</answer>"):
                    output = await self.call_llm(stop=False)
                break

        if "output" not in self.sample_stat:
            self.sample_stat["output"] = ""
        self.sample_stat["prediction"] = extract_answer(self.sample_stat["output"])
        self.total_time = time.time() - self.sample_start_time

    def log_timing(self):
        self.sample_stat["timing"] = {
            "llm_time": self.llm_time,
            "python_time": self.python_time,
            "search_time": self.search_time,
            "total_time": self.total_time,
        }


class SampleProcessorCompletion(SampleProcessor):

    def call_python_max_limit(self):
        limit_message = f"<result>The maximum python call limit is exceeded. You are not allowed to use python.</result>"
        self.log_output("user", limit_message)
        question = self.sample_stat["input"]
        # print(f'Python limit reached for question: "{question}"')

    def call_search_max_limit(self):
        limit_message = f"<result>The maximum search limit is exceeded. You are not allowed to search.</result>"
        self.log_output("user", limit_message)
        question = self.sample_stat["input"]
        # print(f'Search limit reached for question: "{question}"')

    def call_search_same_query(self):
        limit_message = f"<result>You have searched this query. Please refer to previous results.</result>"
        self.log_output("user", limit_message)
        question = self.sample_stat["input"]
        # print(f'Repeated search for question: "{question}"')

    async def run(self):
        self.sample_start_time = time.time()
        self.process_input()
        while True:
            output = await self.call_llm()
            if not output:
                # print("[Warning] LLM inference failed!!!")
                break
            tool_tag = self.tool_executor.identify_tool(output)
            if tool_tag == "python":
                if self.python_rounds < self.args.max_python_times:
                    python_code = self.tool_executor.extract_content(output, "python")
                    await self.call_python(python_code)
                else:
                    self.call_python_max_limit()
            elif tool_tag == "search":
                if self.search_rounds < self.args.max_search_times:
                    search_query = self.tool_executor.extract_content(output, "search")
                    await self.call_search(search_query)
                else:
                    self.call_search_max_limit()
            else:
                # if "</answer>" not in output:
                #     print(
                #         "[Warning] LLM fails to generate final answers, sample is: ",
                #         self.sample_stat["input"],
                #     )
                break
        self.sample_stat["prediction"] = extract_answer(self.sample_stat["output"])
        self.total_time = time.time() - self.sample_start_time
