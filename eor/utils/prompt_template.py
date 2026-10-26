import string

from .openai_tools import CHAT_MODEL_LIST

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""

# Llama 3.1 使用更简洁的系统提示
LLAMA3_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

CHATGPT_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

class PromptTemplate:
    def __init__(self, language: str, model_name: str, template: str, system_message: str = None,
                 template_id: int = None, task_name: str = None, use_system_message: bool = True):
        """
        language: str - specific the language of instruction, support 'en' and 'zh'
        model_name: str - model_name help to determine whether to use chat_completion format for turbo/gpt-4
        or completion for normal causal LM
        template: str - template str with placeholder around with curly bracket
        system_message: str - while model_name is turbo/gpt-4, system message is need to use chat_completions
        return_fill_info: bool - return fill stage info contains template, text, template id, task name
        template_id: str - if return fill info, template id will also be contained
        task_name: str - if return info, task name will also be contained
        """
        assert language in ["zh", "en"], "please use 'zh' for chinese, 'en' for english."
        self.language = language
        self.template = template
        self.placeholders = self.parse_template_placeholder(template)
        self.template_id = template_id
        self.task_name = task_name
        
        self.use_system_message = use_system_message
        self.model_name = model_name.lower()  # 保存模型名称用于判断

        if model_name in CHAT_MODEL_LIST:
            self.chat = True
            if system_message is None:
                system_message = CHATGPT_DEFAULT_SYSTEM_PROMPT
        else:
            self.chat = False
        
        self.system_message = system_message

        # 检测模型类型并应用相应的模板格式
        if "llama" in self.model_name:
            if "llama3" in self.model_name or "llama-3" in self.model_name or "3.1" in self.model_name:
                # Llama 3.x 系列 (包括 llama3.1-8b-instruct)
                self.template = self.build_llama3_instruct_template(self.template)
            elif "chat" in self.model_name:
                # Llama 2 chat 系列
                self.template = self.build_llama2_chat_template(self.template)
        

    def build_openai_chat_message(self, prompt):
        """
        change input format from completion to chat
        parameters:
        prompt: str
        """
        system_message = {"role": "system", "content": self.system_message}
        query_message = {"role": "user", "content": prompt}
        return [system_message, query_message]
    
    def build_llama2_chat_template(self, prompt):
        """
        modify from llama2 tokenizer _build_conversation_input_ids
        """
        if self.system_message is None:
            self.system_message = LLAMA2_DEFAULT_SYSTEM_PROMPT
        llama2_bos_token = "<s>"
        llama2_B_INST, llama2_E_INST = "[INST]", "[/INST]"
        llama2_B_SYS, llama2_E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        if (not prompt.startswith(llama2_B_SYS) or llama2_E_SYS not in prompt) and self.use_system_message:
            system_info = llama2_B_SYS + self.system_message.strip() + llama2_E_SYS
        else:
            system_info = ""
        return f"{llama2_bos_token}{llama2_B_INST} {system_info}{prompt.strip()} {llama2_E_INST}"

    def build_llama3_instruct_template(self, prompt):
        """
        Build prompt template for Llama 3.x Instruct models.
        Llama 3 使用新的特殊token格式:
        <|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        注意: 不添加 <|begin_of_text|>，因为 tokenizer 会自动添加
        """
        if self.system_message is None:
            self.system_message = LLAMA3_DEFAULT_SYSTEM_PROMPT
        
        # Llama 3 special tokens (不包含 BOS，tokenizer 会自动添加)
        START_HEADER = "<|start_header_id|>"
        END_HEADER = "<|end_header_id|>"
        EOT = "<|eot_id|>"
        
        parts = []
        
        # System message (如果使用)
        if self.use_system_message and self.system_message:
            parts.append(f"{START_HEADER}system{END_HEADER}\n\n")
            parts.append(f"{self.system_message.strip()}{EOT}")
        
        # User message
        parts.append(f"{START_HEADER}user{END_HEADER}\n\n")
        parts.append(f"{prompt.strip()}{EOT}")
        
        # Assistant header (让模型开始生成)
        parts.append(f"{START_HEADER}assistant{END_HEADER}\n\n")
        
        return "".join(parts)

    def build_prompt(self, text):
        """
        fill in the templates
        parameters:
        text: Dict{str: str} - key is the name of placeholder in the template, value is the str to be fill

        return:
        prompt: List[str] or List[List[dict]] - list of prompts generated. If completion, return List[str],
        if chat completion, List[List[dict]]
        info: fill info
        """
        # TODO add clean function to process query and doc in text
        prompt, info = self.fill(text)

        if self.chat:
            prompt = self.build_openai_chat_message(prompt)
            info["system_message"] = self.system_message

        return prompt, info

    def fill(self, text):
        """
        fill text in prompt placeholders.
        :param text: dict - key is placeholder name, value is the str to be fill in
        :return:
        instruction: prompt with placeholders filled
        """
        assert set(self.placeholders).issubset(
            set(text.keys())), f"{set(self.placeholders) - set(text.keys())} should be given"

        prompt = self.template.format(**text)

        info = {
            "template": self.template,
            "text": text,
            "result": prompt,
            "template_id": self.template_id,
            "task_name": self.task_name
        }

        return prompt, info

    @staticmethod
    def parse_template_placeholder(template):
        """
        return the name of all placeholders in the template

        Parameters:
        template: str - normally there should be placeholder(s) around with curly bracket inside.

        Return:
        placeholders: list -  list of all placeholder name in the template

        examples:
        instructions = "请用不超过50字的摘要如实的总结下述文档,摘要应该包含用于回答'{query}'这一问题的最主要的信息：{doc}\n"
        parse_template_placeholder(instructions)

        -> ['query', 'doc']
        """
        placeholders = []

        for parse_result in string.Formatter().parse(template):
            name = parse_result[1]
            if name is None:
                continue
            elif name == "":
                raise TypeError(f"no placeholder name is given in '{template}', check curly bracket")
            else:
                placeholders.append(name)

        return placeholders
