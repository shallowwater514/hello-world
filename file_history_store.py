from typing import Sequence, List

from langchain_core.chat_history import BaseChatMessageHistory
import os
import json
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from prompt_toolkit.history import FileHistory


def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    # 初始化方法，创建实例时自动调用
    def __init__(self, session_id: str, storage_path: str = "./chat_history"):
        # session_id: 会话标识符，比如 "user_001"，不同用户的对话分开存
        self.session_id = session_id
        # storage_path: 存放历史文件的文件夹路径，默认是当前目录下的 chat_history 文件夹
        self.storage_path = storage_path
        # os.makedirs 创建文件夹，exist_ok=True 表示如果文件夹已存在就不报错
        os.makedirs(self.storage_path, exist_ok=True)
        # 拼接完整的文件路径：文件夹路径 + 会话id + .json，例如 "./chat_history/user_001.json"
        self.file_path = os.path.join(self.storage_path, f"{session_id}.json")

    # 这个方法是 LangChain 要求的：添加一批新消息到历史里
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # 先读取已有的消息（调用下面的 messages 属性，它会从文件加载）
        existing = self.messages   # 这里 self.messages 会触发 @property 下面的方法
        # 把已有的和新来的消息合并成一个列表
        all_messages = list(existing) + list(messages)
        # 把消息对象列表转换成字典列表，因为 json.dump 只能存字典/列表等基本类型，不能直接存对象
        dict_list = [message_to_dict(msg) for msg in all_messages]
        # 以写模式打开 JSON 文件，utf-8 编码保证中文正常
        with open(self.file_path, "w", encoding="utf-8") as f:
            # json.dump 把字典列表写入文件，ensure_ascii=False 让中文不是 \uXXXX 形式
            json.dump(dict_list, f, ensure_ascii=False, indent=2)

    # @property 装饰器：下面这个方法可以像属性一样被访问，比如 history.messages 会执行这个方法
    @property
    def messages(self) -> List[BaseMessage]:
        """读取文件中的所有消息，返回 BaseMessage 对象列表"""
        try:
            # 尝试打开 JSON 文件并读取内容
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)        # data 是一个字典列表
                # messages_from_dict 把字典列表变回 BaseMessage 对象列表
                return messages_from_dict(data)
        except FileNotFoundError:
            # 如果文件不存在（第一次运行），返回空列表
            return []

    # 清空当前会话的所有历史消息
    def clear(self) -> None:
        # 用空列表覆盖文件内容
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
