import os

from requests_toolbelt.adapters import source
from datetime import datetime
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_string_md5(input_str: str, encoding='utf-8') -> str:
    """将传入的字符串转化为 MD5 字符串"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex

def check_md5(md5_str: str) -> bool:
    """检查传入的 md5 字符串是否已经被处理过（存在于 md5 记录文件中）"""
    if not os.path.exists(config.md5_path):
        # 如果文件不存在，创建空文件并返回 False（未处理）
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == md5_str:
                return True
    return False

def save_md5(md5_str: str) -> None:
    """将 MD5 字符串保存到文件（记录已处理）"""
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)   #确保文件夹存在
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       #每个文本段最大长度
            chunk_overlap=config.chunk_overlap,   #连续文本段字符重叠数量
            separators=config.separators,             #自然段落划分符号
            length_function=len,
        )


        self.splitter = None   # 注意拼写：Splitter -> splitter

    def upload_by_str(self, data, filename):
        """根据字符串内容上传到知识库（待实现）"""
        #先得到md5值
        md5_hex=get_string_md5(data)

        if check_md5(md5_hex):
            return      "[跳过]内容已经在知识库中"
        if len(data) > config.max_split_char_number:
            knowledge_chunks:list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]




        metadata={
            "source":filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"wxy",
        }
        self.chroma.add_texts(
        texts=knowledge_chunks,
        metadatas=[metadata for _ in knowledge_chunks],

        )
        save_md5(md5_hex)
        return"[成功]内容已经成功载入向量库"


if __name__ == '__main__':
    service=KnowledgeBaseService()
    res=service.upload_by_str("wxy","testfile")
    print(res)
    # 测试 check_md5 和 save_md5（需要先确保 config.md5_path 有值）
    test_md5 = get_string_md5("测试内容")
    if not check_md5(test_md5):
        save_md5(test_md5)
        print(f"已保存 MD5: {test_md5}")
    else:
        print(f"MD5 已存在: {test_md5}")
        if __name__ == '__main__':
            kb = KnowledgeBaseService()
            print(dir(kb))  # 打印所有方法名，看里面有没有 'upload_by_str'
            result = kb.upload_by_str("测试", "test.txt")
            print(result)