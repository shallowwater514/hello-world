from langchain_chroma import Chroma
import os
import config_data as config
import hashlib
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str: str) -> bool:
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == md5_str:
                return True
    return False

def save_md5(md5_str: str) -> None:
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str, encoding='utf-8') -> str:
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()

class KnowledgeBaseService:
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename: str) -> str:
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过] 内容已经存在知识库中"

        # 分块处理
        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 元数据（每个块都带上）
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "operator": "王新宇"
        }
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)
        return "[成功] 内容已经载入向量库"

# 单独测试时运行
if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("秦失其鹿，天下共逐之", "testfile")
    print(r)