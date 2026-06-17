from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        """param embedding: 嵌入模型实例"""
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回检索器，k 为返回的相似文档数量"""
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings

    # 初始化服务（使用与上传时相同的嵌入模型）
    vector_service = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4"))
    retriever = vector_service.get_retriever()

    # 检索测试
    query = "我的体重190斤，应该穿什么码的衣服"
    docs = retriever.invoke(query)

    if docs:
        print(f"找到 {len(docs)} 条相关内容：")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:200]}")
    else:
        print("未找到相关内容，请确认向量库中已有数据。")