from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from operator import itemgetter
import config_data as config
from file_history_store import get_history
from vector_store import VectorStoreService
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

class RagService:
    def __init__(self):
        # 初始化向量服务（注意：这里假设 VectorStoreService 内部有 vector_store 属性）
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的已知参考资料为主。简洁并且专业地回答用户问题。参考资料:\n{context}"),
            MessagesPlaceholder("history"),
            ("user", "用户提问: {input}")
        ])
        self.chat_model = ChatTongyi(model=config.chat_model_name,streaming=True)
        self.chain = self._build_chain()

    def _format_docs(self, docs: list[Document]) -> str:
        if not docs:
            return "无参考资料"
        return "\n\n".join([f"文档片段: {d.page_content}\n元数据: {d.metadata}" for d in docs])

    def _retrieve(self, query: str) -> list[Document]:
        # 直接使用 vector_store 的 similarity_search
        return self.vector_service.vector_store.similarity_search(query, k=3)

    def _build_chain(self):
        # 1. 定义一个检索步骤（RunnableLambda 包装函数）
        retrieve_step = RunnableLambda(lambda inputs: self._retrieve(inputs["input"]))

        # 2. 定义格式化步骤
        format_step = RunnableLambda(lambda docs: self._format_docs(docs))

        # 3. 构建并行字典：context = 检索+格式化, input 和 history 直接透传
        chain = (
            {
                "context": retrieve_step | format_step,
                "input": itemgetter("input"),
                "history": itemgetter("history"),
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        # 4. 包装历史记忆
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

if __name__ == "__main__":
    session_config = {"configurable": {"session_id": "user_001"}}
    rag = RagService()
    res = rag.chain.invoke({"input": "我体重180斤，推荐什么尺码"}, session_config)
    print(res)