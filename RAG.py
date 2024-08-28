import chromadb

import os
os.environ["QIANFAN_AK"] = "XXXX"
os.environ["QIANFAN_SK"] = "XXXX"
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import ChatPromptTemplate
chat_model = QianfanChatEndpoint()# 指定模型名称

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 定义文件路径，指向需要加载的PDF文档
file_path = "xxx.pdf"# 指定文件路径
def read_PDF(file_path):
    # 使用PyPDFLoader加载PDF文档
    loader = PyPDFLoader(file_path)
    # 加载并拆分PDF文档的每一页
    pages = loader.load_and_split()
    # 初始化一个空列表，用于存储每个页面的内容和元数据
    pdf_content = []
    # 遍历每一页，将内容和元数据存储到pdf_content列表中
    for page in pages:
        pdf_content.append({'content': page.page_content,
                            'metadata': page.metadata})
    # 将每个页面的内容和元数据转换为Document对象
    documents = [Document(page_content=x['content'], metadata=x['metadata']) for x in pdf_content]
    return documents

def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # 零宽空格
            "\uff0c",  # 全角逗号
            "\u3001",  # 表意逗号
            "\uff0e",  # 全角句号
            "\u3002",  # 表意句号
            "",
        ],
        # 已有的参数
        # 设置一个非常小的块大小，只是为了展示。
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks):
    client = chromadb.PersistentClient(path='/DataBase/chroma_db')# 指定数据库路径
    vectorstore = Chroma.from_documents(
        chunks,
        QianfanEmbeddingsEndpoint(model="bge-large-zh"),
        client=client,
        collection_name="test_collection",
    )
    return vectorstore

def embed_query(query):
    embeddings_model = QianfanEmbeddingsEndpoint(model="bge-large-zh")# 指定模型名称:嵌入模型
    embedded_query = embeddings_model.embed_query(query)
    return embedded_query
def search(vectorstore, query, top_k=5):
    retriever_top_k = vectorstore.as_retriever(search_kwargs={"k": top_k})
    search_results = retriever_top_k.get_relevant_documents(query)
    print("指定前{}个结果:".format(top_k))
    for ans in search_results:
        print("-------------------------------------------------------")
        print(ans.page_content)
    return search_results


if __name__ == "__main__":
    documents = read_PDF(file_path)
    # 输出指定页面的Document对象，这里输出第9页（索引从0开始）
    # print(documents[8])
    chunks = text_split(documents)
    # print(chunks)
    vectorstore = create_vectorstore(chunks)
    query = "XXXX"# 输入查询问题
    # embedded_query = embed_query(query)
    # search(vectorstore, query)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = QianfanChatEndpoint(model="ERNIE-Bot-turbo")
    system_prompt = (
        "您是一个用于问答任务的助手。"
        "使用以下检索到的上下文片段来回答问题。"
        "如果您不知道答案，请说您不知道。"
        "最多使用三句话，保持回答简洁。"
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    print(response["answer"])













