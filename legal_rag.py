import os
import shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv()

# ============================================================
# 初始化
# ============================================================

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

app = FastAPI(title="法律法规问答API")


# ============================================================
# 构建知识库
# ============================================================

def load_legal_documents(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".txt"]:
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")

            docs = loader.load()
            all_docs.extend(docs)
            print(f"已加载：{filename}，{len(docs)}页")
        except Exception as e:
            print(f"加载失败：{filename}，{e}")

    print(f"\n共加载{len(all_docs)}页文档")
    return all_docs


def build_knowledge_base(folder_path, db_path="./legal_db"):
    if os.path.exists(db_path):
        print("加载已有知识库...")
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    print("构建法律知识库...")
    docs = load_legal_documents(folder_path)

    # 法律文档切片策略：chunk_size稍大，保留完整条款
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["第", "条", "\n\n", "\n", "。"]
    )
    chunks = splitter.split_documents(docs)
    print(f"切片完成：{len(chunks)}个片段")

    # 打印来源分布
    sources = {}
    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "未知"))
        sources[source] = sources.get(source, 0) + 1
    for source, count in sources.items():
        print(f"  {source}：{count}个片段")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"知识库构建完成，共{vectorstore._collection.count()}个向量")
    return vectorstore


print("构建知识库中...")
vectorstore = build_knowledge_base("./law_docs")
print("知识库构建完成\n")


# ============================================================
# HyDE优化检索（核心亮点）
# ============================================================

def hyde_search(question, vectorstore, k=5):
    # 第一步：让模型生成一个假设性回答
    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个法律专家。
根据用户的问题，生成一段假设性的法律条文回答。
不需要完全准确，只需要和问题相关的法律语言风格。
回答控制在100字以内。"""),
        ("user", "{question}")
    ])

    hyde_chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_answer = hyde_chain.invoke({"question": question})

    # 第二步：用假设性回答去检索，比用原始问题检索更准
    results = vectorstore.similarity_search(hypothetical_answer, k=k)
    return results, hypothetical_answer


# ============================================================
# 问答函数
# ============================================================

def legal_answer(question, chat_history=[]):
    # 用HyDE检索
    relevant_chunks, hypothetical_answer = hyde_search(question, vectorstore)

    context = "\n\n".join([
        f"片段{i+1}（来源：{os.path.basename(chunk.metadata.get('source', '未知'))}，"
        f"第{chunk.metadata.get('page', '?')+1}页）：\n{chunk.page_content}"
        for i, chunk in enumerate(relevant_chunks)
    ])

    history_text = ""
    if chat_history:
        history_text = "\n\n历史对话：\n"
        for msg in chat_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            history_text += f"{role}：{msg.get('content', '')}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的法律顾问助手。
请根据以下法律条文回答用户的问题。

规则：
1. 只根据提供的法律条文回答，不要编造法律条款
2. 回答时引用具体的法律条文，比如"根据《劳动合同法》第X条"
3. 如果涉及多部法律，分别说明
4. 如果提供的条文里没有相关内容，明确说"现有法律条文中未找到相关规定"
5. 语言专业但易于理解

参考法律条文：
{context}
{history_text}"""),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "history_text": history_text,
        "question": question
    })

    sources = []
    for chunk in relevant_chunks:
        source = os.path.basename(chunk.metadata.get("source", "未知"))
        page = chunk.metadata.get("page", 0) + 1
        source_info = f"{source} 第{page}页"
        if source_info not in sources:
            sources.append(source_info)

    return answer, sources, hypothetical_answer


# ============================================================
# FastAPI接口
# ============================================================

class AskRequest(BaseModel):
    question: str
    chat_history: list[dict] = []


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    hypothetical_answer: str  # 暴露HyDE生成的假设回答，方便调试


@app.get("/")
def root():
    return {
        "message": "法律法规问答API",
        "chunks": vectorstore._collection.count()
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer, sources, hypothetical_answer = legal_answer(
        request.question,
        request.chat_history
    )
    return AskResponse(
        answer=answer,
        sources=sources,
        hypothetical_answer=hypothetical_answer
    )


# ============================================================
# 命令行测试
# ============================================================

def run_cli():
    print("法律法规问答系统启动")
    print("输入'退出'结束")
    print("=" * 50)

    chat_history = []

    while True:
        question = input("\n你的问题：").strip()
        if question == "退出":
            break
        if not question:
            continue

        print("\n检索中（HyDE优化）...")
        answer, sources, hyde = legal_answer(question, chat_history)

        print(f"\n假设性回答（用于检索）：{hyde[:50]}...")
        print(f"\n回答：\n{answer}")
        print(f"\n参考来源：{' | '.join(sources)}")

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    # 命令行模式
    # run_cli()

    # API模式（注释掉命令行，取消注释下面这行）
    uvicorn.run(app, host="0.0.0.0", port=8000)