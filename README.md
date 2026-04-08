# 法律法规问答系统

基于LangChain+HyDE优化的法律法规智能问答系统。

## 技术栈
- LangChain：LLM应用框架
- Chroma：本地向量数据库
- BAAI/bge-small-zh：本地Embedding模型
- HyDE：检索优化技术
- DeepSeek：大语言模型
- FastAPI：REST接口

## 功能特性
- 支持PDF/txt法律文档加载
- HyDE优化检索准确率
- 引用具体法律条文和页码
- 提供FastAPI REST接口

## 文档准备
将法律PDF放入docs/文件夹，推荐：
- 劳动合同法.pdf
- 消费者权益保护法.pdf
- 个人信息保护法.pdf

下载地址：flk.npc.gov.cn

## 运行
pip install -r requirements.txt
python legal_rag.py
