import os
import pickle
from src.parser.pdf_parse import load_pdf, texts_split
from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever
from src.constant import raw_docs_path, clean_docs_path, split_docs_path, image_save_dir
from src.client.llm_clean_client import request_llm_clean


# 创建缓存、图片与索引目录。
for path in (raw_docs_path, clean_docs_path, split_docs_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(raw_docs_path), "../saved_index"), exist_ok=True)

# 解析 PDF
if not os.path.exists(raw_docs_path):
    raw_docs = load_pdf()
    print("文档page数:", len(raw_docs))
    pickle.dump(raw_docs, open(raw_docs_path, "wb"))
else:
    raw_docs = pickle.load(open(raw_docs_path, "rb"))
    print("加载文档page数:", len(raw_docs))

# 文本清洗和整理
if not os.path.exists(clean_docs_path):
    clean_docs = request_llm_clean(raw_docs)
    print("清洗后文档page数:", len(clean_docs))
    pickle.dump(clean_docs, open(clean_docs_path, "wb"))
else:
    clean_docs = pickle.load(open(clean_docs_path, "rb"))
    print("加载清洗文档page数:", len(clean_docs))

# 文档切分
if not os.path.exists(split_docs_path):
    split_docs = texts_split(clean_docs)
    print("解析后文档总数:", len(split_docs))
    pickle.dump(split_docs, open(split_docs_path, "wb"))

else:
    split_docs = pickle.load(open(split_docs_path, "rb"))
    print("加载解析文档总数:", len(split_docs))


# 索引入库

# Note: sparse - 字面匹配（e.g. 精准匹配的查询），代表：bm25，tfidf
bm25_retriever = BM25(split_docs)
candidate_docs = bm25_retriever.retrieve_topk("仪表盘可以显示哪些车辆状态信息", topk=3)
print("BM25召回样例:")
print(candidate_docs)
print("="*100)

# Note: dense - 向量检索（e.g. 精准匹配的查询），代表：bge-m3, Qwen-embedding的embedding模型 + Milvus/FAISS这样的向量数据库
milvus_retriever = MilvusRetriever(split_docs)
candidate_docs = milvus_retriever.retrieve_topk("仪表盘可以显示哪些车辆状态信息", topk=3)
print("BGE-M3召回样例:")
print(candidate_docs)
