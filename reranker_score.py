"评估检索排序结果的命中率、召回率及排序指标。"

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Set

from langchain_core.documents import Document

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs


def get_doc_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    return metadata.get("parent_id") or metadata.get("unique_id", "")


def hit_recall(doc_ids: List[str], gold_ids: Set[str], k: int) -> Dict[str, float]:
    topk = doc_ids[:k]
    hits = [doc_id for doc_id in topk if doc_id in gold_ids]
    hit_k = float(len(hits) > 0)
    recall_k = len(hits) / len(gold_ids) if gold_ids else 0.0
    return {"hit": hit_k, "recall": recall_k}


def mrr(doc_ids: List[str], gold_ids: Set[str], k: int) -> float:
    for idx, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / idx
    return 0.0


def ndcg(doc_ids: List[str], gold_ids: Set[str], k: int) -> float:
    def dcg(rank_list):
        score = 0.0
        for idx, doc_id in enumerate(rank_list, start=1):
            rel = 1.0 if doc_id in gold_ids else 0.0
            if rel > 0:
                score += rel / math.log2(idx + 1)
        return score

    rank_list = doc_ids[:k]
    ideal_list = list(gold_ids)[:k]
    ideal_dcg = dcg(ideal_list)
    if ideal_dcg == 0:
        return 0.0
    return dcg(rank_list) / ideal_dcg


def evaluate(
    data_path: Path,
    bm25_topk: int = 5,
    milvus_topk: int = 10,
    rerank_topk: int = 5,
    ks: List[int] = None,
):
    if ks is None:
        ks = [1, 3, 5]

    records = json.loads(Path(data_path).read_text(encoding="utf-8"))
    bm25 = BM25(docs=None, retrieve=True)
    milvus = MilvusRetriever(docs=None, retrieve=True)
    reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)

    metrics_template = lambda: {k: 0.0 for k in ks}
    coarse_metrics = {
        "hit": metrics_template(),
        "recall": metrics_template(),
        "mrr": metrics_template(),
        "ndcg": metrics_template(),
    }
    rerank_metrics = {
        "hit": metrics_template(),
        "recall": metrics_template(),
        "mrr": metrics_template(),
        "ndcg": metrics_template(),
    }

    evaluated = 0
    for item in records:
        gold_ids = set(item.get("gold_doc_ids", []))
        if not gold_ids:
            continue
        query = item["question"].strip()
        bm25_docs = bm25.retrieve_topk(query, topk=bm25_topk)
        milvus_docs = milvus.retrieve_topk(query, topk=milvus_topk)
        merged_docs = merge_docs(bm25_docs, milvus_docs)
        merged_ids = [get_doc_id(doc) for doc in merged_docs]
        ranked_docs = reranker.rank(query, merged_docs, topk=rerank_topk)
        ranked_ids = [get_doc_id(doc) for doc in ranked_docs]

        for k in ks:
            coarse_hit_recall = hit_recall(merged_ids, gold_ids, k)
            rerank_hit_recall = hit_recall(ranked_ids, gold_ids, k)
            coarse_metrics["hit"][k] += coarse_hit_recall["hit"]
            coarse_metrics["recall"][k] += coarse_hit_recall["recall"]
            rerank_metrics["hit"][k] += rerank_hit_recall["hit"]
            rerank_metrics["recall"][k] += rerank_hit_recall["recall"]

            coarse_metrics["mrr"][k] += mrr(merged_ids, gold_ids, k)
            rerank_metrics["mrr"][k] += mrr(ranked_ids, gold_ids, k)
            coarse_metrics["ndcg"][k] += ndcg(merged_ids, gold_ids, k)
            rerank_metrics["ndcg"][k] += ndcg(ranked_ids, gold_ids, k)

        evaluated += 1

    if evaluated == 0:
        print("No samples with `gold_doc_ids` were found.")
        return

    print(f"Total evaluated samples: {evaluated}")
    for k in ks:
        print(f"\n=== Top-{k} Metrics ===")
        print(
            f"Coarse Hit@{k}: {coarse_metrics['hit'][k] / evaluated:.3f}, "
            f"Recall@{k}: {coarse_metrics['recall'][k] / evaluated:.3f}, "
            f"MRR@{k}: {coarse_metrics['mrr'][k] / evaluated:.3f}, "
            f"NDCG@{k}: {coarse_metrics['ndcg'][k] / evaluated:.3f}"
        )
        print(
            f"Rerank Hit@{k}: {rerank_metrics['hit'][k] / evaluated:.3f}, "
            f"Recall@{k}: {rerank_metrics['recall'][k] / evaluated:.3f}, "
            f"MRR@{k}: {rerank_metrics['mrr'][k] / evaluated:.3f}, "
            f"NDCG@{k}: {rerank_metrics['ndcg'][k] / evaluated:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval/reranker metrics.")
    parser.add_argument("--data", type=str, default="data/qa_pairs/test_qa_pair_verify.json")
    parser.add_argument("--bm25_topk", type=int, default=5)
    parser.add_argument("--milvus_topk", type=int, default=10)
    parser.add_argument("--rerank_topk", type=int, default=5)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    evaluate(
        data_path=Path(args.data),
        bm25_topk=args.bm25_topk,
        milvus_topk=args.milvus_topk,
        rerank_topk=args.rerank_topk,
        ks=args.ks,
    )
