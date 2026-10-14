import json
import hashlib
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_chat_client import request_chat
from src.reranker.qwen3_reranker_vllm import Qwen3ReRankervLLM
from src.constant import qwen3_4b_reranker_model_path
from src.utils import merge_docs, post_processing
from src.checkpoint import (
    ProcessedTracker,
    JsonlWriter,
    dedup_by_id,
    ensure_parent_dir,
    load_jsonl,
)

random.seed(42)

SUMMARY_TRAIN_TMP = "./data/summary_data/train.jsonl"
SUMMARY_TEST_TMP = "./data/summary_data/test.jsonl"
SUMMARY_TRAIN_PATH = "./data/summary_data/train.json"
SUMMARY_TEST_PATH = "./data/summary_data/test.json"

RERANK_CANDIDATES_TMP = "./data/rerank_data/candidates.jsonl"
RERANK_TEST_TMP = "./data/rerank_data/test.jsonl"
RERANK_TRAIN_PATH = "./data/rerank_data/train.json"
RERANK_DEV_PATH = "./data/rerank_data/dev.json"
RERANK_TEST_PATH = "./data/rerank_data/test.json"

PROCESSED_IDS_PATH = "./data/summary_data/sft_processed_ids.txt"
TRAIN_QA_SOURCE = "./data/qa_pairs/train_qa_pair.json"
TRAIN_DATA_PATH = "./data/qa_pairs/train_data.json"
RAG_SKIP_LOG = "./log/rag_skipped_queries.log"

RUN_RAG_GENERATION = False
MAX_INPUT_SIZE = 4096
RERANK_DEV_SIZE = 1000
TEST_RATE = 0.08
SFT_MAX_WORKERS = 8
RAG_MAX_WORKERS = 8


def load_existing_queries(path: str):
    "读取已生成的查询，用于跳过重复样本。"
    if not os.path.exists(path):
        return set()
    queries = set()
    with open(path, "r", encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = item.get("query", "").strip()
            if query:
                queries.add(query)
    return queries


LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是上汽大众polo系列车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}},{{引用编号2}},...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""


def run_rag_generation():
    "生成训练用问答记录，并按查询保存检查点。"
    print("Initializing retrievers and reranker for RAG data generation...")
    bm25_retriever = BM25(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
    qwen3_reranker = Qwen3ReRankervLLM(model_path=qwen3_4b_reranker_model_path)

    existing_queries = load_existing_queries(TRAIN_DATA_PATH)
    ensure_parent_dir(TRAIN_DATA_PATH)

    with open(TRAIN_QA_SOURCE, "r", encoding="utf-8") as fd:
        qa_pairs = json.load(fd)

    pending_pairs = [
        item for item in qa_pairs
        if item.get("question", "").strip() and item["question"].strip() not in existing_queries
    ]

    if not pending_pairs:
        print("No pending QA samples to regenerate.")
        return

    print(f"Need to generate {len(pending_pairs)} QA samples.")

    def log_skip(reason: str, query: str):
        ensure_parent_dir(RAG_SKIP_LOG)
        with skip_log_lock:
            with open(RAG_SKIP_LOG, "a", encoding="utf-8") as log_fd:
                log_fd.write(f"{reason}: {query}\n")

    def process_one(item: dict):
        query = item["question"].strip()
        try:
            if len(query) < 2:
                print(f"[Skip] Query too short: {query!r}")
                log_skip("short_query", query)
                return
            bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
            milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
            merged_docs = merge_docs(bm25_docs, milvus_docs)
            if not merged_docs:
                print(f"[Skip] No docs found for: {query[:20]}...")
                log_skip("no_docs", query)
                return
            with rerank_lock:
                ranked_docs = qwen3_reranker.rank(query, merged_docs, topk=5)
            if not ranked_docs:
                print(f"[Skip] No rerank result for: {query[:20]}...")
                log_skip("no_rerank", query)
                return
            context = "\n".join(
                f"{idx + 1}.{doc.page_content}" for idx, doc in enumerate(ranked_docs)
            )
            response = request_chat(query, context)
            _ = post_processing(response, ranked_docs)
            ranked_contents = [doc.page_content for doc in ranked_docs]
            merged_contents = [doc.page_content for doc in merged_docs]
            info = {
                "query": query,
                "context": ranked_contents,
                "response": response,
                "merged_docs": merged_contents,
            }
            with write_lock:
                writer.write(info)
        except Exception as exc:
            print(f"\n[Error] Query: {query[:50]} | Msg: {exc}")
            log_skip(f"error:{exc}", query)
            time.sleep(1)

    ensure_parent_dir(RAG_SKIP_LOG)
    skip_log_lock = threading.Lock()
    write_lock = threading.Lock()
    rerank_lock = threading.Lock()
    with JsonlWriter(TRAIN_DATA_PATH, flush_every=1) as writer:
        with ThreadPoolExecutor(max_workers=RAG_MAX_WORKERS) as executor:
            futures = [executor.submit(process_one, item) for item in pending_pairs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="RAG Generation"):
                pass


def parse_response(info):
    response = info["response"]
    all_cites = re.findall(r"[【](.*?)[】]", response)
    cites = []
    for cite in all_cites:
        cite = re.sub(r"[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    cites = sorted(set(cites))
    cite_text = ",".join(str(c) for c in cites)
    answer = re.sub(r"[【](.*?)[】]", "", response)
    answer = re.sub(r"[{}【】]", "", answer)
    formatted = f"{answer}【{cite_text}】" if cite_text else "无答案"
    return formatted


def process_sft_and_rerank():
    def process_line(line: str):
        info = json.loads(line)
        line_id = hashlib.md5(line.encode("utf-8")).hexdigest()
        format_answer = parse_response(info)
        context = "\n".join(f"{idx + 1}.{doc}" for idx, doc in enumerate(info["context"]))
        if len(context) > MAX_INPUT_SIZE:
            context = context[:MAX_INPUT_SIZE]

        query = info["query"].strip()
        instruction = LLM_CHAT_PROMPT.format(query=query, context=context)
        item = {
            "query": query,
            "context": context,
            "instruction": instruction,
            "input": "",
            "output": format_answer,
        }
        neg_docs = [doc for doc in info["merged_docs"] if doc not in info["context"]]
        rng_value = random.random()
        rerank_candidates = []
        rerank_tests = []

        if rng_value < TEST_RATE:
            summary_target = "test"
            if format_answer != "无答案":
                content_list = [info["context"][0], random.choice(info["context"][-2:])]
                if neg_docs:
                    content_list.append(random.choice(neg_docs))
                rerank_tests.append({"_id": line_id, "query": query, "content": content_list})
        else:
            summary_target = "train"
            entry_idx = 0
            if format_answer != "无答案":
                positive = info["context"][0]
                middle = random.choice(info["context"][-2:])
                entry_idx += 1
                rerank_candidates.append(
                    {
                        "_id": f"{line_id}-cand-{entry_idx}",
                        "query": query,
                        "content": positive,
                        "label": 2,
                    }
                )
                entry_idx += 1
                rerank_candidates.append(
                    {
                        "_id": f"{line_id}-cand-{entry_idx}",
                        "query": query,
                        "content": middle,
                        "label": 1,
                    }
                )
                if neg_docs:
                    negative = random.choice(neg_docs)
                    entry_idx += 1
                    rerank_candidates.append(
                        {
                            "_id": f"{line_id}-cand-{entry_idx}",
                            "query": query,
                            "content": negative,
                            "label": 0,
                        }
                    )
            else:
                negative = random.choice(info["merged_docs"])
                entry_idx += 1
                rerank_candidates.append(
                    {
                        "_id": f"{line_id}-cand-{entry_idx}",
                        "query": query,
                        "content": negative,
                        "label": 0,
                    }
                )
        return {
            "line_id": line_id,
            "summary_target": summary_target,
            "summary_item": {"_id": line_id, **item},
            "rerank_candidates": rerank_candidates,
            "rerank_tests": rerank_tests,
        }

    with ProcessedTracker(PROCESSED_IDS_PATH) as tracker, \
        open(TRAIN_DATA_PATH, "r", encoding="utf-8") as fd, \
        JsonlWriter(SUMMARY_TRAIN_TMP) as sft_train_writer, \
        JsonlWriter(SUMMARY_TEST_TMP) as sft_test_writer, \
        JsonlWriter(RERANK_CANDIDATES_TMP) as rerank_cand_writer, \
        JsonlWriter(RERANK_TEST_TMP) as rerank_test_writer:

        lines_to_process = []
        for line in fd:
            line = line.strip()
            if not line:
                continue
            line_id = hashlib.md5(line.encode("utf-8")).hexdigest()
            if tracker.has(line_id):
                continue
            lines_to_process.append(line)

        if not lines_to_process:
            return

        with ThreadPoolExecutor(max_workers=SFT_MAX_WORKERS) as executor:
            futures = [executor.submit(process_line, line) for line in lines_to_process]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                line_id = result["line_id"]
                if result["summary_target"] == "test":
                    sft_test_writer.write(result["summary_item"])
                else:
                    sft_train_writer.write(result["summary_item"])

                for cand in result["rerank_candidates"]:
                    rerank_cand_writer.write(cand)
                for test_item in result["rerank_tests"]:
                    rerank_test_writer.write(test_item)

                tracker.add(line_id)


def finalize_outputs():
    summary_train = dedup_by_id(load_jsonl(SUMMARY_TRAIN_TMP))
    summary_test = dedup_by_id(load_jsonl(SUMMARY_TEST_TMP))

    ensure_parent_dir(SUMMARY_TRAIN_PATH)
    with open(SUMMARY_TRAIN_PATH, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_train, ensure_ascii=False, indent=4))

    ensure_parent_dir(SUMMARY_TEST_PATH)
    with open(SUMMARY_TEST_PATH, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_test, ensure_ascii=False, indent=4))

    rerank_candidates = dedup_by_id(load_jsonl(RERANK_CANDIDATES_TMP))
    rerank_candidates = [
        item for item in rerank_candidates if item["query"] and item["content"]
    ]
    random.shuffle(rerank_candidates)

    if rerank_candidates:
        dev_split_size = min(RERANK_DEV_SIZE, len(rerank_candidates))
        rerank_dev = rerank_candidates[-dev_split_size:]
        rerank_train = rerank_candidates[:-dev_split_size]
    else:
        rerank_dev = []
        rerank_train = []

    rerank_test = dedup_by_id(load_jsonl(RERANK_TEST_TMP))

    ensure_parent_dir(RERANK_TRAIN_PATH)
    with open(RERANK_TRAIN_PATH, "w", encoding="utf-8") as fh:
        for item in rerank_train:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    ensure_parent_dir(RERANK_DEV_PATH)
    with open(RERANK_DEV_PATH, "w", encoding="utf-8") as fh:
        for item in rerank_dev:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    ensure_parent_dir(RERANK_TEST_PATH)
    with open(RERANK_TEST_PATH, "w", encoding="utf-8") as fh:
        for item in rerank_test:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        "Summary Train size:", len(summary_train),
        "Summary Test size:", len(summary_test)
    )
    print(
        "Rerank Train size:", len(rerank_train),
        "Rerank Dev size:", len(rerank_dev),
        "Rerank Test size:", len(rerank_test),
    )


def main():
    if RUN_RAG_GENERATION:
        run_rag_generation()
    process_sft_and_rerank()
    finalize_outputs()


if __name__ == "__main__":
    main()
