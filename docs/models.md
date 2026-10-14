# 模型配置

| 模型 | 用途 | 配置位置 |
|---|---|---|
| m3e-small | 语义切分 | `src/constant.py` |
| BGE-M3 | 稠密、稀疏检索 | `src/retriever/milvus_retriever.py` |
| BGE-reranker-v2-m3 及微调检查点 | 候选排序 | `RAG_RERANKER_PATH`、`training/config/bge_reranker.yaml` |
| Qwen3-8B、LoRA SFT、AWQ INT4 产物 | 回答生成 | `training/config/`、`training/awq_quant.py` |
| Qwen3-Reranker-4B | 可选训练数据构造 | `scripts/generate_sft_data.py` |
| OpenAI 兼容模型接口 | 文本整理、问答生成 | `DOUBAO_API_KEY`、`DOUBAO_BASE_URL`、`DOUBAO_MODEL_NAME` |

权重与训练产物需在本地准备。`RAG_LLM_MODEL` 是服务名称，默认 `qwen3-polo`，客户端与 vLLM 服务端必须一致；`RAG_MODEL_DIR` 指定待服务的模型目录。

LoRA 配置用于 LLaMA-Factory，reranker 配置用于 RAG-Retrieval；这两套训练工具需另行安装，仓库仅提供相关配置。
