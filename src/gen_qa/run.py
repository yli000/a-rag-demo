### python内置库

import os

import re

import pickle

import time

import random

import threading

import json

import hashlib


### 第三方库

import concurrent.futures

from tqdm.auto import tqdm

from openai import OpenAI

from langchain_core.documents import Document


random.seed(42) # 为了让每次随机的结果都一样，便于复现


MINMAL_CHUNK_SIZE = 100

MAX_WORKERS = 20

INPUT_PATH = './data/processed_docs/clean_docs.pkl'

QA_PATH = "./data/qa_pairs/qa_pair.json"

CHATS_PATH = "./data/ut/raw_general_chats.txt"

OUTPUT_PATH = "./data/qa_pairs/expand_qa_pair.json"

TRAIN_PATH = "./data/qa_pairs/train_qa_pair.json"

TEST_PATH = "./data/qa_pairs/test_qa_pair.json"

TEST_KEYWORDS_PATH = "./data/qa_pairs/test_keywords_pair.json"


# 控制不同阶段是否重新生成，默认只读取已有产物

RUN_QA_FROM_SPLIT_DOCS = False

RUN_GENERALIZE_FROM_QA = False


def count_jsonl(path: str) -> int:

    """统计 jsonl 文件中有效行数"""

    count = 0

    with open(path, "r", encoding="utf-8") as fd:

        for line in fd:

            if line.strip():

                count += 1

    return count


def _strip_code_fence(text: str) -> str:

    "移除模型输出外围的 Markdown 代码围栏。"

    if not text:

        return text

    stripped = text.strip()

    if stripped.startswith("```"):

        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()

        if stripped.endswith("```"):

            stripped = stripped[:-3].strip()

    return stripped


def load_json_response(raw_resp: str):

    """

    处理大模型返回的不规范 JSON：

    1. 去掉 ```json 代码块包裹

    2. 将非法的反斜杠转义为合法 JSON

    """

    text = _strip_code_fence(raw_resp)

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        try:

            return json.loads(repaired)

        except json.JSONDecodeError as exc:

            print(f"[warn] json loads failed: {exc}\n{text}\n")

            return []


# 问答prompt：问题相关；独立回答；综合大段文本；返回JSON；<document></document>


CONTEXT_PROMPT_TPL = """

我会给你一段文本（<document></document>之间的部分），你需要阅读这段文本，分别针对这段文本生成5个问题，和基于这段文本对问题的回答，回答请保持完整，无须重复问题。



对问题、答案的要求：

1.问题：问题要与这段文本相关，不要询问类似“这个问题的答案在哪一章”这样的问题;

2.答案：回答请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用其他章节和页码，例如答案内容不能出现请参阅xx页码;

3.5个问题里面至少要包含一个需要综合*大段*文本才能回答的问题，但不要问类似“这一段主要讲了什么内容”这样的问题;



对输出的要求：

1.返回结果以JSON形式组织，格式为[{"question": "...", "answer": "..."}, ...]。

2.如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。



下方是文本：

<document>

{{document}}

</document>



请生成结果：

"""


# 泛化prompt：不同问答、口语化、返回格式、强调要求


GENERALIZE_PROMPT_TPL = """

你是一个造句大师，请根据我输入的问题，生成具有意思相近的5个问题。



要求：

1.生成的问题要表达相近的意思，请探索采用不同的问法。

2.生成问题尽量口语化一点，可以不遵循原来的句式，例如：怎么打开车窗=》这个车的窗子要怎么才能开启

3.每一个问题用回车符连接，前面用序号开头，例如：1., 2., 3.



注意：不是回答问题，任务是输出5个同义句。



下方是输入的问题：

<question>

{{document}}

</question>



请生成结果：

"""


# 关键词抽取prompt：结合 Polo 手册真实术语

KEYWORDS_PROMPT_TPL = """

你是一名专业的汽车领域NLP工程师，任务是从给定的汽车行业文本中提取核心关键词。请按以下要求操作：



抽取原则：

1.优先提取汽车专业术语（如"涡轮增压"、"OBD系统"）

2.保留产品型号/规格（如"Polo"、"1.5L EA211发动机"）

3.包含技术特征（如"前部辅助系统"、"ESP电子稳定系统"）

4.提取关键动作（如"关闭"、"解锁"）





输出要求：

1.重点关注：车身结构/汽车零部件/智能网联/辅助驾驶/售后服务

2.过滤通用词汇（如"使用"、"包括"）

3.请输出最重要的关键词，关键词数量不得超过5个

4.如果没有关键词，请直接输出“无”





输出格式：

关键词列表，用逗号分隔

例如：驻车距离报警传感器,雨量传感器,前部辅助系统





下方是输入的问题：

<question>

{{document}}

</question>



请生成结果：



"""


# 质检prompt：定义好坏


QA_QUALITY_PROMPT_TPL = """

你是一个汽车领域的专家，现在有人根据一份汽车用车手册，构造了一些问题，并对问题进行了回答。

你的任务是对这些问题（<question></question>之间的部分）和回答（<answer></answer>）进行打分。



结果请以JSON形式组织，格式如下（<result></result>之间的部分）：

<result>

{"score": ..., "reason": ...}

</result>

其中score是对问题-回答的打分，分值是一个int类型的值，取值范围为1-5。reason是打分的理由。



好的问题，应该是询问事实、观点等，不好的问题，通常要求做一些文本摘要等初级文字处理工作，类似于“这一段描述了什么”，“文本描述了什么”；或者询问的内容是图相关的，例如“图4展示了什么数据？”。

好的答案，应该能够回应问题，而不是回答无关的内容，不好的回答，会给出在原文中的引用，例如“第3章”等。



问题：

<question>

{{question}}

</question>



答案：

<answer>

{{answer}}

</answer>



请进返回JSON格式的数据即可，不要添加其他任何描述性信息。

"""


llm_client = OpenAI(

    api_key=os.environ['DOUBAO_API_KEY'],

    base_url=os.environ['DOUBAO_BASE_URL']

)


### api实例是：client = OpenAI(

    # 未显式传入密钥时，客户端默认读取 OPENAI_API_KEY 环境变量。


#

# 有无get的区别：api未设置时，有get会返回None，无get会报KeyError异常


def build_qa_prompt(prompt_tmpl, text):

    prompt = prompt_tmpl.replace('{{document}}', text).strip()

    return prompt


### Q temperature和top_p为什么设置得那么高 -> 数据增强，那格式怎么办？


def chat(prompt, max_retry=3, debug=False, temperature=0.85, top_p=0.95):


    def do_chat(prompt):

        completion = llm_client.chat.completions.create(

            model=os.environ["DOUBAO_MODEL_NAME"],

            messages=[

                {"role": "system", "content": "你是一个有用的人工智能助手."},

                {"role": "user", "content": prompt}

            ],

            top_p=top_p,

            temperature=temperature

        )

        return completion.choices[0].message.content


    x = do_chat(prompt)

    while max_retry > 0:

        try:

            return do_chat(prompt)

        # 错误 e.g. 网络超时、API服务器返回错误等 跳转到 except 部分来执行“补救措施”:防止因为一次偶然的网络波动就导致整个几十个小时的程序崩溃

        except Exception as e:

            max_retry -= 1

            sleep_seconds = random.randint(1, 4)

            if debug:

                print(f"{str(e)}, remain retry: {max_retry}, sleeping {sleep_seconds}s {prompt}")

            time.sleep(sleep_seconds)

    return None


### 并发处理:高效地为一大批文档（splitted_docs）并发地调用 chat 函数，生成问答数据，并安全地将结果实时写入文件

def gen_qa(splitted_docs, prompt_tmpl, qa_ckpt_filename, expand=False):

    qa_ckpt = {}

    if os.path.exists(qa_ckpt_filename):

        with open(qa_ckpt_filename, "r", encoding="utf-8") as fd:

            for line in fd:

                line = line.strip()

                if not line:

                    continue

                try:

                    info = json.loads(line)

                except json.JSONDecodeError:

                    continue

                qa_ckpt[info["unique_id"]] = info

        if qa_ckpt:

            print(f"[resume] Loaded {len(qa_ckpt)} records from {qa_ckpt_filename}")


    ### threading 是 Python 标准库的一部分，用于处理多线程

    file_lock = threading.Lock() # 建一个锁

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        if expand: # executor 非阻塞

            futures = {doc.page_content: executor.submit(chat, build_qa_prompt(

                prompt_tmpl, doc.page_content), 3, True) for doc in splitted_docs

                       if doc.page_content not in qa_ckpt} # 使用文档正文作为检查点的唯一键。

        else:

            futures = {doc.metadata['unique_id']: executor.submit(chat, build_qa_prompt(

                prompt_tmpl, doc.page_content), 3, True) for doc in splitted_docs

                       if len(doc.page_content.replace('\n', '')) >= MINMAL_CHUNK_SIZE and

                           doc.metadata['unique_id'] not in qa_ckpt}

        for unique_id in tqdm(futures):

            future = futures[unique_id]

            result = future.result() # 阻塞

            if result is None:

                continue


            item = {'unique_id': unique_id, 'raw_resp': result}

            qa_ckpt[unique_id] = item


            file_lock.acquire()


            try:

                with open(qa_ckpt_filename, 'a') as f:

                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            except Exception as e:

                print(e)

            finally:

                file_lock.release()

    return qa_ckpt


def _load_splitted_docs():

    "读取切分后的文档。"

    if not hasattr(_load_splitted_docs, "_cache"):

        _load_splitted_docs._cache = pickle.load(open(INPUT_PATH, "rb"))

    return _load_splitted_docs._cache


def _qa_target_docs():

    """有效 QA doc 列表"""

    docs = _load_splitted_docs()

    if not hasattr(_qa_target_docs, "_cache"): # hasattr: 用来判断一个对象是否有某个属性，可以复用缓存

        filtered = []

        for doc in docs:

            if len(doc.page_content.replace('\n', '')) >= MINMAL_CHUNK_SIZE:

                filtered.append(doc)

        _qa_target_docs._cache = filtered

    return _qa_target_docs._cache


if __name__ == "__main__":


    load_splitted_docs = _load_splitted_docs # 清洗后的文档缓存

    get_qa_docs = _qa_target_docs #


    # 要不要重新生成QA

    run_qa_stage = RUN_QA_FROM_SPLIT_DOCS or (not os.path.exists(QA_PATH))

    if (not run_qa_stage) and os.path.exists(QA_PATH):

        existing_qa = count_jsonl(QA_PATH)

        total_docs = len(get_qa_docs())

        if existing_qa < total_docs:

            print(f"[resume] QA ckpt {existing_qa}/{total_docs}, continue generating...")

            run_qa_stage = True


    if run_qa_stage:

        docs = get_qa_docs()

        print("待处理文件数:", len(docs))

        if docs:

            print(docs[0])

        qa_dict = gen_qa(docs, CONTEXT_PROMPT_TPL, QA_PATH) # 生成QA对：一页5对

    else:

        qa_dict = {}

        with open(QA_PATH, "r", encoding="utf-8") as fd:

            for line in fd:

                info = json.loads(line)

                qa_dict[info["unique_id"]] = info


    # 构建问题文档供泛化阶段使用

    question_docs = []

    idx = 0

    for unique_id, info in qa_dict.items():

        resp = load_json_response(info["raw_resp"])

        for qa in resp:

            question_docs.append(

                Document(page_content=qa["question"], metadata={"unique_id": str(idx)})

            )

            idx += 1 # 为每个问题分配唯一的索引 ID


    run_expand_stage = RUN_GENERALIZE_FROM_QA or (not os.path.exists(OUTPUT_PATH))

    if (not run_expand_stage) and os.path.exists(OUTPUT_PATH):

        existing_expand = count_jsonl(OUTPUT_PATH)

        if existing_expand < len(question_docs):

            print(f"[resume] Expand ckpt {existing_expand}/{len(question_docs)}, continue generating...")

            run_expand_stage = True


    if run_expand_stage:

        print("待泛化问题数:", len(question_docs))

        expand_qa_dict = gen_qa(question_docs, GENERALIZE_PROMPT_TPL, OUTPUT_PATH, expand=True) # 给每个问题生成5个泛化问题

    else:

        expand_qa_dict = {}

        with open(OUTPUT_PATH, "r", encoding="utf-8") as fd:

            for line in fd:

                info = json.loads(line)

                expand_qa_dict[info["unique_id"]] = info


    # 整理结果：字典：{原问题ID: [扩展问句1, 扩展问句2, …]}

    expand_qa_pairs = {}

    for unique_id, info in expand_qa_dict.items():

        question = info["unique_id"]

        expand_questions = info["raw_resp"]

        expand_questions = expand_questions.split("\n")

        expand_questions = [re.sub(r'^\d[.. ]', '', item).strip() for item in expand_questions]

        expand_qa_pairs[question] = expand_questions


    train_qa_pairs = []

    test_qa_pairs = []

    for unique_id, info in qa_dict.items():

        resp = load_json_response(info["raw_resp"])

        for qa in resp:

            question = qa["question"].strip()

            answer = qa["answer"].strip()


            # 跳过这对QA

            if "无法准确" in answer or "未提及" in answer:

                continue


            ### 将1个原始问题和5个泛化问题组合在一起，实现了数据量的6倍扩增。

            expand_questions = [question] + expand_qa_pairs[question]

            for query in expand_questions:

                unique_id = hashlib.md5(query.encode('utf-8')).hexdigest()

                item = {

                    "unique_id": unique_id,

                    "question": query,

                    "answer": answer

                }

                if random.random() < 0.9:

                    train_qa_pairs.append(item)

                else:

                    test_qa_pairs.append(item)


    print("训练集QA数：", len(train_qa_pairs), "测试集QA数：", len(test_qa_pairs))


    # 去重 + 抽取关键词

    test_answer_docs = []

    unique_test_answers = set([item["answer"] for item in test_qa_pairs])

    for idx, answer in enumerate(unique_test_answers):

        test_answer_docs.append(Document(page_content=answer, metadata={"unique_id": str(idx)}))


    print("待抽取关键词docs数：", len(test_answer_docs))

    keywords_dict = gen_qa(test_answer_docs, KEYWORDS_PROMPT_TPL, TEST_KEYWORDS_PATH, expand=True)


    keywords_mapping = {}

    for unique_id, info in keywords_dict.items():

        keywords = info["raw_resp"].split(",")

        kewyords = [item for item in keywords if item not in ["无", "Polo"]]

        keywords_mapping[info["unique_id"]] = kewyords


    for info in test_qa_pairs:

        keywords = keywords_mapping[info["answer"]]

        info["keywords"] = keywords


    # 负样本

    chats_data = open(CHATS_PATH).readlines()

    chats_data = [item.strip() for item in chats_data]


    random.seed(42)

    for line in chats_data:

        if random.random() < 0.95:

            train_qa_pairs.append({

                "unique_id": hashlib.md5(line.encode('utf-8')).hexdigest(),

                "question": line,

                "answer": "无答案"

            })


        else:

            test_qa_pairs.append({

                "unique_id": hashlib.md5(line.encode('utf-8')).hexdigest(),

                "question": line,

                "answer": "无答案",

                "keywords": []


            })


    random.seed(42)

    with open(TRAIN_PATH, "w") as fd:

        random.shuffle(train_qa_pairs)

        fd.write(json.dumps(train_qa_pairs, ensure_ascii=False, indent=2))

        print("训练集已写入:", TRAIN_PATH, len(train_qa_pairs))


    random.seed(42)

    with open(TEST_PATH, "w") as fd:

        random.shuffle(test_qa_pairs)

        fd.write(json.dumps(test_qa_pairs, ensure_ascii=False, indent=2))

        print("测试集已写入:", TEST_PATH, len(test_qa_pairs))
