from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any
import os, asyncio

# ===== MongoDB =====
MONGO_OK = False
try:
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
    client.admin.command("ping")
    db = client["ai_coach"]
    chat_col = db["chat_history"]    # ✅ 벡터 저장 + RAG 검색까지 여기서 처리
    profile_col = db["profile"]      # ✅ persona 유지
    MONGO_OK = True
except:
    chat_col = None
    profile_col = None

# ===== Embedding =====
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
def embed(text:str):
    return embed_model.encode(text, normalize_embeddings=True, convert_to_numpy=True).tolist()

# ===== LLM =====
from llama_cpp import Llama
MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=(os.cpu_count() or 4)-1, chat_format="chatml")

# ===== Request Schemas =====
class ChatInput(BaseModel):
    user_id: str
    message: str

class ChatWithAnalysisInput(BaseModel):
    user_id: str
    message: str
    analysis: Dict[str, Any]

# ===== Save chat + embedding =====
def save_chat(user_id:str, message:str, response:str):
    if not MONGO_OK: return
    chat_col.insert_one({
        "user_id": user_id,
        "message": message,
        "response": response,
        "embedding": embed(message),   # ✅ embedding 저장
        "timestamp": datetime.now()
    })

# ===== Vector Search =====
def search_similar(user_id:str, query:str, k=4):
    if not MONGO_OK: return []
    qvec = embed(query)
    pipeline = [
        {"$vectorSearch": {
            "index": "chat_vec_index",
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": 100,
            "limit": k,
            "filter": {"user_id": user_id}
        }},
        {"$project": {"message":1, "_id":0}}
    ]
    return [d["message"] for d in chat_col.aggregate(pipeline)]

# ===== Persona =====
def get_persona(user_id:str):
    if not MONGO_OK: return ""
    doc = profile_col.find_one({"user_id": user_id})
    return doc.get("persona", "") if doc else ""

async def update_persona(user_id:str):
    if not MONGO_OK: return
    count = chat_col.count_documents({"user_id": user_id})
    if count < 3 or count % 3 != 0:
        return
    logs = list(chat_col.find({"user_id": user_id}).sort("timestamp",-1).limit(10))
    text = "\n".join([f"User:{c['message']}\nAI:{c['response']}" for c in logs])

    out = llm.create_chat_completion(
        messages=[
            {"role":"system", "content":"사용자의 운동 습관/패턴을 5줄 요약."},
            {"role":"user", "content": text}
        ],
        temperature=0.2, max_tokens=150
    )
    summary = out["choices"][0]["message"]["content"].strip()
    profile_col.update_one({"user_id":user_id},{"$set":{"persona":summary}},upsert=True)

# ===== Prompt Context Build =====
def build_context(user_id:str, msg:str):
    persona = get_persona(user_id)
    similar = search_similar(user_id, msg, k=4)
    ctx = ""
    if persona:
        ctx += f"[사용자 특징]\n{persona}\n\n"
    if similar:
        ctx += "[과거 유사 코칭]\n" + "\n".join(f"- {m}" for m in similar) + "\n\n"
    return ctx

# ===== LLM Generate =====
async def generate(user_id:str, message:str, analysis:Dict[str,Any]):
    from_text = build_context(user_id, message)
    messages = [
        {"role":"system", "content":"너는 한국 PT다. 단호하지만 따뜻하게 말해라."},
        {"role":"user", "content": from_text + message}
    ]
    out = await asyncio.to_thread(lambda:
        llm.create_chat_completion(messages=messages, temperature=0.25)["choices"][0]["message"]["content"].strip()
    )
    return out

# ===== FastAPI =====
app = FastAPI()

@app.post("/chat")
async def chat(data:ChatInput, background_tasks:BackgroundTasks):
    answer = await generate(data.user_id, data.message, {})
    save_chat(data.user_id, data.message, answer)
    background_tasks.add_task(update_persona, data.user_id)
    return {"answer": answer}


# # server.py
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from typing import Dict, Any, Optional
# import os, asyncio
# import numpy as np

# # ---- Optional Mongo ----
# MONGO_OK = False
# try:
#     from pymongo import MongoClient
#     client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
#     _ = client.admin.command("ping")
#     db = client["ai_coach"]
#     chat_col = db["chat_history"]
#     profile_col = db["profile"]
#     MONGO_OK = True
# except:
#     chat_col = None
#     profile_col = None

# # ---- Embedding ----
# from sentence_transformers import SentenceTransformer
# embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

# def embed(text: str):
#     return embed_model.encode(text, normalize_embeddings=True, convert_to_numpy=True).tolist()

# # ---- LLM ----
# from llama_cpp import Llama
# MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=4096,
#     n_threads=max(1, (os.cpu_count() or 2) - 1),
#     n_batch=128,
#     verbose=False,
#     chat_format="chatml"
# )

# # ---- Schemas ----
# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# class ChatWithAnalysisInput(BaseModel):
#     user_id: str
#     message: str
#     analysis: Dict[str, Any]

# # ---- Helper: Convert angle → human-friendly state ----
# def describe(angle: Optional[float]) -> str:
#     if angle is None:
#         return "데이터 없음"
#     if angle > 170:
#         return "잘 펴져 있어"
#     if angle > 150:
#         return "무난해"
#     if angle > 120:
#         return "조금 더 펴면 좋아"
#     return "힘이 덜 실림"

# def describe_back(angle: Optional[float]) -> str:
#     if angle is None:
#         return "데이터 없음"
#     if angle > 40:
#         return "허리가 뒤로 꺾임"
#     if angle < 15:
#         return "허리가 말려 있음"
#     return "허리 중립 좋아"



# # ---- System Prompt (말투 1번) ----
# SYSTEM_PROMPT = (
#     "너는 한국 헬스 퍼스널 트레이너다. 반드시 한국어만 사용해라.\n"
#     "말투는 짧고 단호하지만 따뜻하게. 혼내는 것이 아니라, '잡아주는 느낌'으로.\n\n"

#     "입력되는 관절각/스테이지/정확도는 내부 참고용이다.\n"
#     "❗ 중요: describe() 결과(예: '무난해', '잘 펴져 있어', '힘이 덜 실림')는 "
#     "직접 말하지 말고, 그 의미를 **자연스러운 코칭 표현으로 바꿔서** 설명한다.\n"
#     "즉, 숫자/평가 단어를 그대로 말하지 않는다.\n"
#     "사용자는 '지금 몸이 어떻게 보이는지' + '어떻게 고치면 되는지'만 이해하면 된다.\n\n"

#     "출력 형식:\n"
#     "① 자세 느낌 요약 (2문장, 실제 눈으로 본 듯 묘사)\n"
#     "② 잘한 점 (칭찬 1개, 감정이 담긴 표현)\n"
#     "③ 개선할 점 (2~3개, 간단/명확하게)\n"
#     "④ 코칭 큐 (불릿 3~5개, 4~8글자의 명령형 예: '가슴 들어', '발바닥 눌러')\n"
#     "⑤ 자세 점수: XX/100 (점수만 말하기, 감점 이유 언급 금지)\n"
#     "⑥ 다음 세트 목표 (의지 끌어올리는 한 문장)\n\n"

#     "절대 금지:\n"
#     "- describe() 결과 단어를 그대로 출력\n"
#     "- 각도/수치 나열\n"
#     "- '상태는 ~~로 보입니다' 같은 기계식 말투\n"
#     "- 의료 진단 뉘앙스\n"
# )


# # ---- Build prompt (no angles shown) ----
# def build_user_prompt(user_msg: str, analysis: Dict[str, Any]) -> str:
#     ex = analysis.get("detected_exercise", "운동 이름 없음")
#     stage = analysis.get("stage", "단계 정보 없음")
#     joints = analysis.get("pose_data", {}).get("joints", {})

#     # Convert numeric → state
#     left_elbow = describe(joints.get("left_elbow_angle"))
#     right_elbow = describe(joints.get("right_elbow_angle"))
#     left_knee = describe(joints.get("left_knee_angle"))
#     right_knee = describe(joints.get("right_knee_angle"))
#     back = describe_back(joints.get("back_angle"))

#     return (
#         f"[사용자 말]\n{user_msg}\n\n"
#         f"[운동] {ex}\n"
#         f"[단계] {stage}\n\n"
#         f"팔 상태: 좌 {left_elbow}, 우 {right_elbow}\n"
#         f"무릎 상태: 좌 {left_knee}, 우 {right_knee}\n"
#         f"허리 상태: {back}\n\n"
#         "위 상태를 실제 PT처럼 자연스럽게 해석해서 코칭 피드백을 작성해줘."
#     )


# # ---- LLM Call ----
# async def generate_async(user_msg: str, analysis: Dict[str, Any]) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": build_user_prompt(user_msg, analysis)}
#     ]

#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.25,
#             top_p=0.85,
#             max_tokens=320,
#             stop=["</s>", "<|im_end|>"]
#         )
#         return out["choices"][0]["message"]["content"].strip()

#     return await asyncio.to_thread(_run)

# # ---- Persona (optional) ----
# async def update_persona_background(user_id: str):
#     if not MONGO_OK: return
#     chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(10))
#     text = "\n".join([f"User:{c['message']}\nAI:{c['response']}" for c in chats])
#     messages = [
#         {"role": "system", "content": "사용자 운동 습관과 목표를 5줄로 요약."},
#         {"role": "user", "content": text},
#     ]
#     def _run():
#         out = llm.create_chat_completion(messages=messages, temperature=0.2, max_tokens=120)
#         return out["choices"][0]["message"]["content"].strip()
#     summary = await asyncio.to_thread(_run)
#     profile_col.update_one(
#         {"user_id": user_id},
#         {"$set": {"persona": summary, "updated_at": datetime.now()}},
#         upsert=True
#     )

# # ---- FastAPI ----
# app = FastAPI(title="MediBear LLM Server")

# @app.post("/chat")
# async def chat_with_ai(data: ChatInput, background_tasks: BackgroundTasks):
#     answer = await generate_async(data.message, analysis={})
#     return {"answer": answer}

# @app.post("/chat_with_analysis")
# async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):
#     answer = await generate_async(data.message, data.analysis)
#     return {"answer": answer}




# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import asyncio
# import os
# from typing import Dict, Any

# from llama_cpp import Llama

# app = FastAPI(title="MediBear LLM Server")

# # ---------------- MongoDB ----------------
# client = MongoClient("mongodb://localhost:27017")
# db = client["ai_coach"]
# chat_col = db["chat_history"]
# profile_col = db["profile"]

# # ---------------- Embedding ----------------
# # embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
# embed_model = SentenceTransformer("intfloat/multilingual-e5-base")


# # ---------------- LLM ----------------
# MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=1024,
#     n_threads=8,
#     n_batch=128,
#     logits_all=False,
#     verbose=False,
#     chat_format="chatml"
# )


# # ---------------- 스코어 계산 ----------------
# def score_form(left_elbow, right_elbow, left_knee, right_knee, back_angle):
#     score = 100

#     elbow_avg = (left_elbow + right_elbow) / 2
#     if elbow_avg < 155:
#         score -= min(30, (155 - elbow_avg) * 0.5)

#     knee_avg = (left_knee + right_knee) / 2
#     if knee_avg < 160:
#         score -= min(15, (160 - knee_avg) * 0.3)

#     if not (30 <= back_angle <= 60):
#         score -= 20

#     return max(0, int(score))


# # ---------------- 데이터 모델 ----------------
# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# class ChatWithAnalysisInput(BaseModel):
#     user_id: str
#     message: str
#     analysis: Dict[str, Any]


# # ---------------- LLM 생성 ----------------
# async def generate_async(persona, context_text, message, summary_block):
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "당신은 한국 헬스 퍼스널 트레이너입니다.\n"
#                 "말투는 반말, 친절하지만 단호.\n"
#                 "지적은 짧게, 해결법은 명확하게.\n"
#                 "무조건 한국어만 사용. 영어, 한자 금지.\n\n"

#                 "응답 형식:\n"
#                 "1) 현재 자세 요약 (1문장)\n"
#                 "2) 잘한 점 (1개)\n"
#                 "3) 보완할 점 (1~3개)\n"
#                 "4) 즉시 적용 가능 코칭 큐 (명령형 짧게)\n"
#                 "5) 자세 점수 (ex: 82/100)\n"
#                 "6) 다음 세트 목표 (1문장)"
#             ),
#         },
#         {
#             "role": "user",
#             "content": summary_block
#         }
#     ]

#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.4,
#             top_p=0.9,
#             max_tokens=220
#         )
#         return out["choices"][0]["message"]["content"].strip()

#     return await asyncio.to_thread(_run)


# # ---------------- ✅ 운동 분석 + 대화 ----------------
# @app.post("/chat_with_analysis")
# async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):

#     joints = data.analysis.get("pose_data", {}).get("joints", {})
#     left_elbow = joints.get("left_elbow_angle", 0)
#     right_elbow = joints.get("right_elbow_angle", 0)
#     left_knee = joints.get("left_knee_angle", 0)
#     right_knee = joints.get("right_knee_angle", 0)
#     back_angle = joints.get("back_angle", 0)

#     score = score_form(left_elbow, right_elbow, left_knee, right_knee, back_angle)

#     summary_block = f"""
# 운동 종류: {data.analysis.get('detected_exercise')}
# 단계: {data.analysis.get('stage')}
# 정확도: {data.analysis.get('exercise_confidence')}

# 관절각:
# - 팔꿈치 평균: {(left_elbow+right_elbow)/2:.1f}
# - 무릎 평균: {(left_knee+right_knee)/2:.1f}
# - 등 각도(back_angle): {back_angle}

# 자세 점수: {score}/100

# 사용자 메시지: {data.message}
# """

#     profile = profile_col.find_one({"user_id": data.user_id})
#     persona = profile["persona"] if profile else ""

#     history = list(chat_col.find({"user_id": data.user_id}).sort("timestamp", -1).limit(5))
#     context_text = "\n".join([f"User: {h['message']}\nAI: {h['response']}" for h in history])

#     answer = await generate_async(persona, context_text, data.message, summary_block)

#     emb = embed_model.encode(data.message, convert_to_tensor=True)
#     emb = emb.cpu().numpy().tolist()

#     chat_col.insert_one({
#         "user_id": data.user_id,
#         "message": data.message,
#         "response": answer,
#         "vector": emb,
#         "timestamp": datetime.now()
#     })

#     return {"answer": answer}



# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import asyncio
# import os

# from llama_cpp import Llama  # ✅ CPU 최적화 엔진

# app = FastAPI()

# # ---------------- MongoDB ----------------
# client = MongoClient("mongodb://localhost:27017")
# db = client["ai_coach"]
# chat_col = db["chat_history"]
# profile_col = db["profile"]

# # ---------------- Embedding (빠름/한글OK) ----------------
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ---------------- LLM (Qwen 1.5B GGUF with llama.cpp) ----------------
# MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# # 스레드: Mac Intel은 물리코어-1 권장, EC2 t2.micro는 1
# _default_threads = max(1, (os.cpu_count() or 2) - 1)
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=1024, 
#     n_threads= 8,
#     n_batch= 128,            # 토큰 배치(속도 ↑, 메모리 영향)
#     logits_all=False,       # 메모리 절약
#     verbose=False,
#     chat_format="chatml"    # Qwen은 ChatML 포맷 호환 (<|im_start|> ... )
# )

# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# def cosine_similarity(a, b):
#     a, b = np.array(a), np.array(b)
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# # ---------------- LLM 호출 (Chat 형식, 반복 억제) ----------------
# async def generate_async(user_msg: str, persona: str, context_text: str):

#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "당신은 개인 맞춤형 건강/운동 상담 코치 AI입니다.\n"
#                 "사용자의 지난 대화 내용(persona 요약 + 최근 대화 context)을 참고하여 "
#                 "사용자의 상태와 감정, 습관을 기억하고 자연스럽게 이어지는 대화를 하세요.\n"
#                 "⚠️ **이전 답변을 그대로 복붙하지 말 것.**\n"
#                 "⚠️ 과거 내용을 '핵심만 요약하여 대답 속에 녹여서' 자연스럽게 반영할 것.\n"
#                 "사용자가 질문하면 현재 질문에 답하면서, 이전 상태/목표/습관을 반영하세요."
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"[사용자 요약 정보]\n{persona}\n\n"
#                 f"[최근 관련 대화]\n{context_text}\n\n"
#                 f"[현재 질문]\n{user_msg}"
#             ),
#         },
#     ]

#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.35,
#             top_p=0.9,
#             max_tokens=240,
#             stop=["</s>", "<|im_end|>"]
#         )
#         return out["choices"][0]["message"]["content"].strip()

#     return await asyncio.to_thread(_run)


# # async def generate_async(user_msg: str, persona: str, context_text: str):
# #     """
# #     Qwen ChatML 포맷으로 반복/에코 방지.
# #     """
# #     messages = [
# #         {
# #             "role": "system",
# #             "content": (
# #                 "당신은 운동 자세 교정과 건강 상담을 제공하는 한국어 AI 코치입니다. "
# #                 "친절하고 차분하며, 핵심만 명확하게 설명합니다. "
# #                 "이전 대화가 비정상/노이즈면 무시하세요. 영어 질문엔 한국어/영어 모두 자연스럽게 답해도 됩니다."
# #             ),
# #         },
# #         {
# #             "role": "user",
# #             "content": (
# #                 f"[사용자 성향 요약]\n{persona}\n\n"
# #                 f"[참고용 최근 대화(있으면 참고, 이상하면 무시)]\n{context_text}\n\n"
# #                 f"[사용자 질문]\n{user_msg}"
# #             ),
# #         },
# #     ]

# #     def _run():
# #         # create_chat_completion 사용 → 포맷 자동 처리
# #         out = llm.create_chat_completion(
# #             messages=messages,
# #             temperature=0.4,
# #             top_p=0.9,
# #             max_tokens=160,
# #             stop=["</s>", "<|im_end|>"]
# #         )
# #         return out["choices"][0]["message"]["content"].strip()

# #     return await asyncio.to_thread(_run)

# # ---------------- Persona 요약 (백그라운드) ----------------
# async def update_persona_background(user_id: str):
#     chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(10))
#     text_block = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])

#     messages = [
#     {"role": "system",
#      "content": (
#          "아래의 최근 대화 기록을 분석하여 사용자의 개인 정보를 5항목으로 요약하세요.\n"
#          "요약 기준:\n"
#          "1) 통증, 불편 부위, 건강 관련 상태\n"
#          "2) 운동 목표 또는 개선하고 싶은 점\n"
#          "3) 생활 습관 패턴 (예: 오래 앉음, 수면 부족 등)\n"
#          "4) 말투 / 대화 스타일 선호도\n"
#          "5) 기타 기억해두면 향후 상담에 도움 될 정보\n\n"
#          "중복 표현 없이, 짧고 명확하게 정리하세요."
#      )},
#     {"role": "user", "content": text_block or "(대화기록 없음)"},
# ]

#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.2,
#             top_p=0.9,
#             max_tokens=120,
#             stop=["</s>", "<|im_end|>"]
#         )
#         return out["choices"][0]["message"]["content"].strip()

#     summary = await asyncio.to_thread(_run)

#     profile_col.update_one(
#         {"user_id": user_id},
#         {"$set": {"persona": summary, "updated_at": datetime.now()}},
#         upsert=True
#     )

# # ---------------- API ----------------
# @app.post("/chat")
# async def chat_with_ai(data: ChatInput, background_tasks: BackgroundTasks):
#     # 1) 임베딩 (numpy 미사용 경로로 안전 변환)
#     emb = embed_model.encode(
#         data.message,
#         convert_to_tensor=True,
#         normalize_embeddings=True
#     )
#     user_vec = emb.cpu().tolist()

#     # 2) 과거 대화 로드 & 유사도 상위 3개
#     history = list(chat_col.find({"user_id": data.user_id}).sort("timestamp", -1).limit(10))
#     contexts = []
#     for h in history:
#         if "vector" in h:
#             sim = cosine_similarity(user_vec, h["vector"])
#             contexts.append((sim, h["message"], h.get("response", "")))
#     # contexts = sorted(contexts, key=lambda x: x[0], reverse=True)[:3]
#     # context_text = "\n".join([f"User: {m}\nAI: {r}" for _, m, r in contexts])
#     # 유사도 점수가 낮아도 일단 최근 기록 반영하도록 수정
#     if len(contexts) == 0:
#         context_text = "\n".join([f"User: {h['message']}\nAI: {h['response']}" for h in history[:3]])
#     else:
#         contexts = sorted(contexts, key=lambda x: x[0], reverse=True)[:3]
#         context_text = "\n".join([f"User: {m}\nAI: {r}" for _, m, r in contexts])

    
#     # 3) persona
#     profile = profile_col.find_one({"user_id": data.user_id})
#     persona = profile["persona"] if profile else "특징 미파악 사용자"

#     # 4) LLM 호출 (비동기 + Chat형식)
#     answer = await generate_async(data.message, persona, context_text)

#     # 5) 저장
#     chat_col.insert_one({
#         "user_id": data.user_id,
#         "message": data.message,
#         "response": answer,
#         "vector": user_vec,
#         "timestamp": datetime.now()
#     })

#     # 6) 요약은 백그라운드로
#     if len(history) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)

#     return {"answer": answer, "persona_summary": persona}
