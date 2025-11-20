# # ===== Load .env =====
# # from dotenv import load_dotenv
# # load_dotenv()

# # exercise_llm.py
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from typing import Dict, Any, Optional, List
# from pymongo import MongoClient
# import numpy as np
# import asyncio
# import os
# import re

# # ===== Embedding (384-d fixed) =====
# from sentence_transformers import SentenceTransformer
# EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
# embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

# def embed(text: str) -> List[float]:
#     return embed_model.encode(text, normalize_embeddings=True).tolist()

# VECTOR_DIM = len(embed_model.encode("dim"))

# # ===== LLM (Qwen GGUF 모델 사용) =====
# from llama_cpp import Llama
# # MODEL_PATH = "../../models/exercise_models/qwen2.5-3b-instruct-q4_k_m.gguf"   # ✅ 네가 다운받은 모델 이름 그대로  -> 상대 경로
# MODEL_PATH = r"C:\develop\medibear_fastapi\qwen2.5-3b-instruct-q4_k_m.gguf"


# llm = Llama(
#     model_path=MODEL_PATH,
#     # 이전 : 2048
#     n_ctx=2048,
#     n_threads=max(1, (os.cpu_count() or 2) - 1),
#     n_batch=128,
#     logits_all=False,
#     verbose=False,
#     chat_format="chatml",
# )

# # ===== FastAPI =====
# app = FastAPI(title="MediBear LLM Server (Local Mongo + RAG)")

# # ===== MongoDB =====
# client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
# db = client["ai_coach"]
# chat_col = db["chat_history"]
# profile_col = db["profile"]

# # ===== Input Models =====
# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# class ChatWithAnalysisInput(BaseModel):
#     user_id: str
#     message: str
#     analysis: Dict[str, Any]

# # ===== Utilities =====
# def cosine_similarity(a, b) -> float:
#     a, b = np.asarray(a), np.asarray(b)
#     na, nb = np.linalg.norm(a), np.linalg.norm(b)
#     return float(np.dot(a, b) / (na * nb)) if na * nb != 0 else 0.0

# def safe_get_vec(doc) -> Optional[List[float]]:
#     vec = doc.get("embedding") or doc.get("vector")
#     if isinstance(vec, list) and len(vec) == VECTOR_DIM:
#         return vec
#     return None

# def clean_text_korean_only(text: str) -> str:
#     return re.sub(r"[^가-힣0-9\s\.\,\?\!]", "", text)


# # ===== RAG =====
# def build_rag_context(user_id: str, user_msg: str, topk: int = 3, mode: str = "auto") -> str:
#     qvec = embed(user_msg)

#     if mode == "exercise":
#         query = {"user_id": user_id, "type": "exercise"}
#     elif mode == "general":
#         query = {"user_id": user_id, "type": "general"}
#     else:
#         query = {"user_id": user_id}

#     history = list(
#         chat_col.find(query)
#                .sort("timestamp", -1)
#                .limit(60)
#     )

#     scored = []
#     for h in history:
#         vec = safe_get_vec(h)
#         if vec:
#             cleaned = clean_text_korean_only(h["message"])
#             if cleaned.strip():
#                 scored.append((cosine_similarity(qvec, vec), cleaned))

#     if not scored:
#         return ""

#     scored.sort(key=lambda x: x[0], reverse=True)
#     return "\n".join([f"User said before: {msg}" for _, msg in scored[:topk]])


# # ===== Prompt 정의 =====
# SYSTEM_PROMPT_EXERCISE = (
#     "너는 한국어 퍼스널 트레이너이다.\n"
#     "사용자가 어떤 운동을 하고 있는지 반드시 첫 줄에서 자연스럽게 언급한다.\n"
#     "분석 데이터는 참고하지만 수치를 그대로 언급하지 않는다.\n"
#     "숫자, 각도, cm, %, ° 등 수치 언급 금지.\n"
#     "한자, 영어, 전문용어, 과학적 표현 금지.\n"
#     "운동 동작에 대한 느낌, 긴장, 힘의 흐름, 체중 분배 중심으로 피드백한다.\n\n"
#     "출력 형식:\n"
#     "① 지금 하고 있는 운동 이름 + 자세 느낌 요약 (1~2문장)\n"
#     "② 잘한 점 (1문장)\n"
#     "③ 개선할 점 (• 불릿 2~3개)\n"
#     "④ 코칭 큐 (• 불릿 3~5개, 명령형 4~8글자)\n"
#     "⑤ 다음 세트 목표 (1문장)\n"
#     "(1문장) 이런거는 언급 금지\n"
# )


# SYSTEM_PROMPT_GENERAL = (
#     "너는 한국어 헬스케어 상담 AI다.\n"
#     "공감 + 짧고 단호 + 따뜻한 톤.\n"
#     "운동 템플릿(①~⑤) 절대 사용 금지.\n"
# )


# # ===== LLM Wrapper =====
# async def llm_generate(messages):
#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.2,
#             top_p=0.9,
#             repeat_penalty=1.12,
#             # 이전 : 600
#             max_tokens=256, 
#         )
#         return out["choices"][0]["message"]["content"].strip()
#     return await asyncio.to_thread(_run)


# # ===== Persona 요약 =====
# async def update_persona_background(user_id: str):
#     chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(15))
#     if not chats:
#         return
#     text = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])
#     summary = await llm_generate([
#         {"role": "system", "content": "사용자의 통증 경향/운동 습관/말투를 5줄로 요약하라."},
#         {"role": "user", "content": text},
#     ])
#     profile_col.update_one({"user_id": user_id}, {"$set": {"persona": summary}}, upsert=True)


# # ===== 답변 생성 =====
# async def generate_answer(user_id: str, user_msg: str, analysis: Dict[str, Any]):
#     persona = profile_col.find_one({"user_id": user_id}) or {}
#     persona_text = persona.get("persona", "")

#     # ✅ 여기 추가: 운동 분석 JSON 키 맞춰주기
#     if analysis and "exercise" in analysis and "detected_exercise" not in analysis:
#         analysis["detected_exercise"] = analysis["exercise"]

#     # ✅ 운동 여부 판별 (이제 정상 동작)
#     is_exercise = bool(analysis and analysis.get("detected_exercise"))

#     # ✅ 운동명 텍스트 앞에 붙이기
#     if is_exercise:
#         exercise_name = analysis["detected_exercise"]
#         user_msg = f"{exercise_name} 운동 중: {user_msg}"

#     if is_exercise:
#         system_prompt = SYSTEM_PROMPT_EXERCISE
#         rag = build_rag_context(user_id, user_msg, mode="exercise")
#     else:
#         system_prompt = SYSTEM_PROMPT_GENERAL
#         rag = build_rag_context(user_id, user_msg, mode="general")

#     user_prompt = (rag + "\n\n" + user_msg) if rag else user_msg

#     messages = [
#         {"role": "system", "content": system_prompt + ("\n[사용자 요약]\n" + persona_text if persona_text else "")},
#         {"role": "user", "content": user_prompt},
#     ]

#     return await llm_generate(messages)




# # ===== 저장 =====
# def save_chat(user_id, message, response, embedding, analysis):
#     chat_col.insert_one({
#         "user_id": user_id,
#         "message": message,
#         "response": response,
#         "embedding": embedding,
#         "analysis": analysis or {},
#         "timestamp": datetime.now(),
#         "type": "exercise" if (analysis and analysis.get("detected_exercise")) else "general",
#     })


# # ===== Endpoints =====
# @app.post("/chat")
# async def chat_plain(data: ChatInput, background_tasks: BackgroundTasks):
#     qvec = embed(data.message)
#     answer = await generate_answer(data.user_id, data.message, analysis={})
#     save_chat(data.user_id, data.message, answer, qvec, {})
#     if chat_col.count_documents({"user_id": data.user_id}) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)
#     return {"answer": answer}

# @app.post("/chat_with_analysis")
# async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):

#     # ✅ message 비어있으면 자동 생성
#     user_msg = data.message if data.message and data.message.strip() else "운동 자세 피드백 요청"

#     qvec = embed(user_msg)

#     print("\n")
#     print(data.analysis)
#     print("\n")

#     answer = await generate_answer(data.user_id, user_msg, data.analysis)

#     save_chat(data.user_id, user_msg, answer, qvec, data.analysis)

#     if chat_col.count_documents({"user_id": data.user_id}) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)

#     return {"answer": answer}


# ===== Load .env =====
# from dotenv import load_dotenv
# load_dotenv()



###############################################올라마로 바꾼부분 ##########################################
# exercise_llm.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import numpy as np
import asyncio
import os
import re
import requests

# ===== Embedding (384-d fixed) =====
from sentence_transformers import SentenceTransformer
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

def embed(text: str) -> List[float]:
    return embed_model.encode(text, normalize_embeddings=True).tolist()

VECTOR_DIM = len(embed_model.encode("dim"))


# ============================
# 🔥 LLM = Ollama API
# ============================
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"


# ============================
# 🔥 LLM Wrapper (ChatML 강제)
# ============================
async def llm_generate(messages):
    def _run():
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }
        )
        data = res.json()
        try:
            return data["message"]["content"].strip()
        except:
            return "LLM 응답 오류"
    return await asyncio.to_thread(_run)



# ===== FastAPI =====
app = FastAPI(title="MediBear LLM Server (Local Mongo + RAG)")

# ===== MongoDB =====
client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
db = client["ai_coach"]
chat_col = db["chat_history"]
profile_col = db["profile"]


# ===== Input Models =====
class ChatInput(BaseModel):
    user_id: str
    message: str

class ChatWithAnalysisInput(BaseModel):
    user_id: str
    message: str
    analysis: Dict[str, Any]


# ===== Utilities =====
def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na * nb != 0 else 0.0

def safe_get_vec(doc) -> Optional[List[float]]:
    vec = doc.get("embedding") or doc.get("vector")
    if isinstance(vec, list) and len(vec) == VECTOR_DIM:
        return vec
    return None

def clean_text_korean_only(text: str) -> str:
    return re.sub(r"[^가-힣0-9\s\.\,\?\!]", "", text)


# ===== RAG =====
def build_rag_context(user_id: str, user_msg: str, topk: int = 3, mode: str = "auto") -> str:
    qvec = embed(user_msg)

    if mode == "exercise":
        query = {"user_id": user_id, "type": "exercise"}
    elif mode == "general":
        query = {"user_id": user_id, "type": "general"}
    else:
        query = {"user_id": user_id}

    history = list(
        chat_col.find(query)
               .sort("timestamp", -1)
               .limit(60)
    )

    scored = []
    for h in history:
        vec = safe_get_vec(h)
        if vec:
            cleaned = clean_text_korean_only(h["message"])
            if cleaned.strip():
                scored.append((cosine_similarity(qvec, vec), cleaned))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)

    # >>> 변경됨: RAG를 ChatML assistant 히스토리 형식으로 래핑
    rag_messages = []
    for _, msg in scored[:topk]:
        rag_messages.append(f"<|im_start|>assistant\n{msg}\n<|im_end|>")

    return "\n".join(rag_messages)



# ===== Prompt 정의 =====
SYSTEM_PROMPT_EXERCISE = (
    "너는 한국어 퍼스널 트레이너이다.\n"
    "사용자가 어떤 운동을 하고 있는지 첫 줄에서 자연스럽게 언급한다.\n"
    "분석 수치는 언급하지 않는다.\n"
    "숫자, 각도, cm, %, ° 등 수치 언급 금지.\n"
    "한자, 영어, 전문용어, 과학적 표현 금지.\n"
    "운동 동작은 느낌과 체중 흐름 중심으로 설명한다.\n\n"

    "🚫 [매우 중요한 규칙]\n"
    "사용자가 '아프다', '통증', '찌릿하다', '뻐근하다', '힘들다' 등 통증 관련 표현을 하면\n"
    "그 부위로 운동을 절대 시키지 말고 즉시 중단하도록 안내하라.\n"
    "아픈 부위에 자극이 가는 운동, 스트레칭, 버티기, 회전 동작을 절대 추천하지 말라.\n"
    "대신 무리가 없는 대체 운동이나 휴식을 권장하라.\n\n"

    "출력 형식:\n"
    "① 지금 하고 있는 운동 이름 + 자세 느낌 요약 (1~2문장)\n"
    "② 잘한 점 (1문장)\n"
    "③ 개선할 점 (• 불릿 2~3개)\n"
    "④ 코칭 큐 (• 불릿 3~5개, 명령형 4~8글자)\n"
    "⑤ 다음 세트 목표 (1문장)\n"
)


SYSTEM_PROMPT_GENERAL = (
    "너는 한국어 헬스케어 상담 AI이다.\n"
    "절대 운동 템플릿을 사용하지 말고, 짧고 따뜻하고 단호하게 답한다.\n"
    "영어, 한자, 전문용어 사용 금지.\n"
)



# ===== Persona 요약 =====
async def update_persona_background(user_id: str):
    chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(15))
    if not chats:
        return

    text = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])

    summary = await llm_generate([
        {
            "role": "system",
            "content": "<|im_start|>system\n사용자의 말투/통증 경향/운동 습관을 한국어로 5줄로 요약하라.\n<|im_end|>"
        },
        {
            "role": "user",
            "content": f"<|im_start|>user\n{text}\n<|im_end|>"
        }
    ])
    profile_col.update_one({"user_id": user_id}, {"$set": {"persona": summary}}, upsert=True)



# ===== 답변 생성 =====
async def generate_answer(user_id: str, user_msg: str, analysis: Dict[str, Any]):
    persona = profile_col.find_one({"user_id": user_id}) or {}
    persona_text = persona.get("persona", "")

    # 운동명 세팅
    if analysis and "exercise" in analysis and "detected_exercise" not in analysis:
        analysis["detected_exercise"] = analysis["exercise"]

    is_exercise = bool(analysis and analysis.get("detected_exercise"))

    if is_exercise:
        exercise_name = analysis["detected_exercise"]
        user_msg = f"{exercise_name} 운동 중: {user_msg}"

    mode = "exercise" if is_exercise else "general"
    rag = build_rag_context(user_id, user_msg, mode=mode)

    # 🔥 system prompt 하나에 persona + RAG 모두 포함
    system_prompt = SYSTEM_PROMPT_EXERCISE if is_exercise else SYSTEM_PROMPT_GENERAL

    system_content = f"""
{system_prompt}

👤 [사용자 말투/습관 요약]
{persona_text}

📚 [사용자 이전 대화 기록]
{rag}
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_msg},
    ]

    return await llm_generate(messages)



# ===== 저장 =====
def save_chat(user_id, message, response, embedding, analysis):
    chat_col.insert_one({
        "user_id": user_id,
        "message": message,
        "response": response,
        "embedding": embedding,
        "analysis": analysis or {},
        "timestamp": datetime.now(),
        "type": "exercise" if (analysis and analysis.get("detected_exercise")) else "general",
    })



# ===== Endpoints =====
@app.post("/chat")
async def chat_plain(data: ChatInput, background_tasks: BackgroundTasks):
    qvec = embed(data.message)
    answer = await generate_answer(data.user_id, data.message, analysis={})
    save_chat(data.user_id, data.message, answer, qvec, {})
    if chat_col.count_documents({"user_id": data.user_id}) >= 3:
        background_tasks.add_task(update_persona_background, data.user_id)
    return {"answer": answer}


@app.post("/chat_with_analysis")
async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):
    user_msg = data.message if data.message and data.message.strip() else "운동 자세 피드백 요청"

    qvec = embed(user_msg)

    answer = await generate_answer(data.user_id, user_msg, data.analysis)

    save_chat(data.user_id, user_msg, answer, qvec, data.analysis)

    if chat_col.count_documents({"user_id": data.user_id}) >= 3:
        background_tasks.add_task(update_persona_background, data.user_id)

    return {"answer": answer}

