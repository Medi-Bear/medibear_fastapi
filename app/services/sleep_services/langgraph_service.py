def general_chat_node(state: MessagesState):
    """일반 대화용 노드"""
    user_msg = state["messages"][-1]["content"]
    prompt = f"""
    너는 수면 코치지만, 친구처럼 편하게 대화해주는 챗봇이야.
    사용자가 수면, 스트레스, 피로, 루틴, 휴식 등에 대해 물어보면 답변해줘.
    모든 답변은 4-5줄로 답변해줘.
    단, 너무 과학적이거나 의학적인 설명은 피하고,
    짧고 부드러운 문장으로 이야기하듯 답해줘.

    사용자 질문: {user_msg}
    """
    response = llm.invoke(prompt).content
    state["messages"].append({"role": "ai", "content": response})
    return state


def save_chat_node(state: MessagesState):
    """DB 저장"""
    # 예: save_chat(user_id, user_message, ai_message, chat_type)
    user_id = state.get("user_id", "unknown")
    msg = state["messages"][-2]["content"]
    res = state["messages"][-1]["content"]
    save_chat(user_id, msg, res, chat_type="general")
    return state