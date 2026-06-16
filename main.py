import dotenv

dotenv.load_dotenv()

import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession, WebSearchTool

st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="centered")
st.title("🌱 Life Coach Agent")
st.caption("당신의 성장을 응원하는 AI 라이프 코치입니다")

# Agent 초기화
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions="""
        당신은 따뜻하고 열정적인 라이프 코치입니다.

        역할:
        - 사용자의 목표 달성과 자기 성장을 적극적으로 지원합니다
        - 동기부여, 자기계발 팁, 습관 형성 조언을 제공합니다
        - 항상 공감하며 격려하는 긍정적인 말투를 유지합니다
        - 사용자의 고민을 경청하고 구체적인 해결책을 제시합니다

        지침:
        - 웹 검색으로 최신 자기계발 연구와 검증된 방법을 찾아 활용하세요
        - 조언은 단계별로 나누어 실행하기 쉽게 설명하세요
        - 사용자를 이름 대신 "당신"으로 부르며 친근하게 대화하세요
        - 반드시 한국어로 답변하세요
        - 답변 마지막에 격려의 말 한 마디를 꼭 추가하세요
        """,
        tools=[WebSearchTool()],
    )

agent = st.session_state["agent"]

# SQLite 세션 (에이전트 메모리)
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-session",
        "chat-gpt-clone-memory.db",
    )

session = st.session_state["session"]

# 화면 표시용 메시지 히스토리
if "messages" not in st.session_state:
    st.session_state["messages"] = []


async def run_agent(message: str):
    stream = Runner.run_streamed(agent, message, session=session)

    response_text = ""
    search_queries = []

    with st.chat_message("assistant"):
        search_status = st.empty()
        text_placeholder = st.empty()

        async for event in stream.stream_events():
            # 도구 호출 이벤트 감지 (웹 검색)
            if event.type == "run_item_stream_event":
                item = getattr(event.data, "item", None)
                if item and getattr(item, "type", None) == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    if raw:
                        query = None
                        # WebSearchTool 쿼리 추출
                        if hasattr(raw, "parameters"):
                            query = getattr(raw.parameters, "query", None)
                        elif hasattr(raw, "input"):
                            import json
                            try:
                                inp = json.loads(raw.input) if isinstance(raw.input, str) else raw.input
                                query = inp.get("query", "")
                            except Exception:
                                pass
                        if query:
                            search_queries.append(query)
                            search_status.info(f"🔍 웹 검색 중: **{query}**")

            # 텍스트 스트리밍
            elif event.type == "raw_response_event":
                if hasattr(event.data, "type") and event.data.type == "response.output_text.delta":
                    response_text += event.data.delta
                    text_placeholder.markdown(response_text)

        # 검색 완료 후 상태 숨기기
        if search_queries:
            search_status.empty()

    return response_text, search_queries


# 사이드바 - 메모리 관리
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        asyncio.run(session.clear_session())
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.subheader("📋 대화 기록")
    history = asyncio.run(session.get_items())
    st.write(f"저장된 메시지: {len(history)}개")

    st.divider()
    st.markdown("**💡 대화 주제 예시**")
    st.markdown("- 아침 루틴 만들기")
    st.markdown("- 습관 형성 방법")
    st.markdown("- 목표 설정과 달성")
    st.markdown("- 집중력 향상 팁")
    st.markdown("- 스트레스 관리")

# 기존 메시지 표시
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 채팅 입력
prompt = st.chat_input("코치에게 고민을 이야기해보세요...")

if prompt:
    # 사용자 메시지 표시 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 에이전트 실행 및 응답 스트리밍
    response_text, search_queries = asyncio.run(run_agent(prompt))

    # 응답 저장
    if response_text:
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
