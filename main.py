import dotenv

dotenv.load_dotenv()

import asyncio
import json
import os
from datetime import datetime
import streamlit as st
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool

# ─── 상수 ───────────────────────────────────────────────────────────────────
JOURNAL_FILE = "journal.txt"
VS_ID_FILE = "vector_store_id.txt"
GOALS_FILE = "goals.txt"

# ─── OpenAI 클라이언트 (벡터 스토어 관리용) ──────────────────────────────────
openai_client = OpenAI()


# ─── 벡터 스토어 관리 함수 ───────────────────────────────────────────────────

def load_or_create_vector_store() -> str:
    """벡터 스토어 ID 로드 또는 신규 생성"""
    if os.path.exists(VS_ID_FILE):
        with open(VS_ID_FILE, "r", encoding="utf-8") as f:
            vs_id = f.read().strip()
        try:
            openai_client.vector_stores.retrieve(vs_id)
            return vs_id
        except Exception:
            pass  # 유효하지 않으면 새로 생성

    vs = openai_client.vector_stores.create(name="Life Coach Memory")
    with open(VS_ID_FILE, "w", encoding="utf-8") as f:
        f.write(vs.id)
    return vs.id


def upload_bytes_to_vector_store(vs_id: str, content: bytes, filename: str) -> str:
    """바이트 데이터를 OpenAI 파일로 업로드 후 벡터 스토어에 추가"""
    # 파일 업로드
    file_obj = openai_client.files.create(
        file=(filename, content, "text/plain"),
        purpose="assistants",
    )
    # 벡터 스토어에 연결 (완료 대기)
    openai_client.vector_stores.files.create(
        vector_store_id=vs_id,
        file_id=file_obj.id,
    )
    return file_obj.id


def upload_local_file_to_vector_store(vs_id: str, path: str) -> str:
    """로컬 파일을 벡터 스토어에 업로드"""
    with open(path, "rb") as f:
        content = f.read()
    return upload_bytes_to_vector_store(vs_id, content, os.path.basename(path))


def append_journal_entry(text: str):
    """일기 항목을 journal.txt 에 추가 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## 일기 [{timestamp}]\n{text}\n")


# ─── 앱 초기화 ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="centered")
st.title("🌱 Life Coach Agent")
st.caption("목표와 일기를 기억하는 AI 라이프 코치입니다")

# 벡터 스토어 ID (세션)
if "vs_id" not in st.session_state:
    with st.spinner("목표 문서 저장소 초기화 중..."):
        vs_id = load_or_create_vector_store()
        st.session_state["vs_id"] = vs_id

        # 최초 실행 시 기본 goals.txt 자동 업로드
        if os.path.exists(GOALS_FILE) and "goals_uploaded" not in st.session_state:
            try:
                upload_local_file_to_vector_store(vs_id, GOALS_FILE)
                st.session_state["goals_uploaded"] = True
            except Exception as e:
                st.warning(f"기본 목표 파일 업로드 실패: {e}")

vs_id = st.session_state["vs_id"]

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

        도구 활용 지침:
        - **파일 검색(file_search)**: 사용자의 목표나 진행 상황에 관한 질문이 오면
          반드시 먼저 파일 검색으로 업로드된 목표 문서와 일기를 확인하세요.
          예) "내 목표가 뭐야?", "운동 목표는?", "이번 달 계획은?" 등
        - **웹 검색(web_search)**: 파일 검색 후, 목표와 관련된 최신 팁·연구·방법을
          웹에서 추가로 검색하여 개인화된 추천을 제공하세요.
        - 두 도구를 함께 사용해 "개인 목표 + 최신 정보"를 결합한 조언을 하세요.

        답변 형식:
        - 조언은 단계별로 나누어 실행하기 쉽게 설명하세요
        - 사용자를 "당신"으로 부르며 친근하게 대화하세요
        - 반드시 한국어로 답변하세요
        - 답변 마지막에 격려의 말 한 마디를 꼭 추가하세요
        """,
        tools=[
            FileSearchTool(
                vector_store_ids=[vs_id],
                max_num_results=5,
            ),
            WebSearchTool(),
        ],
    )

agent = st.session_state["agent"]

# SQLite 세션 (대화 메모리)
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-session",
        "chat-gpt-clone-memory.db",
    )

session = st.session_state["session"]

# 화면 표시용 메시지 히스토리
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ─── 에이전트 실행 ────────────────────────────────────────────────────────────

async def run_agent(message: str):
    stream = Runner.run_streamed(agent, message, session=session)

    response_text = ""
    active_tools: list[str] = []

    with st.chat_message("assistant"):
        status_area = st.empty()
        text_placeholder = st.empty()

        async for event in stream.stream_events():
            # 도구 호출 감지 — RunItemStreamEvent는 event.item 을 직접 가짐
            if event.type == "run_item_stream_event":
                item = getattr(event, "item", None)

                # 파일 검색 (ToolSearchCallItem)
                if item and getattr(item, "type", None) == "tool_search_call_item":
                    active_tools.append("📂 목표 문서 검색 중...")
                    status_area.info("\n\n".join(active_tools))

                # 일반 도구 호출 (ToolCallItem) — 웹 검색 포함
                elif item and getattr(item, "type", None) == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    if raw:
                        raw_type = str(type(raw))
                        query = None

                        # 웹 검색: ResponseFunctionWebSearch
                        if "web_search" in raw_type.lower() or "web_search" in str(getattr(raw, "type", "")):
                            action = getattr(raw, "action", None)
                            if action:
                                query = getattr(action, "query", None)
                            if query:
                                active_tools.append(f"🔍 웹 검색: **{query}**")
                                status_area.info("\n\n".join(active_tools))

            # 텍스트 스트리밍
            elif event.type == "raw_response_event":
                if hasattr(event.data, "type") and event.data.type == "response.output_text.delta":
                    response_text += event.data.delta
                    text_placeholder.markdown(response_text)

        status_area.empty()

    return response_text


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 설정")

    # ── 목표 문서 업로드 ──────────────────────────────────────────────
    st.subheader("📎 목표 문서 업로드")
    st.caption("PDF 또는 TXT 파일을 업로드하면 코치가 참고합니다.")
    uploaded_file = st.file_uploader(
        "파일 선택",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("📤 업로드", use_container_width=True):
            with st.spinner("파일을 목표 저장소에 업로드 중..."):
                try:
                    upload_bytes_to_vector_store(
                        vs_id,
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                    st.success(f"✅ '{uploaded_file.name}' 업로드 완료!")
                except Exception as e:
                    st.error(f"업로드 실패: {e}")

    st.divider()

    # ── 일기 / 진행 상황 기록 ─────────────────────────────────────────
    st.subheader("📝 오늘의 일기")
    st.caption("목표 달성 현황이나 오늘의 기록을 남겨보세요.")
    journal_text = st.text_area(
        "일기 내용",
        placeholder="예) 오늘 운동 30분 완료! 물 2리터 마심. 독서 1챕터 읽었다.",
        height=120,
        label_visibility="collapsed",
    )
    if st.button("💾 일기 저장 & 업로드", use_container_width=True):
        if journal_text.strip():
            with st.spinner("일기를 저장 중..."):
                try:
                    append_journal_entry(journal_text.strip())
                    upload_local_file_to_vector_store(vs_id, JOURNAL_FILE)
                    st.success("✅ 일기가 저장되었습니다!")
                except Exception as e:
                    st.error(f"일기 저장 실패: {e}")
        else:
            st.warning("일기 내용을 입력해주세요.")

    st.divider()

    # ── 대화 관리 ─────────────────────────────────────────────────────
    st.subheader("💬 대화 관리")
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        asyncio.run(session.clear_session())
        st.session_state["messages"] = []
        st.rerun()

    history = asyncio.run(session.get_items())
    st.caption(f"저장된 메시지: {len(history)}개")

    st.divider()

    # ── 예시 질문 ─────────────────────────────────────────────────────
    st.subheader("💡 예시 질문")
    st.markdown("- 내 운동 목표 달성은 잘 되어가고 있어?")
    st.markdown("- 이번 달 중점 과제가 뭐였지?")
    st.markdown("- 독서 목표에 대한 팁을 알려줘")
    st.markdown("- 아침 루틴 만드는 방법")
    st.markdown("- 재정 목표 달성 전략이 궁금해")


# ─── 채팅 화면 ────────────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("코치에게 고민을 이야기해보세요...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_text = asyncio.run(run_agent(prompt))

    if response_text:
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
