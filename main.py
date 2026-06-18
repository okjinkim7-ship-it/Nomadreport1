import dotenv

dotenv.load_dotenv()

import asyncio
import base64
import json
import os
import uuid
from datetime import datetime
import streamlit as st
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool, ImageGenerationTool

# ─── 상수 ───────────────────────────────────────────────────────────────────
JOURNAL_FILE = "journal.txt"
VS_ID_FILE = "vector_store_id.txt"
GOALS_FILE = "goals.txt"
SESSIONS_FILE = "sessions.json"
DB_PATH = "chat-gpt-clone-memory.db"

openai_client = OpenAI()


# ─── 세션 메타데이터 관리 ─────────────────────────────────────────────────────

def load_session_list() -> list[dict]:
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_session_list(sessions: list[dict]):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def upsert_session_meta(session_id: str, title: str):
    sessions = load_session_list()
    existing = next((s for s in sessions if s["id"] == session_id), None)
    if existing:
        existing["title"] = title
    else:
        sessions.insert(0, {
            "id": session_id,
            "title": title,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    save_session_list(sessions)


def remove_session_meta(session_id: str):
    sessions = [s for s in load_session_list() if s["id"] != session_id]
    save_session_list(sessions)


# ─── 벡터 스토어 관리 ────────────────────────────────────────────────────────

def load_or_create_vector_store() -> str:
    if os.path.exists(VS_ID_FILE):
        with open(VS_ID_FILE, "r", encoding="utf-8") as f:
            vs_id = f.read().strip()
        try:
            openai_client.vector_stores.retrieve(vs_id)
            return vs_id
        except Exception:
            pass

    vs = openai_client.vector_stores.create(name="Life Coach Memory")
    with open(VS_ID_FILE, "w", encoding="utf-8") as f:
        f.write(vs.id)
    return vs.id


def upload_bytes_to_vector_store(vs_id: str, content: bytes, filename: str) -> str:
    file_obj = openai_client.files.create(
        file=(filename, content, "text/plain"),
        purpose="assistants",
    )
    openai_client.vector_stores.files.create(
        vector_store_id=vs_id,
        file_id=file_obj.id,
    )
    return file_obj.id


def upload_local_file_to_vector_store(vs_id: str, path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return upload_bytes_to_vector_store(vs_id, content, os.path.basename(path))


def append_journal_entry(text: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## 일기 [{timestamp}]\n{text}\n")


# ─── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="wide")

# ─── 벡터 스토어 초기화 ───────────────────────────────────────────────────────

if "vs_id" not in st.session_state:
    with st.spinner("목표 문서 저장소 초기화 중..."):
        vs_id = load_or_create_vector_store()
        st.session_state["vs_id"] = vs_id
        if os.path.exists(GOALS_FILE) and "goals_uploaded" not in st.session_state:
            try:
                upload_local_file_to_vector_store(vs_id, GOALS_FILE)
                st.session_state["goals_uploaded"] = True
            except Exception:
                pass

vs_id = st.session_state["vs_id"]


# ─── 현재 세션 관리 ───────────────────────────────────────────────────────────

def make_agent(vs_id: str) -> Agent:
    return Agent(
        name="Life Coach",
        instructions="""
        당신은 따뜻하고 열정적인 라이프 코치입니다.

        역할:
        - 사용자의 목표 달성과 자기 성장을 적극적으로 지원합니다
        - 동기부여, 자기계발 팁, 습관 형성 조언을 제공합니다
        - 항상 공감하며 격려하는 긍정적인 말투를 유지합니다
        - 사용자의 고민을 경청하고 구체적인 해결책을 제시합니다

        도구 활용 지침:
        1. **파일 검색(file_search)**: 목표·일기·진행 상황 관련 질문이 오면
           반드시 먼저 파일을 검색하여 개인 목표 문서와 일기를 확인하세요.
        2. **웹 검색(web_search)**: 최신 팁·연구·방법을 검색해 개인 목표와 결합한
           맞춤 조언을 제공하세요.
        3. **이미지 생성(image_generation)**: 다음 상황에서 적극적으로 이미지를 생성하세요.
           - 사용자가 목표를 달성했을 때 → 축하 이미지
           - 비전 보드 요청 → 목표들을 담은 영감을 주는 비전 보드
           - 동기부여 포스터 요청 → 맞춤 메시지가 담긴 포스터
           - 진행 상황 시각화 요청 → 진행 상황을 보여주는 이미지
           프롬프트는 **반드시 영어**로 작성하고, 밝고 긍정적이며 고해상도 스타일로 요청하세요.

        답변 형식:
        - 조언은 단계별로 나누어 실행하기 쉽게 설명하세요
        - 사용자를 "당신"으로 부르며 친근하게 대화하세요
        - 반드시 한국어로 답변하세요
        - 답변 마지막에 격려의 말 한 마디를 꼭 추가하세요
        """,
        tools=[
            FileSearchTool(vector_store_ids=[vs_id], max_num_results=5),
            WebSearchTool(),
            ImageGenerationTool(
                tool_config={"type": "image_generation", "quality": "low", "size": "1024x1024"}
            ),
        ],
    )


def new_chat():
    """새 대화 세션 생성"""
    session_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = session_id
    st.session_state["agent"] = make_agent(vs_id)
    st.session_state["session"] = SQLiteSession(session_id, DB_PATH)
    st.session_state["messages"] = []
    # 세션별 시각 메시지 저장소 초기화
    if "session_messages" not in st.session_state:
        st.session_state["session_messages"] = {}
    st.session_state["session_messages"][session_id] = []


def switch_chat(session_id: str):
    """기존 대화 세션으로 전환"""
    st.session_state["current_session_id"] = session_id
    st.session_state["agent"] = make_agent(vs_id)
    st.session_state["session"] = SQLiteSession(session_id, DB_PATH)
    # 저장된 시각 메시지 복원 (이미지는 바이트로 저장됨)
    saved = st.session_state.get("session_messages", {}).get(session_id, [])
    st.session_state["messages"] = saved


# 최초 실행 시 새 대화 생성
if "current_session_id" not in st.session_state:
    new_chat()

agent = st.session_state["agent"]
session = st.session_state["session"]
current_sid = st.session_state["current_session_id"]


# ─── 에이전트 실행 ────────────────────────────────────────────────────────────

async def run_agent(message: str):
    stream = Runner.run_streamed(agent, message, session=session)

    response_text = ""
    active_tools: list[str] = []
    generated_images: list[bytes] = []
    shown_image_ids: set[str] = set()  # 중복 표시 방지

    with st.chat_message("assistant"):
        status_area = st.empty()
        text_placeholder = st.empty()
        image_area = st.container()

        async for event in stream.stream_events():
            if event.type == "run_item_stream_event":
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None) if item else None

                if item_type == "tool_search_call_item":
                    if "📂 목표 문서 검색 중..." not in active_tools:
                        active_tools.append("📂 목표 문서 검색 중...")
                    status_area.info("\n\n".join(active_tools))

                elif item_type == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    if raw:
                        raw_item_type = getattr(raw, "type", "")

                        if raw_item_type == "image_generation_call":
                            result_b64 = getattr(raw, "result", None)
                            img_id = getattr(raw, "id", "")

                            # result가 있으면 즉시 표시 (status가 "generating"이어도 결과가 담겨 옴)
                            if result_b64 and img_id not in shown_image_ids:
                                shown_image_ids.add(img_id)
                                img_bytes = base64.b64decode(result_b64)
                                generated_images.append(img_bytes)
                                image_area.image(img_bytes, use_container_width=True)
                            elif not result_b64:
                                # 아직 생성 중 — 상태 표시
                                if "🎨 이미지 생성 중..." not in active_tools:
                                    active_tools.append("🎨 이미지 생성 중...")
                                    status_area.info("\n\n".join(active_tools))

                        elif "web_search" in str(raw_item_type):
                            action = getattr(raw, "action", None)
                            if action:
                                query = getattr(action, "query", None)
                                if query:
                                    active_tools.append(f"🔍 웹 검색: **{query}**")
                                    status_area.info("\n\n".join(active_tools))

            elif event.type == "raw_response_event":
                if hasattr(event.data, "type") and event.data.type == "response.output_text.delta":
                    response_text += event.data.delta
                    text_placeholder.markdown(response_text)

        status_area.empty()

    return response_text, generated_images


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌱 Life Coach")

    # ── 새 대화 버튼 ──────────────────────────────────────────────────
    if st.button("✏️ 새 대화", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.divider()

    # ── 대화 히스토리 목록 ────────────────────────────────────────────
    st.subheader("💬 대화 목록")
    session_list = load_session_list()

    if not session_list:
        st.caption("대화 기록이 없습니다.")
    else:
        for s in session_list:
            sid = s["id"]
            title = s.get("title", "새 대화")
            created = s.get("created_at", "")
            is_active = sid == current_sid

            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"**{title}**" if is_active else title
                if st.button(
                    label,
                    key=f"sess_{sid}",
                    use_container_width=True,
                    help=created,
                ):
                    if not is_active:
                        switch_chat(sid)
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{sid}", help="삭제"):
                    # SQLite 세션 데이터 삭제
                    tmp = SQLiteSession(sid, DB_PATH)
                    asyncio.run(tmp.clear_session())
                    remove_session_meta(sid)
                    # 시각 메시지도 삭제
                    st.session_state.get("session_messages", {}).pop(sid, None)
                    if sid == current_sid:
                        new_chat()
                    st.rerun()

    st.divider()

    # ── 목표 문서 업로드 ──────────────────────────────────────────────
    st.subheader("📎 목표 문서")
    uploaded_file = st.file_uploader(
        "PDF/TXT 업로드",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("📤 업로드", use_container_width=True):
            with st.spinner("업로드 중..."):
                try:
                    upload_bytes_to_vector_store(vs_id, uploaded_file.getvalue(), uploaded_file.name)
                    st.success(f"✅ '{uploaded_file.name}' 완료!")
                except Exception as e:
                    st.error(f"실패: {e}")

    st.divider()

    # ── 일기 기록 ────────────────────────────────────────────────────
    st.subheader("📝 오늘의 일기")
    journal_text = st.text_area(
        "일기",
        placeholder="오늘 운동 30분 완료! 물 2리터 마심.",
        height=100,
        label_visibility="collapsed",
    )
    if st.button("💾 저장 & 업로드", use_container_width=True):
        if journal_text.strip():
            with st.spinner("저장 중..."):
                try:
                    append_journal_entry(journal_text.strip())
                    upload_local_file_to_vector_store(vs_id, JOURNAL_FILE)
                    st.success("✅ 저장됐습니다!")
                except Exception as e:
                    st.error(f"실패: {e}")
        else:
            st.warning("내용을 입력하세요.")


# ─── 메인 채팅 화면 ───────────────────────────────────────────────────────────

st.title("🌱 Life Coach Agent")
st.caption("목표를 기억하고, 조언하고, 비전 보드를 만드는 AI 라이프 코치")

# 기존 메시지 표시
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        for img_bytes in msg.get("images", []):
            st.image(img_bytes, use_container_width=True)

# 채팅 입력
prompt = st.chat_input("코치에게 고민을 이야기해보세요...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    user_msg = {"role": "user", "content": prompt, "images": []}
    st.session_state["messages"].append(user_msg)

    # 세션 메타데이터 저장 (첫 메시지로 제목 설정)
    session_list = load_session_list()
    existing = next((s for s in session_list if s["id"] == current_sid), None)
    title = (prompt[:28] + "...") if len(prompt) > 28 else prompt
    if not existing:
        upsert_session_meta(current_sid, title)

    # 에이전트 실행
    response_text, generated_images = asyncio.run(run_agent(prompt))

    if response_text or generated_images:
        asst_msg = {"role": "assistant", "content": response_text, "images": generated_images}
        st.session_state["messages"].append(asst_msg)

        # 세션별 시각 메시지 저장
        if "session_messages" not in st.session_state:
            st.session_state["session_messages"] = {}
        st.session_state["session_messages"][current_sid] = st.session_state["messages"]
