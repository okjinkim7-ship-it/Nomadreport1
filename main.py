import dotenv

dotenv.load_dotenv()

import asyncio
import base64
import json
import os
from datetime import datetime
import streamlit as st
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool, ImageGenerationTool

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


# ─── 앱 초기화 ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Life Coach", page_icon="🌱", layout="centered")
st.title("🌱 Life Coach Agent")
st.caption("목표를 기억하고, 조언하고, 비전 보드를 만드는 AI 라이프 코치")

# 벡터 스토어 ID
if "vs_id" not in st.session_state:
    with st.spinner("목표 문서 저장소 초기화 중..."):
        vs_id = load_or_create_vector_store()
        st.session_state["vs_id"] = vs_id

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
            FileSearchTool(
                vector_store_ids=[vs_id],
                max_num_results=5,
            ),
            WebSearchTool(),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "low",
                    "size": "1024x1024",
                }
            ),
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
# 각 메시지: {"role": str, "content": str, "images": list[bytes]}
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ─── 에이전트 실행 ────────────────────────────────────────────────────────────

async def run_agent(message: str):
    stream = Runner.run_streamed(agent, message, session=session)

    response_text = ""
    active_tools: list[str] = []
    generated_images: list[bytes] = []

    with st.chat_message("assistant"):
        status_area = st.empty()
        text_placeholder = st.empty()
        image_area = st.container()

        async for event in stream.stream_events():

            # ── 도구 호출 / 결과 이벤트 ─────────────────────────────────
            if event.type == "run_item_stream_event":
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None) if item else None

                # 파일 검색 (ToolSearchCallItem)
                if item_type == "tool_search_call_item":
                    if "📂 목표 문서 검색 중..." not in active_tools:
                        active_tools.append("📂 목표 문서 검색 중...")
                    status_area.info("\n\n".join(active_tools))

                # 일반 도구 호출 (ToolCallItem)
                elif item_type == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    if raw:
                        raw_item_type = getattr(raw, "type", "")

                        # 이미지 생성
                        if raw_item_type == "image_generation_call":
                            status = getattr(raw, "status", "")
                            if status in ("in_progress", "generating"):
                                if "🎨 이미지 생성 중..." not in active_tools:
                                    active_tools.append("🎨 이미지 생성 중...")
                                    status_area.info("\n\n".join(active_tools))
                            elif status == "completed":
                                result_b64 = getattr(raw, "result", None)
                                if result_b64:
                                    img_bytes = base64.b64decode(result_b64)
                                    generated_images.append(img_bytes)
                                    image_area.image(img_bytes, use_container_width=True)

                        # 웹 검색 (ResponseFunctionWebSearch)
                        elif "web_search" in str(raw_item_type):
                            action = getattr(raw, "action", None)
                            if action:
                                query = getattr(action, "query", None)
                                if query:
                                    active_tools.append(f"🔍 웹 검색: **{query}**")
                                    status_area.info("\n\n".join(active_tools))

            # ── 텍스트 스트리밍 ─────────────────────────────────────────
            elif event.type == "raw_response_event":
                if hasattr(event.data, "type") and event.data.type == "response.output_text.delta":
                    response_text += event.data.delta
                    text_placeholder.markdown(response_text)

        status_area.empty()

    return response_text, generated_images


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
    st.markdown("- 올해 목표로 비전 보드를 만들어줘")
    st.markdown("- 동기부여 포스터 하나 만들어줘")
    st.markdown("- 독서 목표 달성했어! 축하 이미지 만들어줘")
    st.markdown("- 이번 달 재정 목표 전략 알려줘")


# ─── 채팅 화면 ────────────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        for img_bytes in msg.get("images", []):
            st.image(img_bytes, use_container_width=True)

prompt = st.chat_input("코치에게 고민을 이야기해보세요...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt, "images": []})

    response_text, generated_images = asyncio.run(run_agent(prompt))

    if response_text or generated_images:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text,
            "images": generated_images,
        })
