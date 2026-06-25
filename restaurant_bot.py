import dotenv

dotenv.load_dotenv()

import asyncio
import streamlit as st
from agents import Agent, Runner

# ─── 메뉴 데이터 ─────────────────────────────────────────────────────────────

MENU_DATA = """
🍽️ 레스토랑 메뉴

## 🥩 메인 요리
1. 안심 스테이크 - 45,000원 (소고기 안심, 감자, 로즈마리)
2. 연어 스테이크 - 35,000원 (노르웨이 연어, 레몬버터소스)
3. 트러플 파스타 - 28,000원 (트러플오일, 파마산치즈, 크림)
4. 해산물 리조또 - 30,000원 (새우, 홍합, 오징어, 파마산치즈)
5. 채식 버거 - 18,000원 (콩패티, 통밀빵, 아보카도) [채식]
6. 두부 스테이크 - 22,000원 (유기농 두부, 표고버섯, 테리야키소스) [채식]
7. 그린 파스타 - 24,000원 (바질페스토, 호두, 체리토마토) [채식/비건]

## 🥗 에피타이저 & 샐러드
8. 시저 샐러드 - 14,000원 (로메인, 파마산, 크루통, 앤초비드레싱) [견과류]
9. 카프레제 - 12,000원 (모짜렐라, 토마토, 바질) [채식]
10. 새우 칵테일 - 16,000원 (타이거새우, 칵테일소스) [갑각류]
11. 버섯 수프 - 10,000원 (표고·양송이·느타리, 크림) [유제품]

## 🍰 디저트
12. 티라미수 - 12,000원 (마스카포네, 에스프레소, 코코아) [유제품/달걀]
13. 과일 셔벗 - 9,000원 (망고·패션프루트·라즈베리) [비건]
14. 초콜릿 브라우니 - 11,000원 (다크초콜릿, 호두) [견과류/유제품]

## 🥤 음료
15. 스파클링 워터 - 5,000원
16. 생과일 주스 - 8,000원 (오렌지/사과/당근)
17. 하우스 와인 (글라스) - 12,000원 (레드/화이트)

## ⚠️ 알레르기 정보
- 견과류: 8, 14번
- 유제품: 3, 4, 8, 11, 12, 14번
- 갑각류/해산물: 4, 10번
- 글루텐: 1, 3, 5, 7, 8, 14번
- 달걀: 12번
"""

# ─── 전문 에이전트 정의 ───────────────────────────────────────────────────────

menu_agent = Agent(
    name="Menu Agent",
    handoff_description="메뉴, 재료, 알레르기 관련 질문을 처리하는 에이전트",
    instructions=f"""당신은 레스토랑의 메뉴 전문가입니다.

역할:
- 메뉴 항목, 가격, 재료에 대한 질문에 답변합니다.
- 알레르기 관련 질문에 정확하게 답변합니다.
- 채식/비건 메뉴를 안내합니다.
- 메뉴 추천을 해줍니다.

다음은 현재 레스토랑 메뉴입니다:
{MENU_DATA}

답변 지침:
- 항상 친절하고 전문적으로 답변하세요.
- 알레르기 관련 질문은 특히 정확하게 답변하세요.
- 반드시 한국어로 답변하세요.
- 메뉴에 없는 음식을 질문하면 정중히 없다고 안내하세요.
""",
)

order_agent = Agent(
    name="Order Agent",
    handoff_description="주문을 받고 확인하는 에이전트",
    instructions=f"""당신은 레스토랑의 주문 담당입니다.

역할:
- 고객의 주문을 받고 확인합니다.
- 주문 내역을 정리하여 보여줍니다.
- 합계 금액을 계산합니다.
- 추가 요청사항(알레르기 제외, 추가 토핑 등)을 처리합니다.

다음은 현재 레스토랑 메뉴입니다:
{MENU_DATA}

주문 처리 지침:
- 주문을 받으면 항목, 수량, 가격을 표로 정리하세요.
- 총 합계를 계산하여 보여주세요.
- 주문 확인 시 "주문이 확정되었습니다"라고 안내하세요.
- 반드시 한국어로 답변하세요.
- 메뉴에 없는 음식을 주문하면 정중히 안내하세요.
""",
)

reservation_agent = Agent(
    name="Reservation Agent",
    handoff_description="테이블 예약을 처리하는 에이전트",
    instructions="""당신은 레스토랑의 예약 담당입니다.

역할:
- 테이블 예약을 처리합니다.
- 예약에 필요한 정보를 수집합니다: 날짜, 시간, 인원수, 예약자 이름, 연락처.
- 예약 확인을 제공합니다.

예약 처리 지침:
- 필요한 정보를 하나씩 친절하게 물어보세요.
- 영업시간: 11:30 ~ 22:00 (라스트 오더 21:00)
- 최대 수용 인원: 8명 (초과 시 단체석 안내)
- 예약 가능 시간: 30분 단위
- 모든 정보가 모이면 예약 내역을 정리하여 확인해주세요.
- 반드시 한국어로 답변하세요.
""",
)

# Triage 에이전트 (라우터) - handoffs로 전문 에이전트 연결
triage_agent = Agent(
    name="Triage Agent",
    handoff_description="고객 요청을 파악하여 적절한 전문 에이전트로 연결하는 라우터",
    instructions="""당신은 레스토랑의 안내 데스크 담당입니다.

역할:
- 고객이 무엇을 원하는지 파악합니다.
- 적절한 전문 에이전트로 연결합니다.

라우팅 규칙:
- 메뉴, 재료, 알레르기, 음식 추천 관련 → Menu Agent로 연결
- 주문, 음식 시키기, 결제 관련 → Order Agent로 연결
- 예약, 테이블, 방문 일정 관련 → Reservation Agent로 연결
- 인사나 일반 대화 → 직접 응대하며, 도움이 필요한지 물어보세요

답변 지침:
- 연결 시 간단히 어떤 담당에게 연결하는지 안내하세요.
- 반드시 한국어로 답변하세요.
- 밝고 친절한 톤을 유지하세요.
""",
    handoffs=[menu_agent, order_agent, reservation_agent],
)

# 각 전문 에이전트도 Triage로 돌아갈 수 있게 설정
menu_agent.handoffs = [triage_agent, order_agent, reservation_agent]
order_agent.handoffs = [triage_agent, menu_agent, reservation_agent]
reservation_agent.handoffs = [triage_agent, menu_agent, order_agent]


# ─── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Restaurant Bot", page_icon="🍽️", layout="wide")

# ─── 세션 상태 초기화 ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent


# ─── 에이전트 실행 ────────────────────────────────────────────────────────────

async def run_agent(message: str):
    current_agent = st.session_state["agent"]
    result = await Runner.run(current_agent, message)

    # handoff가 일어났는지 확인하고, 마지막 에이전트를 저장
    st.session_state["agent"] = result.last_agent

    # handoff 이벤트 추출
    handoff_events = []
    for item in result.new_items:
        item_type = getattr(item, "type", None)
        if item_type == "handoff_output_item":
            source = getattr(item, "source_agent", None)
            target = getattr(item, "target_agent", None)
            if source and target:
                handoff_events.append({
                    "source": source.name,
                    "target": target.name,
                })

    return result.final_output, handoff_events


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🍽️ Restaurant Bot")
    st.divider()

    st.subheader("🤖 현재 담당")
    current = st.session_state["agent"]
    agent_icons = {
        "Triage Agent": "🏠",
        "Menu Agent": "📋",
        "Order Agent": "🛒",
        "Reservation Agent": "📅",
    }
    icon = agent_icons.get(current.name, "🤖")
    st.info(f"{icon} **{current.name}**")

    st.divider()

    st.subheader("📌 에이전트 안내")
    st.markdown("""
    - 🏠 **Triage** - 안내 데스크
    - 📋 **Menu** - 메뉴/재료/알레르기
    - 🛒 **Order** - 주문 처리
    - 📅 **Reservation** - 예약 관리
    """)

    st.divider()

    if st.button("🔄 새 대화", use_container_width=True, type="primary"):
        st.session_state["messages"] = []
        st.session_state["agent"] = triage_agent
        st.rerun()


# ─── 메인 채팅 화면 ───────────────────────────────────────────────────────────

st.title("🍽️ Restaurant Bot")
st.caption("메뉴 안내, 주문, 예약을 도와드리는 AI 레스토랑 봇")

# 기존 메시지 표시
for msg in st.session_state["messages"]:
    if msg["role"] == "handoff":
        st.info(msg["content"], icon="🔀")
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 채팅 입력
prompt = st.chat_input("무엇을 도와드릴까요? (메뉴, 주문, 예약)")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 에이전트 실행
    with st.spinner("답변 준비 중..."):
        response_text, handoff_events = asyncio.run(run_agent(prompt))

    # handoff 이벤트 표시
    for event in handoff_events:
        target_icon = agent_icons.get(event["target"], "🤖")
        handoff_msg = f"{target_icon} **{event['target']}**에게 연결합니다..."
        st.info(handoff_msg, icon="🔀")
        st.session_state["messages"].append({
            "role": "handoff",
            "content": handoff_msg,
        })

    # 응답 표시
    if response_text:
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text,
        })

    st.rerun()
