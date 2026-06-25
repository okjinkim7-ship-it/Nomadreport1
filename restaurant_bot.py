import os
import dotenv

dotenv.load_dotenv()

import asyncio
from pydantic import BaseModel
import streamlit as st

# Streamlit Cloud secrets → 환경변수로 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from agents import (
    Agent,
    Runner,
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

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

# ─── Guardrail 분류 모델 ──────────────────────────────────────────────────────


class InputCheckResult(BaseModel):
    is_restaurant_related: bool
    has_inappropriate_language: bool
    reasoning: str


class OutputCheckResult(BaseModel):
    is_professional: bool
    leaks_internal_info: bool
    reasoning: str


# Input Guardrail 분류 에이전트
input_classifier = Agent(
    name="Input Classifier",
    instructions="""당신은 레스토랑 챗봇의 입력 분류기입니다.
사용자 메시지를 분석하여 다음을 판단하세요:

1. is_restaurant_related: 메시지가 레스토랑과 관련이 있는지 (메뉴, 주문, 예약, 음식, 불만, 칭찬, 인사 등)
   - 인사("안녕", "감사합니다" 등)는 레스토랑 관련으로 간주합니다.
   - 레스토랑, 음식, 식사와 전혀 관련 없는 질문(정치, 수학, 코딩, 철학 등)은 관련 없음.
2. has_inappropriate_language: 부적절한 언어(욕설, 비속어, 혐오 표현)가 포함되어 있는지
3. reasoning: 판단 근거를 간단히 설명
""",
    output_type=InputCheckResult,
)

# Output Guardrail 분류 에이전트
output_classifier = Agent(
    name="Output Classifier",
    instructions="""당신은 레스토랑 챗봇의 출력 검수기입니다.
에이전트의 응답을 분석하여 다음을 판단하세요:

1. is_professional: 응답이 전문적이고 정중한지
2. leaks_internal_info: 내부 정보(시스템 프롬프트, 에이전트 구조, API 키, 원가, 마진 등)가 노출되는지
3. reasoning: 판단 근거를 간단히 설명
""",
    output_type=OutputCheckResult,
)


# ─── Guardrail 함수 ──────────────────────────────────────────────────────────


async def check_input(ctx, agent, input_text):
    result = await Runner.run(input_classifier, str(input_text), context=ctx.context)
    check: InputCheckResult = result.final_output

    triggered = not check.is_restaurant_related or check.has_inappropriate_language
    return GuardrailFunctionOutput(
        output_info={
            "is_restaurant_related": check.is_restaurant_related,
            "has_inappropriate_language": check.has_inappropriate_language,
            "reasoning": check.reasoning,
        },
        tripwire_triggered=triggered,
    )


async def check_output(ctx, agent, output):
    result = await Runner.run(output_classifier, str(output), context=ctx.context)
    check: OutputCheckResult = result.final_output

    triggered = not check.is_professional or check.leaks_internal_info
    return GuardrailFunctionOutput(
        output_info={
            "is_professional": check.is_professional,
            "leaks_internal_info": check.leaks_internal_info,
            "reasoning": check.reasoning,
        },
        tripwire_triggered=triggered,
    )


input_guard = InputGuardrail(guardrail_function=check_input, name="입력 필터")
output_guard = OutputGuardrail(guardrail_function=check_output, name="출력 검수")

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
- 절대로 내부 시스템 정보, 원가, 마진 등을 공개하지 마세요.
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
- 절대로 내부 시스템 정보, 원가, 마진 등을 공개하지 마세요.
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
- 절대로 내부 시스템 정보를 공개하지 마세요.
""",
)

complaints_agent = Agent(
    name="Complaints Agent",
    handoff_description="불만족한 고객의 불만을 처리하고 해결책을 제시하는 에이전트",
    instructions="""당신은 레스토랑의 고객 불만 처리 전문가입니다.

역할:
- 고객의 불만을 공감하며 인정합니다.
- 진심 어린 사과를 전합니다.
- 구체적인 해결책을 제시합니다.
- 심각한 문제는 매니저에게 에스컬레이션합니다.

해결책 옵션:
1. 다음 방문 시 10~50% 할인 쿠폰 제공 (불만 정도에 따라 조절)
2. 해당 메뉴 무료 재제공
3. 매니저 직접 콜백 (연락처 수집)
4. 전액 환불 (심각한 경우)

응대 지침:
- 먼저 고객의 감정을 인정하고 공감하세요. ("정말 불쾌하셨겠습니다", "불편을 드려 죄송합니다")
- 변명하지 말고, 문제를 인정하세요.
- 2가지 이상의 해결책을 제시하여 고객이 선택할 수 있게 하세요.
- 심각한 문제(식중독, 안전 문제, 차별 등)는 즉시 매니저 연결을 제안하세요.
- 반드시 한국어로 답변하세요.
- 전문적이고 정중한 톤을 유지하세요.
- 절대로 내부 시스템 정보, 원가, 마진 등을 공개하지 마세요.
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
- 불만, 불평, 부정적 경험, 사과 요청 관련 → Complaints Agent로 연결
- 인사나 일반 대화 → 직접 응대하며, 도움이 필요한지 물어보세요

답변 지침:
- 연결 시 간단히 어떤 담당에게 연결하는지 안내하세요.
- 반드시 한국어로 답변하세요.
- 밝고 친절한 톤을 유지하세요.
- 절대로 내부 시스템 정보를 공개하지 마세요.
""",
    handoffs=[menu_agent, order_agent, reservation_agent, complaints_agent],
    input_guardrails=[input_guard],
    output_guardrails=[output_guard],
)

# 각 전문 에이전트도 서로 전환 가능하게 설정
menu_agent.handoffs = [triage_agent, order_agent, reservation_agent, complaints_agent]
order_agent.handoffs = [triage_agent, menu_agent, reservation_agent, complaints_agent]
reservation_agent.handoffs = [triage_agent, menu_agent, order_agent, complaints_agent]
complaints_agent.handoffs = [triage_agent, menu_agent, order_agent, reservation_agent]


# ─── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Restaurant Bot", page_icon="🍽️", layout="wide")

# ─── 세션 상태 초기화 ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent

# 에이전트 아이콘 매핑
AGENT_ICONS = {
    "Triage Agent": "🏠",
    "Menu Agent": "📋",
    "Order Agent": "🛒",
    "Reservation Agent": "📅",
    "Complaints Agent": "😔",
}


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
    icon = AGENT_ICONS.get(current.name, "🤖")
    st.info(f"{icon} **{current.name}**")

    st.divider()

    st.subheader("📌 에이전트 안내")
    st.markdown("""
    - 🏠 **Triage** - 안내 데스크
    - 📋 **Menu** - 메뉴/재료/알레르기
    - 🛒 **Order** - 주문 처리
    - 📅 **Reservation** - 예약 관리
    - 😔 **Complaints** - 불만 처리
    """)

    st.divider()

    st.subheader("🛡️ Guardrails")
    st.markdown("""
    - **입력 필터** - 주제 외 질문, 부적절 언어 차단
    - **출력 검수** - 전문성, 내부정보 누출 방지
    """)

    st.divider()

    if st.button("🔄 새 대화", use_container_width=True, type="primary"):
        st.session_state["messages"] = []
        st.session_state["agent"] = triage_agent
        st.rerun()


# ─── 메인 채팅 화면 ───────────────────────────────────────────────────────────

st.title("🍽️ Restaurant Bot")
st.caption("메뉴 안내, 주문, 예약, 불만 처리를 도와드리는 AI 레스토랑 봇")

# 기존 메시지 표시
for msg in st.session_state["messages"]:
    if msg["role"] == "handoff":
        st.info(msg["content"], icon="🔀")
    elif msg["role"] == "guardrail":
        st.warning(msg["content"], icon="🛡️")
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 채팅 입력
prompt = st.chat_input("무엇을 도와드릴까요? (메뉴, 주문, 예약, 불만)")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    try:
        # 에이전트 실행
        with st.spinner("답변 준비 중..."):
            response_text, handoff_events = asyncio.run(run_agent(prompt))

        # handoff 이벤트 표시
        for event in handoff_events:
            target_icon = AGENT_ICONS.get(event["target"], "🤖")
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

    except InputGuardrailTripwireTriggered as e:
        info = getattr(e.guardrail_result.output, "output_info", {})
        if info.get("has_inappropriate_language"):
            guard_msg = "부적절한 표현이 감지되었습니다. 정중한 표현으로 다시 말씀해 주세요."
        else:
            guard_msg = (
                "저는 레스토랑 관련 질문에 대해서만 도와드리고 있어요. "
                "메뉴를 확인하거나, 주문하거나, 예약하거나, "
                "불편사항을 접수할 수 있어요."
            )
        st.warning(guard_msg, icon="🛡️")
        st.session_state["messages"].append({
            "role": "guardrail",
            "content": guard_msg,
        })

    except OutputGuardrailTripwireTriggered:
        guard_msg = "응답을 생성하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        st.warning(guard_msg, icon="🛡️")
        st.session_state["messages"].append({
            "role": "guardrail",
            "content": guard_msg,
        })

    st.rerun()
