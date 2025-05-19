import os
from dotenv import load_dotenv
import requests

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from tool_service import place_search_tool

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    model_name= "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature= 0.7,
    streaming= True
)

memory= ConversationBufferMemory(
    memory_key= "chat_history",
    return_messages= True
)

guardrail_message= SystemMessage(
    content = """
당신은 여행지 추천과 계획 작성 전문 에이전트입니다.
반드시 아래의 행동 규칙을 따르세요:
1. 생각(Thought), 행동(Action), 행동 입력(Action Input), 관찰(Observation), 생각(Thought), 최종 답변(Final Answer)의 순서를 지켜야 합니다.
2. Action과 Final Answer를 동시에 출력하지 마세요.
3. Observation이 반드시 출력된 후에만 Final Answer를 출력하세요.
4. 필요하지 않은 Action은 출력하지 마세요.
출력 예시는 다음과 같습니다:

Thought: 사용자 요청을 이해했습니다.
Action: PlaceSearch
Action Input: 서울 여행지
Observation: 서울의 인기 여행지 검색 완료
Thought: 이제 추천할 수 있습니다.
Final Answer: 서울의 추천 여행지는 경복궁, 남산타워, 한강공원입니다.
"""
)

agent= initialize_agent(
    tools= [
        place_search_tool
    ],
    llm= llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose= True,
    handle_parsing_errors= True,
    system_message= guardrail_message,
    memory= memory
)

def plan_travel(prompt: str) -> str:
    return agent.run(prompt)