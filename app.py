import os
from dotenv import load_dotenv

import streamlit as st

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from state import PlannerState
from flow import build_flexible_planner_graph

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    model_name= "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature= 0.7,
    streaming= True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history= []
if "planner_state" not in st.session_state:
    st.session_state.planner_state= PlannerState()
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage= "initial"

st.title= ("여행 계획 멀티 에이전트")
user_prompt= st.text_input("메시지를 입력하세요", "")

def stream_llm_response(prompt: str):
    messages= [ HumanMessage(content=prompt) ]
    for chunk in llm.stream(messages):
        if hasattr(chunk, "content"):
            yield chunk.content or ""
        else:
            yield ""

if st.button("전송") and user_prompt:
    st.session_state.planner_state.user_input= user_prompt
    
    graph= build_flexible_planner_graph()
    res= graph.invoke(st.session_state.planner_state)
    st.write_stream( stream_llm_response(user_prompt) )
    
    if res.last_plan_summary:
        st.write("### 여행 계획서")
        st.write(res.last_plan_summary)