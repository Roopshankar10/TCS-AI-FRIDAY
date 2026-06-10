# Chatbot AI - Deepseek

import streamlit as st
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Setup Page Configuration
st.set_page_config(page_title="Hackathon AI Assistant", page_icon="🤖")
st.title("🤖 11:59 Squad - Team 01")
st.caption("Powered by DeepSeek-V3 via TCS GenAI Lab")

# 2. Initialize the Model (Using their exact sample code configuration)
@st.cache_resource
def load_llm():
    # Disables SSL verification just like client = httpx.Client(verify=False)
    http_client = httpx.Client(verify=False) 
    
    return ChatOpenAI(
        base_url="https://genailab.tcs.in/v1",  # Added /v1 which LangChain appends under the hood
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="sk-0qxBx5yz7XnVigOpOVHNCQ",
        http_client=http_client
    )

try:
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")

# 3. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a specialized AI assistant designed to solve the provided hackathon problem statement."}
    ]

# 4. Display past chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# 5. Handle User Input
if user_input := st.chat_input("Ask the AI solution a question..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

# 6. Call the LangChain Model with Full Memory
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        try:
            # Construct the full message history array for LangChain
            formatted_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    formatted_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AIMessage(content=msg["content"]))
                
            # PASS THE ENTIRE HISTORY, NOT JUST USER_INPUT
            response = llm.invoke(formatted_messages)
                
            # Extract the text content from LangChain's AIMessage object
            ai_response = response.content
                
            st.write(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
        except Exception as e:
            st.error(f"Error communicating with DeepSeek-V3: {e}")
