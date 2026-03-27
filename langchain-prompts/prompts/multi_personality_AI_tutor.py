from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an AI tutor.

Teaching Style: {style}
Student Level: {level}

Instructions:
- Explain clearly
- Use real-world examples
- Adapt based on level
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

chain = prompt | llm

# UI
st.title("🤖 AI Tutor (ChatGPT Style)")

style = st.sidebar.selectbox("Style", ["Formal", "Casual", "Standard"])
level = st.sidebar.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])

# Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    response = chain.invoke({
        "input": user_input,
        "style": style,
        "level": level,
        "chat_history": st.session_state.chat_history
    })

    # Show AI response
    with st.chat_message("assistant"):
        st.write(response.content)

    # Save memory
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response.content))

# Clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []