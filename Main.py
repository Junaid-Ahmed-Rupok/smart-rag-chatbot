import streamlit as st
from Config import APP_NAME, APP_ICON
from design.components import apply_custom_style

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")
apply_custom_style()

st.title(f"{APP_ICON} {APP_NAME}")
st.caption("Professional RAG Chatbot with Design System")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = "Ready! Add RAG logic in Rag.py"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
