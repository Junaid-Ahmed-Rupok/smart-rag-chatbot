"""Smart RAG Chatbot: entry point. Run with:  streamlit run Main.py"""
import streamlit as st

from smartrag import embeddings
from smartrag.settings import APP_ICON, APP_NAME
from smartrag.ui import chat, sidebar, state, styles

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

styles.inject()
state.init()
embeddings.warm_up(state.settings().embedding_model)   # load the search model in the background, once per server
# The chat is drawn first so the sidebar (Clear chat, Download chat) always reflects the latest turn.
chat.render()
sidebar.render()
