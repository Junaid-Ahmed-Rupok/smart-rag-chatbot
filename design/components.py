import streamlit as st
from design.theme import COLORS

def apply_custom_style():
    st.markdown(f"""
        <style>
        .stApp {{ background: {COLORS['background']}; }}
        .stButton > button {{
            background: {COLORS['primary']};
            color: white;
            border-radius: 8px;
        }}
        </style>
    """, unsafe_allow_html=True)
