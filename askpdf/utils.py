import streamlit as st
import os
from typing import List

def set_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key

def create_sidebar():
    with st.sidebar:
        st.markdown(
            """
            ## How to use\n
            1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below\n
            2. Upload a pdf file\n
            3. Ask any question\n"""
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="OpenAI API key (sk-###)",
            help="API key can be retrieved from https://platform.openai.com/account/api-keys.",
        )

        if api_key_input:
            set_api_key(api_key_input)
    return


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def is_valid(index, query):
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please configure your OpenAI API key!")
        return False
    if not index:
        st.error("Please upload a document!")
        return False
    if not query:
        st.error("Please enter a question!")
        return False
    return True
