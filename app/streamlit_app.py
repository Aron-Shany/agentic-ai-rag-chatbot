"""Optional Streamlit UI. Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.rag_graph import rag_graph

st.set_page_config(page_title="Agentic AI eBook Chatbot", layout="wide")
st.title("Agentic AI eBook Chatbot")

question = st.chat_input("Ask a question about the eBook")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Thinking..."):
        result = rag_graph.invoke({"question": question})
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.caption(f"Confidence: {result['confidence']}")
    with st.sidebar:
        st.header("Retrieved context")
        for i, c in enumerate(result["context_chunks"], 1):
            st.markdown(f"**Chunk {i}** (page {c['page']}, score {c['score']})")
            st.write(c["text"])
