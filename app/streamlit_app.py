# app/streamlit_app.py
# Simple UI for the Company Knowledge Assistant.
# This talks to the FastAPI backend (app/api.py) over HTTP - it doesn't
# touch the RAG/ingestion logic directly.

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Company Knowledge Assistant")
st.title("Company Knowledge Assistant")

# ---- Ask a question ----
st.header("Ask a question")
question = st.text_input("What do you want to know?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        else:
            st.write(data["answer"])
            if data["sources"]:
                st.caption("Sources: " + ", ".join(data["sources"]))

st.divider()

# ---- Load documents into the knowledge base ----
st.header("Load documents into the knowledge base")

if st.button("Start ingestion"):
    try:
        response = requests.post(f"{API_URL}/ingest", timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Could not start ingestion: {e}")
    else:
        st.success("Ingestion started")

try:
    status = requests.get(f"{API_URL}/ingest/status", timeout=10).json()
except Exception as e:
    st.error(f"Could not reach the API: {e}")
else:
    st.write(f"Status: **{status['status']}**")
    if status["stats"]:
        st.write(status["stats"])
    if status["error"]:
        st.error(status["error"])
