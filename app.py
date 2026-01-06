import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from annoy import AnnoyIndex
from openai import OpenAI

# Load env (local only)
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("📘 AI Study Assistant – Final Version")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Sidebar – PDF Upload
# -------------------------
st.sidebar.header("📂 Upload Study PDF (Optional)")
pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

pdf_text = ""

if pdf_file:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        pdf_text += page.extract_text()

    st.sidebar.success("PDF loaded successfully ✅")

# -------------------------
# Create Embeddings
# -------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

annoy_index = None
chunks = []

if pdf_text:
    chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
    annoy_index = AnnoyIndex(1536, "angular")

    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        annoy_index.add_item(i, emb)

    annoy_index.build(10)

# -------------------------
# Main Question Input
# -------------------------
question = st.text_input("❓ Ask your question")

if st.button("Ask AI"):
    if not question:
        st.warning("Please enter a question")
    else:
        context = ""

        if annoy_index:
            q_emb = get_embedding(question)
            ids = annoy_index.get_nns_by_vector(q_emb, 3)
            context = " ".join([chunks[i] for i in ids])

        prompt = f"""
You are a helpful study assistant.

Context (if any):
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        st.session_state.history.append((question, answer))

        st.success("Answer generated ✅")
        st.write(answer)

# -------------------------
# History Section
# -------------------------
st.divider()
st.subheader("🕘 Question History")

for q, a in reversed(st.session_state.history):
    with st.expander(q):
        st.write(a)





