import streamlit as st
import tempfile
import os
import uuid
from rag import app_graph
from ingest import load_text, process_and_store, get_collection, reset_collection

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stChatMessage { border-radius: 12px; padding: 6px; }
    h1 {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stCaption { opacity: 0.7; }
</style>
""", unsafe_allow_html=True)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
collection_name = f"session_{st.session_state.session_id}"


with st.sidebar:
    st.header("About this project")
    st.markdown(
        "A **Retrieval-Augmented Generation** chatbot that answers "
        "questions grounded in documents you provide, orchestrated with **LangGraph**."
    )
    st.divider()
    st.markdown("**Stack**")
    st.markdown("- `Chroma` — vector store\n- `sentence-transformers` — embeddings\n- `Claude` — generation\n- `LangGraph` — orchestration\n- `Streamlit` — UI")
    st.divider()

    st.subheader("Add a document")
    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
    if uploaded_file and st.button("Ingest file", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            text = load_text(tmp_path)
            collection = get_collection(collection_name)
            success = process_and_store(text, source_name=uploaded_file.name, collection=collection)
            os.unlink(tmp_path)
        if success:
            st.success("Added!")
        else:
            st.error("Couldn't extract text from that file.")

    st.divider()
    if st.button("🗑️ Clear documents", use_container_width=True):
        reset_collection(collection_name)
        st.success("Knowledge base cleared.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Document RAG Chatbot")
st.caption("Ask questions about documents you've added — your uploads are private to this session and answers are grounded only in retrieved context.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources used"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**[{i}]** {doc[:300]}...")


if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents..."):
            result = app_graph.invoke({
                "question": question,
                "collection_name": collection_name,
                "retry_count": 0,
            })

        answer = result["answer"]
        sources = result["retrieved_docs"]

        st.markdown(answer)
        with st.expander("📄 Sources used"):
            for i, doc in enumerate(sources):
                st.markdown(f"**[{i}]** {doc[:300]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


st.divider()
st.caption("Powered by Claude · LangGraph · Chroma")