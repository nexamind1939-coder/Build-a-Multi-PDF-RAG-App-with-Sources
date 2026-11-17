
import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# 1. Configuration
# --------------------------------

load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.write("Upload multiple PDFs and ask questions across all documents.")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please configure your .env file.")
    st.stop()


# --------------------------------
# 2. Session State
# --------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# --------------------------------
# 3. PDF Upload
# --------------------------------

uploaded_files = st.file_uploader(
    "Upload multiple PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------
# 4. Build Vector Database
# --------------------------------

def build_vector_database(files):

    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    for uploaded_file in files:

        st.write(f"Processing: {uploaded_file.name}")

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            # Load PDF pages
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            # Preserve original filename and page metadata
            for document in documents:
                document.metadata["source"] = uploaded_file.name

            # Split into chunks
            chunks = splitter.split_documents(documents)

            all_chunks.extend(chunks)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    # Create one vector database for every PDF
    embeddings = OpenAIEmbeddings(api_key=api_key)

    vector_store = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    return vector_store, len(all_chunks)


# Identify whether uploaded files have changed
if uploaded_files:

    signature_data = "".join(
        f"{f.name}:{hashlib.md5(f.getvalue()).hexdigest()}"
        for f in uploaded_files
    )

    current_signature = hashlib.md5(
        signature_data.encode()
    ).hexdigest()

    if current_signature != st.session_state.file_signature:

        st.session_state.store = None
        st.session_state.file_signature = current_signature
        st.session_state.indexed_files = [
            f.name for f in uploaded_files
        ]
        st.session_state.chunk_count = 0


# --------------------------------
# 5. Index Documents
# --------------------------------

if uploaded_files and st.session_state.store is None:

    if st.button("Build Knowledge Base", type="primary"):

        try:
            with st.spinner("Reading PDFs and creating embeddings..."):

                vector_store, total_chunks = build_vector_database(
                    uploaded_files
                )

                st.session_state.store = vector_store
                st.session_state.chunk_count = total_chunks

            st.success(
                f"Successfully indexed {total_chunks} chunks!"
            )

        except Exception as e:
            st.error(f"Error while indexing PDFs: {e}")


# --------------------------------
# 6. Display Index Information
# --------------------------------

if st.session_state.store is not None:

    st.success("Knowledge base is ready!")

    st.write("### Indexed Documents")

    for filename in st.session_state.indexed_files:
        st.write(f"📄 {filename}")

    st.metric(
        "Total chunks indexed",
        st.session_state.chunk_count
    )


# --------------------------------
# 7. Ask Questions
# --------------------------------

def ask_question(question):

    # Retrieve relevant chunks
    retriever = st.session_state.store.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        filename = doc.metadata.get("source", "Unknown file")

        # PyPDFLoader uses zero-based page numbers
        page_number = int(doc.metadata.get("page", 0)) + 1

        context_parts.append(
            f"""
            SOURCE {i}
            File: {filename}
            Page: {page_number}
            Content: {doc.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    # Prompt that handles conflicting information
    prompt = ChatPromptTemplate.from_template("""
You are a careful Multi-PDF RAG assistant.

Answer the question using ONLY the supplied document context.

Rules:
1. Do not invent facts.
2. If the documents do not contain the answer, say:
   "I don't know from the provided documents."
3. Compare information from different documents.
4. If two documents disagree, clearly explain the disagreement.
5. Never silently choose one conflicting answer.
6. Mention the relevant filenames and page numbers in your answer.
7. Keep the answer clear and concise.

Document Context:
{context}

User Question:
{question}

Answer:
""")

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=api_key
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, docs


# --------------------------------
# 8. Streamlit Question Interface
# --------------------------------

if st.session_state.store is not None:

    st.divider()

    st.subheader("💬 Ask your documents")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the technology stack?"
    )

    if st.button("Get Answer"):

        if question.strip():

            with st.spinner("Searching documents and generating answer..."):

                try:
                    answer, source_docs = ask_question(question)

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("📌 Sources")

                    seen_sources = set()

                    for doc in source_docs:

                        filename = doc.metadata.get(
                            "source", "Unknown file"
                        )

                        page = int(
                            doc.metadata.get("page", 0)
                        ) + 1

                        source_key = (filename, page)

                        if source_key not in seen_sources:

                            st.caption(
                                f"📄 {filename} — Page {page}"
                            )

                            seen_sources.add(source_key)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

        else:
            st.warning("Please enter a question.")
