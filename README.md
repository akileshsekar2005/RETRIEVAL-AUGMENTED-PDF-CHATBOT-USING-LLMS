# 📚 Chat with Multiple PDFs

A RAG (Retrieval-Augmented Generation) based chatbot that allows you to upload multiple PDF documents and ask questions about them in natural language.

## 🚀 Live Demo
[Click here to try the app](https://pdf-chat-akilesh.streamlit.app)

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| PDF Processing | pypdf |
| Framework | LangChain |
| Language | Python 3.11 |

## ⚙️ How It Works

1. **Upload PDFs** — Upload one or more PDF files from the sidebar
2. **Processing** — Text is extracted, split into chunks, and stored as vectors in FAISS
3. **Ask Questions** — Ask anything about your documents in the chat input
4. **Get Answers** — The app retrieves relevant chunks and generates answers using LLaMA 3.1

## 🏗️ Architecture
PDF Upload → Text Extraction → Text Chunking → Vector Embeddings → FAISS Store
↓
User Question → Query Embedding → Similarity Search → Context Retrieval → LLM → Answer

## 🐛 Challenges Faced & Fixes

### 1. Python Version Conflict
- **Problem:** Project was running on Python 3.14 which caused pydantic v1 compatibility issues with LangChain
- **Fix:** Downgraded to Python 3.11 and created a fresh virtual environment using `py -3.11 -m venv venv`

### 2. LangChain Memory Import Error
- **Problem:** `from langchain.memory import ConversationBufferMemory` failed due to pydantic conflicts
- **Fix:** Rewrote using modern LangChain LCEL approach with `RunnableWithMessageHistory` and `InMemoryChatMessageHistory`

### 3. LangChain Chains Import Error
- **Problem:** `from langchain.chains import create_retrieval_chain` failed in newer LangChain versions
- **Fix:** Rewrote the chain manually using `RunnablePassthrough` and `StrOutputParser` from `langchain_core`

### 4. OpenAI Quota Error
- **Problem:** `openai.RateLimitError: insufficient_quota` — OpenAI API key had no credits
- **Fix:** Switched to Groq (free, fast) using `langchain_groq` with `llama-3.1-8b-instant`

### 5. Groq Model Decommissioned
- **Problem:** `llama3-8b-8192` model was decommissioned
- **Fix:** Updated to `llama-3.1-8b-instant`

### 6. PDF Text Extraction Issue
- **Problem:** PDFs exported from tools like Gamma had garbled/encoded text when extracted with PyPDF2
- **Fix:** Switched from `PyPDF2` to `pypdf` which handles font-encoded PDFs better

### 7. Empty PDF Error
- **Problem:** `IndexError: list index out of range` when uploading image-based or scanned PDFs
- **Fix:** Added a guard in `get_vectorstore()` to raise a friendly error if no text is extracted

## 📦 Complete Installation Steps

### Step 1 — Install Python 3.11
- Download Python 3.11 from [python.org](https://python.org)
- Install with default settings

### Step 2 — Create Project Folder
```bash
mkdir "MULTIPLE PDF CHATBOT"
cd "MULTIPLE PDF CHATBOT"
```

### Step 3 — Create Virtual Environment with Python 3.11
```bash
py -3.11 -m venv venv
```

### Step 4 — Activate Virtual Environment
```bash
# Windows PowerShell
venv\Scripts\activate
```

You should see `(venv)` in your terminal.

### Step 5 — Confirm Python Version
```bash
python --version
# Should show Python 3.11.x
```

### Step 6 — Install Dependencies
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu sentence-transformers streamlit pypdf python-dotenv openai langchain-groq
```

### Step 7 — Create `.env` File
Create a file named `.env` in your project folder:
GROQ_API_KEY=your_groq_api_key_here

Get your free Groq API key from [console.groq.com](https://console.groq.com)

### Step 8 — Create `app.py`
Create `app.py` with the full application code (see below).

### Step 9 — Run the App
```bash
streamlit run app.py
```

## 💻 Full Application Code

```python
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text could be extracted from the PDF. It may be a scanned or image-based PDF.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def build_rag_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question using ONLY "
         "the context below. If the answer isn't in the context, say so.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Analyzer", page_icon=":books:")

    for key, default in [("retriever", None), ("chain", None), ("chat_history", [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.header("Chat with Multiple PDFs :books:")

    with st.sidebar:
        st.subheader("🕘 Recent Chats")

        if "all_sessions" not in st.session_state:
            st.session_state.all_sessions = []

        if st.button("+ New Chat"):
            if st.session_state.chat_history:
                existing_titles = [s["title"] for s in st.session_state.all_sessions]
                new_title = st.session_state.chat_history[0]["content"][:40] + "..."
                if new_title not in existing_titles:
                    st.session_state.all_sessions.append({
                        "title": new_title,
                        "history": st.session_state.chat_history.copy()
                    })
            st.session_state.chat_history = []
            st.session_state.chain = None
            st.rerun()

        recent = list(reversed(st.session_state.all_sessions))[:3]
        for i, session in enumerate(recent):
            if st.button(f"💬 {session['title']}", key=f"session_{i}"):
                st.session_state.chat_history = session["history"]
                st.rerun()

        st.divider()

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Extracting text…"):
                    raw_text = get_pdf_text(pdf_docs)

                with st.spinner("Chunking text…"):
                    chunks = get_text_chunks(raw_text)
                    st.success(f"{len(chunks)} chunks created.")

                with st.spinner("Building vector store…"):
                    try:
                        vs = get_vectorstore(chunks)
                        retriever, chain = build_rag_chain(vs)
                        st.session_state.retriever = retriever
                        st.session_state.chain = chain
                        st.session_state.chat_history = []
                        st.success("Ready! Ask your questions.")
                    except ValueError as e:
                        st.error(str(e))

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    st.markdown("**Ask a question about your documents:**")
    user_question = st.chat_input("Ask a question about your documents…")

    if user_question:
        if st.session_state.chain is None:
            st.warning("Please upload and process PDFs first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    history = [
                        HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"])
                        for m in st.session_state.chat_history[:-1]
                    ]
                    answer = st.session_state.chain.invoke({
                        "input": user_question,
                        "chat_history": history,
                    })
                    st.write(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
```

## 🚀 Deployment Steps (Streamlit Cloud)

### Step 1 — Install Git
- Download from [git-scm.com](https://git-scm.com)
- Add to PATH: `C:\Program Files\Git\cmd`

### Step 2 — Create requirements.txt
```bash
pip freeze > requirements.txt
```

### Step 3 — Create .gitignore
Create a file named `.gitignore`:
.env
venv/
pycache/

### Step 4 — Create GitHub Repository
- Go to [github.com](https://github.com)
- Click **"+"** → **"New repository"**
- Name it `pdf-chatbot`
- Keep it **Public**
- Click **"Create repository"**

### Step 5 — Configure Git
```bash
git config --global user.email "youremail@gmail.com"
git config --global user.name "YourName"
```

### Step 6 — Push to GitHub
```bash
git init
git add app.py requirements.txt .gitignore README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/yourusername/pdf-chatbot.git
git push -u origin main
```

### Step 7 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"** → **"Deploy a public app from GitHub"**
4. Fill in:
   - Repository: `yourusername/pdf-chatbot`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: `3.11`
5. Click **"Advanced settings"** → add secret:
GROQ_API_KEY = "your_groq_api_key_here"
6. Click **"Deploy"**

Your app will be live at:
https://yourappname.streamlit.app

## ✨ Features

- 📄 Upload multiple PDFs at once
- 💬 Chat with your documents naturally
- 🕘 Recent chat history in sidebar
- 🆕 Start new chat sessions
- ⚡ Fast responses using Groq
- 🔒 Secure API key handling
- ⚠️ Friendly error for scanned/image PDFs

## 🔑 API Keys Required

- **Groq API Key** — Get it free from [console.groq.com](https://console.groq.com)

## 📊 Models Used

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `llama-3.1-8b-instant` via Groq

## 👨‍💻 Developed By

**Akilesh** — St. Joseph's College of Engineering, Chennai

## 📄 License

MIT License