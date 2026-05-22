#  CHAT WITH MULTIPLE PDFs

A RAG (Retrieval-Augmented Generation) based chatbot that allows you to upload multiple PDF documents and ask questions about them in natural language.

##  LIVE DEMO
[Click here to Try the App](https://multi-pdf-chatbot-akilesh.streamlit.app/)

## TECH STACK USED

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| PDF Processing | pypdf |
| Framework | LangChain |
| Language | Python 3.11 |

## HOW IT WORKS?

1. **Upload PDFs** — Upload one or more PDF files from the sidebar
2. **Processing** — Text is extracted, split into chunks, and stored as vectors in FAISS
3. **Ask Questions** — Ask anything about your documents in the chat input
4. **Get Answers** — The app retrieves relevant chunks and generates answers using LLaMA 3.1

## ARCHITECTURE
PDF Upload → Text Extraction → Text Chunking → Vector Embeddings → FAISS Store
↓
User Question → Query Embedding → Similarity Search → Context Retrieval → LLM → Answer

## WHAT ARE ALL THE CHALLENGES FACES AND FIXED?

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

## COMPLETE INSTALLATION STEPS

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


## DEPLOYMENT STEPS, AS I USED STREAMLIT CLOUD

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

## FEATURES

-  Upload multiple PDFs at once
-  Chat with your documents naturally
-  Recent chat history in sidebar
-  Start new chat sessions
-  Fast responses using Groq
-  Secure API key handling
-  Friendly error for scanned/image PDFs

##  API Keys Required

- **Groq API Key** — Get it free from [console.groq.com](https://console.groq.com)

##  Models Used

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `llama-3.1-8b-instant` via Groq

##  Developed By

**Akilesh** — Shiv Nadar University, Chennai

##  License

MIT License
