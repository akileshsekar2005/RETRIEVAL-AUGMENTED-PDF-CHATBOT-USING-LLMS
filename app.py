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
                text += page.extract_text() or ""
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

    st.header("CHAT WITH MULTIPLE PDFs :books:")

    with st.sidebar:
        st.subheader("Recent Chats")
    
        if "all_sessions" not in st.session_state:
            st.session_state.all_sessions = []

        if st.button("+ New Chat"):
            if st.session_state.chat_history:
                existing_titles = [s["title"] for s in st.session_state.all_sessions]
                new_title = st.session_state.chat_history[0]["content"][:40] + "..."
                if new_title not in existing_titles:
                    st.session_state.all_sessions.append({
                        "title": st.session_state.chat_history[0]["content"][:40] + "...",
                        "history": st.session_state.chat_history.copy()
                    })
            st.session_state.chat_history = []
            st.session_state.chain = None
            st.rerun()

        recent = list(reversed(st.session_state.all_sessions))[:3]
        for i, session in enumerate(reversed(st.session_state.all_sessions)):
            if st.button(f" {session['title']}", key=f"session_{i}"):
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