import os

import streamlit as st
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

_ = load_dotenv(override=True)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# setup langchain tracing
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "edu-track-qa-assistant"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# hugging face init
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# streamlit init
st.title("EduTrack QA Assistant")
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = {}

# Groq init
groq_api_key = st.text_input("Groq api key", type="password", key="groq_api_key")

if groq_api_key:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key
    )

    # chat history
    session_id = st.text_input("Session Id", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Initialize ui_messages for this session if not exists
    if session_id not in st.session_state.ui_messages:
        st.session_state.ui_messages[session_id] = []

    # Ingestion pdf upload
    uploaded_files = st.file_uploader("Choose the pdf files", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(file.getvalue())
                file_name = file.name
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Retriever for contextualizing question
        contextualize_question_system_prompt = (
            "Given a chat history and the latest user question,"
            "which might reference the context in the chat history"
            "formulate a question that is more specific and relevant to the chat history."
            "reformulate if needed otherwise return the same question."
        )

        contextualize_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", contextualize_question_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        chat_history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_prompt
        )

        # Rag Chain for the answers
        system_prompt = (
            "You are an assistant that answers questions based on the context"
            "Use the following retrieved context to answer the questions"
            "Give concise answers in 50 words or less"
            "\n\n"
            "{context}"
        )

        answer_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)
        rag_chain = create_retrieval_chain(
            retriever=chat_history_aware_retriever,
            combine_docs_chain=qa_chain
        )


        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = InMemoryChatMessageHistory()
            return st.session_state.store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        ## Reload of app refresh/rerun - display messages for current session
        for message in st.session_state.ui_messages[session_id]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask your question"):
            # Append user message first to current session
            st.session_state.ui_messages[session_id].append({
                "role": "user",
                "content": user_input
            })

            # display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            response = conversational_rag_chain.invoke(
                input={"input": user_input},
                config=RunnableConfig(
                    configurable={
                        "session_id": session_id
                    }
                )
            )

            # Append assistant message to current session
            st.session_state.ui_messages[session_id].append({
                "role": "assistant",
                "content": response["answer"]
            })

            # display assistant message
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

else:
    st.error("Please provide a Groq api key")
