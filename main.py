# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import os

# --- IMPORTANT: SET YOUR GOOGLE API KEY ---
# It's best practice to set this as an environment variable in your terminal.
# On Windows: $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
# On macOS/Linux: export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
# You will also need to run: pip install -U langchain-huggingface

# --- Part 1: Processing the PDF (Only needs to be run once) ---
def create_vector_db():
    if os.path.exists("faiss_index"):
        print("Vector DB index already exists. Skipping creation.")
        return
    print("Creating vector DB index...")
    loader = PyPDFLoader("data/project_plan.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    print("Vector DB index created and saved successfully.")

# --- Part 2: The Conversational Q&A Application (Modern LCEL approach) ---
def run_qa_app():
    # Load the saved vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Set up the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

    # 1. Contextualize Question Prompt: This prompt takes the chat history and a new question,
    # and creates a standalone question.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answering Prompt: This prompt takes the retrieved documents and the user's question
    # to generate a final answer.
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Combine the chains into a single RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 4. Set up message history management
    demo_ephemeral_chat_history = ChatMessageHistory()
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: demo_ephemeral_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Start the interactive loop
    print("\n✅ Your Conversational AI Assistant is ready!")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        
        result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "demo_session"}}
        )
        print(f"AI: {result['answer']}")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    
    create_vector_db()
    run_qa_app()
