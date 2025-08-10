# 🤖 Personal Conversational AI Assistant
A powerful, context-aware AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on your own documents. This project is built with Python, LangChain, and Google's Gemini API, providing a solid foundation for a personal knowledge base you can chat with.

This assistant can remember the context of your conversation and can be easily adapted to use any PDF document as its knowledge source.

# ✨ Features
Conversational Memory: Remembers previous parts of the conversation to answer follow-up questions naturally.

Retrieval-Augmented Generation (RAG): Grounds its answers in a specific document, reducing hallucinations and providing accurate, context-aware responses.

Powered by Gemini 1.5 Flash: Leverages Google's fast and efficient language model for high-quality responses.

Customizable Knowledge Base: Easily swap out the included project_plan.pdf with any PDF to create an expert on any topic.

Open-Source & Extendable: Built with modern, modular code that's easy to understand and extend.

# 🚀 How It Works
This assistant uses a sophisticated, multi-step process to answer your questions:

Vector Database Creation: When first run, the script processes your PDF document. It splits the text into manageable chunks, converts them into numerical representations (embeddings) using a Hugging Face model, and stores them in a local FAISS vector database. This is a one-time setup.

Contextualizing the Question: When you ask a question, the AI first looks at the chat history to see if your new question is a follow-up. It then creates a "standalone question" that includes all the necessary context.

Retrieval: The standalone question is used to search the vector database for the most relevant text chunks from your document.

Generation: The original question, the chat history, and the retrieved document chunks are all passed to the Gemini LLM, which generates a final, coherent answer.

# 🛠️ Getting Started
Follow these steps to get the assistant running on your local machine.

Prerequisites
Python 3.9+

A Google Gemini API Key

1. Clone the Repository
git clone 
cd Personal_Assistant

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
Install all the required packages using the requirements.txt file.

pip install -r requirements.txt



4. Set Up Your API Key
Set your Google Gemini API key as an environment variable. This is the most secure way to handle your key.

# For macOS/Linux
export GOOGLE_API_KEY='YOUR_API_KEY_HERE'

# For Windows (PowerShell)
$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"

5. Run the Application
You're all set! Run the main script to start chatting with your assistant.

python main.py

# 🔌 Advanced: Swapping the AI Model
This project is built on LangChain, which makes it incredibly flexible. While it's configured to use Google's Gemini API by default, you can easily swap it out to use other services like OpenAI or even run it completely offline with a local model via Ollama.

Using OpenAI (GPT-4, etc.)
Install the package: pip install langchain-openai

Set your API Key: export OPENAI_API_KEY='YOUR_SK_KEY_HERE'

Update the code: In main.py, change the LLM initialization:

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Running Offline with Ollama
Install and run Ollama: Follow the instructions at ollama.com and pull a model (e.g., ollama run llama3).

Update the code: In main.py, change the LLM initialization:

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
llm = ChatOllama(model="llama3") # No API key needed!

# 📄 Customizing the Knowledge Base
To use your own PDF as the knowledge source, simply:

Place your PDF file inside the data/ directory.

Open main.py and change the filename in the create_vector_db function:

# Before
loader = PyPDFLoader("data/project_plan.pdf")

# After
loader = PyPDFLoader("data/your_new_file.pdf")

Delete the existing faiss_index folder.

Run python main.py. The script will automatically process your new PDF and create a new vector index for it.

# 🌐 Deploying as a Web Application
You can wrap this application in a web framework like FastAPI to create an API that can power a web frontend.

Create an API Endpoint: Use FastAPI to create an endpoint that takes a user's question.

Call the RAG Chain: Inside the endpoint, call your conversational_rag_chain.invoke() function.

Return the Answer: Return the AI's answer as a JSON response.

Connect a Frontend: Build a simple frontend using HTML, CSS, and JavaScript (or a framework like React/Vue) that makes API calls to your FastAPI backend.

📦 requirements.txt

langchain
langchain-community
langchain-core
langchain-google-genai
langchain-huggingface
faiss-cpu
sentence-transformers
pypdf

Users can then install all of these with a single command: pip install -r requirements.txt.

# 🤝 Contributing
Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.

Commit your changes (git commit -m 'Add some amazing feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
