import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import glob
import shutil
import tempfile

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from Streamlit secrets or environment variables
def get_openai_api_key():
    # First try to get from Streamlit secrets
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        # If not in Streamlit secrets, try environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.")
        return api_key

# Set OpenAI API key
api_key = get_openai_api_key()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("No OpenAI API key found. Please provide an API key to use this application.")
    st.stop()

# Constants
# Use a directory we can write to in Streamlit Cloud
# Using a subdirectory of tempfile.gettempdir() ensures we have write permissions
TEMP_DIR = os.path.join(tempfile.gettempdir(), "streamlit_rag")
os.makedirs(TEMP_DIR, exist_ok=True)
CHROMA_PATH = os.path.join(TEMP_DIR, "chroma")
DATA_PATH = "data/books"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = []

# Initialize the database and models
@st.cache_resource
def initialize_rag():
    # Use default initialization which will read from environment variables
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = ChatOpenAI()
    return db, model

# Function to check if Chroma DB exists and has data
def check_db_status():
    if not os.path.exists(CHROMA_PATH):
        return "Database not found. Please create embeddings first."
    
    try:
        # Use default initialization which will read from environment variables
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        collection = db._collection
        count = collection.count()
        if count == 0:
            return "Database exists but contains no documents. Please create embeddings first."
        return f"Database found with {count} embeddings."
    except Exception as e:
        return f"Error checking database: {str(e)}"

# Function to load documents
def load_documents():
    documents = []
    
    # Get all markdown files in the data directory
    md_files = glob.glob(f"{DATA_PATH}/*.md")
    
    for md_file in md_files:
        try:
            loader = TextLoader(md_file)
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {md_file}: {str(e)}")
    
    return documents

# Function to split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# Function to create embeddings
def create_embeddings():
    with st.spinner("Loading documents..."):
        documents = load_documents()
        
        if len(documents) == 0:
            st.error("No documents found in data directory. Please add markdown files to data/books/")
            return False
            
        st.info(f"Loaded {len(documents)} document(s)")
    
    with st.spinner("Splitting documents into chunks..."):
        chunks = split_text(documents)
        st.info(f"Split into {len(chunks)} chunks")
        
        # Show a sample chunk
        if len(chunks) > 0:
            with st.expander("Sample chunk"):
                st.write(chunks[0].page_content)
    
    with st.spinner("Creating embeddings... This may take a while."):
        # Remove old database if it exists
        try:
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            st.warning(f"Could not remove old database: {str(e)}. Will attempt to overwrite.")
            
        # Make sure the directory exists
        os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)
            
        try:
            # Use default initialization which will read from environment variables
            embedding_function = OpenAIEmbeddings()
            db = Chroma.from_documents(
                chunks, 
                embedding_function, 
                persist_directory=CHROMA_PATH
            )
            db.persist()
            st.success(f"Created embeddings for {len(chunks)} chunks and saved to disk.")
            return True
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False

# Function to get available documents
def get_available_docs():
    md_files = glob.glob(f"{DATA_PATH}/*.md")
    return [os.path.basename(f) for f in md_files]

# Function to get response
def get_response(query_text, db, model):
    # Lower the relevance threshold to 0.5 (from 0.7)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Debug info
    if len(results) == 0:
        st.session_state.debug_info = "No results found in the database."
        return None, None
    elif results[0][1] < 0.5:
        st.session_state.debug_info = f"Results found but relevance score too low: {results[0][1]}"
        # Continue anyway, just for debugging
    else:
        st.session_state.debug_info = f"Found {len(results)} results with top score: {results[0][1]}"
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = model.predict(prompt)
    
    # Store the actual text passages instead of just file paths
    source_texts = [doc.page_content for doc, _score in results]
    
    return response_text, source_texts

# Streamlit UI
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for dark theme readability
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #1E1E1E;
    }
    .chat-message.user {
        background-color: #E8F4FA;
        border-left: 5px solid #2980B9;
    }
    .chat-message.assistant {
        background-color: #F9F9F9;
        border-left: 5px solid #27AE60;
    }
    .chat-message .header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .chat-message .header .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 8px;
    }
    .chat-message .header .name {
        font-weight: bold;
    }
    .chat-message .content {
        margin-top: 0.5rem;
    }
    .sources {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
    .sources-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .source-text {
        margin-bottom: 1rem;
        white-space: pre-line;
    }
    .debug-info {
        background-color: #FFF3CD;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    .clear-button {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("💬 About this App")
    st.markdown("""
    This is a **Retrieval-Augmented Generation (RAG)** system that:
    
    1. Uses **Vector Search** to find relevant passages from documents
    2. Provides the passages as context to a **Large Language Model**
    3. Generates accurate answers based on the retrieved context
    
    ### Available Documents:
    """)
    
    docs = get_available_docs()
    if docs:
        for doc in docs:
            st.markdown(f"- {doc}")
    else:
        st.markdown("No documents found in data directory.")
    
    st.markdown("### Database Status:")
    db_status = check_db_status()
    st.markdown(db_status)
    
    if "Database not found" in db_status or "contains no documents" in db_status:
        if st.button("Create Embeddings", type="primary"):
            success = create_embeddings()
            if success:
                st.rerun()
    
    st.markdown("---")
    st.markdown("### Sample Questions:")
    
    # Sample questions - clicking these will set the query
    sample_questions = [
        "Who is Alice?",
        "What happens when Alice meets the Queen?",
        "What is the significance of the rabbit hole?",
        "Describe the Mad Hatter's tea party."
    ]
    
    for question in sample_questions:
        if st.button(question):
            st.session_state.current_question = question
            st.rerun()

# Main content
st.title("📚 Alice in Wonderland RAG")
st.markdown("""
    Ask questions about Alice in Wonderland. The system will retrieve relevant passages from the book and generate an answer.
""")

# Clear chat button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear Chat", key="clear", type="primary"):
        clear_chat_history()
        st.rerun()

# Check if database exists before initializing
try:
    db_status = check_db_status()
    if "Database not found" in db_status or "contains no documents" in db_status:
        st.warning(db_status)
        st.info("Please create embeddings using the button in the sidebar.")
        db, model = None, None
    else:
        db, model = initialize_rag()
except Exception as e:
    st.error(f"Error initializing the database: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    source_texts = message.get("source_texts", None)
    
    with st.container():
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="header">
                    <div class="avatar">👤</div>
                    <div class="name">You</div>
                </div>
                <div class="content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Prepare sources HTML inside the assistant message
            sources_html = ""
            if source_texts:
                sources_html = """<div class="sources">
                <div class="sources-title">Sources:</div>
                """
                for source_text in source_texts:
                    sources_html += f"""<div class="source-text">---\n\n{source_text}\n\n---</div>"""
                sources_html += "</div>"
            
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="header">
                    <div class="avatar">🤖</div>
                    <div class="name">Assistant</div>
                </div>
                <div class="content">{content}</div>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)

# Query input
query = st.text_input(
    "Enter your question:", 
    value=st.session_state.get("current_question", ""),
    placeholder="e.g., Who is Alice?",
    disabled=db is None
)

# Clear current_question after use
if "current_question" in st.session_state:
    del st.session_state.current_question

# Process query
if query and db is not None:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Searching for relevant information..."):
        response, source_texts = get_response(query, db, model)
        
        # Display debug info (can be removed in production)
        if "debug_info" in st.session_state:
            st.markdown(f"""
            <div class="debug-info">
                Debug: {st.session_state.debug_info}
            </div>
            """, unsafe_allow_html=True)
        
        if response:
            # Add assistant message to chat history with source texts
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "source_texts": source_texts
            })
        else:
            # Add a message explaining the issue
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I couldn't find relevant information for your query. This could be because the database hasn't been properly initialized."
            })
    
    # Rerun to display the updated chat history
    st.rerun() 