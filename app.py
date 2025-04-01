import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import os
import requests
import json
import glob
import pickle
import tempfile
import numpy as np

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
if not api_key:
    st.error("No OpenAI API key found. Please provide an API key to use this application.")
    st.stop()

# Create a custom embeddings class that uses direct API requests
class DirectOpenAIEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using direct API calls."""
        embeddings = []
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {
                "model": "text-embedding-ada-002",
                "input": batch
            }
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response_data = response.json()
            
            if "error" in response_data:
                raise ValueError(f"API Error: {response_data['error']['message']}")
                
            embeddings.extend([item["embedding"] for item in response_data["data"]])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using direct API calls."""
        payload = {
            "model": "text-embedding-ada-002",
            "input": [text]
        }
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        response_data = response.json()
        
        if "error" in response_data:
            raise ValueError(f"API Error: {response_data['error']['message']}")
            
        return response_data["data"][0]["embedding"]

# A simple custom vector store that doesn't depend on scikit-learn
class SimpleVectorStore:
    def __init__(self, documents=None, embeddings=None, embedding_function=None):
        self.documents = documents or []
        self.embedding_vectors = embeddings or []
        self.embedding_function = embedding_function
        self.docstore = {"_dict": {}} if documents else {"_dict": {}}
    
    @classmethod
    def from_documents(cls, documents, embedding):
        """Create a vector store from documents."""
        texts = [doc.page_content for doc in documents]
        embeddings = embedding.embed_documents(texts)
        
        # Initialize docstore
        docstore = {"_dict": {}}
        for i, doc in enumerate(documents):
            docstore["_dict"][i] = doc
        
        return cls(documents=documents, embeddings=embeddings, embedding_function=embedding, docstore=docstore)
    
    def similarity_search_with_score(self, query, k=4):
        """Search for documents similar to query."""
        query_embedding = np.array(self.embedding_function.embed_query(query))
        
        results = []
        for i, doc_embedding in enumerate(self.embedding_vectors):
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, np.array(doc_embedding))
            results.append((self.documents[i], similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Constants
# Use a directory we can write to in Streamlit Cloud
# Using a subdirectory of tempfile.gettempdir() ensures we have write permissions
TEMP_DIR = os.path.join(tempfile.gettempdir(), "streamlit_rag")
os.makedirs(TEMP_DIR, exist_ok=True)
VECTOR_STORE_PATH = os.path.join(TEMP_DIR, "simple_vector_store.pkl")
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
    # Use our direct API call embeddings class
    embedding_function = DirectOpenAIEmbeddings(api_key=api_key)
    
    # Load the vector store if it exists
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            db = pickle.load(f)
    else:
        st.error("Vector database not found. Please create embeddings first.")
        return None, None
        
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    return db, model

# Function to check if vector store exists
def check_db_status():
    if not os.path.exists(VECTOR_STORE_PATH):
        return "Database not found. Please create embeddings first."
    
    try:
        with open(VECTOR_STORE_PATH, "rb") as f:
            db = pickle.load(f)
            
        # Get approximate document count
        count = len(db.documents) if hasattr(db, 'documents') else 0
            
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
        try:
            # Use our direct API call embeddings class
            embedding_function = DirectOpenAIEmbeddings(api_key=api_key)
            
            # Create our simple vector store
            texts = [doc.page_content for doc in chunks]
            embeddings = embedding_function.embed_documents(texts)
            
            db = SimpleVectorStore(
                documents=chunks,
                embeddings=embeddings,
                embedding_function=embedding_function
            )
            
            # Save the vector store
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(db, f)
                
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
    # Get similar documents with relevance scores
    results = db.similarity_search_with_score(query_text, k=3)
    
    # Debug info
    if len(results) == 0:
        st.session_state.debug_info = "No results found in the database."
        return None, None
    else:
        # Make sure scores are in a reasonable range
        top_score = results[0][1]
        if top_score < 0.5:  # May need adjustment based on how scores work
            st.session_state.debug_info = f"Results found but relevance score too low: {top_score:.3f}"
        else:
            st.session_state.debug_info = f"Found {len(results)} results with top score: {top_score:.3f}"
    
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