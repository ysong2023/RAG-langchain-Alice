# RAG-LangChain Project

A Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and OpenAI.

## Features

- Create embeddings from text documents using OpenAI
- Store and retrieve embeddings using ChromaDB vector database
- Query the database using natural language questions
- Get context-aware responses using OpenAI's LLM

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_goes_here
   ```

## Usage

### Creating the Vector Database

Run the following command to process documents in the `data/books` directory and create embeddings:

```
python create_database.py
```

This will:
- Load the document(s) in the data/books directory
- Split them into smaller chunks
- Create embeddings using OpenAI
- Store the embeddings in a ChromaDB database

### Querying the Database

Ask questions about the documents using:

```
python query_data.py "your question here"
```

Example:
```
python query_data.py "Who is Alice?"
```

The system will:
1. Find the most relevant passages from the documents
2. Use them as context to generate an accurate answer
3. Display the response along with the source documents

## Customization

- Modify `CHROMA_PATH` in the scripts to change the database location
- Adjust `DATA_PATH` to point to different document directories
- Change the `chunk_size` and `chunk_overlap` in `split_text()` for different segmentation

## Troubleshooting

If you encounter errors:
- Ensure your OpenAI API key is valid and properly set in the .env file
- Check that the data directory contains valid markdown files
- Verify that all dependencies are installed correctly
