"""
Pinot Confluence Chatbot
This script demonstrates how to:
1. Load content from Apache Pinot Confluence pages using different APIs
2. Index the content in Pinecone
3. Build a chatbot that can answer questions based on the indexed content
"""

import os
import re
import requests
import json
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import ConfluenceLoader
import pinecone
import tempfile
from urllib.parse import urljoin, urlparse
from atlassian import Confluence

# Load environment variables
load_dotenv('conf/.env', override=True)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', 'gcp-starter')
CONFLUENCE_USERNAME = os.environ.get('CONFLUENCE_USERNAME', '')
CONFLUENCE_PASSWORD = os.environ.get('CONFLUENCE_PASSWORD', '')
CONFLUENCE_URL = os.environ.get('CONFLUENCE_URL', 'https://cwiki.apache.org/confluence')
CONFLUENCE_SPACE_KEY = os.environ.get('CONFLUENCE_SPACE_KEY', 'PINOT')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# ============================================================================
# CONFLUENCE CONTENT EXTRACTION - PYTHON CONFLUENCE API
# ============================================================================

def extract_confluence_content_python_api(space_key=CONFLUENCE_SPACE_KEY, limit=100):
    """
    Extracts content from a Confluence space using the Python Confluence API.
    
    Args:
        space_key (str): Key of the Confluence space
        limit (int): Maximum number of pages to retrieve
        
    Returns:
        str: Extracted text content
    """
    print(f"Fetching content from Confluence space '{space_key}' using Python API...")
    
    # Initialize the Confluence client
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        password=CONFLUENCE_PASSWORD,
        verify_ssl=False
    )
    
    # Get all pages in the space
    pages = confluence.get_all_pages_from_space(space_key, start=0, limit=limit)
    
    all_content = ""
    
    # Process each page
    for page in pages:
        page_id = page['id']
        page_title = page['title']
        page_url = f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={page_id}"
        
        print(f"Processing page: {page_title} (ID: {page_id})")
        
        # Get the page content
        page_content = confluence.get_page_by_id(page_id, expand='body.storage')
        
        # Extract the text content
        if 'body' in page_content and 'storage' in page_content['body']:
            html_content = page_content['body']['storage']['value']
            
            # Parse HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Clean up the text
            text_content = re.sub(r'\n{3,}', '\n\n', text_content)
            
            # Add page metadata
            page_content = f"TITLE: {page_title}\nURL: {page_url}\n\n{text_content}"
            
            all_content += page_content + "\n\n"
            print(f"Extracted {len(text_content)} characters of text from {page_title}")
        else:
            print(f"No content found for page {page_title}")
    
    print(f"Total content extracted: {len(all_content)} characters")
    return all_content

# ============================================================================
# CONFLUENCE CONTENT EXTRACTION - LANGCHAIN CONFLUENCE API
# ============================================================================

def extract_confluence_content_langchain_api(space_key=CONFLUENCE_SPACE_KEY, limit=100):
    """
    Extracts content from a Confluence space using the Langchain Confluence API.
    
    Args:
        space_key (str): Key of the Confluence space
        limit (int): Maximum number of pages to retrieve
        
    Returns:
        list: List of documents
    """
    print(f"Fetching content from Confluence space '{space_key}' using Langchain API...")
    
    # Initialize the Confluence loader
    loader = ConfluenceLoader(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        api_key=CONFLUENCE_PASSWORD,
        space_key=space_key,
        limit=limit
    )
    
    # Load the documents
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents from Confluence")
    return documents

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_text_documents(text):
    """
    Processes the extracted text into documents for indexing.
    
    Args:
        text (str): Text content
        
    Returns:
        list: List of document chunks
    """
    # Save the content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp_file:
        temp_file.write(text)
        temp_file_path = temp_file.name
    
    # Load the document
    loader = TextLoader(temp_file_path)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
    
    print(f"Split content into {len(chunks)} chunks")
    return chunks

def process_langchain_documents(documents):
    """
    Processes the Langchain documents into chunks for indexing.
    
    Args:
        documents (list): List of Langchain documents
        
    Returns:
        list: List of document chunks
    """
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split content into {len(chunks)} chunks")
    return chunks

# ============================================================================
# VECTOR STORE
# ============================================================================

def create_pinecone_index(chunks, index_name="pinot-confluence"):
    """
    Creates a Pinecone index from document chunks.
    
    Args:
        chunks (list): List of document chunks
        index_name (str): Name of the Pinecone index
        
    Returns:
        Pinecone: Pinecone vector store
    """
    # Check if the index already exists
    if index_name in pinecone.list_indexes():
        print(f"Index '{index_name}' already exists. Deleting it...")
        pinecone.delete_index(index_name)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create the Pinecone index
    print(f"Creating Pinecone index '{index_name}'...")
    vectorstore = Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"Index created with {len(chunks)} vectors")
    return vectorstore

# ============================================================================
# CHATBOT
# ============================================================================

def create_chatbot(vectorstore):
    """
    Creates a chatbot using the vector store.
    
    Args:
        vectorstore (Pinecone): Pinecone vector store
        
    Returns:
        RetrievalQA: Question answering chain
    """
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create a retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    
    return qa_chain

def answer_question(qa_chain, question):
    """
    Answers a question using the QA chain.
    
    Args:
        qa_chain (RetrievalQA): Question answering chain
        question (str): Question to answer
        
    Returns:
        dict: Answer and source documents
    """
    result = qa_chain({"query": question})
    return result

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the Pinot Confluence chatbot.
    """
    # Choose the method to extract content
    method = input("Choose the method to extract content (1: Python API, 2: Langchain API): ")
    
    if method == "1":
        # Extract content using Python Confluence API
        content = extract_confluence_content_python_api()
        
        # Save the content to a file for reference
        with open("pinot_confluence_content.txt", "w", encoding="utf-8") as f:
            f.write(content)
        
        # Process the content into documents
        chunks = process_text_documents(content)
    else:
        # Extract content using Langchain Confluence API
        documents = extract_confluence_content_langchain_api()
        
        # Process the documents into chunks
        chunks = process_langchain_documents(documents)
    
    # Create a Pinecone index
    vectorstore = create_pinecone_index(chunks)
    
    # Create a chatbot
    chatbot = create_chatbot(vectorstore)
    
    # Interactive question answering
    print("\n" + "="*50)
    print("Pinot Confluence Chatbot")
    print("Type 'exit' to quit")
    print("="*50 + "\n")
    
    while True:
        question = input("\nAsk a question about Apache Pinot: ")
        if question.lower() == 'exit':
            break
        
        try:
            result = answer_question(chatbot, question)
            print("\nAnswer:")
            print(result["result"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"]):
                print(f"{i+1}. {doc.page_content[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 