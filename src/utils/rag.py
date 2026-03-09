from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from typing import List, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
llm = ChatGroq(model=os.getenv("LLM_MODEL_NAME"), temperature=0.9)


embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def load_data(path: str):
    loader = DirectoryLoader(
        f"../../../{UPLOAD_DIRECTORY}",
        glob="**/*",
        show_progress=True
    )

    data = loader.load()
    return data


def split_docs(data: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = splitter.split_documents(data)
    return chunks


def save_to_vectorDB(data: List[Document]):
    vector_store = Milvus.from_documents(
        documents=data,
        embedding=embeddings, 
        connection_args={
            "uri": "http://localhost:19530",
        },
        drop_old=True
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever


def get_relevent_docs(query: str, retriever: Any):
    relevent_docs = retriever.get_relevant_documents(query)
    return relevent_docs


def get_relevent_docs_with_score(query: str, retriever: Any):
    relevent_docs_with_score = retriever.get_relevant_documents_with_score(query)
    return relevent_docs_with_score


def generate_ans_using_retriever(query: str, retriever: Any, option: int = 1):
    relevent_docs = None
    if option == 1:
        relevent_docs = get_relevent_docs(query, retriever)
    elif option == 2:
        relevent_docs = get_relevent_docs_with_score(query, retriever)
    
    prompt = f"""You are a helpful assistant, that answer the question based on the given context only: 
    
    Question: {query}
    
    <Context>
    {relevent_docs}
    </Context>    

    """

    ans = llm(prompt)
    return ans