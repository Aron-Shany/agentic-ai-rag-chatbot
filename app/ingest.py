"""One-time ingestion: PDF -> chunks -> embeddings -> Pinecone."""
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app import config


def ensure_index(pc: Pinecone) -> None:
    if config.INDEX_NAME in pc.list_indexes().names():
        print(f"Index '{config.INDEX_NAME}' already exists.")
        return
    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(config.INDEX_NAME).status["ready"]:
        time.sleep(1)
    print(f"Created index '{config.INDEX_NAME}'.")


def main() -> None:
    print("Loading PDF...")
    pages = PyPDFLoader(config.PDF_PATH).load()
    print(f"Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks.")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc)

    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=config.INDEX_NAME
    )
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
