import os
import shutil

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    documents = []

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    files = os.listdir(DATA_PATH)

    for filename in files:

        filepath = os.path.join(DATA_PATH, filename)

        if filename.lower().endswith(".pdf"):

            print(f"Loading PDF: {filename}")

            loader = PyPDFLoader(filepath)
            docs = loader.load()

        elif filename.lower().endswith(".docx"):

            print(f"Loading DOCX: {filename}")

            loader = Docx2txtLoader(filepath)
            docs = loader.load()

        else:
            continue

        for doc in docs:
            doc.metadata["source_file"] = filename

        documents.extend(docs)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks


def create_vector_database(chunks):

    # Remove old database if it exists
    if os.path.exists(CHROMA_PATH):

        print("Removing old Chroma database...")

        try:
            shutil.rmtree(CHROMA_PATH)

        except PermissionError:

            print("\nERROR: ChromaDB is currently being used.")
            print("Please stop Streamlit/Ollama-related Python processes")
            print("and run ingestion again.")

            return False

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print("Creating ChromaDB...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("\nVector database created successfully!")

    return True


def ingest():

    print("\n================================")
    print("      SAMACHEER RAG INGESTION")
    print("================================\n")

    documents = load_documents()

    if not documents:

        print("No PDF or DOCX files found.")
        print("Put your textbooks inside the data folder.")

        return

    print(f"\nLoaded {len(documents)} document pages.")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings and vector database...")

    success = create_vector_database(chunks)

    if success:
        print("\n================================")
        print("       INGESTION COMPLETE")
        print("================================\n")


if __name__ == "__main__":
    ingest()