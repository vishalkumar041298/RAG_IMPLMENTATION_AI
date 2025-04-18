import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def load_document(file_path: str):
    """Loads a document from the given file path."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_document(document, chunk_size=1000, chunk_overlap=0):
    """Splits the document into chunks using CharacterTextSplitter."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(document)


def create_vector_store(documents, index_name):
    """Embeds the documents and stores them in Pinecone VectorStore."""
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)


def main():
    load_dotenv()
    file_path = '/docs/mediumblog1.txt'
    index_name = os.getenv('INDEX_NAME')

    if not index_name:
        raise ValueError("INDEX_NAME is not set in environment variables.")

    document = load_document(file_path)
    chunks = split_document(document)
    create_vector_store(chunks, index_name)
    print("Documents successfully loaded into Pinecone VectorStore.")


if __name__ == '__main__':
    main()
