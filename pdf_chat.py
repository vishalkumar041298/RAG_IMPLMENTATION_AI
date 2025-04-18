import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


def load_pdf_documents(pdf_path: str):
    """Loads and splits a PDF into chunks."""
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    return text_splitter.split_documents(documents)


def create_faiss_vectorstore(docs, embeddings, index_dir: str):
    """Creates a FAISS vectorstore from documents and saves locally."""
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_dir)


def load_faiss_vectorstore(index_dir: str, embeddings):
    """Loads an existing FAISS vectorstore from local storage."""
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def create_react_retrieval_chain(vectorstore, model_name="gpt-4o-mini"):
    """Creates a retrieval chain using the ReAct prompt from LangChain hub."""
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI(model=model_name)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)


def main():
    load_dotenv()
    pdf_path = "./docs/react_paper.pdf"
    index_dir = "faiss_index_react"
    
    embeddings = OpenAIEmbeddings()
    
    # Load and split PDF
    docs = load_pdf_documents(pdf_path)

    # Create and save FAISS index
    create_faiss_vectorstore(docs, embeddings, index_dir)

    # Load FAISS index
    vectorstore = load_faiss_vectorstore(index_dir, embeddings)

    # Create and run retrieval chain
    retrieval_chain = create_react_retrieval_chain(vectorstore)
    query = "Give me the gist of ReAct in 3 sentences"
    result = retrieval_chain.invoke({"input": query})
    
    print("\n[ReAct Summary]:")
    print(result["answer"])


if __name__ == '__main__':
    main()
