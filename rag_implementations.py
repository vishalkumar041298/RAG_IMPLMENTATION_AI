import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()


# Shared setup
def initialize_components():
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model='gpt-4o-mini')
    vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
    return llm, vectorstore


# RAG Method 1: Using LangChain hub (retrieval-qa-chat)
def run_hub_rag(llm, vectorstore, query):
    retrievalqa_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    combine_docs_chain = create_stuff_documents_chain(llm, retrievalqa_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(input={'input': query})
    print("\n[Hub RAG Result]:")
    print(result)


# RAG Method 2: Manual PromptTemplate with document formatting
def run_manual_rag(llm, vectorstore, query):
    template = """Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Use three sentences maximum and keep the answer as concise as possible.
                Always say "thanks for asking!" at the end of the answer.

                {context}

                Question: {question}

                Helpful Answer:"""
                    
    custom_rag_prompt = PromptTemplate.from_template(template=template)

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_chain = (
        {'context': vectorstore.as_retriever() | format_docs, 'question': RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke(query)
    print("\n[Manual RAG Result]:")
    print(result.content)


# RAG Method 3: Simple PromptTemplate directly from query (non-RAG baseline)
def run_simple_prompt(llm, query):
    prompt = PromptTemplate.from_template(template=query)
    chain = prompt | llm
    result = chain.invoke(input={})
    print("\n[Simple Prompt Result]:")
    print(result.content)


# Main Execution
if __name__ == '__main__':
    query = 'what is pinecone in machine learning?'
    llm, vectorstore = initialize_components()

    # Uncomment the ones you want to run
    # run_hub_rag(llm, vectorstore, query)
    run_manual_rag(llm, vectorstore, query)
    # run_simple_prompt(llm, query)
