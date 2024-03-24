import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import chain

# Accessing the API key securely using Streamlit secrets
nvidia_api_key = "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"

# Setting up loaders and models
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
embeddings = NVIDIAEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
model = ChatNVIDIA(model="mistral_7b", nvidia_api_key=nvidia_api_key)

# Setting up templates and transformers
hyde_template = """Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question:
{question}"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_query_transformer = hyde_prompt | model | StrOutputParser()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
answer_chain = prompt | model | StrOutputParser()

# Defining Streamlit app
def main():
    st.title("Langsmith Q&A App")

    # Input question
    question = st.text_input("Enter your question:")

    # Button to trigger processing
    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            # Process the question
            result = get_answer(question)
            st.write(result)

# Function to process the question
@chain
def get_answer(question):
    documents = hyde_retriever.invoke(question)
    answers = []
    for s in answer_chain.stream({"question": question, "context": documents}):
        answers.append(s)
    return "".join(answers)

# Function to retrieve documents
@chain
def hyde_retriever(question):
    hypothetical_document = hyde_query_transformer.invoke({"question": question})
    return retriever.invoke(hypothetical_document)

if __name__ == "__main__":
    main()
