import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set NVIDIA API Key from secrets
os.environ['NVIDIA_API_KEY'] = st.secrets['NVIDIA_API_KEY']

# Load documents
loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
docs = loader.load()

# Initialize embeddings
embeddings = NVIDIAEmbeddings()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Create vector store
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Initialize Chat model
model = ChatNVIDIA(model="mistral_7b")

# Define Streamlit app
def main():
    st.title("LangChain 🦜️ QA App")

    # Add sidebar
    st.sidebar.title("About LangChain")
    st.sidebar.write("""
    LangChain is a framework for developing applications powered by language models. It enables applications that:
    - Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
    - Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)
    """)

    # Add input field for the question
    question = st.text_input("Ask a question", "")

    # Add 'Get Answer' button
    if st.button("Get Answer"):
        if question:
            # Display loading spinner
            with st.spinner("Fetching answer..."):
                # Define template
                template = """Answer the question based only on the following context: {context} Question: {question}"""
                prompt = ChatPromptTemplate.from_template(template)

                # Get answer
                answer_chain = prompt | model | StrOutputParser()
                answer = answer_chain.invoke({'question': question, 'context': documents})

                # Display answer
                st.write("Answer:", answer)
                # Display balloons once output is shown
                st.balloons()
        else:
            st.warning("Please provide a question.")

    # Add some additional text for guidance
    st.write("")
    st.write("### Tips for asking questions:")
    st.write("- Keep questions concise.")
    st.write("- Provide context if necessary.")
    st.write("- Be patient for the answer.")

if __name__ == "__main__":
    main()
