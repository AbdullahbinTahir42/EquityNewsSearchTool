import os
import streamlit as st
from dotenv import load_dotenv

# Modern Langchain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
# FIX 1: Switched to WebBaseLoader which is much more reliable for news articles
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- Configuration and Initialization ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or Streamlit Secrets.")
    st.stop()
    
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Initialize Model and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("News Search Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
faiss_dir = "faiss_index"
main_placeholder = st.empty()

for i in range(3):
    url = st.sidebar.text_input(f"Article URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.sidebar.warning("Please enter at least one valid URL.")
        st.stop()

    # Loading data using the more reliable WebBaseLoader
    loader = WebBaseLoader(web_paths=valid_urls)
    main_placeholder.text("Loading data...")
    try:
        data = loader.load()
    except Exception as e:
        main_placeholder.error(f"Error loading URLs: {e}")
        st.stop()

    # Splitting text 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text(f"Splitting documents...")
    docs = text_splitter.split_documents(data)

    # FIX 2: Prevent the IndexError by checking if docs is empty before passing to FAISS
    if not docs:
        main_placeholder.error("Error: No text could be extracted from these URLs. The websites might be blocking bots or require JavaScript rendering.")
        st.stop()

    # Embedding and Vectorstore Creation
    main_placeholder.text(f"Creating embeddings for {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save to local directory
    main_placeholder.text("Storing vectorstore...") 
    vectorstore.save_local(faiss_dir)
    
    main_placeholder.success("Vector Store created and saved successfully!")

query = st.text_input("Question: ")

if query:
    if not os.path.exists(faiss_dir):
        st.error("Vector Store not found. Please process URLs first.")
    else:
        main_placeholder.empty() 

        # Load FAISS index safely
        vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Set up standard LCEL retrieval chain
        system_prompt = (
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        st.text("Searching documents and generating answer...")
        
        try:
            # Execute chain
            response = chain.invoke({"input": query})
            
            st.header("Answer:")
            st.write(response["answer"])

            # Extract sources
            sources = set([doc.metadata.get("source") for doc in response["context"] if doc.metadata.get("source")])
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(f"- {source}")
                    
        except Exception as e:
            st.error(f"An error occurred during chain execution: {e}")
