import os
import streamlit as st
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter as rec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
# Using the recommended community imports for embeddings and loaders
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration and Initialization ---

gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key is None:
    # Use st.error instead of raise for better Streamlit display
    st.error("GEMINI_API_KEY environment variable is not set. Please set it.")
    st.stop()
    
# Set the environment variable used by the underlying Google SDK client
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# 1. FIX: Changed model name to standard lowercase ("gemini-2.5-pro") 
# to avoid NotFound or Invalid Argument errors caused by incorrect casing.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

st.title("News Search Tool📈")
st.sidebar.title("New Article URLS")

urls = []
file_path = "faiss_index.pkl"
main_placeholder = st.empty()

for i in range(3):
    url = st.sidebar.text_input(f"Article URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    # Check if any non-empty URLs were provided
    valid_urls = [url for url in urls if url]
    if not valid_urls:
        st.sidebar.warning("Please enter at least one valid URL.")
        st.stop()

    # Loading data
    loader = UnstructuredURLLoader(urls=valid_urls)
    main_placeholder.text("Loading data...")
    try:
        data = loader.load()
    except Exception as e:
        main_placeholder.error(f"Error loading URLs: {e}")
        st.stop()

    # Splitting text 
    text_splitter = rec(
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text(f"Splitting {len(data)} documents...")
    docs = text_splitter.split_documents(data)

    # Embedding
    # 2. FIX: Using HuggingFaceEmbeddings from langchain_community
    embeddings = HuggingFaceEmbeddings()
    main_placeholder.text(f"Creating embeddings for {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Store
    main_placeholder.text("Storing vectorstore...") 

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    main_placeholder.success("Vector Store created and saved successfully!")

query = st.text_input("Question: ")

if query:
    if not os.path.exists(file_path):
        st.error("Vector Store not found. Please process URLs first.")
    else:
        main_placeholder.empty() # Clear the previous success message/placeholder

        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # 3. CRITICAL FIX: Changed 'stuff' to 'map_reduce' chain type.
            # The 'stuff' chain puts ALL documents into a single prompt, which likely
            # caused the 'Invalid Argument' error due to context window overload.
            # 'map_reduce' processes documents individually (mapping) and then
            # synthesizes the answers (reducing), handling large inputs gracefully.
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type="map_reduce" 
            )

            st.text("Searching documents and generating answer...")
            
            try:
                # This is the line that caused the error previously
                response = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer:")
                st.subheader(response["answer"])

                sources = response.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    source_list = sources.split("\n")
                    for source in source_list:
                        if source.strip(): # Avoid empty lines
                            st.write(f"- {source}")
            except Exception as e:
                st.error(f"An error occurred during chain execution. Try a simpler query or check the logs: {e}")
