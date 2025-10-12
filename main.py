import os
import streamlit as st
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter as rec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
os.environ["GOOGLE_API_KEY"] = gemini_api_key
llm = ChatGoogleGenerativeAI(model="gemini-2.5-Pro", temperature=0)

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
    #loading data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...")
    data = loader.load()
    
    #split text 
    text_splitter = rec(
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text("Splitting data...")
    docs = text_splitter.split_documents(data)

    #embeding
    embeddings = HuggingFaceEmbeddings()
    main_placeholder.text("Creating embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    #store 
    main_placeholder.text("Storing vectorstore...") 

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # Use from_chain_type instead of direct instantiation
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff"  # default, can be 'map_reduce', etc.
            )

            response = chain({"question": query}, return_only_outputs=True)
            st.header("Answer:")
            st.subheader(response["answer"])

            sources = response.get("sources", "")
            if sources:
                st.subheader("Sources:")
                source_list = sources.split("\n")
                for source in source_list:
                    st.write(f"- {source}")
