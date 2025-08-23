import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetreivalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


st.title("News Search Tool📈")
st.sidebar.title("New Article URLS")


for i in range(3):
    url = st.sidebar.text_input(f"Article URL {i + 1}")

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    pass