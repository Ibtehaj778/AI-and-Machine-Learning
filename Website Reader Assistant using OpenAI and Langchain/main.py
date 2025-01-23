import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv

file_path = "vector_FAISS.pkl"
load_dotenv()
llm = OpenAI(temperature=0.9,max_tokens=500)

st.title("News Research Tool")
st.sidebar.title("News Article Urls")
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_clicked = st.sidebar.button("Process Urls")

mainplace_folder = st.empty()
if process_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    mainplace_folder.text("Data Loading started....")
    data = loader.load()
    mainplace_folder.text("Data Loaded....")
    #split data
    mainplace_folder.text("Splitting Data.....")
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.'],chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    mainplace_folder.text("Creating Embeddings.....")
    #embedd data and store in vectors(save it to FAISS)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorindex_hf = FAISS.from_documents(chunks, embeddings)
    
    mainplace_folder.text("Embeddings Created....")
    #save vectors in pkl file
    with open(file_path,"wb") as f:
        pickle.dump(vectorindex_hf,f)
    

query = mainplace_folder.text_input("Query: ")

if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorindex_hf = pickle.load(f)
            
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorindex_hf.as_retriever())
        result = chain({"question":query},return_only_outputs=True)
        st.header("Answer")
        st.subheader(result['answer'])
        
        sources = result.get("sources","")
        if sources:
            st.subheader("sources")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    
    
    
    
    
    
    
    
    
    
    
