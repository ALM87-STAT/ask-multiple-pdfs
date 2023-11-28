import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma, FAISS
import chromadb
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms import PalM


credential_path = r"proyecto-llm-405317-1760c1276351.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

from google.cloud import documentai_v1beta3 as documentai
from google.api_core.client_options import ClientOptions

# Set endpoint to EU
options = ClientOptions(api_endpoint="eu-documentai.googleapis.com:443")
# Instantiates a client
client = documentai.DocumentProcessorServiceClient(client_options=options)


loader = PyPDFLoader("2023_GPT4All_Technical_Report.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, length_function=len, chunk_overlap=200
)

documents = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


llm = ChatVertexAI()

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "De qué se trata el documento?"

qa_chain.run(query)
