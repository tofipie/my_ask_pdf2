from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
import streamlit as st
from utils import get_data_files, reset_conversation


import os
from datetime import datetime

import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

from botocore.config import Config

#free models
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub


from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain


def create_RetrievalQA_chain(query):

  qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT})
  
  answer = qa({"query": query}) #return all answer from llm #qa({"query": query})['result'] return just answer of llm
  return answer

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


DATA_PATH = "pdfs/"
HUGGINGFACEHUB_API_TOKEN = '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'
#HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN'


loader = PyPDFDirectoryLoader("./pdfs/")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)


vectorstore_faiss = FAISS.from_documents(docs, embeddings)

# Main Streamlit app
temperature= st.number_input(label="Temperature",step=.1,format="%.2f", value=0.7)

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                    model_kwargs={"temperature":temperature, "max_length":512},
                     huggingfacehub_api_token='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')
def main():
    st.title("PDF Chat Using AWS Bedrock and Anthropic Claude 💬")
    

    with st.sidebar:
        st.title('💬 PDF Chat App')
        st.button('New Chat', on_click=reset_conversation)
        st.write("Files loaded in VectorDB:")
        for file in get_data_files():
          st.markdown("- " + file)
        st.write('Made by Noa Cohen')
            
  
    user_input = st.text_area("Ask Your Question")
    button = st.button("Generate Answer")
    if user_input and button:
            summary = create_RetrievalQA_chain(user_input)
            st.write(summary)
if __name__ == "__main__":
    main()
