#https://medium.com/@gaurav.jaik86/building-an-ai-powered-chat-with-pdf-app-with-streamlit-langchain-faiss-and-llama2-affadea65737
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
#from loadllm import Loadllm
from streamlit_chat import message
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
#from utils import get_data_files, reset_conversation
import os
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

DB_FAISS_PATH = "vectorstores/db_faiss"
DATA_PATH = "pdfs/"

loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)
    

# Create embeddings using Sentence Transformers
#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Create a FAISS vector store and save embeddings

# Load the language model
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                    model_kwargs={"temperature":0.5, "max_length":512},
                             huggingfacehub_api_token='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')

        # Create a conversational chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.messages = []

def get_data_files():
    data_files = []
    for dirname, _, filenames in os.walk("pdfs"):
        for filename in filenames:
            data_files.append(os.path.join(filename))
    return data_files
        
st.title("PDF Chat Using AWS Bedrock and Anthropic Claude 💬")

#max_tokens = st.number_input('Max Tokens', value=1000)
#temperature= st.number_input(label="Temperature",step=.1,format="%.2f", value=0.7)
#llm_model = st.selectbox("Select LLM", ["Anthropic Claude V2", "Amazon Titan Text Express v1", "Ai21 Labs Jurassic-2 Ultra"])


# Create a sidebar and a button to reset the chat, using our reset_conversation function from utils.py
# Also display a list of all files in the data folder that were used to train our model
st.sidebar.title("App Description")
with st.sidebar:
    st.button('New Chat', on_click=reset_conversation)
    st.write("Files loaded in VectorDB:")
    for file in get_data_files():
        st.markdown("- " + file)
    st.write('Made by Noa Cohen')

        # Function for conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

        # Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

        # Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = []#["Hello ! Ask me(LLAMA2) about " + " 🤗"] + self.uploaded_file.name + 

if 'past' not in st.session_state:
    st.session_state['past'] = []#["Hey ! 👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
        with st.form(key='my_form', clear_on_submit=True):
               # user_input = st.text_input("Query:", placeholder="שאל שאלה...", key='input')
                user_input = st.chat_input("שאל שאלה...")
            #    submit_button = st.form_submit_button(label='Send')

       # if submit_button and user_input:
        if user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
        with response_container:
                for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                        message(st.session_state["generated"][i], key=str(i), avatar_style="🦖")
