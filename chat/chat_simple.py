import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv

load_dotenv()

# Configurar la página de Streamlit
st.set_page_config(page_title="GlobeBotter", page_icon="🌎")
st.header("Bienvenido al chat Bariloche, soy tu guía en Bariloche. ¿Qué te gustaría hacer?")

# Agregar la clave de API de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

# Cargar y dividir el documento PDF

current_dir = os.getcwd()  # Obtener el directorio actual
pdf_path = os.path.join(current_dir, 'pdf/GuiaViajeBariloche.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

raw_documents = PyPDFLoader(pdf_path).load()
documents = text_splitter.split_documents(raw_documents)

# Crear la base de datos FAISS con embeddings de OpenAI
db = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))

# Crear la memoria del chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Crear el modelo de lenguaje
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

# Crear la herramienta de recuperación
tool = create_retriever_tool(db.as_retriever(), "GuiaViajeBariloche", "Buscar y retornar documentos sobre Bariloche")
tools = [tool]

# Crear el agente conversacional
agent_executor = create_conversational_retrieval_agent(
    llm=llm,
    tools=tools,
    memory_key='chat_history',
    verbose=True
)

# Inicializar el estado de la sesión para almacenar mensajes y memoria
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿Cómo puedo ayudarte?"}]
if "memory" not in st.session_state:
    st.session_state["memory"] = memory

# Mostrar el historial de chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Capturar la consulta del usuario en Streamlit
user_query = st.text_input("¿Qué te gustaría hacer en Bariloche?", placeholder="Pregúntame cualquier cosa...")

if user_query:
    # Agregar la consulta del usuario al historial
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor(user_query, callbacks=[st_cb])['output']

        # Agregar la respuesta del asistente al historial
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)

# Botón para reiniciar el historial de chat
if st.sidebar.button("Reset chat history"):
    st.session_state["messages"] = []
