import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict, Annotated
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph.message import add_messages
# from PyPDF2 import PdfReader
# from langchain.schema import Document
# import operator


load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    # extraContext: str = ''

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
global vectorstore

# Initialize LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

workflow = StateGraph(CustomState)

# Define the function that calls the model
def call_model(state: CustomState):
    # right now we are building context on each call but we can look into storing it on graph state later
    context = ''
    # passing a hardcoded file path
    file_content = extract_text_from_file("./uploads/yourfile.pdf")
    # Index the extracted content
    p_store: FAISS = index_document(file_content)
    query = state["messages"][-1].content
    context = retrieve(query , p_store)

    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability. "
        f"Use the following context if it is relevant: {context}"
        "If the context is not relevant use your own knowledge set to answer the question."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("call_model", call_model)
workflow.add_edge(START, "call_model")

# Add simple in-memory checkpointer
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

# Extract text from a PDF file
# def extract_text_from_pdf(file_path):
#     reader = PdfReader(file_path, allow_dangerous_pickle=False)
#     text = ""

#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# If we extract content and want to conver to Langchain document.
# def pdf_to_documents(file_name):
#     file_path = os.path.join(UPLOAD_FOLDER, file_name)
#     text = extract_text_from_pdf(file_path)
#     # Split the text into smaller chunks (optional)
#     chunks = text.split("\n\n")  # Split by paragraphs or custom logic
#     documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
#     return documents

def handle_file_download(file_url):
    """ Download the file from WhatsApp and saves it locally. """

    file_name = file_url.split("/")[-1] + ".pdf"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
   
    response = requests.get(file_url)
    # write to local folder
    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_name

def extract_text_from_file(file_name):
    """Extracts text from the uploaded file, stored in the local folder."""
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    # for now we suppoer pdf file format
    if file_path.endswith(".pdf"):
       # we should be looping through all the files under the upload folder but for now a single file
       loader = PyPDFLoader(file_path)
       documents = loader.load()
       
       return documents
    else:
        raise ValueError("Unsupported file type")

def index_document(documents):
    """Indexes the extracted content into the vector store."""
   
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    # Create vectors
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Persist the vectors locally on disk
    vectorstore.save_local("faiss_index_constitution")

    # Load from local storage
    persisted_vectorstore = FAISS.load_local("faiss_index_constitution", embeddings, allow_dangerous_deserialization=True)
  
    return persisted_vectorstore


@app.route('/whatsapp/incoming', methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').strip()  # User's WhatsApp message
    incoming_file_url = request.values.get('MediaUrl0')  # File URL if provided
    
    response = MessagingResponse()
    msg = response.message()
    
    # Temp config, for prod can ue the incoming number 
    config = {"configurable": {"thread_id": "abc123"}}
   
    try:
        if incoming_file_url:
            file_name: str = handle_file_download(incoming_file_url)
            file_content = extract_text_from_file(file_name)
            
            # Index the extracted content
            index_document(file_content)
            
            msg.body("File content indexed successfully. Now you can ask questions related to the uploaded document.")
        else:
            # Use the LangChain agent to handle the user's message
            input_messages = [HumanMessage(incoming_msg)]
            print(f"input_messages: {input_messages}")

            agent_response = agent.invoke({"messages": input_messages}, config)
            ai_response = agent_response["messages"][-1]
            msg.body(str(ai_response.content))
    except Exception as e:
        msg.body(f"Oops! Something went wrong: {e}")

    return str(response)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
