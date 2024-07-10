import warnings
warnings.filterwarnings("ignore")

from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
pc = Pinecone(api_key = pinecone_api_key) 
docsearch = PineconeVectorStore.from_existing_index(index_name = index_name, embedding = embeddings)

PROMPT = PromptTemplate.from_template(template = prompt_template)

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.8, max_output_tokens = 512, 
                             safety_settings ={
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                             }
        )

qa_chain = create_stuff_documents_chain(llm = llm, prompt = PROMPT)
rag_chain = create_retrieval_chain(docsearch.as_retriever(search_kwards = {'k': 5}), qa_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = rag_chain.invoke({"input": input})
    print("Response: ", result["answer"])
    return str(result["answer"])

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 8080)