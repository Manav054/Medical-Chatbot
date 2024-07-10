import warnings
warnings.filterwarnings("ignore")

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os, time

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

extracted_data = load_pdf(data = "data/")
text_chunks = text_split(extracted_data = extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key = pinecone_api_key)
index_name = "medical-chatbot"
existing_index = [index_info["name"] for index_info in pc.list_indexes()]

docsearch = None

if index_name not in existing_index:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name, pinecone_api_key = pinecone_api_key)