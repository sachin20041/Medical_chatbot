from flask import Flask, request, render_template
from dotenv import load_dotenv
from pinecone import Pinecone
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# Load env explicitly
load_dotenv("D:/medical_chatbot/Medical_chatbot/.env")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Check .env file path.")

app = Flask(__name__)

print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings ready")


print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever ready")


print("Loading LLM...")
pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)
print("LLM ready")

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a medical assistant. Use the following medical context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}\n\n"
        "Answer:"
    )
)


qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    response = rag_chain.invoke({"input": user_input})
    return response["answer"]

if __name__ == "__main__":
    app.run(port=5000, debug=True)
