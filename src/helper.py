from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

_embeddings = None

def load_pdf_documents(data_folder: str) -> List[Document]:
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs

def text_split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)
def filter_chunks(chunks):
    cleaned = []
    for c in chunks:
        text = c.page_content.strip()
        if len(text) > 50 and not text.lower() in ["ii", "iii", "(ii)", "(iii)"]:
            cleaned.append(c)
    return cleaned

def download_hugging_face_embeddings():
    global _embeddings
    if _embeddings is None:
        print(">>> Loading embedding model...", flush=True)
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        print(">>> Embedding model loaded", flush=True)
    return _embeddings
