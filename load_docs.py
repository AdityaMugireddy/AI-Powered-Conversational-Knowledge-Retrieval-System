from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pdf_loader = PyMuPDFLoader("C:\\Users\\mugir\\OneDrive\\Desktop\\Data for agent 1\\Financial-Management-for-Small-Businesses-2nd-OER-Edition-1627674276.pdf")
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
chroma_db = Chroma(persist_directory="./db", embedding_function=embedding_function)

for i, doc in enumerate(documents):
    chroma_db.add_texts(
        texts=[doc.page_content],
        metadatas=[{"source": "Financial-Management-for-Small-Businesses.pdf"}]
    )

print("✅ Documents stored in vector DB with embeddings!")
