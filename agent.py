
import ollama
import time
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# 🔹 Initialize Embedding Model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Load the Vector Database (ChromaDB)
vector_db = Chroma(persist_directory="./db", embedding_function=embedding_function)

# 🔹 Initialize Memory (for tracking previous questions)
memory = ConversationSummaryBufferMemory(
    llm=OllamaLLM(model="deepseek-r1:7b"), 
    memory_key="chat_history", 
    return_messages=True,
    output_key="answer",
    max_token_limit=2048  # Ensures enough context retention
)

# 🔹 Initialize Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# 🔹 Load AI Model
llm = OllamaLLM(model="deepseek-r1:7b", streaming=True)

# 🔹 Create Conversational Q&A Chain (Ensure memory is used)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, return_source_documents=True
    

)

# 🔹 Function to Ask Questions
def ask_question(query):
    print("Query Running")
    retrieval_start = time.time()

    retrieved_docs = retriever.get_relevant_documents(query)

    retrieval_end = time.time()  # Retrieval is handled automatically

    if not retrieved_docs:
        print("⚠️ No relevant documents found in the database.")
        return {"answer": "⚠️ No relevant documents found.", "sources": []}

    print(f"\n🔎 Retrieved {len(retrieved_docs)} documents in {retrieval_end - retrieval_start:.2f} seconds")
    sources = []
    for i, doc in enumerate(retrieved_docs):
        print(f"\n📄 Source {i+1}:")
        print(f"📜 Content Preview: {doc.page_content[:300]}...")  
        if "source" in doc.metadata:
            print(f"📂 File: {doc.metadata['source']}") 
            sources.append({"source": doc.metadata["source"]})

    model_start = time.time()
    response = qa_chain.invoke({"question": query, "chat_history": memory.load_memory_variables({})["chat_history"]})
    model_end = time.time()

    ai_answer = response.get("answer", "⚠️ Sorry, I couldn't generate a response.")

    print("\n🧠 Chat Memory Updated:", memory.load_memory_variables({}))
    print("\n⏱️ Retrieval Time:", f"{retrieval_end - retrieval_start:.2f} seconds")
    print("⏱️ Model Response Time:", f"{model_end - model_start:.2f} seconds")

    formatted_answer = f"🤖 **AI Answer:**\n{ai_answer}\n\n🔗 **Sources:**\n"
    if sources:
        formatted_answer += "\n".join([f"- {s['source']}" for s in sources])
    else:
        formatted_answer += "No sources found."

    return {"answer": formatted_answer, "sources": sources}

if __name__ == "__main__":
    print("\n🗨️ Interactive AI Chat with Memory & Retrieval Enabled!\n(Type 'exit' to end the conversation.)\n")

    while True:
        question = input("\n❓ Your Question: ")
        if question.lower() == "exit":
            print("👋 Exiting chat. Have a great day!")
            break

        result = ask_question(question)

        print("\n🤖 AI Answer:", result["answer"])