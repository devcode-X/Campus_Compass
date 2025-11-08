# src/retriever.py
import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# === Environment Variables ===
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct:novita")
PERSIST_DIR = os.getenv("FAISS_DIR", "faiss_index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# === Load Vectorstore ===
emb_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(PERSIST_DIR, embeddings=emb_model, allow_dangerous_deserialization=True)

# === Prompt Template ===
PROMPT = """You are Campus Compass, a helpful AI assistant for college students.
Use ONLY the provided context to answer. If the information is not present, say you don't know.

Context:
{context}

Question: {question}

Answer (with sources):
"""

# === Hugging Face Router Endpoint ===
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

def hf_llama_inference(prompt: str) -> str:
    """Query Llama 3 8B via Hugging Face router (OpenAI-style)."""
    HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": [
            {"role": "system", "content": "You are Campus Compass, a helpful AI for college students."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 400,
        "temperature": 0.1
    }

    response = requests.post(HF_CHAT_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"❌ API Error {response.status_code}: {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"]

def answer_question(question: str):
    """Search context from FAISS and query Llama model for answer."""
    results = db.similarity_search(question, k=4)
    context = "\n\n".join([r.page_content for r in results])
    filled_prompt = PROMPT.format(context=context, question=question)

    answer = hf_llama_inference(filled_prompt)
    sources = [f"{r.metadata.get('source')}:{r.metadata.get('chunk')}" for r in results]

    return {"answer": answer, "sources": sources}
# src/retriever.py
# import os
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFaceEndpoint
# from langgraph.graph import StateGraph, END
# from langchain_core.documents import Document


# # =====================
# # CONFIG
# # =====================
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
# HF_TOKEN = os.getenv("HF_API_TOKEN")

# # =====================
# # EMBEDDING + VECTORSTORE
# # =====================
# def load_vectorstore():
#     embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vs = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
#     return vs

# # =====================
# # PROMPT TEMPLATE
# # =====================
# PROMPT_TEMPLATE = """
# You are an intelligent campus assistant with access to multiple official RGIPT documents.
# Use the following retrieved context to answer the user's question clearly and completely.

# If information is missing, say "I don't know" — do not hallucinate.

# Question: {question}
# Context: {context}

# Answer:
# """

# # =====================
# # LLM via Hugging Face Endpoint
# # =====================
# def get_llm():
#     return HuggingFaceEndpoint(
#         repo_id=LLM_MODEL,
#         task="text-generation",
#         max_new_tokens=500,
#         temperature=0.2,
#         repetition_penalty=1.1,
#         huggingfacehub_api_token=HF_TOKEN,
#     )

# # =====================
# # AGENTIC GRAPH DEFINITION
# # =====================
# def build_agent():
#     vs = load_vectorstore()
#     retriever = vs.as_retriever(search_kwargs={"k": 5})
#     llm = get_llm()

#     # Define nodes
#     def retrieve(state):
#         question = state["question"]
#         results = retriever.invoke(question)
#         docs = "\n\n".join([r.page_content for r in results])
#         sources = [r.metadata["source"] for r in results]
#         state["context"] = docs
#         state["sources"] = sources
#         return state

#     def generate(state):
#         prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         llm_chain = prompt | llm | StrOutputParser()
#         answer = llm_chain.invoke({
#             "question": state["question"],
#             "context": state["context"]
#         })
#         state["answer"] = answer
#         return state

#     # Build LangGraph workflow
#     graph = StateGraph(dict)
#     graph.add_node("retrieve", retrieve)
#     graph.add_node("generate", generate)
#     graph.set_entry_point("retrieve")
#     graph.add_edge("retrieve", "generate")
#     graph.add_edge("generate", END)
#     return graph.compile()

# # =====================
# # MAIN ENTRY POINT
# # =====================
# def answer_question(question: str):
#     agent = build_agent()
#     result = agent.invoke({"question": question})
#     return {
#         "answer": result["answer"],
#         "sources": result["sources"],
#     }
