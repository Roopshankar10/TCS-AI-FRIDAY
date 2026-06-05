import requests
import urllib3
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests
import os 
import httpx

client = httpx.Client(
    verify=False
)


load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_fda_data(drug_name):
    url = "https://api.fda.gov/drug/label.json"

    params = {
        "search": f'openfda.generic_name:"{drug_name}" openfda.brand_name:"{drug_name}"',
        "limit": 1
    }

    try:
        response = requests.get(url, params=params, verify=False)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_drug_rag_object(drug_name: str):
    raw_data = get_fda_data(drug_name)

    if not raw_data or "results" not in raw_data:
        return {
            "drug_name": drug_name,
            "found": False,
            "chunks": []
        }

    record = raw_data["results"][0]
    openfda = record.get("openfda", {})

    generic = openfda.get("generic_name", [drug_name])[0]
    brand = openfda.get("brand_name", ["N/A"])[0]

    active_ing = record.get("active_ingredient", ["Not listed"])[0]
    purpose = record.get("purpose", ["Not listed"])[0]

    moa = openfda.get("pharm_class_moa", ["Not categorized"])[0]
    pe = openfda.get("pharm_class_pe", ["Not categorized"])

    warnings = record.get("warnings", [""])[0]
    interactions = record.get("drug_interactions", [""])[0]
    safety = interactions if interactions else warnings

    ask_doctor = record.get(
        "ask_doctor_or_pharmacist",
        record.get("ask_doctor", ["Consult healthcare provider"])
    )[0]

    stop_use = record.get(
        "stop_use",
        ["Stop use if adverse effects occur"]
    )[0]

    return {
        "drug_name": drug_name,
        "found": True,
        "metadata": {
            "generic_name": generic,
            "brand_name": brand
        },
        "clinical": {
            "active_ingredient": active_ing,
            "purpose": purpose
        },
        "mechanism": {
            "moa": moa,
            "physiological_effects": pe
        },
        "safety": {
            "warnings_and_interactions": safety,
            "ask_doctor": ask_doctor,
            "stop_use": stop_use
        },
        "chunks": [
            {
                "chunk_id": "metadata",
                "text": f"{generic} {brand}"
            },
            {
                "chunk_id": "clinical",
                "text": f"{active_ing} | {purpose}"
            },
            {
                "chunk_id": "mechanism",
                "text": f"{moa} | {', '.join(pe)}"
            },
            {
                "chunk_id": "safety",
                "text": f"{safety}"
            },
            {
                "chunk_id": "guidance",
                "text": f"{ask_doctor} | {stop_use}"
            }
        ]
    }

testing = get_drug_rag_object('dolo')


def chunks_to_documents(drug_rag_object):
    docs = []

    for chunk in drug_rag_object["chunks"]:
        docs.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    **chunk.get("metadata", {}),
                    "chunk_id": chunk["chunk_id"],
                    "drug_name": drug_rag_object["drug_name"]
                }
            )
        )

    return docs

t1= chunks_to_documents(testing)

def build_faiss_index(drug_name: str):
    save_path = f"faiss_db/{drug_name}"

    rag_object = get_drug_rag_object(drug_name)

    if not rag_object["found"]:
        print(f"No data found for {drug_name}")
        return None

    docs = chunks_to_documents(rag_object)

    embeddings = OpenAIEmbeddings(
        model="azure/genailab-maas-text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        http_client=client
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(save_path)

    print(f"Vector DB saved at: {save_path}")

    return vectorstore

build_faiss_index("dolo")

# def load_vectorstore(drug_name: str):
#     embeddings = OpenAIEmbeddings(
#         model="azure/genailab-maas-text-embedding-3-large",
#         api_key=os.getenv("OPENAI_API_KEY"),
#         base_url=os.getenv("BASE_URL"),
#     )

#     path = f"faiss_db/{drug_name}"

#     vectorstore = FAISS.load_local(
#         path,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     return vectorstore

# def retrieve_chunks(vectorstore, query: str, k: int = 3):
#     docs = vectorstore.similarity_search(query, k=k)
#     return docs


# def ask_llm(query: str, docs):
#     llm = ChatOpenAI(
#         model="azure_ai/genailab-maas-DeepSeek-V3-0324",
#         api_key="sk-0qxBx5yz7XnVigOpOVHNCQ",
#         base_url="https://genailab.tcs.in",
#         # temperature=0
#     )

#     context = "\n\n".join(
#         [f"{d.page_content}" for d in docs]
#     )

#     prompt = f"""
# You are a medical assistant.

# Use ONLY the context below to answer the question.

# Context:
# {context}

# Question:
# {query}

# Answer clearly and safely:
# """

#     response = llm.invoke(prompt)
#     print(response)
#     return response.content


# # # build / load vector DB
# # vectorstore = build_faiss_index("dolo")

# # # retrieve relevant chunks
# # docs = retrieve_chunks(vectorstore, "what is dolo used for")

# # # get LLM answer
# # answer = ask_llm("what is dolo used for", docs)

# # # PRINT LLM RESPONSE
# # print("\n===== LLM RESPONSE =====\n")
# # print(answer)