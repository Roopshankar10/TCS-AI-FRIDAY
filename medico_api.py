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

requests.get = lambda *args, **kwargs: requests.api.get(*args, verify=False, **kwargs)


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

# result_of_medico = get_drug_rag_object("dolo")

# print(result_of_medico)



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

def ingest_drug(drug_name):
    rag_obj = get_drug_rag_object(drug_name)

    if not rag_obj["found"] or len(rag_obj["chunks"]) == 0:
        print("❌ No data found for drug")
        return None

    docs = chunks_to_documents(rag_obj)

    print(f"✅ Created {len(docs)} documents")

    db = store_in_faiss(docs, embeddings)

    print("✅ FAISS index created & saved")

    return db

db = ingest_drug("dolo")


embeddings = OpenAIEmbeddings(
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    http_client=client
)



def store_in_faiss(docs, embeddings, index_path="drug_faiss_index"):
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    return db

def load_faiss(index_path, embeddings):
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

results = db.similarity_search("what are warnings", k=3)

for r in results:
    print("\n---")
    print(r.page_content)
    print(r.metadata)