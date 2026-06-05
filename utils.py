from langchain_openai import ChatOpenAI  
from langchain_openai import OpenAIEmbeddings
import os  
import inti.medico_api as medico_api
import httpx  
client = httpx.Client(verify=False) 



def load_vectorstore(drug_name: str):
    embeddings = OpenAIEmbeddings(
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="sk-0qxBx5yz7XnVigOpOVHNCQ",
        base_url="https://genailab.tcs.in",
    )

    path = f"faiss_db/{drug_name}"

    vectorstore = medico_api.FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("load_vectorestore is working")
    return vectorstore

def retrieve_chunks(vectorstore, query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    print("retrival is working")
    return docs


def ask_llm(query: str, docs):
    llm = ChatOpenAI(
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="sk-0qxBx5yz7XnVigOpOVHNCQ",
        base_url="https://genailab.tcs.in",
        http_client= client
        # temperature=0
    )

    context = "\n\n".join(
        [f"{d.page_content}" for d in docs]
    )

    prompt = f"""
You are a medical assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and safely:
"""

    response = llm.invoke(prompt)
    print(response)
    return response.content


# build / load vector DB
vectorstore = medico_api.build_faiss_index("dolo")

# retrieve relevant chunks
docs = retrieve_chunks(vectorstore, "what is dolo used for")

# get LLM answer
answer = ask_llm("what is dolo used for", docs)

# PRINT LLM RESPONSE
print("\n===== LLM RESPONSE =====\n")
print(answer)