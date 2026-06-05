import os
import sqlite3
import httpx
import urllib3
import pandas as pd
from dotenv import load_dotenv

# LangChain Extensions
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize Environment Context
load_dotenv()

# Override context proxies for secure enterprise networks
client = httpx.Client(verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PharmaSQLAgent:
    def __init__(self, db_path="pharma_workbench.db"):
        self.db_path = db_path
        
        # Initialize your Azure/OpenAI Chat model wrapper cleanly
        self.llm = ChatOpenAI(
            model="azure/genailab-maas-gpt-4o", # Ensure this matches your active Chat model deployment deployment name
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            http_client=client,
            temperature=0.0 # Strict accuracy parameter setting for coding/SQL tasks
        )
        
        # Automatically grab table structural configurations to give context to the AI
        self.schema_context = self._get_database_schema()

    def _get_database_schema(self):
        """Extracts table schema structure from SQLite to inject into the system prompt."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch table create instructions
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='drug_interactions';")
            schema_row = cursor.fetchone()
            
            # Grab a 3-row data snippet to show text content formatting layout
            sample_df = pd.read_sql_query("SELECT * FROM drug_interactions LIMIT 3;", conn)
            sample_text = sample_df.to_string(index=False)
            
            conn.close()
            
            schema_info = f"""
Database Table: drug_interactions
DDL Schema Definition:
{schema_row[0] if schema_row else "Table not located"}

Sample Rows for Data Formatting Context:
{sample_text}
"""
            return schema_info
        except Exception as e:
            return f"Error gathering metadata profiles: {e}"

    def execute_read_only_query(self, sql_query):
        """Runs the generated SELECT statement safely against the local engine."""
        # Security Guardrail: Explicit read-only selection check
        clean_query = sql_query.strip().strip("`").replace("sql", "").strip()
        
        if not clean_query.lower().startswith("select"):
            return "Error: Destruction/Modification statements are barred on this interface registry context."
            
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql_query(clean_query, conn)
            conn.close()
            
            if result_df.empty:
                return "No matching rows located for target pair parameter metrics."
            return result_df.to_string(index=False)
        except Exception as e:
            return f"SQL Execution Error: {e}"

    def ask(self, user_question):
        """Processes the natural language query, creates SQL, runs it, and reads out results."""
        print(f"\n🧠 [AGENT LOGIC] Processing researcher query: '{user_question}'")
        
        # Step 1: Tell the AI how to write a correct query matching our explicit pairwise columns
        sql_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an elite SQL query generator for a clinical drug interaction database.
Given a user request, convert it into a syntactically correct SQLite SELECT statement.

CRITICAL RULES:
1. Return ONLY the raw SQL query code block string. No explanations, no markdown ticks.
2. The database table contains lowercase columns: drug_1, drug_2, interaction_description.
3. Because interactions are pairwise, always evaluate entries both ways using an OR condition. Example:
   (drug_1 = 'DrugA' AND drug_2 = 'DrugB') OR (drug_1 = 'DrugB' AND drug_2 = 'DrugA')
4. Always apply a LIKE or exact text matching approach based on the input text.

Database Authority Schema Profile:
{schema}
"""),
            ("human", "Generate the SQLite query for this question: {question}")
        ])
        
        # Generate SQL Query
        chain = sql_generation_prompt | self.llm
        response = chain.invoke({"schema": self.schema_context, "question": user_question})
        generated_sql = response.content.strip()
        
        print(f"📡 [AGENT LOGIC] Generated SQL Command:\n{generated_sql}")
        
        # Step 2: Execute the query locally
        db_output = self.execute_read_only_query(generated_sql)
        print(f"💾 [AGENT LOGIC] Retrieved Database Output:\n{db_output}")
        
        # Step 3: Package the row back into a pristine conversational synthesis
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Clinical Pharmacologist AI Agent. 
Take the user's initial question and the raw ground-truth rows pulled from our verified database tables, and synthesize a clear, conversational, and direct scientific summary answer. 
Do not hallucinate or invent parameters outside the database output text data records."""),
            ("human", """Researcher Question: {question}
Database Data Records Row: {data}

Synthesize your final summary brief below:""")
        ])
        
        synthesis_chain = synthesis_prompt | self.llm
        final_answer = synthesis_chain.invoke({"question": user_question, "data": db_output})
        
        return final_answer.content

# --- Standalone Testing Entrypoint ---
if __name__ == "__main__":
    # Initialize the engine
    agent = PharmaSQLAgent()
    
    # Try searching for a known pairwise link combination from your dataset snippet!
    query = "What happens if a patient takes Tenofovir disoproxil with Ganciclovir?"
    answer = agent.ask(query)
    
    print("\n" + "="*60)
    print("🔬 FINAL AGENT CONVERSATIONAL RESPONSE:")
    print("="*60)
    print(answer)
