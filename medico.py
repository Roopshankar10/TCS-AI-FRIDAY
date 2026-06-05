import requests
import urllib3
import json

# Suppress enterprise proxy / network SSL warnings in the terminal
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_fda_data(drug_name):
    """Hits openFDA API and returns the raw JSON payload."""
    url = "https://api.fda.gov/drug/label.json"
    params = {
        "search": f'openfda.generic_name:"{drug_name}" openfda.brand_name:"{drug_name}"',
        "limit": 1
    }
    try:
        response = requests.get(url, params=params, verify=False)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"❌ No official FDA label found for '{drug_name}'. Checking spelling...")
            return None
    except Exception as e:
        print(f"💥 Network Error: {e}")
        return None

def format_pharma_response(raw_data, drug_name):
    """
    Parses the raw openFDA JSON payload and extracts content 
    to map directly to the hackathon's solution expectations.
    """
    if not raw_data or "results" not in raw_data:
        return "No processable data found."
    
    # Extract the primary label record
    record = raw_data["results"][0]
    openfda_meta = record.get("openfda", {})
    
    # -------------------------------------------------------------
    # EXTRACT DATA FIELDS FROM PAYLOAD
    # -------------------------------------------------------------
    generic_name = openfda_meta.get("generic_name", [drug_name.upper()])[0]
    brand_name = openfda_meta.get("brand_name", ["N/A"])[0]
    
    # 1. Active Ingredient & Purpose
    active_ing = record.get("active_ingredient", ["Not explicitly listed"])[0]
    purpose = record.get("purpose", ["Not explicitly listed"])[0]
    
    # 2. Extract Biological Mechanisms
    moa = openfda_meta.get("pharm_class_moa", ["Mechanism classification not categorized in indexing"])[0]
    pe = openfda_meta.get("pharm_class_pe", ["Physiological effects not categorized in indexing"])
    pe_string = ", ".join(pe)
    
    # 3. Extract Warnings & Interactions
    # openFDA combines drug-drug interactions under 'warnings' or 'drug_interactions' depending on the OTC vs Prescription format
    warnings_text = record.get("warnings", [""])[0]
    explicit_interactions = record.get("drug_interactions", [""])[0]
    
    # Combine or fallback to ensure we catch everything
    combined_warnings = explicit_interactions if explicit_interactions else warnings_text
    
    # Extract clinical actions/mitigations (Ask Doctor / Stop Use sections)
    ask_doctor = record.get("ask_doctor_or_pharmacist", record.get("ask_doctor", ["Consult a physician before combining therapies."]))[0]
    stop_use = record.get("stop_use", ["Discontinue immediately if side effects present."])[0]

    # -------------------------------------------------------------
    # MAP TO HACKATHON SOLUTION REQUIREMENTS
    # -------------------------------------------------------------
    formatted_output = f"""
================================================================================
🔬 CLINICAL BRIEF GENERATED FOR: {generic_name.upper()} ({brand_name.upper()})
================================================================================

[1] IDENTIFICATION & CORE INGREDIENTS
--------------------------------------------------------------------------------
* Generic Compound: {generic_name}
* Active Ingredient Summary: {active_ing}
* Primary Clinical Purpose: {purpose}

[2] UNDERLYING BIOLOGICAL MECHANISM
--------------------------------------------------------------------------------
* Mechanism of Action (MoA): {moa}
* Physiological Effects (PE): {pe_string}

[3] INTERACTION TYPES & SEVERITY INDICATION (EXTRACTED SAFETY SUMMARY)
--------------------------------------------------------------------------------
{combined_warnings[:2500]} ... [Truncated for processing]

[4] SUGGESTED CLINICAL ACTIONS & RISK MITIGATION
--------------------------------------------------------------------------------
* Pre-Combination Advisories: {ask_doctor}
* Intervention / Stop-Use Protocols: {stop_use}

================================================================================
"""
    return formatted_output

# --- Standalone Execution ---
if __name__ == "__main__":
    print("=" * 60)
    print("💊 openFDA Structured Data Extractor Engine")
    print("=" * 60)
    
    target_drug = input("Enter drug name to process and format: ").strip()
    
    if target_drug:
        # Step 1: Fetch the raw data block
        raw_payload = get_fda_data(target_drug)
        
        if raw_payload:
            # Step 2: Format the raw data directly against the problem statement
            final_report = format_pharma_response(raw_payload, target_drug)
            
            # Print to screen
            print(final_report)
            
            # Step 3: Save to file so your team can use it immediately!
            filename = f"{target_drug.lower()}_formatted_brief.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"💾 Success! Formatted clinical summary saved locally as '{filename}'")
