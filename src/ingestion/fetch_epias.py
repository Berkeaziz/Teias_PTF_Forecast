import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CAS_URL ="https://giris.epias.com.tr/cas/v1/tickets"
MCP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"

def get_tgt(username: str, password: str) -> str:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
    }
    payload = {"username": username, "password": password}

    r = requests.post(CAS_URL, headers=headers, data=payload, timeout=30)
    r.raise_for_status()

    tgt = r.text.strip()  
    if not tgt.startswith("TGT-"):
        raise RuntimeError(
            f"TGT beklenirken farklı cevap geldi. "
            f"status={r.status_code} content-type={r.headers.get('Content-Type')} "
            f"text={r.text[:300]}"
        )
    return tgt

def fetch_mcp_raw(start_date: str, end_date: str, tgt: str) -> dict:
    headers = {"TGT": tgt, "Content-Type": "application/json"}
    payload = {
        "startDate": f"{start_date}T00:00:00+03:00",
        "endDate": f"{end_date}T23:59:59+03:00",
    }

    r = requests.post(MCP_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()  

def save_raw_json(raw:dict,start_date : str,end_date :str) ->str:
    os.makedirs("data/raw",exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname=f"ptf_mcp_{start_date}__{end_date}__fetched_{ts}.json"
    path =os.path.join("data/raw",fname)

    with open(path,"w",encoding="utf-8") as f:
        json.dump(raw,f,ensure_ascii=False,indent=2)
    return path

def main():
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        raise RuntimeError("EPIAS_USERNAME ve EPIAS_PASSWORD Invalid")
    
    start_date ="2020-01-01"
    end_date="2026-03-06"

    tgt = get_tgt(username,password)
    raw =fetch_mcp_raw(start_date,end_date,tgt)

    out_path =save_raw_json(raw,start_date,end_date)
    print(f"RAW JSON SAVED: {out_path}")

if __name__ == "__main__":
    main()