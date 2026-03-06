import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
MCP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"

OUT_DIR = "data/raw/ptf_mcp"  

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
            f"Insert Ansvers"
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
    if r.status_code != 200:
        print("MCP status:", r.status_code)
        print("MCP text head:", r.text[:500])
    r.raise_for_status()
    return r.json()

def daterange_chunks(start_date: str, end_date: str, chunk_days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)

def _find_first_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_first_list(v)
            if found is not None:
                return found
    return None

def raw_to_df(raw: dict, chunk_start: str, chunk_end: str) -> pd.DataFrame:
    rows = _find_first_list(raw)
    if not isinstance(rows, list) or len(rows) == 0:
        return pd.DataFrame()

    df = pd.json_normalize(rows)


    df["_chunk_start"] = chunk_start
    df["_chunk_end"] = chunk_end


    for col in ["date", "datetime", "time", "period", "periodDate", "periodTime"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    return df

def write_parquet_partitioned(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return 0


    ts_col = None
    for c in ["date", "datetime", "time", "period", "periodDate"]:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
            ts_col = c
            break

    if ts_col:
        df["year"] = df[ts_col].dt.year.astype("int16")
        df["month"] = df[ts_col].dt.month.astype("int8")
    else:
        cs = pd.to_datetime(df["_chunk_start"].iloc[0])
        df["year"] = int(cs.year)
        df["month"] = int(cs.month)

    os.makedirs(out_dir, exist_ok=True)


    df.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="snappy",
        index=False,
        partition_cols=["year", "month"],
    )
    return len(df)

def main():
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        raise RuntimeError("EPIAS_USERNAME ve EPIAS_PASSWORD Invalid")

    start_date = "2018-01-01"
    end_date = "2026-03-05"
    chunk_days = 30

    tgt = get_tgt(username, password)
    print("TGT OK")

    ok_chunks = 0
    fail_chunks = 0
    total_rows = 0

    for s, e in daterange_chunks(start_date, end_date, chunk_days=chunk_days):
        try:
            print(f"Fetching: {s} -> {e}")
            raw = fetch_mcp_raw(s, e, tgt)

            df = raw_to_df(raw, s, e)
            written = write_parquet_partitioned(df, OUT_DIR)

            print(f"  WROTE {written} rows to parquet dataset: {OUT_DIR}")
            total_rows += written
            ok_chunks += 1

            time.sleep(0.2)

        except requests.HTTPError as ex:
            fail_chunks += 1
            print(f"  FAILED: {s}->{e} | HTTPError: {ex}")
            continue

    print(f"DONE. OK_CHUNKS={ok_chunks}, FAIL_CHUNKS={fail_chunks}, TOTAL_ROWS={total_rows}")
    print(f"PARQUET DATASET: {OUT_DIR}")

if __name__ == "__main__":
    main()