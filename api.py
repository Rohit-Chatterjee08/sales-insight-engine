import os
import uuid
import hashlib
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from main import run_crew

# --- Caching Configuration ---
CACHE_DIR = "cache"

app = FastAPI(
    title="FMCG Agentic API",
    description="An API to trigger an AI crew for analyzing FMCG sales data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Filesystem Caching Functions ---
def get_report_from_cache(file_hash: str):
    """Checks the cache directory for a report matching the file hash."""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None

def save_report_to_cache(report_data: dict, file_hash: str):
    """Saves the generated report to a file in the cache directory."""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    with open(cache_path, 'w') as f:
        json.dump(report_data, f)

def calculate_file_hash(file_path: str):
    """Calculates the SHA256 hash of a file by reading it in chunks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@app.post("/generate-report/", summary="Upload dataset and generate a sales report")
async def generate_sales_report(file: UploadFile = File(...)):
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Use a more robust temporary file naming scheme
    temp_file_path = os.path.join(os.getcwd(), f"temp_{uuid.uuid4()}_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        file_hash = calculate_file_hash(temp_file_path)
        cached_data = get_report_from_cache(file_hash)
        if cached_data:
            print(f"Cache hit from filesystem for file hash: {file_hash[:10]}...")
            return cached_data

        print(f"Cache miss for file hash: {file_hash[:10]}.... Generating new report.")
        result_dict = run_crew(file_path=temp_file_path)
        
        save_report_to_cache(result_dict, file_hash)
        return result_dict
        
    except Exception as e:
        error_detail = f"An unexpected error occurred: {str(e)}"
        print(f"ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Mount the frontend directory
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

