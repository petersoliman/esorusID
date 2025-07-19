from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    file_path = os.path.join("static", "index.html")
    print(f"DEBUG: Trying to open {file_path}")  # <-- Debug print
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"DEBUG: index.html size = {len(content)} bytes")  # <-- Debug print
            return content
    except Exception as e:
        print(f"ERROR: Failed to read index.html: {e}")  # <-- Debug print
        return HTMLResponse(content="<h1>Failed to load index.html</h1>", status_code=500)
