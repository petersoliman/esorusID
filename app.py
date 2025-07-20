from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil

print("🚀 app.py starting...")

app = FastAPI()
print("📦 FastAPI instance created.")

# Static folder for assets (CSS, JS, uploads)
app.mount("/static", StaticFiles(directory="static"), name="static")
print("📂 Mounted static directory.")

# Templates folder
templates = Jinja2Templates(directory="templates")
print("🧩 Jinja2 templates set to 'templates/'.")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    print("📥 GET / - Rendering index.html")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    print("📤 POST /search - File upload started")
    print(f"📎 Received file: {file.filename}")

    # Save the uploaded file
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"✅ File saved to: {file_path}")

    # Build the URL to the uploaded image
    uploaded_image_url = f"/static/uploads/{file.filename}"
    print(f"🖼️ Uploaded image URL: {uploaded_image_url}")

    # TODO: Replace this block with real recommendation logic
    recommended_images = os.listdir("static/recommended")
    recommended_images = sorted(recommended_images)  # Sort alphabetically
    recommended_image_urls = [f"/static/recommended/{img}" for img in recommended_images]

    print(f"🔍 Recommended images: {recommended_image_urls}")

    # Render result page with uploaded + recommended images
    return templates.TemplateResponse("result.html", {
        "request": request,
        "uploaded_image": uploaded_image_url,
        "recommended_images": recommended_image_urls
    })
