from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil

print("🚀 Starting app.py")

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
print("📂 Static directory mounted")

# Jinja2 templates folder
templates = Jinja2Templates(directory="templates")
print("🧩 Jinja2 templates loaded from 'templates/'")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("📥 GET /")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    try:
        print("📤 POST /search")
        print(f"📎 Uploaded file: {file.filename}")

        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        print("📁 Upload directory ensured")

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ File saved to {file_path}")

        # Construct URL for uploaded image
        uploaded_image_url = f"/static/uploads/{file.filename}"
        print(f"🖼️ Uploaded image URL: {uploaded_image_url}")

        # Dummy recommended images (use static placeholder images)
        recommended_images = [
            "/static/images/rec1.jpg",
            "/static/images/rec2.jpg",
            "/static/images/rec3.jpg"
        ]
        print(f"🔍 Recommended images: {recommended_images}")

        return templates.TemplateResponse("result.html", {
            "request": request,
            "uploaded_image": uploaded_image_url,
            "recommended_images": recommended_images
        })

    except Exception as e:
        print(f"❌ ERROR in /search: {e}")
        return HTMLResponse(content=f"<h1>Server Error: {e}</h1>", status_code=500)
