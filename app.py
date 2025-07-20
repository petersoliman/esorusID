from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/search")
async def search(request: Request, file: UploadFile = File(...)):
    print("🟢 POST /search called")
    print(f"🟢 Received file: {file.filename}")

    try:
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        file_location = os.path.join(upload_folder, file.filename)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"🟢 Saved file to {file_location}")

        # Now prepare your recommended images list (dummy for now)
        recommended_images = [
            "/static/images/recommend1.jpg",
            "/static/images/recommend2.jpg",
            "/static/images/recommend3.jpg"
        ]

        print("🟢 Rendering result.html")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "uploaded_image": f"/static/uploads/{file.filename}",
            "recommended_images": recommended_images
        })
    except Exception as e:
        print(f"❌ Error in /search: {e}")
        return HTMLResponse(f"<h1>Error: {e}</h1>", status_code=500)

@app.get("/test")
async def test(request: Request):
    print("🟢 /test called")
    recommended_images = [
        "/static/images/recommend1.jpg",
        "/static/images/recommend2.jpg",
        "/static/images/recommend3.jpg"
    ]
    uploaded_image = "/static/images/sample_upload.jpg"  # Put a sample image here manually
    return templates.TemplateResponse("result.html", {
        "request": request,
        "uploaded_image": uploaded_image,
        "recommended_images": recommended_images
    })