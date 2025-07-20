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

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    try:
        print(f"📤 Received file: {file.filename}")

        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_image_url = f"/static/uploads/{file.filename}"
        print(f"✅ File saved at {uploaded_image_url}")

        # Dummy recommendations
        recommended_images = [
            "/static/images/18868_0.jpg",
            "/static/images/18868_1.jpg",
            "/static/images/18869_0.jpg"
        ]

        return templates.TemplateResponse("result.html", {
            "request": request,
            "uploaded_image": uploaded_image_url,
            "recommended_images": recommended_images
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(content=f"<h1>Server Error: {e}</h1>", status_code=500)
