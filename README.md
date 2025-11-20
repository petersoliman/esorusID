# esorusID

Image search and recommendation engine using OpenAI's CLIP model and FAISS.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Images**
   Place your images in `static/recommendations`.
   ```bash
   mkdir -p static/recommendations
   # Copy images here
   ```

3. **Build Index**
   Run the indexing script to process images:
   ```bash
   python3 index_images.py
   ```

## Running

Start the server:
```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.
