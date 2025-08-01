#!/usr/bin/env python3
"""
Script to upload images to Railway deployment
"""
import os
import shutil
from pathlib import Path
import requests
import json

def upload_images_to_railway():
    """Upload images to Railway using the reindex endpoint"""
    print("🚀 Uploading images to Railway...")
    
    # Railway app URL
    railway_url = "https://esorusid-production.up.railway.app"
    
    try:
        # Call the reindex endpoint to generate images and index
        print("📞 Calling reindex endpoint...")
        response = requests.get(f"{railway_url}/reindex", timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            print(f"🌐 Check your app at: {railway_url}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. The indexing process may still be running.")
        print(f"🌐 Check your app at: {railway_url}")
    except Exception as e:
        print(f"❌ Error uploading to Railway: {e}")

def copy_images_to_railway_storage():
    """Copy images to Railway persistent storage (if using Railway CLI)"""
    print("📁 Copying images to Railway storage...")
    
    # Source directory (local)
    source_dir = Path("static/recommendations")
    
    # Check if Railway storage is available
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Count images
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
    print(f"📸 Found {len(image_files)} images to upload")
    
    if not image_files:
        print("❌ No images found to upload")
        return
    
    print("💡 To upload images to Railway, you can:")
    print("1. Use Railway CLI: railway up")
    print("2. Use the reindex endpoint: curl https://esorusid-production.up.railway.app/reindex")
    print("3. Upload via Railway dashboard")

if __name__ == "__main__":
    print("🎯 Railway Image Upload Tool")
    print("=" * 40)
    
    # Try to upload via API
    upload_images_to_railway()
    
    print("\n" + "=" * 40)
    # Show manual options
    copy_images_to_railway_storage() 