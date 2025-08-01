#!/usr/bin/env python3
"""
Script to upload images to Railway and trigger reindexing
"""

import requests
import os
from pathlib import Path
import time

def upload_images_to_railway():
    """Upload images to Railway and trigger reindexing"""
    
    railway_url = "https://esorusid-production.up.railway.app"
    
    print("🚀 Uploading images to Railway...")
    
    # Check if app is running
    try:
        health_response = requests.get(f"{railway_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Railway app is running")
        else:
            print("❌ Railway app is not responding properly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Railway: {e}")
        return
    
    # Check if ML is available
    health_data = health_response.json()
    if not health_data.get('ml_available', False):
        print("⚠️ ML dependencies not available yet. Waiting for deployment...")
        print("Please wait a few more minutes for the ML deployment to complete.")
        return
    
    print("✅ ML dependencies are available")
    
    # Trigger reindex
    print("🔄 Triggering reindex...")
    try:
        reindex_response = requests.get(f"{railway_url}/reindex", timeout=300)  # 5 minutes timeout
        if reindex_response.status_code == 200:
            result = reindex_response.json()
            if result.get('status') == 'success':
                print("✅ Reindex completed successfully!")
                print(f"📊 {result.get('message', '')}")
            else:
                print(f"❌ Reindex failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ Reindex request failed with status {reindex_response.status_code}")
    except Exception as e:
        print(f"❌ Error during reindex: {e}")

def manual_upload_instructions():
    """Print manual upload instructions"""
    print("\n📋 Manual Upload Instructions:")
    print("=" * 50)
    print("1. Go to: https://esorusid-production.up.railway.app/images")
    print("2. Upload your images using the web interface")
    print("3. Wait for the ML deployment to complete")
    print("4. Visit: https://esorusid-production.up.railway.app/reindex")
    print("5. This will process all uploaded images and create the search index")
    print("\n💡 Tip: You can also upload images directly to the home page")
    print("   and they will be processed automatically during search.")

if __name__ == "__main__":
    print("🎯 Railway Image Upload Tool")
    print("=" * 30)
    
    # Try automatic upload
    upload_images_to_railway()
    
    # Always show manual instructions
    manual_upload_instructions() 