"""
Downloads pre-built ChromaDB from GitHub Releases
"""
import os
import requests
import tarfile

CHROMADB_URL = "https://github.com/ihateusernamesfuckthis/kommunal-rag-agent/releases/download/v1.0.0/chroma_data.tar.gz"
CHROMADB_PATH = "./chroma_data"

def download_chromadb():
    """Downloads and extracts ChromaDB if it doesn't exist"""

    # Check if ChromaDB already exists and has data
    if os.path.exists(CHROMADB_PATH) and os.listdir(CHROMADB_PATH):
        print(f"✅ ChromaDB already exists at {CHROMADB_PATH}, skipping download")
        return

    print("📦 Downloading pre-built ChromaDB from GitHub Releases...")

    # Download the archive
    response = requests.get(CHROMADB_URL, stream=True)
    response.raise_for_status()

    archive_path = "chroma_data.tar.gz"

    # Save to disk with progress
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(archive_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"⏳ Download progress: {progress:.1f}%", end='\r')

    print("\n📂 Extracting ChromaDB...")

    # Extract the archive
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall('.')

    # Clean up archive
    os.remove(archive_path)

    print("✅ ChromaDB downloaded and extracted successfully!")

if __name__ == "__main__":
    download_chromadb()
