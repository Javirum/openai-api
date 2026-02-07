# Install: pip install datasets
from datasets import load_dataset
import requests
from PIL import Image
import pandas as pd
from pathlib import Path
import base64
from io import BytesIO


def image_to_base64(image, format="JPEG"):
    """Convert a PIL Image to base64 string for API transmission."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_base64_data_url(image, format="JPEG"):
    """Convert PIL Image to a data URL for OpenAI API."""
    mime_type = f"image/{format.lower()}"
    b64_string = image_to_base64(image, format)
    return f"data:{mime_type};base64,{b64_string}"


# Load dataset from HuggingFace
print("Loading product dataset...")
try:
    # Try loading the dataset
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:5]")  # First 100 samples
    print(f"✓ Loaded {len(dataset)} products")
    
    # Convert to pandas for easier manipulation
    products_df = pd.DataFrame(dataset)
    print(f"Dataset columns: {products_df.columns.tolist()}")
    
except Exception as e:
    print(f"⚠ Could not load HuggingFace dataset: {e}")
    print("Using local images instead...")
    
    # Alternative: Use local images
    # Create a products.json file with product information
    products_data = [
        {
            "id": 1,
            "name": "Wireless Headphones",
            "price": 79.99,
            "category": "Electronics",
            "image_path": "images/product1.jpg"
        },
        # Add more products...
    ]
    
    products_df = pd.DataFrame(products_data)

# Create images directory
images_dir = Path("product_images")
images_dir.mkdir(exist_ok=True)

print("Downloading product images...", products_df.sort_index())

print(f"\n✓ Dataset prepared!")
print(f"  Total products: {len(products_df)}")

# Convert images to base64 for API transmission
print("\nConverting images to base64...")
base64_images = []

for idx, row in products_df.iterrows():
    if "image" in row and row["image"] is not None:
        img = row["image"]
        b64_url = get_base64_data_url(img)
        base64_images.append(b64_url)

        if idx < 3:  # Show first 3 as examples
            print(f"  Product {idx}: {len(b64_url)} chars")

products_df["image_base64"] = base64_images
print(f"✓ Converted {len(base64_images)} images to base64")
