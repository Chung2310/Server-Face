import requests
import json
import os
import sys
import io

BASE_URL = "http://127.0.0.1:8000/api/v1"
IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
FACE_API_KEY = os.environ.get("FACE_API_KEY", "")

def main():
    # Force stdout to use UTF-8 to prevent cp1252 rendering errors on Windows.
    # Kept inside main() so pytest collection does not break capture.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== STARTING INTEGRATION TEST ===")
    
    # 1. Download sample face image
    print(f"Downloading sample image from: {IMAGE_URL}...")
    img_response = requests.get(IMAGE_URL)
    if img_response.status_code != 200:
        print("Failed to download sample image.")
        return
        
    image_bytes = img_response.content
    print(f"Downloaded {len(image_bytes)} bytes.")

    # 2. Test Detect endpoint
    print("\nTesting `/face/detect` endpoint...")
    files = {"file": ("lena.jpg", image_bytes, "image/jpeg")}
    detect_res = requests.post(f"{BASE_URL}/face/detect", files=files)
    print(f"Status Code: {detect_res.status_code}")
    if detect_res.status_code == 200:
        faces = detect_res.json()
        print(f"Detected {len(faces)} face(s).")
        for idx, face in enumerate(faces):
            bbox = face['bbox']
            print(f"  Face {idx+1}: Box [{bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}], Confidence: {face['confidence']:.2f}, Age: {face['age']:.1f}, Gender: {face['gender']}")
    else:
        print(f"Error: {detect_res.text}")
        return

    # 3. Test Embedding endpoint
    print("\nTesting `/face/embedding` endpoint...")
    files = {"file": ("lena.jpg", image_bytes, "image/jpeg")}
    emb_res = requests.post(f"{BASE_URL}/face/embedding", files=files, params={"largest_only": True})
    print(f"Status Code: {emb_res.status_code}")
    if emb_res.status_code == 200:
        embeddings = emb_res.json()
        embedding = embeddings[0]["embedding"]
        print(f"Successfully extracted embedding vector. Dimension: {len(embedding)}")
        print(f"First 5 elements: {embedding[:5]}")
    else:
        print(f"Error: {emb_res.text}")
        return

    # 4. Test Register endpoint (multipart image upload, X-API-Key protected)
    print("\nTesting `/face/register` endpoint...")
    files = {"file": ("lena.jpg", image_bytes, "image/jpeg")}
    reg_res = requests.post(
        f"{BASE_URL}/face/register",
        data={"user_id": "lena_test_user"},
        files=files,
        headers={"X-API-Key": FACE_API_KEY},
    )
    print(f"Status Code: {reg_res.status_code}")
    print(f"Response: {reg_res.json()}")

    # 4b. Registration status lookup
    status_res = requests.get(
        f"{BASE_URL}/face/register/lena_test_user",
        headers={"X-API-Key": FACE_API_KEY},
    )
    print(f"Status lookup: {status_res.status_code} {status_res.json() if status_res.ok else status_res.text}")

    # 5. Test Search endpoint
    print("\nTesting `/face/search` endpoint...")
    search_payload = {
        "embedding": embedding,
        "limit": 3
    }
    search_res = requests.post(f"{BASE_URL}/face/search", json=search_payload)
    print(f"Status Code: {search_res.status_code}")
    if search_res.status_code == 200:
        matches = search_res.json()["matches"]
        print(f"Found {len(matches)} match(es):")
        for match in matches:
            print(f"  User ID: {match['user_id']}, Similarity: {match['similarity']:.4f}")
    else:
        print(f"Error: {search_res.text}")

    # 6. Test Verify Employee endpoint
    print("\nTesting `/face/verify-employee` endpoint...")
    files = {"file": ("lena.jpg", image_bytes, "image/jpeg")}
    verify_payload = {"user_id": "lena_test_user"}
    verify_res = requests.post(f"{BASE_URL}/face/verify-employee", data=verify_payload, files=files)
    print(f"Status Code: {verify_res.status_code}")
    if verify_res.status_code == 200:
        res_data = verify_res.json()
        print(f"Verified: {res_data['verified']}")
        print(f"Reason: {res_data['reason']}")
        if res_data['similarity'] is not None:
            print(f"Similarity: {res_data['similarity']:.4f}")
    else:
        print(f"Error: {verify_res.text}")

if __name__ == "__main__":
    main()
