# face_auth/register_face_api.py
# Optimized — reuses cached model

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import base64
from config import FACES_DIR


def get_embedding(frame_bgr):
    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            frame_bgr,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )
        if result and len(result) > 0:
            arr = np.array(result[0]["embedding"])
            if np.linalg.norm(arr) > 1.0:
                return arr
        return None
    except Exception as e:
        print(f"DeepFace error: {e}")
        return None


def register_from_images(username: str, images_base64: list) -> dict:
    """Register face from browser-captured images."""
    from face_auth.authenticate import get_model, load_registered_faces
    get_model()  # ensure model is loaded

    username = username.strip().lower()
    save_path = FACES_DIR / f"{username}.npy"
    embeddings = []

    for i, img_b64 in enumerate(images_base64):
        try:
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            emb = get_embedding(frame)
            if emb is not None:
                embeddings.append(emb)
                print(f"  ✓ Sample {len(embeddings)}/{len(images_base64)}")
        except Exception as e:
            print(f"  Frame {i} error: {e}")

    if len(embeddings) < 3:
        return {
            "success": False,
            "samples": len(embeddings),
            "message": f"Only {len(embeddings)} valid samples. Need at least 3. Try better lighting."
        }

    mean_embedding = np.mean(embeddings, axis=0)
    np.save(save_path, mean_embedding)

    # Force reload faces cache
    load_registered_faces(force_reload=True)

    return {
        "success": True,
        "samples": len(embeddings),
        "message": f"Registered with {len(embeddings)} samples."
    }