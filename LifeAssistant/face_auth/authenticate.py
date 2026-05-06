# face_auth/authenticate.py
# Optimized — model loaded ONCE at startup, cached globally

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from typing import Optional
import time
import base64
from config import FACES_DIR, FACE_TOLERANCE

# ── Load DeepFace model ONCE at module import ─────────────
# This is the key optimization — instead of loading FaceNet
# every time a frame arrives (2-3 sec each), we load it once
# when the server starts and keep it in memory.
_deepface_model = None
_registered_faces = None
_faces_loaded_at = 0

def get_model():
    """Load DeepFace model once and cache it."""
    global _deepface_model
    if _deepface_model is None:
        print("🔄 Loading FaceNet model (one-time, ~3 seconds)...")
        from deepface import DeepFace
        # Warm up by running a dummy image — forces model load
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                dummy,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv"
            )
        except Exception:
            pass
        _deepface_model = True  # flag that model is loaded
        print("✅ FaceNet model ready (cached for all future requests)")
    return True


def load_registered_faces(force_reload: bool = False) -> dict:
    """Load face encodings, cache them in memory."""
    global _registered_faces, _faces_loaded_at
    # Reload if empty or forced or older than 60 seconds
    if _registered_faces is None or force_reload or (time.time() - _faces_loaded_at > 60):
        _registered_faces = {}
        for face_file in FACES_DIR.glob("*.npy"):
            username = face_file.stem
            _registered_faces[username] = np.load(face_file)
        _faces_loaded_at = time.time()
        print(f"👤 Loaded {len(_registered_faces)} registered face(s): {list(_registered_faces.keys())}")
    return _registered_faces


def get_embedding(frame_bgr):
    """Get face embedding — model already loaded, so this is fast (<200ms)."""
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
    except Exception:
        return None


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))


def authenticate_from_image(image_base64: str) -> Optional[str]:
    """
    Fast authentication from browser image.
    Model is pre-loaded so each call takes ~200-400ms instead of 3s.
    """
    # Ensure model is loaded (instant if already cached)
    get_model()
    registered_faces = load_registered_faces()

    if not registered_faces:
        print("⚠️  No registered faces found.")
        return None

    # Decode base64 image
    try:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        img_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
    except Exception as e:
        print(f"❌ Image decode error: {e}")
        return None

    # Get embedding (fast — model already in memory)
    t0 = time.time()
    emb = get_embedding(frame)
    print(f"   Embedding time: {(time.time()-t0)*1000:.0f}ms")

    if emb is None:
        return None

    # Compare against registered faces
    SIMILARITY_THRESHOLD = 0.70
    best_match, best_score = None, 0.0

    for uname, reg_emb in registered_faces.items():
        score = cosine_similarity(emb, reg_emb)
        print(f"   '{uname}': similarity={score:.3f}")
        if score > best_score:
            best_score = score
            best_match = uname

    if best_score >= SIMILARITY_THRESHOLD:
        print(f"✅ Authenticated: {best_match} ({best_score:.3f}) in {(time.time()-t0)*1000:.0f}ms")
        return best_match

    print(f"❌ No match. Best: {best_score:.3f}")
    return None


def authenticate(timeout_seconds: int = 20) -> Optional[str]:
    """Legacy webcam auth — kept as fallback."""
    get_model()
    registered_faces = load_registered_faces()
    if not registered_faces:
        return None

    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        cap.release()

    if not cap or not cap.isOpened():
        return None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    start_time = time.time()
    matched_user = None
    SIMILARITY_THRESHOLD = 0.70

    while time.time() - start_time < timeout_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        remaining = int(timeout_seconds - (time.time() - start_time))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            emb = get_embedding(frame)
            if emb is not None:
                best_match, best_score = None, 0.0
                for uname, reg_emb in registered_faces.items():
                    score = cosine_similarity(emb, reg_emb)
                    if score > best_score:
                        best_score = score
                        best_match = uname
                if best_score >= SIMILARITY_THRESHOLD:
                    matched_user = best_match
                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)
                else:
                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,0,255), 2)

        cv2.putText(display, f"Authenticating... {remaining}s",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.imshow("Authentication", display)
        cv2.waitKey(1)
        if matched_user:
            time.sleep(1)
            break

    cap.release()
    cv2.destroyAllWindows()
    return matched_user