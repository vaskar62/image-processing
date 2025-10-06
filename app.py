import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ----------------------------------------------------------
# App setup
# ----------------------------------------------------------
app = FastAPI(title="CleanVision – Blur Detector")

# Ensure upload folder exists
os.makedirs("uploads", exist_ok=True)

# Mount folders
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# ----------------------------------------------------------
# Threshold (adjust this)
# ----------------------------------------------------------
# Lower = more strict (more images will be marked blurry)
BLUR_THRESHOLD = 12000.0



def detect_blur(image_path: str):
    """Detect if an image is blurry using Laplacian + Tenengrad metrics."""
    img = cv2.imread(image_path)
    if img is None:
        return True, 0.0

    # Convert to grayscale and normalize lighting
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Laplacian variance (focus/sharpness measure)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Tenengrad method (edge clarity)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(gx ** 2 + gy ** 2)

    # Weighted average (gives smoother detection)
    combined_score = (lap_var * 0.6) + (tenengrad * 0.4)

    print(f"DEBUG → Laplacian: {lap_var:.2f}, Tenengrad: {tenengrad:.2f}, Combined: {combined_score:.2f}")

    # Determine if blurry
    is_blurry = combined_score < BLUR_THRESHOLD
    return is_blurry, combined_score


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def main_page(request: Request):
    """Render main upload page."""
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_url": None})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    """Handle upload and analyze the image."""
    file_path = os.path.join("uploads", file.filename)

    # Save uploaded image
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Detect blur
    is_blurry, score = detect_blur(file_path)

    # Build result
    result = {
        "status": "❌ FAIL – Blurry Image" if is_blurry else "✅ PASS – Sharp Image",
        "reason": "Image appears blurry or out of focus." if is_blurry else "Image is sharp and well-focused.",
        "sharpness_score": round(score, 2),
        "confidence": f"{min(round(score / BLUR_THRESHOLD * 100, 1), 100)}%",
        "is_blurry": is_blurry,
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image_url": f"/uploads/{file.filename}",
        },
    )


# ----------------------------------------------------------
# Run (example)
# ----------------------------------------------------------
# uvicorn app:app --reload --port 8005
