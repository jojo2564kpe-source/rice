import os, json, base64, datetime
import cv2, numpy as np, pandas as pd, joblib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from features import extract_features, validate_image

MODEL_PATH  = "rice_model.pkl"
LOG_DIR     = "user_uploads"
CONF_THRESH = 0.50
CLASS_NAMES = {0: "ไม่พบแผล (ปกติ)", 1: "แผลขนาดเล็ก",
               2: "แผลขนาดปานกลาง", 3: "แผลขนาดใหญ่ / รุนแรงมาก"}
ADVICE = {
    0: "ใบข้าวอยู่ในสภาพปกติ ควรเฝ้าระวังและตรวจแปลงอย่างสม่ำเสมอ",
    1: "พบการติดเชื้อระยะเริ่มต้น ควรลดการใส่ปุ๋ยไนโตรเจนและระบายน้ำในแปลง",
    2: "การระบาดอยู่ในระดับปานกลาง ควรพิจารณาใช้สารป้องกันกำจัดเชื้อราตามคำแนะนำ",
    3: "การระบาดรุนแรง ควรเข้าควบคุมโดยเร่งด่วนและปรึกษาเจ้าหน้าที่เกษตรในพื้นที่",
}

app = FastAPI(title="ระบบตรวจวัดระดับความรุนแรงโรคไหม้ในใบข้าว")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
os.makedirs(LOG_DIR, exist_ok=True)

BUNDLE = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def to_b64(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    info = {}
    if BUNDLE:
        info = {"test": BUNDLE.get("test_score", 0), "cv": BUNDLE.get("cv_mean", 0)}
    return templates.TemplateResponse(request, "index.html", {"info": info})



@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...), true_label: str = Form("ไม่ระบุ")):
    if BUNDLE is None:
        return JSONResponse({"error": "ยังไม่มีไฟล์โมเดล rice_model.pkl"}, 500)

    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "ไม่สามารถอ่านไฟล์ภาพได้"}, 400)

    warns, fatal = validate_image(img)
    if fatal:
        return JSONResponse({"error": "ภาพนี้ไม่ใช่รูปใบข้าว กรุณาอัปโหลดใหม่",
                             "warnings": warns}, 422)

    feats, mask, vis = extract_features(img, return_debug=True)
    X = pd.DataFrame([feats]).fillna(0).reindex(columns=BUNDLE["cols"], fill_value=0)

    model = BUNDLE["model"]
    proba = model.predict_proba(X)[0]
    classes = [int(c) for c in model.classes_]
    idx = int(np.argmax(proba))
    level, conf = classes[idx], float(proba[idx])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(os.path.join(LOG_DIR, f"{ts}.jpg"), img)
    with open(os.path.join(LOG_DIR, "log.jsonl"), "a", encoding="utf-8") as fp:
        fp.write(json.dumps({"ts": ts, "pred": level, "conf": round(conf, 4),
                             "user_label": true_label}, ensure_ascii=False) + "\n")

    if conf < CONF_THRESH:
        warns.append(f"ความมั่นใจต่ำกว่า {CONF_THRESH:.0%} ผลลัพธ์อาจคลาดเคลื่อน")

    return {
        "level": level,
        "name": CLASS_NAMES.get(level, "-"),
        "advice": ADVICE.get(level, ""),
        "confidence": conf,
        "reliable": conf >= CONF_THRESH and not warns,
        "proba": [{"level": c, "name": CLASS_NAMES.get(c, str(c)), "p": float(p)}
                  for c, p in zip(classes, proba)],
        "lesion_ratio": feats["lesion_ratio"],
        "spot_density": feats["spot_density"],
        "warnings": warns,
        "vis": to_b64(vis),
        "mask": to_b64(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)