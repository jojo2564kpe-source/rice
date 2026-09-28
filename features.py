import cv2, numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

TARGET_W = 1024

def imread_unicode(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def gray_world_wb(img):
    b, g, r = cv2.split(img.astype(np.float32))
    k = (b.mean() + g.mean() + r.mean()) / 3.0
    b *= k / (b.mean() + 1e-6); g *= k / (g.mean() + 1e-6); r *= k / (r.mean() + 1e-6)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def preprocess(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (TARGET_W, max(1, int(h * TARGET_W / w))), interpolation=cv2.INTER_AREA)
    img = gray_world_wb(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, bb]), cv2.COLOR_LAB2BGR)

def get_leaf_mask(img):
    b, g, r = cv2.split(img.astype(np.float32))
    s = b + g + r + 1e-6
    exg = 2 * (g / s) - (r / s) - (b / s)
    exr = 1.4 * (r / s) - (g / s)
    score = np.maximum(exg, exr * 0.9)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        mask = np.uint8(lab_ == big) * 255
    if mask.sum() == 0:
        mask = np.full(mask.shape, 255, np.uint8)
    return mask

def extract_features(image, return_debug=False):
    img  = preprocess(image)
    mask = get_leaf_mask(img)
    px   = mask > 0
    leaf_area = float(px.sum()) + 1e-6
    f = {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    for name, ch in [('h', hsv[:,:,0]), ('s', hsv[:,:,1]), ('v', hsv[:,:,2]),
                     ('a', lab[:,:,1]), ('b', lab[:,:,2])]:
        v = ch[px].astype(np.float32)
        f[f'{name}_mean'] = float(v.mean()); f[f'{name}_std'] = float(v.std())
        for p in (10, 25, 50, 75, 90):
            f[f'{name}_p{p}'] = float(np.percentile(v, p))
        f[f'{name}_skew'] = float(((v - v.mean())**3).mean() / (v.std()**3 + 1e-6))

    h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    healthy = ((h_ > 35) & (h_ < 85)) & px
    lesion  = (~healthy) & px
    f['healthy_ratio']   = float(healthy.sum() / leaf_area)
    f['lesion_ratio']    = float(lesion.sum() / leaf_area)
    f['chlorosis_ratio'] = float((((h_ >= 20) & (h_ <= 35)) & px).sum() / leaf_area)
    f['necrosis_ratio']  = float((((h_ < 20) | (h_ > 160)) & (v_ < 140) & px).sum() / leaf_area)
    f['lowsat_ratio']    = float(((s_ < 60) & px).sum() / leaf_area)

    lm = cv2.morphologyEx(np.uint8(lesion) * 255, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(lm, 8)
    areas = np.sort(st[1:, cv2.CC_STAT_AREA])[::-1] if n > 1 else np.array([], dtype=int)
    areas = areas[areas >= 20]
    f['spot_density'] = float(len(areas) / leaf_area * 1e4)
    f['spot_max']  = float(areas[0] / leaf_area)        if len(areas) else 0.0
    f['spot_mean'] = float(areas.mean() / leaf_area)    if len(areas) else 0.0
    f['spot_top3'] = float(areas[:3].sum() / leaf_area) if len(areas) else 0.0
    f['spot_std']  = float(areas.std() / leaf_area)     if len(areas) else 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g8 = (gray // 32).astype(np.uint8); g8[~px] = 0
    glcm = graycomatrix(g8, distances=[1,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    for prop in ['contrast','homogeneity','energy','correlation','dissimilarity']:
        vals = graycoprops(glcm, prop)
        f[f'glcm_{prop}_mean'] = float(vals.mean())
        f[f'glcm_{prop}_std']  = float(vals.std())

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp[px], bins=10, range=(0,10), density=True)
    for i, val in enumerate(hist):
        f[f'lbp_{i}'] = float(val)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True) + 1e-6
        f['leaf_solidity'] = float(cv2.contourArea(c) / (cv2.contourArea(cv2.convexHull(c)) + 1e-6))
        f['leaf_circular'] = float(4 * np.pi * cv2.contourArea(c) / (peri ** 2))
    else:
        f['leaf_solidity'] = f['leaf_circular'] = 0.0

    if return_debug:
        ov = img.copy(); ov[lesion] = (0, 0, 255)
        vis = cv2.addWeighted(img, 0.65, ov, 0.35, 0)
        if cnts:
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
        return f, mask, vis
    return f

def validate_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warns, fatal = [], False
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 60:
        warns.append(f"ภาพอาจเบลอ (ค่าความคมชัด {blur:.0f}) ควรถ่ายใหม่ให้โฟกัสชัด")
    b = gray.mean()
    if b < 40:    warns.append("ภาพมืดเกินไป")
    elif b > 220: warns.append("ภาพสว่างจ้าเกินไป")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plant = (((hsv[:,:,0] > 10) & (hsv[:,:,0] < 95)) & (hsv[:,:,1] > 40)).mean()
    if plant < 0.03:
        warns.append("ไม่พบพื้นที่ที่มีลักษณะเป็นใบพืช — อาจไม่ใช่รูปใบข้าว")
        fatal = True
    return warns, fatal