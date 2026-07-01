"""
YOLO 옷 크롭 (LookMatch 전처리) — 필터링 + 비율유지 패딩 + 그리드 검수
======================================================================
DeepFashion2 YOLO로 옷을 탐지·크롭하면서:
  - 나쁜 크롭 자동 필터 (신뢰도/크기/잘림/종횡비)
  - 비율 유지 패딩 (바지 111x360 왜곡 방지)
  - 캡션 부위 매칭 (학습용: "니트"면 상의만)
  - 검수용 그리드 이미지 생성 (샘플 눈으로 확인)

모델: Bingsu/adetailer 의 deepfashion2_yolov8s-seg.pt

설치: pip install ultralytics huggingface_hub pillow pandas tqdm
사용:
  process_for_training()  # 학습 데이터 크롭 (부위 매칭)
  crop_for_inference()    # 추론: 이미지 1장의 모든 옷 크롭
"""

import math
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ============================================================
# 설정
# ============================================================
CONF_THRESHOLD = 0.5      # 신뢰도 임계값 (이하 제외)
MIN_SIZE = 80             # 최소 크롭 크기(px, 가로·세로) (이하 제외)
CUT_MARGIN = 5            # 경계 접촉 판정 여백(px)
CUT_MAX_TOUCH = 2         # 경계 몇 변 이상 붙으면 "잘림"으로 제외
MAX_RATIO = 4.0           # 종횡비 상한 (이상이면 제외) - 바지 고려해 여유
PAD_SIZE = 512            # 패딩 후 정사각형 크기
PAD_COLOR = (255, 255, 255)   # 패딩 색 (흰색)
BATCH = 16                # YOLO 배치 크기

# DeepFashion2 13개 클래스 → 부위
DF2_CLASSES = {
    0: "short_sleeve_top", 1: "long_sleeve_top", 2: "short_sleeve_outwear",
    3: "long_sleeve_outwear", 4: "vest", 5: "sling", 6: "shorts",
    7: "trousers", 8: "skirt", 9: "short_sleeve_dress",
    10: "long_sleeve_dress", 11: "vest_dress", 12: "sling_dress",
}
CLASS_TO_PART = {
    0: "upper", 1: "upper", 4: "upper", 5: "upper",
    2: "outer", 3: "outer",
    6: "lower", 7: "lower", 8: "lower",
    9: "dress", 10: "dress", 11: "dress", 12: "dress",
}

# 네이버 키워드 → 부위
UPPER_KW = ["니트","맨투맨","셔츠","블라우스","후드티","티셔츠","카라티","피케","나시","민소매","크롭티","조끼","베스트"]
LOWER_KW = ["청바지","슬랙스","치마","스커트","반바지","레깅스","팬츠","조거","큐롯","진","바지"]
OUTER_KW = ["코트","패딩","자켓","가디건","블레이저","집업","바람막이","야상","무스탕","플리스","후리스","점퍼"]
DRESS_KW = ["원피스","점프수트","투피스","셋업","정장세트","정장"]

def keyword_to_part(keyword):
    # 구체 키워드(정장바지/정장자켓/정장치마)를 세트(정장)보다 먼저 매칭
    if "정장바지" in keyword or "정장치마" in keyword:
        return "lower"
    if "정장자켓" in keyword:
        return "upper"
    # 일반 매칭: 하의 > 원피스/세트 > 아우터 > 상의
    if any(k in keyword for k in LOWER_KW): return "lower"
    if any(k in keyword for k in DRESS_KW): return "dress"
    if any(k in keyword for k in OUTER_KW): return "outer"
    if any(k in keyword for k in UPPER_KW): return "upper"
    return "unknown"


# ============================================================
# 모델 로드
# ============================================================
_model = None
def get_model():
    global _model
    if _model is None:
        path = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")
        _model = YOLO(path)
    return _model


# ============================================================
# 나쁜 크롭 필터
# ============================================================
def is_cut_off(x1, y1, x2, y2, img_w, img_h):
    """경계에 여러 변이 붙으면 잘린 이미지로 판단"""
    touch = 0
    if x1 <= CUT_MARGIN: touch += 1
    if y1 <= CUT_MARGIN: touch += 1
    if x2 >= img_w - CUT_MARGIN: touch += 1
    if y2 >= img_h - CUT_MARGIN: touch += 1
    return touch >= CUT_MAX_TOUCH

def passes_filter(x1, y1, x2, y2, conf, img_w, img_h, part=None):
    """크롭 품질 필터. 통과하면 True, 걸러지면 (False, 사유)"""
    w, h = x2 - x1, y2 - y1
    if conf < CONF_THRESHOLD:
        return False, "low_conf"
    if w < MIN_SIZE or h < MIN_SIZE:
        return False, "too_small"
    if is_cut_off(x1, y1, x2, y2, img_w, img_h):
        return False, "cut_off"
    ratio = max(w, h) / max(min(w, h), 1)
    # 하의(바지)는 원래 길쭉하니 상한 완화
    limit = MAX_RATIO + 1.0 if part == "lower" else MAX_RATIO
    if ratio > limit:
        return False, "bad_ratio"
    return True, "ok"


# ============================================================
# 비율 유지 패딩
# ============================================================
def pad_to_square(img, size=PAD_SIZE, fill=PAD_COLOR, max_ratio=MAX_RATIO):
    """비율 유지하며 정사각형으로. 극단적 비율은 중앙크롭으로 완화."""
    w, h = img.size
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > max_ratio:
        if h > w:
            new_h = int(w * max_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        else:
            new_w = int(h * max_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
    img = img.copy()
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


# ============================================================
# 탐지 → 박스 추출 (필터 적용)
# ============================================================
def detect(image_path):
    """이미지에서 옷 박스 추출 → [{part,class,xyxy,conf,area}, ...] (필터 통과분만)"""
    model = get_model()
    results = model(image_path, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]
    img_w, img_h = r.orig_shape[1], r.orig_shape[0]
    boxes = []
    for box in r.boxes:
        cls_id = int(box.cls)
        part = CLASS_TO_PART.get(cls_id, "unknown")
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf)
        ok, reason = passes_filter(x1, y1, x2, y2, conf, img_w, img_h, part)
        if not ok:
            continue
        boxes.append({
            "part": part, "class": DF2_CLASSES.get(cls_id, "?"),
            "xyxy": (x1, y1, x2, y2), "conf": conf,
            "area": (x2 - x1) * (y2 - y1),
        })
    return boxes


# ============================================================
# [학습용] 캡션 부위에 맞는 크롭
# ============================================================
def process_for_training(meta_csv="clothes_dataset/metadata.csv",
                         crop_dir="clothes_dataset/crops",
                         out_csv="clothes_dataset/metadata_cropped.csv",
                         grid_dir="clothes_dataset/grid_check"):
    """
    수집 데이터를 캡션 부위에 맞게 크롭 + 패딩.
    성공한 것만 새 CSV로. 검수용 그리드 이미지도 생성.
    """
    df = pd.read_csv(meta_csv)
    crop_dir = Path(crop_dir); crop_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = Path(grid_dir); grid_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    reasons = {"no_detect": 0, "no_match": 0}
    sample_crops = []   # 그리드용

    for _, row in tqdm(df.iterrows(), total=len(df), desc="학습 크롭"):
        img_path = row["image_path"]
        keyword = str(row.get("keyword", ""))
        if not Path(img_path).exists():
            continue

        target = keyword_to_part(keyword)
        boxes = detect(img_path)
        if not boxes:
            reasons["no_detect"] += 1
            continue

        # 부위 매칭
        if target == "unknown":
            cands = boxes
        else:
            cands = [b for b in boxes if b["part"] == target]
            if not cands and target == "dress":
                cands = [b for b in boxes if b["part"] in ("upper", "outer")]
        if not cands:
            reasons["no_match"] += 1
            continue

        # 가장 크고 신뢰도 높은 박스
        best = max(cands, key=lambda b: b["conf"] * b["area"])
        x1, y1, x2, y2 = best["xyxy"]

        img = Image.open(img_path).convert("RGB")
        crop = img.crop((x1, y1, x2, y2))
        crop = pad_to_square(crop)

        crop_name = Path(img_path).stem + "_crop.jpg"
        crop_path = crop_dir / crop_name
        crop.save(crop_path)

        new_row = row.to_dict()
        new_row["crop_path"] = str(crop_path)
        new_row["crop_part"] = best["part"]
        new_row["crop_conf"] = round(best["conf"], 3)
        kept.append(new_row)
        if len(sample_crops) < 100:
            sample_crops.append(crop_path)

    pd.DataFrame(kept).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 검수용 그리드 이미지
    if sample_crops:
        make_grid(sample_crops, grid_dir / "sample_grid.jpg")

    print(f"\n크롭 성공: {len(kept)}개")
    print(f"탐지 실패: {reasons['no_detect']}개 / 부위 매칭 실패: {reasons['no_match']}개")
    print(f"저장: {out_csv}")
    print(f"검수 그리드: {grid_dir}/sample_grid.jpg (눈으로 100장 확인)")


# ============================================================
# [추론용] 이미지 1장의 모든 옷 크롭
# ============================================================
def crop_for_inference(image_path, out_dir="infer_crops"):
    """추론: 이미지의 모든 옷을 크롭+패딩. [(crop_path, part, class, conf), ...]"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    boxes = detect(image_path)
    img = Image.open(image_path).convert("RGB")
    out = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["xyxy"]
        crop = pad_to_square(img.crop((x1, y1, x2, y2)))
        cpath = out_dir / f"crop_{i}_{b['part']}_{b['class']}.jpg"
        crop.save(cpath)
        out.append((str(cpath), b["part"], b["class"], round(b["conf"], 3)))
    return out


# ============================================================
# 검수용 그리드 이미지 (100장을 한 장으로)
# ============================================================
def make_grid(crop_paths, out_path, cols=10, thumb=128):
    n = len(crop_paths)
    rows = math.ceil(n / cols)
    grid = Image.new("RGB", (cols * thumb, rows * thumb), (240, 240, 240))
    for idx, p in enumerate(crop_paths):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb))
            r, c = divmod(idx, cols)
            grid.paste(im, (c * thumb, r * thumb))
        except Exception:
            continue
    grid.save(out_path)


if __name__ == "__main__":
    # [학습] 수집 데이터 크롭
    process_for_training()

    # [추론 테스트] 이미지 1장
    # crops = crop_for_inference("test.jpg")
    # for cpath, part, cls, conf in crops:
    #     print(f"{part:8s} {cls:20s} conf={conf} → {cpath}")
