"""
네이버 쇼핑 API 일일 수집 (옷 전용) — 매일 25,000개씩 중복 없이 누적
====================================================================
오늘 25,000개 수집 → 내일 실행하면 어제 안 받은 것부터 또 25,000개.
며칠 반복하면 중복 없이 데이터가 쌓임.

수집 항목 (핵심 3개 + 메타):
  ★ image_url / image_path : 옷 사진
  ★ caption                : 상품명 (클렌징)
  ★ product_link           : 상품 URL
    brand / price / mall    : 메타 (나중 필터용)

중복 방지 3단:
  1. progress.json : 키워드별 어디까지 받았나 → 이어받기
  2. seen_ids.json : 받은 productId 누적 → 같은 상품 skip
  3. pHash         : 시각적 중복 제거

상태 파일(자동 생성/갱신, 절대 삭제 X):
  clothes_dataset/seen_ids.json
  clothes_dataset/progress.json
  clothes_dataset/seen_phash.json
  clothes_dataset/metadata.csv   ← 결과 (매일 append)

설치: pip install requests pillow imagehash
실행: python collect_daily.py     (매일 1회)
"""

import os
import re
import csv
import json
import time
import random
import hashlib
import requests
from pathlib import Path
from PIL import Image
import imagehash

# ============================================================
# 설정
# ============================================================
CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "여기에_CLIENT_ID")
CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "여기에_CLIENT_SECRET")

SEARCH_URL = "https://openapi.naver.com/v1/search/shop.json"

DAILY_LIMIT = 25000      # 하루 수집 목표
DISPLAY = 100            # API 1회 호출당 (최대 100)
MAX_START = 1000         # 네이버 start 최대 (키워드당 최대 1000개)
SORT = "sim"
PHASH_THRESHOLD = 5      # 시각적 중복 판정
SLEEP = 0.05            # 호출 간격

OUT_DIR = Path("clothes_dataset")
IMG_DIR = OUT_DIR / "images"
META_CSV = OUT_DIR / "metadata.csv"
SEEN_IDS_FILE = OUT_DIR / "seen_ids.json"
PROGRESS_FILE = OUT_DIR / "progress.json"
PHASH_FILE = OUT_DIR / "seen_phash.json"

# ============================================================
# 옷 키워드 (일상복, 131개)
# ============================================================
KEYWORDS = [
    # 상의 - 니트
    "니트", "여성니트", "남성니트", "오버사이즈니트", "크롭니트",
    "터틀넥니트", "라운드넥니트", "브이넥니트", "골지니트", "캐시미어니트",
    # 상의 - 맨투맨
    "맨투맨", "여성맨투맨", "남성맨투맨", "오버핏맨투맨", "기모맨투맨",
    # 상의 - 셔츠/블라우스
    "셔츠", "여성셔츠", "남성셔츠", "오버핏셔츠", "린넨셔츠", "체크셔츠", "옥스포드셔츠",
    "블라우스", "쉬폰블라우스", "퍼프블라우스", "프릴블라우스",
    # 상의 - 후드/티
    "후드티", "여성후드티", "남성후드티", "기모후드티",
    "티셔츠", "반팔티셔츠", "긴팔티셔츠", "여성티셔츠", "남성티셔츠",
    "카라티", "피케티", "나시", "민소매", "크롭티", "오버핏티셔츠",
    # 하의 - 팬츠
    "청바지", "여성청바지", "남성청바지", "와이드팬츠", "스키니진", "부츠컷청바지",
    "슬랙스", "여성슬랙스", "남성슬랙스", "와이드슬랙스", "치노팬츠", "코듀로이팬츠",
    "트레이닝팬츠", "조거팬츠", "코튼팬츠", "카고팬츠", "밴딩팬츠",
    # 하의 - 스커트/반바지
    "치마", "미니스커트", "롱스커트", "미디스커트", "플리츠스커트", "데님스커트", "에이라인스커트",
    "반바지", "여성반바지", "남성반바지", "버뮤다팬츠", "데님반바지",
    "레깅스", "여성레깅스", "큐롯",
    # 원피스/세트
    "원피스", "여름원피스", "롱원피스", "미니원피스", "셔츠원피스",
    "니트원피스", "플라워원피스", "데님원피스", "린넨원피스",
    "점프수트", "투피스", "셋업", "정장세트",
    # 아우터 - 코트/패딩
    "코트", "여성코트", "남성코트", "트렌치코트", "울코트", "롱코트", "더플코트", "핸드메이드코트",
    "패딩", "여성패딩", "남성패딩", "숏패딩", "롱패딩", "경량패딩", "패딩조끼", "퀼팅자켓",
    # 아우터 - 자켓/가디건
    "자켓", "여성자켓", "남성자켓", "데님자켓", "가죽자켓", "블레이저", "사파리자켓", "트러커자켓",
    "가디건", "여성가디건", "남성가디건", "롱가디건", "니트가디건",
    "후드집업", "니트집업", "바람막이", "야상", "무스탕", "플리스", "후리스",
    # 정장/포멀
    "정장", "여성정장", "남성정장", "정장바지", "정장자켓", "정장치마",
    "조끼", "베스트", "니트베스트",
]

# ============================================================
# 제목 클렌징 → 캡션
# ============================================================
NOISE_PATTERNS = [
    r"<[^>]+>",
    r"\[[^\]]*\]",
    r"\([^)]*\)",
    r"[★☆▶◀♥♡■◆●▣]+",
    r"무료배송|당일발송|빠른배송|로켓배송|오늘출발",
    r"\b[0-9]+%\s*(할인|세일|쿠폰)?",
    r"정품|특가|핫딜|신상|best|BEST|NEW|new|sale|SALE",
    r"\b[0-9]+\s*~\s*[0-9]+\s*(호|size|사이즈)\b",
]

def clean_title(title):
    text = title
    for p in NOISE_PATTERNS:
        text = re.sub(p, " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ============================================================
# 상태 파일
# ============================================================
def load_json(path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# ============================================================
# API 호출
# ============================================================
def fetch_page(keyword, start):
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    params = {"query": keyword, "display": DISPLAY, "start": start, "sort": SORT}
    try:
        r = requests.get(SEARCH_URL, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  [API오류] {keyword} start={start}: {r.status_code} {r.text[:80]}")
            return None   # None = 중단 신호 (한도 초과 등)
        return r.json().get("items", [])
    except Exception as e:
        print(f"  [예외] {keyword} start={start}: {e}")
        return []

# ============================================================
# 이미지 다운로드
# ============================================================
def download_image(url):
    if not url:
        return None
    fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    fpath = IMG_DIR / fname
    if fpath.exists():
        return str(fpath)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        with open(fpath, "wb") as f:
            f.write(r.content)
        Image.open(fpath).verify()
        return str(fpath)
    except Exception:
        if fpath.exists():
            fpath.unlink()
        return None

def get_phash(image_path):
    try:
        return str(imagehash.phash(Image.open(image_path).convert("RGB")))
    except Exception:
        return None

# ============================================================
# 메인 — 하루치 수집
# ============================================================
def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    seen_ids = set(load_json(SEEN_IDS_FILE, []))
    progress = load_json(PROGRESS_FILE, {})
    seen_phash = load_json(PHASH_FILE, [])

    print("=" * 55)
    print(f"기존 누적: {len(seen_ids):,}개")
    print(f"오늘 목표: {DAILY_LIMIT:,}개")
    print("=" * 55 + "\n")

    # 아직 안 끝난 키워드만 (완료된 건 제외)
    remaining = [k for k in KEYWORDS if progress.get(k, 1) <= MAX_START]
    if not remaining:
        print("모든 키워드 수집 완료! 더 받으려면 KEYWORDS에 추가하세요.")
        return
    # 카테고리 골고루 섞기 위해 셔플 (매일 다른 순서)
    random.shuffle(remaining)

    collected_today = 0
    new_rows = []
    write_header = not META_CSV.exists()
    stopped = False

    for keyword in remaining:
        if collected_today >= DAILY_LIMIT:
            break

        start = progress.get(keyword, 1)
        while start <= MAX_START and collected_today < DAILY_LIMIT:
            items = fetch_page(keyword, start)

            if items is None:        # 한도 초과 → 저장 후 종료
                print("\nAPI 한도 도달로 추정. 상태 저장 후 종료.")
                progress[keyword] = start
                stopped = True
                break
            if not items:            # 결과 없음 → 이 키워드 완료
                progress[keyword] = MAX_START + 1
                break

            for item in items:
                pid = item.get("productId", "")
                if not pid or pid in seen_ids:
                    continue
                caption = clean_title(item.get("title", ""))
                if not caption:
                    continue
                image_url = item.get("image", "")
                img_path = download_image(image_url)
                if not img_path:
                    continue
                # pHash 시각적 중복
                ph = get_phash(img_path)
                if ph:
                    dup = any((imagehash.hex_to_hash(ph) - imagehash.hex_to_hash(sp)) <= PHASH_THRESHOLD
                              for sp in seen_phash[-2000:])
                    if dup:
                        Path(img_path).unlink(missing_ok=True)
                        continue
                    seen_phash.append(ph)

                seen_ids.add(pid)
                new_rows.append({
                    "image_path": img_path,
                    "image_url": image_url,
                    "product_link": item.get("link", ""),
                    "caption": caption,
                    "keyword": keyword,
                    "naver_category": item.get("category3", "") or item.get("category2", ""),
                    "brand": item.get("brand", "") or item.get("maker", ""),
                    "price": item.get("lprice", ""),
                    "mall_name": item.get("mallName", ""),
                    "product_id": pid,
                    "phash": ph or "",
                })
                collected_today += 1
                if collected_today >= DAILY_LIMIT:
                    break

            start += DISPLAY
            progress[keyword] = start
            time.sleep(SLEEP)

        print(f"  '{keyword}' 완료  (오늘 누적 {collected_today:,}개)")
        if stopped:
            break

    # 저장
    save_json(SEEN_IDS_FILE, list(seen_ids))
    save_json(PROGRESS_FILE, progress)
    save_json(PHASH_FILE, seen_phash)
    if new_rows:
        fields = ["image_path", "image_url", "product_link", "caption", "keyword",
                  "naver_category", "brand", "price", "mall_name", "product_id", "phash"]
        mode = "w" if write_header else "a"
        with open(META_CSV, mode, newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)

    # 요약
    done_kw = sum(1 for k in KEYWORDS if progress.get(k, 1) > MAX_START)
    print("\n" + "=" * 55)
    print(f"오늘 신규 수집 : {collected_today:,}개")
    print(f"전체 누적      : {len(seen_ids):,}개")
    print(f"완료 키워드    : {done_kw}/{len(KEYWORDS)}개")
    print(f"저장 위치      : {META_CSV}")
    print("=" * 55)
    if new_rows:
        print("\n[샘플 3개]")
        for r in new_rows[:3]:
            print(f"  · {r['caption']}")
    print("\n→ 내일 다시 실행하면 이어서 수집됩니다.")


if __name__ == "__main__":
    main()
