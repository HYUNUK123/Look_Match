"""
캡션 정리 (LookMatch) — 빈 캡션 행 삭제
========================================
generate_captions.py 결과에서 qwen_caption이 빈 행(색상 나열로 제외된 것)을
제거해 깨끗한 학습용 CSV를 만든다. 이미지 파일은 삭제하지 않음(안전).

입력:  metadata_captioned.csv (qwen_caption 컬럼, 일부 빈 값)
출력:  metadata_final.csv       (빈 캡션 제거된 학습용)
       excluded_captions.csv    (제외된 목록 — 오판 확인용)

실행:  python clean_captions.py
"""

import os
from pathlib import Path
import re
import pandas as pd


# ============================================================
# 캡션 정리 (/ → 공백, 이모지·외국어·반복 처리)
# ============================================================
# 한글, 영어, 숫자, 공백만 유효. 그 외는 제거.
_VALID = re.compile(r"[가-힣a-zA-Z0-9\s]")
# 외국어 문자 감지: 한글·영어·숫자·공백·기본기호가 아닌 "문자"가 있으면 외국어로 간주
# (일본어 가나·한자, 중국어, 러시아어 키릴, 기타 스크립트 모두 포함)
_NON_KO_EN = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s/·,.()\-]")


def has_foreign(text):
    """한글/영어/숫자 외의 외국어 문자(일본어·중국어·러시아어 등)가 있으면 True"""
    return bool(_NON_KO_EN.search(str(text)))


def has_repetition(text, max_repeat=3):
    """같은 단어가 max_repeat번 이상 반복되면 True (모델 생성 폭주)"""
    words = str(text).replace("/", " ").split()
    if not words:
        return False
    # 단어별 등장 횟수
    from collections import Counter
    counts = Counter(words)
    return any(c >= max_repeat for c in counts.values())


def tidy_caption(text):
    """/ → 공백, 이모지·기호 제거, 중복 단어 정리"""
    text = str(text)
    # 슬래시를 공백으로 (하이웨스트/밴딩/포켓 → 하이웨스트 밴딩 포켓)
    text = text.replace("/", " ").replace("／", " ")
    # 유효 문자(한글·영어·숫자·공백)만 남기고 나머지 제거 (이모지 등)
    text = "".join(ch if _VALID.match(ch) else " " for ch in text)
    # 중복 단어 제거 (순서 유지)
    seen, words = set(), []
    for w in text.split():
        if w not in seen:
            seen.add(w)
            words.append(w)
    return " ".join(words).strip()


# ============================================================
# 경로 설정 — 로컬 / Colab / 드라이브 자동 감지 (다른 스크립트와 동일)
# ============================================================
def setup_data_dir():
    env_dir = os.environ.get("LOOKMATCH_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        import google.colab  # noqa
        is_colab = True
    except ImportError:
        is_colab = False
    if is_colab:
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            from google.colab import drive
            drive.mount("/content/drive")
        return drive_root / "clothes_dataset"
    return Path("clothes_dataset")


DATA_DIR = setup_data_dir()

# 입력/출력 경로 (테스트면 _test 파일 사용하도록 자동 선택)
IN_CSV = DATA_DIR / "metadata_captioned.csv"
if not IN_CSV.exists():
    # 전체 파일이 없으면 테스트 파일 시도
    test_csv = DATA_DIR / "metadata_captioned_test.csv"
    if test_csv.exists():
        IN_CSV = test_csv

OUT_CSV = DATA_DIR / (IN_CSV.stem + "_final.csv")
EXCLUDED_CSV = DATA_DIR / (IN_CSV.stem + "_excluded.csv")


def is_empty_caption(v):
    """빈 캡션 판정: NaN, 빈 문자열, 공백만"""
    if pd.isna(v):
        return True
    if str(v).strip() == "":
        return True
    return False


def main():
    print(f"입력: {IN_CSV}")
    if not IN_CSV.exists():
        print(f"[오류] 입력 파일이 없습니다: {IN_CSV}")
        print("→ generate_captions.py를 먼저 실행하세요.")
        return

    df = pd.read_csv(IN_CSV)
    total = len(df)
    print(f"전체: {total}개")

    if "qwen_caption" not in df.columns:
        print("[오류] qwen_caption 컬럼이 없습니다.")
        return

    # 1) 외국어(일본어/중국어/러시아어 등) 섞인 캡션 표시 → 제외 대상
    foreign_mask = df["qwen_caption"].apply(has_foreign)
    n_foreign = int(foreign_mask.sum())

    # 1-2) 같은 단어 반복(모델 폭주) 캡션 표시 → 제외 대상
    repeat_mask = df["qwen_caption"].apply(has_repetition)
    n_repeat = int(repeat_mask.sum())

    # 2) 캡션 정리 (/ → 공백, 이모지 제거, 중복 제거)
    df["qwen_caption"] = df["qwen_caption"].apply(tidy_caption)

    # 3) 빈 캡션 마스크 (원래 빈 것 + 정리 후 빈 것)
    empty_mask = df["qwen_caption"].apply(is_empty_caption)

    # 제외 = 빈 캡션 OR 외국어 OR 반복 OR 너무 짧은 것
    too_short = df["qwen_caption"].str.len() < 5
    drop_mask = empty_mask | foreign_mask | repeat_mask | too_short

    n_empty = int(empty_mask.sum())
    n_short = int((too_short & ~empty_mask & ~foreign_mask & ~repeat_mask).sum())
    n_drop = int(drop_mask.sum())
    n_keep = total - n_drop

    # 제외된 것 저장 (오판 확인용) — 제외 사유 표시
    excluded = df[drop_mask].copy()
    if len(excluded) > 0:
        excluded["exclude_reason"] = ""
        excluded.loc[too_short[drop_mask], "exclude_reason"] = "너무짧음"
        excluded.loc[repeat_mask[drop_mask], "exclude_reason"] = "단어반복"
        excluded.loc[foreign_mask[drop_mask], "exclude_reason"] = "외국어"
        excluded.loc[empty_mask[drop_mask], "exclude_reason"] = "빈캡션"
        excluded.to_csv(EXCLUDED_CSV, index=False, encoding="utf-8-sig")

    # 깨끗한 데이터 저장
    clean = df[~drop_mask].copy()
    clean.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 요약
    print(f"\n{'='*50}")
    print(f"유지 (학습용): {n_keep}개")
    print(f"제외 합계    : {n_drop}개")
    print(f"  - 빈 캡션(색상나열 등): {n_empty}개")
    print(f"  - 외국어(일본어/러시아어 등): {n_foreign}개")
    print(f"  - 단어 반복(모델 폭주): {n_repeat}개")
    print(f"  - 너무 짧음          : {n_short}개")
    print(f"{'='*50}")
    print(f"학습용 저장: {OUT_CSV}")
    if n_drop > 0:
        print(f"제외 목록  : {EXCLUDED_CSV} (사유 포함, 오판 확인용)")
        print(f"\n[제외된 이미지 샘플 — 오판 없나 확인용]")
        for _, r in excluded.head(5).iterrows():
            fname = str(r["crop_path"]).replace("\\", "/").split("/")[-1]
            print(f"  [{r.get('exclude_reason','')}] {fname}")
    print(f"\n→ 이미지 파일은 삭제하지 않았습니다 (안전).")
    print(f"→ 학습에는 {OUT_CSV.name} 를 사용하세요.")


if __name__ == "__main__":
    main()
