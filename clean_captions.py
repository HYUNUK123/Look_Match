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
import pandas as pd


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

    # 빈 캡션 마스크
    empty_mask = df["qwen_caption"].apply(is_empty_caption)
    n_empty = int(empty_mask.sum())
    n_keep = total - n_empty

    # 제외된 것 저장 (오판 확인용)
    excluded = df[empty_mask].copy()
    if len(excluded) > 0:
        excluded.to_csv(EXCLUDED_CSV, index=False, encoding="utf-8-sig")

    # 깨끗한 데이터 저장 (빈 캡션 제거)
    clean = df[~empty_mask].copy()
    clean.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 요약
    print(f"\n{'='*50}")
    print(f"유지 (캡션 있음): {n_keep}개")
    print(f"삭제 (빈 캡션)  : {n_empty}개")
    print(f"{'='*50}")
    print(f"학습용 저장: {OUT_CSV}")
    if n_empty > 0:
        print(f"제외 목록  : {EXCLUDED_CSV} (오판 확인용)")
        # 제외된 것 중 crop_path 샘플 몇 개 (눈으로 확인할 수 있게)
        print(f"\n[제외된 이미지 샘플 — 오판 없나 확인용]")
        for p in excluded["crop_path"].head(5):
            fname = str(p).replace("\\", "/").split("/")[-1]
            print(f"  {fname}")
    print(f"\n→ 이미지 파일은 삭제하지 않았습니다 (안전).")
    print(f"→ 학습에는 {OUT_CSV.name} 를 사용하세요.")


if __name__ == "__main__":
    main()
