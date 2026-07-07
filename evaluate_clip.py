"""
Jina-CLIP v2 성능 평가 (LookMatch) — 3가지 실험
================================================
검색 모델(CLIP)의 성능을 Recall@K로 평가한다.

실험 1: 캡션 스타일 비교 (파인튜닝 X)
  원본 모델로 같은 이미지를 두 캡션으로 각각 검색
  → 네이버 제목 vs Qwen 캡션 중 CLIP이 뭘 잘 매칭하나

실험 2: 파인튜닝 전후 비교
  같은 검증셋(Qwen 캡션)으로 원본 vs 파인튜닝 모델
  → 파인튜닝 효과

실험 3: 학습 캡션별 모델 비교
  네이버로 학습한 모델 vs Qwen으로 학습한 모델
  → 데이터 품질 효과

지표: Recall@1 / @5 / @10 (텍스트→이미지 + 이미지→이미지)

입력:  metadata_captioned_final.csv (crop_path, caption, qwen_caption)
       (실험 2,3은 파인튜닝된 LoRA 필요)

실행:  python evaluate_clip.py
       → 어떤 실험을 돌릴지 아래 RUN_* 플래그로 제어
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ============================================================
# 설정
# ============================================================
def setup_data_dir():
    env_dir = os.environ.get("LOOKMATCH_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        import google.colab  # noqa
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            from google.colab import drive
            drive.mount("/content/drive")
        return drive_root / "clothes_dataset"
    except ImportError:
        return Path("clothes_dataset")


DATA_DIR = setup_data_dir()
IN_CSV = DATA_DIR / "metadata_captioned_final.csv"
if not IN_CSV.exists():
    for alt in ["metadata_captioned.csv", "metadata_captioned_test_final.csv",
                "metadata_captioned_test.csv"]:
        if (DATA_DIR / alt).exists():
            IN_CSV = DATA_DIR / alt
            break

CROPS_SUBDIR = "crops"
LOCAL_CROPS = Path("/content/crops_local")   # Colab 로컬 복사본이 있으면 사용

MODEL_NAME = "jinaai/jina-clip-v2"
TRUNCATE_DIM = 512
FINETUNED_QWEN_DIR = str(DATA_DIR / "clip_finetuned")            # Qwen 캡션 학습 결과
FINETUNED_NAVER_DIR = str(DATA_DIR / "clip_finetuned_naver")     # 네이버 제목 학습 결과

# 평가 설정
N_EVAL = 1000            # 평가에 쓸 샘플 수 (검증셋 크기). 전체면 None
K_LIST = [1, 5, 10]      # Recall@K
BATCH = 64
SEED = 42

# 어떤 실험을 돌릴지 (파인튜닝 안 됐으면 실험1만 True)
RUN_EXP1 = True          # 캡션 스타일 비교 (파인튜닝 불필요)
RUN_EXP2 = False         # 파인튜닝 전후 (clip_finetuned 필요)
RUN_EXP3 = False         # 네이버 vs Qwen 학습 (둘 다 필요)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_image_path(crop_path):
    fname = str(crop_path).replace("\\", "/").split("/")[-1]
    for cand in [LOCAL_CROPS / fname, Path(crop_path), DATA_DIR / CROPS_SUBDIR / fname]:
        if Path(cand).exists():
            return str(cand)
    return None


# ============================================================
# 데이터 로드 (평가용 검증셋)
# ============================================================
def load_eval_data():
    df = pd.read_csv(IN_CSV)
    # 이미지·캡션 유효한 것만
    rows = []
    for _, r in df.iterrows():
        img = resolve_image_path(r["crop_path"])
        naver = str(r.get("caption", "")).strip()
        qwen = str(r.get("qwen_caption", "")).strip()
        if img and qwen:  # qwen 캡션은 필수, naver는 실험1에서만
            rows.append({"image": img, "naver": naver, "qwen": qwen})
    # 평가 샘플 수 제한
    random.shuffle(rows)
    if N_EVAL:
        rows = rows[:N_EVAL]
    print(f"평가 샘플: {len(rows)}개")
    return rows


# ============================================================
# 모델 로드
# ============================================================
def load_model(finetuned_dir=None):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
    ).to(DEVICE)
    if finetuned_dir and Path(finetuned_dir).exists():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, finetuned_dir).to(DEVICE)
            print(f"  파인튜닝 로드: {finetuned_dir}")
        except Exception as e:
            print(f"  [경고] 파인튜닝 로드 실패({e}) → 원본 사용")
    model.eval()
    return model


def get_base(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


# ============================================================
# 임베딩 계산
# ============================================================
@torch.no_grad()
def embed(model, images=None, texts=None):
    base = get_base(model)
    if images is not None:
        emb = base.encode_image(images, truncate_dim=TRUNCATE_DIM)
    else:
        emb = base.encode_text(texts, truncate_dim=TRUNCATE_DIM)
    if not torch.is_tensor(emb):
        emb = torch.tensor(np.array(emb), dtype=torch.float32)
    return F.normalize(emb.to(DEVICE).float(), dim=-1)


@torch.no_grad()
def embed_all(model, rows, caption_key):
    """검증셋 전체를 이미지·텍스트 임베딩으로. caption_key: 'naver' 또는 'qwen'"""
    img_embs, txt_embs = [], []
    for i in tqdm(range(0, len(rows), BATCH), desc=f"임베딩({caption_key})", leave=False):
        chunk = rows[i:i+BATCH]
        images = [Image.open(r["image"]).convert("RGB") for r in chunk]
        texts = [r[caption_key] for r in chunk]
        img_embs.append(embed(model, images=images))
        txt_embs.append(embed(model, texts=texts))
    return torch.cat(img_embs), torch.cat(txt_embs)


# ============================================================
# Recall@K 계산
# ============================================================
def recall_at_k(txt_emb, img_emb, k_list, direction="t2i"):
    """
    direction:
      't2i' 텍스트→이미지 (i번째 캡션의 정답은 i번째 이미지)
      'i2i' 이미지→이미지 (자기 제외 최근접이 같은 인덱스일 순 없으니 t2i 위주 권장)
    """
    if direction == "t2i":
        sims = txt_emb @ img_emb.t()      # (N, N)
    else:  # i2i: 이미지로 이미지 (자기 자신 제외)
        sims = img_emb @ img_emb.t()
        sims.fill_diagonal_(-1e9)         # 자기 자신 제외

    results = {}
    n = sims.size(0)
    for k in k_list:
        topk = sims.topk(min(k, sims.size(1)), dim=1).indices
        correct = sum(i in topk[i] for i in range(n))
        results[k] = correct / n
    return results


def print_recall(title, results):
    line = " / ".join(f"R@{k}={v:.3f}" for k, v in results.items())
    print(f"  {title}: {line}")


# ============================================================
# 실험 1: 캡션 스타일 비교 (파인튜닝 X)
# ============================================================
def experiment_1(rows):
    print("\n" + "="*60)
    print("[실험 1] 캡션 스타일 비교 (원본 모델, 파인튜닝 X)")
    print("  같은 이미지를 네이버 제목 vs Qwen 캡션으로 검색")
    print("="*60)
    model = load_model()   # 원본

    # 네이버 제목이 있는 것만 (일부는 비었을 수 있음)
    naver_rows = [r for r in rows if r["naver"]]
    print(f"\n네이버 제목 있는 샘플: {len(naver_rows)}개로 평가")

    # 네이버 제목으로 검색
    img_emb, txt_emb = embed_all(model, naver_rows, "naver")
    r_naver = recall_at_k(txt_emb, img_emb, K_LIST, "t2i")

    # Qwen 캡션으로 검색 (같은 이미지셋)
    img_emb2, txt_emb2 = embed_all(model, naver_rows, "qwen")
    r_qwen = recall_at_k(txt_emb2, img_emb2, K_LIST, "t2i")

    print("\n결과 (텍스트→이미지 검색):")
    print_recall("네이버 제목으로 검색", r_naver)
    print_recall("Qwen 캡션으로 검색  ", r_qwen)
    print("\n해석: Recall이 높은 캡션이 CLIP과 더 잘 매칭됨")
    return {"naver": r_naver, "qwen": r_qwen}


# ============================================================
# 실험 2: 파인튜닝 전후 비교
# ============================================================
def experiment_2(rows):
    print("\n" + "="*60)
    print("[실험 2] 파인튜닝 전후 비교 (Qwen 캡션 기준)")
    print("="*60)

    # 원본
    print("\n원본 모델:")
    model_before = load_model()
    img_b, txt_b = embed_all(model_before, rows, "qwen")
    r_before_t2i = recall_at_k(txt_b, img_b, K_LIST, "t2i")
    r_before_i2i = recall_at_k(txt_b, img_b, K_LIST, "i2i")
    del model_before; torch.cuda.empty_cache() if DEVICE=="cuda" else None

    # 파인튜닝
    print("파인튜닝 모델:")
    model_after = load_model(FINETUNED_QWEN_DIR)
    img_a, txt_a = embed_all(model_after, rows, "qwen")
    r_after_t2i = recall_at_k(txt_a, img_a, K_LIST, "t2i")
    r_after_i2i = recall_at_k(txt_a, img_a, K_LIST, "i2i")

    print("\n결과 (텍스트→이미지):")
    print_recall("파인튜닝 전(원본)", r_before_t2i)
    print_recall("파인튜닝 후      ", r_after_t2i)
    print("\n결과 (이미지→이미지):")
    print_recall("파인튜닝 전(원본)", r_before_i2i)
    print_recall("파인튜닝 후      ", r_after_i2i)

    # 개선폭
    print("\n개선폭 (텍스트→이미지):")
    for k in K_LIST:
        diff = r_after_t2i[k] - r_before_t2i[k]
        print(f"  R@{k}: {r_before_t2i[k]:.3f} → {r_after_t2i[k]:.3f} ({diff:+.3f})")
    return {"before": r_before_t2i, "after": r_after_t2i}


# ============================================================
# 실험 3: 학습 캡션별 모델 비교
# ============================================================
def experiment_3(rows):
    print("\n" + "="*60)
    print("[실험 3] 학습 캡션별 모델 비교 (같은 검증셋: Qwen 캡션)")
    print("  네이버 제목으로 학습 vs Qwen 캡션으로 학습")
    print("="*60)

    # 네이버로 학습한 모델
    print("\n네이버 제목으로 학습한 모델:")
    model_naver = load_model(FINETUNED_NAVER_DIR)
    img_n, txt_n = embed_all(model_naver, rows, "qwen")   # 검증은 Qwen 캡션 고정
    r_naver = recall_at_k(txt_n, img_n, K_LIST, "t2i")
    del model_naver; torch.cuda.empty_cache() if DEVICE=="cuda" else None

    # Qwen으로 학습한 모델
    print("Qwen 캡션으로 학습한 모델:")
    model_qwen = load_model(FINETUNED_QWEN_DIR)
    img_q, txt_q = embed_all(model_qwen, rows, "qwen")
    r_qwen = recall_at_k(txt_q, img_q, K_LIST, "t2i")

    print("\n결과 (검증셋 Qwen 캡션 고정, 텍스트→이미지):")
    print_recall("네이버 제목으로 학습", r_naver)
    print_recall("Qwen 캡션으로 학습  ", r_qwen)
    print("\n해석: 같은 검증셋에서 더 높은 쪽이 우수한 학습 데이터")
    return {"naver_trained": r_naver, "qwen_trained": r_qwen}


# ============================================================
# 메인
# ============================================================
def main():
    print(f"입력: {IN_CSV}")
    print(f"디바이스: {DEVICE}")
    if not IN_CSV.exists():
        print("[오류] 입력 CSV 없음. 캡션 생성·정리를 먼저 하세요.")
        return

    rows = load_eval_data()
    summary = {}

    if RUN_EXP1:
        summary["exp1"] = experiment_1(rows)
    if RUN_EXP2:
        if not Path(FINETUNED_QWEN_DIR).exists():
            print(f"\n[실험 2 건너뜀] 파인튜닝 결과 없음: {FINETUNED_QWEN_DIR}")
        else:
            summary["exp2"] = experiment_2(rows)
    if RUN_EXP3:
        if not (Path(FINETUNED_QWEN_DIR).exists() and Path(FINETUNED_NAVER_DIR).exists()):
            print(f"\n[실험 3 건너뜀] 두 파인튜닝 결과가 모두 필요합니다.")
        else:
            summary["exp3"] = experiment_3(rows)

    print("\n" + "="*60)
    print("평가 완료")
    print("="*60)


if __name__ == "__main__":
    main()
