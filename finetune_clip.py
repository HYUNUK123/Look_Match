"""
Jina-CLIP v2 LoRA 파인튜닝 (LookMatch) — 로컬 / Colab / 구글드라이브 겸용
========================================================================
crop된 옷 이미지 + qwen_caption 쌍으로 Jina-CLIP v2를 도메인 파인튜닝.
대조학습(대칭 InfoNCE)으로 "이 옷 이미지 ↔ 이 캡션"을 정렬.

특징:
  - 환경 자동 감지 (로컬 / Colab / 드라이브)
  - 이미지를 로컬로 복사 후 학습 (드라이브 I/O 병목 제거)
  - LoRA 파인튜닝 (가볍고 빠름)
  - 대칭 InfoNCE 손실 (이미지↔텍스트 양방향)
  - Recall@K 평가 + 최고 성능 저장
  - 체크포인트 (끊겨도 이어서)

입력:  <데이터폴더>/metadata_captioned_final.csv (crop_path, qwen_caption)
출력:  <데이터폴더>/clip_finetuned/  (LoRA 가중치)

설치:
  [로컬]  pip install torch transformers peft pillow pandas tqdm einops timm
  [Colab] !pip install -q transformers peft einops timm pillow pandas tqdm

실행:  python finetune_clip.py

※ 주의:
  - Jina-CLIP v2는 trust_remote_code=True 필요, cc-by-nc-4.0 라이선스(비상업).
  - encode_text/encode_image는 gradient가 흐르므로 파인튜닝 가능.
  - 처음엔 SMOKE_TEST=True로 소량 검증 후 전체 실행 권장.
"""

import os
import gc
import time
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# ============================================================
# 경로 설정 — 로컬 / Colab / 드라이브 자동 감지
# ============================================================
def setup_data_dir():
    env_dir = os.environ.get("LOOKMATCH_DIR")
    if env_dir:
        print(f"데이터 경로(환경변수): {env_dir}")
        return Path(env_dir)
    try:
        import google.colab  # noqa
        is_colab = True
    except ImportError:
        is_colab = False
    if is_colab:
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            print("구글 드라이브 연결 중...")
            from google.colab import drive
            drive.mount("/content/drive")
        data_dir = drive_root / "clothes_dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"데이터 경로(구글드라이브): {data_dir}")
        return data_dir, True
    data_dir = Path("clothes_dataset")
    print(f"데이터 경로(로컬): {data_dir.resolve()}")
    return data_dir, False


_result = setup_data_dir()
if isinstance(_result, tuple):
    DATA_DIR, IS_COLAB = _result
else:
    DATA_DIR, IS_COLAB = _result, False

# 입력 CSV: final 우선, 없으면 captioned
IN_CSV = DATA_DIR / "metadata_captioned_final.csv"
if not IN_CSV.exists():
    for alt in ["metadata_captioned.csv", "metadata_captioned_test_final.csv",
                "metadata_captioned_test.csv"]:
        if (DATA_DIR / alt).exists():
            IN_CSV = DATA_DIR / alt
            break

OUT_DIR = str(DATA_DIR / "clip_finetuned")
CROPS_SUBDIR = "crops"
# Colab이면 이미지를 로컬로 복사해 I/O 병목 제거
LOCAL_CROPS = Path("/content/crops_local") if IS_COLAB else (DATA_DIR / CROPS_SUBDIR)

# ============================================================
# 하이퍼파라미터
# ============================================================
MODEL_NAME = "jinaai/jina-clip-v2"
TRUNCATE_DIM = 512       # Matryoshka 차원 (512면 저장·속도 이득, None이면 1024)
BATCH_SIZE = 32          # 대조학습은 배치 클수록 유리 (OOM 시 줄이기)
EPOCHS = 3
LR = 1e-5
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
VAL_RATIO = 0.1
SEED = 42
SMOKE_TEST = True        # True면 소량(200개)으로 빠르게 검증. 검증되면 False로.
SMOKE_N = 200

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def resolve_image_path(crop_path):
    """윈도우/리눅스 경로 혼용 처리 + 로컬 복사본 우선"""
    fname = str(crop_path).replace("\\", "/").split("/")[-1]
    # 로컬 복사본 우선
    local = LOCAL_CROPS / fname
    if local.exists():
        return str(local)
    # 원본 경로
    if Path(crop_path).exists():
        return str(crop_path)
    cand = DATA_DIR / CROPS_SUBDIR / fname
    if cand.exists():
        return str(cand)
    return None


# ============================================================
# 이미지 로컬 복사 (Colab I/O 병목 제거)
# ============================================================
def copy_images_local(df):
    """드라이브 crops → 로컬로 복사 (Colab에서 학습 속도 대폭 향상)"""
    if not IS_COLAB:
        return
    LOCAL_CROPS.mkdir(parents=True, exist_ok=True)
    src_dir = DATA_DIR / CROPS_SUBDIR
    print(f"이미지를 로컬로 복사 중... ({src_dir} → {LOCAL_CROPS})")
    copied = 0
    for crop_path in tqdm(df["crop_path"], desc="복사"):
        fname = str(crop_path).replace("\\", "/").split("/")[-1]
        dst = LOCAL_CROPS / fname
        if dst.exists():
            continue
        src = src_dir / fname
        if src.exists():
            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception:
                pass
    print(f"복사 완료: {copied}개")


# ============================================================
# 데이터셋
# ============================================================
class ClothesDataset(Dataset):
    def __init__(self, df):
        self.rows = []
        for _, row in df.iterrows():
            img = resolve_image_path(row["crop_path"])
            cap = str(row.get("qwen_caption", "")).strip()
            if img and cap:
                self.rows.append((img, cap))
        print(f"  유효 샘플: {len(self.rows)}개")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, caption = self.rows[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (512, 512), (255, 255, 255))
        return {"image": image, "caption": caption}


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "captions": [b["caption"] for b in batch],
    }


# ============================================================
# 모델 로드 + LoRA
# ============================================================
def load_model():
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    print(f"모델 로드: {MODEL_NAME} (처음이면 다운로드에 시간 걸림)")

    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)

    # LoRA 적용 시도 (target_modules는 실제 구조에 맞게 자동 탐색)
    try:
        from peft import LoraConfig, get_peft_model
        # Jina-CLIP 내부 attention 계층 이름 후보 (여러 개 넣어 매칭되는 것만 적용)
        target_candidates = ["q_proj", "k_proj", "v_proj", "out_proj",
                             "query", "key", "value", "dense",
                             "Wqkv", "in_proj", "wo", "wq", "wk", "wv"]
        # 실제 존재하는 모듈만 필터
        existing = set()
        for name, _ in model.named_modules():
            for t in target_candidates:
                if name.endswith(t):
                    existing.add(t)
        targets = list(existing) if existing else target_candidates
        print(f"LoRA target_modules: {targets}")

        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=targets, bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"[LoRA 경고] {e}")
        print("→ LoRA 없이 전체 파인튜닝으로 진행")

    return model, device


def get_base(model):
    """peft 래핑 여부와 무관하게 encode 메서드 접근"""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


# ============================================================
# 임베딩 (gradient 흐름 유지)
# ============================================================
def encode_batch(model, images, captions, device):
    base = get_base(model)
    # Jina-CLIP encode_image / encode_text (PIL·문자열 리스트 입력)
    img_emb = base.encode_image(images, truncate_dim=TRUNCATE_DIM)
    txt_emb = base.encode_text(captions, truncate_dim=TRUNCATE_DIM)
    # numpy로 반환되면 텐서로 (gradient 필요 시 아래 fallback 사용)
    if not torch.is_tensor(img_emb):
        img_emb = torch.tensor(np.array(img_emb), device=device, dtype=torch.float32)
    if not torch.is_tensor(txt_emb):
        txt_emb = torch.tensor(np.array(txt_emb), device=device, dtype=torch.float32)
    return img_emb.to(device).float(), txt_emb.to(device).float()


def clip_loss(img_emb, txt_emb, temperature=0.07):
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    logits = (img_emb @ txt_emb.t()) / temperature
    labels = torch.arange(len(img_emb), device=img_emb.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


@torch.no_grad()
def evaluate(model, loader, device, k=5):
    model.eval()
    all_i, all_t = [], []
    for b in loader:
        ie, te = encode_batch(model, b["images"], b["captions"], device)
        all_i.append(F.normalize(ie, dim=-1))
        all_t.append(F.normalize(te, dim=-1))
    if not all_i:
        return 0.0
    ie, te = torch.cat(all_i), torch.cat(all_t)
    sims = te @ ie.t()
    topk = sims.topk(min(k, sims.size(1)), dim=1).indices
    correct = sum(i in topk[i] for i in range(len(topk)))
    return correct / len(topk)


# ============================================================
# 학습
# ============================================================
def train():
    print(f"입력 CSV: {IN_CSV}")
    if not IN_CSV.exists():
        print("[오류] 입력 CSV 없음. generate_captions.py → clean_captions.py 먼저 실행.")
        return

    df = pd.read_csv(IN_CSV)
    df = df[df["qwen_caption"].notna() & (df["qwen_caption"].astype(str).str.strip() != "")]
    if SMOKE_TEST:
        df = df.head(SMOKE_N).copy()
        print(f"[SMOKE TEST] {len(df)}개로 빠른 검증 (검증되면 SMOKE_TEST=False)")
    print(f"학습 대상: {len(df)}개")

    # 이미지 로컬 복사 (Colab I/O 병목 제거)
    copy_images_local(df)

    # 분할
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(len(df) * VAL_RATIO))
    val_df, train_df = df[:n_val], df[n_val:]
    print("학습셋:", end=" "); train_ds = ClothesDataset(train_df)
    print("검증셋:", end=" "); val_ds = ClothesDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    model, device = load_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    print(f"\n학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})\n")
    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for b in pbar:
            optimizer.zero_grad()
            ie, te = encode_batch(model, b["images"], b["captions"], device)
            if not ie.requires_grad:
                # encode_*가 gradient를 안 흘리면 경고 (파인튜닝 불가 신호)
                print("\n[경고] 임베딩에 gradient가 없습니다. "
                      "encode_* 대신 저수준 forward가 필요할 수 있습니다.")
            loss = clip_loss(ie, te)
            loss.backward()
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        recall = evaluate(model, val_loader, device, k=5)
        dt = time.time() - t0
        print(f"  Epoch {epoch+1}: loss={total/max(len(train_loader),1):.4f}, "
              f"Recall@5={recall:.3f}, {dt:.0f}초")

        if recall >= best:
            best = recall
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            try:
                model.save_pretrained(OUT_DIR)
                print(f"  → 저장: {OUT_DIR} (Recall@5={recall:.3f})")
            except Exception as e:
                print(f"  저장 경고: {e}")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"학습 완료! 최고 Recall@5 = {best:.3f}")
    print(f"LoRA 가중치: {OUT_DIR}")
    if SMOKE_TEST:
        print("\n※ SMOKE TEST 완료. 정상이면 SMOKE_TEST=False로 바꿔 전체 학습하세요.")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()
