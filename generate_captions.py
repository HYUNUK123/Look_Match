"""
Qwen-VL 캡션 생성 (LookMatch) — Colab / 로컬 겸용
==================================================
crop된 옷 이미지 + 네이버 상품명(힌트) → 깨끗한 한국어 캡션 생성.
네이버 제목의 SEO 키워드 나열을 자연스러운 캡션으로 정제.

특징:
  - Colab / 로컬 자동 감지
  - GPU 메모리에 따라 모델 크기 자동 선택 (2B / 7B)
  - 배치 처리로 속도 ↑
  - 이어받기 (중간에 끊겨도 재실행하면 이어서)
  - 샘플 검수 출력

입력:  clothes_dataset/metadata_cropped.csv (crop_path, caption)
출력:  clothes_dataset/metadata_captioned.csv (+ qwen_caption 컬럼)

설치 (Colab):
  !pip install -q git+https://github.com/huggingface/transformers accelerate qwen-vl-utils
  !pip install -q qwen-vl-utils bitsandbytes
실행:
  python generate_captions.py
"""

import os
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
IN_CSV = "clothes_dataset/metadata_cropped.csv"
OUT_CSV = "clothes_dataset/metadata_captioned.csv"
PROGRESS_FILE = "clothes_dataset/caption_progress.json"

BATCH_SIZE = 8           # 배치 크기 (VRAM 부족하면 줄이기)
MAX_NEW_TOKENS = 40      # 출력 길이 제한 (짧게 = 빠름, 간결)
SAVE_EVERY = 200         # N개마다 중간 저장 (이어받기용)

# 프롬프트 — CLIP 파인튜닝 최적화
# 핵심: 검색 쿼리 형태 + 시각 속성 중심 + 간결한 명사구 + 일관된 구조
# (사용자 검색어와 캡션 형태를 맞춰 검색 정확도를 높임)
PROMPT_TEMPLATE = (
    "너는 패션 이미지 검색 데이터셋의 캡션을 만드는 전문가야. "
    "아래 옷 이미지를 보고, 이미지-텍스트 검색 모델 학습에 쓸 캡션을 한국어로 만들어줘.\n\n"
    "[규칙]\n"
    "1. 눈에 보이는 시각적 속성만 쓴다: 색상, 옷 종류, 핏/실루엣, 넥라인, 소매길이, 패턴/디테일, 소재감.\n"
    "2. '여름', '편안한', '데일리', '예쁜' 같은 추상적·감성적 표현은 절대 쓰지 않는다.\n"
    "3. 문장이 아니라 명사·형용사 위주의 키워드 나열로 쓴다. (예: '베이지 하이웨이스트 밴딩 와이드 슬랙스')\n"
    "4. 순서는 [색상] → [패턴/소재] → [핏/실루엣] → [디테일] → [옷 종류] 를 지향한다.\n"
    "5. 6~12단어 이내로 간결하게. 브랜드명·가격·배송 문구는 넣지 않는다.\n"
    "6. 상품명은 참고만 하고, 이미지에 실제로 보이는 것을 우선한다.\n\n"
    "참고 상품명: {title}\n\n"
    "캡션(키워드 나열만 출력):"
)


# ============================================================
# 환경 감지
# ============================================================
def detect_env():
    """Colab인지, GPU 있는지, VRAM 얼마인지 감지"""
    try:
        import google.colab  # noqa
        is_colab = True
    except ImportError:
        is_colab = False

    has_gpu = torch.cuda.is_available()
    vram_gb = 0
    if has_gpu:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"환경: {'Colab' if is_colab else '로컬'}")
    print(f"GPU: {'있음' if has_gpu else '없음 (CPU - 매우 느림)'}")
    if has_gpu:
        print(f"VRAM: {vram_gb:.1f}GB ({torch.cuda.get_device_name(0)})")
    return is_colab, has_gpu, vram_gb


# ============================================================
# 모델 선택 & 로드 (VRAM에 맞게)
# ============================================================
def load_model(vram_gb):
    """VRAM에 따라 모델 크기·양자화 자동 선택"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # VRAM 기준 모델 선택
    if vram_gb >= 20:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        quant = None
        print(f"→ 7B 모델 (VRAM 충분)")
    elif vram_gb >= 12:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        quant = "4bit"
        print(f"→ 7B 모델 + 4bit 양자화")
    else:
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        quant = None
        print(f"→ 2B 모델 (VRAM 제한, 가벼움)")

    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    # 4bit 양자화 (VRAM 절약)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # 이미지 토큰 수 제한 (속도·메모리)
    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels
    )
    model.eval()
    return model, processor


# ============================================================
# 캡션 정제 (후처리)
# ============================================================
def clean_caption(text):
    """Qwen 출력에서 불필요한 부분 제거 → 깨끗한 키워드 나열만"""
    text = text.strip()
    # 접두어 제거 (모델이 붙이는 군더더기)
    prefixes = ["캡션:", "캡션", "답:", "특징:", "키워드:",
                "이 옷은", "이 옷", "이미지에는", "이미지에", "사진에는", "사진에",
                "옷 종류:", "색상:"]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
    # 따옴표·기호 제거
    for junk in ['"', "'", "`", "```", "*", "-", "•", "·", ":", "\n"]:
        text = text.replace(junk, " ")
    # 문장 끝 표현 제거 (혹시 문장으로 나온 경우)
    for tail in ["입니다", "이에요", "예요", "습니다", "같습니다", "보입니다"]:
        text = text.replace(tail, "")
    # 공백 정리
    text = " ".join(text.split())
    return text.strip(" .·:")[:100]


# ============================================================
# 배치 캡션 생성
# ============================================================
def caption_batch(model, processor, rows):
    """이미지+제목 배치 → 캡션 리스트"""
    messages_list = []
    valid_idx = []
    for i, row in enumerate(rows):
        img_path = row["crop_path"]
        if not Path(img_path).exists():
            continue
        title = str(row.get("caption", ""))[:100]  # 네이버 제목(힌트)
        messages_list.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": PROMPT_TEMPLATE.format(title=title)},
            ],
        }])
        valid_idx.append(i)

    if not messages_list:
        return {}

    # 배치 전처리
    from qwen_vl_utils import process_vision_info
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    # 생성
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    outputs = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)

    # idx → 캡션 매핑
    result = {}
    for local_i, cap in zip(valid_idx, outputs):
        result[local_i] = clean_caption(cap)
    return result


# ============================================================
# 메인
# ============================================================
def main():
    is_colab, has_gpu, vram_gb = detect_env()
    if not has_gpu:
        print("\n⚠️ GPU가 없어 매우 느립니다. Colab GPU 런타임을 권장합니다.")

    df = pd.read_csv(IN_CSV)
    print(f"\n총 {len(df)}개 캡션 생성 대상\n")

    # 이어받기: 이미 처리한 인덱스 로드
    done = set()
    captions = {}
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
            captions = {int(k): v for k, v in saved.items()}
            done = set(captions.keys())
        print(f"이어받기: {len(done)}개 이미 완료\n")

    # 모델 로드
    model, processor = load_model(vram_gb)

    # 배치 처리
    todo = [i for i in range(len(df)) if i not in done]
    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="캡션 생성"):
        batch_idx = todo[start:start + BATCH_SIZE]
        rows = [df.iloc[i].to_dict() for i in batch_idx]

        try:
            batch_result = caption_batch(model, processor, rows)
            for local_i, cap in batch_result.items():
                captions[batch_idx[local_i]] = cap
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM! BATCH_SIZE를 줄이세요 (현재 {BATCH_SIZE}). 진행분 저장.")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"\n배치 오류(건너뜀): {e}")
            continue

        # 중간 저장 (이어받기용)
        if (start // BATCH_SIZE) % (SAVE_EVERY // BATCH_SIZE + 1) == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in captions.items()}, f, ensure_ascii=False)
            gc.collect()
            if has_gpu:
                torch.cuda.empty_cache()

    # 최종 저장
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in captions.items()}, f, ensure_ascii=False)

    df["qwen_caption"] = df.index.map(lambda i: captions.get(i, ""))
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 요약 + 샘플
    n_done = sum(1 for v in captions.values() if v)
    print(f"\n{'='*55}")
    print(f"캡션 생성 완료: {n_done}/{len(df)}개")
    print(f"저장: {OUT_CSV}")
    print(f"\n[샘플 5개 — 네이버 제목 → Qwen 캡션]")
    shown = 0
    for i in range(len(df)):
        if captions.get(i):
            orig = str(df.iloc[i].get("caption", ""))[:40]
            print(f"  원본: {orig}")
            print(f"  정제: {captions[i]}\n")
            shown += 1
            if shown >= 5:
                break
    print("="*55)


if __name__ == "__main__":
    main()
