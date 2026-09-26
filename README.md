# Look_Match

1. LookMatch_Data_Collection.py -> 네이버쇼핑 API 데이터 수집
2. crop_clothes.py -> YOLO 크롭 + 512 패딩
3. generate_captions_colab_upgrade.ipynb -> Qwen3-VL-8B 캡션 생성
4. clean_captions.py -> 캡션 후처리(한자/반복/이모지 정제)
5. finetune_clip.ipynb, finetune_siglip2_full.ipynb -> JinaClipv2, SigLIP2 LoRA 파인튜닝
6. evaluate_jina_clip_experiment2_val7000.ipynb, evaluate_siglip2_full_experiment2_val7000.ipynb ->  Recall 평가 (텍스트→이미지)
7. evaluate_vlm_judge_gpt_changequery2_margin.ipynb -> VLM-as-Judge 평가 (nDCG/margin/Wilcoxon)
8. search_lookmatch_local_valonly_price_gradio_naverpicture_text+imagesearch_reason2_rerank_crossencoder(textonly).ipynb -> Qdrant 검색 + BGE 리랭킹 + Gradio UI
