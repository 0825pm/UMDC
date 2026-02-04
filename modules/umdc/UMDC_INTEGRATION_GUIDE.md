# UMDC 통합 가이드

## 📁 파일 구조

```
MVREC/
├── modules/
│   ├── umdc/                    # ← 새로 추가
│   │   ├── __init__.py
│   │   ├── classifier.py        # UnifiedZipAdapterF
│   │   ├── dataset.py           # UnifiedDataset
│   │   ├── sampler.py           # EpisodicSampler
│   │   └── loss.py              # EpisodicLoss
│   ├── classifier.py            # ← 수정 필요
│   ├── model.py                 # ← 수정 필요
│   └── ...
├── run_unified.py               # ← 새로 추가
└── ...
```

---

## 🔧 Step 1: umdc 폴더 복사

```bash
cp -r umdc/ ~/Projects/research/MVREC/modules/
cp run_unified.py ~/Projects/research/MVREC/
```

---

## 🔧 Step 2: model.py 수정

### 2-1. init_classifier() 메서드에 추가 (Line 151 근처)

**assert 문 수정:**
```python
assert classifier in  [ "ClipZeroShot",
                        "CosimClassfier",
                       "EchoClassfier" ,"EchoClassfierT","EchoClassfierF","EchoClassfier_text","EchoClassfierF_text",
                       "TipAdapter","TipAdapterF",
                       "LinearProbeClassifier",
                       "TransformerClassifier",
                       "ClipAdapter",
                       "EuclideanClassifier",
                       "KNNClassifier",
                       "ClassificationAdapter",
                       "UnifiedZipAdapterF",  # ← 추가!
                       ]
```

### 2-2. classifier 분기 추가 (Line 190 근처)

```python
elif classifier=="ClassificationAdapter":
    head=ClassificationAdapter(text_features)
elif classifier=="UnifiedZipAdapterF":  # ← 추가!
    from modules.umdc import UnifiedZipAdapterF
    head=UnifiedZipAdapterF(text_features)
else:
    raise Exception(f"unknown head name {classifier}")
```

---

## 🔧 Step 3: 실행

```bash
cd ~/Projects/research/MVREC

# UMDC 통합 평가 (5-shot)
python run_unified.py --k_shot 5 --classifier UnifiedZipAdapterF

# 1-shot
python run_unified.py --k_shot 1

# 기존 MVREC 방식 (비교용)
bash run.sh
```

---

## 📊 예상 출력

```
============================================================
[UMDC] Unified Multi-category Defect Classification
============================================================
  K-shot: 5
  Classifier: UnifiedZipAdapterF
  Categories: 14

[UMDC] Loading unified dataset...
  - Total categories: 14
  - Total classes: 68
  - Total samples: 720

[UMDC] Creating model...
[UMDC] Extracting features...

[UMDC] Sampling 1/5
  Support: 340 samples, 68 classes
  Accuracy: 0.XXXX

...

============================================================
[UMDC] Final Results:
  Accuracy: 0.XXXX ± 0.XXXX
============================================================
```

---

## 🏗️ 핵심 설계

### Episodic Training
- Global 68개 class가 아닌 **episode 내 상대적 label** 사용
- 새로운 카테고리도 동일하게 처리 가능

### Category-Agnostic
- 카테고리 정보 불필요
- Support set이 곧 카테고리 정의

### MVREC 호환
- `init_weight()`, `forward()` 인터페이스 동일
- 기존 파이프라인에 드롭인 교체 가능

---

## ⚠️ 주의사항

1. **mv_method**: `"mso"` 사용 (param_space.py의 `"msv"`가 아님)
2. **input_shape**: `224` 사용
3. **GPU 메모리**: 68개 class 통합이라 더 많은 메모리 필요

---

## 🔜 다음 단계 (Phase 2)

1. Dinomaly 기법 추가 (Linear Attention, Noisy Bottleneck)
2. Cross-category episode training
3. Detection + Classification 통합
