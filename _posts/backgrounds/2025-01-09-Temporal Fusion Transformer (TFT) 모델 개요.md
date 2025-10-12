---
title: Temporal Fusion Transformer (TFT) 모델 개요
date: 2025-01-09 00:00:00 +09:00
categories: [backgrounds]
math: true
---
> - 본 글에서는 Temporal Fusion Transformer(TFT)의 구조와 수요예측에서의 적용 가능성을 설명합니다.  
> - TFT는 시계열 데이터의 **단기 패턴**, **장기 추세**, **카테고리, 연속형 피처 상호작용**을 모두 학습할 수 있는 모델입니다.

---

## 1. 모델 개요

- TFT는 2020년 Google Research에서 발표된 모델로, LSTM 기반의 temporal encoder와 Transformer의 self-attention 구조를 결합하여 **multi-horizon forecasting** 문제를 해결합니다.
- 관련하여 발표된 논문은 : [Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207021000637)

---

## 2. 아키텍처 구성

1. **Variable Selection Network (VSN)**
  - 각 시점별 피처의 중요도를 동적으로 학습
  - Feature-level attention을 통해 불필요한 변수 제거

2. **LSTM Encoder/Decoder**
  - 시계열 내의 temporal dependency 학습
  - 과거 컨텍스트를 인코딩, 미래 시점을 디코딩

3. **Static Covariate Encoder**
  - 점포, SKU 등 변하지 않는 특성 반영
  - 각 그룹별 embedding 벡터를 생성하여 context conditioning에 활용

4. **Interpretable Multi-Head Attention**
  - Transformer 기반 attention을 통해 장기 추세(long-term trend) 및 최근 신호(short-term pattern)를 통합

---

## 3. 수요예측 분야에서 사용 시 부각되는 장점
### 요약

| 항목 | 설명 |
|------|------|
| Feature Selection | 시점별 동적 중요도 반영 (VSN) |
| Temporal Modeling | LSTM 기반 sequence dependency |
| Long-term Pattern | Self-attention 기반 global context |
| Interpretability | Variable importance 시각화 가능 |
| Robustness to Missing Future Features | 미래 시점 피처가 없어도 인코더 표현만으로 예측 가능 (Lim et al., NeurIPS 2020) |

TFT는 기존 시계열 모델(LSTM, XGBoost 등)과 비교했을 때 **특히 “입력 피처의 불완전성”에 강하다**는 점이 인상적입니다.  
즉, 일부 피처의 **미래 시점 데이터가 존재하지 않더라도 안정적으로 예측을 수행**할 수 있습니다.

이 특성은 TFT의 입력 구조 설계에서 비롯됩니다. TFT는 입력 변수를 크게 두 그룹으로 구분합니다:

- **Observed Inputs (과거에만 존재하는 피처)**: 과거 시점에서만 관측 가능한 변수 (예: 판매량, 재고, 클릭수 등)
- **Known Inputs (미래에도 알려진 피처)**: 미래 시점에도 미리 정의 가능한 변수 (예: 요일, 프로모션 일정, 공휴일 여부 등)

이때, LSTM 인코더는 **과거 observed inputs만으로 latent context를 구성**하고, 디코더 단계에서 known inputs가 존재할 경우에만 이를 추가적으로 반영합니다.  
즉, **미래 피처가 일부 비어 있더라도 인코더 representation만으로 예측을 이어갈 수 있는 구조**입니다.

개인적으로는 이 "**미래 피처 결측에 대한 강건성(robustness)**"이 TFT의 가장 큰 실무적 장점이라고 생각합니다. e-commerce 수요예측에서는 향후 할인, 이벤트 등 미래 요인이 **사전에 확정되지 않은 경우**가 많기 때문입니다.  

이럴 때 대부분의 모델은 "미래 피처를 생성하거나 가정해야만" 예측이 가능하지만, TFT는 과거 피처만으로도 **multi-horizon 예측을 안정적으로 수행**할 수 있다고 하니, 이만한 장점을 가진 모델을 찾기는 어려울 것 같습니다(써봐야 알겠지만요 ㅎㅎ).

---

## 4. 사용 방법 (PyTorch Forecasting 기반)

[PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) 라이브러리를 이용하여 모델 학습을 구현해봅니다.

이 라이브러리는 시계열 예측에 필요한 **데이터셋 구성 -> 모델 정의 -> 학습 및 예측** 과정을 일관된 방식으로 제공하며, 다음 세 가지 클래스를 핵심으로 사용합니다.


| 클래스 | 역할 | 설명                                      |
|--------|------|-----------------------------------------|
| `TimeSeriesDataSet` | 입력데이터 구성 | 과거(encoder) 구간과 미래(decoder) 구간을 자동으로 분리 |
| `TemporalFusionTransformer` | 핵심 모델 | LSTM + Attention 기반 멀티히라이즌 예측 모델        |
| `Baseline` | 비교 모델 | 단순 평균, 중앙값 등으로 baseline 성능 확인용          |


### 4.1 데이터 구성 — `TimeSeriesDataSet`

TFT는 데이터를 **인코더(과거)** 와 **디코더(미래)** 구간으로 나누어 학습합니다. `TimeSeriesDataSet`는 이것을 자동화해주는 클래스입니다.

```python
from pytorch_forecasting import TimeSeriesDataSet

max_encoder_length = 28   # 과거 4주 데이터 사용 하는 경우
max_prediction_length = 7 # 다음 1주 예측한다고 가정

training = TimeSeriesDataSet(
    df,                               # 시계열 DataFrame
    time_idx="time_idx",              # 시계열 인덱스 (정수형)
    target="y",                       # 예측 타깃 (수요량)
    group_ids=["str_no", "sku_id"],  # SKU 단위 그룹
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,

    # --- 미래에도 미리 알 수 있는 변수 (없어도 됨) ---
    time_varying_known_reals=["dow", "is_holiday"],  # 예: 요일, 공휴일
    time_varying_known_categoricals=[],

    # --- 과거에만 관측되는 변수 (필수) ---
    time_varying_unknown_reals=["y", "inv_cnt"],

    # --- 점포, 카테고리 등 고정 특성 ---
    static_categoricals=["ctg_cls"]
)
```

이렇게 정의하면, 내부적으로 입력 데이터가 (batch, time, feature) 형태의 텐서로 자동 변환되어 LSTM 인코더와 Attention 디코더가 바로 사용할 수 있는 구조가 됩니다.

### 4.2 모델 정의 — `TemporalFusionTransformer`

TemporalFusionTransformer는 LSTM 기반 인코더, Variable Selection Network(VSN), 그리고 Attention 블록으로 구성된 해석 가능한 시계열 모델입니다.

```python
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0001, 
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    lstm_layers=1,
    loss=QuantileLoss(),    # 0.1, 0.5, 0.9 quantile 기반 예측을 가정
    reduce_on_plateau_patience=3,
)
```
