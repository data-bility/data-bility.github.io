---
title: Bayesian opt 기반 모델 선택에서 SVR은 왜 잘 선택되지 않을까?
date: 2024-03-15 00:00:00 +09:00
categories: [experiments]
math: true
---

## 1. 문제 제기
지난 2월에 진행한 [**"SSG 1DAY 배송 서비스 자동발주예측 ML모델 고도화"**](https://data-bility.github.io/posts/SSG-1DAY-%EB%B0%B0%EC%86%A1-%EC%84%9C%EB%B9%84%EC%8A%A4-%EC%9E%90%EB%8F%99%EB%B0%9C%EC%A3%BC%EC%98%88%EC%B8%A1-ML%EB%AA%A8%EB%8D%B8-%EA%B3%A0%EB%8F%84%ED%99%94/) 프로젝트에는 원래 SVR(Support Vector Regresson)도 모델 후보로 추가하려고 했습니다.

하지만, 여러 회귀모델(XGBoost, LightGBM, RandomForest, Ridge, Lasso, ElasticNet, SVR 등)을 **Bayesian Optimizer** 기반으로 튜닝하여 **Validation MAE**가 가장 낮은 모델을 선택하는 구조를 사용했더니 과거 데이터에 대한 시뮬레이션 과정에서 다음과 같은 현상이 관찰되었습니다.

> - SVR(Support Vector Regression)이 거의 선택되지 않음.
> - 선택되더라도 Ridge나 XGBoost보다 예측 성능이 낮음.
> - 하이퍼파라미터 튜닝을 해도 오히려 Validation Loss가 불안정함

이 글에서는 "왜 SVR이 Validation MAE 모델 선택 구조에서 잘 작동하지 않는가"를 살펴보려고 합니다.

---
## 2. SVR의 기본 구조 복기
[**"Support Vector Regression은 선형모델일까, 비선형모델일까?"**](https://data-bility.github.io/posts/Support-Vector-Regression%EC%9D%80-%EC%84%A0%ED%98%95%EB%AA%A8%EB%8D%B8%EC%9D%BC%EA%B9%8C,-%EB%B9%84%EC%84%A0%ED%98%95%EB%AA%A8%EB%8D%B8%EC%9D%BC%EA%B9%8C/)에서 다룬 내용을 복기해봅니다. 

* SVR은 **마진 최대화**(margin maximization)에 기반한 회귀모형으로, 입력 공간에서 선형평면을 찾되, 오차가 ε 이내인 구간은 무시하는 **epsilon-insensitive loss**를 사용합니다. 
* 즉, 가능한 한 정규화 된 함수로 데이터를 근사하면서, 허용 오차를 넘는 포인트에 대해서만 벌점(C)을 주는 구조입니다.

---
## 3. 수요예측 데이터의 특성

* SVR의 원리는 **노이즈가 적고 구조적 경계가 명확한 문제**에서는 매우 강력하지만, 수요예측 데이터의 특성과는 잘 맞지 않습니다.
* 수요 데이터는 대부분 다음과 같은 특징을 갖습니다.

  | 구분 | 설명 |
  |------|------|
  | **잡음(Noise)** | 이벤트, 재고, 할인, 날씨 등 외생변수가 많음 |
  | **불연속성(Discontinuity)** | 간헐적(intermittent) 또는 0-inflated 수요 존재 |
  | **비선형성(Non-linearity)** | 피처 간 상호작용 강함 (예: 요일×프로모션) |
  | **스케일 다양성(Scale variance)** | SKU별 평균 판매량 차이가 큼 |
  | **비정상성(Non-stationarity)** | 주기나 트렌드가 시점마다 변함 |

즉, **고정된 마진 내에서 평면을 유지하려는 SVR의 특성**이 이러한 동적 구조와 잘 맞지 않습니다.

---

## 4. Bayesian Optimizer Validation MAE 기반 모델 선택 방법에서의 불리함

Bayesian Optimization은 각 하이퍼파라미터 조합에 대해 **평활(smooth)** 한 성능 함수(즉, validation metric이 부드럽게 변하는 함수)를 가정하고 탐색합니다.

그러나 SVR의 주요 하이퍼파라미터 공간은 다음과 같이 **비연속적이고 민감**합니다.

| 파라미터 | 의미 | 문제점                                        |
|-----------|--------|--------------------------------------------|
| `C` | 규제 강도 (over/underfitting 조절) | 작은 변화에도 예측 함수의 support vector 구성이 급변       |
| `epsilon` | 허용 오차 폭 | epsilon 범위 내 데이터 비율이 달라지면 손실 함수의 모양 자체가 바뀜 |
| `gamma` | RBF 커널 폭 | 너무 작으면 과적합, 너무 크면 평탄해짐 — tuning 난이도 높음     |

결과적으로,
- Loss landscape이 매끄럽지 않아 **Bayesian optimizer의 surrogate function**이 잘 근사하지 못함.
- 탐색이 불안정하거나 **local minima에 빠지기 쉬운** 특성을 갖는것으로 보임.

---
## 5. Ridge나 XGBoost보다 예측 성능이 나빴던 이유

타 모델에 비해서 성능이 안좋은 이유에 대해서도 모델이 갖는 특징을 기준으로 생각해봤습니다.

| 항목 | SVR | Ridge / ElasticNet | XGBoost / LightGBM |
|------|------|--------------------|---------------------|
| **모델 구조** | 평면 기반 (마진 중심) | 선형 회귀 (L2/L1 규제) | 트리 기반 (비선형 상호작용) |
| **잡음 내성** | 약함 (outlier에 민감) | 강함 | 매우 강함 |
| **비선형 표현력** | 커널 의존 | 제한적 | 자동 학습 |
| **스케일 민감도** | 높음 | 낮음 | 낮음 |
| **튜닝 안정성** | 낮음 | 높음 | 높음 |

결국, 수요예측의 현실적 문제(비선형, 노이즈, 불연속)에 SVR은 **모델 구조적으로 불리**하고, **튜닝적으로 불안정**합니다.

한편, 문헌에서 공통적으로 지적되는 SVR의 단점 중 하나는 **학습 시간과 메모리 복잡도**입니다.

- **복잡도:** 학습 복잡도는 $ O(n^2) $ ~ $ O(n^3) $, 예측은 $ O(n) $에 근접합니다.<br>
  (데이터 개수가 n 일 때)
- **원인:** 커널 행렬을 전부 계산해야 하며, support vector가 많을수록 계산량이 증가합니다.

실제로 모델을 추가하여 과거 기간에 대해 시뮬레이션 해보았을 때,
1. 학습 시간은 RF 수준으로 오래걸리고
2. 선택 받는 경우는 적고
3. 선택되더라도 성능이 떨어지는 문제로 

후보에서 제외하게 되었습니다.

---
## 6. 결론

- **이론적 측면:** SVR은 epsilon-insensitive loss로 노이즈 내성을 높이려 하지만, 이 손실 구조는 비평활하며 튜닝 민감도가 높다.
- **계산적 측면:** 커널 기반 계산 특성상 계산 복잡도가 높아 학습 시간이 오래 걸린다.
- **탐색적 측면:** Bayesian optimizer 또는 grid search를 이용한 하이퍼파라미터 탐색 시 최적화가 어렵고 시간이 많이 소요된다.
- **실무적 결론:** SVR은 **명확한 패턴을 가진 예측 문제**에서는 여전히 유효하지만, 대부분의 점포/SKU 단위 출하 데이터에 대한 모델로써는 부적합했다.
