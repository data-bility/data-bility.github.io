---
title: Non-negative Linear Regression의 개념 및 사용
date: 2024-11-07 00:00:00 +09:00
categories: [backgrounds]
math: true
---
## 1. 개요

> - **Non-negative Linear Regression (비음수 선형회귀)** 은 회귀계수에 음수가 허용되지 않도록 제약을 가한 모델입니다.
> - 즉, 모든 피처가 타깃 변수에 **양의 방향으로만 기여한다**는 가정을 반영합니다.

---
## 2. 수학적 정의

일반적인 선형회귀는 다음과 같은 최소제곱문제를 풉니다.

$$
\min_{\beta} ||y - X\beta||_2^2
$$

반면, **Non-negative Linear Regression(NNLS)** 은 다음과 같은 제약을 추가합니다.

$$
\min_{\beta \ge 0} ||y - X\beta||_2^2
$$

- 이 제약조건은 각 계수가 "기여도(weight)"로 해석될 수 있도록 하며, 피처가 타깃값을 감소시키는 방향으로 작용하지 않도록 합니다.
- 실질적으로는 [Lawson–Hanson 알고리즘](https://en.wikipedia.org/wiki/Non-negative_least_squares) 을 사용하여 모든 회귀계수를 0 이상으로 유지합니다.
  
---
## 3. 장점 및 활용 포인트

| 구분         | 설명 |
|------------|------|
| **해석 용이성** | 모든 계수가 0 이상이므로, 피처가 얼마나 긍정적으로 기여하는지만 파악 가능 |
| **실무적 관점** | 경영/물류 등 의사결정 현장에서 ‘음의 영향’ 해석이 어려운 경우 유용 |
| **모델 단순성** | 비선형 모델보다 학습, 해석, 배포가 용이 |

---
## 4. 사용 방법

`scikit-learn`의 `LinearRegression`은 `positive=True` 옵션을 제공하여 손쉽게 비음수 회귀를 구현할 수 있습니다.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(positive=True)
model.fit(X, y)
print(model.coef_)
```

---
## 5. 일반 선형회귀 vs 비음수 회귀 비교 실험
- 아래는 모든 진짜 계수는 양수인데, 특징 간 공선성(collinearity)과 노이즈 때문에 OLS가 음의 계수를 뽑아낼 수 있음을 보여주는 예시입니다.
- NNLS는 제약($ \beta >= 0 $) 덕분에 해석 가능한(모두 양수) 계수를 반환하지만, 경우에 따라 약간의 예측 성능 손실이 있을 수 있습니다(편향–분산 트레이드오프).

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(811)
n = 1000

# 강한 공선성: X2 ≈ X1 + 작은 잡음
X1 = np.random.lognormal(mean=0.1, sigma=0.6, size=n)
X2 = X1 + np.random.normal(0, 0.05, size=n)
X3 = np.abs(np.random.normal(1.0, 0.5, size=n))
X4 = np.abs(np.random.normal(0.3, 0.2, size=n))

# 진짜 계수 (모두 양수)
beta_true = np.array([0.5, 0.4, 0.2, 0.1])

# X2에 음의 잡음(엔도지니티) 추가 = 부호 반전 유도
noise = np.random.normal(0, 0.05, size=n)
y = 0.5*X1 + 0.4*X2 + 0.2*X3 + 0.1*X4 - 0.7*X2 + noise

X = np.column_stack([X1, X2, X3, X4])
cols = ["X1_price", "X2_price_like", "X3_margin", "X4_discount"]

# 모델 학습
ols  = LinearRegression(positive=False).fit(X, y)
nnls = LinearRegression(positive=True ).fit(X, y)

coef_df = pd.DataFrame({
  "Feature": cols,
  "True beta": beta_true,
  "OLS beta": ols.coef_,
  "NNLS beta (>=0)": nnls.coef_
})
coef_df

```


| Feature       | True beta | OLS beta   | NNLS beta (>=0) |
| ------------- |-----------|------------|-----------------|
| X1_price      | 0.5       | 0.457      | 0.201           |
| X2_price_like | 0.4       | **-0.256** | **0.000**       |
| X3_margin     | 0.2       | 0.194      | 0.194           |
| X4_discount   | 0.1       | 0.105      | 0.105           |


- OLS 
  - 다중공선성, 그리고 내생성(endogeneity)의 결합으로 인해 계수의 부호가 불안정해질 수 있음.
    - 다중공선성(collinearity): 두 설명변수끼리 너무 비슷해서 "누가 영향을 주는지" 구분이 어려운 문제
    - 엔도지니티(endogeneity): 설명변수가 오차항과 상관되어 "원인과 결과"가 섞인 문제
  - 예측 성능은 유지되지만, 해석상 "가격이 높을수록 매출이 감소한다" 같은 비논리적 결과가 발생할 수 있음.

- NNLS 
  - 부호 제약(beta >= 0)을 통해 부호 뒤집힘 문제를 방지 
  - 약간의 성능 손실이 있을 수 있으나, 해석 가능성과 일관성이 보장됨

--- 

## 6. 결론 요약

- 일반 선형회귀는 데이터 공선성이나 오차의 상관성에 따라 계수 부호가 뒤집히는 현상이 자주 발생합니다.
- Non-negative Linear Regression(NNLS)은 이 문제를 구조적으로 차단하고, 각 feature의 영향 방향을 항상 양수로 유지합니다.
- 실무적으로는 해석가능성을 유용하게 사용 가능할 것으로 보입니다.

