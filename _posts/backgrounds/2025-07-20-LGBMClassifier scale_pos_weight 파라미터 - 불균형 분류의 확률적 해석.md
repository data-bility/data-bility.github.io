---
title: LGBMClassifier scale_pos_weight 파라미터 - 불균형 분류의 확률적 해석
date: 2025-07-20 00:00:00 +09:00
categories: [backgrounds]
math: true
---
> - 이 글은 불균형 이진 분류에서 **scale_pos_weight** 파라미터가 LGBMClassifier에서 어떻게 작동하는지 확률, 통계적 관점에서 접근해봅니다. 

---

## 1. 왜 데이터 불균형이 문제가 될 수 있는가?

불균형 이진 분류에서 양성 비율 $r = \frac{\text{pos}}{\text{pos} + \text{neg}}$ 이 매우 작을 때:

- 정확도(Accuracy)는 **음성만 잘 맞춰도** 과대평가될 수 있음.
- 학습 시 손실함수에서 **양성 오차의 기여도가 작아져** 양성 탐지(Recall) 저하.
- 결과적으로 Cut-off를 조정해도 **PR 곡선이 개선되지 않거나, Precision–Recall 트레이드오프가 왜곡**됨.

---
## 2. scale_pos_weight의 수학적 의미

LightGBM/XGBoost 류의 그래디언트 부스팅은 각 데이터 포인트의 손실 기여도를 가중합으로 최적화합니다. 

양성 샘플에 가중치 $ w_+ $ 를 주면, 손실은 대략 다음과 같이 표현됩니다. :

$$
\mathcal{L} = \sum_{i \in \text{neg}} \ell_i + \sum_{j \in \text{pos}} w_+ \cdot \ell_j
$$

**scale\_pos\_weight** 는 보통 다음과 같이 설정합니다. : 

$$
w_+ = k \cdot \frac{N_{neg}}{N_{pos}}
$$

직관적으로 생각해보면 :

- 양성이 희소할수록 $\frac{N_{neg}}{N_{pos}}$ 가 커지므로, **양성 오차의 상대적 중요도**를 키워 학습 단계에서 양성에 더 민감하게 만듭니다.
- 추가 배율 $ k $ 로 과소/과대 보정을 미세 조정하면서 성능을 확인해봐야 합니다.

> **효과** 
> - 일반적으로 Recall 상승, Precision 하강 하는 경향.  
> - 지나친 가중은 **과잉 탐지(FP가 커짐)** 로 F1이 오히려 악화될 수 있습니다.

---

## 3. 그래디언트 관점에서 탐구

이진 로지스틱 손실에서 예측 logit $ f(x) $ 에 대한 그래디언트(1차 미분 term)는

$$
g = (\hat{p} - y)
$$

이며, 양성($y=1$)에 가중 $w_+$ 를 주면 **유효 그래디언트**는

$$
g' =
\begin{cases}
\hat{p} - 0, & y=0 \\w_+ (\hat{p} - 1), & y=1
\end{cases}
$$

- 즉, **양성 샘플의 업데이트가 더 크게 반영**되어, 결정경계가 **양성 쪽으로 이동**합니다.
- Cut-off를 고정(예: 0.7)해도 예측확률 분포 자체가 변하므로 PR 곡선 전체가 이동할 수 있습니다.

---

## 4. 일반적인 사용 방식 

공식 문서 기준으로 발췌해보았습니다.

- **LightGBM 공식 문서 요약**
  - `scale_pos_weight`는 **이진 분류에서 양성 클래스의 가중치(weight of positive class)** 를 의미합니다. ([LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html))
  - 기본값은 `1.0`이며, `is_unbalance=True` 옵션과 **동시에 사용할 수 없습니다.**
  - 공식 문서에서는 `scale_pos_weight` 또는 `is_unbalance` 중 **하나만 설정**할 것을 권장합니다. ([AWS SageMaker LightGBM Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/lightgbm-hyperparameters.html))

- **XGBoost 공식 문서 요약**
  - `scale_pos_weight`는 **양성과 음성 클래스 간의 가중치 균형을 조정**하는 파라미터입니다. ([XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html))
  
  - 일반적으로 다음의 경험적 설정이 권장됩니다.
  
    $$
    \text{scale_pos_weight} = \frac{\text{number of negative samples}}{\text{number of positive samples}}
    $$


  - 이 값은 **불균형 비율의 역수**로서, 클래스 불균형이 심할수록 양성 샘플의 기여도를 높여줍니다.

- **Best Practice**
  - 초기 설정으로 `scale_pos_weight = N_neg / N_pos` 를 사용하고, 이후 k $ \in $  {0.5, 1, 2} 범위 내에서 조정하며 **Precision/F1 기반으로 최적화**.
  - 지나치게 큰 값은 FP 증가로 **Precision 하락**을 유발할 수 있으므로 주의.
  - `is_unbalance` 옵션은 내부적으로 동일한 효과를 주므로, 재현성과 명시적 제어를 원하면 `scale_pos_weight` 설정이 더 권장됩니다.

---

## 5. 코드 예시

```python
from lightgbm import LGBMClassifier

# 불균형 비율 계산
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
base_weight = n_neg / max(n_pos, 1)

for k in [0.5, 1.0, 2.0]:
    clf = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        scale_pos_weight=k * base_weight,
        random_state=42
    )
    clf.fit(X_train, y_train)

```
