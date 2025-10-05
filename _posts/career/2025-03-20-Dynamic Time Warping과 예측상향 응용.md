---
title: Dynamic Time Warping과 예측상향 응용
date: 2025-03-20 00:00:00 +09:00
categories: [career]
math: true
---

## 1. 기본 개념
> 두 시계열 데이터의 **비선형적 시간왜곡**(non-linear time distortion)을 허용하면서 **유사도를 측정하는 알고리즘**입니다.  
즉, 단순히 시점이 일치하지 않더라도 형태(패턴)가 비슷한 시계열끼리의 거리를 계산할 수 있습니다.

유클리드 거리는 두 시계열의 길이가 같고, 시점이 정확히 일치할 때만 비교가 가능합니다. 하지만 현실의 시계열 데이터는 다음과 같은 특징을 갖는 경우가 있습니다.
- 동일한 패턴이라도 시간 축이 늘어나거나 압축되는 경우
- 일부 구간에서만 속도가 다른 경우

예를 들어, 다음 두 시계열이 있다고 할 때:
- A: [1, 2, 3, 4, 5]
- B: [1, 1, 2, 3, 5, 6]

이 두 시계열은 전체적인 패턴 자체는 비슷하지만, B는 일부 구간에서 더 느리거나 빠르게 진행됩니다.  
이 경우 단순한 유클리드 거리로는 유사도를 올바르게 계산하기 어렵습니다.  
DTW는 이러한 시간 축 왜곡을 보정하여 **두 시계열이 가장 잘 정렬되는 최소 거리**, 즉 **warping path**를 계산합니다.

---

## 2. 작동 원리

DTW는 두 시계열 \( X \)와 \( Y \)의 **warping path** \( W \)를 찾습니다.

$$
X = (x_1, x_2, \dots, x_n), \quad Y = (y_1, y_2, \dots, y_m)
$$

워핑 경로 \( W \)는 다음과 같이 정의됩니다.

$$
W = \{ (i_1, j_1), (i_2, j_2), \dots, (i_K, j_K) \}
$$

단, 다음 제약 조건을 만족해야 합니다.

| 제약 | 설명 |
|------|------|
| **시작/끝 조건 (Boundary)** | 경로는 (1,1)에서 시작해 (n,m)에서 끝나야 함 |
| **단조성 (Monotonicity)** | 인덱스는 감소하지 않아야 함 (시간의 역행 불가) |
| **연속성 (Continuity)** | 한 스텝에서 다음 스텝으로의 이동은 (1,0), (0,1), (1,1) 중 하나 |

즉, 한 시점에서 다음 시점으로 갈 때 한쪽만 움직이거나(시간이 느리게 흐르는 경우), 동시에 움직이는(속도가 동일한 경우) 식으로 **비선형 정렬**이 가능하게 됩니다.

---

## 3. 누적 거리 행렬 (Accumulated Distance Matrix)

DTW는 각 시점 간의 국소 거리(local distance)로부터 **누적 최소 거리**를 동적 계획법(Dynamic Programming)으로 계산합니다.

먼저, 각 시점 간의 국소 거리 행렬 \( C \)를 정의합니다.

$$
C_{i,j} = |x_i - y_j|
$$

그 다음, 누적 거리 행렬 \( D \)는 다음 점화식으로 계산됩니다.

$$
D_{i,j} = C_{i,j} + \min(D_{i-1,j},\; D_{i,j-1},\; D_{i-1,j-1})
$$

여기서
- $D_{i-1,j}$ : 위쪽 셀(시계열 X의 진행)
- $D_{i,j-1}$ : 왼쪽 셀(시계열 Y의 진행)
- $D_{i-1,j-1}$ : 대각선 셀(두 시계열이 동시에 진행)

이며, 가장 작은 누적 거리를 선택해 경로를 구성합니다.  
초기 조건은 다음과 같습니다.

$$
D_{1,1} = C_{1,1}, \quad D_{i,0} = D_{0,j} = +\infty
$$

최종적으로 $D_{n,m}$이 두 시계열 간의 최소 누적 거리(DTW distance)가 됩니다.

---

## 4. Warping Path 복원

최소 누적 거리 $D_{n,m}$이 계산되면, 역방향으로 추적(backtracking)하여 실제 워핑 경로를 복원할 수 있습니다.

1. $(n,m)$ 에서 시작
2. $\min(D_{i-1,j}, D_{i,j-1}, D_{i-1,j-1})$ 방향으로 이동
3. $(1,1)$ 에 도달할 때까지 반복

이 과정을 통해 두 시계열이 **어떤 시점끼리 매칭되었는지**를 알 수 있습니다.

---

## 5. 응용 : 품절 상품의 예측값 상향 보정(DTW 기반 휴리스틱)
> **맥락**  
> 수요예측 운영에서 **품절**이 발생한 날의 판매량은 잠재 수요를 반영하지 못해, 예측 모델이 **체계적으로 과소예측**하는 문제가 생길 수 있습니다.  
> 본 사례는 DTW를 이용해 “보정된 예측 시계열”이 **정상 영업일 수요 시계열**과 더 자연스럽게 정렬되도록 만드는 **휴리스틱 상향 보정** 절차를 설명합니다.


### 5.1 요약
- **목표**: 품절 영향으로 왜곡된 예측값을 **가까운 미래**로 적절히 **이월 또는 상향**하여 잠재 수요를 반영.
- **핵심 아이디어**:
  - 원본 출하량 $y$에 대한 시계열 데이터에서 **품절 발생일자**에 해당하는 데이터를 제거해 $y_{\text{compressed}}$을 만들고,
  - 예측 $pred$도 구간별 비율에 따라 **전이(transfer)** 하여 $pred_{\text{compressed}}$를 생성,
  - 둘의 **DTW 거리**를 기준으로 **segment size**(전이 강도)를 선택해 **보정된 전체 시계열** $pred_{\text{corrected_full}}$을 복원합니다.

### 5.2 절차
1. **품절 정의 및 압축**
- 원본 $y$에서 품절과 관련된 피쳐가 특정 값인 날을 **품절 발생일자**로 정의하고 제거합니다 : $y_{\text{compressed}}$.
2. **구간별 전이 비율 계산**
- 길이 $T$를 segment size $s$로 나눠 구간 $g$마다
  $$
  \alpha_g=\frac{\{t\in g: y_t>0\}}{|g|},\qquad
  \text{transfer_ratio}_g = 1-\alpha_g
  $$<br>
  (구간이 정상일수 비중이 낮을수록 전이 비율이 높아지게끔 모델링)
3. **예측값 전이(이월)**
- 품절일의 $pred_t$를 해당 구간의 **transfer\_ratio**만큼 **다음 정상일**로 이월하고, 정상일의 $pred$는 누적시킵니다.
- 결과: $pred_{\text{compressed}}$ (길이는 $y_{\text{compressed}}$와 동일해집니다.)
4. **원래 길이로 복원**
- $pred_{\text{compressed}}$를 원본 타임라인으로 되돌려 $pred_{\text{corrected_full}}$ 생성(품절일은 0 처리).
5. **DTW 기반 선택**
- 기준값: $\text{DTW}(pred[y>0],\, y[y>0])$ (보정 전).
- 후보 $s \in \{1,\dots,S_{\max}\}$에 대해
  $$
  \text{ratio}(s)=\frac{\text{DTW}\bigl(pred_{\text{corrected\_full}}[y>0],\,y[y>0]\bigr)}{\text{DTW}\bigl(pred[y>0],\,y[y>0]\bigr)}
  $$<br>
  를 계산하고, $\text{ratio}(s)\le \tau$ (예: $\tau=1.10$)를 만족하면서 **가장 개선 효과가 큰** $s^*$를 선택.

> 응용 의도
> - DTW는 시간 늘어짐/당김을 허용하므로, 보정된 예측이 정상일 수요와 **형태적으로 더 비슷**해지면 $\text{DTW}$가 개선됩니다.
> - 단, **과도한 이월**은 과적합/왜곡을 낳을 수 있으므로 $\tau$로 변화 폭을 제한합니다.

### 5.3 간단한 의사코드
~~~text
# 입력: y (실판매), pred (예측), oper_cnt, segment_candidates, threshold τ
# 출력: pred_corrected_full, best_segment

# 1) y>0 인덱스와 y_compressed 구성
nz_idx = [t for t in range(T) if oper_cnt[t] > 0]
y_compressed = y[nz_idx]

best_segment, best_diff = None, -inf
for s in segment_candidates:
    # 2) 구간별 alpha, transfer_ratio 산출
    alpha_seg = segmentwise_nonzero_ratio(y, s)
    # 3) 품절일 pred를 transfer_ratio 만큼 다음 정상일로 이월 → pred_compressed
    pred_compressed = shift_to_next_nonzero(pred, y, alpha_seg, s)
    # 4) 원래 길이로 복원
    pred_full = expand_back(pred_compressed, nz_idx, T)
    # 5) DTW 평가
    base = DTW(pred[oper_cnt>0], y[oper_cnt>0])
    after = DTW(pred_full[oper_cnt>0], y[oper_cnt>0])
    ratio = after / base
    if ratio <= τ and (after - base) > best_diff:
        best_diff = after - base
        best_segment = s
        pred_corrected_full = pred_full
~~~
### 5.4 함수 구현
~~~python
import numpy as np

def dtw_distance_and_path(A, B):
    n, m = len(A), len(B)
    dp = np.full((n+1, m+1), np.inf)
    dp[0,0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(A[i-1] - B[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    dist = dp[n,m]
    # backtracking (0-based 반환이 필요하면 -1 보정)
    path = []
    i, j = n, m
    while i>0 and j>0:
        path.append((i-1, j-1))
        neighbors = [dp[i-1,j], dp[i,j-1], dp[i-1,j-1]]
        move = np.argmin(neighbors)
        if move == 0:   i-=1
        elif move == 1: j-=1
        else:           i-=1; j-=1
    path.reverse()
    return dist, dp, path

def compress_with_partial_scale_by_segment(y, pred, segment_size=5):
    T = len(y)
    nz_idx = [i for i in range(T) if y[i]!=0]  # 정상일
    pred_acc = np.zeros(len(nz_idx), dtype=float)

    n_segments = (T+segment_size-1)//segment_size
    alpha_seg = np.zeros(n_segments, dtype=float)

    for seg in range(n_segments):
        s = seg*segment_size
        e = min((seg+1)*segment_size, T)
        L = e - s
        alpha_seg[seg] = (np.sum(y[s:e]!=0)/L) if L>0 else 1.0

    nz_ptr = 0
    for i in range(T):
        seg_idx = i//segment_size
        transfer_ratio = 1.0 - alpha_seg[seg_idx]
        if y[i]!=0:
            pred_acc[nz_ptr] += pred[i]
            nz_ptr += 1
        else:
            if nz_ptr < len(nz_idx):
                pred_acc[nz_ptr] += transfer_ratio * pred[i]

    y_compressed = np.array([y[i] for i in nz_idx], dtype=float)
    return y_compressed, pred_acc, alpha_seg, nz_idx

def build_corrected_pred_full(y, nz_idx, pred_c):
    T = len(y)
    full = np.zeros(T, dtype=float)
    cptr = 0
    for i in range(T):
        if y[i]!=0:
            full[i] = pred_c[cptr]; cptr += 1
        # else: 품절일은 0 유지
    return full

~~~

### 5.5 시각화 예시
<img src="/assets/img/dtw example.png" width="900px" height="720px" alt="dtw example">

위 그림은 DTW 기반 보정 절차의 전 과정을 시각화한 예시입니다.<br>
(A) Original y vs. pred
- **파란색 (`y`)**: 실제 판매량(실측 수요). 품절일에는 0으로 급락.
- **주황색 (`pred`)**: 모델 예측값으로, 품절 구간에서도 예측이 유지되어 과소예측 형태 발생.
- 두 시계열의 형태는 비슷하지만 **시간축 정렬이 어긋나 DTW 거리**가 큽니다.

(B) Compressed (segment_size = 7)
- 품절일을 제거하고 구간별 전이 비율을 적용한 결과입니다.
- `segment_size=7` 기준으로 구간별 정상일 비율(α)을 계산하여 품절 영향이 큰 구간일수록 예측 이월 비율(transfer_ratio)이 커집니다.
- `y_compressed`와 `pred_compressed`가 구조적으로 더 유사해져 DTW 정렬 경로가 한층 안정화됩니다.

(C) Corrected pred (full timeline) vs. original y
- **빨간색 (`pred_corrected_full`)**: 품절 구간은 0으로 두되, 이월된 예측을 반영한 시계열.
- **보라색 점선**: 정상일 기준으로 이월된 예측이 실제 판매량의 흐름과 더 잘 정렬됨을 보여줌.
- **파란 점선 (`pred_before`)**: 보정 전 예측이며, 형태적으로 더 불안정합니다.

(D) 수치 요약

| 항목 | 값 |
|------|------|
| 선택된 segment_size | 7 |
| 보정 전 DTW distance | 147.51 |
| 보정 후 DTW distance | 169.90 |
| distance ratio (after/before) | 1.0914 (≤ 1.1) |

> **해석:** 보정 후 DTW 거리가 일정 수준(≤ 1.1) 이내에서 유지되므로, 품절 보정이 시계열의 형태적 일관성을 해치지 않고 정상일 수요 흐름에 더 잘 정렬된 예측을 생성했음을 의미합니다.
<br>

### 5.6 작업에 대한 회고
우선, 이 방법은 기본적으로 다음 문제 상황을 조금이나마 완화하기 위함이지 예측 성능이 개선되는 방식은 아님에 주의해야합니다.
> 품절발생! → 잠재 수요 반영 불가 → 체계적 과소예측이 발생한다 → 품절 이후 발주일에 정작 더 낮은 예측값을 활용하게 된다.

따라서, 다음과 같은 한계점이 분명히 존재했습니다.

(1) 품절일의 **잠재 수요(latent demand)** 반영 한계
- 품절일(`oper_cnt=0`)은 판매량이 0으로 관측되지만, 실제로는 **판매할 재고가 없었던 상태**입니다.
- 즉, 해당 시점의 `y=0`은 “수요가 없었다”가 아니라 **“공급이 제한되어 관측되지 않은 수요가 존재했다”**는 의미입니다.
- 보정 과정에서 이를 단순히 이월(transfer)시키는 것은,
  $$
  \text{잠재 수요} = \text{실제 판매} + \text{미판매 수요}
  $$
  중 후자를 **정확히 추정하지 못한 채 형태적으로만 보정하는 것**에 그칠 수 있습니다.
- 따라서 DTW 보정은 **패턴 일치도 개선용(Feature Engineering)** 으로는 유효하지만, **예측 성능 지표 개선**을 직접적으로 보장하지는 않습니다.

(2) 보정 후 시계열의 **스케일 왜곡(scale distortion)**
- `transfer_ratio`에 따라 특정 구간의 예측값이 과도하게 이월되면, 국소적으로 예측치가 “부풀려진(high bias)” 상태가 됩니다.
- 이는 RMSE나 MAE 관점에서 **일부 구간의 과대 예측(over-prediction)** 으로 작용할 수 있습니다.
- 특히 품절 구간이 길거나 연속된 경우, 이월 비율이 누적되면서 **총합(sum) 기준의 오차**가 커질 수 있습니다.

(3) DTW 거리 개선 != 실제 오차 개선
- DTW는 시점 간 **위상(warping)** 을 허용하므로, 일정 부분의 “시간 어긋남”을 보정한 것만으로도 거리 값이 개선될 수 있습니다.
- 그러나 실제 비즈니스 관점(예: 일자별 예측 정확도, 재고 계획)은 **시간 정렬 오차** 자체가 중요하기 때문에, DTW 상의 개선이 반드시 예측 품질 향상을 의미하지는 않습니다.
- 즉, DTW는 “형태적으로 비슷하다”를 보장할 뿐 “언제 얼마가 팔릴지를 맞췄다”를 보장하지 않습니다.

(4) **과보정(Over-correction)** 위험
- segment_size, threshold_ratio(τ) 등의 설정을 너무 느슨하게 두면, DTW 기준으로는 개선된 듯 보여도 **실제 데이터 분포가 왜곡**될 수 있습니다.
- 예를 들어, 품절이 잦은 SKU의 경우 이월 비율이 높아져 전체 예측 시계열이 **평탄화(flatten)** 되거나 **시계열 간 동조화(synchronization)** 가 과도하게 일어날 수 있습니다.

(5) **평가 지표의 다면적 해석 필요**
- 보정 후에는 반드시 **DTW + MAE + MAPE + under_rate/over_rate** 등을 함께 모니터링해야 합니다.
- DTW 개선만으로는 실제 모델의 “정확도 개선”을 단정하기 어렵습니다.
- 특히 **실제 발주량/재고량 시뮬레이션 기반 평가**와 함께 검증해야 보정이 실질적인 효용을 가지는지 판단할 수 있습니다.
