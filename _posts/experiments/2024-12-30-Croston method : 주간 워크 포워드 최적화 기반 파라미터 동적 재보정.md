---
title: Croston method - 주간 워크 포워드 최적화 기반 파라미터 동적 재보정
date: 2024-12-30 00:00:00 +09:00
categories: [experiments]
math: true
---
> - 본 글은 신규 센터(NE.O.004)와 기존 이마트 PP센터의 자동발주 예측 시스템에 대해, **주간 워크 포워드 최적화 기반 파라미터 동적 재보정** 방식을 적용한 사례를 다룹니다.
> - 두 센터 모두 간헐(intermittent)한 수요 특성이 강하여, Croston method의 지수평활계수 $ \alpha $를 **주간 단위 사후평가 기반으로 동적으로 재보정**하는 접근을 통해 안정적인 기초예측 성능을 확보했습니다.
> - Croston 기법 자체의 수식과 원리는 이전 글을 참고하세요: [Croston method: 간헐적 수요 예측을 위한 고전적 접근방법](https://data-bility.github.io/posts/Croston-method-%EA%B0%84%ED%97%90%EC%A0%81-%EC%88%98%EC%9A%94-%EC%98%88%EC%B8%A1%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B3%A0%EC%A0%84%EC%A0%81-%EC%A0%91%EA%B7%BC%EB%B0%A9%EB%B2%95/)
> - Demand Categorization에 대해서는 이전 글을 참고하세요: [ADI, CV-squared를 이용한 Demand Categorization](https://data-bility.github.io/posts/ADI,-CV-squared%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-Demand-Categorization/)

---

## 1. 프로젝트 개요

| 구분 | NE.O.004 자동화물류센터     | 이마트 PP센터                            |
|------|----------------------|-------------------------------------|
| **기간** | 2024.12              | 2024.10 ~ 2024.11                   |
| **상황** | 신규 오픈, 데이터 부족        | 기존 센터, 간헐 수요 비중 높음                  |
| **핵심 목표** | 통계 기반 안정적 기초예측 확보    | 간헐 수요 대응 및 자동발주 확장                  |
| **핵심 접근** | $ \alpha $ 주간 동적 재보정 | $ \alpha $ 재보정 + 주간 사후평가 기반 모형 선택   |
| **결과 요약** | 신규 센터 자동발주 안정화       | 과소예측 6% 감소(개선), 자동발주 비중 2.5% 향상(개선) |

---

## 2. 수요 패턴 및 기초예측 후보

### 2.1 수요 패턴 특성

- NE.O.004 및 PP센터 모두 **Intermittent/Lumpy** 패턴의 SKU가 높은 비중을 차지.
- ADI(Average Demand Interval)와 CV²(Coefficient of Variation squared) 분석 결과,  
  정규 수요 대비 **불규칙적·간헐적 판매** SKU가 다수 확인됨.


| 패턴 | 설명 | 특징 |
|------|------|------|
| Smooth | 지속적 수요 | 안정적 |
| Erratic | 지속되나 변동 큼 | 보정 필요 |
| Intermittent | 불규칙 수요 발생 | Zero-inflated |
| Lumpy | 간격·수량 모두 불규칙 | 예측 난이도 높음 |


---

### 2.2 기초예측 후보

두 센터 모두 다음 3개 기초모형을 동일하게 비교 대상으로 사용하였습니다.

| 모형 | 설명 | 수식 |
|------|------|------|
| **MA28** | 최근 28일 이동평균 | $ \hat{y}_t = \frac{1}{28}\sum_{i=t-27}^t y_i $ |
| **DOW49** | 49일(7주) 요일평균 | $ \hat{y}_{t,d} = \text{mean}(y_d)$ |
| **Croston method** | 간헐 수요 전용 지수평활 | $ \hat{y}_{t+1} = \frac{\hat{z}_t}{\hat{p}_t}$ |


---


## 3. 주간 워크 포워드 최적화 설계
### 3.1 실행 구조

- 스케줄: *월요일 주배치*
- 실행 순서: **`기초모형 주간 사후 평가` -> `최종 모델 선택(Bayesian Optimizer 기반 모델 선택 로직 통합 가능)`**
- 평가 기간: 최근 **365일** (센터 특성 및 가용 데이터에 따라 조정 가능)
- 평가 제외: **명절/특수휴일 제외**, **점포 휴무 제외**, **상품판매운영시간 > 0 (=각 SKU 정상 운영일)**

### 3.2 평가 지표와 선택 규칙

- 기본 지표: **sMAPE**, **MAE**, **정규화 MAE (nMAE = MAE / 평균실적)**
- 선택 규칙(요지):
  1) sMAPE가 정상적으로 계산되는 SKU에 대해서는 **sMAPE 최소** alpha 선택
  2) sMAPE가 비정상(예: 극단치로 200 상한 처리) 또는 결측이면 **nMAE 최소** alpha 선택
  3) nMAE 비교가 곤란하면 **MAE 최소**로 fallback 처리

> 이 규칙은 **월요일 = 생성 기준일**을 기준으로 삼아, **지난 주(또는 정의한 고정 리드타임 오프셋)** 의 예측값과 실제를 맞춰 보는 순수 **사후 평가** 방식입니다.

### 3.3 기초모형 주간 사후 평가 의사코드 (지표 산출 + alpha 선택)

```text
# INPUT
- 판매실적 from feature store
- 예측결과:
  - weeklyma(MA of weekday)
  - moveavg30(MA30)
  - croston_method(alpha=0.1, 0.2, 0.3 각각 산출되어 있다고 가정)
- 휴일/명절 테이블 (평가 제외용)
- critn_dt := 월요일(기준일)

# PREP
- 평가대상일: [critn_dt - 365, critn_dt - 1]  ∩ (명절 X, 점포휴무 X, oper_cnt > 0)
- join(실적 y, 각 기초예측 predt) on (salestr_no, bizcnd_cd, offline_item_id, predt_dt)

# METRIC
- sMAPE := mean(  |y - predt| / ((|y| + |predt|)/2) * 100  )
- MAE   := mean(  |y - predt|  )
- y_avg := mean( y )
- nMAE  := MAE / y_avg

# alpha SELECTION (Croston only)
for each (salestr_no, bizcnd_cd, offline_item_id) as id:
  for alpha in {0.1, 0.2, 0.3}:
    compute sMAPE, MAE, nMAE on evaluation window
  if valid(sMAPE):
    pick alpha with min sMAPE
  else if valid(nMAE):
    pick alpha with min nMAE
  else:
    pick alpha with min MAE
  persist metrics (by id, alpha, critn_dt) -> base_mdl_metric/mdl=croston_method/critn_dt=each monday

# OTHER BASELINES
- 동일 윈도우/필터로 weeklyma, moveavg30 지표도 함께 산출, 저장
```

- 산출물은 base_mdl_metric 영역에 모델별(weeklyma / moveavg30 / croston_method), 기준일별로 저장됩니다. 
- croston_method의 경우에는 이때 '선택된 alpha'에 해당하는 성능 지표가 기록됩니다.

---

## 4. 기초예측 모델 선택 및 자동발주 연계

다음 단계 `최종 모델 선택`에서는 3.에서 저장한 성능 지표들을 불러와, 기초예측 모델( MA28 / DOW49 / croston_method) 중 SKU별로 가장 우수한 후보를 선택합니다.

### `최종 모델 선택` 의사코드
```text
# INPUT
- base_mdl_metric (from base_mdl_eval output):
  id, mdl in {weeklyma, moveavg30, croston_method}, sMAPE, nMAE, MAE, critn_dt
- 기준일 critn_dt := 월요일

# PICK BEST BASELINE PER SKU
for each id:
  # 지표 사용 우선순위: sMAPE 정상 < nMAE < MAE
  candidate := argmin_by_priority(id, mdl, sMAPE, nMAE, MAE)
  mark candidate.mdl as best_baseline

# OUTPUT
- 영업점/SKU 별 최종 선택 모델
```

두 과정을 요약하면 : 
1) 월요일마다 “지난 1년 평가 결과”를 최신화
2) alpha 선정
3) 3개 기초모형 간 베스트 선택
4) 자동발주 가능 여부 산출의 순환을 자동화

## 5. 왜 워크 포워드 최적화 기반으로 alpha를 동적 선택하게 설계 했는가? 

- 데이터 부족 기간의 현실적 제약: 사전 튜닝용 홀드아웃 구성도 어려운 SKU가 많음 
- 안정성: 지난 1년의 운영일 기준 결과를 주기적으로 재평가하면, 과대적합 위험을 줄이면서도 alpha를 환경 변화에 맞춰 서서히 업데이트 가능
- 운영 단순성: alpha 후보를 작게(0.1/0.2/0.3) 두면 계산비용, 운영복잡도를 관리하면서도 반응성(최근성) <-> 안정성(평활)의 트레이드오프를 SKU별로 자동 보정
  - 한편, 문헌 상으로도 croston method의 alpha 값은 최대 0.3 정도로 사용하는 것이 일반적이었음.

## 6. 성과


| 구분        | NE.O.004 센터                   | 이마트 PP센터                      |
| --------- |-------------------------------|-------------------------------|
| **성과 요약** | 신규센터 자동발주 안정화                 | 자동발주 수량 비율 +2.5%              |
| **추가 효과** | ML 도입 전 안정적 기초예측              | 과소예측률 6% 감소                   |
| **핵심 기술** | $ \alpha $ 주간 동적 재보정 | $ \alpha $ 재보정 + 주간 사후평가 기반 모형 선택   |


- 공통 : 기초모형 후보 확장, 파라미터 재보정 방식 도입
  - Croston alpha의 주간 재선정으로 운영 SKU별 예측 민감도 자동 보정
- 신규 오픈 NE.O.004 센터
  - 자동발주 수요예측 서비스 오픈(2024.12)
  - ML 학습 파이프라인 도입 전까지 안정적 기초예측 제공
  - 기존 Bayesian Optimizer 로직의 확장성을 고려하여 설계했으므로, 이후 ML(예: XGBoost/Ridge/ElasticNet) 기반 최적화 파이프라인으로 무리 없이 전환 가능
- 기존 이마트 PP센터
  - 자동발주 비율 +2.5% 향상(개선)
  - 과소예측률 6% 감소(개선)


## 7. 회고 및 한계
### 7.1 한계
- alpha 후보가 {0.1, 0.2, 0.3}로 제한됨 -> 연속값 최적화에 비해 정밀도는 낮을 것임.
- sMAPE 특성상 0 근처에서 불안정 -> 보정(상한 처리 등)과 nMAE 폴백 규칙이 필수적이었음.
- 수요 급변 시점에는 주간 단위 갱신만으로는 반응이 늦을 수 있음.

### 7.2 개선 아이디어
- SKU별 변동성 레짐 탐지 후 일시적으로 alpha 상향 (예: 이벤트/프로모션 구간)
- 가중 sMAPE / 비용가중 MAE(under/over 비용 차등)로 선택 규칙 고도화
- 일정 데이터 축적 뒤, 베이지안 최적화+크로스밸리데이션으로 ML 전환 및 혼합(blending)
