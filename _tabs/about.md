---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---
# 진대현 (Dae-Hyun Jin)
> - ML Engineer / Demand Forecasting @ SSG.COM (2022.03 ~ 현재) <br>
> - Developer Relations @ SSG.COM (2024.05 ~ 현재)

이커머스 수요예측, 품절분류, 가격최적화 문제를 해결하는 ML Engineer입니다. <br>
수요예측 자동화와 가격탄력성 추정을 통해 운영 효율을 높이고, <br>
모델 해석과 인사이트로 고객 경험의 품질을 개선하는 일에 집중합니다.

D/I 본부 Devleoper Reltaions을 겸하고 있습니다. <br>
기술 부채 청산 캠페인을 실시하여 사내 개발문화를 개선하고, <br>
[SSG TECH BLOG](https://medium.com/ssgtech)를 공동 운영합니다.

---

## 경력 요약

| 기간           | 소속                           | 직무                  |
|--------------|------------------------------|---------------------|
| 2022.03 ~ 현재 | SSG.COM D/I 본부 데이터예측팀 | ML Engineer         |
| 2024.05 ~ 현재 | SSG.COM D/I 본부               | Developer Relations |

- PySpark, Scala 기반 수요예측 파이프라인 개발
- TabPFN, TFT, DML, XGBoost, LightGBM, Ridge/Lasso/ElasticNet 활용 모델링
- Croston/Poisson 등 간헐 수요 통계모형 연구 및 ML 전환
- Mesos, Ray 병렬추론 기반 CausalForestDML 가격탄력성 추정
- Airflow 기반 주간 학습 및 일간 예측 자동화 파이프라인 구축
- [SSG TECH BLOG](https://medium.com/ssgtech) 콘텐츠 개발 및 운영


---

## 기술 스택

- **언어:** Python, Scala, SQL
- **ML:** XGBoost, LightGBM, Scikit-learn, PyTorch Forecasting, EconML
- **Data:** PySpark, Pandas, Numpy, SparkSQL, Snowflake, S3, GCP, GCS, BigQuery
- **Infra:** Airflow, Ray, Git
- **협업:** Confluence, Jira, MS365

---


## 주요 프로젝트

### (WIP) 행사·할인 반응을 반영한 단기 수요예측 모델 고도화
> 2025.11 ~ 현재 / 개발 1명, 기여도 100% <br>
> **목적:** 행사·할인 구간에서 급변하는 수요를 기존 통계/트리 기반 모델 대비 예측 성능 향상, zero-inflated SKU의 단기 예측 성능 개선

#### ① Pre-trained model 기반 수요예측 베이스라인 모델 개발, 이벤트 민감형 모델 PoC
- **기술:** PySpark, Python, Ray(GPU Inference)
- **성과**
  - 현재 개발된 Baseline 모델에서, 평시 구간 RMSE/MAE 기준 기존 모델 대비 점포 별 약 5~20% 개선
  - 고가·고변동 SKU에서 수요 급등 구간 추종 성능 향상, 행사 구간 내 예측 성능 극대화 초점으로 고도화 진행중
  - 기존 수요예측 파이프라인에 보조 예측 모델(ensemble 후보) 로 통합 가능성 검증
- **설명:** <br>
  SKU 단위 수요 데이터의 0 다빈도(zero-inflated) 및 행사·할인 시 급격한 분포 변화 문제를 해결하기 위해 Tabular datad에 적합한 Pre-trained model을 도입했습니다.
  Spark에서 생성한 할인(dcrt)·이벤트·캘린더 기반 피처를 활용해, 학습 데이터 내 유사 패턴을 조건부(in-context)로 참조하는 방식으로 단기 수요 반응을 직접 추론하도록 설계했습니다.
  특히 행사 전·중·후 구간에서 기존 OPS/GBDT 모델 대비 예측 지연 및 과소예측 문제를 완화하는 데 초점을 두었습니다.

### 1. 품절상품 분류 및 워크포워드 기반 Cut-off 최적화 로직 개발
> 2025.06 ~ 2025.09 / 개발 1명, 기여도 100% <br>
> **목적:** 품절 발생 확률 조기 예측 및 주간별 threshold 자동 보정 로직 구축

#### ① [주간 품절상품 분류 모델 개발](https://data-bility.github.io/posts/%EC%A3%BC%EA%B0%84-%ED%92%88%EC%A0%88%EC%83%81%ED%92%88-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C%EA%B8%B0/)
- **기술:** PySpark, LightGBM
- **성과**
  - 평균 Precision 약 0.65 / F1-score 약 0.46
  - 수요예측 프로덕트에 품절상품 조기 탐지 기능을 탑재
- **설명:**  
  주간 단위로 발생하는 품절상품(Abst 상품)의 특성을 분류하고, `abst_signal`을 생성해 품절 발생 가능성을 조기 탐지하는 ML 모델을 개발했습니다. <br>
  주차별 품절 발생 패턴과 재고 소진 속도를 feature로 통합, 정량화했으며 품절상품을 조기 탐지합니다.

#### ② [워크포워드 최적화 기반 절대·상대 혼합 Cut-off 로직 개발](https://data-bility.github.io/posts/%EC%9B%8C%ED%81%AC-%ED%8F%AC%EC%9B%8C%EB%93%9C-%EC%B5%9C%EC%A0%81%ED%99%94-%EA%B8%B0%EB%B0%98-%EC%A0%88%EB%8C%80,-%EC%83%81%EB%8C%80-%ED%98%BC%ED%95%A9-%EB%B6%84%EB%A5%98-Cut-off-%EB%A1%9C%EC%A7%81-%EA%B0%9C%EB%B0%9C/)
- **기술:** PySpark, SparkSQL, Weekly Rolling Validation Framework
- **성과:**
  - Cut-off 자동 보정으로 **운영 실적 대비 예측 안정성 향상**
  - threshold 고정 대비 recall 변동폭 **-35% → ±10%** 수준으로 안정화
  - 조기 탐지된 품절상품에 대해 Quantile Regression 모델 접목하여 예측 수준 상향
- **설명:**  
  품절 분류 결과를 활용해 **절대값 cut-off** 와 **상대적 cut-off** 을 결합한 cut-off를 정의하고, 주간 검증 데이터를 이용해 워크포워드 방식으로 자동 보정했습니다. <br>
  이를 통해 threshold drift 문제를 해결하고, 품절 탐지의 민감도(sensitivity)와 안정성 간 균형을 확보했습니다. 

---

### 2. [자동화물류센터 수요예측 고도화 - Poisson XGB, TFT 모델](https://data-bility.github.io/posts/%EC%9E%90%EB%8F%99%ED%99%94%EB%AC%BC%EB%A5%98%EC%84%BC%ED%84%B0-%EC%88%98%EC%9A%94%EC%98%88%EC%B8%A1-%EA%B3%A0%EB%8F%84%ED%99%94-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-Poisson-XGB,-TFT-%EB%AA%A8%EB%8D%B8-%EB%8F%84%EC%9E%85/)
> 2025.03 ~ 2025.06 / 개발 1명, 기여도 100% <br>
> **목적:** 통계적 모델에서 ML 예측체계로 전환, 데이터 부족 문제 해결

- **기술:** XGBoost(`count:poisson`), TFT(Temporal Fusion Transformer), PySpark
- **성과:**
  - 주차별 RMSE **약 -12.8% 개선**, MAPE **약 -9.4% 개선**
  - Poisson objective 기반으로 **과소예측 문제 개선**
- **설명:** NE.O.004 자동화센터 오픈 초기 데이터 부족 상황에서 ML 예측체계로 전환

---

### 3. [입하 CAPA 기반 취소우선순위 ML모델](https://data-bility.github.io/posts/%EC%9E%85%ED%95%98-CAPA%EB%A5%BC-%EA%B3%A0%EB%A0%A4%ED%95%9C-%EC%9E%90%EB%8F%99%EB%B0%9C%EC%A3%BC-%EC%B7%A8%EC%86%8C-%EC%9A%B0%EC%84%A0%EC%88%9C%EC%9C%84-Rank-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C/)
> 2025.01 ~ 2025.02 / 개발 1명, 기여도 100% <br>
> **목적:** 입하 취소의 수리적 타당성, 로직의 해석가능성 확보
- **기술:** Ray, NNLS, Scala, PySpark
- **성과:**
  - NNLS regression (Non-Negative Least-Squares regression) 기반 해석 가능성 확보
  - 수요예측 프로덕트에 자동발주 취소우선순위 기능 및 대시보드 탑재
- **설명:** 자동화센터 CAPA 초과 시 취소우선순위 결정 모델 개발

---

### 4. [주간 워크 포워드 최적화 기반 파라미터 동적 재보정](https://data-bility.github.io/posts/Croston-method-%EC%A3%BC%EA%B0%84-%EC%9B%8C%ED%81%AC-%ED%8F%AC%EC%9B%8C%EB%93%9C-%EC%B5%9C%EC%A0%81%ED%99%94-%EA%B8%B0%EB%B0%98-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%EB%8F%99%EC%A0%81-%EC%9E%AC%EB%B3%B4%EC%A0%95/)
> 2024.09 ~ 2024.10 / 개발 1명, 기여도 100% <br>
> **목적:** 통계적 모델의 고도화
- **기술:** Python, Pyspark, Airflow, Scala
- **성과**
  - α=0.3 고정 방식 대비 예측오차(MAPE) **-12% 개선**
  - 신규 점포 초기 8주차 이내 안정화
- **설명:** Croston method의 smoothing parameter(α)를 주간 사후평가 기반으로 동적 조정

---

### 5. [SSG 1DAY 배송서비스 자동발주예측 ML모델 고도화](https://data-bility.github.io/posts/SSG-1DAY-%EB%B0%B0%EC%86%A1-%EC%84%9C%EB%B9%84%EC%8A%A4-%EC%9E%90%EB%8F%99%EB%B0%9C%EC%A3%BC%EC%98%88%EC%B8%A1-ML%EB%AA%A8%EB%8D%B8-%EA%B3%A0%EB%8F%84%ED%99%94/)
> 2024.05 ~ 2024.07 / 개발 1명, 기여도 100% <br> 
> **목적:** 배송 서비스 유형 특화 ML 모델 도입, 고도화
- **기술:** Ray, XGBoost, LightGBM, Ridge/Lasso/ElasticNet, Airflow, PySpark
- **성과**
  - 이동평균 대비 RMSE **-21%**, SMAPE **-15%** 개선
  - Bayesian Optimization 기반 모델 선택 자동화
- **설명:** Ray cluster 기반 영업점/SKU 단위 ML 학습 및 예측 파이프라인 구축


---

### 6. 가격 최적화 모델 (CausalForestDML)
- **설명:** CausalForestDML 기반 가격탄력성 추정 및 수익최대화 시뮬레이션
- **성과:**
  - (WIP) 모델 학습/예측 결과의 설명가능성을 위한 개선 작업, 그래프 모델 구조 개선 작업 진행중
  - [Double Machine Learning (DML) — 인과추론과 머신러닝의 경계 위에서](https://data-bility.github.io/posts/DML-%EA%B0%9C%EB%85%90/)

---

# 연구 및 교육사항
## 출판물
* Donghee Kim, Daehyun Jin, Changki Kim, Hyun-Goo Kim, & Yung-Seop Lee (2021-11-10). Model Development of Solar Radiation Model Output Statistics based on Satellite Data and UM-LDAPS Data. 한국태양에너지학회 학술대회논문집, 전남. [(Link)](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11199744)
* Yung-Seop Lee, Daehyun Jin, Donghee Kim, Chang Ki Kim, & Hyun-Goo Kim (2021-07-13). A Comparison of Several Time Series Models for Day-ahead Solar Power Generation Forecasting. 한국신·재생에너지학회 학술대회 초록집, 전남. [(Link)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10591841)
* Dae-Hyun Jin, Sung-Hwan Jang, Hee-Kyung Kim, & Yung-Seop Lee (2021). A trend analysis of seasonal average temperatures over 40 years in South Korea using Mann-Kendall test and sen’s slope. 응용통계연구, 34(3), 439-447. [(Link)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11406051)

## 교육사항
* SnowFlake Fundamentals, 2025.08 : [Link](https://data-bility.github.io/posts/SnowFlake-Fundamentals/)

### 동국대학교 일반대학원(서울)
* 2020.03 ~ 2022.02
* 통계학 석사
* 주요 연구 분야 : 데이터 마이닝, 경향성 분석

### 동국대학교(서울)
* 2011.03 ~ 2018.02
* 산업시스템공학, 통계학 복수전공
