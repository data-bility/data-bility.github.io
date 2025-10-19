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
- XGBoost, LightGBM, Ridge/Lasso/ElasticNet, TFT, DML 모델 활용
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

### 1. 품절상품 분류 및 워크포워드 기반 Cut-off 최적화 로직 개발
> **목적:** 품절 발생 확률 조기 예측 및 주간별 threshold 자동 보정 로직 구축

#### ① [주간 품절상품 분류 모델 개발](https://data-bility.github.io/posts/%EC%A3%BC%EA%B0%84-%ED%92%88%EC%A0%88%EC%83%81%ED%92%88-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C%EA%B8%B0/)
- **설명:**  
  주간 단위로 발생하는 품절상품(Abst 상품)의 특성을 분류하고, `abst_signal`을 생성해 품절 발생 가능성을 조기 탐지하는 ML 모델을 개발했습니다. <br>
  주차별 품절 발생 패턴과 재고 소진 속도를 feature로 통합해 품절 전조(early-signal)를 정량화했습니다.
- **기술:** PySpark, LightGBM
- **성과:**
  - F1-score **+11%p**, Precision **+9%p**, Recall **+12%p** 개선
  - 품절 조기 탐지 정확도 향상 및 오탐율(False Positive) 감소
  - `abst_signal` 누적값을 feature로 활용해 **다음 주 품절 가능성 예측 정확도 향상**

#### ② [워크포워드 최적화 기반 절대·상대 혼합 Cut-off 로직 개발](https://data-bility.github.io/posts/%EC%9B%8C%ED%81%AC-%ED%8F%AC%EC%9B%8C%EB%93%9C-%EC%B5%9C%EC%A0%81%ED%99%94-%EA%B8%B0%EB%B0%98-%EC%A0%88%EB%8C%80,-%EC%83%81%EB%8C%80-%ED%98%BC%ED%95%A9-%EB%B6%84%EB%A5%98-Cut-off-%EB%A1%9C%EC%A7%81-%EA%B0%9C%EB%B0%9C/)
- **설명:**  
  품절 분류 결과를 활용해 **절대값 기준(oper_cnt, dd_inv_dcnt)** 과 **상대 기준(abst_signal 비율)** 을 결합한 cut-off를 정의하고, 주간 검증 데이터를 이용해 워크포워드 방식으로 자동 보정했습니다. <br>
  이를 통해 threshold drift 문제를 해결하고, 품절 탐지의 민감도(sensitivity)와 안정성 간 균형을 확보했습니다.
- **기술:** PySpark, SparkSQL, Weekly Rolling Validation Framework
- **성과:**
  - Cut-off 자동 보정으로 **운영 실적 대비 예측 안정성 향상**
  - threshold 고정 대비 recall 변동폭 **-35% → ±10%** 수준으로 안정화
  - 자동발주 품목의 품절 탐지 precision **+7%p** 개선

---

### 2. [자동화물류센터 수요예측 고도화 - Poisson XGB, TFT 모델](https://data-bility.github.io/posts/%EC%9E%90%EB%8F%99%ED%99%94%EB%AC%BC%EB%A5%98%EC%84%BC%ED%84%B0-%EC%88%98%EC%9A%94%EC%98%88%EC%B8%A1-%EA%B3%A0%EB%8F%84%ED%99%94-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-Poisson-XGB,-TFT-%EB%AA%A8%EB%8D%B8-%EB%8F%84%EC%9E%85/)
- **설명:** NE.O.004 자동화센터 오픈 초기 데이터 부족 상황에서 ML 예측체계로 전환
- **기술:** XGBoost(`count:poisson`), TFT(Temporal Fusion Transformer), PySpark
- **성과:**
  - 주차별 nRMSE **-18% 개선**, MAPE **-14% 개선**
  - 기존 Croston 대비 cold-start SKU 커버리지 **+25% 확대**
  - Poisson objective 기반으로 **과소예측 문제 개선**

---

### 3. [입하 CAPA 기반 취소우선순위 ML모델](https://data-bility.github.io/posts/%EC%9E%85%ED%95%98-CAPA%EB%A5%BC-%EA%B3%A0%EB%A0%A4%ED%95%9C-%EC%9E%90%EB%8F%99%EB%B0%9C%EC%A3%BC-%EC%B7%A8%EC%86%8C-%EC%9A%B0%EC%84%A0%EC%88%9C%EC%9C%84-Rank-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C/)
- **설명:** 자동화센터 CAPA 초과 시 취소우선순위 결정 모델 개발
- **기술:** Ray, NNLS, Scala, PySpark
- **성과:**
  - NNLS(Non-Negative LinearRegression) 기반 해석 가능성 확보
  - 자동발주로 인한 품절율 **-2.5% 감소(개선)**

---

### 4. [주간 워크 포워드 최적화 기반 파라미터 동적 재보정](https://data-bility.github.io/posts/Croston-method-%EC%A3%BC%EA%B0%84-%EC%9B%8C%ED%81%AC-%ED%8F%AC%EC%9B%8C%EB%93%9C-%EC%B5%9C%EC%A0%81%ED%99%94-%EA%B8%B0%EB%B0%98-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%EB%8F%99%EC%A0%81-%EC%9E%AC%EB%B3%B4%EC%A0%95/)
- **설명:** Croston method의 smoothing parameter(α)를 주간 사후평가 기반으로 동적 조정
- **성과:**
  - α=0.3 고정 방식 대비 예측오차(MAPE) **-12% 개선**
  - 신규 점포 초기 8주차 이내 안정화

---

### 5. [SSG 1DAY 배송서비스 자동발주예측 ML모델 고도화](https://data-bility.github.io/posts/SSG-1DAY-%EB%B0%B0%EC%86%A1-%EC%84%9C%EB%B9%84%EC%8A%A4-%EC%9E%90%EB%8F%99%EB%B0%9C%EC%A3%BC%EC%98%88%EC%B8%A1-ML%EB%AA%A8%EB%8D%B8-%EA%B3%A0%EB%8F%84%ED%99%94/)
- **설명:** Ray cluster 기반 영업점/SKU 단위 ML 학습 및 예측 파이프라인 구축
- **기술:** Ray, XGBoost, LightGBM, Ridge/Lasso, Airflow, PySpark
- **성과:**
  - 이동평균 대비 RMSE **-21%**, SMAPE **-15%** 개선
  - Bayesian Optimization 기반 모델 선택 자동화


---

### 6. [가격 최적화 모델 (CausalForestDML)](https://data-bility.github.io/posts/price-optimization-dml/)
- **설명:** CausalForestDML 기반 가격탄력성 추정 및 수익최대화 시뮬레이션
- **성과:**
  - 현재 개선작업 진행중
  - [Double Machine Learning (DML) — 인과추론과 머신러닝의 경계 위에서](https://data-bility.github.io/posts/DML-%EA%B0%9C%EB%85%90/)

---

# 연구 및 교육사항
## 출판물
* Donghee Kim, Daehyun Jin, Changki Kim, Hyun-Goo Kim, & Yung-Seop Lee (2021-11-10). Model Development of Solar Radiation Model Output Statistics based on Satellite Data and UM-LDAPS Data. 한국태양에너지학회 학술대회논문집, 전남. [(Link)](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11199744)
* Yung-Seop Lee, Daehyun Jin, Donghee Kim, Chang Ki Kim, & Hyun-Goo Kim (2021-07-13). A Comparison of Several Time Series Models for Day-ahead Solar Power Generation Forecasting. 한국신·재생에너지학회 학술대회 초록집, 전남. [(Link)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10591841)
* Dae-Hyun Jin, Sung-Hwan Jang, Hee-Kyung Kim, & Yung-Seop Lee (2021). A trend analysis of seasonal average temperatures over 40 years in South Korea using Mann-Kendall test and sen’s slope. 응용통계연구, 34(3), 439-447. [(Link)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11406051)

## 교육사항
* SnowFlake Fundamentals, 2025.08

### 동국대학교 일반대학원(서울)
* 2020.03 ~ 2022.02
* 통계학 석사
* 주요 연구 분야 : 데이터 마이닝, 경향성 분석

### 동국대학교(서울)
* 2011.03 ~ 2018.02
* 산업시스템공학, 통계학 복수전공
