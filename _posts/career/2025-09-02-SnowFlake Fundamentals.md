---
title: Snowflake Fundamentals 교육 이수 후기 - 데이터 클라우드의 구조 이해
date: 2025-09-02 00:00:00 +09:00
categories: [career]
math: false
---

> 본 글은 **Snowflake Fundamentals 공식 교육 과정(2025.08)**을 이수하며 배운 주요 개념과 실습 경험, 그리고 SSG.COM의 데이터 플랫폼 전환 과정에서 느낀 인사이트를 정리한 것입니다.

---

## 1. 교육 개요

Snowflake Fundamentals 과정은 Snowflake의 핵심 구조와 철학을 다루는 **입문자용 필수 교육**으로, <br>
데이터 아키텍처, 컴퓨팅 자원 관리, 보안, 복제(Replication)까지 전반적인 내용을 실습과 함께 학습하는 프로그램이다.

- **교육 기관**: Snowflake 공식 Training Portal
- **기간**: 2025년 8월
- **구성**:
  - Overview & Architecture
  - Snowflake와 연결 (Snowsight, SnowSQL)
  - Data Protection (Cloning, Time Travel, Fail-Safe)
  - SQL Support & Virtual Warehouse
  - Replication, Security, and Cost Management

실습 환경은 Snowflake 계정을 부여받아 **Snowsight UI**, **SnowSQL CLI**, **워크시트 및 대시보드 생성** 등을 직접 수행할 수 있도록 구성되어 있었다.

---

## 2. Snowflake의 철학과 아키텍처

### 2.1 컴퓨팅과 스토리지의 완전한 분리

가장 인상 깊었던 부분은 **Snowflake의 구조적 단순함과 확장성**이었다.  <br>
Snowflake는 `컴퓨팅(Compute)`과 `스토리지(Storage)`를 완전히 분리하여, <br>
서로 다른 팀이 독립적인 가상 웨어하우스(Virtual Warehouse)를 통해 동시에 쿼리를 수행하더라도 리소스 경합이 발생하지 않는다.

> “Snowflake는 리소스 경합을 없애기 위해 컴퓨팅과 스토리지를 분리합니다.”  <br>
> — Fundamentals 교안 중

이는 내가 기존 GCP BigQuery 기반 환경에서 겪었던 **쿼리 슬롯 경합 문제**와 대비되는 구조였다.  <br>
특히 SSG.COM의 Snowflake 전환 프로젝트를 병행하며, 이 개념이 **비용 효율성과 워크로드 분리의 핵심**임을 체감할 수 있었다.

---

### 2.2 마이크로 파티션(Micro-Partition)과 컬럼 기반 스토리지

데이터가 50MB~500MB 단위로 자동 파티셔닝되어 저장되는 구조이다.

Snowflake는 데이터 로드 시 물리적 순서를 유지하면서도, 각 파티션에 메타데이터(MIN/MAX, NULL 수 등)를 저장하여 **쿼리 프루닝(Pruning)**을 자동으로 수행한다. <br>
덕분에 별도의 인덱스나 파티션 키 설계 없이도, 매우 빠른 조회 성능을 보인다.

---

### 2.3 Time Travel & Cloning: “데이터를 되감는가”

이전까지는 복원 목적의 백업을 별도의 ETL로 스케줄링했지만, Snowflake는 기본적으로 모든 데이터 변경을 **불변(Immutable)** 구조로 저장하기 때문에 `CREATE TABLE ... CLONE ...` 문 하나로 과거 시점의 데이터를 즉시 복제할 수 있다.

> `CREATE TABLE T_B CLONE T_A`  
> : 원본 마이크로파티션을 공유하되, 수정 시점부터 새 파티션을 생성하는 구조.

데이터 실험(예: A/B 실험용 테이블 생성) 시 이 개념은 **SSG 데이터예측팀의 분석 반복성**을 높이는 데 매우 유용하게 활용될 수 있다.

---

## 3. 실습에서 배운 것들

### Snowsight 인터페이스
- SQL 워크시트, Notebook, Dashboard를 통합 관리 가능
- Streamlit을 내장하여 간단한 웹앱 형태로 모델 예측값 시각화 가능

### SnowSQL CLI
- `~/.snowsql/config`를 통한 계정별 설정 관리
- `USE ROLE`, `SELECT CURRENT_ROLE()` 등 권한 기반 쿼리 컨텍스트 이해

### Time Travel 실습
- `UNDROP TABLE`로 삭제 복원, `AT(TIMESTAMP => …)` 쿼리로 시점별 데이터 조회
- 실습 중 일부 단계는 의도적 오류를 발생시켜 복구 과정을 직접 체험하도록 구성되어 있었음

---

## 4. 교육을 통해 얻은 인사이트

| 구분 | 기존 환경(GCP BigQuery 등) | Snowflake 전환 후 기대 효과 |
|------|-----------------------------|------------------------------|
| 리소스 격리 | 쿼리 슬롯 공유로 인한 경합 | Virtual Warehouse로 완전 분리 |
| 복제/백업 | 데이터셋 복사 + 스냅샷 스케줄링 | Zero-Copy Cloning, Time Travel |
| 데이터 관리 | 외부 테이블 기반 스테이징 | Stage/Integration 통한 중앙 관리 |
| 권한 모델 | 프로젝트/데이터셋 단위 권한 | 역할(Role) 기반 계층적 권한 부여 |

Snowflake는 단순한 DWH(Data Warehouse)가 아니라, **“Data Cloud”라는 이름에 걸맞게 모든 데이터 자산을 공유 가능한 형태로 조직화하는 플랫폼**이었다. <br>
특히 AI/ML 확장성 측면에서 Snowpark를 통한 Python 기반 연산, Serverless Task와 Stream을 통한 ETL 자동화 구조는 향후 우리 팀의 MLOps에도 적용 가능성을 느꼈다.

---

## 5. 개인적인 소감

> “데이터 엔지니어링과 분석의 경계가 허물어진다.”

Snowflake Fundamentals 과정을 통해 느낀 점은, Snowflake가 단순히 쿼리 속도가 빠른 DWH가 아니라 <br>
**데이터 플랫폼 운영 그 자체의 철학을 바꾼 도구**라는 것이다.

특히,
- SQL, Python, Streamlit이 하나의 인터페이스(Snowsight)로 통합되어 있다는 점,
- 데이터 복제, 버전 관리, 보안 설정이 ‘SQL 한 줄’로 제어된다는 점에서 운영 부담이 대폭 줄어드는 것을 체감했다.

지난 8월에 이미 GCP -> SnowFlake로 한번 더 플랫폼 전환을 진행했기 때문에, <br>
이번 Fundamentals 교육은 그 기초를 다지는 의미 있는 출발점이었다.
