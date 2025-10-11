---
title: Support Vector Regression은 선형모델일까, 비선형모델일까?
date: 2023-11-30 00:00:00 +09:00
categories: [backgrounds]
math: true
---

## 1. 질문: Support Vector Regression(SVR)은 선형모델일까, 비선형모델일까?

SVM(Support Vector Machine)을 분류(classification) 문제에서만 접하다가, 회귀(regression) 문제로 확장된 **SVR(Support Vector Regression)** 을 처음 보고 다음과 같은 의문이 생겼습니다.

> "SVR은 선형모델일까, 아니면 비선형모델일까?"

구글링 결과는 다음과 같습니다.  
> **모델의 형태는 선형이지만, 커널을 사용하면 비선형 관계를 학습할 수 있습니다.**

솔직히 아리송하네요. 오늘은 SVR에 대해 수학적으로 접근해보겠습니다. 

---
## 2. 기본 구조: 선형 모델로서의 SVR
> 아래 내용은 참조문헌의 내용을 재가공한 것입니다.

SVR의 기본식은 다음과 같이 표현됩니다.

$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b
$$

전형적인 **선형 회귀모형(linear regression)** 형태입니다.  
즉, 입력 변수 $ \mathbf{x} $에 대해 가중치 $ \mathbf{w} $를 선형 결합한 뒤, 절편 $ b $를 더하는 구조입니다.

다만, SVR은 일반적인 최소제곱(OLS) 회귀와 달리 **epsilon-insensitive loss**를 사용합니다. <br>
즉, 오차가 $ \epsilon $ 이하인 경우는 무시하고, 그 이상만 페널티를 주는 방식입니다. 작은 오차는 신경쓰지 않습니다.

$$
L_{\varepsilon}(y, f(x)) = \begin{cases}
0, & \text{if } |y - f(x)| \le \varepsilon \\
|y - f(x)| - \varepsilon, & \text{otherwise}
\end{cases}
$$

이 손실함수를 최소화하는 문제는 다음과 같이 정규화됩니다.

$$
\min_{\mathbf{w}, b, \xi_i, \xi_i^*}
\frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
$$

subject to

$$
\begin{cases}
y_i - \mathbf{w}^\top \mathbf{x}_i - b \le \varepsilon + \xi_i \\
\mathbf{w}^\top \mathbf{x}_i + b - y_i \le \varepsilon + \xi_i^* \\
\xi_i, \xi_i^* \ge 0
\end{cases}
$$

즉, $ \epsilon$ 영역 내의 오차는 용인하면서도, 전체적인 평면 $ \mathbf{w}^\top \mathbf{x} + b $ 를 가능한 한 평평하게 유지하도록 합니다.

이때까진 명백히 **선형모델**입니다.

---

## 3. 커널 트릭(Kernel Trick): 비선형성을 부여하는 장치

하지만 실제 데이터는 대부분 선형관계로 설명되지 않습니다.

SVR이 강력한 이유는, **"입력 공간(input space)"을 "고차원 특징공간(feature space)"으로 mapping** 하여 비선형 관계를 선형적으로 풀 수 있게 하는 커널 트릭 덕분입니다.

이를 수식으로 보면,

$$
f(\mathbf{x}) = \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b
$$

이때 $K(\mathbf{x}_i, \mathbf{x}) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}) \rangle$ 은 **커널함수(kernel function)** 로, 입력 $\mathbf{x}$를 고차원으로 변환하는 함수 $\phi(\cdot)$를 직접 계산하지 않고도 그 내적(inner product)을 계산해주는 역할을 합니다.

---
## 4. 선형 vs 비선형을 구분하는 기준

| 구분 | 커널함수 | 모델 형태 | 해석 |
|------|-----------|------------|------|
| **선형 SVR** | Linear kernel: $K(\mathbf{x}_i, \mathbf{x}) = \mathbf{x}_i^\top \mathbf{x}$ | $f(x) = w^\top x + b$ | 선형회귀와 유사 |
| **비선형 SVR** | RBF, Polynomial, Sigmoid 등 | $f(x) = \sum_i (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, x) + b$ | 비선형 경계 학습 가능 |

따라서 **SVR은 "기본 구조는 선형"이지만, 커널에 따라 "비선형 관계를 표현"할 수 있는 모델**로 볼 수  있습니다.

---

## 5. 해석적 관점에서 본 SVR의 본질

요약하자면,

- **모델의 구조** : 선형
- **데이터 표현 방식** : 커널을 통해 비선형도 가능
- **학습 방식** : 마진 극대화 기반 최적화
- **결과적 특성** : 선형/비선형 모두 표현 가능

즉, SVR은 "선형성과 비선형성의 경계를 자유롭게 넘나드는 모델" 로 요약 가능합니다. 단순한 회귀 모델보다 **일반화 성능(Generalization Performance)** 이 뛰어난 이유이기도 합니다.

---

## 참고 문헌
- Vapnik, V. (1995). *The Nature of Statistical Learning Theory.* Springer.
- Smola, A. J., & Schölkopf, B. (2004). *A tutorial on support vector regression.* Statistics and Computing, 14(3), 199–222.
