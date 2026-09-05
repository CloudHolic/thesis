# VSRG 난이도 측정 파일럿 구현 명세

## 문서의 목적과 구현자의 책임

이 문서는 새 레포를 처음부터 구축할 코딩 에이전트를 위한 handoff 명세다. 연구의 전체 철학은 `RESEARCH_OVERVIEW.md`에 있으며, 여기서는 현재 파일럿에서 **무엇을 왜 구현해야 하는지**, 무엇을 임의로 바꾸면 안 되는지에 집중한다.

이 파일럿은 완성형 chart encoder를 만드는 단계가 아니다. 다음 두 질문을 검증한다.

1. ZOI Beta-IRT, 플레이어 능력 주변화, MAP/ELBO 추론이 의도한 확률모형대로 작동하는가?
2. 학습된 문항 파라미터 `τ`에서 VSRG 난이도를 어떤 함수로 읽어야 하는가?

일반적인 ML 관습에 맞춘다는 이유로 likelihood를 단순화하거나, SR을 label로 넣거나, `θ`를 사용자 embedding으로 바꾸지 말 것. 그런 변경은 구현 개선이 아니라 연구 질문 변경이다.

작업을 시작하기 전에 반드시 §10 금지사항과 §11 미결사항을 읽는다.

## 1. 파일럿의 범위와 완료 조건

구현 범위는 다음과 같다.

- random pool 데이터의 선행 진단 쿼리와 person-major 데이터셋 구성
- 채보 이벤트 파싱 및 지정된 단순 feature 추출
- 정확한 ZOI Beta-IRT log-kernel
- `θ ~ Normal(0,1)`에 대한 Gauss-Hermite 주변화
- 동일 인터페이스의 ID, linear, MLP encoder
- 이 프로젝트에서 “MLE 버전”이라 부르는 MAP 목적함수와 full-covariance ELBO
- 합성 데이터 parameter recovery
- accuracy와 score 양쪽의 fit 및 검증표
- multi-key linking 가능성에 따른 실험 범위 결정

최소 완료 조건은 합성 recovery가 통과하고, ID encoder 실데이터 기준선과 linear/MLP 결과를 동일한 평가 파이프라인에서 비교할 수 있는 것이다. 실데이터 loss 감소만으로 완료 처리하지 않는다.

## 2. 확률모형

### 2.1 기호와 좌표계를 섞지 말 것

인코더 또는 자유 문항 테이블이 내는 제약 없는 최적화 좌표를 `z_i`라 한다.

```text
z_i = (ℓa_i, b_i, ω_i, γ_0i, λ_i) ∈ R^5
```

likelihood가 사용하는 문항 파라미터를 `τ_i`라 한다.

```text
τ_i = (a_i, b_i, ω_i, γ_0i, γ_1i)
```

결정론적 변환 `T: z → τ`는 정확히 다음과 같다.

```text
a_i   = exp(ℓa_i)
b_i   = b_i
ω_i   = ω_i
γ_0i  = γ_0i
γ_1i  = γ_0i + exp(λ_i)
```

따라서 `a_i > 0`, `γ_1i > γ_0i`가 자동으로 성립한다. `T`는 학습 가능한 층이 아니다. `γ_1 > γ_0`에 별도 penalty를 추가하지 않는다.

출력은 데이터에 `y=0`이 없어도 항상 5차원이다. `γ_0`를 제거하지 않는다.

### 2.2 `b`의 부호 규약

이 명세의 선형 예측자는 다음과 같다.

```text
η_pi = a_i θ_p + b_i
```

`a_i > 0`이므로 `θ`가 커질수록 Beta 성분 평균이 커진다. 같은 `θ`에서 `b_i`가 커져도 평균 성과가 커진다. 따라서 `b_i` 자체는 부호상 **easiness/intercept**이며, 값이 클수록 쉬운 방향이다. 변수명을 관례적으로 `b`로 유지하되 코드 주석과 문서에서 이를 “difficulty”라고 부르지 않는다.

난이도 방향의 threshold는 다음과 같이 읽는다.

```text
b*_i(y) = (logit(y) - b_i) / a_i
        = (logit(y) - b_i) · exp(-ℓa_i)
```

`b*`가 클수록 해당 성과 수준에 더 높은 능력이 필요하므로 어려운 방향이다. 부호를 임의로 뒤집거나 `η=a(θ-b)` 규약과 혼용하지 않는다.

### 2.3 정확한 ZOI Beta-IRT kernel

응답 `y_pi ∈ [0,1]`는 0의 점질량, `(0,1)`의 Beta 밀도, 1의 점질량으로 구성된다. 아래 식을 source of truth로 구현한다.

문항 `i`, 능력 `θ`에 대해:

```text
η       = a_i θ + b_i
G_0     = sigmoid(γ_0i - a_i θ)
G_1     = sigmoid(γ_1i - a_i θ)

π_0     = G_0
π_B     = G_1 - G_0
π_1     = 1 - G_1

α_B     = exp(( η + ω_i) / 2)
β_B     = exp((-η + ω_i) / 2)
```

kernel은 다음과 같다.

```text
k(y | θ, τ_i) =
    π_0                                      if y = 0
    π_B · BetaPDF(y; α_B, β_B)              if 0 < y < 1
    π_1                                      if y = 1
```

따라서 log-kernel은 branch별로 직접 계산한다.

```text
log k =
    log G_0                                                   if y = 0
    log(G_1 - G_0) + log BetaPDF(y; α_B, β_B)                if 0 < y < 1
    log(1 - G_1)                                              if y = 1
```

`jnp.where`만 믿고 모든 branch의 표현식을 먼저 평가하지 말 것. 프레임워크의 평가 방식 때문에 `y=0` 또는 `y=1`에서 Beta log-density가 `inf`/`nan`을 만들 수 있다. 안전한 interior surrogate로 Beta 항을 계산한 뒤 mask하거나, 실제 branch 제어를 사용한다.

`log(G_1-G_0)`는 작은 차이에서 안정적인 `logdiffexp`/`log1mexp` 형태로 계산한다. `log(1-G_1)`도 stable log-sigmoid를 사용한다. 논문의 결합 점수 `s`를 경유하지 말고 `log k`를 직접 구현한다.

### 2.4 `ω`의 정확한 의미

`ω_i`는 ZOI Beta-IRT의 **dispersion parameter**다. 독립적인 Beta precision `φ`라고 부르지 않는다.

Beta 성분 평균은 다음과 같다.

```text
μ_B = α_B / (α_B + β_B) = sigmoid(η)
```

그러나 실제 conditional precision은 다음과 같아 `θ`와 `η`에도 의존한다.

```text
φ(θ) = α_B + β_B
     = 2 exp(ω_i / 2) cosh(η / 2)
```

따라서 `ω`는 평균 위치와 완전히 독립된 precision이 아니다. 다른 조건이 같을 때 `ω`가 커지면 Beta 성분이 더 집중되는 방향이지만, 결과 해석에서는 난이도와 별개의 response-dispersion/일관성 축 후보로 취급한다.

인플레이션은 정확히 0 또는 정확히 1인 값만 처리한다. 경계 근처의 값은 Beta 성분이 처리한다.

## 3. 플레이어 능력 `θ`의 주변화

파일럿에서는 다음을 고정한다.

```text
θ_p ~ Normal(0,1)
```

이는 IRT 척도의 위치·크기를 고정하는 역할과 함께 latent population shape를 표준정규로 두는 모델 가정이다. 현재 파일럿에서는 추정하거나 변경하지 않는다. 사용자별 `θ_p` 또는 `q(θ_p)`를 만들지 않는다.

person `p`의 관측 문항 집합을 `O_p`라 할 때 주변 log-likelihood는 다음과 같다.

```text
log p(y_p | τ)
= log ∫ [∏_{i∈O_p} k(y_pi | θ, τ_i)] Normal(θ;0,1) dθ
```

Gauss-Hermite quadrature로 계산한다.

```text
log p(y_p | τ)
≈ logsumexp_q [log w_q + Σ_{i∈O_p} log k(y_pi | θ_q, τ_i)]
```

표준 Hermite node `x_q`, weight `w̃_q`를 다음처럼 변환한다.

```text
θ_q = sqrt(2) · x_q
w_q = w̃_q / sqrt(π)
```

`Q ∈ {11, 21, 31, 41}` 민감도 검사를 반드시 수행한다. 손실은 cell-major가 아니라 **person-major**로 분해한다.

## 4. 사전분포와 추론 목적함수

### 4.1 prior 좌표와 수치 정의

MAP와 ELBO 모두 동일한 제약 없는 `z` 좌표에 prior를 둔다.

```text
p(z_i) = Normal(0, 10² I_5)
```

코드 수준에서는 각 좌표에 독립적인 `Normal(loc=0, scale=10)`이다. `10`을 variance로 해석하지 않는다. prior는 `τ=(a,b,ω,γ_0,γ_1)`에 직접 걸지 않는다. `T`의 Jacobian을 추가하지 않는다. 이는 처음부터 `z` 공간에서 정의한 프로젝트 prior다.

### 4.2 MLE 버전이라고 부르는 MAP

프로젝트 명명 규약을 유지하여 코드와 보고서에서는 “MLE 버전”이라 부른다. 실제 목적함수에는 prior가 있으므로 통계적으로는 MAP이다.

```text
L_MLE
= -Σ_p log p(y_p | {T(z_i)}) - Σ_i log p(z_i)
```

prior는 항상 켠다. 최소한의 발산 방지 장치다.

### 4.3 full-covariance ELBO 버전

각 문항의 변분 사후는 제약 없는 공간의 5차원 Gaussian이다.

```text
q(z_i | chart_i) = Normal(μ_i, L_i L_iᵀ)
```

`L_i`는 양의 대각을 갖는 lower-triangular Cholesky factor여야 한다. 대각 transform과 numerical floor의 정확한 값은 §11 미결사항으로 남긴다. 구현자가 임의의 상수를 확정하지 않는다.

최소화할 손실은 다음과 같다.

```text
L_ELBO
= -Σ_p E_q[log p(y_p | {T(z_i)})]
  + λ_KL Σ_i KL(q(z_i | chart_i) || p(z_i))
```

KL은 Gaussian끼리의 closed form을 사용한다. `q`를 `τ` 공간에 직접 두지 않는다.

MAP와 VI는 동일한 generative model과 동일한 `p(z)`에 대한 두 추론 방식이다. 구현 인터페이스는 공유하되, 정상적인 Gaussian ELBO가 분산 0의 점질량에서 그대로 MAP로 수렴한다고 가정하지 않는다.

full covariance를 유지하는 이유는 다음과 같다.

- `b*(y)`의 불확실성이 `ℓa`–`b` 공분산에 의존한다.
- 경계 관측이 부족할 때 `(γ_0, λ)`에 강한 상관과 flat direction이 생길 수 있다.

mean-field로 바꾸지 않는다.

### 4.4 KL annealing과 minibatch scaling

`λ_KL`은 0에서 1로 warm-up할 수 있으나 최종 평가는 반드시 1에서 한다. 영구적으로 작은 값을 사용하지 않는다.

person minibatch를 사용할 때 likelihood 합에는 `N_persons / batch_size`를 곱해 full-data scale을 복원하거나, 동치인 방식으로 KL과의 비율을 맞춘다. 우도는 person 합이고 KL은 item 합이라는 점을 코드에 명시한다.

한 optimization step에서는 필요한 각 item의 `z_i`를 한 번 reparameterized sample하고 해당 minibatch의 모든 person term에서 공유한다. 이것이 이 프로젝트의 MC estimator 정책이다. 한 사람의 joint likelihood 안에서는 반드시 동일한 item sample을 사용한다.

## 5. 인코더 인터페이스

모든 인코더는 같은 `z` 인터페이스를 제공한다.

| 단계 | 입력 | 역할 |
|---|---|---|
| ID | item ID/one-hot에 동치인 embedding table | 문항별 자유 파라미터, 도달 가능한 fit의 상한 |
| linear | 지정 feature vector | 파이프라인·부호·계수 디버깅 기준선 |
| MLP | 동일 feature vector | 현 단계의 주력 feature encoder |
| 향후 | 채보 직접 표현 | 이번 파일럿 구현 대상 아님 |

MAP head는 `μ=z`만 출력한다. ELBO head는 `(μ,L)`을 출력한다. ID와 feature 모델은 likelihood, quadrature, prior, optimizer와 평가 코드를 공유해야 한다. ID만 별도의 통계모형으로 구현하지 않는다.

ID와 feature 버전을 나란히 실행한다. ID–feature 격차는 선택 feature가 response-derived 문항 구조를 얼마나 설명하는지 나타낸다.

### 5.1 feature 계산 규칙

rate를 feature 계산 전에 timestamp에 적용한다. DT는 모든 시각에 `1/1.5`를 곱한 뒤 그 결과에서 아래 feature를 계산한다.

| 이름 | 정의 | log 허용 | 활성 조건 |
|---|---|---|---|
| `D` | `(N_tap + N_LN) / T` | 예 | 항상 |
| `T` | 첫 노트 시작부터 마지막 노트 끝까지, LN은 tail 사용, 중간 공백 포함 | 예 | 항상 |
| `N` | `N_tap` | 예 | 항상 |
| `R` | `N_LN / (N_tap + N_LN)` | 아니오 | 항상 |
| `K` | 키 수 | 예 | 다중 키모드일 때 |

feature spec은 `(name, formula, log_allowed, active_condition)` 레지스트리로 관리한다. 모든 feature에 log를 일괄 적용하지 않는다. 0 가능 feature의 변환은 안전해야 하며, 정확한 처리 선택을 설정과 테스트에 드러낸다.

`D`의 LN 가중치는 1이다. `T`에서 중간 공백을 제거하지 않는다. `D`와 `N`은 각각 처리 속도와 누적 판정 이벤트 수를 나타내므로 둘 다 둔다. `R`은 tap이 0인 실제 채보에서도 유한하도록 위 정의를 유지한다.

`feature_scale ∈ {raw, log}`는 단순 전처리 튜닝이 아니라 두 가설 공간의 비교다. 기본은 log, raw는 대조군이다. 계수 해석은 raw, 성능 비교는 log를 우선한다. log 좌표의 공선성 때문에 개별 계수를 과해석하지 않는다.

### 5.2 키 수 `K`

`K`는 별도 feature로 두고 log를 허용한다. `D`를 컬럼당 밀도 `N/(T·K)`로 하드코딩하지 않는다. linear log 모형에서 `log D` 계수 `α`, `log K` 계수 `κ`를 분리해 `κ/α`를 조사한다. 예상 방향은 `κ > -α`다.

8키 초과의 꺾임 항은 이번 단계에서 구현하지 않는다. 향후 직접 인코더는 가변 `K`를 전제로 하며 고정 폭 column one-hot을 사용하지 않는다.

## 6. 데이터 파이프라인

### 6.1 소스와 사용 범위

홈랩 PostgreSQL 18(Docker)에 적재된 osu! 공개 덤프의 2026년 1–6월 monthly snapshot, top-10000 및 random-10000 pool 가운데 **random pool만 사용한다**. top+random 혼합은 사용하지 않는다.

스키마와 실제 column/function name은 Obsidian vault의 원본 노트를 확인한다. 이 문서만 보고 이름을 추측해 쿼리를 작성하지 않는다.

### 6.2 응답변수

accuracy와 ScoreV1 score를 모두 사용해 별도로 비교한다.

- accuracy: `[0.9,1.0]`에 압축되어 있으며 1.0의 실제 점질량이 있다.
- score: 1M 상한이며 더 넓게 분포한다. HT는 500k이므로 정규화 규칙이 필요하다.

ScoreV1의 Bonus meter 때문에 miss가 뭉치면 흩어진 경우보다 점수가 높을 수 있다. miss pattern에 대한 순진한 해석을 사용하지 않는다.

### 6.3 중복 제거와 저장 구조

`(user_id, beatmap_id, rate_group)`별 최고 결과 하나만 남긴다. 이는 iid 반복 관측이 아니라 최고값 순서통계량을 택하는 것이므로 구조적 편향으로 기록한다.

채보 파싱 원본은 `(timestamp, column, type)` event list로 저장한다. feature와 향후 grid는 여기서 파생한다.

응답은 다음 person-major ragged 구조로 저장한다.

```text
(person_id, [item_ids], [responses])
```

### 6.4 선행 쿼리 1: multi-key linking

가장 먼저 random pool에서 4K와 7K 양쪽에 일정 수 이상 응답한 사용자 수를 구한다. 4K와 7K에는 공통 item이 없으므로 두 모드를 모두 플레이한 사람이 척도를 연결한다.

- 충분하면 4K+7K 혼합 fit을 진행한다.
- 부족하면 mode별 독립 filter 대신 linking user를 보존하는 통합 k-core를 시도한다.
- 그래도 부족하면 mode를 따로 fit하고 mode 간 절대 난이도 비교와 `K` 계수 해석을 포기한다. mode 내 검증만 수행한다.

“충분”의 수치는 아직 확정되지 않았다. 임의로 정하지 말고 결과를 보고 사용자와 결정한다.

### 6.5 선행 쿼리 2: 양방향 k-core 곡선

item 응답 수 threshold `N`과 person 응답 수 threshold `M`을 수렴할 때까지 번갈아 적용한다. threshold sweep마다 다음을 기록한다.

- item 수와 item당 응답 수 중앙값의 trade-off
- 확보되는 동일 곡 NM/DT/HT rate-group pair 수
- 키모드별 분포
- linking user 보존량

### 6.6 holdout

- cell holdout: 응답 일부를 숨겨 likelihood와 calibration 검증
- item holdout: 채보 전체를 숨겨 amortization 검증. 규모가 허용하면 반드시 포함

## 7. 합성 recovery

실데이터 fit 전에 반드시 수행한다.

1. 명시된 prior 또는 통제된 parameter grid에서 `z_i`를 생성한다.
2. `T(z_i)`로 `τ_i`를 만든다.
3. `θ_p ~ Normal(0,1)`을 생성한다.
4. §2.3의 정확한 mixture kernel에서 response를 생성한다.
5. person-major 데이터로 변환해 실제 학습 코드와 동일한 경로로 fit한다.
6. true `z`, true `τ`, 추정값과 파생 난이도 후보를 비교한다.

검증 항목:

- branch별 kernel normalization과 finite log-probability
- `T`의 positivity/order constraint
- `Q ∈ {11,21,31,41}`에 따른 objective와 parameter 변화
- item별 response 수와 경계 질량에 따른 recovery 성능
- `ℓa,b,ω,γ_0,λ`의 bias/RMSE/rank recovery
- `b*(y)`, `γ_1/a` 등 파생량의 recovery
- MAP와 ELBO의 point estimate 비교 및 ELBO interval coverage
- seed 반복에 대한 안정성

합성 recovery가 실패하면 실제 데이터나 feature encoder로 넘어가지 않는다. kernel, quadrature, masking, transform, prior, optimizer 순서로 원인을 좁힌다.

## 8. 난이도 후보와 검증표

### 8.1 계산할 후보

```text
b*_i(y) = (logit(y)-b_i)/a_i,  y ∈ {0.90, 0.95, 0.98}
γ_1i/a_i
γ_0i/a_i
ω_i
```

`b*(y)`는 **Beta 연속 성분의 조건부 평균**이 `y`가 되는 `θ`다. ZOI 전체 기대 응답이 `y`가 되는 지점이라고 쓰지 않는다. 전체 기대값은 다음과 같다.

```text
E[Y|θ,τ] = π_B(θ) · μ_B(θ) + π_1(θ)
```

필요하다면 이를 수치적으로 푼 별도 후보를 추가할 수 있으나, 기존 후보를 대체하거나 이름을 혼용하지 않는다.

`γ_1/a`는 one-inflation threshold와 연결된 perfect/MAX 난도 후보다. `γ_0/a`는 zero threshold 후보이며 0 관측이 없을 때 약하게 식별될 수 있다. `ω`는 난이도보다 dispersion/일관성 축으로 해석한다.

### 8.2 필수 산출물

1. 난이도 후보 간 Spearman rank-correlation matrix
2. 각 후보와 SR의 rank correlation
3. 동일 곡 rate-group에서 `P(difficulty_DT > difficulty_NM)` 등 순서 정확도
4. accuracy와 score 결과의 비교
5. 후보 간 rank reversal이 큰 채보 목록과 원자료 링크/식별자
6. ID–linear–MLP의 held-out log-likelihood와 calibration
7. ID 상한과 feature model의 격차
8. `Q` 민감도
9. ELBO의 posterior scale과 held-out error의 관계
10. ELBO에서 `(γ_0,λ)` covariance와 weak-identification 양상

후보끼리 불일치한다고 즉시 학습 실패로 판정하지 않는다. 서로 다른 난이도 측면일 수 있으므로 likelihood, calibration과 실제 채보 사례를 함께 조사한다. SR과의 상관은 external comparison일 뿐 합격선 또는 학습 target이 아니다.

MAP에도 prior가 있으므로 데이터가 약한 방향의 점추정은 prior 중심으로 갈 수 있다. MAP와 ELBO의 평균이 비슷한 것은 ELBO 실패 증거가 아니다. 차이는 uncertainty와 covariance에서 평가한다.

## 9. 수치 안정성과 테스트 요구사항

- stable `logsigmoid`, `log1mexp`, `logdiffexp` 사용
- `y=0/1`에서 Beta density가 실제로 평가되지 않도록 안전한 branching/masking
- `α_B`, `β_B`, `log k`, person marginal likelihood의 finite check
- raw probability를 곱하지 않고 log space에서 합산
- GH weight normalization test
- vectorized 결과와 작은 scalar reference implementation의 일치 test
- MAP/ELBO가 동일한 kernel, `T`, prior definition을 공유하는지 test

원문에는 “Beta shape clipping”이 요구되어 있으나 정확한 범위가 정해지지 않았다. 항목 자체는 삭제하지 말되, 임의의 영구 clip 범위를 숨겨 넣지 않는다. 우선 log-domain 계산과 finite diagnostics를 구현하고, clipping이 실제로 필요하면 §11의 결정 절차를 거쳐 명시적 config와 sensitivity test로 도입한다.

## 10. 금지사항과 보존할 결정

- SR 또는 pp를 feature나 target으로 사용하지 않는다.
- SR/pp 예측으로 사전학습한 encoder의 frozen intermediate representation도 사용하지 않는다.
- SR은 평가 단계의 external comparator로만 사용한다.
- `γ_1 > γ_0` constraint에 penalty를 추가하지 않는다.
- `log k`를 결합 점수 `s`를 통해 계산하지 않는다.
- `θ` 또는 `q(θ_p)`를 학습하지 않는다.
- `D`를 `N/(T·K)`로 하드코딩하지 않는다.
- top+random pool을 섞지 않는다.
- schema, column name, function name 또는 미정 수치를 추측하지 않는다.
- multi-key encoder에 고정 폭 column one-hot을 사용하지 않는다.
- 데이터에 `y=0`이 없다는 이유로 `γ_0`를 제거하거나 출력 차원을 줄이지 않는다.
- mean-field posterior로 full covariance를 대체하지 않는다.
- 최종 평가에서 `λ_KL < 1`을 사용하지 않는다.

이미 폐기된 다음 논거를 다시 설계 근거로 사용하지 않는다.

- “posterior width 자체가 다음 stage의 학습 신호다”
- “얇은 item은 단일 stage에서 자동 하향 가중된다”
- “MLE는 uncertainty를 낼 수 없다”

다음 후보는 현재 구현 범위가 아니다.

- one-inflated-only 전환: ZOI가 올바르게 구현된 뒤 작동하지 않을 때만 검토
- MCMC sample에서 5차원 joint posterior를 학습하는 mixture density network
- `max(0,K-8)` 등의 8키 초과 손가락 배정 항
- transformer 기반 직접 chart encoder

## 11. 미결사항

다음은 구현자가 임의로 확정하지 않는다. 진단 결과 또는 사용자 확인이 필요하다.

| 항목 | 상태와 결정 기준 |
|---|---|
| 파일럿 규모 | linking과 k-core 결과로 결정. 수십 item 제안과 수백 item 검증 필요성 사이에서 선택 |
| 키모드 구성 | 4K+7K가 목표이나 linking user 수에 따라 혼합 또는 분리 |
| rate_group 포함 | 순서 검정의 장점과 동일 곡 변형이 item을 차지하는 trade-off 검토 |
| snapshot month | 여러 달 결합 시 월간 중복 처리 규칙 필요 |
| ELBO `L` head | `μ`만 amortize하고 item별 `L`을 둘지, 둘 다 amortize할지 미정. 전 item 공유 global `L`은 금지 |
| Cholesky diagonal | positive transform, floor와 초기값의 정확한 수치 미정 |
| Beta shape clipping | 필요 여부, 범위와 적용 방식 미정. 진단·민감도 검사 후 결정 |
| MLP 크기 | linear → shallow MLP → 3–4 layer 순으로 ID gap과 overfit을 보고 결정 |
| 초기화 | `b`를 전체 평균의 logit 근처, `ℓa=0`, `ω`를 분산 기반, ELBO log-scale 약 -2로 시작하는 안은 후보이며 확정값 아님 |
| `R` 상단 포화 | 추출 후 `R>0.9` 질량 확인 |
| Python 3.14 | JAX/equinox 호환성 확인 |
| repo 이름·위치 | 기존 `vsrg-irt` 재사용 여부 미정. 기본 방향은 새 구축 |

질문이 생기면 먼저 Obsidian vault의 관련 원문을 확인한다. 그래도 결정되지 않은 값만 사용자에게 묻는다. 지도교수 제안, 사용자 결정, 구현상 제안을 구분해 기록한다.

## 12. 레포 구조와 기술 스택

기존 `vsrg-irt`는 two-stage와 NumPyro SVI 전제를 갖고 있어 재사용 당위성이 낮다. 기본 방향은 새 레포 구축이다. 다만 osu! lazer C# SR algorithm의 Python port인 `mania_sr.py`는 수치 일치가 검증되었으므로 평가 모듈로 가져온다.

```text
data/
  DB query, linking 진단, iterative k-core, holdout, person-major dataset
chart/
  event parser, rate transform, feature registry/extraction
model/
  z↔τ transform, ZOI log-kernel, GH quadrature, id/linear/mlp encoders
train/
  MLE(MAP)/ELBO objectives, KL schedule/scaling, optax loops
eval/
  recovery, b*(y), gamma thresholds, omega, calibration,
  rank correlations, rate order, SR comparison
tests/
  kernel branches, normalization, transforms, GH reference,
  synthetic recovery smoke tests
configs/
  dataset, feature scale, Q, inference mode, seed and explicit safeguards
```

기술 스택:

- Windows, Python 현재 3.14, `uv`
- JAX CPU
- `optax` 직접 loop 권장
- neural layers는 `equinox` 정도의 경량 library
- PostgreSQL 18 Docker, 홈랩

NumPyro AutoGuide는 guide가 chart input을 받는 amortized 구조와 잘 맞지 않으므로 기본 선택이 아니다. 손실식이 명시적이므로 직접 구현한다.

## 13. 실행 순서

1. Obsidian schema note 확인
2. multi-key linking 선행 쿼리
3. iterative k-core sweep과 파일럿 범위 결정
4. event parser, rate transform, feature registry
5. scalar reference ZOI kernel과 branch unit test
6. vectorized kernel과 GH marginalization, reference 일치 test
7. synthetic generator와 recovery; 통과 전 실데이터 금지
8. ID encoder 실데이터 MAP fit
9. ID encoder ELBO fit
10. linear, shallow MLP, 필요 시 더 깊은 MLP
11. accuracy와 score 각각에 대한 §8 검증표
12. 미결사항과 알려진 편향을 포함한 결과 보고

## 14. 알려진 구조적 한계

- 플레이어의 선곡은 MNAR이며 어려운 item의 난이도를 과소추정할 수 있다.
- 응답이 두꺼운 item 선택은 인기·랭크맵 편향을 만든다.
- 최고 기록 중복 제거는 order-statistic bias를 만든다.
- random pilot의 `τ` 분포는 전체 VSRG 모집단을 대표하지 않는다.
- linking이 약하면 키모드 간 공통 척도가 성립하지 않는다.
- `θ ~ Normal(0,1)`은 식별 규약뿐 아니라 population-shape 가정이며 후속 민감도 분석 대상이다.

이 한계를 숨기거나 파일럿에서 모두 해결하려 하지 않는다. 재현 가능한 진단과 결과 메타데이터로 남긴다.

## 15. 참고 문헌과 비교 기준

- Molenaar, Cúri & Bazán (2022), *Zero and One Inflated Item Response Theory Models for Bounded Continuous Data*, JEBS
- Chen et al. (2019), β³-IRT, AISTATS
- Cheng et al. (2023), Interval Target Regression, ICML
- Noel & Dauvier (2007), Beta IRT
- osu!mania lazer SR: 규칙 기반 external comparator
- Etterna MinaCalc: 규칙 기반 다축 비교 대상

Molenaar 논문의 완전 주변우도는 model-selection quantity이며 이 파일럿의 training objective로 구현하지 않는다. 이 파일럿은 person ability를 GH로 주변화하고 item `z`에 대해 MAP 또는 variational inference를 수행한다.
