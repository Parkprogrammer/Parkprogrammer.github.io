---
layout: default
title: "Active Learning and Transductive Learning"
date: 2025-04-27
---

<nav>
  <a href="/">Home</a>
  <a href="/blog/">Blog</a>
  <a href="/publications/">Publications</a>
  <a href="/assets/files/CV_PHJ.pdf">CV</a>
</nav>  

## Active Learning and Transductive Learning

개인적인 연구를 진행하던 도중, 어려웠던 개념을 위주로 설명을 간단히 작성한다.

<div style="text-align: center;">
  <img src="/assets/images/posts/ALTL-0" alt="이미지 설명" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
</div>
<div style="text-align: center;">
  <a href="https://opencv.org/blog/understanding-transductive-few-shot-learning/" style="font-size: 0.9em; color: #888;">https://opencv.org/blog/understanding-transductive-few-shot-learning/</a>
</div>

### **Active learning** <br/>
기존 지도학습(Supervised Learning)은 방대한 양의 학습 데이터(A)와 정답 라벨(label)을 기반으로 End-to-End로 학습시키는 수동적인(Passive) 방식을 따른다. 반면에, 능동학습(Active Learning)은 전체 데이터셋(A)에서 **신뢰도(모델의 logit, entropy 등)**를 기반으로 **여러 샘플링(Sampling) 기법을 통해** *정보량이 크거나* *불확실성이 높은* 데이터를 모델이 **직접 선택(B)하여**, 라벨링하고 학습하는 방식을 뜻한다.  <br/><br/> *정보를 기반으로 선택(Selection)을 한다는 점에서 무한한 State-Space를 다루고자 Regularization을 기반으로 탐험(exploration)하는 RL과 유사한 면이 있다.<br/>
### **Transductive learning** <br/>
직관적으로는 쉽지만, 깊게 들어가면 난해해지는 개념이다. 우선 이 개념에 대해 설명하기에 앞서, *test데이터*가 무엇인지부터 정의하고 가고자 한다. 보통 모델을 학습시킬 때, 엔지니어는 지금까지 수집한 데이터 및 라벨을 *train/test/validation set*으로 분리하고 *train loss*와 *test/validation accuracy*를 기준으로 학습의 성공 여부를 판단한다. <br/><br/>
하지만 확률분포 개념을 기반으로 생각해보면, **이 분할(split) 과정에는 편향(bias)이 존재한다.** <br/><br/>
<div style="text-align: center;">
  <img src="/assets/images/posts/TAL-0.jpg" alt="이미지 설명" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
</div>
<div style="text-align: center;">
  <a href="https://www.sciencedirect.com/topics/computer-science/training-error" style="font-size: 0.9em; color: #888;">https://www.sciencedirect.com/topics/computer-science/training-error</a>
</div>
*test accuracy*나 *validation accuracy*는 모델의 완벽한 성능을 나타내지 못한다. 위 그림에서 볼 수 있듯이, 전체 데이터(A) 중에서 테스트(B)나 검증(C) 샘플은 전체 모집단 분포 *P*에 대한 유한 표본이자 추정치(*P'*)이지, 이 표본이 전체 분포를 나타낸다는 보장은 없으며,<br/> **일반화에 대한 완벽한 척도가 될 수 없기 때문이다.** (물론 현재 많은 Scaling Law에 의존하는 모델들은 이런 문제를 신경안쓰는 듯 하다)

하지만 이보다 더 확실하게 데이터에 대한 학습여부를 나타내는 지표가 아직까지는 없다. 그렇기에  학습과 검증 과정을 분리해서 보지 못한(*unseen*) 데이터에 대해 검증을 하는 방식을 **귀납적(*inductive*)**으로 학습한다고 하며, 많은 모델들의 학습방식이 이 패러다임을 따른다.

반면에 해당 논문에서 소개할 *transductive-learning*의 경우, *test* 데이터를 *train* 데이터와 함께 가지고 학습을 수행한다. 여기서 주의할 점은, ***test* 데이터의 경우 *input*만 있으며 *label*이 없는 반면 *train* 데이터의 경우 *label*이 있다는 것이다.**

label이 없는 학습? train/test가 존재하면 너무 많은 데이터가 있는것이 아닌가? 등의 질문들이 자연스레 떠오르지만, 이는 label에 의한 loss에 의존하지 않고, Sample들의 구조(*manifold*)를 학습하는 *meta-learning*이자 *metric-learning*의 개념으로 보면 된다.

더 자세한 내용은 추후 [Learning to propagate labels: Transductive propagation network for few-shot learning(ICLR2019)](https://arxiv.org/pdf/1805.10002) 논문에서 다루고자 한다.

<div class="math-display">
$$
\min_f \frac{1}{\ell} \sum_{i=1}^\ell V(f(x_i), y_i) + \gamma_A \|f\|_K^2 + \gamma_I \frac{1}{(\ell+u)^2} f^T Lf
$$
</div>


<br/><br/><br/><br/><br/><br/>