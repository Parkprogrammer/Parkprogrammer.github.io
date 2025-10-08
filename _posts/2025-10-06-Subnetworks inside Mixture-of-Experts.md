---
layout: default
title: "Is Mixture-of-Experts the best architecture there is?"
date: 2025-10-06
---

<nav>
  <a href="/">Home</a>
  <a href="/blog/">Blog</a>
  <a href="/publications/">Publications</a>
  <a href="/assets/files/CV_PHJ.pdf">CV</a>
</nav>  

## Is Mixture-of-Experts the best architecture there is?

<div style="text-align: center;">
  <img src="/assets/images/posts/SMOE-1.png" alt="이미지 설명" style="max-width: 50%; height: auto; display: block; margin: 0 auto;">
</div>

Recently, i received the *Grand Prize Award* for Optimizing Large Language Model at Samsung 2025 AI challenge and was awarded with *$7,500*. Especially, the problem was about *expert merging* and *expert pruning* on a **Qwen-A3B-30B Model**. 

But while compressing the model and optimizing the *MoE Layer*, one question kept nagging on me and after some experiments, the question grew. 

<div style="text-align: center;">
  <blockquote>
    <strong>"Is there a better architecture(inductive bias), that can outperform Mixture-of-Experts?"</strong>
  </blockquote>
</div>

<br/>Here are some of my thoughts on what is going on inside this architecture, and what better algorithm could solve this mystery.

## How did MoE replace Vanilla Feed-Forward-Network?

Scaling laws have shown that *(more parameters ≈ higher performance)*. So if we want to make our model more powerful, we need bigger parameter space, or **bigger matrices**.

And to achieve this, *Switch-Transformers, Qwen, Deepseek ...*, used this architecture called Mixture-of-Experts, and many papers have shown that this architecture achieved a big leap in pursuing the **Scaling laws**. But actually, (in my opinion) this was **not a new inductive-bias** from the original transformers, but **another optimization technique** to the transformers.<br/>

<div style="text-align: center;">
  <img src="/assets/images/posts/SMOE-2.jpg" alt="이미지 설명" style="max-width: 40%; height: auto; display: block; margin: 0 auto;">
  <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;">GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding</p>
</div>

To me, i think this optimization worked so well for two reasons.<br/><br/>
1) The original FFN was originally very *sparse*, meaning that there was a chance of deviding the Big Matrix(function) into smaller sub-networks. (More superposition) <br/>
2) This architecture was so hardware-friendly, meaning that computation costs decreased so well to ignore the VRAM costs.<br/>

>> This post will focus more on the first characteristic of FFN.

I think the most important feature addressing this architecture is ***sparsity***. But why does this sparsity even happen? There are many reasons for this such as *residual-learning* and *gradient*, but on this post i will focus on a more transformer related reason, which is ***Attention***.

## Rank Loss by attention(in residual), and information search

Nikolas Adaloglou's writing on [Why multi-head self attention works: math, intuitions and 10+1 hidden insights](https://theaisummer.com/self-attention/) tells a very important feature of what the FFN did. This work shows that *softmax computation* is *low rank* in-itself.

$$
  P = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

The *softmax* is applied to the input vectors(or residual stream) to choose which **value token** must have more information(attention).

$$
\begin{array}{cccc}
 & \langle \text{head 1} \rangle & \langle \text{head 2} \rangle & \cdots \\[6pt]
\text{I} &
[\,0.3\;\;0.2\;\;0.1\,] &
[\,0.8\;\;0.8\;\;0.8\,] &
\cdots \\[8pt]
\text{think} &
[\,0.3\;\;0.4\;\;0.4\,] &
[\,0.1\;\;0.1\;\;0.1\,] &
\cdots \\[8pt]
\text{about} &
[\,0.4\;\;0.4\;\;0.5\,] &
[\,0.1\;\;0.1\;\;0.1\,] &
\cdots \\[4pt]
\vdots & \vdots & \vdots & \\[-2pt]
\end{array}
$$

Let's say the model is trying to predict what comes after "I think about _". Looking closer at head #2, assuming that the head focuses on nouns and person detection, the softmax computation puts more **attention** into the value vector of the token **[I]**. It seems reasonable and a good mechanism for sequence modeling. But what if this massive attention towards token **[I]** happend at layer 2~3?

If another attention layer comes right after this computation, and another and another.. it seems quite obvious that the predicted next token will only have information about the **token [I]**, which would lead to a sentence like *"I think about I I I I ..."* and so on.

But this kind of behavior almost never accurs in LLMs, because of 2 reasons(or maybe more).<br/><br/>
First, the **Multi-Head** Attention enables the model to learn and express more reach behavior, where each head learns different feature from another. More information or studies could be found in [Transformer circuits](https://transformer-circuits.pub/) by Anthropic.<br/>

Second, the **Feed-Forward-Network** works as a information storage, which performs a kind of **key-value search** in the models parameters to enrich the vectors' features. This kind of interpretation is shown in the papers [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/pdf/2012.14913). 

Another more mathematical interpretation shows that this network as a ***Rank Restoration*** procedure in the paper [Attention is not all you need: pure attention loses rank doubly exponentially with depth](https://arxiv.org/pdf/2103.03404).

<div style="text-align: center;">
  <img src="/assets/images/posts/SMOE-3.jpg" alt="이미지 설명" style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;">A figure from the paper above which shows the rank destortion as layers go deeper.</p>
</div>

In my opinion, although not right in mathematical terms, *sparsity* is a form of *low-rank*, since in terms of information, these two terms means that information is low. In other words, *sparsity* is a form of *low-rank* caused by the attention layers, and to ***restore this sparsity the FFN in itself shows a form of sparse network***.

>> Just a idea not experimented. But is there a way to see the attention-sparsity as a null-space of attention?


## Sparse features vs Dense features

So this Feed-Forward-Network input is sparse, due to the sparsity that the softmax function has inside the attention layer. And of course, the non-linear functions inside this network as *ReLU* or *GeLU* that encourages more sparsity in the FFN's output due to it's cut-off functionality.

But this sparsity showed another propety which showed posibilities towards more optimization, which in my opinion, is the key towards finding a better inductive-bias than MoE. This sparsity worked as ***a way of implicitly expressing more superposition in the neurons of the Transformer architecture***.<br/><br/>

<div style="text-align: center;">
  <img src="/assets/images/posts/SMOE-4.jpg" alt="이미지 설명" style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;"> From the Toy Models of Superposition article by Anthropic.</p>
</div>

This superposition hypothesis is well described in the blog [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html#phase-change) by Anthropic. The figure above shows the idea behind superposition, which is that *dense output vectors* could not express larger feature than the dimension of the vector F, while *sparse output vectores* could express more features than the dimension due to the superposition characteristic.

It was quite counterintuitive for me at the beginning and still is, but here is what i think is happening. A dense feature vector say<br/>

$$
\begin{array}{ccc}
 & \text{Dense output } f & \text{Sparse output } f' \\[6pt]
\text{Distinct-feature}_1 &
[\,2\;\;3\;\;3\;\;1\;\;5\;\;7\;\;8\;\;\cdots\,] &
[\,2\;\;0\;\;0\;\;1\;\;0\;\;7\;\;0\;\;\cdots\,] \\[6pt]
\text{Distinct-feature}_2 &
[\,4\;\;1\;\;2\;\;6\;\;3\;\;5\;\;2\;\;\cdots\,] &
[\,0\;\;0\;\;5\;\;0\;\;0\;\;0\;\;9\;\;\cdots\,] \\[6pt]
\text{Distinct-feature}_3 &
[\,3\;\;4\;\;3\;\;5\;\;2\;\;4\;\;1\;\;\cdots\,] &
[\,0\;\;8\;\;0\;\;0\;\;0\;\;0\;\;1\;\;\cdots\,] \\[2pt]
\vdots & \vdots & \vdots
\end{array}
$$

$f$ and a sparse vector $f'$. Since a feature vector must be distinct from one another for a model to detect differnet features from a sequence, it is plausible to say that the features must be sort of $orthogonal$. And a dense feature $f$ has fewer $orthogonal$ or $\text{orthogonal-like}$ features from sparse vector $f'$, the model could express more features than the dimension(basis) when the vector is ***sparse***.


### Mathematical Reason(Maybe?)
Formally: let the feature directions be $$ V=[v_1,\dots,v_F]\in\mathbb{R}^{d\times F},\quad \|v_i\|_2=1,\quad x=Va. $$ If a feature vector must be distinct from another, it suffices that active directions can be separated by simple scoring; define the mutual coherence
$$ \mu := \max_{i\ne j}\, |v_i^\top v_j|. $$ 

For a \($$k$$\)-sparse coefficient \(a\) (i.e., \(\|$$a$$\| = $$k$$ << $$d$$\)), with nonzero entries of magnitude \(c>0\), we have

$$
\underbrace{\langle x, v_i\rangle}_{i\in \mathrm{supp}(a)} \;\ge\; c - \mu (k-1)c,
\qquad
\underbrace{|\langle x, v_j\rangle|}_{j\notin \mathrm{supp}(a)} \;\le\; \mu k c.
$$

Hence a simple top-\(k\) (or threshold) rule correctly recovers the active set whenever

$$
c - \mu (k-1)c \;>\; \mu k c
\;\;\Longleftrightarrow\;\;
1 \;>\; \mu (2k-1).
\tag{f}
$$

In words: sparser outputs (smaller \(k\)) tolerate more non-orthogonal (‘orthogonal-like’) directions (larger \(F\)) in the same \(d\)-dimensional space, because separation remains valid under the above bound; combinatorially, the number of distinct supports with \($$\le$$ k) actives is


$$
\sum_{j=1}^{k} \binom{d}{j},
$$

easily exceeding \(d\) even for moderate \(k\). By contrast, dense outputs (large \($$k \approx d$$)) violate \($$1>\mu(2k-1)$$\) unless the directions are nearly orthogonal, effectively forcing \($$F \lesssim d$$). Therefore, while dense \(f\) affords fewer orthogonal/orthogonal-like distinguishable features, a sparse \(f'\) can express more features than the dimension via superposition under the coherence condition above.

>> And more, i think this comes from the many layers of Multi-Head(Attention) process, where each heads perform there softmax in different dimension in the vector, and aggregate with a Output Matrix. Need more verification with QK curcuit and OV curcuits...

## Sparse and Huge Network, devided into sub-networks by a router

Finally, the characteristics of sparse features above imply that only a few directions are active per input, and superposition means that many directions are available overall. Under this regime, inputs that share similar support patterns cluster in the router’s gating space, so even a simple router can conditionally separate them with a Top-𝑘 k rule. 

Therefore each selected expert then specializes on a low-interference subspace, reducing gradient conflict and improving capacity utilization.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>  