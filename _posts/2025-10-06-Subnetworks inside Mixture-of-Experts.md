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
1) The original FFN was originally very *sparse*, meaning that there was a chance of deviding the Big Matrix(function) into smaller sub-networks. (Less superposition?) <br/>
2) This architecture was so hardware-friendly, meaning that computation costs decreased so well to ignore the VRAM costs.<br/>

>> This post will focus more on the first characteristic of FFN.

I think the most important feature addressing this architecture is ***sparsity***. But why does this sparsity even happen? There are many reasons for this such as *residual-learning* and *gradient*, but on this post i will focus on a more transformer related reason, which is ***Attention***.

## Rank Loss by attention, and information search

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


## Sparse and Huge Network, devided into sub-networks by a router

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>  