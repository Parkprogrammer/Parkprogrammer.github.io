---
layout: default
title: "Compromising with the bitter lesson: Designing how to search"
date: 2026-10-06
---

<nav>
  <a href="/">Home</a>
  <a href="/blog/">Blog</a>
  <a href="/publications/">Publications</a>
  <a href="/assets/files/CV_PJH.pdf">CV</a>
</nav>  

## Compromising with the bitter lesson: Designing how to search

<br/>

<div style="text-align: center;">
  <img src="/assets/images/posts/bitter_lesson_1.png" alt="The bitter lesson" style="max-width: 70%; height: auto; display: block; margin: 0 auto;">
  <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;">From Lance's Blog</p>
</div>

*"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation." - Richard Sutton* 

For years, after diving into the field of AI and trying to optimize the models into somewhat *General-like(e.g Gaussian Process, Bayesian Inference, SLMs)* the ego-inflation became massive with no actual progress. Papers and Companies argue why some architecture should be awsome (e.g JEPA) and a road to AGI.

But, we fall into the same hole once again. As Geoffrey Hinton said in his latest interview, we don't know why these machines actually work. Every model should fall into a *local-minima*, and cannot go beyond the *bias-variance tradeoff*. But, attention and residual networks goes through every breakthrough.

## Just Transformers? - Make it search through it's program, parameter, posterior .... space

<br/>

<div style="text-align: center;">
  <img src="/assets/images/posts/bitter_lesson_2.jpeg" alt="이미지 설명" style="max-width: 50%; height: auto; display: block; margin: 0 auto;">
  <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;">yann lecun's talk</p>
</div>

*Anti-LLM researchers* usually argure about two things. 1) LLM's need too much information(data) to learn and 2) LLM's cannot search. But in my opinion both are wrong. The metaphor between LLM and a child miss the information density, and the searchability of LLM in Chain-of-Thought is 100% true(which will be discussed in another post). The emergent capability of LLMs never seems to stop.

But after the astonishing success, there seemed to be a never-ending stop towards this machine's capability. Spatial Reasoning, millennial math problems, Robotic manipulation, Machanical Engineering ... It was glorious at first, but came as a disaster to what we humans should do in this *Super-Intelligence-ish* machine.

Now maybe we should abandon inductive-bias.

## Making a Search Engine for AI(not RAG), but make it faster once again.

<div style="max-width: 900px; margin: 0 auto;">
  <div style="display: flex; gap: 16px; align-items: flex-start; justify-content: center; flex-wrap: wrap;">
    <div style="flex: 121 1 0; min-width: 260px;">
      <img src="/assets/images/posts/bitter_lesson_3.png" alt="François Chollet on base LLMs vs. LRMs" style="width: 100%; height: auto; display: block;">
    </div>
    <div style="flex: 168 1 0; min-width: 260px;">
      <img src="/assets/images/posts/bitter_lesson_4.png" alt="AI Learning to Play Chess leaderboard" style="width: 100%; height: auto; display: block;">
    </div>
  </div>
  <p style="text-align: center; color: #888; font-size: 14px; margin-top: 8px; margin-bottom: 0;">From François Chollet (@fchollet), Oct 2, 2026</p>
</div>

The inductive <-> transductive transition made searching at test-time more important. How can we design a search-space more suitable for AI?








<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
