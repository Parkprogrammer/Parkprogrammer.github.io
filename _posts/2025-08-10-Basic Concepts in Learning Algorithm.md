---
layout: default
title: "Basic Concepts in Learning Algorithm"
date: 2025-08-10
---

<head>
  <link rel="icon" type="/assets/images/logo.png" href="/assets/images/logo.png">
</head>

<nav>
  <a href="/">Home</a>
  <a href="/blog/">Blog</a>
  <a href="/publications/">Publications</a>
  <a href="/assets/files/CV_PHJ.pdf">CV</a>
</nav>  

## Basic Concepts in Learning Algorithm

It seems that many publications nowdays focus on Solving application problems *eg. Vision, Motion, Code* with the assumption that the backbone of these models are already optimized.

But to my opinion, there are still a lot of room for imporvement and optimization, which would probably solve the shortage of Hardware (GPUs) with much faster and efficient *learning and inference.*

>> **"Intelligence manifests most clearly when you can do a lot with little in terms of input." - David Krakauer**

## So what is a Neural Network?

Backbone ideas are from the video of Welch Labs *[Why Deep Learning Works Unreasonably Well](https://www.youtube.com/watch?v=qx7hirqgfuU)*

A Neural Network can be bluntly seen as a *Matrix-Multiplication* process. Which transforms a *input-vector* to a *output-vector*. But how does this ***transformation*** make sense? What does it mean by ***transformation***?<br/><br/>

<div style="text-align: center;">
  <img src="/assets/images/posts/BCLA" alt="이미지 설명" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
</div>

The above *SVD(Singular Vector Decomposition)* figure shows what it does. ($U$) and ($V^T$) ***Rotates*** the *input-vector* as they are *Orthogonal Matrices*, and ($Σ$) ***Scales*** each element in the vector as it's a *Diagonal Matrix*.

- We can see this ***Rotation*** as adjusting the coordinate system to enable the Matrix to scale the information(*singular value*) that is of much importance(σ1, σ2, σ3 .. σn) to the matrix system. 
- The ***Scaling*** is increasing the *importance* of the component(*singular value*). It can be also interpretated as a Signal amplication.

So, the ***transformation*** can be interpreted as a ***Selective information transformation***.
But this only scratches the surface of what a Neural Net actually does. There must also be the clarification of what a ***Hidden Layer*** does, and the ***Activation function*** does.
<br/><br/>

<div style="display: flex; justify-content: center; align-items: center; max-width: 600px; margin: 0 auto;">
  <div style="flex: 1; margin: 0 2px;">
    <img src="/assets/images/posts/BCLA-1" alt="이미지 설명" style="max-width: 100%; height: auto; display: block;">
    <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;">Hidden Layer</p>
  </div>
  <div style="flex: 1; margin: 0 2px;">
    <img src="/assets/images/posts/BCLA-2" alt="이미지 설명" style="max-width: 100%; height: auto; display: block;">
    <p style="text-align: center; color: #888; font-size: 14px; margin-top: 5px; margin-bottom: 0;">Activation Layer Visualization</p>
  </div>
</div>

***MLPs***, so well used nowdays are overlooked in its mechanism, which was actually a crucial breakthrough that solved the famous *XOR* problem. A *hidden layer* is a simple concept of sequentially applying *N* number of *matrix multiplication*. But as we all know, $$ W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) $$ shows that any sequential number of *matrix multiplication* can always be represented as a single layer of *matrix multiplication*. **So how does a *hidden layer* exist?** ***Activation functions***.

Before Neural Networks, *SVMs* were a great way to solve classification problems. Apart from the *soft vectors margin algorithm* that created robust classifiers, there was another method that made huge breakthroughs. ***Kernel maps***, which projected input data into *high dimensional space* solved many non-linear problems. From **Polynomial Kernels** to **Gaussian Kernels**, each non-linear map was used to extract meaningful features to classify it's data.

***Activation functions*** are these kind of *non-linear maps* that enable Matrix Systems to learn much more complex structure or features. The above figure shows the ***geometrical folding of feature spaces*** that these activation functions apply. <br/>
> There are a lot of discussions and improvements needed on this *activation function*. which i believe is a crucial part in optimization such as ***Sparsity, Depth, Kernels***.. Will Come back to it in another post.

So a Neural Network is a system that uses ***non-linear function*** to learn complex features from input data with ***Selective information transformation*** and ***geometrical folding***. But there are so many other things known and unknown that enables Neural Networks to perform extreemly well.

The most important feature must be ***Depth***.<br/><br/>

<div style="text-align: center;">
  <img src="/assets/images/posts/BCLA-3" alt="이미지 설명" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
</div>

## Why do we need so many layers?

Backbone ideas are from the Article in Lesswrong *[A Simple Introduction to Neural Networks](https://www.lesswrong.com/posts/Madwb2t79LGrLqWLH/a-simple-introduction-to-neural-networks)*


## How does it learn?

<br/><br/><br/><br/><br/><br/>