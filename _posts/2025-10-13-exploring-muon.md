---
title: "Exploring Muon"
layout: post
featured-img:
mathjax: true
pinned: false
categories: [Algorithms, Rust]
summary: Deep diving into one element of Karpathy's nanochat
---

# Background

## Gradient Descent

### Adam (Adaptive Moment Estimation)

Adam builds upon gradient descent by keeping track of two extra parameters:

- square gradients
- past gradients themselves (momentum)

> Rather than updating parameters directly from the raw gradients, Adam combines the momentum with an adaptive scaling vector derived from the squared gradients

Adam requires keeping two extra parameters per every model parameter. As a result, the optimizer takes up about twice as much memory as the model itself.

Normally, Adam treats all parameters as a single long vector updating each value independently. It uses _vector based optimization_.
