---
title: "Exploring Muon"
layout: post
featured-img:
mathjax: true
pinned: false
python-interactive: true
categories: [Algorithms, A.I., M.L.]
summary: Deep diving into one element of Karpathy's nanochat
---

# Background

[`nanochat`][nanochat] just dropped a couple of weeks ago and one element that I was extremely interested in was [muon][muon-keller-jordan]. It's a pretty recent state of the art optimizer that has shown competitive performance in training speed challenges.

First of all, if you are not familiar with some of this, you should start with Keller Jordan's blog that I linked above. He's the creator of the approach and it's pretty ingenious. Second of all, if you're not familiar with linear algebra at all (which is ok), I'd recommend this [Little Book of Linear Algebra][linalg-book]. I ran through it over the past couple weeks so that I could ensure a strong base / have a refresher for some of the concepts that I haven't seen since college. You can check out the [Jupyter notebooks here][linalg-book-code].

This post is going to try and take you as close from $0 \to 1$ as possible (one huge benefit of running through the book + lab linked above is my latex got way better. Not going to help me land a job at Anthropic but c'est la vie).

# Table of Contents

- [Background](#background)
- [Table of Contents](#table-of-contents)
- [(optional) Reading + Videos](#optional-reading--videos)
- [1. Deep Learning](#1-deep-learning)
- [Optimizers](#optimizers)
  - [Loss Function](#loss-function)
    - [Visualization](#visualization)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [SGD with Momentum](#sgd-with-momentum)
    - [Computational Cost of Momentum](#computational-cost-of-momentum)
  - [Adaptive Learning Rates (AdaGrad / RMSProp)](#adaptive-learning-rates-adagrad--rmsprop)
    - [AdaGrad](#adagrad)
    - [RMSProp](#rmsprop)
  - [Bias Correction (meet Adam)](#bias-correction-meet-adam)
  - [Weight Decay Coupling (the "W" in AdamW)](#weight-decay-coupling-the-w-in-adamw)

# (optional) Reading + Videos

These are a couple of helpful resources for you all to get started. I would actually think that if you're starting from close to scratch or near scratch (haven't studied AdamW) then you should probably come back to these after my article.

- Videos
  - [**This Simple Optimizer Is Revolutionizing How We Train AI (Muon)**][muon-video] (p.s. god the amount of clickbaiting people do is just suffocating me... however, this is a good video)
- Reading
  - [**Muon: An optimizer for hidden layers in neural networks**][muon-keller-jordan] - _Keller Jordan_
  - [**Deriving Muon**][deriving-muon] - _Jeremy Bernstein_
  - [**Understanding Muon**][understanding-muon-laker] - _Laker Newhouse_
    - this series (after doing my own research and investigation) is hilariously written. lots of Matrix allusions

# 1. Deep Learning

I'm not going to take you from the very beginning, but the language of deep learning is basically just... linear algebra.

We have these "deep learning" models that are really neural networks. All that means is that they're layers of parameters (weights and biases) that take various inputs and make predictions. They normally are _affine transformations_ followed by a (usually) non-linear activation.

Generally, the flow for training in deep learning goes like this:

1. forward pass (feeding data in)
2. loss function (so we know how we did)
3. backward pass (so we know how to adapt)
4. gradient descent (or flavors thereof... where we actually adjust our weights)

There's fascinating math at all points of this process. However, we're going to spend the day focusing on step 4 - and specifically on the subset of **optimizers**. Modern optimizers modify gradients using momentum, adaptive learning rates, etc.

Here is a high level visualization of what's happening:

<div class="video-container">
  <div class="video-wrapper-dark">
    <video 
      src="https://www.dropbox.com/scl/fi/11h7n3gwa30gmo57yj0zo/ch1-ml-training-process.mp4?rlkey=z29nnmou3ab8zvvj5hliphi25&st=nov43ilo&raw=1"
      muted
      autoplay
      loop
      controls
      style="width: 100%; height: auto;">
    </video>
  </div>
</div>

<div class="image-caption">Courtesy of me and Claude hammering on manim</div>
<br>

Note, that $ \eta $ here is the learning rate.

# Optimizers

Ok the canonical example with optimizers is that we're basically trying to find the lowest point in a valley. This is assuming our search space is $\mathbb{R}^3$ really but that's fine for now.

So like let's take an actual example with the Grand Canyon. Imagine you're standing on top of the Grand Canyon - how are you going to find the lowest point in the Grand Canyon?

![top-of-grand-canyon](https://www.jasonweissphotography.com/images/960/grand-canyon-toroweap-sunrise.jpg){: .center-shrink .lightbox-image}

<div class="image-caption"><a href="https://www.jasonweissphotography.com/photo/grand-canyon-sunrise-toroweap/">Kudos</a> to Jason Weiss</div>
<br>

Now, the optimizer is basically telling us _how_ to walk down that space. It's obviously a lot easier if we have a topographic map, but we certainly do not in deep learning, and even with the topographic map, it can be tough to search across.

![grand-canyon-topo](https://databayou.com/grand/images/grandcanyonelevation.webp){: .center-shrink .lightbox-image}

<div class="image-caption"><a href="https://databayou.com/grand/canyon.html">Kudos</a> to DataByYou</div>
<br>

In this analogy, elevation is basically how "wrong" we are. You can think of it as the output of our loss function $L(\hat{y}, y)$. So we compute gradients to determine which direction reduces that loss. However, we still don't know how big each step would be (the $\eta$ mentioned above) or how to adjust over time or how to avoid getting caught in local minima, etc.

## Loss Function 

### Visualization

I don't have a loss function that is equivalent to the Grand Canyon (sadly), but we are going to look at the [Styblinski Tang function][styblinski-tang-fn] as our example loss function. This isn't going to be accurate, but imagine that the loss function of our deep learning process is only in 3D and has a shape that can be described by a function. In 2D, the Styblinski Tang function looks like this:

$$
\begin{align}
f(x,y) &= \frac{1}{2}\sum_{i=1}^{d} \big(x_i^4 - 16x_i^2 + 5x_i \big) \\
f(x,y) &= \frac{1}{2}(x_i^4 - 16x_i^2 + 5x_i ) (y_i^4 - 16y_i^2 + 5y_i )
\end{align}
$$

Here's a visualization of this function:

<div class="video-container">
  <div class="video-wrapper-dark">
    <video 
      src="https://www.dropbox.com/scl/fi/evcgoniyxavkrqs8gajru/ch2-loss-function.mp4?rlkey=r74573378jt9njeltk19w5912&st=rmyax9oh&raw=1"
      muted
      autoplay
      loop
      controls
      style="width: 100%; height: auto;">
    </video>
  </div>
</div>

<div class="image-caption">Courtesy of me and Claude hammering on manim</div>
<br>

## Stochastic Gradient Descent

Conceptually with standard stochastic gradient descent (SGD), we update our weights so that we move in the opposite direction of the gradient (given gradient points to highest uphill direction).

Mathematically speaking, this is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} L (\theta_t)
$$

SGD works pretty well but it's far from the best. Think about it back to our Grand Canyon approach. Imagine there are steep stairs but they zig-zag back and forth down the grand canyon. Potentially there is a ramp that is less steep but still more directly gets us to the lowest point in the valley quicker. If our landscape is more dynamic than just a vanilla bowl,that path is almost certainly not straight, and therefore SGD isn't the most _efficient_. This is basically what happens to SGD in ravines. There is high curvature in one dimension, but not in another.

Furthermore, this step size for the gradient descent isn't dynamic enough. Having one step size doesn't take into nuance the steps per model param / model param derivative that we need to adjust by, so we can overblow our targets.

Here's an example of where SGD could get caught in a local minima.

<div class="video-container">
  <div class="video-wrapper-dark">
    <video 
      src="https://www.dropbox.com/scl/fi/nx9o5mbx5kc206ksf2wc1/ch3-sgd-trap.mp4?rlkey=znd4b6bl69dg3roi1t7aox3sq&st=21l0we8j&raw=1"
      muted
      autoplay
      loop
      controls
      style="width: 100%; height: auto;">
    </video>
  </div>
</div>

<div class="image-caption">Courtesy of me and Claude hammering on manim</div>
<br>

And if 3D isn't really your style (especially given my `manima` skills are pretty poor). Here's some Python code that will visualize SGD as a topological 2D portion:

<!-- prettier-ignore-start -->
<div class="interactive-python">
<pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def styblinski_tang_fn(x: float, y: float) -> float:
    return 0.5 * ((x**4 - 16 * x**2 + 5 * x) + (y**4 - 16 * y**2 + 5 * y))

def styblinski_tang_grad(x: float, y: float) -> np.ndarray:
    dfx = 2 * x**3 - 16 * x + 2.5
    dfy = 2 * y**3 - 16 * y + 2.5
    return np.array([dfx, dfy], dtype=float)

"""learning params"""
eta = 0.01
steps = 80
theta = np.array([3.5, -3.5], dtype=float)

"""SGD!!! This is the important part here. Implementing the exact math above."""
path = [theta.copy()]
for _ in range(steps):
    grad = styblinski_tang_grad(*theta)
    theta -= eta * grad
    path.append(theta.copy())
path = np.array(path)

"""find stationary points (we can just look at derivative because repeated)"""
roots = np.roots([2.0, 0.0, -16.0, 2.5])
roots = np.real(roots[np.isreal(roots)])          # keep real roots
"""this is basically using second derivative to determine minima"""
minima_1d = [r for r in roots if (6*r*r - 16) > 0]  # two minima
local_minima_2d = np.array(list(product(minima_1d, repeat=2)), dtype=float)
vals = np.array([styblinski_tang_fn(x, y) for x, y in local_minima_2d])
gmin_idx = np.argmin(vals)
gmin_pt = local_minima_2d[gmin_idx]
gmin_val = vals[gmin_idx]

"""viz"""
x = y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)
Z = styblinski_tang_fn(X, Y)
plt.figure(figsize=(7, 6))
cs = plt.contour(X, Y, Z, levels=40, cmap="viridis", alpha=0.85)
plt.clabel(cs, inline=True, fmt="%.0f", fontsize=8)
plt.plot(path[:, 0], path[:, 1], 'r.-', label='GD Path', zorder=2)
plt.scatter(path[0, 0], path[0, 1], color='orange', s=80, label='Start', zorder=3)
plt.scatter(path[-1, 0], path[-1, 1], color='blue', s=80, label='End', zorder=3)
mask = np.ones(len(local_minima_2d), dtype=bool)
mask[gmin_idx] = False
if np.any(mask):
    plt.scatter(local_minima_2d[mask, 0], local_minima_2d[mask, 1],
                marker='v', s=120, edgecolor='k', facecolor='white',
                label='Local minima', zorder=4)
plt.scatter(gmin_pt[0], gmin_pt[1], marker='*', s=220, edgecolor='k',
            facecolor='gold', label=f'Global min ({gmin_pt[0]:.4f}, {gmin_pt[1]:.4f})\n f={gmin_val:.4f}', zorder=5)
plt.title("Gradient Descent on Styblinski–Tang: Local vs Global Minima")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(loc='upper right'); plt.grid(alpha=0.3); plt.tight_layout();
plt.show()
</code></pre>
</div>
<!-- prettier-ignore-end-->


## SGD with Momentum

So the natural progression is how can we do better than normal SGD.

This idea has been around forever (1964) compared to Muon which is basically 2025. Boris Polyak introduced momentum with physical intuition. If you roll a heavy ball down a hill and there are valleys, it doesn't get trapped in a local minima. It has momentum to carry it over local minimum which helps find a global min.

Mathematically, it's a pretty simple extension from our previous. The general idea is that now we have two equations governing how we update our parameters:

$$
\begin{align}
v_{t+1} &= \beta v_t - \eta \nabla_{\theta} L (\theta_{t}) \\
\theta_{t+1} &= \theta_{t} + v_{t+1}
\end{align}
$$

We've got some new parameters, so let's define those:

- $v_{t}$ - is the "velocity", it's the accumulated gradient basically our physical momentum
- $\beta$ - is the "momentum coefficient". controls how much history we remember and how much we want to propagate
- $\eta$ - is still our learning rate

A key insight is that if you take $\beta \to 0$ and substitute $v_{t+1}$ then our whole thing falls back to SGD (which is good).

A core paradigm shift here was that this was the first time gradient descent carried with it the notion of memory. It's a bit more stateful.

Once again, a 3D version, and a 2D version.

<div class="video-container">
  <div class="video-wrapper-dark">
    <video 
      src="https://www.dropbox.com/scl/fi/ihq86vaebowf17g7dacql/ch4-sgd-mom.mp4?rlkey=1c5xshmajrfy66z7brnfq61hu&st=oladf1ox&raw=1"
      muted
      autoplay
      loop
      controls
      style="width: 100%; height: auto;">
    </video>
  </div>
</div>

<div class="image-caption">Courtesy of me and Claude hammering on manim</div>
<br>


And the 2D visualization:

<div class="interactive-python">
<pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def styblinski_tang_fn(x: float, y: float) -> float:
    return 0.5 * ((x**4 - 16 * x**2 + 5 * x) + (y**4 - 16 * y**2 + 5 * y))

def styblinski_tang_grad(x: float, y: float) -> np.ndarray:
    dfx = 2 * x**3 - 16 * x + 2.5
    dfy = 2 * y**3 - 16 * y + 2.5
    return np.array([dfx, dfy], dtype=float)

def stationary_points_and_global_min():
    roots = np.roots([2.0, 0.0, -16.0, 2.5])
    roots = np.real(roots[np.isreal(roots)])
    minima_1d = [r for r in roots if (6*r*r - 16) > 0]
    mins2d = np.array(list(product(minima_1d, repeat=2)), dtype=float)
    vals = np.array([styblinski_tang_fn(x, y) for x, y in mins2d])
    gidx = np.argmin(vals)
    return mins2d, mins2d[gidx], vals[gidx]

def run_sgd(theta0, eta=0.02, steps=1200):
    theta = np.array(theta0, float)
    path = [theta.copy()]
    for _ in range(steps):
        theta -= eta * styblinski_tang_grad(*theta)
        path.append(theta.copy())
    return np.array(path)

"""
again, re-call beta is our momentum coefficient
eta is still our learning rate
extension: Nesterov Momentum
"""
def run_momentum(theta0, eta=0.02, beta=0.90, steps=1200):
    theta = np.array(theta0, float)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        grad = styblinski_tang_grad(*theta)
        v = beta * v - eta * grad
        theta = theta + v
        path.append(theta.copy())
    return np.array(path)

"""params"""
theta_start = np.array([4.1, 4.5], dtype=float)
eta = 0.02
beta = 0.90
steps = 1200
use_nesterov = False  # flip to True to experiment

sgd_path = run_sgd(theta_start, eta=eta, steps=steps)
mom_path = run_momentum(theta_start, eta=eta, beta=beta, steps=steps)
mins2d, gmin_pt, gmin_val = stationary_points_and_global_min()

"""viz"""
x = y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = styblinski_tang_fn(X, Y)

plt.figure(figsize=(8, 7))
cs = plt.contour(X, Y, Z, levels=50, cmap="viridis", alpha=0.85)
plt.clabel(cs, inline=True, fmt="%.0f", fontsize=7)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'r.-', lw=1.5, ms=3, label='SGD')
plt.plot(mom_path[:, 0], mom_path[:, 1], 'b.-', lw=1.5, ms=3,
         label=f'Momentum (β={beta}, nesterov={use_nesterov})')
plt.scatter(sgd_path[0, 0], sgd_path[0, 1], c='orange', s=80, label='Start', zorder=3)
plt.scatter(sgd_path[-1, 0], sgd_path[-1, 1], c='red', s=70, label='SGD End', zorder=3)
plt.scatter(mom_path[-1, 0], mom_path[-1, 1], c='blue', s=70, label='Momentum End', zorder=3)
vals = np.array([styblinski_tang_fn(x0, y0) for x0, y0 in mins2d])
mask = np.ones(len(mins2d), dtype=bool)
mask[np.argmin(vals)] = False
if np.any(mask):
    plt.scatter(mins2d[mask, 0], mins2d[mask, 1],
                marker='v', s=120, edgecolor='k', facecolor='white',
                label='Local minima', zorder=4)
plt.scatter(gmin_pt[0], gmin_pt[1], marker='*', s=220, edgecolor='k',
            facecolor='gold', label=f'Global min ({gmin_pt[0]:.4f}, {gmin_pt[1]:.4f})\n f={gmin_val:.4f}', zorder=5)

plt.title("SGD vs Momentum on Styblinski–Tang: Escaping a Local Minimum")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()
</code></pre>

</div>


### Computational Cost of Momentum

So while momentum is great and improves training, let's look at the change. In SGD, we have:

- for each parameter $\theta_i$:
  - parameter itself
  - gradient $\nabla_{\theta_{i}} L$

But now that we're carrying around velocity $v_i$ for each parameter, we store:

- memory wise:
  - one extra tensor the same size as $\theta$ (i.e. the same number of parameters we have )
- comp wise:
  - this is relatively inexpensive given it's basically one more computation to make
  - but a good callout is that it's not free

Again, even if you didn't look at all at the code above, and just ran it, see this part:

```python
def run_momentum(theta0, eta=0.02, beta=0.90, steps=1200):
    theta = np.array(theta0, float)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        grad = styblinski_tang_grad(*theta)
        v = beta * v - eta * grad
        theta = theta + v
        path.append(theta.copy())
    return np.array(path)
```

That `v` didn't exist before with standard SGD.

This is a general tradeoff that we'll need to think about optimizer design. We need to be thinking about this on a massive magnitude of training and that each operation has significant impact leading to real $ signs.


## Adaptive Learning Rates (AdaGrad / RMSProp)

Great, so momentum is going to help us smooth learning. The next area of improvement was for people to focus on $\eta$. It sucks that it's the same for every parameter, so the whole notion was that we want to have our learning rate be adaptive per parameter.

This section is where the math starts to get a bit more interesting. 

### AdaGrad

Adaptive gradient came first. [Here's the original paper][adagrad-paper] written in 2010 by John Duchi, Elad Hazan, and Yoram Singer.

The general idea is:

- we keep track of how large each parameter's past gradients have been
- we use the history to scale down updates for params that have seen a lot of gradient action

So the core idea here is that we're going to track the **sum of each parameters squared gradients over time**. And this helps a ton of things with things like vanishing and exploding gradients (which actually was also an annoyance with [**Teaching a Computer How to Write**][gen-handwriting].

In other words, 

$$ r_{t,i} = \sum_{k=1}^t g_{k,i}^2 $$ 

So basically $r_{t,i}$ for the $i$th parameter at time $t$ is going to tell you how much more or less "energy". 

Then our **update rule** rescales the learning rate for each param coordinates:

$$ \theta_{t+1, i} = \theta_{t,i} - \frac{\eta}{\sqrt{r_{t,i}} + \varepsilon} g_{t,i} $$

This can be written in a vectorized format like:

$$ \theta_{t+1} = \theta_{t} - \eta D_{t}^{-1/2} g_{t} $$

where $D_t = \text{diag}(r_t)$ and each diagonal element corresponds to one coordinate's cumulative gradient magnitude. So we're basically embedding the $i$ into the shape of the vectors and dimension. 

Again, DL loves big matrices. 

I am not going to try and do a 3D visualization given those take me awhile to get to an acceptable place.

<div class="interactive-python">
<pre><code class="language-python">
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def styblinski_tang_fn(x: float, y: float) -> float:
    return 0.5 * ((x**4 - 16 * x**2 + 5 * x) + (y**4 - 16 * y**2 + 5 * y))

def styblinski_tang_grad(x: float, y: float) -> np.ndarray:
    dfx = 2 * x**3 - 16 * x + 2.5
    dfy = 2 * y**3 - 16 * y + 2.5
    return np.array([dfx, dfy], dtype=float)

def stationary_points_and_global_min():
    roots = np.roots([2.0, 0.0, -16.0, 2.5])
    roots = np.real(roots[np.isreal(roots)])
    minima_1d = [r for r in roots if (6*r*r - 16) > 0]
    mins2d = np.array(list(product(minima_1d, repeat=2)), dtype=float)
    vals = np.array([styblinski_tang_fn(x, y) for x, y in mins2d])
    gidx = np.argmin(vals)
    return mins2d, mins2d[gidx], vals[gidx]

def run_sgd(theta0, eta=0.02, steps=1200):
    theta = np.array(theta0, float)
    path = [theta.copy()]
    for _ in range(steps):
        theta -= eta * styblinski_tang_grad(*theta)
        path.append(theta.copy())
    return np.array(path)

def run_momentum(theta0, eta=0.02, beta=0.90, steps=1200):
    theta = np.array(theta0, float)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        grad = styblinski_tang_grad(*theta)
        v = beta * v - eta * grad
        theta = theta + v
        path.append(theta.copy())
    return np.array(path)

def run_adagrad(theta0, eta=0.40, eps=1e-8, steps=1200):
    """
    r_t <- r_{t-1} + g_t^2
    theta <- theta - (eta / (sqrt(r_t) + eps)) * g_t
    """
    theta = np.array(theta0, float)
    r = np.zeros_like(theta)         
    path = [theta.copy()]
    for _ in range(steps):
        g = styblinski_tang_grad(*theta)
        r = r + g * g
        lr = eta / (np.sqrt(r) + eps) # elementwise effective LR
        theta = theta - lr * g
        path.append(theta.copy())
    return np.array(path)

"""params"""
theta_start = np.array([4.1, 4.5], dtype=float)
eta = 0.02
beta = 0.90
steps = 1200
eta_adagrad = 0.40
eps_adagrad = 1e-8

sgd_path = run_sgd(theta_start, eta=eta, steps=steps)
mom_path = run_momentum(theta_start, eta=eta, beta=beta, steps=steps)
ada_path = run_adagrad(theta_start, eta=eta_adagrad, eps=eps_adagrad, steps=steps)
mins2d, gmin_pt, gmin_val = stationary_points_and_global_min()

"""viz"""
x = y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = styblinski_tang_fn(X, Y)

plt.figure(figsize=(8, 7))
cs = plt.contour(X, Y, Z, levels=50, alpha=0.85)   # (kept close; removed explicit cmap for portability)
plt.clabel(cs, inline=True, fmt="%.0f", fontsize=7)

plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'r.-', lw=1.5, ms=3, label='SGD')
plt.plot(mom_path[:, 0], mom_path[:, 1], 'b.-', lw=1.5, ms=3,
         label=f'Momentum (β={beta})')
plt.plot(ada_path[:, 0], ada_path[:, 1], 'g.-', lw=1.5, ms=3,
         label=f'AdaGrad (η₀={eta_adagrad})')

plt.scatter(sgd_path[0, 0], sgd_path[0, 1], c='orange', s=80, label='Start', zorder=3)
plt.scatter(sgd_path[-1, 0], sgd_path[-1, 1], c='red', s=70, label='SGD End', zorder=3)
plt.scatter(mom_path[-1, 0], mom_path[-1, 1], c='blue', s=70, label='Momentum End', zorder=3)
plt.scatter(ada_path[-1, 0], ada_path[-1, 1], c='green', s=70, label='AdaGrad End', zorder=3)

vals = np.array([styblinski_tang_fn(x0, y0) for x0, y0 in mins2d])
mask = np.ones(len(mins2d), dtype=bool)
mask[np.argmin(vals)] = False
if np.any(mask):
    plt.scatter(mins2d[mask, 0], mins2d[mask, 1],
                marker='v', s=120, edgecolor='k', facecolor='white',
                label='Local minima', zorder=4)
plt.scatter(gmin_pt[0], gmin_pt[1], marker='*', s=220, edgecolor='k',
            facecolor='gold', label=f'Global min ({gmin_pt[0]:.4f}, {gmin_pt[1]:.4f})\n f={gmin_val:.4f}', zorder=5)

plt.title("SGD vs Momentum vs AdaGrad on Styblinski–Tang")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()
</code></pre>

</div>

So again

### RMSProp


Again, I am not going to try and do a 3D visualization given those take me awhile to get to an acceptable place.

## Bias Correction (meet Adam)



## Weight Decay Coupling (the "W" in AdamW)

[comment]: <> (Bibliography)
[nanochat]: https://github.com/karpathy/nanochat
[muon-keller-jordan]: https://kellerjordan.github.io/posts/muon/
[linalg-book]: https://little-book-of.github.io/linear-algebra/
[linalg-book-code]: https://github.com/johnlarkin1/little-book-of-linalg
[muon-video]: https://www.youtube.com/watch?v=bO5nvE289ec
[deriving-muon]: https://jeremybernste.in/writing/deriving-muon
[understanding-muon-laker]: https://www.lakernewhouse.com/writing/muon-1
[adagrad-paper]: https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
[styblinski-tang-fn]: https://www.sfu.ca/~ssurjano/stybtang.html
[gen-handwriting]: {{ site.baseurl }}/2025/teaching-a-computer-to-write/