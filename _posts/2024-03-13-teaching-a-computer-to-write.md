---
title: 'Teaching a Computer How to Write'
layout: post
featured-gif:
mathjax: true
categories: [Favorites, Algorithms, Development, A.I., M.L.]
summary: Jeez. For basically 7 years this has been on my wishlist.
---

<div class="markdown-alert markdown-alert-disclaimer">
<p>So first of all... this was a pretty large "pet" project. A bit of a Frankenstein. And if you came here excited to read about transformers and LLMs and GPTs, sorry you've come to the wrong place. Which maybe was a bad use of my time, but had an itch to scratch. LSTMs are a precursor, largely solved at sequential data. They aren't as smart (in a way) as LLMs and they don't utilize attention in the same way whatsoever. But it's fun math and an interesting project. </p>

<p> Second of all, I'll outright say some of the structure and logical code was adapted from the brilliant (yet mysterious) <a href="https://github.com/sjvasquez">sjvasquez</a>. Pretty sure they’re at Facebook on the AI side, but yeah it’s their one popular public repo.</p>

<p>Third, that code doesn’t work on the latest Tensorflow version (2.16.1 at time of writing). Additionally, I provide quite a couple of models so we can see the progression from a simple neural net to a basic LSTM to Peephole LSTM to a stacked cascade of Peephole LSTMs to Mixture Density Networks to Attention Mechanism to Attention RNN to finally throwing it all together to the full Handwriting Synthesis Network that Graves originally wrote about.</p>

<p>Fourth, this was by far the most energy and effort I’ve put into a blog post, so please - even if you don’t read it - enjoy the visualizations. I spent basically two weeks in between <a href="https://dropbox.com/">Dropbox</a> and <a href="https://www.mojo.com/">Mojo</a> working on this like… a solid amount. It was a large <em>large</em> time commitment, and even a financial commitment, given some GPU costs. I ended up using <a href="https://vast.ai/">vast.ai</a> to train my model on a more performant machine and then <code class="language-plaintext highlighter-rouge">scp</code> the weights / <code class="language-plaintext highlighter-rouge">.keras</code> file over. I probably should have been learning a bit more about Scylla coupled with Go for Mojo, but c'est la vie.</p>

<p>Finally, I know people are going to give me some flack for not embracing the time off more, but as my family and my ex said, this was my time and I can do with it what I wish. I also can't understate how badly I wanted this project to work. This was my senior engineering project and it’s basically been eating my brain for 7 years since I graduated. I enjoy (and also hate) these problems of dimensionality, complex math, understanding how some ML models are so brilliantly put together. Alex Graves and Chris Olah were basically engineering legends to me in college and that remains true.</p>

<p>Enjoy!</p>

</div>

# Motivating Visualizations

Ok all that being said, let's have some visualizations to motivate us.

TODO

[See here for more](#generative-handwriting-visualizations).

# Table of Contents

- [Motivating Visualizations](#motivating-visualizations)
- [Table of Contents](#table-of-contents)
- [Motivation](#motivation)
- [History](#history)
  - [Tom and My Engineering Thesis](#tom-and-my-engineering-thesis)
- [Acknowledgements](#acknowledgements)
- [Software](#software)
  - [Tensorflow](#tensorflow)
    - [Programming Paradigm](#programming-paradigm)
    - [Versions - How the times have changed](#versions---how-the-times-have-changed)
    - [Tensorboard](#tensorboard)
- [Base Neural Network Theory](#base-neural-network-theory)
  - [Lions, Bears, and Many Neural Networks, oh my](#lions-bears-and-many-neural-networks-oh-my)
  - [Basic Neural Network](#basic-neural-network)
    - [Hyper Parameters](#hyper-parameters)
  - [Feedforward Neural Network](#feedforward-neural-network)
    - [Backpropagation](#backpropagation)
  - [Recurrent Neural Network](#recurrent-neural-network)
  - [Long Short Term Memory Networks](#long-short-term-memory-networks)
    - [Understanding the LLM Structure](#understanding-the-llm-structure)
- [Putting Theory into Code](#putting-theory-into-code)
  - [LSTM Cell with Peephole Connections](#lstm-cell-with-peephole-connections)
    - [Theory](#theory)
    - [Code](#code)
  - [Gaussian Mixture Models](#gaussian-mixture-models)
    - [Theory](#theory-1)
    - [Code](#code-1)
  - [Mixture Density Networks](#mixture-density-networks)
    - [Theory](#theory-2)
    - [Code](#code-2)
  - [Attention Mechanism](#attention-mechanism)
    - [Theory](#theory-3)
    - [Code](#code-3)
  - [Stacked LSTM](#stacked-lstm)
    - [Theory](#theory-4)
    - [Code](#code-4)
  - [Final Result](#final-result)
- [Code](#code-5)
  - [Plan of Attack](#plan-of-attack)
  - [Looking back at college code 😬](#looking-back-at-college-code-)
  - [Data and Loading Data](#data-and-loading-data)
  - [Model and Tensorflow Updates](#model-and-tensorflow-updates)
  - [MDN Predictions with Simple Neural Network](#mdn-predictions-with-simple-neural-network)
  - [Predictive Handwriting Network](#predictive-handwriting-network)
  - [Synthetic Handwriting Generation Network](#synthetic-handwriting-generation-network)
- [Results](#results)
  - [Basic MDN Predictions with Simple Network](#basic-mdn-predictions-with-simple-network)
  - [Predictive Handwriting Visualizations](#predictive-handwriting-visualizations)
  - [Generative Handwriting Visualizations](#generative-handwriting-visualizations)
- [Troubles](#troubles)
- [Hardware](#hardware)
- [Conclusion](#conclusion)

# Motivation

I never actually ended up publishing it, but I had a draft blog post written maybe right after college, about some of the work that a good friend (and absolutely incredible engineer) [Tom Wilmots][tom] and I did in college.

I'm going to pull pieces of that, but the time has changed, and I wanted to revisit some of the work we did back in college, and clean it up some, given the boom of generative AI. Also I'm a better engineer and we had a small bug in some of the modeling that I want to flush out.

# History

[Tom] and I were very interested in the concept of teaching a computer how to write in college. There is a very famous [paper] that was published around 2013 from Canadian computer scientist [Alex Graves][ag], titled _Generating Sequences With Recurrent Neural Networks_. At [Swarthmore][swat], you have to do Engineering theses, called [E90s][e90].

## Tom and My Engineering Thesis

For the actual paper that we wrote, check it out here:

<div style="text-align: center;">
    <embed src="{{ '/pdfs/Handwriting-Synthesis-E90.pdf' | prepend: site.baseurl }}" width="500" height="375" type="application/pdf">
</div>

You can also check it out here: [**Application of Neural Networks with Handwriting Samples**][paper].

# Acknowledgements

- **[Tom Wilmots][tom]** - One of the brightest and best engineers I've worked with. He was an Engineering and Economics double major from [Swarthmore][swat]. He was the inspiration for this project in college, and I wouldn't have gotten anywhere without him.
- **[Matt Zucker][mz]** - Absolutely no surprise here. One of my role models and constant inspirations, Matt was kind enough to be Tom and my academic advisor for this final engineering project. He's an outstanding professor at Swarthmore College and the institution is beyond fortunate to have him.
- **[Alex Graves][ag]** - Another genius that both Tom and I had the pleasure of working with. **He actually responded to our emails!**. I remember that in college, I lost my mind when he emailed back. That a professional (who was a professor at the University of Toronto, see [here][gravesToronto]). He is the author of [this paper][paper], which Matt found for us and pretty much was the basis of our project. Alex is at Google now and is crushing it. He's also the creator of the [Neural Turing Machine][ntm], which peaked my interest after having taken [Theory of Computation][toc], with my other fantastic professor [Lila Fontes][lila] and learning about [Turing machines][turingmachines].
- **[David Ha][dha]** - Yet another genius who we had the priviledge of corresponding with. Check out his blog [here][otoro]. It's beautiful. He also is very prolific on [ArXiv][arxiv] which is always cool to see.

# Software

In college, we decided between [Tensorflow][tensorflow] and [Pytorch][pytorch]. I thought about re-implementing this in [pytorch], but I kind of figured that I wanted to see what's changed with [tensorflow] and that I was a bit more familiar with that programming paradigm.

## Tensorflow

### Programming Paradigm

Tensorflow has this interesting programming paradigm, where you are more or less creating a graph. So you define `Tensor`s and then when you run your dependency graph, those things are actually translated.

I have this quote from the Tensorflow API:

> There's only two things that go into Tensorflow.
>
> 1. Building your computational dependency graph.
> 2. Running your dependency graph.

This was the old way, but now that's not totally true. Apparently, Tensorflow 2.0 helped out a lot with the computational model and the notion of eagerly executing, rather than building the graph and then having everything run at once.

### Versions - How the times have changed

So - another fun fact - when we were doing this in college, we were on **tensorflow version v0.11**!!! They hadn't even released a major version. Now, I'm doing this on Tensorflow **2.16.1**. So the times have definitely changed.

![being-old](/images/generative-handwriting/being_old.jpeg){: .center-shrink }

Definitely haven't been able to keep up with all those changes.

### Tensorboard

Another cool thing about [Tensorflow][tf] that should be mentioned is the ability to utilize the [Tensorboard][tensorboard]. This is a visualization suite that creates a local website where you can interactively and with a live stream visualize your dependency graph. You can do cool things like confirm that the error is actually decreasing over the epochs.

We used this a bit more in college. I didn't get a real chance to dive into the updates made from this.

# Base Neural Network Theory

I am not going to dive into details as much as we did for our senior E90 thesis, but I do want to cover a couple of the building blocks.

## Lions, Bears, and Many Neural Networks, oh my

I would highly encourage you to check out this website: <https://www.asimovinstitute.org/neural-network-zoo/>. I remember seeing it in college when working on this thesis and was stunned. If you're too lazy to click, check out the fun picture:

![neural-network-zoo](/images/generative-handwriting/neural_network_zoo.png){: .center-shrink }
_Reference for the image.[^1]_{: .basic-center}

We're going to explore some of the zoo in a bit more detail, specifically, focusing on [LSTMs][lstm].

## Basic Neural Network

![basic-nn](https://www.researchgate.net/publication/336440220/figure/fig3/AS:839177277042688@1577086862947/A-illustration-of-neural-networks-NNs-a-a-basic-NN-is-composed-of-the-input-output.jpg)
_Reference for the image.[^2]_{: .basic-center}

The core structure of a neural network is the connections between all of the neurons. Each connection carries an activatino signal of varying strength. If the incoming signal to a neuron is strong enough, then the signal is permeated through the next stages of the network.

There is a input layer that feeds the data into the hidden layer. The outputs from the hidden layer are then passed to the output layer. Every connection between nodes carries a weight determining the amount of information that gets passed through.

### Hyper Parameters

For a basic neural network, there are generally three [hyperparameters]:

- pattern of connections between all neurons
- weights of connections between neurons
- activation functions of the neurons

In our project however, we focus on a specific class of neural networks called Recurrent Neural Networks (RNNs), and the more specific variation of RNNs called Long Short Term Memory networks ([LSTMs][lstm]).

However, let's give a bit more context. There's really two broad types of neural networks:

- **[Feedforward Neural Network][fnn]**
- **[Recurrent Neural Networks][rnn]**

## Feedforward Neural Network

These neural networks channel information in **one direction**.

The figure above is showing a feedforward neural network because **the connectinos do not allow for the same input data to be seen multiple times by the same node.**

These networks are generally very well used for mapping raw data to categories. For example, classifying a face from an image.

Every node outputs a numerical value that it then passes to all its successor nodes. In other words:

$$
\begin{align}
y_j = f(x_j)
\end{align}
\tag{1}
$$

where

$$
\begin{align}
x_j = \sum_{i \in P_j} w_{ij} y_i
\end{align}
\tag{2}
$$

where

- $y_j$ is the output of node $j$
- $x_j$ is the total weighted input for node $j$
- $w_{ij}$ is the weight from node $i$ to node $j$
- $y_i$ is the output from node $i$
- $P_j$ represents the set of predecessor nodes to node $j$

Also note, $f(x)$ should be a smooth non-linear activation function that maps outputs to a reasonable domain. Some common activation functions include $\tanh(x)$ or the [sigmoid function][sigmoid]. These complex functions are necessary because the neural network is _literally_ trying to learn a non-linear pattern.

### Backpropagation

[Backpropagation][backprop] is the mechanism in which we pass the error back through the network starting at the output node. Generally, we minimize using stochastic gradient descent. Again, lots of different ways we can define our error, but we can use sum of squared residuals between our $k$ targets and the output of $k$ nodes of the network.

$$
\begin{align}
E = \frac{1}{2} \sum_{k}(t_k - y_k)^2
\end{align}
\tag{3}
$$

The gradient descent part comes in next. We generate the set of all gradients with respect to error and minimize these gradients. We're minimizing this:

$$
\begin{align}
g_{ij} = - \frac{\delta E}{\delta w_{ij}}
\end{align}
\tag{4}
$$

So overall, we're continually altering the weights and minimizing their individual effect oin the overall error of the outputs.

The major downfall of this simple network is that we don't have full context. With sequences, there's not enough information about the previous words, so the context is missing. And that leads us to our next structure.

## Recurrent Neural Network

[Recurrent Neural Networks (RNNs)][rnn] have a capacity to remember. This memory stems from the fact that their input is not only the current input vector but also a variation of what they output at previous time steps.

This visualization from [Christopher Olah][colah] (who holy hell i just realized is a co-founder of [Anthropic][anthropic], but who Tom and I used to follow closely in college) is a great visualization:

![rnn-unrolled](/images/generative-handwriting/rnn_unrolled.png){: .center-shrink }
_Reference for the image.[^3]_{: .basic-center}

This RNN module is being unrolled over multiple timestamps. Information is passed within a module at time step $t$ to the module at $t+1$.

Per Tom and my paper,

> An ideal RNN would theoretically be able to remember as far back as was necessary in order to to make an accurate prediction. However, as with many things, the theory does not carry over to reality. RNNs have trouble learning long term dependencies due to the vanishing gradient problem. An example of such a long term dependency might be if we are trying to predict the last word in the following sentence ”My family originally comes from Belgium so my native language is PREDICTION”. A normal RNN would possibly be able to recognize that the prediction should be a language but it would need the earlier context of Belgium to be able to accurately predict DUTCH.

Topically, this is why the craze around LLMs is so impressive. There's a lot more going on with LLMs with vector databases and how much native knowledge they have already ingested.

The notion of [backpropagation][backprop] is basically the same just we also have the added dimension of time.

The crux of the issue is that RNNs have many layers and as we begin to push the derivatives to zero. The gradients become too small and cause underflow. In actual meaning, the networks then cease to be able to learn.

However, [Sepp Hochreiter][sepp-hochreiter] and [Juergen Schmidhuber][juergen-schmidhuber] developed the [Long Short Term Memory (LSTM)][lstm] unit that solved this vanishing gradient problem.

## Long Short Term Memory Networks

[Long Short Term Memory (LSTM)][lstm] networks are specifically designed to learn long term dependencies.

Every form of RNN has repeating modules that pass information across timesteps, and LSTMs are no different. Where they different is the inner structure of each module. While a standard RNN might have a single neural layer, LSTMs have four.

![lstm-viz](/images/generative-handwriting/lstm.png){: .center-shrink }
_Reference for the image.[^3]_{: .basic-center}

### Understanding the LLM Structure

So let's better understand the structure above. There's a way more comprehensive walkthrough [here][colah]. I'd encourage you to check out that walkthrough.

![lstm-viz](/images/generative-handwriting/single_lstm_module.png){: .center-shrink }
_Reference for the image.[^3]_{: .basic-center}

The top line is key to the LSTM's ability to remember. It is called the cell state. We'll reference it as $C_t$.

The first neural network layer is a sigmoid function. It takes as input the concatenation between the current input $x_t$ and the output of the previous module $h_{t-1}$. This is coined as the forget gate. It is in control of what to forget for the cell state.

We piecewise multiply the output of the sigmoid layer $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$, with the cell state from the previous module $C_{t-1}$, forgetting the things that it doesn't see as important.

Then right in the center of the image above there are two neural network layers which make up the update gate. First, $x_t \cdot h_{t-1}$ is pushed through both a sigmoid ($\sigma$) layer and a $\tanh$ layer. The output of this sigmoid layer $i_t = \sigma (W_i \cdot [h_{t-1}, x_t] + b_C)$ determines which values to use to update, and the output of the $\tanh$ layer $\hat{C} = \sigma (W_C \cdot [h_{t-1}, x_{t} + b_C$, proposes an entirely new cell state. These two results are then piecewise multiplied and added to the current cell state (which we just edited using the forget layer) outputting the new cell state $C_t$.

The final neural network layer is called the output gate. It determines the relevant portion of the cell state to output as $h_t$. Once again, we feed $x_t \cdot h_{t-1}$ through a sigmoid layer whose output, $o_t = \sigma (W_o \cdot [h_{t-1}, x_t] + b_o)$, we piecewise multiple with $\tanh(C_t)$. The result of the multiplication determines the output of the LSTM module. Note that the <span style="color:purple">**purple**</span> $\tanh$ is not a neural network layer, but a piecewise multiplication intended to push the current cell state into a reasonable domain.

# Putting Theory into Code

I thought for awhile about how to structure this part. I, at first, was going to cover theory and code, but hopefully I've built the code out in a way that's easy to read (largely thanks to Tensorflow). So I'm going to walk through a couple of building blocks, and split each section into theory and code.

## LSTM Cell with Peephole Connections

### Theory

The basic LSTM cell (`tf.keras.layers.LSTMCell`) does not actually have the notion of peephole connections.

According to the very functional code that [sjvasquez] wrote, I don't think we actually need it, but I figured it would be fun to implement regardless. Back in the old days, when Tensorflow would support add-ons, there was some work around this [here][lstm-peep], but that project was deprecated.

### Code

The adjustment in code to the basic LSTM cell is pretty light, the code comments should be pretty descriptive.

```python
    def call(self, inputs: tf.Tensor, state: Tuple[tf.Tensor, tf.Tensor]):
        """
        This is basically implementing Graves's equations on page 5
        https://www.cs.toronto.edu/~graves/preprint.pdf
        equations 5-11.

        From the paper,
        * sigma is the logistic sigmoid function
        * i -> input gate
        * f -> forget gate
        * o -> output gate
        * c -> cell state
        * W_{hi} - hidden-input gate matrix
        * W_{xo} - input-output gate matrix
        * W_{ci} - are diagonal
          + so element m in each gate vector only receives input from
          + element m of the cell vector
        """
        # Both of these are going to be shape (?, num_lstm_units)
        h_tm1, c_tm1 = state
        # Compute linear combinations for input, forget, and output gates, and cell candidate
        # Basically the meat of eq, 7, 8, 9, 10
        z = (
            tf.matmul(inputs, self.kernel)
            + tf.matmul(h_tm1, self.recurrent_kernel)
            + self.bias
        )
        # Split the transformations into input, forget, cell, and output components
        i, f, c_candidate, o = tf.split(z, num_or_size_splits=4, axis=1)

        if self.should_apply_peephole:
            # Peephole connections before the activation functions
            i += c_tm1 * self.peephole_weights[:, 0]
            f += c_tm1 * self.peephole_weights[:, 1]

        # apply the activations - first step for eq. 7, eq. 8. eq. 10
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)

        if self.should_clip_gradients:
            # Per Graves, we need to apply gradient clipping to still fight off
            # the exploding derivative issue. It's a bit weird
            # to do it here maybe so that's why this bool defaults to off.
            i = tf.clip_by_value(i, -self.clip_value, self.clip_value)
            f = tf.clip_by_value(f, -self.clip_value, self.clip_value)
            o = tf.clip_by_value(o, -self.clip_value, self.clip_value)
            c_candidate = tf.clip_by_value(
                c_candidate, -self.clip_value, self.clip_value
            )

        c_candidate = tf.tanh(c_candidate)
        c = f * c_tm1 + i * c_candidate
        if self.should_apply_peephole:
            # Adjusting the output gate with peephole connection after computing new cell state
            o += c * self.peephole_weights[:, 2]

        # Compute final hidden state -> Equation 11
        h = o * tf.tanh(c)
        return h, [h, c]
```

## Gaussian Mixture Models

### Theory

![gmm-viz](https://miro.medium.com/v2/resize:fit:996/1*kJYirC6ewCqX1M6UiXmLHQ.gif){: .basic-center }
_Reference for the image... I didn't create this one[^4]_{: .basic-center}

[Gaussian Mixture Models][gmm] are an unsupervised technique to learn an underlying probabilistic model.

Brilliant has an incredible explanation walking through the theory [here][gmm]. I'd encourage you to check it out, but at a very high level:

1. A number of Gaussians is specified by the user
2. The algo learns various parameters that represent the data while maximizing the likelihood of seeing such data

So if we have $k$ components, for a _multivariate_ Gaussian mixture model, we'll learn $k$ means, $k$ variances, $k$ mixture weights, $k$ correlations through expectation maximization.

From [Brilliant][brilliant], there are really two steps for the EM step:

> The first step, known as the expectation step or E step, consists of calculating the expectation of the component assignments $C_k$ for each data point $x_i \in X$ given the model parameters $\phi_k, \mu_k$ , and $\sigma_k$ .
>
> The second step is known as the maximization step or M step, which consists of maximizing the expectations calculated in the E step with respect to the model parameters. This step consists of updating the values $\phi_k, \mu_k$ , and $\sigma_k$ .

### Code

There's actually not a whole lot of code to provide here. GMMs are more of the technique that we'll combine with the output of a neural network. That leads us smoothly to our next section.

## Mixture Density Networks

### Theory

[Mixture Density Networks][mdn] are an extension of GMMs that predict the parameters of a mixture probability distribution.

![mdn-viz](https://www.researchgate.net/profile/Baptiste-Feron/publication/325194613/figure/fig2/AS:643897080434701@1530528435323/Mixture-Density-Network-The-output-of-a-neural-network-parametrizes-a-Gaussian-mixture.png){: .basic-center }
_Reference for the image.[^6]_{: .basic-center}

Per our paper:

> The idea is relatively simple - we take the output from a neural network and parametrize the learned parameters of the GMM. The result is that we can infer probabilistic prediction from our learned parameters. If our neural network is reason- ably predicting where the next point might be, the GMM will then learn probabilistic parameters that model the distribution of the next point. This is different in a few key aspects. Namely, we now have target values because our data is sequential. Therefore, when we feed in our targets, we minimize the log likelihood based on those expectations, thus altering the GMM portion of the model to learn the predicted values.

More or less though, the problem we're trying to solve is predicting the next input given our output vector. Essentially, we're asking for $\text{Pr}(x_{t+1} \| y_t)$. I'm not going to show the proof (we didn't in our paper right), but the equation for the conditional probability is shown below:

$$
\begin{align}
\text{Pr}(x_{t+1} | y_t) = \sum_{j=1}^{M} \pi_{j}^t \mathcal{N} (x_{t+1} \mid \mu_j^t, \sigma_j^t, \rho_j^t)
\end{align}
\tag{5}
$$

where

$$
\begin{align}
\mathcal{N}(x \mid \mu, \sigma, \rho) = \frac{1}{2\pi \sigma_1 \sigma_2 \sqrt[]{1-\rho^2}} \exp \left[\frac{-Z}{2(1-\rho^2)}\right]
\end{align}
\tag{6}
$$

and

$$
\begin{align}
Z = \frac{(x_1 - \mu_1)^2 }{\sigma_1^2} + \frac{(x_2 - \mu_2)^2}{\sigma_2^2} - \frac{2\rho (x_1 - \mu_1) (x_2 - \mu_2) }{\sigma_1 \sigma_2}
\end{align}
\tag{7}
$$

Now, there's a slight variation here because we have a handwriting specific _end-of-stroke_ parameter. So we modify our conditional probability formula to result in our final calculation of:

$$
\begin{align}
\textrm{Pr}(x_{t+1} \mid y_t ) = \sum\limits_{j=1}\limits^{M} \pi_j^t \; \mathcal{N} (x_{t+1} \mid \mu_j^t, \sigma_j^t, \rho_j^t)
\begin{cases}
e_t & \textrm{if } (x_{t+1})_3 = 1 \\
1-e_t & \textrm{otherwise}
\end{cases}
\end{align}
\tag{8}
$$

And that's it! That's our final probability output from the MDN. Once we have this, performing our expectation maximization is simple as our loss function that we choose to minimize is just:

$$
\begin{align}
\mathcal{L}(\mathbf{x}) = - \sum\limits_{t=1}^{T} \log \textrm{Pr}(x_{t+1} \mid y_t)
\end{align}
\tag{9}
$$

### Code

Here's the corresponding code section for my mixture density network.

<details>
  <summary><b>FULL CODE HERE</b></summary>

<!-- prettier-ignore-start -->
{% highlight python %}
import tensorflow as tf
import numpy as np

from constants import NUM_MIXTURE_COMPONENTS_PER_COMPONENT


class MixtureDensityLayer(tf.keras.layers.Layer):
    def __init__(self, num_components, name="mdn", **kwargs):
        super(MixtureDensityLayer, self).__init__(name=name, **kwargs)
        self.num_components = num_components
        # The number of parameters per mixture component: 2 means, 2 standard deviations, 1 correlation
        # Plus 1 for the mixture weights and 1 for the end-of-stroke probability
        self.output_dim = num_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1
        self.name = name

    def build(self, input_shape):
        # Weights for mixture weights
        self.batch_size = input_shape[0]
        self.sequence_length = input_shape[1]
        self.W_pi = self.add_weight(
            name=f"{self.name}_W_pi",
            shape=(input_shape[-1], self.num_components),
            initializer="uniform",
            trainable=True,
        )
        # Weights for means
        self.W_mu = self.add_weight(
            name=f"{self.name}_W_mu",
            shape=(input_shape[-1], self.num_components * 2),
            initializer="uniform",
            trainable=True,
        )
        # Weights for standard deviations
        self.W_sigma = self.add_weight(
            name=f"{self.name}_W_sigma",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
            trainable=True,
        )
        # Weights for correlation coefficients
        self.W_rho = self.add_weight(
            name=f"{self.name}_W_rho",
            shape=(input_shape[-1], self.num_components),
            initializer="uniform",
            trainable=True,
        )
        # Weights for end-of-stroke probability
        self.W_eos = self.add_weight(
            name=f"{self.name}_W_eos",
            shape=(input_shape[-1], 1),
            initializer="uniform",
            trainable=True,
        )
        # Bias for mixture weights
        self.b_pi = self.add_weight(
            name=f"{self.name}_b_pi",
            shape=(self.num_components,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for means
        self.b_mu = self.add_weight(
            name=f"{self.name}_b_mu",
            shape=(self.num_components * 2,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for standard deviations
        self.b_sigma = self.add_weight(
            name=f"{self.name}_b_sigma",
            shape=(self.num_components * 2,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for correlation coefficients
        self.b_rho = self.add_weight(
            name=f"{self.name}_b_rho",
            shape=(self.num_components,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for end-of-stroke probability
        self.b_eos = self.add_weight(
            name=f"{self.name}_b_eos", shape=(1,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        eps = 1e-8
        sigma_eps = 1e-4

        pi = tf.nn.softmax(tf.matmul(inputs, self.W_pi) + self.b_pi)

        mu = tf.matmul(inputs, self.W_mu) + self.b_mu
        mu1, mu2 = tf.split(mu, num_or_size_splits=2, axis=2)

        sigma = tf.exp(tf.matmul(inputs, self.W_sigma) + self.b_sigma)
        sigmas = tf.clip_by_value(sigma, sigma_eps, np.inf)
        sigma1, sigma2 = tf.split(sigmas, num_or_size_splits=2, axis=2)

        rho = tf.tanh(tf.matmul(inputs, self.W_rho) + self.b_rho)
        rho = tf.clip_by_value(rho, eps - 1.0, 1.0 - eps)

        eos = tf.sigmoid(tf.matmul(inputs, self.W_eos) + self.b_eos)
        eos = tf.reshape(eos, [-1, inputs.shape[1], 1])

        outputs = tf.concat([pi, mu1, mu2, sigma1, sigma2, rho, eos], axis=2)

        tf.debugging.assert_shapes(
            [
                (pi, (self.batch_size, self.sequence_length, self.num_components)),
                (mu1, (self.batch_size, self.sequence_length, self.num_components)),
                (mu2, (self.batch_size, self.sequence_length, self.num_components)),
                (sigma1, (self.batch_size, self.sequence_length, self.num_components)),
                (sigma2, (self.batch_size, self.sequence_length, self.num_components)),
                (rho, (self.batch_size, self.sequence_length, self.num_components)),
                (eos, (self.batch_size, self.sequence_length, 1)),
                (outputs, (self.batch_size, self.sequence_length, self.output_dim)),
            ]
        )
        return outputs


@tf.keras.utils.register_keras_serializable()
def mdn_loss(y_true, y_pred, stroke_lengths, num_components, eps=1e-8):
    """Calculate the mixture density loss with masking for valid sequence lengths.

    Args:
    - y_true: The true next points in the sequence, with shape [batch_size, seq_length, 3].
    - y_pred: The concatenated MDN outputs, with shape [batch_size, seq_length, num_components * 6 + 1].
    - stroke_lengths: The actual lengths of each sequence in the batch, with shape [batch_size].
    - num_components: The number of mixture components.

    Returns:
    - The calculated loss.
    """

    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_rho, out_eos = tf.split(
        y_pred,
        [num_components] * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + [1],
        axis=2,
    )

    tf.debugging.assert_shapes(
        [
            (out_pi, (None, None, num_components)),
            (out_mu1, (None, None, num_components)),
            (out_mu2, (None, None, num_components)),
            (out_sigma1, (None, None, num_components)),
            (out_sigma2, (None, None, num_components)),
            (out_rho, (None, None, num_components)),
            (out_eos, (None, None, 1)),
        ]
    )
    tf.debugging.assert_greater(
        out_sigma1, 0.0, message="out_sigma1 has non-positive values"
    )
    tf.debugging.assert_greater(
        out_sigma2, 0.0, message="out_sigma2 has non-positive values"
    )
    tf.debugging.assert_near(
        tf.reduce_sum(out_pi, axis=-1),
        tf.ones_like(tf.reduce_sum(out_pi, axis=-1)),
        atol=1e-5,
        message="out_pi is not close to 1.0",
    )
    x_data, y_data, eos_data = tf.split(y_true, [1, 1, 1], axis=-1)
    norm = 1.0 / (
        2 * np.pi * out_sigma1 * out_sigma2 * tf.sqrt(1 - tf.square(out_rho) + eps)
    )
    Z = (
        tf.square((x_data - out_mu1) / (out_sigma1))
        + tf.square((y_data - out_mu2) / (out_sigma2))
        - (
            2
            * out_rho
            * (x_data - out_mu1)
            * (y_data - out_mu2)
            / (out_sigma1 * out_sigma2)
        )
    )

    exp = -Z / (2 * (1 - tf.square(out_rho)))
    gaussian_likelihoods = tf.exp(exp) * norm
    gmm_likelihood = tf.reduce_sum(out_pi * gaussian_likelihoods, axis=2)
    gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

    bernoulli_likelihood = tf.squeeze(
        tf.where(tf.equal(tf.ones_like(eos_data), eos_data), out_eos, 1 - out_eos)
    )
    bernoulli_likelihood = tf.clip_by_value(bernoulli_likelihood, eps, 1.0 - eps)

    nll = -1 * (tf.math.log(gmm_likelihood) + tf.math.log(bernoulli_likelihood))

    # Create a mask for valid sequence lengths
    if stroke_lengths is not None:
        max_len = tf.shape(y_true)[1]
        mask = tf.sequence_mask(stroke_lengths, maxlen=max_len, dtype=tf.float32)

        # Apply the mask to the negative log-likelihood
        masked_nll = nll * mask
        masked_nll = tf.where(
            tf.not_equal(mask, 0), masked_nll, tf.zeros_like(masked_nll)
        )

        # Calculate the loss, considering only the valid parts of each sequence
        loss = tf.reduce_sum(masked_nll) / tf.reduce_sum(mask)
        return loss
    else:
        return tf.reduce_sum(nll)

<!-- prettier-ignore-end -->

{% endhighlight %}

</details>
<br/>

That being said, I just want to explicitly highlight a couple of portions. We build the various parameters that we train, but then here's where we're doing the math above:

```python
        pi = tf.nn.softmax(tf.matmul(inputs, self.W_pi) + self.b_pi)

        mu = tf.matmul(inputs, self.W_mu) + self.b_mu
        mu1, mu2 = tf.split(mu, num_or_size_splits=2, axis=2)

        sigma = tf.exp(tf.matmul(inputs, self.W_sigma) + self.b_sigma)
        sigmas = tf.clip_by_value(sigma, sigma_eps, np.inf)
        sigma1, sigma2 = tf.split(sigmas, num_or_size_splits=2, axis=2)

        rho = tf.tanh(tf.matmul(inputs, self.W_rho) + self.b_rho)
        rho = tf.clip_by_value(rho, eps - 1.0, 1.0 - eps)

        eos = tf.sigmoid(tf.matmul(inputs, self.W_eos) + self.b_eos)
        eos = tf.reshape(eos, [-1, inputs.shape[1], 1])

        outputs = tf.concat([pi, mu1, mu2, sigma1, sigma2, rho, eos], axis=2)
```

And then here is where we're computing the loss (similar to above):

```python
@tf.keras.utils.register_keras_serializable()
def mdn_loss(y_true, y_pred, stroke_lengths, num_components, eps=1e-8):
    """Calculate the mixture density loss with masking for valid sequence lengths.

    Args:
    - y_true: The true next points in the sequence, with shape [batch_size, seq_length, 3].
    - y_pred: The concatenated MDN outputs, with shape [batch_size, seq_length, num_components * 6 + 1].
    - stroke_lengths: The actual lengths of each sequence in the batch, with shape [batch_size].
    - num_components: The number of mixture components.

    Returns:
    - The calculated loss.
    """

    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_rho, out_eos = tf.split(
        y_pred,
        [num_components] * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + [1],
        axis=2,
    )
    x_data, y_data, eos_data = tf.split(y_true, [1, 1, 1], axis=-1)
    norm = 1.0 / (
        2 * np.pi * out_sigma1 * out_sigma2 * tf.sqrt(1 - tf.square(out_rho) + eps)
    )
    Z = (
        tf.square((x_data - out_mu1) / (out_sigma1))
        + tf.square((y_data - out_mu2) / (out_sigma2))
        - (
            2
            * out_rho
            * (x_data - out_mu1)
            * (y_data - out_mu2)
            / (out_sigma1 * out_sigma2)
        )
    )

    exp = -Z / (2 * (1 - tf.square(out_rho)))
    gaussian_likelihoods = tf.exp(exp) * norm
    gmm_likelihood = tf.reduce_sum(out_pi * gaussian_likelihoods, axis=2)
    gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

    bernoulli_likelihood = tf.squeeze(
        tf.where(tf.equal(tf.ones_like(eos_data), eos_data), out_eos, 1 - out_eos)
    )
    bernoulli_likelihood = tf.clip_by_value(bernoulli_likelihood, eps, 1.0 - eps)

    nll = -1 * (tf.math.log(gmm_likelihood) + tf.math.log(bernoulli_likelihood))

    if stroke_lengths is not None:
        max_len = tf.shape(y_true)[1]
        mask = tf.sequence_mask(stroke_lengths, maxlen=max_len, dtype=tf.float32)

        # Apply the mask to the negative log-likelihood
        masked_nll = nll * mask
        masked_nll = tf.where(
            tf.not_equal(mask, 0), masked_nll, tf.zeros_like(masked_nll)
        )

        # Calculate the loss, considering only the valid parts of each sequence
        loss = tf.reduce_sum(masked_nll) / tf.reduce_sum(mask)
        return loss
    else:
        return tf.reduce_sum(nll)

```

## Attention Mechanism

### Theory

The attention mechanism really only comes into play with the Synthesis Network which sadly [Tom][tom] and I never got to in college. The idea (similar to most attention notions) is that we need to tell our model more specifically where to focus. This isn't like the transformer notion of attention from the famous "Attention is All You Need" paper, but it's the idea that we have various Gaussians to indicate probabilistically where we should be focusing. We utilize one-hot encoding vectors over our input characters so that we can more clearly identify the numerical representation. So the question we're basically answering is like "oh, i see a 'w' character, generally how far along do we need to write for that?" to help also answer the question of when do we need to terminate.

The mathematical representation is here:

> Given a length $U$ character sequence $\mathbf{c}$ and a length $T$ data sequence $\mathbf{x}$, the soft window $w_t$ into $\mathbf{c}$ at timestep $t$ ($1 \leq t \leq T$) is defined by the following discrete convolution with a mixture of $K$ Gaussian functions
>
> $$
> \begin{align}
> \phi(t, u) &= \sum_{k=1}^K \alpha^k_t\exp\left(-\beta_t^k\left(\kappa_t^k-u\right)^2\right)\\
> w_t &= \sum_{u=1}^U \phi(t, u)c_u
> \end{align}
> $$
>
> where $\phi(t, u)$ is the \emph{window weight} of $c_u$ at timestep $t$.
>
> Intuitively, the $\kappa_t$ parameters control the location of the window, the $\beta_t$ parameters control the width of the window and the $\alpha_t$ parameters control the importance of the window within the mixture.
>
> The size of the soft window vectors is the same as the size of the character vectors $c_u$ (assuming a one-hot encoding, this will be the number of characters in the alphabet).
>
> Note that the window mixture is not normalised and hence does not determine a probability distribution; however the window weight $\phi(t, u)$ can be loosely interpreted as the network's belief that it is writing character $c_u$ at time $t$.

### Code

## Stacked LSTM

### Theory

The one distinction between Graves's setup and a standard LSTM is that Graves uses a _cascade_ of LSTMs. So we use the MDN to generate a probabilistic prediction however our neural network is the cascade of LSTMs.

Per our paper:

> The LSTM cascade buys us a few different things. As Graves aptly points out, it mitigates the vanishing gradient problem even more greatly than a single LSTM could. This is because of the skip-connections. All hidden layers have access to the input and all hidden layers are also directly connected to the output node. As a result, there are less processing steps from the bottom of the network to the top.

So it looks something like this:

![graves-stacked-lstm](/images/generative-handwriting/graves_stacked_lstm.png){: .center-shrink }
_Alex Graves 2013.[^5]_{: .basic-center}

The one thing to note is that there is a dimensionality increase given we now have these hidden layers. Tom and I broke this down in our paper here:

> Let's observe the $x_{t-1}$ input. $h_{t-1}^1$ only has $x_{t-1}$ as its input which is in $\mathbb{R}^3$ because $(x, y, eos)$. However, we also pass our input $x_{t-1}$ into $h_{t-1}^2$. We assume that we simply concatenate the original input and the output of the first hidden layer. Because LSTMs do not scale dimensionality, we know the output is going to be in $\mathbb{R}^3$ as well. Therefore, after this concatenation, the input into the second hidden layer will be in $\mathbb{R}^6$. We can follow this process through and see that, the input to the third hidden layer will be in $\mathbb{R}^9$. Finally, we concatenate all of the LSTM cells (i.e. the hidden layers) together, thus getting a final dimension of $\mathbb{R}^{18}$ fed into our MDN. Note, this is for $m=3$ hidden layers, but more generally, we can observe the relation as
>
> $$\begin{align} \textrm{final dimension} = k \frac{m(m+1)}{2} \end{align}$$

### Code

This is where the various `cell` vs `layer` concept in Tensorflow was very nice.

From our top level model, we have:

```python
class DeepHandwritingSynthesisModel(tf.keras.Model):
    """
    A similar implementation to the previous model, but with a different approach to the attention mechanism. This is batched for efficiency
    """

    def __init__(
        self,
        units=400,
        num_layers=3,
        num_mixture_components=20,
        num_chars=73,
        num_attention_gaussians=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_chars = num_chars
        self.lstm_cells = [LSTMPeepholeCell(units, idx) for idx in range(num_layers)]
        self.attention_mechanism = AttentionMechanism(
            num_gaussians=num_attention_gaussians, num_chars=num_chars
        )
        self.attention_rnn_cell = AttentionRNNCell(
            self.lstm_cells, self.attention_mechanism, self.num_chars
        )
        self.rnn_layer = tf.keras.layers.RNN(
            self.attention_rnn_cell, return_sequences=True
        )
        self.mdn_layer = MixtureDensityLayer(num_mixture_components)
```

You can see here how the parts all come together smoothly. The custom RNN cell takes the lstm_cells (which are stacked), and then can basically abstract out and operate on the individual time steps without having to worry about actually introducing another `for` loop. This is beneficial because of the batching and GPU win we can get when it eventually becomes time.

For the actual `AttentionRNNCell`, this is what I'm talking about:

```python
class AttentionRNNCell(tf.keras.layers.Layer):
    def __init__(self, lstm_cells, attention_mechanism, num_chars, **kwargs):
        super().__init__(**kwargs)
        self.lstm_cells = lstm_cells
        self.attention_mechanism = attention_mechanism
        self.num_chars = num_chars
        self.state_size = [cell.state_size for cell in lstm_cells] + [
            tf.TensorShape([attention_mechanism.num_gaussians]),
            tf.TensorShape([num_chars]),
        ]
        self.output_size = lstm_cells[-1].output_size
        # The one-hot encoding of the character sequence
        # will be set by the model before calling the cell
        self.char_seq_one_hot = None
        # Same for the length of the character sequence
        self.char_seq_len = None

    def call(self, inputs, states):
        assert self.char_seq_one_hot is not None, "char_seq_one_hot is not set"
        assert self.char_seq_len is not None, "char_seq_len is not set"

        x_t = inputs
        (
            s1_state_h,
            s1_state_c,
            s2_state_h,
            s2_state_c,
            s3_state_h,
            s3_state_c,
            kappa,
            w,
        ) = states

        # LSTM layer 1
        s1_in = tf.concat([w, x_t], axis=1)
        s1_out, s1_state_new = self.lstm_cells[0](s1_in, [s1_state_h, s1_state_c])

        # Attention
        attention_inputs = tf.concat([w, x_t, s1_out], axis=1)
        w_new, kappa_new = self.attention_mechanism(
            attention_inputs, kappa, self.char_seq_one_hot, self.char_seq_len
        )

        # LSTM layer 2
        s2_in = tf.concat([x_t, s1_out, w_new], axis=1)
        s2_out, s2_state_new = self.lstm_cells[1](s2_in, [s2_state_h, s2_state_c])

        # LSTM layer 3
        s3_in = tf.concat([x_t, s2_out, w_new], axis=1)
        s3_out, s3_state_new = self.lstm_cells[2](s3_in, [s3_state_h, s3_state_c])

        # Preparing new states as a list to return
        new_states = [
            s1_state_new[0],
            s1_state_new[1],
            s2_state_new[0],
            s2_state_new[1],
            s3_state_new[0],
            s3_state_new[1],
            kappa_new,
            w_new,
        ]

        return s3_out, new_states
```

And the various concatenation between inputs is what Graves means when he says "deep in space and time".

## Final Result

Alright finally! So what do we have, and what can we do now?

We now are going to feed the output from our LSTM cascade into the GMM in order to build a probabilistic prediction model for the next stroke. The GMM will then be fed the actual next point, in order to create some idea of the deviation os that the loss can be properly minimized.

# Code

Ok so that's all well and good and some fun math and neural network construction, but the meat of this project is about what we're actually building with this theory. So let's lay out our to do list.

## Plan of Attack

## Looking back at college code 😬

Yeah, so I haven't looked at a lot of my college work recently, but damn. This was SO messy. And it's fair, I was a senior in college and didn't have industry experience, and you know we basically were treating our codebase like many scripts tied together.

But yeah... here's a look at some of the way we had organized things:

![being-old](/images/generative-handwriting/college_messy_structure.png){: .center-super-shrink }

**Ouch**. 🧹

<div class="markdown-alert markdown-alert-note">
  <p>So... I basically started fresh with a new repo, given Tensorflow has changed so much, and I couldn't really bear to look at the old code.</p>
</div>

## Data and Loading Data

We're using the [IAM Online Handwriting Database][iam-database]. Specifically, I'm looking at `data/lineStrokes-all.tar.gz`, which is XML data that looks like this:

![data](/images/generative-handwriting/example_data.png){: .center-super-shrink }

There's also this note:

> The database is divided into 4 parts, a training set, a first validation set, a second validation set and a final test set. The training set may be used for training the recognition system, while the two validation sets may be used for optimizing some meta-parameters. The final test set must be left unseen until the final test is performed. Note that you are allowed to use also other data for training etc, but report all the changes when you publish your experimental results and let the test set unchanged (It contains 3859 sequences, i.e. XML-files - one for each text line).

So that determines our training set, validation set, second validation set, and a final test set.

## Model and Tensorflow Updates

One thing to note. For the sake of my time (given I'm no longer doing an engineering thesis, I'm going to lean a bit more on Tensorflow than Tom and I did in college). So for example, [Tensorflow][tensorflow] has a [pre-built LSTM component][tensorflow-lstm] which I'll be taking advantage of.

## MDN Predictions with Simple Neural Network

Ok so... this was a **humbling** experience. Given how much Tensorflow has changed, and how rusty I am at building out neural network models, I decided to start from the scratch (similar to Tom and my paper back in the day). Step 1: just implement a basic neural network layer, some 2D data, and feed that into the network and then the MDN to get prediction of the x and y data.

🎉 However, voila - not tooooo painful 😅. Here's the code, it's relatively straightforward.

```python
class MDNLayer(tf.keras.layers.Layer):
    def __init__(self, num_components, **kwargs):
        super(MDNLayer, self).__init__(**kwargs)
        self.num_components = num_components
        # The number of parameters per mixture component: 2 means, 2 standard deviations, 1 correlation
        # Plus 1 for the mixture weights and 1 for the end-of-stroke probability
        self.output_dim = num_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1

    def build(self, input_shape):
        # Weights for mixture weights
        self.W_pi = self.add_weight(
            name="W_pi", shape=(input_shape[-1], self.num_components), initializer="uniform", trainable=True
        )
        # Weights for means
        self.W_mu = self.add_weight(
            name="W_mu", shape=(input_shape[-1], self.num_components * 2), initializer="uniform", trainable=True
        )
        # Weights for standard deviations
        self.W_sigma = self.add_weight(
            name="W_sigma",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),  # Small stddev
            trainable=True,
        )
        # Weights for correlation coefficients
        self.W_rho = self.add_weight(
            name="W_rho", shape=(input_shape[-1], self.num_components), initializer="uniform", trainable=True
        )
        # Weights for end-of-stroke probability
        self.W_eos = self.add_weight(name="W_eos", shape=(input_shape[-1], 1), initializer="uniform", trainable=True)
        # Bias for mixture weights
        self.b_pi = self.add_weight(name="b_pi", shape=(self.num_components,), initializer="zeros", trainable=True)
        # Bias for means
        self.b_mu = self.add_weight(name="b_mu", shape=(self.num_components * 2,), initializer="zeros", trainable=True)
        # Bias for standard deviations
        self.b_sigma = self.add_weight(
            name="b_sigma", shape=(self.num_components * 2,), initializer="zeros", trainable=True
        )
        # Bias for correlation coefficients
        self.b_rho = self.add_weight(name="b_rho", shape=(self.num_components,), initializer="zeros", trainable=True)
        # Bias for end-of-stroke probability
        self.b_eos = self.add_weight(name="b_eos", shape=(1,), initializer="zeros", trainable=True)

        # Again, i think we need this?
        self.built = True

    def call(self, inputs):
        pi = tf.nn.softmax(tf.matmul(inputs, self.W_pi) + self.b_pi)
        mu = tf.matmul(inputs, self.W_mu) + self.b_mu
        sigma = tf.exp(tf.clip_by_value(tf.matmul(inputs, self.W_sigma) + self.b_sigma, -5, 5))
        sigma = tf.debugging.check_numerics(sigma, "Check sigma: ")
        rho = tf.tanh(tf.matmul(inputs, self.W_rho) + self.b_rho)
        eos = tf.sigmoid(tf.matmul(inputs, self.W_eos) + self.b_eos)
        eos = tf.reshape(eos, [-1, inputs.shape[1], 1])
        outputs = tf.concat([pi, mu, sigma, rho, eos], axis=2)
        return outputs

def mdn_loss(y_true, y_pred, num_components):
"""Calculate the mixture density loss.

    Args:
    - y_true: The true next points in the sequence, with shape [batch_size, seq_length, 3].
              The last dimension is (delta x, delta y, end_of_stroke).
    - y_pred: The concatenated MDN outputs, with shape [batch_size, seq_length, num_components * 6 + 1].
              This consists of mixture weights, means, log standard deviations, correlation coefficients,
              and end stroke probabilities.
    - num_components: The number of mixture components.

    Returns:
    - The calculated loss.
    """

    out_pi, out_sigma1, out_sigma2, out_rho, out_mu1, out_mu2, out_eos = tf.split(
        y_pred, [num_components] * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + [1], axis=-1
    )
    out_pi = tf.debugging.check_numerics(out_pi, "Check out_pi: ")
    out_sigma1 = tf.maximum(out_sigma1, EPSILON)
    out_sigma2 = tf.maximum(out_sigma2, EPSILON)

    # Since direct broadcasting to a target shape with None is not possible,
    # adjust shapes without relying on tf.broadcast_to
    # Expand dimensions of x_data, y_data to allow TensorFlow to broadcast automatically
    x_data, y_data, eos_data = tf.split(y_true, [1, 1, 1], axis=-1)

    # HEre, we should just be taking advantage of Tensy's broadcasting
    z = (
        tf.square((x_data - out_mu1) / out_sigma1)
        + tf.square((y_data - out_mu2) / out_sigma2)
        - 2 * out_rho * (x_data - out_mu1) * (y_data - out_mu2) / (out_sigma1 * out_sigma2)
    )

    # Ensure out_rho is within a valid range to avoid sqrt of negative numbers
    safe_out_rho = tf.clip_by_value(out_rho, -1 + EPSILON, 1 - EPSILON)

    # Add an epsilon to denom to ensure it cannot be zero or too close to zero
    denom = 2 * np.pi * out_sigma1 * out_sigma2 * tf.sqrt(1 - tf.square(safe_out_rho)) + EPSILON

    # Recalculate gaussian_exp_arg with clipping to avoid extreme values
    max_exp_arg = 50
    gaussian_exp_arg = -z / (2 * (1 - tf.square(safe_out_rho)))
    gaussian_exp_arg = tf.clip_by_value(gaussian_exp_arg, -max_exp_arg, max_exp_arg)

    # Calculate gaussian, ensuring the result is within a valid probability range
    gaussian = tf.exp(gaussian_exp_arg) / denom

    # I was getting slammed by nans and derivatives going to infinity
    gaussian = tf.debugging.check_numerics(gaussian, "Check gaussian after division: ")

    weighted_gaussian = out_pi * gaussian
    weighted_gaussian = tf.debugging.check_numerics(weighted_gaussian, "Check weighted_gaussian: ")
    weighted_sum = tf.reduce_sum(weighted_gaussian, axis=2, keepdims=True)
    weighted_sum = tf.debugging.check_numerics(weighted_sum, "Check weighted_sum: ")

    eos_likelihood = eos_data * out_eos + (1 - eos_data) * (1 - out_eos)

    loss_gaussian = -tf.reduce_sum(tf.math.log(tf.maximum(weighted_sum, EPSILON)))
    loss_gaussian = tf.debugging.check_numerics(loss_gaussian, "Check loss_gaussian: ")
    loss_eos = -tf.reduce_sum(tf.math.log(tf.maximum(eos_likelihood, EPSILON)))
    loss_eos = tf.debugging.check_numerics(loss_eos, "Check loss_eos: ")
    negative_log_likelihood = (loss_gaussian + loss_eos) / tf.cast(tf.size(x_data), dtype=tf.float32)

    return negative_log_likelihood
```

## Predictive Handwriting Network

Ok so let's just look at our model. This is by far my favorite part of this whole process. I'm not smart enough (like Alex Graves) to figure out the architectures and build such generational impact as something like an LSTM cell, but I do love that we can translate mathematical equations to code. Let's observe:

<details>
<summary><b>Full Code</b></summary>

<div class="language-python highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="k">class</span> <span class="nc">LSTMCellWithPeepholes</span><span class="p">(</span><span class="n">tf</span><span class="p">.</span><span class="n">keras</span><span class="p">.</span><span class="n">layers</span><span class="p">.</span><span class="n">Layer</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">num_lstm_units</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">(</span><span class="n">LSTMCellWithPeepholes</span><span class="p">,</span> <span class="bp">self</span><span class="p">).</span><span class="n">__init__</span><span class="p">(</span><span class="o">**</span><span class="n">kwargs</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span> <span class="o">=</span> <span class="n">num_lstm_units</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">state_size</span> <span class="o">=</span> <span class="p">[</span><span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span><span class="p">,</span> <span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span><span class="p">]</span>

    <span class="k">def</span> <span class="nf">build</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">input_shape</span><span class="p">):</span>
        <span class="s">"""
        Building the LSTM cell with peephole connections.
        Basically defining all of the appropriate weights, biases, and peephole weights.
        """</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">kernel</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">add_weight</span><span class="p">(</span>
            <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">input_shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">],</span> <span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span> <span class="o">*</span> <span class="mi">4</span><span class="p">),</span> <span class="n">initializer</span><span class="o">=</span><span class="s">"uniform"</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s">"kernel"</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">recurrent_kernel</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">add_weight</span><span class="p">(</span>
            <span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span><span class="p">,</span> <span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span> <span class="o">*</span> <span class="mi">4</span><span class="p">),</span> <span class="n">initializer</span><span class="o">=</span><span class="s">"uniform"</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s">"recurrent_kernel"</span>
        <span class="p">)</span>

        <span class="c1"># Peephole weights for input, forget, and output gates

</span> <span class="bp">self</span><span class="p">.</span><span class="n">peephole_weights</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">add_weight</span><span class="p">(</span>
<span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span><span class="p">,</span> <span class="mi">3</span><span class="p">),</span> <span class="n">initializer</span><span class="o">=</span><span class="s">"uniform"</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s">"peephole_weights"</span>
<span class="p">)</span>
<span class="bp">self</span><span class="p">.</span><span class="n">bias</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">add_weight</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span> <span class="o">\*</span> <span class="mi">4</span><span class="p">,),</span> <span class="n">initializer</span><span class="o">=</span><span class="s">"zeros"</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s">"bias"</span><span class="p">)</span>

        <span class="c1"># Apparently - and again struggles of Tensorflow - this is imperative because if you define

</span> <span class="c1"># a built method then it's more or less lazy loaded, and so you need to set this
</span> <span class="c1"># property so tensorflow knows that you're totally live for your Layer.
</span> <span class="bp">self</span><span class="p">.</span><span class="n">built</span> <span class="o">=</span> <span class="bp">True</span>

    <span class="k">def</span> <span class="nf">call</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">inputs</span><span class="p">:</span> <span class="n">tf</span><span class="p">.</span><span class="n">Tensor</span><span class="p">,</span> <span class="n">states</span><span class="p">:</span> <span class="n">Tuple</span><span class="p">[</span><span class="n">tf</span><span class="p">.</span><span class="n">Tensor</span><span class="p">,</span> <span class="n">tf</span><span class="p">.</span><span class="n">Tensor</span><span class="p">]):</span>
        <span class="s">"""
        This is basically implementing Graves's equations on page 5
        https://www.cs.toronto.edu/~graves/preprint.pdf
        equations 5-11.

        From the paper,
        * sigma is the logistic sigmoid function
        * i -&gt; input gate
        * f -&gt; forget gate
        * o -&gt; output gate
        * c -&gt; cell state
        * W_{hi} - hidden-input gate matrix
        * W_{xo} - input-output gate matrix
        * W_{ci} - are diagonal
          + so element m in each gate vector only receives input from
          + element m of the cell vector
        """</span>
        <span class="c1"># Both of these are going to be shape (?, num_lstm_units)

</span> <span class="n">h_tm1</span><span class="p">,</span> <span class="n">c_tm1</span> <span class="o">=</span> <span class="n">states</span>

        <span class="c1"># Compute linear combinations for input, forget, and output gates, and cell candidate

</span> <span class="n">z</span> <span class="o">=</span> <span class="n">tf</span><span class="p">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">inputs</span><span class="p">,</span> <span class="bp">self</span><span class="p">.</span><span class="n">kernel</span><span class="p">)</span> <span class="o">+</span> <span class="n">tf</span><span class="p">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">h_tm1</span><span class="p">,</span> <span class="bp">self</span><span class="p">.</span><span class="n">recurrent_kernel</span><span class="p">)</span> <span class="o">+</span> <span class="bp">self</span><span class="p">.</span><span class="n">bias</span>

        <span class="c1"># Split the transformations into input, forget, cell, and output components

</span> <span class="n">i</span><span class="p">,</span> <span class="n">f</span><span class="p">,</span> <span class="n">c_candidate</span><span class="p">,</span> <span class="n">o</span> <span class="o">=</span> <span class="n">tf</span><span class="p">.</span><span class="n">split</span><span class="p">(</span><span class="n">z</span><span class="p">,</span> <span class="n">num_or_size_splits</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

        <span class="c1"># Peephole connections: directly use without reshaping, assuming self.peephole_weights is defined with shape (num_lstm_units, 3)

</span> <span class="n">i*peephole</span> <span class="o">=</span> <span class="n">c_tm1</span> <span class="o">*</span> <span class="bp">self</span><span class="p">.</span><span class="n">peephole*weights</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="c1"># For input gate
</span> <span class="n">f_peephole</span> <span class="o">=</span> <span class="n">c_tm1</span> <span class="o">*</span> <span class="bp">self</span><span class="p">.</span><span class="n">peephole*weights</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span> <span class="c1"># For forget gate
</span>
<span class="c1"># Update cell state -&gt; element-wise multiply forget gate with cell state
</span> <span class="c1"># This is equation 9 from Graves's paper
</span> <span class="n">c</span> <span class="o">=</span> <span class="n">f</span> <span class="o">*</span> <span class="n">c*tm1</span> <span class="o">+</span> <span class="n">i</span> <span class="o">*</span> <span class="n">tf</span><span class="p">.</span><span class="n">tanh</span><span class="p">(</span><span class="n">c_candidate</span><span class="p">)</span>

        <span class="n">o_peephole</span> <span class="o">=</span> <span class="n">c</span> <span class="o">*</span> <span class="bp">self</span><span class="p">.</span><span class="n">peephole_weights</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">]</span>  <span class="c1"># For output gate

</span>
        <span class="c1"># add the peephole adjustment before the gate activations
</span>        <span class="n">i</span> <span class="o">+=</span> <span class="n">i_peephole</span>
        <span class="n">f</span> <span class="o">+=</span> <span class="n">f_peephole</span>
        <span class="n">o</span> <span class="o">+=</span> <span class="n">o_peephole</span>

        <span class="c1"># apply the activations - first step for eq. 7, eq. 8. eq 9

</span> <span class="n">i</span> <span class="o">=</span> <span class="n">tf</span><span class="p">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>
<span class="n">f</span> <span class="o">=</span> <span class="n">tf</span><span class="p">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
<span class="n">o</span> <span class="o">=</span> <span class="n">tf</span><span class="p">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">o</span><span class="p">)</span>

        <span class="c1"># Compute final hidden state -&gt; Equation 11

</span> <span class="n">h</span> <span class="o">=</span> <span class="n">o</span> <span class="o">\*</span> <span class="n">tf</span><span class="p">.</span><span class="n">tanh</span><span class="p">(</span><span class="n">c</span><span class="p">)</span>
<span class="k">return</span> <span class="n">h</span><span class="p">,</span> <span class="p">[</span><span class="n">h</span><span class="p">,</span> <span class="n">c</span><span class="p">]</span>

    <span class="k">def</span> <span class="nf">get_config</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="n">config</span> <span class="o">=</span> <span class="nb">super</span><span class="p">(</span><span class="n">LSTMCellWithPeepholes</span><span class="p">,</span> <span class="bp">self</span><span class="p">).</span><span class="n">get_config</span><span class="p">()</span>
        <span class="n">config</span><span class="p">.</span><span class="n">update</span><span class="p">({</span><span class="s">"units"</span><span class="p">:</span> <span class="bp">self</span><span class="p">.</span><span class="n">num_lstm_units</span><span class="p">})</span>
        <span class="k">return</span> <span class="n">config</span>

</code></pre></div></div>

</details>
<br>

This code:

```python
c = f * c_tm1 + i * tf.tanh(c_candidate)

o_peephole = c * self.peephole_weights[:, 2]  # For output gate

# add the peephole adjustment before the gate activations
i += i_peephole
f += f_peephole
o += o_peephole

# apply the activations - first step for eq. 7, eq. 8. eq 9
i = tf.sigmoid(i)
f = tf.sigmoid(f)
o = tf.sigmoid(o)

# Compute final hidden state -> Equation 11
h = o * tf.tanh(c)
return h, [h, c]
```

corresponds exactly to these equations:

$$
\begin{align}
i_t &= \sigma\left(W_{xi} x_t + W_{hi} h_{t-1} + W_{ci} c_{t-1}  + b_i\right) \tag{7} \\
f_t &= \sigma\left(W_{xf} x_t + W_{hf} h_{t-1} + W_{cf} c_{t-1} + b_f \right) \tag{8} \\
c_t &= f_t c_{t-1} + i_t \tanh \left(W_{xc} x_t + W_{hc} h_{t-1} + b_c\right) \tag{9} \\
o_t &= \sigma\left(W_{xo} x_t + W_{ho} h_{t-1} + W_{co} c_{t} + b_o\right) \tag{10} \\
h_t &= o_t \tanh(c_t) \tag{11}
\end{align}
$$

<div class="markdown-alert markdown-alert-note">
  <p>These equation tags are from the paper so there is overlap. </p>
</div>

## Synthetic Handwriting Generation Network

# Results

## Basic MDN Predictions with Simple Network

## Predictive Handwriting Visualizations

## Generative Handwriting Visualizations

# Troubles

God dang, I _almost_ forgot from college how annoying debugging Tensorflow can be.

![first-time-debugging](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiaqcJg0701YTca6MJxrTUyX5MBA7xtbrHfzob4TOEPnT7DCl1iDMyU82PTn9kpoQKrvTWhZgm0bBXWCTssu841uf5XsIgwOZx7wsLAhxsm1Jk6aBs9Jw9JTimFx7wDa0rAROLKImifIoE/s1600/47446306_2163002007046355_8400391397395398656_n.png){: .center-super-shrink }

The big thing is just figuring out all of the dimensionality of your various `tensor`s and making sure that broadcasting works smoothly.

So the **core** of our processing was done in this dimension:

**Input Dimension:**

$$
\left[\text{batch size}, \text{sequence length}, 3\right]
$$

The final $3$ dimension is because we're operating on $(x, y, \text{eos})$ data.

**Output Dimension from Mixture Density Network (MDN):**

$$
\left[
\text{batch size}, \text{sequence length}, \text{num mixtures} * 6
\right]
$$

This output dimension arises from the MDN and includes parameters for each mixture component. Each mixture component requires six parameters:

- Weight ($\pi_{i}$) for each mixture component.
- Mean ($\mu_{i}$) for both the $x$ and $y$ dimensions of each mixture component.
- Standard deviation ($\sigma_{i}$) for both the $x$ and $y$ dimensions of each mixture component.
- End of stroke ($e_{i}$) as a lone dimension representing the end of stroke probability (from Bernoulli)

# Hardware

So - I need a new laptop - but I did all of the local development on my Macbook 2020 that has 500GB of SSD and only 8GB of RAM. And no fancy Nvidia GPU enabled to speed this up exponentially. So I wrote the code locally but then just rented an AWS EC2 instance to actually run the code.

# Conclusion

When re-reading my old draft blog post, I liked the way I ended things. So here it is:

> Finally, I want to leave with a quote from our academic advisor [Matt Zucker][mz]. When I asked him when we know that our model is good enough, he responded with the following.
>
> > "Learning never stops."

[^1]: https://www.asimovinstitute.org/neural-network-zoo/
[^2]: https://www.researchgate.net/publication/336440220/figure/fig3/AS:839177277042688@1577086862947/A-illustration-of-neural-networks-NNs-a-a-basic-NN-is-composed-of-the-input-output.jpg
[^3]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[^4]: https://miro.medium.com/v2/resize:fit:996/1*kJYirC6ewCqX1M6UiXmLHQ.gif
[^5]: https://arxiv.org/abs/1308.0850
[^6]: https://www.researchgate.net/profile/Baptiste-Feron/publication/325194613/figure/fig2/AS:643897080434701@1530528435323/Mixture-Density-Network-The-output-of-a-neural-network-parametrizes-a-Gaussian-mixture.png

[comment]: <> (Bibliography)
[code]: https://github.com/johnlarkin1/sudoku-solver
[swat]: http://www.swarthmore.edu/
[tom]: https://www.linkedin.com/in/tom-wilmots-030781a6/
[lse]: http://www.lse.ac.uk/
[mz]: https://mzucker.github.io/
[ag]: https://en.wikipedia.org/wiki/Alex_Graves_(computer_scientist)
[paper]: https://arxiv.org/abs/1308.0850
[gravesToronto]: https://www.cs.toronto.edu/~graves/
[dha]: https://www.linkedin.com/in/david-ha-168a012/
[arxiv]: https://arxiv.org/
[toc]: https://www.cs.swarthmore.edu/~fontes/cs46/17s/index.php
[lila]: https://www.cs.swarthmore.edu/~fontes/
[turingmachines]: https://en.wikipedia.org/wiki/Turing_machine
[ntm]: https://arxiv.org/abs/1410.5401
[tf]: https://www.tensorflow.org/
[virtualenvironments]: http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/
[otoro]: http://blog.otoro.net/
[backprop]: https://en.wikipedia.org/wiki/Backpropagation
[tensorboard]: https://www.tensorflow.org/get_started/summaries_and_tensorboard
[e90]: https://www.swarthmore.edu/engineering/e90-senior-design-project
[twjlpaper]: {{ site.baseurl }}/pdfs/Handwriting-Synthesis-E90.pdf
[tensorflow]: https://www.tensorflow.org/
[keras]: https://keras.io/
[pytorch]: https://pytorch.org/
[hyperparameters]: https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
[rnn]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[fnn]: https://en.wikipedia.org/wiki/Feedforward_neural_network
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
[anthropic]: https://www.anthropic.com/
[colah]: https://colah.github.io/
[understanding-rnn]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[sepp-hochreiter]: https://en.wikipedia.org/wiki/Sepp_Hochreiter
[juergen-schmidhuber]: https://en.wikipedia.org/wiki/J%C3%BCrgen_Schmidhuber
[gmm]: https://brilliant.org/wiki/gaussian-mixture-model/
[mdn]: https://www.katnoria.com/mdn/
[brilliant]: https://brilliant.org/
[iam-database]: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
[tensorflow-lstm]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
[sjvasquez]: https://github.com/sjvasquez
[dropbox]: https://dropbox.com/
[mojo]: https://www.mojo.com/
[vast]: https://vast.ai/
[lstm-peep]: https://www.tensorflow.org/addons/api_docs/python/tfa/rnn/PeepholeLSTMCell
