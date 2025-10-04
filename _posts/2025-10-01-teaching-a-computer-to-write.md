---
title: "Teaching a Computer How to Write"
layout: post
featured-gif: TODO
mathjax: true
pinned: true
categories: [⭐️ Favorites, Algorithms, Dev, A.I., M.L.]
summary: TODO
---

<div class="markdown-alert markdown-alert-disclaimer">

<p>This is a relatively long post! I would encourage you if you're trying to learn from 0 -> 1 to read the whole thing, but feel free to jump around as you so wish. I would say there's three main portions: concept, theory, and code.
</p>

<p>My purpose here was to build up from the basics and really understand the flow. I provide quite a couple of models so we can see the progression from a simple neural net to a basic LSTM to Peephole LSTM to a stacked cascade of Peephole LSTMs to Mixture Density Networks to Attention Mechanism to Attention RNN to the Handwriting Prediction Network to finally throwing it all together to the full Handwriting Synthesis Network that Graves originally wrote about.</p>

<p>Enjoy!</p>

</div>

<div class="markdown-alert markdown-alert-note">
<p>One thing that I would highly recommend - if you're interested in the theory of LSTMs and why sigmoid vs tanh activations were chosen, I would really encourage reading Chris Olah's <b><a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTMs</a></b> blog post. It does a fantastic job.</p>
</div>

<br>

# ✍️ Motivating Visualizations

Today, we're going to learn how to teach a computer to write. I don't mean generating text (which would have been probably a better thing to study in college), I mean learning to write like a human learns how to write with a pen and paper. My results eventually were pretty good, here are some motivating visualizations:

<!-- TODO:@larkin -->

# Table of Contents

- [✍️ Motivating Visualizations](#️-motivating-visualizations)
- [Table of Contents](#table-of-contents)
- [🥅 Motivation](#-motivation)
- [👨‍🏫 History](#-history)
  - [Tom and My Engineering Thesis](#tom-and-my-engineering-thesis)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Concept](#-concept)
- [👾 Software](#-software)
  - [Tensorflow](#tensorflow)
    - [Programming Paradigm](#programming-paradigm)
    - [Versions - How the times have changed](#versions---how-the-times-have-changed)
    - [Tensorboard](#tensorboard)
  - [Pytorch](#pytorch)
  - [JAX](#jax)
    - [Programming Paradigm](#programming-paradigm-1)
- [📊 Data](#-data)
- [🧠 Base Neural Network Theory](#-base-neural-network-theory)
  - [Lions, Bears, and Many Neural Networks, oh my](#lions-bears-and-many-neural-networks-oh-my)
  - [Basic Neural Network](#basic-neural-network)
    - [Hyper Parameters](#hyper-parameters)
  - [Feedforward Neural Network](#feedforward-neural-network)
    - [Backpropagation](#backpropagation)
  - [Recurrent Neural Network](#recurrent-neural-network)
  - [Long Short Term Memory Networks](#long-short-term-memory-networks)
    - [Understanding the LLM Structure](#understanding-the-llm-structure)
- [🧬 Concepts to Code](#-concepts-to-code)
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
- [🏆 Results](#-results)
  - [Vast AI GPU Enabled Execution](#vast-ai-gpu-enabled-execution)
    - [Problem #1 - Gradient Explosion Problem](#problem-1---gradient-explosion-problem)
    - [Problem #2 - OOM Galore](#problem-2---oom-galore)
    - [Sanity Check - Validating Model Dimensions (with AI... so somewhat)](#sanity-check---validating-model-dimensions-with-ai-so-somewhat)
  - [Drawing Code](#drawing-code)
  - [Visualizations](#visualizations)
    - [Learning with Dummy Data](#learning-with-dummy-data)
- [Conclusion](#conclusion)

# 🥅 Motivation

This motivation is clear - this is something that I have wanted to find the time to do right since college. My engineering thesis was on this Graves paper. My senior year, I worked with my good friend (also he's a brilliant engineer) [Tom Wilmots][tom] to understand and dive into this paper.

I'm going to pull pieces of that, but the time has changed, and I wanted to revisit some of the work we did, hopefully clean it up, and finally put a nail in this (so my girlfriend / friends don't have to keep hearing about it).

# 👨‍🏫 History

[Tom] and I were very interested in the concept of teaching a computer how to write in college. There is a very famous [paper] that was published around 2013 from Canadian computer scientist [Alex Graves][ag], titled [_Generating Sequences With Recurrent Neural Networks_][gen-sequences]. At [Swarthmore][swat], you have to do Engineering thesis, called [E90s][e90]. It's basically a year (although I'd argue it's more of a semester when it all shakes out) long project focused on doing a piece of work you're proud of.

## Tom and My Engineering Thesis

For the actual paper that we wrote, check it out here:

<div style="text-align: center;">
    <embed src="{{ '/pdfs/Handwriting-Synthesis-E90.pdf' | prepend: site.baseurl }}" width="500" height="375" type="application/pdf">
</div>

You can also check it out here: [**Application of Neural Networks with Handwriting Samples**][paper].

# 🙏 Acknowledgements

Before I dive in, I do want to make some acknowledgements just given this is a partial resumption of work.

- **[Tom Wilmots][tom]** - One of the brightest and best engineers I've worked with. He was an Engineering and Economics double major from [Swarthmore][swat]. Pretty sure I would have failed my E90 thesis without him.
- **[Matt Zucker][mz]** - One of my role models and constant inspirations, Matt was kind enough to be Tom and my academic advisor for this final engineering project. He is the best professor I've come across.
- **[Alex Graves][ag]** - A professor that both Tom and I had the pleasure of working with. **He responded to our emails, which I'm still very appreciative of**. You can see more about his work at the University of Toronto [here][gravesToronto]). He is the author of [this paper][paper], which Matt found for us and pretty much was the basis of our project. He's also the creator of the [Neural Turing Machine][ntm], which peaked my interest after having taken [Theory of Computation][toc], with my other fantastic professor [Lila Fontes][lila] and learning about [Turing machines][turingmachines].
- **[David Ha][dha]** - Another brilliant scientist who we had the privilege of corresponding with. Check out his blog [here][otoro]. It's beautiful. He also is very prolific on [ArXiv][arxiv] which is always cool to see.

# 📝 Concept

This section is going to be for non-technical people to understand what we were trying to do. It's relatively simple. At a very high level, **we are trying to teach a computer how to generate human looking handwriting**. To do that, we are going to train a neural network. We are going to use a public dataset, called [IAM Online Handwriting Database][iam-database]. This dataset had a ton of people write on a tablet where the data was being recorded. It collected basically sets of `Stroke` data, which were tuples of $(x, y, t)$, where $(x, y)$ are the coordinates on the tablet, and $t$ is the timestamp. We'll use this data to train a model so that across all of the participants we have this blended approach of how to write like a human.

# 👾 Software

In college, we decided between [Tensorflow][tensorflow] and [Pytorch][pytorch]. In college, we used Tensorflow. However, given the times, I wanted to still resume our tensorflow approach with updated designs, but I also wanted to try and use [JAX]. [JAX] is... newer. But it's gotten some hype online and I think there's a solid amount of adoption across the bigger AI labs now. In my opinion, Tensorflow is dying, Pytorch is the new status quo, and JAX is the new kid on the block. However, I'm not an ML researcher clearing millions of dollars. So grain of salt. This [clickbaity article][pytorch-rant] which declares _"Pytorch is dead. Long live JAX"_ got a ton of flak online, but regardless... it piqued my interest enough to try it here.

I'll cover all three here and yeah probably dive deepest into tensorflow... but feel free to skip this section.

## Tensorflow

### Programming Paradigm

Tensorflow has this interesting programming paradigm, where you are more or less creating a graph. You define `Tensor`s and then when you run your dependency graph, those things are actually translated.

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

## Pytorch

[PyTorch][pytorch] is now basically the defacto standard for most serious research labs and AI shops. To me, it seems like things are still somewhat ported to Tensorflow for production, but I'm not totally sure about convention.

Pytorch seems to thread the line between Tensorflow and JAX. Functions don't necessarily need to be pure to be utilized. You can loop and mutate state in a `nn.Module` just fine.

I won't be covering pytorch but I certainly will come back around to it in later projects.

## JAX

The new up and comer! I think it's largely a crowd favorite for it's speed. Documentation is obviously worse. One Redditor summarized it nicely:

<blockquote class="reddit-embed-bq" data-embed-theme="dark" data-embed-height="396"><a href="https://www.reddit.com/r/MachineLearning/comments/1b08qv6/comment/ks6u1e2/">Comment</a><br> by<a href="https://www.reddit.com/user/Few-Pomegranate4369/">u/Few-Pomegranate4369</a> from discussion<a href="https://www.reddit.com/r/MachineLearning/comments/1b08qv6/d_is_it_worth_switching_to_jax_from/"></a><br> in<a href="https://www.reddit.com/r/MachineLearning/">MachineLearning</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>

Documentation is certainly worse and I hit numerous roadblocks where functions weren't actually pure and then the JIT compile portion basically failed on startup.

### Programming Paradigm

JAX and Pytorch are definitely the most like traditional Python imperative flow. The restriction on JAX is largely around pure functions. Tensorflow is also gradually moving away from the compile your graph and then run it paradigm.

# 📊 Data

We're using the [IAM Online Handwriting Database][iam-database]. Specifically, I'm looking at `data/lineStrokes-all.tar.gz`, which is XML data that looks like this:

![data](/images/generative-handwriting/example_data.png){: .center-super-shrink }

There's also this note:

> The database is divided into 4 parts, a training set, a first validation set, a second validation set and a final test set. The training set may be used for training the recognition system, while the two validation sets may be used for optimizing some meta-parameters. The final test set must be left unseen until the final test is performed. Note that you are allowed to use also other data for training etc, but report all the changes when you publish your experimental results and let the test set unchanged (It contains 3859 sequences, i.e. XML-files - one for each text line).

So that determines our training set, validation set, second validation set, and a final test set.

# 🧠 Base Neural Network Theory

I am not going to dive into details as much as we did for our senior E90 thesis, but I do want to cover a couple of the building blocks.

## Lions, Bears, and Many Neural Networks, oh my

I would highly encourage you to check out this website: <https://www.asimovinstitute.org/neural-network-zoo/>. I remember seeing it in college when working on this thesis and was stunned. If you're too lazy to click, check out the fun picture:

![neural-network-zoo](/images/generative-handwriting/neural_network_zoo.png){: .center-shrink }
_Reference for the image.[^1]_{: .basic-center}

We're going to explore some of the zoo in a bit more detail, specifically, focusing on [LSTMs][lstm].

## Basic Neural Network

![basic-nn](https://www.researchgate.net/publication/336440220/figure/fig3/AS:839177277042688@1577086862947/A-illustration-of-neural-networks-NNs-a-a-basic-NN-is-composed-of-the-input-output.jpg)
_Reference for the image.[^2]_{: .basic-center}

The core structure of a neural network is the connections between all of the neurons. Each connection carries an activation signal of varying strength. If the incoming signal to a neuron is strong enough, then the signal is permeated through the next stages of the network.

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

The figure above is showing a feedforward neural network because **the connections do not allow for the same input data to be seen multiple times by the same node.**

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

[Backpropagation][backprop] is the mechanism in which we pass the error back through the network starting at the output node. Generally, we minimize using [stochastic gradient descent][stoch-grad-desc]. Again, lots of different ways we can define our error, but we can use sum of squared residuals between our $k$ targets and the output of $k$ nodes of the network.

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

Topically, this is why the craze around LLMs is so impressive. There's a lot more going on with LLMs... which... I will not cover here.

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

The first neural network layer is a sigmoid function. It takes as input the concatenation between the current input $x_t$ and the output of the previous module $h_{t-1}$. This is coined as the forget gate. It is in control of what to forget for the cell state. The sigmoid function is a good architecture decision here because it basically outputs numbers between [0,1] indicating how much the layer should let through.

We piecewise multiply the output of the sigmoid layer $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$, with the cell state from the previous module $C_{t-1}$, forgetting the things that it doesn't see as important.

Then right in the center of the image above there are two neural network layers which make up the update gate. First, $x_t \cdot h_{t-1}$ is pushed through both a sigmoid ($\sigma$) layer and a $\tanh$ layer. The output of this sigmoid layer $i_t = \sigma (W_i \cdot [h_{t-1}, x_t] + b_C)$ determines which values to use to update, and the output of the $\tanh$ layer $\hat{C} = \sigma (W_C \cdot [h_{t-1}, x_{t} + b_C$, proposes an entirely new cell state. These two results are then piecewise multiplied and added to the current cell state (which we just edited using the forget layer) outputting the new cell state $C_t$.

The final neural network layer is called the output gate. It determines the relevant portion of the cell state to output as $h_t$. Once again, we feed $x_t \cdot h_{t-1}$ through a sigmoid layer whose output, $o_t = \sigma (W_o \cdot [h_{t-1}, x_t] + b_o)$, we piecewise multiple with $\tanh(C_t)$. The result of the multiplication determines the output of the LSTM module. Note that the <span style="color:purple">**purple**</span> $\tanh$ is not a neural network layer, but a piecewise multiplication intended to push the current cell state into a reasonable domain.

<div class="markdown-alert markdown-alert-note">
<p><b>I'm serious... you guys should check out Olah's <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTMs</a>. Here he is back in 2015 strongly foreshadowing transformers given the focus on attention (which is truly the hardest part of all this) blog post.</b></p>
</div>

![olah-attention](/images/generative-handwriting/olah-attention.png){: .center-shrink }
_Reference for the image.[^3]_{: .basic-center}

<br>

# 🧬 Concepts to Code

When very first starting this project, I kind of figured that I would be able to use some of my college code, but looking back. It's quite a mess and I don't think that's the way to go about it.

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

![mdn-viz](https://towardsdatascience.com/wp-content/uploads/2024/05/1UKuoYsGWis22cOV7KpLjVg.png){: .basic-center }
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

```python
@tf.keras.utils.register_keras_serializable()
class AttentionMechanism(tf.keras.layers.Layer):
    """
    Attention mechanism for the handwriting synthesis model.
    This is a version of the attention mechanism used in
    the original paper by Alex Graves. It uses a Gaussian
    window to focus on different parts of the character sequence
    at each time step.
    """

    def __init__(self, num_gaussians, num_chars, name="attention", **kwargs):
        super(AttentionMechanism, self).__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.num_chars = num_chars
        self.name_mod = name

    def build(self, input_shape):
        self.dense_attention = tf.keras.layers.Dense(
            units=3 * self.num_gaussians,
            activation="softplus",
            name=f"{self.name_mod}_dense",
        )
        super().build(input_shape)

    def call(self, inputs, prev_kappa, char_seq_one_hot, sequence_lengths):
        # Generate concatenated attention parameters - just utilizing
        # the dense layer so that I don't have to manually define the matrix
        attention_params = self.dense_attention(inputs)
        alpha, beta, kappa_increment = tf.split(attention_params, 3, axis=1)

        # Normalize and clip kappa and beta...
        alpha = tf.maximum(alpha, 1e-8)
        beta = tf.maximum(beta, 1e-8)
        kappa_increment = tf.maximum(kappa_increment, 1e-8)

        kappa = prev_kappa + kappa_increment
        char_len = tf.shape(char_seq_one_hot)[1]
        batch_size = tf.shape(inputs)[0]
        u = tf.cast(tf.range(0, char_len), tf.float32)  # Shape: [char_len]
        u = tf.reshape(u, [1, 1, -1])  # Shape: [1, 1, char_len]
        u = tf.tile(u, [batch_size, self.num_gaussians, 1])  # Shape: [batch_size, num_gaussians, char_len]

        # gaussian window
        alpha = tf.expand_dims(alpha, axis=-1)  # Shape: [batch_size, num_gaussians, 1]
        beta = tf.expand_dims(beta, axis=-1)  # Shape: [batch_size, num_gaussians, 1]
        kappa = tf.expand_dims(kappa, axis=-1)  # Shape: [batch_size, num_gaussians, 1]

        # phi - attention weights
        phi = alpha * tf.exp(-beta * tf.square(kappa - u))  # Shape: [batch_size, num_gaussians, char_len]
        phi = tf.reduce_sum(phi, axis=1)  # Sum over the gaussians: [batch_size, char_len]

        # sequence mask
        sequence_mask = tf.sequence_mask(sequence_lengths, maxlen=char_len, dtype=tf.float32)
        phi = phi * sequence_mask  # Apply mask to attention weights

        # normalize
        phi_sum = tf.reduce_sum(phi, axis=1, keepdims=True) + 1e-8
        phi = phi / phi_sum

        # window vec
        phi = tf.expand_dims(phi, axis=-1)  # Shape: [batch_size, char_len, 1]
        w = tf.reduce_sum(phi * char_seq_one_hot, axis=1)  # Shape: [batch_size, num_chars]

        return w, kappa[:, :, 0]  # Return updated kappa

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_gaussians": self.num_gaussians,
                "num_chars": self.num_chars,
                "name": self.name_mod,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

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

# 🏆 Results

## Vast AI GPU Enabled Execution

```bash
2024-04-21 19:01:02.183969: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
train data found. Loading...
test data found. Loading...
valid2 data found. Loading...
valid1 data found. Loading...
2024-04-21 19:01:04.798925: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1928] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22455 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:82:00.0, compute capability: 8.6
2024-04-21 19:01:05.887036: I external/local_tsl/tsl/profiler/lib/profiler_session.cc:104] Profiler session initializing.
2024-04-21 19:01:05.887070: I external/local_tsl/tsl/profiler/lib/profiler_session.cc:119] Profiler session started.
2024-04-21 19:01:05.887164: I external/local_xla/xla/backends/profiler/gpu/cupti_tracer.cc:1239] Profiler found 1 GPUs
2024-04-21 19:01:05.917572: I external/local_tsl/tsl/profiler/lib/profiler_session.cc:131] Profiler session tear down.
2024-04-21 19:01:05.917763: I external/local_xla/xla/backends/profiler/gpu/cupti_tracer.cc:1364] CUPTI activity buffer flushed
Epoch 1/10000
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1713726072.109654    2329 service.cc:145] XLA service 0x7ad5bc004600 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1713726072.109731    2329 service.cc:153]   StreamExecutor device (0): NVIDIA GeForce RTX 3090, Compute Capability 8.6
2024-04-21 19:01:12.346749: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
W0000 00:00:1713726072.691839    2329 assert_op.cc:38] Ignoring Assert operator assert_greater/Assert/AssertGuard/Assert
W0000 00:00:1713726072.694098    2329 assert_op.cc:38] Ignoring Assert operator assert_greater_1/Assert/AssertGuard/Assert
W0000 00:00:1713726072.696267    2329 assert_op.cc:38] Ignoring Assert operator assert_near/Assert/AssertGuard/Assert
2024-04-21 19:01:13.095183: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:465] Loaded cuDNN version 8906
2024-04-21 19:01:14.883021: W external/local_xla/xla/service/hlo_rematerialization.cc:2941] Can't reduce memory use below 17.97GiB (19297974672 bytes) by rematerialization; only reduced to 20.51GiB (22027581828 bytes), down from 20.67GiB (22193496744 bytes) originally
I0000 00:00:1713726076.329853    2329 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
167/168 ━━━━━━━━━━━━━━━━━━━━ 0s 685ms/step - loss: 2.8113W0000 00:00:1713726191.427557    2333 assert_op.cc:38] Ignoring Assert operator assert_greater/Assert/AssertGuard/Assert
W0000 00:00:1713726191.429182    2333 assert_op.cc:38] Ignoring Assert operator assert_greater_1/Assert/AssertGuard/Assert
W0000 00:00:1713726191.430622    2333 assert_op.cc:38] Ignoring Assert operator assert_near/Assert/AssertGuard/Assert
2024-04-21 19:03:13.488256: W external/local_xla/xla/service/hlo_rematerialization.cc:2941] Can't reduce memory use below 17.97GiB (19298282069 bytes) by rematerialization; only reduced to 19.75GiB (21203023676 bytes), down from 19.87GiB (21340423652 bytes) originally
168/168 ━━━━━━━━━━━━━━━━━━━━ 0s 709ms/step - loss: 2.8097
Epoch 1: Saving model.

Epoch 1: Loss improved from None to 0.0, saving model.
Model parameters after the 1st epoch:
Model: "deep_handwriting_synthesis_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm_peephole_cell                   │ ?                           │         764,400 │
│ (LSTMPeepholeCell)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_peephole_cell_1                 │ ?                           │       1,404,400 │
│ (LSTMPeepholeCell)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_peephole_cell_2                 │ ?                           │       1,404,400 │
│ (LSTMPeepholeCell)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ attention (AttentionMechanism)       │ ?                           │          14,310 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ attention_rnn_cell                   │ ?                           │       3,587,510 │
│ (AttentionRNNCell)                   │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rnn (RNN)                            │ ?                           │       3,587,510 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ mdn (MixtureDensityLayer)            │ ?                           │          48,521 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,272,064 (27.74 MB)
 Trainable params: 3,636,031 (13.87 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 3,636,033 (13.87 MB)

All parameters:
===============
[[ lstm_peephole_kernel1 ]] shape: (76, 1600)
[[ lstm_peephole_recurrent_kernel1 ]] shape: (400, 1600)
[[ lstm_peephole_weights1 ]] shape: (400, 3)
[[ lstm_peephole_bias1 ]] shape: (1600,)
[[ lstm_peephole_kernel2 ]] shape: (476, 1600)
[[ lstm_peephole_recurrent_kernel2 ]] shape: (400, 1600)
[[ lstm_peephole_weights2 ]] shape: (400, 3)
[[ lstm_peephole_bias2 ]] shape: (1600,)
[[ lstm_peephole_kernel3 ]] shape: (476, 1600)
[[ lstm_peephole_recurrent_kernel3 ]] shape: (400, 1600)
[[ lstm_peephole_weights3 ]] shape: (400, 3)
[[ lstm_peephole_bias3 ]] shape: (1600,)
[[ kernel ]] shape: (476, 30)
[[ bias ]] shape: (30,)
[[ mdn_W_pi ]] shape: (400, 20)
[[ mdn_W_mu ]] shape: (400, 40)
[[ mdn_W_sigma ]] shape: (400, 40)
[[ mdn_W_rho ]] shape: (400, 20)
[[ mdn_W_eos ]] shape: (400, 1)
[[ mdn_b_pi ]] shape: (20,)
[[ mdn_b_mu ]] shape: (40,)
[[ mdn_b_sigma ]] shape: (40,)
[[ mdn_b_rho ]] shape: (20,)
[[ mdn_b_eos ]] shape: (1,)

Trainable parameters:
=====================
[[ lstm_peephole_kernel1 ]] shape: (76, 1600)
[[ lstm_peephole_recurrent_kernel1 ]] shape: (400, 1600)
[[ lstm_peephole_weights1 ]] shape: (400, 3)
[[ lstm_peephole_bias1 ]] shape: (1600,)
[[ lstm_peephole_kernel2 ]] shape: (476, 1600)
[[ lstm_peephole_recurrent_kernel2 ]] shape: (400, 1600)
[[ lstm_peephole_weights2 ]] shape: (400, 3)
[[ lstm_peephole_bias2 ]] shape: (1600,)
[[ lstm_peephole_kernel3 ]] shape: (476, 1600)
[[ lstm_peephole_recurrent_kernel3 ]] shape: (400, 1600)
[[ lstm_peephole_weights3 ]] shape: (400, 3)
[[ lstm_peephole_bias3 ]] shape: (1600,)
[[ kernel ]] shape: (476, 30)
[[ bias ]] shape: (30,)
[[ mdn_W_pi ]] shape: (400, 20)
[[ mdn_W_mu ]] shape: (400, 40)
[[ mdn_W_sigma ]] shape: (400, 40)
[[ mdn_W_rho ]] shape: (400, 20)
[[ mdn_W_eos ]] shape: (400, 1)
[[ mdn_b_pi ]] shape: (20,)
[[ mdn_b_mu ]] shape: (40,)
[[ mdn_b_sigma ]] shape: (40,)
[[ mdn_b_rho ]] shape: (20,)
[[ mdn_b_eos ]] shape: (1,)

Trainable parameter count:
==========================
3636031
168/168 ━━━━━━━━━━━━━━━━━━━━ 133s 728ms/step - loss: 2.7931
Epoch 2/10000
 60/168 ━━━━━━━━━━━━━━━━━━━━ 1:14 686ms/step - loss: 2.4870
```

Ok so that's all well and good and some fun math and neural network construction, but the meat of this project is about what we're actually building with this theory. So let's lay out our to-do list.

### Problem #1 - Gradient Explosion Problem

Somehow on my first run through of this, I was still getting explodient gradients in the later stages of training my model.

As a result, I chose the laborious and time consuming process to run the training model on CPU so that I could print out debugging information and then run `tensorboard`'s Debugger model so I could inspect which gradients were exploding to `nan` or dreaded `inf`.

Here's an example of what that looked like:

![tensorboard](/images/generative-handwriting/tensorboard-debugging.png){: .center-image}

Which was even more annoying because of this: <https://github.com/tensorflow/tensorflow/issues/59215> issue.

### Problem #2 - OOM Galore

Uh oh, looks like the `vast.ai` instance I utilized didn't have enough memory. Here is an example of one of the errors I ran into:

<details>
  <summary style="background-color: #f8d7da; padding: 10px; border-radius: 5px; cursor: pointer; color: #721c24; font-weight: bold;">
    Out of memory error here
  </summary>

<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>Out of memory while trying to allocate 22271409880 bytes.
BufferAssignment OOM Debugging.
BufferAssignment stats:
             parameter allocation:   29.54MiB
              constant allocation:         4B
        maybe_live_out allocation:   27.74MiB
     preallocated temp allocation:   20.74GiB
                 total allocation:   20.77GiB
Peak buffers:
        Buffer 1:
                Size: 3.40GiB
                Operator: op_type="EmptyTensorList" op_name="gradient_tape/deep_handwriting_synthesis_model_1/rnn_1/while/deep_handwriting_synthesis_model_1/rnn_1/while/attention_rnn_cell_1/lstm_peephole_cell_2_1/MatMul/ReadVariableOp_0/accumulator" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,476,1600]
                ==========================

        Buffer 2:
                Size: 3.40GiB
                Operator: op_type="EmptyTensorList" op_name="gradient_tape/deep_handwriting_synthesis_model_1/rnn_1/while/deep_handwriting_synthesis_model_1/rnn_1/while/attention_rnn_cell_1/lstm_peephole_cell_2_1/MatMul/ReadVariableOp_0/accumulator" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,476,1600]
                ==========================

        Buffer 3:
                Size: 2.86GiB
                Operator: op_type="EmptyTensorList" op_name="gradient_tape/deep_handwriting_synthesis_model_1/rnn_1/while/deep_handwriting_synthesis_model_1/rnn_1/while/attention_rnn_cell_1/lstm_peephole_cell_2_1/MatMul_1/ReadVariableOp_0/accumulator" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,400,1600]
                ==========================

        Buffer 4:
                Size: 2.86GiB
                Operator: op_type="EmptyTensorList" op_name="gradient_tape/deep_handwriting_synthesis_model_1/rnn_1/while/deep_handwriting_synthesis_model_1/rnn_1/while/attention_rnn_cell_1/lstm_peephole_cell_2_1/MatMul_1/ReadVariableOp_0/accumulator" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,400,1600]
                ==========================

        Buffer 5:
                Size: 2.86GiB
                Operator: op_type="EmptyTensorList" op_name="gradient_tape/deep_handwriting_synthesis_model_1/rnn_1/while/deep_handwriting_synthesis_model_1/rnn_1/while/attention_rnn_cell_1/lstm_peephole_cell_2_1/MatMul_1/ReadVariableOp_0/accumulator" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,400,1600]
                ==========================

        Buffer 6:
                Size: 556.64MiB
                Operator: op_type="EmptyTensorList" op_name="gradient_tape/deep_handwriting_synthesis_model_1/rnn_1/while/deep_handwriting_synthesis_model_1/rnn_1/while/attention_rnn_cell_1/lstm_peephole_cell_1/MatMul/ReadVariableOp_0/accumulator" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,76,1600]
                ==========================

        Buffer 7:
                Size: 219.73MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,10,75]
                ==========================

        Buffer 8:
                Size: 219.73MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,10,75]
                ==========================

        Buffer 9:
                Size: 219.73MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,10,75]
                ==========================

        Buffer 10:
                Size: 139.45MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,476]
                ==========================

        Buffer 11:
                Size: 139.45MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,476]
                ==========================

        Buffer 12:
                Size: 139.45MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,476]
                ==========================

        Buffer 13:
                Size: 117.19MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,400]
                ==========================

        Buffer 14:
                Size: 117.19MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,400]
                ==========================

        Buffer 15:
                Size: 117.19MiB
                Operator: op_type="While" op_name="deep_handwriting_synthesis_model_1/rnn_1/while" source_file="/root/code/venv/lib/python3.11/site-packages/tensorflow/python/framework/ops.py" source_line=1177
                XLA Label: fusion
                Shape: f32[1200,64,400]
                ==========================


         [[]]

</code></pre></div></div>

</details>

<br/>

### Sanity Check - Validating Model Dimensions (with AI... so somewhat)

So where does AI using AI come in? I wanted to validate that the shapes of my training parameters all looked good. Again, I print out the shapes on the very first epoch to get some more details and do a quick dimensionality alignment check. I then fed that into ChatGPT along with the paper and asked it to double check all my params. Here was it's output:

<details>
  <summary style="background-color: #d4edda; padding: 10px; border-radius: 5px; cursor: pointer; color: #155724; font-weight: bold;">
    Model Summary
  </summary>
<blockquote><h3 id="model-summary">Model Summary</h3><p><strong>Model Name:</strong> deep_handwriting_synthesis_model<br> <strong>Total Parameters:</strong> 7,272,063 (27.74 MB)<br> <strong>Trainable Parameters:</strong> 3,636,031 (13.87 MB)<br> <strong>Non-Trainable Parameters:</strong> 0 (0.00 B)<br> <strong>Optimizer Parameters:</strong> 3,636,032 (13.87 MB)</p><hr><h3 id="layer-wise-breakdown">Layer-wise Breakdown</h3><h4 id="1-lstm-peephole-cells">1. LSTM Peephole Cells</h4><p>You have three LSTM peephole cells:</p><ul><li><code class="language-plaintext highlighter-rouge">lstm_peephole_cell</code></li><li><code class="language-plaintext highlighter-rouge">lstm_peephole_cell_1</code></li><li><code class="language-plaintext highlighter-rouge">lstm_peephole_cell_2</code></li></ul><h5 id="parameters-and-shapes">Parameters and Shapes</h5><p>For each LSTM peephole cell:</p><ul><li><strong>Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(input_dim, 4 * units)</code></li><li><strong>Recurrent Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(units, 4 * units)</code></li><li><strong>Peephole Weights Shape:</strong> <code class="language-plaintext highlighter-rouge">(units, 3)</code></li><li><strong>Bias Shape:</strong> <code class="language-plaintext highlighter-rouge">(4 * units,)</code></li></ul><h5 id="shapes">Shapes:</h5><p><strong>First LSTM Peephole Cell (lstm_peephole_cell):</strong></p><ul><li><strong>Input Dimension:</strong> 76</li><li><strong>Units:</strong> 400</li><li><strong>Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(76, 1600)</code> &nbsp;&nbsp;&nbsp;(76, 4 * 400)</li><li><strong>Recurrent Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(400, 1600)</code> &nbsp;&nbsp;&nbsp;(400, 4 * 400)</li><li><strong>Peephole Weights Shape:</strong> <code class="language-plaintext highlighter-rouge">(400, 3)</code> &nbsp;&nbsp;&nbsp;(400 units, 3 gates)</li><li><strong>Bias Shape:</strong> <code class="language-plaintext highlighter-rouge">(1600,)</code> &nbsp;&nbsp;&nbsp;(4 * 400,)</li></ul><p><strong>Second and Third LSTM Peephole Cells (lstm_peephole_cell_1 and lstm_peephole_cell_2):</strong></p><ul><li><strong>Input Dimension:</strong> 476<br> This includes concatenated inputs from previous layers and attention outputs.</li><li><strong>Units:</strong> 400</li><li><strong>Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(476, 1600)</code> &nbsp;&nbsp;&nbsp;(476, 4 * 400)</li><li><strong>Recurrent Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(400, 1600)</code> &nbsp;&nbsp;&nbsp;(400, 4 * 400)</li><li><strong>Peephole Weights Shape:</strong> <code class="language-plaintext highlighter-rouge">(400, 3)</code> &nbsp;&nbsp;&nbsp;(400 units, 3 gates)</li><li><strong>Bias Shape:</strong> <code class="language-plaintext highlighter-rouge">(1600,)</code> &nbsp;&nbsp;&nbsp;(4 * 400,)</li></ul><h5 id="conclusion">Conclusion</h5><p>The shapes of the LSTM peephole cells’ parameters are correct and align with standard LSTM architectures with peephole connections.</p><hr><h4 id="2-attention-mechanism">2. Attention Mechanism</h4><ul><li><strong>Layer:</strong> <code class="language-plaintext highlighter-rouge">attention</code> (AttentionMechanism)</li></ul><h5 id="parameters-and-shapes-1">Parameters and Shapes</h5><ul><li><strong>Kernel Shape:</strong> <code class="language-plaintext highlighter-rouge">(476, 30)</code></li><li><strong>Bias Shape:</strong> <code class="language-plaintext highlighter-rouge">(30,)</code></li></ul><p><strong>Explanation:</strong></p><ul><li><strong>Input Dimension:</strong> 476<br> This likely comes from concatenating the window vector <code class="language-plaintext highlighter-rouge">w</code>, the current input <code class="language-plaintext highlighter-rouge">x_t</code>, and the output from the first LSTM layer <code class="language-plaintext highlighter-rouge">s1_out</code>.</li><li><strong>Output Dimension:</strong> 30<br> Corresponds to parameters needed for the attention mechanism, such as the attention weights.</li></ul><h5 id="conclusion-1">Conclusion</h5><p>The attention mechanism’s parameter shapes are appropriate.</p><hr><h4 id="3-mixture-density-network-mdn-layer">3. Mixture Density Network (MDN) Layer</h4><ul><li><strong>Layer:</strong> <code class="language-plaintext highlighter-rouge">mdn</code> (MixtureDensityLayer)</li></ul><h5 id="parameters-and-shapes-2">Parameters and Shapes</h5><ul><li><strong>mdn_W_pi:</strong> <code class="language-plaintext highlighter-rouge">(400, 20)</code></li><li><strong>mdn_b_pi:</strong> <code class="language-plaintext highlighter-rouge">(20,)</code></li><li><strong>mdn_W_mu:</strong> <code class="language-plaintext highlighter-rouge">(400, 40)</code></li><li><strong>mdn_b_mu:</strong> <code class="language-plaintext highlighter-rouge">(40,)</code></li><li><strong>mdn_W_sigma:</strong> <code class="language-plaintext highlighter-rouge">(400, 40)</code></li><li><strong>mdn_b_sigma:</strong> <code class="language-plaintext highlighter-rouge">(40,)</code></li><li><strong>mdn_W_rho:</strong> <code class="language-plaintext highlighter-rouge">(400, 20)</code></li><li><strong>mdn_b_rho:</strong> <code class="language-plaintext highlighter-rouge">(20,)</code></li><li><strong>mdn_W_eos:</strong> <code class="language-plaintext highlighter-rouge">(400, 1)</code></li><li><strong>mdn_b_eos:</strong> <code class="language-plaintext highlighter-rouge">(1,)</code></li></ul><p><strong>Explanation:</strong></p><ul><li><strong>Hidden Units from Last LSTM Layer:</strong> 400</li><li><strong>Number of Mixture Components:</strong> 20</li></ul><h5 id="mdn-outputs">MDN Outputs:</h5><ul><li><strong>pi</strong> (mixture weights): 20 components</li><li><strong>mu</strong> (means): 2 coordinates per component * 20 components = 40 outputs</li><li><strong>sigma</strong> (standard deviations): 2 per component * 20 components = 40 outputs</li><li><strong>rho</strong> (correlation coefficients): 20 components</li><li><strong>eos</strong> (end-of-stroke probability): 1 output</li></ul><h5 id="parameter-shapes">Parameter Shapes:</h5><ul><li><strong>Weights:</strong></li><li><code class="language-plaintext highlighter-rouge">mdn_W_pi</code>: <code class="language-plaintext highlighter-rouge">(400, 20)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_W_mu</code>: <code class="language-plaintext highlighter-rouge">(400, 40)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_W_sigma</code>: <code class="language-plaintext highlighter-rouge">(400, 40)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_W_rho</code>: <code class="language-plaintext highlighter-rouge">(400, 20)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_W_eos</code>: <code class="language-plaintext highlighter-rouge">(400, 1)</code></li><li><strong>Biases:</strong></li><li><code class="language-plaintext highlighter-rouge">mdn_b_pi</code>: <code class="language-plaintext highlighter-rouge">(20,)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_b_mu</code>: <code class="language-plaintext highlighter-rouge">(40,)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_b_sigma</code>: <code class="language-plaintext highlighter-rouge">(40,)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_b_rho</code>: <code class="language-plaintext highlighter-rouge">(20,)</code></li><li><code class="language-plaintext highlighter-rouge">mdn_b_eos</code>: <code class="language-plaintext highlighter-rouge">(1,)</code></li></ul><h5 id="conclusion-2">Conclusion</h5><p>The MDN layer’s parameter shapes correctly correspond to the required outputs for a mixture of 20 bivariate Gaussian components.</p><hr><h4 id="4-total-parameters-and-trainable-parameters">4. Total Parameters and Trainable Parameters</h4><ul><li><strong>Total Parameters:</strong> 7,272,063</li><li><strong>Trainable Parameters:</strong> 3,636,031</li></ul><p><strong>Observation:</strong></p><ul><li>The total parameters are approximately double the trainable parameters.</li><li>This may be due to counting both the weights and optimizer states (like moments in the Adam optimizer).</li><li><strong>Optimizer Parameters:</strong> 3,636,032 (slightly off by one, which could be due to rounding or reporting).</li></ul><h5 id="conclusion-3">Conclusion</h5><p>The parameter counts make sense, and the discrepancy is acceptable.</p><hr><h3 id="your-models-parameters">Your Model’s Parameters</h3><h4 id="all-parameters">All Parameters</h4><p>All parameters listed are correctly shaped and correspond to their respective layers.</p><h4 id="trainable-parameters">Trainable Parameters</h4><p>All trainable parameters match the parameters in the model and have the correct shapes.</p><hr><h3 id="answering-your-question">Answering Your Question</h3><p>Yes, based on the model summary and the parameter shapes you’ve provided, everything appears to be in order. The shapes align with the expected dimensions &gt; given your model’s architecture:</p><ul><li><strong>LSTM Peephole Cells:</strong></li><li>Input dimensions and units are consistent.</li><li>Kernels and recurrent kernels have correct shapes.</li><li><p>Peephole weights and biases are appropriately sized.</p></li><li><strong>Attention Mechanism:</strong></li><li><p>The input and output dimensions match the concatenated inputs and expected output size.</p></li><li><strong>MDN Layer:</strong></li><li><p>The weights and biases correspond correctly to the number of mixture components and the outputs required for the mixture density network.</p></li><li><strong>Parameter Counts:</strong></li><li>The total number of parameters is reasonable for a model of this complexity.</li><li>The trainable parameters are correctly identified.</li></ul><hr><h3 id="final-thoughts">Final Thoughts</h3><ul><li><p><strong>Proceed with Training:</strong> You can confidently continue running your training code.</p></li><li><strong>Monitoring:</strong> Keep an eye on training metrics such as loss, especially since you have confirmed that negative loss values are acceptable in your MDN context.</li><li><strong>Validation:</strong> Consider using a validation set to monitor overfitting and generalization performance.</li><li><strong>Performance Checks:</strong> If you encounter any unexpected behavior during training, revisit the data preprocessing steps and ensure that the inputs to your model are correctly shaped and normalized.</li></ul><hr><h3 id="additional-recommendations">Additional Recommendations</h3><ul><li><strong>Logging:</strong> Utilize TensorBoard or similar tools to visualize training progress.</li><li><strong>Batch Size and Sequence Length:</strong> Ensure that your batch sizes and sequence lengths are manageable to prevent memory issues, as discussed earlier.</li><li><strong>Model Saving:</strong> Regularly save model checkpoints to avoid losing progress in case of interruptions.</li></ul></blockquote>
</details>

<br/>
## Drawing Code

The code to actually write, which I dubbed `Calligrapher` was relatively simple as it creates an initial data point and supplied user texts and then queries the model based on those two inputs to what the actual handwriting data should look like. More or less, that code is here:

```python
class Calligrapher:
    def __init__(self, model_path: str, num_output_mixtures: int) -> None:
        self.model_path = model_path
        self.num_output_mixtures = num_output_mixtures
        self.model, self.loaded = load_model_if_exists(
            model_path,
            custom_objects={
                "mdn_loss": mdn_loss,
                "AttentionMechanism": AttentionMechanism,
                "AttentionRNNCell": AttentionRNNCell,
                "MixtureDensityLayer": MixtureDensityLayer,
                "DeepHandwritingSynthesisModel": DeepHandwritingSynthesisModel,
            },
        )
        if not self.loaded:
            raise ValueError(f"Model not loaded from {model_path}")

    def sample_gaussian_2d(self, mu1, mu2, s1, s2, rho):
        """
        Sample a point from a 2D Gaussian.
        """
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x, y = np.random.multivariate_normal([mu1, mu2], cov)
        return x, y

    def adjust_parameters(self, pi_logits, mu1, mu2, sigma1, sigma2, rho, bias):
        """
        Adjust the parameters for biased sampling.
        """
        sigma1_adj = np.exp(np.log(sigma1) - bias)
        sigma2_adj = np.exp(np.log(sigma2) - bias)
        pi_adj = np.exp(np.log(pi_logits) * (1 + bias))
        pi_adj /= np.sum(pi_adj, axis=-1, keepdims=True)
        return pi_adj, mu1, mu2, sigma1_adj, sigma2_adj, rho

    def encode_characters(self, lines, char_to_idx, max_length):
        """
        Encodes text lines to a format suitable for the model.
        """
        encoded_lines = np.zeros((len(lines), max_length), dtype=int)
        for i, line in enumerate(lines):
            encoded_line = [char_to_idx.get(char, 0) for char in line]
            encoded_lines[i, : len(encoded_line)] = encoded_line[:max_length]
        return encoded_lines

    def sample(
        self, lines: list[str], max_char_len=MAX_CHAR_LEN, bias=0.0, temperature=1.0
    ):
        """
        Sample handwriting sequences with optional bias and temperature.
        """

        encoded_lines = np.array([encode_ascii(line) for line in lines])
        batch_size = len(encoded_lines)
        x_in = np.zeros((batch_size, max_char_len, 3), dtype=np.float32)
        chars_seq_len = np.zeros([batch_size])
        char_seq = np.zeros((len(lines), max_char_len), dtype=int)
        for i, line in enumerate(encoded_lines):
            char_seq[i, : len(line)] = line

        # Get MDN outputs
        mdn_outputs = self.model(x_in, char_seq, chars_seq_len)
        pi_logits, mu1, mu2, sigma1, sigma2, rho, eos_logits = tf.split(
            mdn_outputs, [self.num_output_mixtures] * 6 + [1], axis=-1
        )
        pi_logits /= temperature  # Apply temperature to soften pi distribution

        if bias != 0.0:
            pi, mu1, mu2, sigma1, sigma2, rho = self.adjust_parameters(
                pi_logits.numpy(),
                mu1.numpy(),
                mu2.numpy(),
                sigma1.numpy(),
                sigma2.numpy(),
                rho.numpy(),
                bias,
            )
        else:
            pi = tf.nn.softmax(pi_logits, axis=-1).numpy()

        # Sample from the GMM
        indices = [np.random.choice(20, p=pi[i]) for i in range(pi.shape[0])]
        selected_mu1 = np.take_along_axis(
            mu1, np.expand_dims(indices, axis=-1), axis=-1
        )
        selected_mu2 = np.take_along_axis(
            mu2, np.expand_dims(indices, axis=-1), axis=-1
        )
        selected_sigma1 = np.take_along_axis(
            sigma1, np.expand_dims(indices, axis=-1), axis=-1
        )
        selected_sigma2 = np.take_along_axis(
            sigma2, np.expand_dims(indices, axis=-1), axis=-1
        )
        selected_rho = np.take_along_axis(
            rho, np.expand_dims(indices, axis=-1), axis=-1
        )

        sampled_points = [
            self.sample_gaussian_2d(
                selected_mu1[i, 0],
                selected_mu2[i, 0],
                selected_sigma1[i, 0],
                selected_sigma2[i, 0],
                selected_rho[i, 0],
            )
            for i in range(pi.shape[0])
        ]

        # Sample eos with temperature applied
        eos = tf.sigmoid(eos_logits / temperature).numpy()

        return sampled_points, eos

    def write(
        self,
        strokes: list[Tuple[np.ndarray, np.ndarray]],
        lines: list[str],
        filename: str = "output.svg",
        bias=0.0,
        temperature=1.0,
        stroke_colors=None,
        stroke_widths=None,
    ):
        """
        Draws pseudo handwriting given lines of text and visualizes it as an SVG image.
        """
        stroke_colors = stroke_colors or ["black"] * len(lines)
        stroke_widths = stroke_widths or [2] * len(lines)

        # Generate sampled points for each line of text
        strokes = self.sample(lines, bias=bias, temperature=temperature)

        # Visualization
        line_height = 60
        view_width = 1000
        view_height = line_height * (len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename, size=(view_width, view_height))
        dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))

        initial_y_offset = line_height
        for sampled_points, color, width in zip(strokes, stroke_colors, stroke_widths):
            prev_eos = 1.0
            p = "M {},{}".format(0, initial_y_offset)
            for x, y, eos in sampled_points:
                command = "M" if prev_eos == 1.0 else "L"
                p += " {} {},{}".format(command, x, -y + initial_y_offset)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap="round").fill("none")
            dwg.add(path)
            initial_y_offset += line_height

        dwg.save()


if __name__ == "__main__":
    texts = ["Better to have loved", "and lost", "than never loved at all"]
    calligrapher = Calligrapher(model_load_path, num_output_mixtures=1)
    calligrapher.write(texts, "output.svg")
```

## Visualizations

### Learning with Dummy Data

Again, we used dummy data to start with to ensure our various components were learning and converging correctly.

Here is the dummy data:

![dummy_data](/images/generative-handwriting/viz/dummy_data.png){: .center-image}

Here is just our cascade of LSTMs learning on the loop-da-loop data and predicting on a single sequence. This is not optimized or utilizing the mixture density network.

![handwriting_loop_lstm_simple](/images/generative-handwriting/viz/handwriting_loop_simplified.gif){: .center-image}

![handwriting_zig_lstm_simple](/images/generative-handwriting/viz/handwriting_zigzag_simplified.gif){: .center-image}

Here is our entire network and just sampling from the means (not showing the mixture densities) across the entire example datasets. One thing to note here if you can see how the LSTMs can still handle this type of larger contexts. Again, it pales in comparison to modern day transformer context, but still impressive.

![handwriting_loop_lstm_simple](/images/generative-handwriting/viz/loop_epoch200_mixtures5.gif){: .center-image}

![handwriting_zig_lstm_simple](/images/generative-handwriting/viz/zigzag_epoch200_mixtures5.gif){: .center-image}

# Conclusion

This - again - was an absolute bear of a project and took considerable effort and engineering.

I don't think I'll embark on a project of this nature in awhile, unless I can feel some more tangible external benefits of doing something like this. But enjoy! And feel free to pull the code and dive in yourself.

When re-reading my old draft blog post, I liked the way I ended things. So here it is:

> Finally, I want to leave with a quote from our academic advisor [Matt Zucker][mz]. When I asked him when we know that our model is good enough, he responded with the following.
>
> > "Learning never stops."

[^1]: https://www.asimovinstitute.org/neural-network-zoo/
[^2]: https://www.researchgate.net/publication/336440220/figure/fig3/AS:839177277042688@1577086862947/A-illustration-of-neural-networks-NNs-a-a-basic-NN-is-composed-of-the-input-output.jpg
[^3]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[^4]: https://miro.medium.com/v2/resize:fit:996/1*kJYirC6ewCqX1M6UiXmLHQ.gif
[^5]: https://arxiv.org/abs/1308.0850
[^6]: https://towardsdatascience.com/wp-content/uploads/2024/05/1UKuoYsGWis22cOV7KpLjVg.png

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
[JAX]:https://github.com/jax-ml/jax
[hnews]: https://news.ycombinator.com/item?id=45440431
[xlstm-ref]: https://news.ycombinator.com/item?id=45440431#:~:text=What%27s%20with%20all,crypto/DeFi%20pitch.
[xlstm]: https://arxiv.org/abs/2405.04517
[gen-sequences]: https://arxiv.org/abs/1308.0850
[pytorch-rant]: https://neel04.github.io/my-website/blog/pytorch_rant/
[stoch-grad-descent]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
[understanding-lstms]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
