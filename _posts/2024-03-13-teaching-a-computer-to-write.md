---
title: 'Teaching a Computer How to Write'
layout: post
featured-gif:
mathjax: true
categories: [Favorites, Algorithms, Development, A.I., M.L.]
summary: Let's use some LSTMs, GMMs, and MDNs to teach a computer how to handwrite. Keyboardwrite?
---

# Motivating Visualizations

This is a bit of a long one, so I figured I would include some pretty visualizations to get you all excited.

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
  - [Tensorflow's Programming Paradigm](#tensorflows-programming-paradigm)
  - [Tensorflow Version - How the times have changed](#tensorflow-version---how-the-times-have-changed)
  - [Tensorflow's Tensorboard](#tensorflows-tensorboard)
- [Neural Network Theory](#neural-network-theory)
  - [Lions, Bears, and Many Neural Networks, oh my](#lions-bears-and-many-neural-networks-oh-my)
  - [Basic Neural Network](#basic-neural-network)
    - [Hyper Parameters](#hyper-parameters)
  - [Feedforward Neural Network](#feedforward-neural-network)
    - [Backpropagation](#backpropagation)
  - [Recurrent Neural Network](#recurrent-neural-network)
  - [Long Short Term Memory Networks](#long-short-term-memory-networks)
    - [Understanding the LLM Structure](#understanding-the-llm-structure)
- [Probabilistic Model Theory + Graves Extensions](#probabilistic-model-theory--graves-extensions)
  - [Gaussian Mixture Models](#gaussian-mixture-models)
  - [Mixture Density Networks](#mixture-density-networks)
  - [Stacked LSTM and MDN](#stacked-lstm-and-mdn)
  - [Final Result](#final-result)
- [Code](#code)
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

## Tensorflow's Programming Paradigm

Tensorflow has this interesting programming paradigm, where you are more or less creating a graph. So you define `Tensor`s and then when you run your dependency graph, those things are actually translated.

I have this quote from the Tensorflow API:

> There's only two things that go into Tensorflow.
>
> 1. Building your computational dependency graph.
> 2. Running your dependency graph.

## Tensorflow Version - How the times have changed

So - another fun fact - when we were doing this in college, we were on **tensorflow version v0.11**!!! They hadn't even released a major version. Now, I'm doing this on Tensorflow **2.16.1**. So the times have definitely changed.

![being-old](/images/generative-handwriting/being_old.jpeg){: .center-shrink }

There's never enough time in the day.

<div class="markdown-alert markdown-alert-note">
  <p>Apparently, Tensorflow 2.0 helped out a lot with the computational model and the notion of eagerly executing, rather than building the graph and then having everything run at once.</p>
</div>

<!--
<div class="markdown-alert markdown-alert-tip">
  <p><strong>Tip:</strong> Optional information to help a user be more successful.</p>
</div>

<div class="markdown-alert markdown-alert-important">
  <p><strong>Important:</strong> Crucial information necessary for users to succeed.</p>
</div>

<div class="markdown-alert markdown-alert-warning">
  <p><strong>Warning:</strong> Critical content demanding immediate user attention due to potential risks.</p>
</div>

<div class="markdown-alert markdown-alert-caution">
  <p><strong>Caution:</strong> Negative potential consequences of an action.</p>
</div> -->

## Tensorflow's Tensorboard

Another cool thing about [Tensorflow][tf] that should be mentioned is the ability to utilize the [Tensorboard][tensorboard]. This is a visualization suite that creates a local website where you can interactively and with a live stream visualize your dependency graph. You can do cool things like confirm that the error is actually decreasing over the epochs.

# Neural Network Theory

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

# Probabilistic Model Theory + Graves Extensions

There are a couple of other components that Alex used in his paper. We'll explore those here.

## Gaussian Mixture Models

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

## Mixture Density Networks

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

## Stacked LSTM and MDN

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

## Final Result

Alright finally! So what do we have, and what can we do now?

We now are going to feed the output from our LSTM cascade into the GMM in order to build a probabilistic prediction model for the next stroke. The GMM will then be fed the actual next point, in order to create some idea of the deviation os that the loss can be properly minimized.

# Code

Ok so that's all well and good and some fun math and neural network construction, but the meat of this project is about what we're actually building with this theory. So let's get to some of the code.

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

🎉 Voila - not too painful. Here's the code, it's relatively straightforward.

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
