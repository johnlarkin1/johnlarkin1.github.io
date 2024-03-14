---
title: 'Teaching a Computer How to Write'
layout: post
featured-gif: 
mathjax: true
categories: [Favorites, Algorithms, Development, A.I., M.L.]
summary: 
    In the boom of Generative AI, I'm throwing it back to my senior engineering
    thesis which was teaching a computer to write like a human. 
---

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Motivation](#motivation)
- [History](#history)
  - [Tom and My Engineering Thesis](#tom-and-my-engineering-thesis)
- [Acknowledgements](#acknowledgements)
- [Software](#software)
  - [Tensorflow's Programming Paradigm](#tensorflows-programming-paradigm)
  - [Tensorflow Version - How the times have changed](#tensorflow-version---how-the-times-have-changed)
  - [Tensorflow's Tensorboard](#tensorflows-tensorboard)
- [Theory](#theory)
  - [Basic Neural Network](#basic-neural-network)
    - [Hyper Parameters](#hyper-parameters)
- [Code](#code)
  - [Looking back at college code 😬](#looking-back-at-college-code-)
  - [New Code](#new-code)
- [Results](#results)
- [Visualizations](#visualizations)
- [Troubles](#troubles)
- [Conclusion](#conclusion)

# Motivation 

I never actually ended up publishing it, but I had a draft blog post written maybe right after college, about some of the work that a good friend (and absolutely incredible engineer) [Tom Wilmots][tom] and I did in college.

I'm going to pull pieces of that, but the time has changed, and I wanted to revisit some of the work we did back in college, and clean it up some, given the boom of generative AI. Also I'm a better engineer and we had a small bug in some of the modeling that I want to flush out. 

# History

[Tom] and I were very interested in the concept of teaching a computer how to write in college. There is a very famous [paper] that was published around 2013 from Canadian computer scientist [Alex Graves][ag], titled *Generating Sequences With Recurrent Neural Networks*. At [Swarthmore][swat], you have to do Engineering theses, called [E90s][e90]. 

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
> 1. Building your computational dependency graph. 
> 2. Running your dependency graph. 

## Tensorflow Version - How the times have changed 

So - another fun fact - when we were doing this in college, we were on **tensorflow version v0.11**!!! They hadn't even released a major version. Now, I'm doing this on Tensorflow **2.16.1**. So the times have definitely changed. 

![being-old](/images/generative-handwriting/being_old.jpeg){: .center-shrink }

## Tensorflow's Tensorboard

Another cool thing about [Tensorflow][tf] that should be mentioned is the ability to utilize the [Tensorboard][tensorboard]. This is a visualization suite that creates a local website where you can interactively and with a live stream visualize your dependency graph. You can do cool things like confirm that the error is actually decreasing over the epochs. 

# Theory

I am not going to dive into details as much as we did for our senior E90 thesis, but I do want to cover a couple of the building blocks. 

## Basic Neural Network

![basic-nn](https://www.researchgate.net/publication/336440220/figure/fig3/AS:839177277042688@1577086862947/A-illustration-of-neural-networks-NNs-a-a-basic-NN-is-composed-of-the-input-output.jpg)


The core structure of a neural network is the connections between all of the neurons. Each connection carries an activatino signal of varying strength. If the incoming signal to a neuron is strong enough, then the signal is permeated through the next stages of the network. 

There is a input layer that feeds the data into the hidden layer. The outputs from the hidden layer are then passed to the output layer. Every connection between nodes carries a weight determining the amount of information that gets passed through. 

### Hyper Parameters

For a basic neural network, there are generally three [hyperparameters]:

* pattern of connections between all neurons
* weights of connections between neurons 
* activation functions of the neurons 

In our project however, we focus on a specific class of neural networks called Recurrent Neural Networks (RNNs), and the more specific variation of RNNs called Long Short Term Memory networks (LSTMs). 

However, let's give a bit more context. There's two 

# Code 

## Looking back at college code 😬

Yeah, so I haven't looked at a lot of my college work recently, but damn. This was SO messy. And it's fair, I was a senior in college and didn't have industry experience, and you know we basically were treating our codebase like many scripts tied together. 

But yeah... here's a look at some of the way we had organized things: 

![being-old](/images/generative-handwriting/college_messy_structure.png){: .center-shrink }

Ouch.

## New Code

So - sadly, I basically started fresh with a new repo, and a fresh outlook, acting like I know nothing. 



# Results 

# Visualizations 

# Troubles

# Conclusion 

When re-reading my old draft blog post, I liked the way I ended things. So here it is: 

> Finally, I want to leave with a quote from our academic advisor [Matt Zucker][mz]. When I asked him when we know that our model is good enough, he responded with the following. 
>> "Learning never stops."

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
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory