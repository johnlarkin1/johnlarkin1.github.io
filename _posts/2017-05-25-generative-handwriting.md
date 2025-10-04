---
title: GMMs, MDNs, LSTMs, and Generative Handwriting
layout: post
featured-img: network
categories: [Dev]
---

Coming soon... this one has taken me forever to iron out a bug and formalize... for reasons I'll explain in the actual blog post. I'll post it soon.

<!--
Ah where to even start? How about with explaining the acronyms above? Click [here](#theory).

This blog post will summarize work done for [Tom Wilmot][tom] and my final engineering project. Before I start, this was a project that Tom and I did during our senior spring semester at Swat. We presented on May 2nd, 2017. The day I'm writing this is September 23, 2017. That's several months. A lot has happened which I'll touch on in a later post.

There's a bunch of places that I could start. But I think I'm going to go in the following order.

- [Project goals](#pg)
- [Acknowledgements](#acknowledge)
- [Background](#bg)
- [Data and Software](#data)
- [Theory](#theory)
- [*Brief* overlay of model](#model)
- [Visualizations](#viz)
- [Troubles](#trub)
- [Read our paper and](#paper) [link][TWJLpaper]
- [Conclusion](#end)

So... here we go.

Project Goals {#pg}
=============

This project was / is really cool. It's a great example of deep learning. The idea was to be able to teach a computer how to write realistically like a human. **In essence, learn how a human writes.** Which is a sweet freaking concept!! This is computationally intensive for a multitude of reasons. It requires your model to learn long term dependencies. It's an intense prediction problem because you can think about all of the degrees of freedom at every spot as to where the next pen point is going to be.

Acknowledgements {#acknowledge}
================

There are so many different acknowledgements that need to be made. I'm going to attempt to list them all.

- **[Tom Wilmots][tom]** - First and foremost, *I did not do this project by myself.* I had tremendous help from one of my best friends from [Swarthmore][swat] - [Tom Wilmots][tom]. He was an Engineering and Economics double major from Swarthmore and is currently at the [London School of Economics][lse].
- **[Matt Zucker][mz]** - Absolutely no surprise here. One of my role models and constant inspirations, Matt was kind enough to be Tom and my academic advisor for this final engineering project. He's an outstanding professor at Swarthmore College and the institution is beyond fortunate to have him.
- **[Alex Graves][ag]** - Another genius that both Tom and I had the pleasure of working with. *He actually responded to our emails! Do you know how cool that is? That a professional (who was a professor at the University of Toronto, see [here][gravesToronto]). He is the author of [this paper][paper], which Matt found for us and pretty much was the basis of our project. Alex is at Google now and is crushing it. He's also the creator of the [Neural Turing Machine][ntm], which peaked my interest after having taken [Theory of Computation][toc], with my other fantastic professor [Lila Fontes][lila] and learning about [Turing machines][turingmachines].
- **[David Ha][dha]** - Yet another genius who we had the priviledge of corresponding with. Check out his blog [here][otoro]. It's beautiful. He also is very prolific on [ArXiv][arxiv] which is always cool to see.
- My family - For putting up with my incessant rambling about pitfalls in the project. Spitballing ideas and etc. Just generally putting up with me.

Background {#bg}
==========
Tom and I didn't just decide to take three classes and dedicate a semester of work into this project for fun. Swarthmore requires that you accomplish a final engineering project, which is called your E90 after the class you register under. This was ours.

An important note. *Tom and I are far from experts in the subject matter*. However, we did a large amount of research trying to build up our knowledge. We went from no official A.I. and machine learning experience to feeling comfortable implmenting neural nets quickly with Tensorflow.


Data and Software {#data}
==================

Theory {#theory}
======

Let's talk about some theory. Before we get into our model, we need to first understand the different building blocks that go into it. Think about this kind of like a lego set. We don't just build the entire model at once. We build smaller parts, and then connect it all together smoothly (or attempt to) at the end.




Model {#model}
======

Visualizations {#viz}
==============

Troubles {#trub}
=========

### [Tensorflow][tf]
Wowweeeee where to even start. Tensorflow is low-key annoying at first. It's really a different type of programming because now you're really creating a dependency graph. Oh you like debugging? Sucks. Everything's a 'Tensor' until you actually run your dependency graph. As Google's Tensorflow API likes to say:

> There's only two things that go into Tensorflow.
> 1. Building your computational dependency graph.
> 2. Running your dependency graph.

Wooo! So easy. Jk. I really didn't think so. *Note, we used Tensorflow v0.11*. If you use any version above v0.11, I'm 99% sure that it won't work. Another thing to learn about is [virtual environments][virtualenvironments], which Matt taught us about and encouraged us to read up on.

One other cool thing about [Tensorflow][tf] that should be mentioned is the ability to utilize the [Tensorboard][tensorboard]. This is a visualization suite that creates a local website where you can interactively and with a live stream visualize your dependency graph. You can do cool things like confirm that the error is actually decreasing over the epochs.

### Our Model
Our model isn't the best one out there. I'm going to be really up front and blunt about it. We know about [David Ha's website][otoro] and his implementation accomplishes this with much less code and is probably more readable.

That being said, he's also unequivocally had more experience with machine learning, and is probably just a more intelligent person as a baseline.

### TIME
Computers are fast! BUT, they really slow down when there's [backpropagation][backprop] going on essentially everywhere. We had a *bunch* of different variables that were trainable and our dependency graph was incredibly large. I cannot say it enough. This part was incredibly annoying. If we would have used our full datasource on a CPU and trained it for 200 epochs it would have taken well over 75 days.


Paper {#paper}
=====

*do paper shit up here*

Conclusion {#end}
==========
Finally, I want to leave with a quote from our academic advisor Matt Zucker. When I asked him when we know that our model is good enough, he responded with the following.

> "Learning never stops."

Thanks for your time as always!

[comment]: <> (Bibliography)
[code]: https://github.com/johnlarkin1/sudoku-solver
[TWJLpaper]: https://google.com
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

-->
