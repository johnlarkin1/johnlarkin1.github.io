---
title: "Multi Armed Bandit"
layout: post
featured-img: multi-armed-bandit
mathjax: true
python-interactive: false
categories: [Development, A.I]
summary: Exploring a super interesting problem that comes up in probability and reinforcement learning.
---

# Context

Recently, I responded to some recruiters and fielded a couple of interviews. 

I generally abhor interviewing. There are parts I absolutely love - meeting new people, learning about new technical challenges, studying up on businesses or industries - but there are also parts I *abhor*. Getting grilled on usage of the Web Speech API (man oh man was I in the wrong interview) or how to [decode a string][fuck-leetcode] in 2026 does feel... a bit perplexing. I'll rant about it on Substack at some point in time. 

However! I do genuinely enjoy take homes (as exemplified by [Book Brain][book-brain]. Despite often it being a bigger time constraint, and more of a commitment. 

This blog post is going to go over a concept and problem that (embarrassingly enough), I hadn't yet seen before the take home. For more context, I ultimately didn't carve enough time to do this takehome the way I wanted... As a result, I accepted another offer in the same timeframe, and withdrew from the process. It's unfortunate too because I do genuinely believe the company will be a $10BN company in no time, and the engineering seems absolutely fascinating.

While I ultimately withdrew from this interviewing cycle, and sent them only my thoughts on the problem, this blog post is going to talk about a take home question I received from that company. I'm anonymizing the company to keep the sanctity of their interview process. 

The company restricted Ai usage during the take, so I did a ton of research / youtube videos. However, for this blog post, some details of implementation will be left to Claude. The repo has documentation and detail included various transcripts between Claude and I. So let's begin with the problem.

# Problem Setup

The problem focuses around the [multi-armed bandit][mab] problem, which I'll commonly abbreviate as MAB.

## Traditional (Stochastic) Multi-Armed Bandit Problem (MAB)

The [traditional multi-armed bandit][trad-mab] is pretty well encapsulated by a hypothetical situation. I'll give you the long / fun version, and then I'll give you an abbreviated Wikipedia version.

---

Imagine, you wake up. 

![Life is full]({{ site.baseurl }}/assets/svg/multi-armed-bandit/life-is-full.svg){: .center-super-medium-shrink .lightbox-image}

You live in a beautiful city (let's say Cincinnati).

![Cincinnati Skyline](https://assets.simpleviewinc.com/sv-cincy/image/upload/c_fill,h_840,q_75,w_1200/v1/cms_resources/clients/cincy/msvachphotography_Instagram_1244_ig_17864840302845987_89e72393-d2a6-4837-bb9c-865845b1366b.jpg){: .center-shrink .lightbox-image}


<div class="image-caption">Kudos to @msvachphotography for the shot from Mt. Echo Park</div>
<br>

But then you realize you have too much money in your pockets. You decide to gamble (i discourage this, especially after seeing how the sausage is made).

So you hit the casino!

![Life is full]({{ site.baseurl }}/assets/svg/multi-armed-bandit/too-much-mula.svg){: .center-shrink .lightbox-image}

However, because it's Cincinnati, this is a very nice casino. You actually have a chance to win. However, they only have single-armed bandits - commonly known as slot machines! These are unique slot machines, and their underlying probably distributions become more apparent over time.

Despite having too much money in your pockets, you love winning, so you do want to win. Your problem therefore is to figure out the optimal strategy for which machines to play, when to play those machines, how many times to play them, and when you need to switch.

---

Wikipedia more blandly (but also more succinctly) puts this as:

> More generally, it is a problem in which a decision maker iteratively selects one of multiple fixed choices (i.e., arms or actions) when the properties of each choice are only partially known at the time of allocation, and may become better understood as time passes. A fundamental aspect of bandit problems is that choosing an arm does not affect the properties of the arm or other arms.[4]


## Multi-Armed Bandit Variants

The situation I described above is really the stochastic MAB. There's a finite set of arms, and the reward distribution is unknown. As I learned throughout this process, there are many variants and generalizations of this problem. Specifically, these are _generalizations_ where the MAB is extended by adding some information or structure to the problem. Namely:

* [adversarial bandits][adversarial-bandits]
  * this is probably my favorite variant. the notion is that you have an adversary that is trying to **maximize** your regret, while you're trying to minimize your regret. so they're basically trying to trick or con your algorithm.
  * if you're asking yourself (like I did), ok well then why doesn't the adversary just assign $r_{a,t} = 0$ as the reward function for all arms $a$ at time $t$, well... you shouldn't really think about it in terms of reward. Reward is relative. We instead want to think about it in terms of *regret* which I'll talk more about later. There are two subvariants ([oblivious adversary][oblivious-adversary] and [adaptive adversary][adaptive-adversary]), but we're not going to discuss those - although a very interesting extension is the [EXP3][exp3] algorithm.
* [contextual bandits][contextual-bandits]
  * the notion here is that instead of learning $E[r \mid a]$ where again $r$ is the reward and $a$ is the arm you pick, you're learning $E[r \mid x, a]$ where $x$ is some additional bit of context at time $t$ that you're exposed to.
* [dueling bandits][dueling-bandits]
  * an interesting variant where instead of being exposed to the reward, your information is limited to just picking two bandits and only knowing which one is better comparatively... but again it's stochastic. So you can inquire about the same two arms and it's very feasible that you'll get different results for the comparison. The whole notion is that you're building up this preference matrix. Seems like an incredibly difficult problem.

## Take-home Twist Multi-Armed Bandit 

The takehome I received had an interesting twist on this. The change is that: **you are only penalized for a failing server request after $k$ tries.** So you are still trying to maximize your "score" (i.e. reward) but you're also given some leeway.

After further deep research with Claude / ChatGPT, I believe the problem is best framed as a **Bandit with Knapsack** situation.

### Bandit with Knapsack (BwK) Variant

The original paper is from [Ashwinkumar Badanidiyuru][ash], [Robert Kleinberg][klein], and [Aleksandrs Slivkins][sliv]. People who I'd love to be an iota as smart as. You can see the paper [here][bwk]. It's a 55 page paper, and I'd be lying if I said I read past the 

The idea is relatively simple though. Your arms now have resources associated with them that they consume. I honestly think it's easier to draw it out mathematically and reference the actual paper (also shoutout to [alphaxiv], it's got most of the normal [arvix] features, just with some ai native question answering and highlighting which has been nice). 

### BwK Formal Declaration



# Theory


# Things to Talk About

* Hoeffding's Inequality...


[comment]: <> (Bibliography)
[fuck-leetcode]: https://leetcode.com/problems/decode-string/description/
[book-brain]: {{ site.baseurl }}/2024/book-brain 
[mab]: https://en.wikipedia.org/wiki/Multi-armed_bandit
[cincy-skyline]: https://as2.ftcdn.net/v2/jpg/04/00/44/39/1000_F_400443989_4nVyEAyjLLbWGuJgKVfQ08RlEOG72EDx.jpg
[adversarial-bandits]: https://en.wikipedia.org/wiki/Multi-armed_bandit#Adversarial_bandit
[contextual-bandits]: https://towardsdatascience.com/an-overview-of-contextual-bandits-53ac3aa45034/
[dueling-bandits]: https://doogkong.github.io/2017/slides/Yue.pdf
[oblivious-adversary]: https://www.cs.cornell.edu/~rdk/papers/anytime.pdf
[adaptive-adversary]: https://ui.adsabs.harvard.edu/abs/2006cs........2053D/abstract
[exp3]: https://en.wikipedia.org/wiki/Multi-armed_bandit#:~:text=%5Bedit%5D-,Exp3,-%5Bedit%5D
[ash]: https://sites.google.com/site/ashwinkumarbv/home
[klein]: https://www.cs.cornell.edu/~rdk/
[sliv]: https://scholar.google.com/citations?user=f2x233wAAAAJ&hl=en
[bwk]: https://www.alphaxiv.org/abs/1305.2545
[alphaxiv]: https://www.alphaxiv.org/
[arxiv]: https://arxiv.org/
