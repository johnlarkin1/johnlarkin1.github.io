---
title: Advent of Code - 2024
layout: post
featured-img: advent-of-code-2024
mathjax: true
major-category: tech
categories: [Advent of Code, Dev]
summary: Slogging through Advent of Code 2024
---

<!--
# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Background](#background)
- [Goal](#goal)
- [Questions](#questions)
  - [Why am I doing this?](#why-am-i-doing-this)
  - [Aren't LLMs just going to replace everything?](#arent-llms-just-going-to-replace-everything)
  - [Ok fine, but why this year?](#ok-fine-but-why-this-year)
  - [Did you accomplish all your goals? Any excuses?](#did-you-accomplish-all-your-goals-any-excuses)
  - [Beh, sure - well did you solve them all without any help?](#beh-sure---well-did-you-solve-them-all-without-any-help)
  - [Oh so you're a bad engineer?](#oh-so-youre-a-bad-engineer)
  - [What was the community like?](#what-was-the-community-like)
  - [Ok well... what did you learn?](#ok-well-what-did-you-learn)
    - [Z3 for Python](#z3-for-python)
    - [Bron-Kerbosh Algorithm](#bron-kerbosh-algorithm)
    - [Adder Gate Implementation](#adder-gate-implementation)
  - [What were the hardest days?](#what-were-the-hardest-days)
  - [Ok this is sounding better - are you going to do it again?](#ok-this-is-sounding-better---are-you-going-to-do-it-again)
  - [Well... what are you going to do now?](#well-what-are-you-going-to-do-now)
- [Code](#code)
  - [Solutions](#solutions)
  - [Closing Thoughts](#closing-thoughts)
-->

# Introduction

This year (2024) I decided to do the [Advent of Code][aoc] put on by [Eric Wastl][eric] who's a bit of a legend in my opinion now. I'll cover some of the hardest problems I faced, my ramblings / thoughts, and anything that I learned. But here's the final result:

![2024](/images/advent-of-code/2024/complete_border.png){: .center-image}

<div class="image-caption">And yes, I did adjust the padding-left for the screenshot</div>

# Background

What is [Advent of Code?][aoc-about] You can find more there (<-), but the idea is it's a somewhat competitive set of programming challenges. It's a great way to challenge yourself, potentially learn some new things, explore a new language, get involved with a great community, and see some of the very impressive solutions that people put out.

# Goal

My goals were:

- ✅ solve all solutions
  - Note, for day17 and day24 I did have to consult the subreddit and eventually for day24 I had to heavily take inspiration from a post. I'll discuss those in more detail
- 🟡 solve all of these without using ChatGPT or Claude
  - I did end up using LLMs for debugging and to generate some of the helpful input utils, in addition one of the problems required GraphViz, so I used LLMs to handle a lot of that interaction with the third party library
  - For some of the harder problems, I _tried_ to use LLMs but honestly with pretty disappointing results. I'll talk about that more [later](#arent-llms-just-going-to-replace-everything).
- ❌ implement all solutions in both Python and Rust
  - I've been trying to learn Rust for awhile now and have had some exposure at Dropbox, but really want to get more familiar with Rust and figured this could be a good chance
  - That being said, I only got to about [day09] in Rust and the rest I just took with Python

# Questions

I'm going to frame this whole article a bit differently than normal, and basically just have the sections be various questions that either I got asked from others, I asked myself, or I ruminated on during this process.

## Why am I doing this?

Yeah, why _did_ I do this? I asked myself that question a LOT. Especially during some of the more challenging problems (which i'll discuss later).

To be honest, I'm not sure. I haven't really come up with a longer post about the state of Ai ([i'm going to try spelling it like this because of this article][why-ai-spelling]) and LLMs, but I think generally a bit of an Ai doomer.

**So one of the reasons is to get back to my roots.**

Over the past probably 2 years since ChatGPT and the whole LLM craze, the way I work and operate has changed **drastically**. And I think that's good - I can learn quicker, explore more with a customized approach, and build faster (although more efficiently I think there are implications of LLMs there). However... That's not really what attracted me to computer engineering / science. I miss implementing some of the logic and solving these hard problems.

**Another one of the reasons I wanted to do this is to get more Rust exposure.**

Rust is rapidly becoming the developer's language (see [here][rust-hype] and [here][rust-hype-2]). So I figured I should ride that bandwagon some.

**And finally, just to solve hard problems and see if I learn anything new**.

I did, and I'll definitely discuss some of those below.

## Aren't LLMs just going to replace everything?

Yeah so that was pretty interesting. You'd be surprised.

On [day17] and [day24], I definitely tried to use LLMs and they all kind of dropped the ball. They couldn't even really come up with logic. And yes, even [`o1`][o1].

The results actually surprised me. Perhaps because these were pretty novel problems (again, huge shoutout to [Eric][eric]).

I wasn't sure if I was alone in this approach, but it seems like others can confirm perhaps a lackluster / disappointing performance from LLMs. And yes, I know that o3 is going to out perform like 99.9% of developers in competive programming and all of that, so who knows. But I will say actual use cases seemed lackluster at the moment. And I'm sure all the die-hard defenders are going to defend it and sure - i'm sure that some of this could have been prevented or solved with better prompting but that was not the result that I would have hoped for.

**One other thing that I thought was pretty interesting was a fellow solver posted about his findings on LLMs with Advent of Code [here][jerpint-llm]**.

## Ok fine, but why this year?

Seemed a good a year as any. Perhaps it was the fact that I always felt like if I was going to start, then I wanted to finish, and I feel like I had enough time this year or maybe enough breathing room with work that this was possible.

## Did you accomplish all your goals? Any excuses?

I did not accomplish all my goals. This became more of an algorithmic exercise than a "learn-a-new-programming-language-by-doing" exercise - which if anything is just a testament to some of the great work that [Eric][eric] has put together.

I also finished a bit late, but I had a trip with my girlfriend and a family trip so two weeks of December weren't even really work or fun work (like aoc) focused. As a result, I was a bit late. Furthermore, I burned out a bit towards the end (see [day24]).

## Beh, sure - well did you solve them all without any help?

Not really - especially once we got into the `day2*` area, the problems were pretty fascinating and I checked the subreddit on numerous occasions. I'll get into the community in a bit but that was definitely a highlight.

## Oh so you're a bad engineer?

Hm maybe? I'd say I'm not going to be competing in any programming competitions soon.

## What was the community like?

Ok yes, so this is an **awesome** community and I genuinely appreciate and have a ton of respect for the contributors, and lots of the solutions out there.

My personal highlights that I saw?

- [this absolutely beautiful day24.py code][day24-fav] that i spent probably an hour understanding how it works. you can compare against my code, but this is **so** efficient, so minimal, and so thoughtful. It's very impressive and again, not something an LLM would produce which I appreciate even more. Credit to [Guahao](https://github.com/guohao)

- [this novel approach to day23 utilizing DWave's quantum computer][day23-fav]

<blockquote class="reddit-embed-bq" style="height:316px" data-embed-height="316"><a href="https://www.reddit.com/r/adventofcode/comments/1hkyc1y/2024_day_23_part_2python_solved_using_a_quantum/">[2024 Day 23 (Part 2)][Python] Solved using a Quantum Computer!</a><br> by<a href="https://www.reddit.com/user/Few-Example3992/">u/Few-Example3992</a> in<a href="https://www.reddit.com/r/adventofcode/">adventofcode</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>

<br>I'm going to do another writeup about the state of quantum soon, but such a beautiful and cool application, and a great writeup of what they did. This is **not** my code above or my post to be clear.

I think those two really take the cake. I'm not a big poster on Reddit, but maybe next year, I'll get a bit more involved.

## Ok well... what did you learn?

Actually quite a bit! Even more so than I was expecting.

### Z3 for Python

Z3 is theorem prover that Microsoft put out awhile ago. The code is found [here][z3], but I think [this page][z3-basic] is a bit more helpful.

What this means in practicality is that if you can model your problem as an equation, and there is a solution, then you can use this library to help you solve it in an optimal and efficient way.

This library came in handy on [Day17][day17] for me personally.

However, I think it's fun to look at some of the examples that can be provided.

Here's one from the [basic guide][z3-basic-guide-example]:

> Ben Rushin is waiting at a stoplight. When it finally turns green, Ben accelerated from rest at a rate of a 6.00 m/s2 for a time of 4.10 seconds. Determine the displacement of Ben's car during this time period.

```python
from z3 import Reals, set_option, solve

d, a, t, v_i, v_f = Reals("d a t v__i v__f")

equations = [
    d == v_i * t + (a * t**2) / 2,
    v_f == v_i + a * t,
]

# Given v_i, t and a, find d
problem = [v_i == 0, t == 4.10, a == 6]

solve(equations + problem)

# Display rationals in decimal notation
set_option(rational_to_decimal=True)

solve(equations + problem)
```

```bash
╰─➤  python z3_example_1.py
[t = 41/10, v__i = 0, v__f = 123/5, d = 5043/100, a = 6]
[t = 4.1, v__i = 0, v__f = 24.6, d = 50.43, a = 6]
```

Here's another where it's non-linear.

```python
from z3 import Real, set_option, solve

x = Real("x")
y = Real("y")
solve(x**2 + y**2 == 3, x**3 == 2)

set_option(precision=30)
print("Solving, and displaying result with 30 decimal places")
solve(x**2 + y**2 == 3, x**3 == 2)
```

```bash
╰─➤  python z3_example_2.py
[y = -1.1885280594?, x = 1.2599210498?]
Solving, and displaying result with 30 decimal places
[y = -1.188528059421316533710369365015?,
 x = 1.259921049894873164767210607278?]
```

Here's a harder fun one that I asked ChatGPT for a more complex and hard problem

> **Problem:**  
> A ball is launched upward with an initial velocity of $v_0 = 20 \, \text{m/s}$ from a height of $h_0 = 10 \, \text{m}$ above the ground. The motion is affected by air resistance, modeled as a force proportional to the velocity ($F_r = -k \cdot v$), where $k = 0.1$ is the air resistance coefficient. Assume acceleration due to gravity is $g = 9.8 \, \text{m/s}^2$.
>
> Using the following equations, determine:
>
> 1. The time of flight $t$,
> 2. The maximum height $h_{\text{max}}$,
> 3. The velocity when the ball hits the ground $v_f$.
>
> **Equations**
>
> 1. The height of the ball as a function of time:
>    $$
>    h(t) = h_0 + v_0 \cdot t - \frac{1}{2} \cdot g \cdot t^2 - k \cdot t^2
>    $$
> 2. The velocity of the ball as a function of time:
>    $$
>    v(t) = v_0 - g \cdot t - k \cdot t
>    $$
> 3. At the time of impact:
>    $$
>    h(t) = 0
>    $$

```python
from z3 import Real, Solver, sat

# Declare variables
t = Real("t")  # Time of flight
h_max = Real("h_max")  # Maximum height
v_f = Real("v_f")  # Final velocity
g = Real("g")  # Acceleration due to gravity
k = Real("k")  # Air resistance coefficient
h_0 = Real("h_0")  # Initial height
v_0 = Real("v_0")  # Initial velocity

# Equations
equations = [
    # Height as a function of time
    h_0 + v_0 * t - (1 / 2) * g * t**2 - k * t**2 == 0,  # Impact at h = 0
    # Maximum height
    h_max
    == h_0
    + (v_0**2) / (2 * g),  # Max height (ignoring air resistance here for simplicity)
    # Final velocity at impact
    v_f == v_0 - g * t - k * t,
]

# Given values
problem = [
    h_0 == 10,  # Initial height in meters
    v_0 == 20,  # Initial velocity in m/s
    g == 9.8,  # Gravitational acceleration in m/s^2
    k == 0.1,  # Air resistance coefficient
]

# Solve the system
solver = Solver()
for eq in equations + problem:
    solver.add(eq)

if solver.check() == sat:
    model = solver.model()
    print(f"Time of flight (t): {model[t]} seconds")
    print(f"Maximum height (h_max): {model[h_max]} meters")
    print(f"Final velocity (v_f): {model[v_f]} m/s")
else:
    print("No solution found.")
```

Solution:

```bash
╰─➤  python z3_example_3.py
Time of flight (t): 4.4494897427? seconds
Maximum height (h_max): 1490/49 meters
Final velocity (v_f): -24.0499484535? m/s
```

### Bron-Kerbosh Algorithm

This [algorithm][bron-kerbosh] is a famous algorithm for finding the maximum `clique` in an undirected graph.

This came into play on [day23][day23-specific] when we were solving the problem related to the LAN party, and we had to find the largest group of connected computers.

I wasn't familiar with this algorithm before AoC but it's a beautiful algorithm and I want to give it enough time to walk through it with an example.

Here's a visualization courtesy of [this article][bron-kerbosh-article]:

![bron-kerbosh-algo](https://miro.medium.com/v2/resize:fit:1132/format:webp/1*m2UXRqHpxfcKCcCPvwzg3w.gif){: .center-image }

<div class="image-caption">Courtesy of David Pynes from <a href="https://towardsdatascience.com/graphs-paths-bron-kerbosch-maximal-cliques-e6cab843bc2c">here</a></div>

This is an efficient way to find the maximum clique or grouping of entirely interconnected nodes in a graph. Here was my Python implementation:

```python
def bron_kerbosch(graph: NetworkType, R: set[str], P: set[str], X: set[str], cliques: list[set[str]]) -> None:
    """
    https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    """
    if not P and not X:
        # Found a maximal clique
        cliques.append(R)
        return
    for node in list(P):
        neighbors = set(graph[node])
        bron_kerbosch(graph, R.union({node}), P.intersection(neighbors), X.intersection(neighbors), cliques)
        P.remove(node)
        X.add(node)


def largest_clique(graph: NetworkType) -> set[str]:
    cliques = []
    bron_kerbosch(graph, set(), set(graph.keys()), set(), cliques)
    return max(cliques, key=len)
```

### Adder Gate Implementation

Day24 was a beast. Basically broke my brain. I am very appreciative of my good friend [Sebastian][seb] who also attacked it (in a better way might I add) and solved it but also burned a lot of time on it.

The problem started off simply enough in Part 1. It was basically determining the final output through a series of connected gates in a graph. [I used a topological sort][day24] which was an old coding interview problem we would give at [Ab Initio][ab-initio].

I was pleased with my part 1, but then yeah part 2 basically came to the idea of this is meant to be a [Ripple Carry Adder][ripple-carry-adder].

I probably should know more about this from my electrical circuit design courses in college, but man had it been a minute.

I saw people on the reddit thread talking about how to visualize it and then once visualized they could infer the gates to switch. So I started with some visualization(s):

<iframe src="https://srv2.zoomable.ca/viewer.php?i=img03c0f0ab0321996c_day24_gate_network_pure" width="100%" height="700px" frameborder="0" style="border:0" allowfullscreen></iframe>
<div class="image-caption">Made using Zoomable</div>

Note, I used Claude to help me with the PyGraphViz work.

Here's another variation:

<iframe src="https://zoomhub.net/xwB8g" width="100%" height="700px" frameborder="0" style="border:0" allowfullscreen></iframe>
<div class="image-caption">Made using Zoomhub (seemed like higher quality resolution handling for the larger image)</div>

I still wasn't totally able to solve it just by looking at these nodes. I knew the problem areas but I couldn't deduce which ones to flip just by looking at them or even by spending a couple minutes with pen and paper.

This post was the key for me, and unlocked the logic I needed to detect 3 out of the 4 incorrectly swapped gates.

<blockquote class="reddit-embed-bq" style="height:316px" data-embed-height="316"><a href="https://www.reddit.com/r/adventofcode/comments/1hla5ql/2024_day_24_part_2_a_guide_on_the_idea_behind_the/">[2024 Day 24 Part 2] A guide on the idea behind the solution</a><br> by<a href="https://www.reddit.com/user/LxsterGames/">u/LxsterGames</a> in<a href="https://www.reddit.com/r/adventofcode/">adventofcode</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>

<br/>
The final one I also didn't quite do perfectly. The post above has some more details about how to determine exactly the wrong gate (which I got in principle) but I didn't spend enough time coding it up. I just tried all combinations, collected the ones that resulted in the right number, and then tried a range of different test cases for input and made sure it was functioning like a full adder gate.

So... a little bit gritty there but it got the job done.

## What were the hardest days?

In my opinion, [Day 24][day24] with the Ripple Carry Adder was pretty obscenely difficult. I would not have been able to solve it most likely without the subreddit and some key insights there (or just more time than I was able to spend on that).

The other very hard day was [Day 21][day21], which I really wasn't able to solve part 2 without [this insightful article][day21-artcile]. But man oh man what a great problem that was.

## Ok this is sounding better - are you going to do it again?

Honestly? Probably not. Again, I love the problem solving and learning, but... it was a lot of time. And in a world where it's so hard to be present, I don't need to be trying to do or think more about these problems as I'm on a vacation with my girlfriend.

But who knows, I say that now, and maybe it's like running a marathon.

## Well... what are you going to do now?

More specifically, my girlfriend asked me something along the lines, "are you sad that this is done and no one really cares?" and I said I'm sad, but not because no one cares but sad that I didn't do a better job. I suppose that's part of it.

And as for what's next, I have many many other blog posts and side projects that I need to work on as well as my full time job, so I am not hurting for next steps.

# Code

## Solutions

Here are all my solutions:

- [Day 01][day01]
- [Day 02][day02]
- [Day 03][day03]
- [Day 04][day04]
- [Day 05][day05]
- [Day 06][day06]
- [Day 07][day07]
- [Day 08][day08]
- [Day 09][day09]
- [Day 10][day10]
- [Day 11][day11]
- [Day 12][day12]
- [Day 13][day13]
- [Day 14][day14]
- [Day 15][day15]
- [Day 16][day16]
- [Day 17][day17]
- [Day 18][day18]
- [Day 19][day19]
- [Day 20][day20]
- [Day 21][day21]
- [Day 22][day22]
- [Day 23][day23]
- [Day 24][day24]
- [Day 25][day25]

## Closing Thoughts

Again, my solutions are not optimized or polished. I'd love to spend more time on this but I estimate that I've already spent probably 30 hours on this and it's way over my allocated budget (or what I was expecting). Feel free to leave comments or leave my any suggestions per usual.

Until next time!

[comment]: <> (Bibliography)
[aoc]: https://adventofcode.com/2024
[aoc-about]: https://adventofcode.com/2024/about
[jerpint-llm]: https://www.jerpint.io/blog/advent-of-code-llms/
[johnlarkin-aoc-2024-code]: https://github.com/johnlarkin1/advent-of-code
[why-ai-spelling]:https://open.substack.com/pub/stephenfry/p/ai-a-means-to-an-end-or-a-means-to?selection=1e311a57-6e10-4dc2-a175-de328121f94d&utm_campaign=post-share-selection&utm_medium=web
[rust-hype]: https://www.ishir.com/blog/114521/is-rust-programming-language-worth-the-hype-is-the-hype-a-bust-or-a-boom.htm
[rust-hype-2]: https://www.infoworld.com/article/2514539/rust-leaps-forward-in-language-popularity-index.html
[day01]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day01.py
[day02]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day02.py
[day03]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day03.py
[day04]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day04.py
[day05]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day05.py
[day06]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day06.py
[day07]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day07.py
[day08]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day08.py
[day09]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day09.py
[day10]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day10.py
[day11]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day11.py
[day12]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day12.py
[day13]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day13.py
[day14]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day14.py
[day15]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day15.py
[day16]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day16.py
[day17]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day17.py
[day18]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day18.py
[day19]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day19.py
[day20]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day20.py
[day21]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day21.py
[day22]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day22.py
[day23]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day23.py
[day24]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day24.py
[day25]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day25.py
[o1]: https://openai.com/o1/
[eric]: https://was.tl/
[day24-fav]: https://github.com/guohao/advent-of-code/blob/main/2024/24.py#L41-L85
[day23-fav]: https://www.reddit.com/r/adventofcode/comments/1hkyc1y/2024_day_23_part_2python_solved_using_a_quantum/
[z3]: https://github.com/Z3Prover/z3
[z3-basic]: https://ericpony.github.io/z3py-tutorial/guide-examples.htm
[z3-basic-guide-example]: https://ericpony.github.io/z3py-tutorial/guide-examples.htm#:~:text=Ben%20Rushin%20is%20waiting%20at%20a%20stoplight.%20When%20it%20finally%20turns%20green%2C%20Ben%20accelerated%20from%20rest%20at%20a%20rate%20of%20a%206.00%20m/s2%20for%20a%20time%20of%204.10%20seconds.%20Determine%20the%20displacement%20of%20Ben%27s%20car%20during%20this%20time%20period.
[bron-kerbosh]: https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
[day23-specific]: https://github.com/johnlarkin1/advent-of-code/blob/main/2024/python/day23.py#L77
[bron-kerbosh-article]: https://towardsdatascience.com/graphs-paths-bron-kerbosch-maximal-cliques-e6cab843bc2c
[seb]:https://www.linkedin.com/in/sebastian-hoar-a71a5b112
[ab-initio]: https://www.abinitio.com/en/
[ripple-carry-adder]: https://www.sciencedirect.com/topics/computer-science/ripple-carry-adder
[day21-article]: https://observablehq.com/@jwolondon/advent-of-code-2024-day-21
