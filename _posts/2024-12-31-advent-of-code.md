---
title: Advent of Code - 2024
layout: post
featured-img: advent-of-code-2024
major-category: tech
categories: [Advent of Code, Development]
summary: Slogging through Advent of Code 2024
---

- [Introduction](#introduction)
- [Background](#background)
- [Goal](#goal)
- [Discussion](#discussion)
  - [Questions](#questions)
    - [Why am I doing this?](#why-am-i-doing-this)
    - [Aren't LLMs just going to replace everything?](#arent-llms-just-going-to-replace-everything)
    - [Ok fine, but why this year?](#ok-fine-but-why-this-year)
    - [Did you accomplish all your goals? Any excuses?](#did-you-accomplish-all-your-goals-any-excuses)
    - [Beh, sure - well did you solve them all without any help?](#beh-sure---well-did-you-solve-them-all-without-any-help)
    - [Oh so you're a bad engineer?](#oh-so-youre-a-bad-engineer)
    - [What was the community like?](#what-was-the-community-like)
    - [Ok well... what did you learn?](#ok-well-what-did-you-learn)
    - [Ok this is sounding better - are you going to do it again?](#ok-this-is-sounding-better---are-you-going-to-do-it-again)
    - [Well... what are you going to do now?](#well-what-are-you-going-to-do-now)
  - [Hardest Problems](#hardest-problems)

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

# Discussion

I'm going to start this section off with some initial questions. These are questions either my friends asked me, my girlfriend asked me, I asked myself, or I thought about quite a lot.

## Questions

### Why am I doing this?

Yeah, why _did_ I do this? I asked myself that question a LOT. Especially during some of the more challenging problems (which i'll discuss later).

To be honest, I'm not sure. I haven't really come up with a longer post about the state of Ai ([i'm going to try spelling it like this because of this article][why-ai-spelling]) and LLMs, but I think generally a bit of an Ai doomer.

**So one of the reasons is to get back to my roots. **

Over the past probably 2 years since ChatGPT and the whole LLM craze, the way I work and operate has changed **drastically**. And I think that's good - I can learn quicker, explore more with a customized approach, and build faster (although more efficiently I think there are implications of LLMs there). However... That's not really what attracted me to computer engineering / science. I miss implementing some of the logic and solving these hard problems.

**Another one of the reasons I wanted to do this is to get more Rust exposure.** Rust is rapidly becoming the developer's language (see [here][rust-hype] and [here][rust-hype-2]). So I figured I should ride that bandwagon some.

**And finally, just to solve hard problems and see if I learn anything new (which I did).**

### Aren't LLMs just going to replace everything?

Yeah so that was pretty interesting. You'd be surprised.

On [day17] and [day24], I definitely tried to use LLMs and they all kind of dropped the ball. They couldn't even really come up with logic. And yes, even [`o1`][o1].

The results actually surprised me. Perhaps because these were pretty novel problems (again, huge shoutout to [Eric][eric]).

I wasn't sure if I was alone in this approach, but it seems like others can confirm perhaps a lackluster / disappointing performance from LLMs. And yes, I know that o3 is going to out perform like 99.9% of developers in competive programming and all of that, so who knows. But I will say actual use cases seemed lackluster at the moment. And I'm sure all the die-hard defenders are going to defend it and sure - i'm sure that some of this could have been prevented or solved with better prompting but that was not the result that I would have hoped for.

### Ok fine, but why this year?

### Did you accomplish all your goals? Any excuses?

### Beh, sure - well did you solve them all without any help?

### Oh so you're a bad engineer?

### What was the community like?

### Ok well... what did you learn?

### Ok this is sounding better - are you going to do it again?

### Well... what are you going to do now?

More specifically, my girlfriend asked me something along the lines, "are you sad that this is done and no one really cares?" and I said I'm sad, but not because no one cares but sad that I didn't do a better job. I suppose that's part of it.

And as for what's next, I have many many other blog posts and side projects that I need to work on as well as my full time job, so I am not hurting for next steps.

## Hardest Problems

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
