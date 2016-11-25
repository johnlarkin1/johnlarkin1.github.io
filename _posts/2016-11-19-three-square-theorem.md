---
title: Natural Number Representation
layout: post
tags: 
---

What numbers can't be represented as the sum of three squares? 

The question we're trying to answer is what number(s) x cannot be represented as:

$$ 
x = a^2 + b^2 + c^2 
$$ 

First, let's explore where this problem even came from. 

History of Sum of Three Squares Problem
=======================================

### My History
So this problem was actually pitched to me by my bright and awesome friend [Deets][Deets]. Deets (real name Aditi Kulkarni) is a math minor at Swarthmore. Chem major. Premed. Essentially an art and psych minor on the side as well. Anyway, as I write this, she's in number theory, a class that she has very mixed feelings about. Regardless, she was telling me about this problem originally presented in her number theory class. She was like 

> Hey tell me all the numbers that can't be represented by the sum of three other squares. Write a program whatever. 

And I was hooked. The code and algorithm can be found [here][code source]. But here's the actual history of the problem. 

### Real History of the Problem
This is a formulation by [A.-M. Legendre][legendre], a french mathematician. *If you get one thing out of this blog post, **please** let it be clicking on that link and checking out the caricature of A-M Legendre*. It is beyond wild.  

Anyway, so Legendre's three-square theorem actually is the following: 
> A natural number can be represented as the sum of three square integers, $$ n = x^2 +y^2 +z^2 $$ if and only if $$n$$ is **not** of the form $$ n = 4^a (8\cdot b + 7) $$

Aaaand that's kind of the solution. I could have just generated numbers using that formula, but I was curious to see if there was an easy way to generate the numbers in another way, if we didn't already know the answer. While there were multiple speculations and attempts at proofs, Legendre actually found the first proof for his 3-square theorem in 1796. Or at least that's what [this][threesq] article says. I, having not taken number theory, actually struggled for a clean proof on how to do this. I should still probably reach out to Deets, but [this is one that I found][threesqproof].

I'll lay out the base outline of the proof. They assert:

**THEOREM 1:** *If $$m$$ is a positive integer not of the form $$4^n (8n+7)$$, then $$m$$ is the sum of three squares.* 

As of right now (11/24/16 11:38PM), I haven't fulled worked out the math. I don't understand a few of the other theorems needed to prove it, so I am going to come back to this portion. 

[Here's another super helpful link about this proof though][threesqproof2].
 
Algorithm
=========

Feel free to check out my code to the solution. I first wrote up a python script because I wanted to make sure I knew what I was doing and python is essentially like writing pseudocode. Then I made a much nicer (or at least I think so) version in C++. 

I would love any improvements on this brute-force approach to solving this problem. I have perused online and have not found too many programs that solve this in an easier way. It's an interesting problem to see if we can do better than a brute force search. Right now, the program that I have developed is roughly $$ O(n^4) $$. The logic is such: looping over every number between 1 and max range. **Originally**, for every one of those numbers, I am considering the entire range of the possible other three numbers, limiting them to a max value of the maximum number itself.

That's obviously dumb though. We're never going to get to a point where we have some number, and then sum of any number greater than it is going to actually be equal to it. That doesn't make sense. So then let's only go up to the number itself. 

But again that's also not as good as we can do. Realistically, we should go up to the square root of the number. Or like $$ \lfloor \sqrt[]{n} \rfloor$$ for each number.

This greatly helped decrease the runtime. I will actually do an analytic analysis of the expected runtime, just to review a bit of the analytic skills. 

Example Runtime
===============

**Example 1: $$MAX\_RANGE = 250$$**

*Note: The differences on the time will be explained below*.

| Upper Bound     | Real Time | User Time | Sys Time | 
| :-------------- | --------: | --------: | -------: |
| MAX_RANGE (500) | 0m14.641s | 0m14.179s | 0m0.083s |
| number		  | 0m3.944s  | 0m3.864s  | 0m0.031s |
| ceil of sqrt(num) | 0m0.013s| 0m0.005s  | 0m0.003s |

Obviously, I like the last option the most. 

__Time Notation Note__

Let's be clear about what this `time` command is doing with our program. 

`real` - wall clock time. So we hit enter to execute our program and the timer starts and then when it is finished the timer ends. This includes the time to various other modules like I/O for example.  
`user` - actual CPU time to execute the program. `real` and `user` should be relatively similar. So this is when things like the I/O don't actually count.  
`sys` - actual CPU time in the kernel. This is not library code, this is more like time in system calls. A clean (although unrealistic) example of what could create a drag on the `sys` report is if for example, your program needs to allocate like a GB of memory. That's going to take forever and all of that time allocating memory would be clumped under `sys` time.

See [this link][time link] for a highly popular stack overflow page about this. 

Time Analysis
=============

Let's think for a second about the runtime of such a program. Specifically, we'll consider the asymptotic runtime of such a program.

Essentially, what we have is for some $$n$$, which is equivalent to our $$MAX\_RANGE$$, we have:

$$
\sum_{num=0}^{n} \sum_{i=0}^{\sqrt[]{num}} \sum_{j=i}^{\sqrt[]{num}} \sum_{k=j}^{\sqrt[]{num}} k 
$$

So really, there might be a better way to get an exact analytic runtime out of this, but it's essentially going to be dominated by the inner most for loop. Which is essentially running for square root time. But we're doing this three times over, so as of right now, I'm pretty sure we'll be $$ O(n^4) $$, but actually not positive.

Let's break it down logically. 

For each number, we're going to look up to square root of that number. But for each number in that, we're doing in the worst case square root of that number work. For each each number in that we're doing in the worst square root of that work. 

In other words, I believe that we have those inner three dependent for loops being executed for:

$$
\sum_{i=0}^{\sqrt[]{num}} \sum_{j=i}^{\sqrt[]{num}} \sum_{k=j}^{\sqrt[]{num}} k \approx \frac{(\sqrt[]{num}) (\sqrt[]{num}+1) (\sqrt[]{num}+2)}{6} 
$$

This in accordance to the power series. Thus taking into acccount the last for-loop, we can see that this should be $$O(n^{2.5})$$, which is much better than the original $$O(n^4)$$. 

Therefore in conclusion, final estimated runtime:

$$
O(n^{2.5})
$$

Extension
=========

What's better than considering if a number can be composed of three squares? What about if we gave ourselves one more degree of flexibility and said **what numbers can be represented by the sum of four squares**?

The answer is conveniently all of them. Check it out [here][foursq]. The best part is? *All natural numbers can be represented by the sum of four integer squares*. So yeah. It's a bit cooler than the three- square version, but how boring would a program be that just outputs an empty set?

Conclusion
==========

This obviously wasn't really a code heavy project. But it's got some pretty interesting math. I also am still relatively new at this so I figured this would be a good chance to get my feet wet on a project where I wasn't really looking at pseudocode and there wasn't really a clean path. That being said... the solution I implemented was still brute-force just with a slight twist. As always, if people have more clever solutions, then *please* let me know! I'd love to hone my skills. 


[comment]: <> (Bibliography)
[Deets]: http://www.swarthmore.edu/news-events/qa-aditi-kulkarni-17-founder-red-lips-project
[time link]: http://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1
[code source]: https://github.com/johnlarkin1/three-square-problem
[legendre]: https://en.wikipedia.org/wiki/Adrien-Marie_Legendre
[threesq]: https://en.wikipedia.org/wiki/Legendre's_three-square_theorem
[threesqproof]: http://www.ams.org/journals/proc/1957-008-02/S0002-9939-1957-0085275-8/S0002-9939-1957-0085275-8.pdf
[threesqproof2]: http://www.math.uchicago.edu/~may/VIGRE/VIGRE2009/REUPapers/Wong.pdf
[foursq]: https://en.wikipedia.org/wiki/Lagrange%27s_four-square_theorem
