---
title: Euclidean Algorithm - Greatest Common Divisor
layout: post
featured-img: euclidean-color
mathjax: true
categories: [Math, Number Theory]
---

I was dying to take a break from getting thrashed by Theory of Computation and my E90.

![gif](/images/gcd/euc-algo.gif){: .center-super-super-shrink }

<center> <i> Don't understand? Read below. This gif summarizes the subtraction based Euclidean algorithm. </i> </center>

It's been awhile since my last blog post and I was feeling particular frustrated with my Theory of Computation homework and my E90. That was about two days ago, but it made me want to sink my teeth into something a little easier. I'm just getting the chance to push this tonight after continually getting jammed by ToC and others (just check the commit time on this post!). While those have been wicked frustrating at some points throughout this semester, the upside (as always!) is that I've learned a bunch. Hell, hopefully soon I can do some cool writeups on things like [quines][quine] or an [LSTM cascade][lstm] or general grammars (also I'm taking [computer vision][cv] with [Matt][matt] so there's a bunch of cool stuff from that). As a result, I wrote up some cute little programs that essentially amount to doing Euclid's 'Euclidean Algorithm' - which really just computes the greatest common divisor (gcd) between two numbers.

# History of Euclidean Algorithm

As always, wikipedia has a good article about this algorithm. Also, Khan Academy does which is nice to see. This algorithm was first described in Euclid's **Elements** which was written around 300 BC. So yeah. I'm _definitely_ not reinventing the wheel here. There's really not too much else that's interesting about the history of the algorithm. The problem is defined as such:

> Find the largest number that divides both of the two input numbers without leaving a remainder.

# Algorithm

There are multiple commonly found versions of the algorithm now. Things can be expedited using the modulo operator, but Euclid originally proposed his algorithm with subtraction. To see the computational inefficiency that this can solve, you can definitely check out the `gcd_subtract.py` code and stare at your terminal waiting. However, because of it's mathematical cleanliness, I will present it here and then we can analyze it below.

```python
def euclidean_algorithm(a,b):
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a
```

Legitimately, that simple. The quicker division version can also be found here:

```python
def euclidean_algorithm(a, b):
    while num1 != 0:
        t = b
        b = a % b
        a = t
    return a
```

# Proof of Validity

Much more interesting is the mathematical proof of the correctness of this algorithm. Again, if the below argument is confusing, I highly suggest reading through the Khan Academy site [here][khan] for a further explanation.

First, the algorithm is set up so that for each iteration, we use the previous iterations changes. This is why in the first algorithm you see the resetting of either $$ a $$ or $$ b $$ and in the second algorithm, you see we need to store a temporary variable so we still have both pieces of information.

This derivation and proof will closely follow Wikipedia's. I'm doing this partially for myself because I want to understand but partially because I might break it down more and maybe you like my explanation.

Let $$ k $$ be the iteration of the `while` loop.

Let's declare $$ r*{k-1} $$ and $$ r*{k-2} $$ to be nonnegative remainders in our algorithm.

_Assume_, that $$ b $$ is larger than $$ a $$. If it is not, the first step of the algorithm will be to switch those two variables.

In our initial step, that is the very first time the loop is run and $$ k = 0 $$, the remainders are set to be $$ a $$ and $$ b $$. That is,

$$
\begin{align}
r_{k-2} &= r_{0-2}\\
&= r_{-2} \\
&= a
\end{align}
$$

$$
\begin{align}
r_{k-1} &= r_{0-1}\\
&= r_{-1} \\
&= b
\end{align}
$$

We can pose our problem in the following manner. The $$k$$th iteration is trying to find a quotient $$ q_k $$ (that is, quantity produced by dividing two numbers) and a remainder $$ r_k $$ that satisfy:

$$
r_{k-2} = q_k r_{k-1} + r_{k}
$$

Note, we also have the constraint that $$ r*{k} < r*{k-1} $$. Think about the above constraint saying like ok, $$ q_k $$ is going to represent how many times we can subtract one number from the other before we have to switch their values. $$ r_k $$ is going to say ok what's the remainder value after all those subtractions.

Because of the above constraint, it is very important to note that are remainders are **constantly decreasing** with every iteration $$ k $$. Also, we constrain the remainders to be strictly greater than zero. Because of this, it ensures that the while loop will terminate and we have an appropriate algorithm in terms of runtime. We just need to be convinced thoroughly that this program gives the right answer.

With all of this defined, it is easy to generate a few sequences of the algorithm:

$$
\begin{align}
a &= q_0 b + r_ 0 \\
b &= q_1 r_0 + r_1 \\
r_0 &= q_2 r_1 + r_2 \\
r_1 &= q_3 r_2 + r_3 \\
& \; \cdots
\end{align}
$$

With these equations defined, we can make some claims.

The proof can be summarized in two steps and has a flavorful dose of induction involved.

1. The final nonzero remainder $$r_{N-1}$$ is shown to divide both $$ a $$ and $$ b $$. $$r_{N-1}$$ is a common divisor so it must be less than or equal to $$ g $$, the gcd.
2. Any common divisor of $$ a $$ and $$ b $$, including the greatest common divisor $$ g $$ must be less than or equal to $$ r\_{N-1} $$.

These two conclusions are going to be inconsistent UNLESS $$ r\_{N-1} = g $$ (which it does! so we're good).

1. We know that the final remainder $$ r\_{N} = 0 $$ as that is our stopping condition. We therefore know that our equation in reality is just:

$$
r_{N-2} = q_{N} r_{N-1} + 0
$$

This means that $$ r*{N-1} $$ divides its predecessor $$ r*{N-2} $$ because we literally just got a remainder of zero. So we know we have to be able to subtract off $$ q\_{N} $$ times and when we do that subtraction for however many times, we're left with zero.

We can then just carry this logic through. We know that $$ r*{N-2} $$ is divisible by $$ r*{N-1} $$ so then looking at

$$
r_{N-3} = q_{N} r_{N-2} + r_{N-1}
$$

we know that $$ r*{N-1} $$ is also going to divide $$ r*{N-3} $$ because $$ r*{N-1} $$ is divisible by itself and it's also divisble by $$ r*{N-2} $$. Therefore, we can just permeate this logic all the way down until our base case at

$$
\begin{align}
a &= q_0 b + r_ 0 \\
b &= q_1 r_0 + r_1 \\
\end{align}
$$

we know that $$ a $$ and $$ b $$ will therefore also be divisible by $$ r\_{N-1} $$.

Because $$ r\_{N-1} $$ is a common divisor, it must be less than or equal to the **biggest** common divisor, $$ g $$. In other words,

$$
r_{N-1} \leq g
$$

2. Let's look at a common divisor $$ c $$.

By the definition of what a divisor is, it means that there are some natural numbers such that $$ a = mc $$ and $$ b = nc $$. So let's take any remainder, $$ r_k $$. We can rearrange our formula above:

$$
\begin{align}
a &= q_0 b + r_0 \\
r_0 &= a - q_0 b \\
&= mc - q_0 nc \\
&= (m - q_0 n) c
\end{align}
$$

We can also apply this argument for any step $$ k $$. Let's just show one more for $$ r_1 $$. Now, let's pick a common divisor $$c$$ (different from last time) where $$b = mc $$ and $$ r_0 = nc $$. We know that our natural number $$ c $$ is going to be a divisor for both $$ a $$ and $$ r_0 $$ because of our proof above and our definition here. Because $$ n $$ is a natural number, it means $$ c $$ goes into it evenly. If $$ c $$ goes into $$ r_0$$ evenly, then $$ a $$ is also divisible by $$ c $$.

$$
\begin{align}
b &= q_1 r_0 + r_1 \\
r_1 &= b - q_1 r_0 \\
&= mc - q_1 nc \\
&= (m - q_0 n) c
\end{align}
$$

And thus, the logic continues until our stopping condition. This shows that $$ r*{N-1} $$ has to be the largest possible divisor before we have none left because of our stopping condition. In other words, this shows that $$ g \leq r*{N-1} $$.

### Conclusion:

Because we have $$ g \leq r*{N-1} $$ and $$ g \geq r*{N-1} $$, we have that

$$
r_{N-1} = g
$$

_In other words... this algorithm effectively returns the gcd._

# Code

As always, I wrote up a little code. There's two python scripts and a cpp script in a repo [here][code]. Feel free to grab them or as always, toss some improvements / suggestions my way. I'm a big fan.

[comment]: <> (Bibliography)
[quine]: https://en.wikipedia.org/wiki/Quine_(computing)
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[matt]: http://www.swarthmore.edu/NatSci/mzucker1/
[cv]: http://www.swarthmore.edu/NatSci/mzucker1/e27_s2017/index.html
[khan]: https://www.khanacademy.org/computing/computer-science/cryptography/modarithmetic/a/the-euclidean-algorithm
[code]: https://github.com/johnlarkin1/greatest-common-div
