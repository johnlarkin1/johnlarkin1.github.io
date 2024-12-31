---
title: Quantum Exploration
layout: post
featured-img: quantum-circuits
categories: []
summary: Looking a bit at the recent quantum developments.
favorite: true
---

<div class="markdown-alert markdown-alert-disclaimer">
<p>First of all, I want to clarify that I am by no means an expert in anything quantum. I have no PhD. The farthest I ever got was taking Physical Chemistry (PChem) I and II in college. I have not worked at Google, or a leading quantum company. I haven't ever run a program on a quantum board.</p>

<p>That being said, it's always been something I've loved and been captivated by. PChem was one of my favorite classes in college. I still distinctively remember sitting in the second floor of Science Center with <a href="https://www.swarthmore.edu/profile/kathleen-howard">Kathleen Howard</a>. A couple of my brilliant classmates were <a href="https://www.linkedin.com/in/aditi-kulkarni-72a267131">Aditi Kulkarni</a> (my lab partner for most of it), <a href="https://www.instagram.com/bmcorthores/p/Cyw7wYKsUwH/">Brian Gibbs</a>, and <a href="https://www.linkedin.com/in/jacob-kirsh-338754b3">Jacob Kirsh</a>. All brilliant and now either all getting PhDs, MDs, or some other impressive feat. Kirsh is even getting his PhD in biophysical chemistry at Caltech now. Fun quantum related note, one of my friends parents is also CEO of a major (public) quantum computing company. </p>

<p>All of that is really more from a true underling quantum physics perspective, but as for quantum computing, I am even farther from an expert. The one claim to fame is in my Algorithms class with <a href="https://www.cs.swarthmore.edu/~brody/">Josh Brody</a> is him calling out <a href="https://en.wikipedia.org/wiki/Shor%27s_algorithm">Shor's Algorithm</a>, which is a quantum approach for factoring prime numbers. This obviously has huge repercussions for encryption and cyber security.</p>

<p>The only other disclaimer is that I have numerous long positions in a couple of publicly traded quantum computing companies. </p>

<p>Regardless, this will hopefully be a review of some fun things I learned back in PChem, as well as looking at it a little bit more from a true quantum computing angle.</p>

</div>

<br>

If curious, I sadly don't have my PChem notebook or binder given they burned down with my Dad's house (or I might have given it to some of the younger tennis guys 😬), but I did manage to find part of the first homework that I turned in for one of my additional problems for PChem.

<div style="text-align: center;">
    <embed src="{{ '/pdfs/PChem2_HW1.pdf' | prepend: site.baseurl }}" width="500" height="375" type="application/pdf">
</div>

# Motivation

In this post, we're going to:

1. Review some quantum computing fundamentals
2. Look at Google's Willow post
3. Look at some backlash that was trending from Google's post
4. Explore IONQ's trapped ion approach
5. Explore IBM's `qiskit` Python library for designing and building circuits
6. See if we can run some semblance of an experiment (either mirroring Google's random circuit sampling benchmark or even Shor's algorithm as a rough heuristic)

# Quantum Computing Fundamentals

# Google's Willow

## Error Correction and `below threshold`

[comment]: <> (Bibliography)
[brody]: https://www.cs.swarthmore.edu/~brody/
[howard]: https://www.swarthmore.edu/profile/kathleen-howard
[deets]: https://www.linkedin.com/in/aditi-kulkarni-72a267131
[gibbs]: https://www.instagram.com/bmcorthores/p/Cyw7wYKsUwH/
[kirsh]: https://www.linkedin.com/in/jacob-kirsh-338754b3
[shors]: https://en.wikipedia.org/wiki/Shor%27s_algorithm
[willow]: https://blog.google/technology/research/google-willow-quantum-chip/
[willow-backlash-pt1]: TODO
[willow-backlash-pt2]: TODO
