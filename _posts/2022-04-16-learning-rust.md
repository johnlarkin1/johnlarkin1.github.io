---
title: Udemy - Learning Rust
layout: post
featured-img: rust
categories: [Education]
---

Took a stab at a new programming language that we utilize for work. 

Introduction
============
While this is a bit of a 180 from the world of web development (of which I am very much still interested in!!), I wanted to get better at programming in Rust. We use Rust at Dropbox pretty extensively to cover our sync engine (to ensure that all your data is safely, securely, and quickly uploaded). When first reading through some of the code, I was a bit lost syntactically. A teammate (and awesome engineer) at Dropbox led the charge on reading [The Rust Book][rust-book], which as you can tell from the hyperlink has been totally digitalized. It's worth a read and a helpful first overview.  

If you want to skip reading all of this, [here's the code][code] for the class and some other fun portions. 

Comments
========
Overall, there are a lot of nice things about Rust. Firstly, check out their adorable little language mascot:

![rustacean](/images/rust-class/rustacean-flat-happy.png){: .center-shrink }

But speaking technically, it's an incredibly quick and memory efficient language. The compiler is leagues better and more clear with errors than `gcc` or anything like that. It's relatively functional which offers a lot of benefits (and many programming languages/frameworks like React are moving that way).

I’ve gotta say though... I'm not sure if I like the idea of no explicit classes. I feel like Golang does this as well (despite Go basically supporting classes just without the explicit `class` reserved keyword). It seems odd to me. I feel like syntactically it would make things so much easier. Say what you will about Python, C#, or Java, but there are some beautiful designs that come out of canonical OO programming. In order to define relationships between objects, Rust instead turns to `traits`. It has `structs` and you can define methods and implementations for these various traits, structs, and enums, but that's a major division in terms of programming paradigm.

There have been a couple articles ([here (Why not just add classes?)][post1] and [here (where Rust's Enums shine)][post2]) that I think are relevant and good reads. 

Specifically about the class, this class was in turn relatively quick. I thought it was a very good introduction and a little bit more hands on than just reading the book. I would recommend it. That being said, certianly some of the syntax / version of Rust the instructor was using did seem a bit older. Some of the compiler errors that the instructor got, I did not, solely because I was on a later version. 


Certificate
===========
Again... you have to right?

![certificate](/images/rust-class/RustLanguage2022CourseCertificate.png)

Overview
========
You'll note that for this one there's a slight discrepancy in how I took notes. I was doing a bit more programming than note taking, so most of the notes are embedded in the `.rs` files themselves. 

You can see the HTML version of my notes [here][notes-html-preview] (rendered through this nice Github html preview site [here][github-preview]).

Here's a Table of Content generated from [here](https://ecotrust-canada.github.io/markdown-toc/). You can see it's not much (check the actual code!!).

- [The Rust Programming Language](#the-rust-programming-language)
  * [Section 1: Introduction](#section-1--introduction)
    + [2. Installing and Configuring Rust](#2-installing-and-configuring-rust)
    + [3. Hello, Rust](#3-hello--rust)
    + [4. Introducing the Cargo Package Manager](#4-introducing-the-cargo-package-manager)
  * [Section 2: Types and Variables](#section-2--types-and-variables)
    + [6. Numbers on the Computer](#6-numbers-on-the-computer)
  * [Section 3: Control Flow](#section-3--control-flow)

Projects
========
Again, there's a ton of cool stuff that I'm hoping to eventually get to with Rust. For now though, just to actually have some more experience in the wild (besides what I expanded on in the class for fun), I just implemented the first five problems of [Project Euler][euler], which are always a good confirmation that you've got a basic understanding. 


Other Helpful Links
===================
Again, the normal disclaimer. While I've included all of the code, please do not abuse this. You're only doing yourself a disservice in terms of what you learn.

As always feel free to email me with questions - although, I'm far *far* from an expert! I am more than happy to help debug. 

**Once again, you can see the notes [here][notes-html-preview].** 

[comment]: <> (Bibliography)
[rust-book]: https://doc.rust-lang.org/book/
[course]: https://www.udemy.com/course/rust-lang/learn
[github-preview]: https://htmlpreview.github.io/
[notes-html-preview]: https://htmlpreview.github.io/?https://github.com/johnlarkin1/rust-programming-course/blob/main/rust_programming_language/rust_course_2022.html
[euler]: https://projecteuler.net/
[code]: https://github.com/johnlarkin1/rust-programming-course
[post1]: https://users.rust-lang.org/t/why-not-just-add-classes/4618
[post2]: http://smallcultfollowing.com/babysteps/blog/2015/05/05/where-rusts-enum-shines/