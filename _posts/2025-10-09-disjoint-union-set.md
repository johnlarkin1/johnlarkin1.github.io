---
title: "Disjoint Union Set"
layout: post
featured-img:
mathjax: true
pinned: false
categories: [Algorithms]
summary: Looking at a data structure i wasn't familiar with until I got torched in an interview
---

Recently, I did an interview and basically failed it because I wasn't familiar with a Disjoint Union Set (and I certainly couldn't complete the C++ interview in time building this data structure naturally).

I figure I would go back to this and do a deeper dive because I wasn't as familiar with it.

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Theory](#theory)
  - [What is a Disjoint Set?](#what-is-a-disjoint-set)

# Theory

## What is a Disjoint Set?

A _disjoint set_ or _union find_ or _disjoint union set_ are all the same thing. It is a data structure that is optimized for handling various set operations and mainly focuses on two methods: `union` and `find` (hence one of the names).

So we'll mainly have a target set. We're representing each subset as an inverted tree (i.e. all the child nodes are pointing back to the root).

As a reminder, trees are a specific form of a graph where:

- undirected
- at most 1 path between any 2 nodes
- acyclic

Two types: **out-tree** and **in-tree**.

![tree-types](/images/disjoint-union-set/tree-types.png){: .center-image .lightbox-image}
