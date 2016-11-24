---
title: Emacs or VIM?! Let the war rage on...
layout: post
---

Want to be quicker when using vim? Or learn more about the war between Emacs and Vim?  

I use Vim and much of the posts that I've written for this website, as well as much of the backend architecture I wrote in vim. For larger projects, I prefer using [atom](https://atom.io/) or [sublime](https://www.sublimetext.com/). However, for quicker projects, where I want a lightweight editor to access files and navigate a director quickly, I think vim is a perfect tool. Plus it's going to be hard to find a computer that doens't have vim installed. 

As I was coming up with this website and some of the blog posts, I found myself dying to know more shortcuts. I knew the basics, but I thought it might be great to have a post where I could just store / share some of the tricks that I've learned in a convenient place. Essentially, most of the information can be found [here](http://www.keyxl.com/aaa8263/290/VIM-keyboard-shortcuts.htm). However, I thought even that was overkill. 

General Vim Use:
---------------

**Not in insert mode:**

* `y` - yank or copy 
* `x` - cut
* `p` - paste
* `w` - jump to beginning of words (equivalent to shift + arrow keys on mac)
* `W` - jump to beginning of words (ignoring punctuation)
* `e` and `E` - same as above just to the end of the words 
* `b` and `B` - same as above just jumping backwords
* `0` - move to beginning of the line
* `$` - jump to end of the line
* `ctrl + r` - redo
* `u` - undo
* `.` - redo last command
* `J` - join the current line and the line under cursor together
* `v` - start highlight
* `V` - highlight entire line
* `cw` - change the word (essentially deletes the word and then enters insert mode)
* `dd` - delete entire line

**In insert mode**:

* `shift + left || right keys` - move by full words (similar to w and b outside of insert mode) (also \|\| indicates or)
* `i` - enter insert mode at cursor
* `A` - enter insert mode at end of line
* `o` - enter insert mode on new blank line below cursor
* `O` - enter insert mode on new blank line above cursor

**Searching:**

* `/pattern` - searches for said pattern
	* `n` - find next of said pattern
	* `N` - find previous of said pattern 

**Random Cool Tricks:**

* `~` - switch case of highlighted character
* `d*w` - where * is an integer, this will delete * many words

I also thought it would be kind of cool if I provided some history behind Emacs and Vim and a little bit more about how they work. 

History of Vim
===============
So there's apparently a distinction between *Vim* and *Vi* both of which you can use from your terminal to just edit a file. Vim stands for Vi IMproved... over vi. Vim is free and is ranked one of the top editors every year. It was developed by [Bram Moolenaar](https://en.wikipedia.org/wiki/Bram_Moolenaar), who is just a random gnarly Dutch programmer. He developed it in 1988. 
Big perks?
- cross compatable across systems
- it's suuuuuper lightweight. you can make quick changes to a file and close it up in the blink of an eye
- it divides into reading and editing vs insertion mode. helps differentiate tasks.

History of Emacs
================
I was surprised to see that Emacs was actually written in 1976. Given how ubiquitous Vim is, I would have figured that it would have come first, because it's so lightweight. Emacs boasts of having over 2,000 commands. That's sooo many commands, but imagine if you're a master of them. It rewards the users who use it longer.
People customize Emacs heavily. Which is great for their own internal use, but if you try to switch between versions of Emacs, then things are bound to be different and more complex. 
Big perks? 
- highly customizable 
- features keyboard macros for user-defined batches of editing commands
- capable of formatting and printing documents like a word processor 

It was pretty fun to do a bit of research about the two topics. Stack overflow has some very heated articles about it. This is something that people have been debating about and arguing for a long time. There obviously is no right or wrong answer, but there are just strong opinions about each side. 
