---
title: Creating what you're looking at
layout: post
---

How are you seeing this?

There's a lot of different ways that you can go around building something. I actually struggled to generate this page... and this blog. Overall, they were relatively small things, but regardless, the progress was relatively slow. I figured that it might be best if I create a page (largely for myself)) where I can store what I learned and also give attributions. 

Resources 
=========

1. First of all, a lot of what I learned in how to model and create the design of the blog was based off of one of my Swarthmore professor's blogs. Specifically, [Matt Zucker's blog](https://mzucker.github.io/) was *really* helpful. 

2. Second of all (which in hindsite, probably should have been my first resource), the [Jekyll](https://jekyllrb.com/docs/home/) homepage was helpful.

3. Third, I became desperate and needed some help. I turned to the [Jekyll talk page](https://talk.jekyllrb.com/). While the page is primarily meant for asking Jekyll based questions, I also asked some questions about the construction of the sass file and the way to use Jekyll and html. 
 
4. A great and pretty famous resource is [http://www.w3schools.com/](http://www.w3schools.com/). While I haven't been lucky enough to walk through **all** of the tutorials, I have been to various element pages and checked out the documentation. 

5. Finally, just a lot of testing around with the site. These were the core of the resources used to construct the blog. I'm still learning a bunch, so any advice and emails and comments would be greatly appreciated. Obviously, you can see the code on my actual [personal github page](https://github.com/johnlarkin1).

This is going to be more of a running list for myself.

HTML Notes
==========

- `div`: this is going to be the basic building block
- `display`:
	- block - this means display it as a block like it spans across the page; similar to <p>
- `pre`:
	- This guy is super important for defining characteristics over multiple lines... i.e. it's great for code formatting.
	- As this [link](https://perishablepress.com/perfect-pre-tags/) puts, they are ideal for lines that 'need to retain character spacing, display unformatted characters, keep inherent line breaks, and so on.' 
-


SCSS Notes
==========
General note:
SCSS: this stands for syntatically awesome style sheets. This is actually a scripting language that is interpreted into CSS (cascading style sheets!!). First note, that variables can be set with the `$` sign symbol. These are essentially like global variables that you can use throughout the entire program. 

Another great thing, SCSS lets you nest your CSS selectors in a nice clean heirarchy!! This means that:
```
nav {
  ul {
    margin: 0;
    padding: 0;
    list-style: none;
  }
```

will actually be interpreted as the slightly more cluttered CSS version shown below:
```
nav ul {
  margin: 0;
  padding: 0;
  list-style: none;
}
``` 
with the given heirarchy being relatively more clear.





CSS Notes
==========

- `>` selector operator:
	- It means immediate children. For example, as seen on stack overflow [here](http://stackoverflow.com/questions/4459821/css-selector-what-is-it), it defines what tier the operator is affecting.
	- This is like a specific selector, instead of a 'descendant selector', which just means any matching element in that first class. 
	- The selector operator is a bit more picky than that.
- partials
	- There's a careful distinction between partials with css files. 
	- Partials are ___fill this in___
- calculations:
	- you can use `calc` to perform a quick calculation on how to set the web page up
	- for example, that's how the margin of the actual content are calculated for my blog contents:
		- `max-width: -webkit-calc(#{$content-width} - (#{$spacing-unit}));`
		- `max-width: calc(#{$content-width} - (#{$spacing-unit}));`
	 
		You can see here that it is taking the content width and subtracting off the spacing unit. Also note, that the -webkit-calc is for other types of browsers that might not be compatible. 		
- who likes to have bullet points with lists? I don't always. Make sure to include that under the class type of where you're making the list!
	- for example, you want to say: `list-style: none;` under the appropriate css class

Markdown Notes
==============

Installing Latex
================
I'm a big fan of Latex. I was eager to be able to do things like this:

$$ 
x = \frac{3}{2} 
$$

This was simply enabled by including the following bit of code, which loads in the appropriate MathJax javascript. 
~~~javascript
<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
~~~

The syntax isn't too tricky... you can essentially use it like Latex. For inline, you can still just use `$$ ... $$`. You don't have to worry about it automatically creating the block math. The above fraction can be shown as:
```
$$
x = \frac{3}{2}
$$
```

