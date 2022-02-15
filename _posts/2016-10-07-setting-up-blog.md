---
layout: post
title: Creating what you're looking at
featured-img: double-bloom
mathjax: true
categories: [Setup]
---

I talk about how I actually set this blog up and some more helpful resources.

There's a lot of different ways that you can go around building something. I actually struggled to generate this page... and this blog. Overall, they were relatively small things, but regardless, the progress was relatively slow. I figured that it might be best if I create a page (largely for myself)) where I can store what I learned and also give attributions. 

Resources:
=========

1. First of all, a lot of what I learned in how to model and create the design of the blog was based off of one of my Swarthmore professor's blogs. Specifically, [Matt Zucker's blog](https://mzucker.github.io/) was *really* helpful. 

2. Second of all (which in hindsite, probably should have been my first resource), the [Jekyll](https://jekyllrb.com/docs/home/) homepage was helpful.

3. Third, I became desperate and needed some help. I turned to the [Jekyll talk page](https://talk.jekyllrb.com/). While the page is primarily meant for asking Jekyll based questions, I also asked some questions about the construction of the sass file and the way to use Jekyll and html. 
 
4. A great and pretty famous resource is [http://www.w3schools.com/](http://www.w3schools.com/). While I haven't been lucky enough to walk through **all** of the tutorials, I have been to various element pages and checked out the documentation. 

5. Finally, just a lot of testing around with the site. These were the core of the resources used to construct the blog. I'm still learning a bunch, so any advice and emails and comments would be greatly appreciated. Obviously, you can see the code on my actual [personal github page](https://github.com/johnlarkin1).

This is going to be more of a running list for myself.

HTML Notes:
==========

- `div`: this is going to be the basic building block
- `display`:
	- block - this means display it as a block like it spans across the page; similar to <p>
- `pre`:
	- This guy is super important for defining characteristics over multiple lines... i.e. it's great for code formatting.
	- As this [link](https://perishablepress.com/perfect-pre-tags/) puts, they are ideal for lines that "need to retain character spacing, display unformatted characters, keep inherent line breaks, and so on."


SCSS Notes:
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

- Tables are also pretty weird
	- You're going to want to read up on `td`, `th`, `tr`, and `thead`. There are some slight nuances. You can read about some of them [here](http://stackoverflow.com/questions/5395228/html-tables-thead-vs-th).

CSS Notes:
==========

- `>` selector operator:
	- It means immediate children. For example, as seen on stack overflow [here](http://stackoverflow.com/questions/4459821/css-selector-what-is-it), it defines what tier the operator is affecting.
	- This is like a specific selector, instead of a 'descendant selector', which just means any matching element in that first class. 
	- The selector operator is a bit more picky than that.
- partials
	- There's a careful distinction between partials with css files. 
	- Partials are really just little snippets of css code that you can include in scss files
	- You **need** to label it with an underscore so like `_partial.scss` in order for it to be recognized as a partial and not a full css file. 
	- These guys are used with the `@import` command
- calculations:
	- You can use `calc` to perform a quick calculation on how to set the web page up
	- For example, that's how the margin of the actual content are calculated for my blog contents:
		- `max-width: -webkit-calc(#{$content-width} - (#{$spacing-unit}));`
		- `max-width: calc(#{$content-width} - (#{$spacing-unit}));`
	 
		You can see here that it is taking the content width and subtracting off the spacing unit. Also note, that the -webkit-calc is for other types of browsers that might not be compatible. 		
- Who likes to have bullet points with lists? I don't always. Make sure to include that under the class type of where you're making the list!
	- for example, you want to say: `list-style: none;` under the appropriate css class

Markdown Notes:
==============
Markdown makes life sooo much easier. It's essentially just making HTML way more easy to right. Pretty much everything I know about it has come from either peers or professors or [this super helpful link](https://en.support.wordpress.com/markdown-quick-reference/).

Here's a brief summary:
- `*text*` or `_text_` - _italics_ 
- `**text**` or `__text__` - __bold__
- links can be done in multiple ways. you can have a bunch of links at the end of a post so that it's kind of like a bibliography! Otherwise, the syntax is just `[this is going to be hyperlinked](some-url-for-hyperlinking)`
- images are similar just preface the hard brackets with an exclamation point
- You can STILL cover markdown with HTML. For example, if I want to change the font of certain text, I could do: 

`<span style="color: #111111; font-family: Consolas; font-size: 2em;">woo!</span>`

which would give:

<span style="color: #FF69B4; font-family: Consolas;">woo!</span>

So that's a pretty nice thing to keep in the back of your mind for blog posts. 

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

Installing Comments with Disqus:
===============================

This was simply a bit of googling and then comparing with Matt's code. It's pretty straight forward though because Disqus has a bunch of really helpful pages. 

~~~javascript
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/

var disqus_config = function () {
    this.page.url = '{{ page.url | prepend: site.url }}';  // Replace PAGE_URL with your page's canonical URL variable
    this.page.identifier = '{{ page.id }}'; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};

(function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = '//johnlarkin1-github-io.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
})();
</script>
~~~

I literally pulled that code straight after registering my site on Disqus. Overall, there are like two main links:
1. https://johnlarkin1-github-io.disqus.com/admin/install/
2. http://www.perfectlyrandom.org/2014/06/29/adding-disqus-to-your-jekyll-powered-github-pages/

Getting a `favicon.ico` to Show:
================================

This was kind of tricky. I'm still not actually totally sure how to have an animated `favicon.ico` as your little page icon. Essentially, these are the things that in chrome and other browsers show up next to your tab if you're not on the page. They're little like logos. At first, I was like ohhh those are cute, maybe I'll add one... and then I spent like 4 hours researching them. Note, they should be relatively square. Otherwise, they're not really going to work out. I tried to make the mountain in the upper left corner, but it was too elongated.  

I got stuck and had to ask about it on talk jekyll. The answer wasn't totally correct, but I did a bit more googling. Adding a "?" apparently works as a hacky trick, but generally, people aren't sure why. [Check it out](https://talk.jekyllrb.com/t/trouble-creating-visible-icon-in-pages/3315/).

Linking PDFs:
=============

There are several projects that I've worked on where I didn't want to do a full write up, but I've already written up a paper or some report for class. I thought it would be easier to just link the actual pdfs that I've already created. Because Github is the shit and relatively easy to work with other file types, I just had to load the files into a certain public spot (which I was fine with) and then just pass it a command. 
It's also nice to utilize jekyll here, because we can easily specify our baseurl. So for example, if I wanted to link my final E91 (biomedical signals) presentation, I could just write the following code:
~~~html
<a href="{{ site.baseurl }}/pdfs/E91FinalProjectFlowCytometry.pdf"> link </a>
~~~
And that beautiful piece of code, will take you right to here:
<a href="{{ site.baseurl }}/pdfs/E91FinalProjectFlowCytometry.pdf"> link </a>

---------

Ok! I think that's all I have for now. Let me know if you guys have any comments and / or there are other things you think I should add. This is the first post, so I'm still kind of new to what's going on so far.
