---
title: Invisible Ink
layout: post
featured-img: invisible-ink
categories: [Quick]
summary: I'm a sucker for invisible ink
---

I thought it'd be fun to add some invisible ink to the blog. 

This is going to be super quick, and partially because I've been forgetting to blog so much. One of my favorite features on iOS is the ability to do some invisible ink.

You'll have hover over this text. <ink>It's a modest attempt at doing the same</ink>. 

The code is pretty simple. In your css files, you can do:

```css
ink {
    filter: blur(2px);
    transition: filter 1s ease-in;
}

ink:hover {
    filter: blur(0px);
    transition: filter 1s ease-out;
}
```

and then in your corresponding markdown, you can just wrap the text with `<ink> my invisible ink text </ink>`. And :tada: voila :tada: <ink> you're done! </ink>.

In the future, I hope to do something a little bit more fun with animated css. 

Let me know if any suggestions!