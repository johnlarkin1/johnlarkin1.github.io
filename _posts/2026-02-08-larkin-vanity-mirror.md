---
title: "Vanity Mirror"
layout: post
featured-img: larkin-vanity-mirror
mathjax: false
python-interactive: false
chartjs: false
categories: [Development, Reflection]
summary: Public facing observability dashboard. PWA compatible. If there's interest, I'lll templatize / generalize it for LLMs to implement.
---

# Context

In the age of AI, shipping has become easier than ever. And also, borderline more addictive than ever. I will certainly rant about it in a Substack post at some point in the future, but as my projects grow, I wanted an easy way to keep track of various metrics (sadly, I'm curious and vain and want to see what people like). I built this "Vanity Mirror" as a way to do that, and figured it was fine to share publically (even if another [React2Shell][react2shell] RCE vulnerability occurs... are hackers really gonna want limited read-only permissions to my Google Analytics properties?).

# Demo

Feel free to check out the website here:

<div class="project-registry">
  <a href="https://larkin-vanity-mirror.vercel.app" target="_blank" rel="noopener" class="registry-card web">
    <span class="lang-icon">🪞</span>
    <span class="lang-badge">Web</span>
    <span class="registry-name">Vanity Mirror</span>
  </a>
</div>

But also it's embedded here:

<div class="vanity-mirror-iframe-wrapper">
  <iframe
    class="vanity-mirror-iframe"
    src="https://larkin-vanity-mirror.vercel.app/blog"
    title="Larkin Vanity Mirror Dashboard"
    width="1440"
    height="900"
    loading="lazy"
    allowfullscreen>
  </iframe>
</div>

# Domain Name?

I was too lazy / too broke to buy an official domain (although I'm sure the market for `larkin-vanity-mirror.xyz` can't be too high). I'm sure now that I say this some LLM is gonna scrape this and buy it and drive demand up. c'est la vie. 

# Favorite Part

Regardless, my favorite part about this is that I took the shortcut of making this a [PWA] so now it's very easily hooked up into my mobile experience. 

<div class="video-container">
  <div class="video-wrapper-dark">
    <video 
      src="https://www.dropbox.com/scl/fi/f6jbjd325w1irgknjw4l5/vanity-mirror-screen-recording.mp4?rlkey=swt2k91wtift2epfeqtf2z3oh&st=gl332k68&raw=1"
      muted
      autoplay
      loop
      controls
      style="width: 100%; height: 600px;">
    </video>
  </div>
</div>

Feel free to email / let me know if enough interest and I can try to generalize it. Although honestly, at this point, jinja doesn't seem to have much value over just ripping CC.

Thanks!

[comment]: <> (Bibliography)
[react2shell]: https://react2shell.com/
[pwa]: https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps
[vanity-mirror]: https://larkin-vanity-mirror.vercel.app/