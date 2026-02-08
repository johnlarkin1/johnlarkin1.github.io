---
title: iMessage Data Foundry
layout: post
featured-img: imessage-data-foundry
mathjax: false
python-interactive: false
categories: [Development, AI]
summary: Synthetic data generation in the exact sqlite3 schema needed for MacOS for iMessage / AddressBook compatibility
---

<div class="project-registry">
  <a href="https://github.com/johnlarkin1/imessage-data-foundry" target="_blank" rel="noopener" class="registry-card github">
    <span class="lang-icon">🐙</span>
    <span class="lang-badge">Source</span>
    <span class="registry-name">GitHub</span>
  </a>
  <a href="https://pypi.org/project/imessage-data-foundry/" target="_blank" rel="noopener" class="registry-card python">
    <span class="lang-icon">🐍</span>
    <span class="lang-badge">Python</span>
    <span class="registry-name">PyPI</span>
  </a>
</div>

# Context

Recently for one of my projects, I needed synthetic data generated to match a MacOS compatible iMessage `chat.db` as well as the `AddressBook.db`.

This is going to be a short post, because the [Github's][gh]`README.md` has a lot more information. So check that out. 

Alternatively, feel free to watch this demo video:

<div class="video-container">
  <div class="video-wrapper-dark">
    <iframe
      src="https://www.youtube.com/embed/t_6QnWvlkCI"
      title="iMessage Data Foundry Demo"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      allowfullscreen>
    </iframe>
  </div>
</div>

# Installation

There are many ways:

```
$ uv tool install imessage-data-foundry
$ uvx imessage-data-foundry
$ pip install imessage-data-foundry
$ pipx install imessage-data-foundry
``` 

# Conclusion

Thanks! Feel free to check out the GH repo or reach out if there are any questions / concerns. Also it's open source so feel free to submit issues / PRs. 

[comment]: <> (Bibliography)
[gh]: https://github.com/johnlarkin1/imessage-data-foundry