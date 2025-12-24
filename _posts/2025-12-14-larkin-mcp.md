---
title: "larkin-mcp"
layout: post
featured-img: larkin-mcp
mathjax: true
python-interactive: true
categories: [Development, A.I]
summary: Setting up a personalized MCP server. Feel free to install and interact with.
---

![larkin-mcp](/images/larkin-mcp/hero.png){: .center-shrink .lightbox-image}

<div class="image-caption">You can either interact with mine, or clone the repo <a href="https://github.com/johnlarkin1/yourname-mcp">here</a>, and get started with the second one.</div>
<br/>

<style>
.larkin-mcp-registry {
  --python-color: #50c878;
  --typescript-color: #00d4ff;
  --rust-color: #f79359;

  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin: 2rem auto;
  max-width: 720px;
}

.registry-card {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.25rem 1.5rem;
  min-width: 180px;
  border-radius: 12px;
  text-decoration: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.registry-card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
  padding: 1px;
  background: linear-gradient(135deg, var(--accent-color) 0%, transparent 50%, var(--accent-color) 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0.5;
  transition: opacity 0.3s ease;
}

.registry-card:hover::before {
  opacity: 1;
}

.registry-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 32px -8px var(--glow-color);
}

.registry-card.python {
  --accent-color: var(--python-color);
  --glow-color: rgba(80, 200, 120, 0.25);
  background: rgba(80, 200, 120, 0.08);
}

.registry-card.typescript {
  --accent-color: var(--typescript-color);
  --glow-color: rgba(0, 212, 255, 0.25);
  background: rgba(0, 212, 255, 0.08);
}

.registry-card.rust {
  --accent-color: var(--rust-color);
  --glow-color: rgba(247, 147, 89, 0.25);
  background: rgba(247, 147, 89, 0.08);
}

.registry-card .lang-icon {
  font-size: 1.75rem;
  margin-bottom: 0.5rem;
  filter: drop-shadow(0 2px 8px var(--glow-color));
}

.registry-card .lang-badge {
  font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.25rem 0.6rem;
  border-radius: 4px;
  margin-bottom: 0.6rem;
  background: rgba(255, 255, 255, 0.05);
  color: var(--accent-color);
  border: 1px solid var(--accent-color);
  opacity: 0.9;
}

.registry-card .registry-name {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.6);
  font-weight: 400;
  margin: 0;
}

.registry-card:hover .registry-name {
  color: rgba(255, 255, 255, 0.85);
}

@media (max-width: 600px) {
  .larkin-mcp-registry {
    gap: 0.75rem;
  }
  .registry-card {
    min-width: 140px;
    padding: 1rem 1.25rem;
  }
}

/* Template card */
.template-card-wrapper {
  display: flex;
  justify-content: center;
  margin: 0.75rem auto 1.5rem;
}

.template-card {
  --template-color: #a78bfa;
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.65rem 1.1rem;
  border-radius: 8px;
  text-decoration: none;
  background: rgba(167, 139, 250, 0.08);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.template-card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 8px;
  padding: 1px;
  background: linear-gradient(135deg, var(--template-color) 0%, transparent 50%, var(--template-color) 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0.4;
  transition: opacity 0.3s ease;
}

.template-card:hover::before {
  opacity: 1;
}

.template-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px -6px rgba(167, 139, 250, 0.3);
}

.template-card .template-icon {
  font-size: 1.1rem;
}

.template-card .template-text {
  font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--template-color);
  letter-spacing: 0.02em;
}

.template-card .template-arrow {
  font-size: 0.85rem;
  color: var(--template-color);
  opacity: 0.6;
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.template-card:hover .template-arrow {
  opacity: 1;
  transform: translateX(2px);
}
</style>

<div class="larkin-mcp-registry">
  <a href="https://pypi.org/project/larkin-mcp/" target="_blank" rel="noopener" class="registry-card python">
    <span class="lang-icon">🐍</span>
    <span class="lang-badge">Python</span>
    <span class="registry-name">PyPI</span>
  </a>
  <a href="https://www.npmjs.com/package/@johnlarkin1/larkin-mcp" target="_blank" rel="noopener" class="registry-card typescript">
    <span class="lang-icon">📘</span>
    <span class="lang-badge">TypeScript</span>
    <span class="registry-name">npm</span>
  </a>
  <a href="https://crates.io/crates/larkin-mcp" target="_blank" rel="noopener" class="registry-card rust">
    <span class="lang-icon">🦀</span>
    <span class="lang-badge">Rust</span>
    <span class="registry-name">crates.io</span>
  </a>
</div>

<div class="image-caption">Check out any of the links above for the various published packages. Note, Claude did the css here.</div>
<br/>
<div class="template-card-wrapper">
  <a href="https://github.com/johnlarkin1/yourname-mcp" target="_blank" rel="noopener" class="template-card">
    <span class="template-icon">📋</span>
    <span class="template-text">yourname-mcp template</span>
    <span class="template-arrow">→</span>
  </a>
</div>

<br/>

I'm working on a much bigger project, but honestly, needed to take a break from that. It has been a grind. I have burned many early mornings on that.

So as a break, I have wanted to explore building my own MCP server and templatizing this to make it easier for others to install and set this up as well. This is not going to be a long post, but I'm hoping the repos speak for themselves, and this provides ample motivation.

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Motivation](#motivation)
  - [Personal Insights](#personal-insights)
  - [Interactive Timeline](#interactive-timeline)
    - [Example 1:](#example-1)
    - [Example 2:](#example-2)
  - [Personalized Study Guide](#personalized-study-guide)
- [Context](#context)
- [Why?](#why)
- [`yourname-mcp`](#yourname-mcp)
  - [Demo](#demo)
  - [Security](#security)
  - [Rust](#rust)
- [Conclusion](#conclusion)

# Motivation

To provide some motivation (and perhaps earn a few stars on the template repo), here are practical examples of what you can do with this specific MCP server.

## Personal Insights

> What do you think was John Larkin's hardest tennis match?

**Result:**

![larkin-mcp](/images/larkin-mcp/hardest-match.png){: .center-shrink .lightbox-image}

**Rude**!! Hallucination. I didn't get _bageled_, I got _breadsticked_. In other words, it was 1-6 not 0-6. But yes, shoutout to Phillip Locklear...

## Interactive Timeline

### Example 1:

**Prompt:**

> Can you give me John's experience's as a beautiful timeline? Please create a html file with that visualization

**Result:** <a href="{{ '/assets/html/larkin-mcp/john-larkin-profile.html' | relative_url }}" target="_blank" rel="noopener">View the timeline <svg class="external-link-icon" width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-left:2px;"><path d="M10.5 1.5L5.5 6.5M10.5 1.5H7M10.5 1.5V5M10.5 7V10C10.5 10.2761 10.2761 10.5 10 10.5H2C1.72386 10.5 1.5 10.2761 1.5 10V2C1.5 1.72386 1.72386 1.5 2 1.5H5" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/></svg></a>

<div class="mcp-demo-iframe-wrapper">
  <iframe
    class="mcp-demo-iframe-container"
    src="{{ '/assets/html/larkin-mcp/john-larkin-profile.html' | relative_url }}"
    title="John Larkin Timeline"
    width="1120" 
    height="630"
    allowfullscreen>
  </iframe>
</div>
<br/>

### Example 2:

> Can you use your frontend-design skill and build a beautiful interactive timeline of John's work experience and personal project timeline as a single html file visualization?

**Result:** <a href="{{ '/assets/html/larkin-mcp/john-larkin-timeline.html' | relative_url }}" target="_blank" rel="noopener">View the timeline <svg class="external-link-icon" width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-left:2px;"><path d="M10.5 1.5L5.5 6.5M10.5 1.5H7M10.5 1.5V5M10.5 7V10C10.5 10.2761 10.2761 10.5 10 10.5H2C1.72386 10.5 1.5 10.2761 1.5 10V2C1.5 1.72386 1.72386 1.5 2 1.5H5" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/></svg></a>

<div class="mcp-demo-iframe-wrapper">
  <iframe
    class="mcp-demo-iframe-container"
    src="{{ '/assets/html/larkin-mcp/john-larkin-timeline.html' | relative_url }}"
    title="John Larkin Experience Timeline"
    width="1120" 
    height="630"
    allowfullscreen>
  </iframe>
</div>
<br/>

Honestly, the second one is pretty slick although it's a bit... vapid of personality I guess.

fwiw, here is the usage in CC:

![larkin-mcp](/images/larkin-mcp/claude-code-example.png){: .center-super-shrink .lightbox-image}

## Personalized Study Guide

**Prompt:**

> Can you help John Larkin prepare for an Anthropic interview given his resume and past experience? Please search and find open roles and then prepare a study guide for his various gaps.

**Result:**

Not sharing the whole thing, but you can see this from Claude Desktop:

![larkin-mcp](/images/larkin-mcp/claude-desktop-example.png){: .center-small .lightbox-image}

# Context

I wanted to set up a local MCP server that you can install to ask questions about the user. There are two versions:

- `larkin-mcp` - my materialized repo that has details about myself (largely professional, markdown files are online, but I'm guessing in the age of the internet, this level of detail is fine).
- `yourname-mcp` - the templated repo where you can clone this, run a script, and optionally publish (caution: the info that you put in your `resources/content` markdown files will then be indexable / probably ingested from some AI... but my theory is that most of that stuff is already going to be there)

# Why?

Yeah so this was something my PM girlfriend asked me almost immediately. Why do this? Can't you just feed your resume into ChatGPT and it'll basically be able to do the same? I think yes, partially, but (at least in my case), my resume is still missing a ton of context. So I think my response is mutli-fold:

> Can't you just feed your resume into ChatGPT and ask questions of that?

1. Feeding in your resume as a `pdf` or `md` file is going to bloat your context window. MCP provides more selective invocations.
1. I don't want to do that everytime I need something with my context and personality
1. It's still missing a ton of context about who I am and some more ephemeral things about me. (note: i know that 90% of companies won't care about that, and 99.9% of recruiters won't care about it)
1. I wanted to be able to distribute this. There's a world I could imagine where recruiters just run `uvx larkin-mcp` and then ask questions to get a feel for my work and who I am
1. I want to control the level of detail and insight that this MCP server has
1. I wanted to build an MCP server... I hadn't done it, even at work.
1. I wanted to explore the tooling around it as well.
1. I wanted to build an MCP server in Typescript and Rust explicitly, given I'm trying to work on my Rust skills and I'm less involved in those communities
1. I thought it would be a useful thing to templatize and set up some infrastructure so less technical users could `git clone <repo> && ./run-install.sh` and that would ask them a couple of questions, analyze their resume, convert it into markdown, they could write some markdown to provide more context, and then boom, they could also publish it and others could use it if they wanted.
1. As stated previously, I needed a break from my other project.

And if you're thinking like _well, what about Claude memory or ChatGPT memory?_, I'm really not a fan of that. I don't think Simon Willison is either. And I don't trust it to not sycophant it up or pull information that perhaps I don't want for the questions I'm asking.

Hopefully, that's enough rationale for personal motivation.

# `yourname-mcp`

This is hopefully your template of interest. The point is that this has enough scaffolding that you can run the install script, populate a couple markdown files, upload to PYPI and then you're off and running. There will be more info in the actuall repo [here][yourname-mcp].

## Demo

Here is a demo showcasing the functionality:

<div class="video-container">
  <div class="video-wrapper-dark">
    <video 
      src="https://www.dropbox.com/scl/fi/v7ljkkxf3p8d24vk8wlpk/yourname-mcp-demo-lg.mp4?rlkey=95ha9lg6gpwngufkdvq9l2t0q&st=dlhyft2u&raw=1"
      muted
      autoplay
      loop
      controls
      style="width: 100%; height: auto;">
    </video>
  </div>
</div>

## Security

I - like basically every other engineer - am slightly cautious about MCP. There are going to be large amounts of attacks given the trust people are placing into MCP and utilizing binary executables (i.e. `bunx` or `uvx`).

This is from 6 days ago (at time of writing):

<blockquote class="reddit-embed-bq" style="height:316px" data-embed-theme="dark" data-embed-height="316"><a href="https://www.reddit.com/r/MCPservers/comments/1poelh4/is_anyone_else_terrified_by_the_lack_of_security/">Is anyone else terrified by the lack of security in standard MCP?</a><br> by<a href="https://www.reddit.com/user/RaceInteresting3814/">u/RaceInteresting3814</a> in<a href="https://www.reddit.com/r/MCPservers/">MCPservers</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>
<br/>

Even with this project... while I utilize `uvx` and `bunx` for the convenience, I am 100% afraid about impersonations, security attacks, people injecting malicious code from poor distributors. This is obviously nuanced. I am a huge fan of making software easily disseminated but the increase in malicious code and actors (that are only exacerbated from the AI wave) is extremely alarming. I mean just look at npm in the [past][npm-attack-1] [couple][npm-attack-2] [months][npm-attack-3]?

## Rust

I could have used something like [`cargo-binstall`][cargo-binstall], but didn't quite get to it. As a result, if you want to set this up in Claude Code or Claude Desktop, you'll need to do something like `cargo install larkin-mcp` and then point to that corresponding built binary:

```shell
   Compiling larkin-mcp v1.0.2
    Finished `release` profile [optimized] target(s) in 14.83s
  Installing /Users/johnlarkin/.cargo/bin/larkin-mcp
   Installed package `larkin-mcp v1.0.2` (executable `larkin-mcp`)
```

Rust was my favorite to implement, although the code structure is perhaps not as Rust idiomatic as it should be. In my opinion, `rmcp` which is the canonical framework for Rust MCP servers is slightly less ergonomic. They match a lot of the Python decorators in terms of Rust macros but there's some tricks about public traits and understanding what is actually going on given the function calls.

# Conclusion

If you like this, or think it will be useful, please check out the basically templated repo `yourname-mcp` where the `README.md` will walk you through what you need to do! Always feel free to email or leave comments if need be.

[comment]: <> (Bibliography)
[npm-attack-1]: https://semgrep.dev/blog/2025/chalk-debug-and-color-on-npm-compromised-in-new-supply-chain-attack/
[npm-attack-2]: https://securitylabs.datadoghq.com/articles/shai-hulud-2.0-npm-worm/
[npm-attack-3]: https://www.crowdstrike.com/en-us/blog/crowdstrike-falcon-prevents-npm-package-supply-chain-attacks/?utm_source=chatgpt.com
[cargo-binstall]: https://crates.io/crates/cargo-binstall
[yourname-mcp]: https://github.com/johnlarkin1/yourname-mcp
