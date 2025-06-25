---
layout: post
title: "GitHub Repository Embed Demo"
summary: "Demonstrating the new GitHub repository card component"
featured-img: shane-rounce-205187
categories: [Development]
---

This post demonstrates how to embed GitHub repositories using our new visually stunning repository card component.

## Static Version (Recommended)

For better performance and reliability with Jekyll's build process, use the static version:

```liquid
{% raw %}{% include github-repo-card-static.html 
  repo="johnlarkin1/sportradar-tennis-v3"
  name="sportradar-tennis-v3"
  description="Python wrapper for Sportradar Tennis API v3 with support for live matches, player stats, and tournament data"
  language="Python"
  stars="12"
  forks="3"
  topics="tennis,sportradar,api,python,sports-data"
%}{% endraw %}
```

### Live Example

Here's how it looks with your tennis repository:

{% include github-repo-card-static.html 
  repo="johnlarkin1/sportradar-tennis-v3"
  name="sportradar-tennis-v3"
  description="Python wrapper for Sportradar Tennis API v3 with support for live matches, player stats, and tournament data"
  language="Python"
  stars="12"
  forks="3"
  topics="tennis,sportradar,api,python,sports-data"
%}

## Multiple Static Examples

You can embed multiple repositories in a single post:

{% include github-repo-card-static.html 
  repo="facebook/react"
  name="react"
  description="The library for web and native user interfaces"
  language="JavaScript"
  stars="220000"
  forks="45000"
  topics="react,javascript,library,frontend,ui"
%}

{% include github-repo-card-static.html 
  repo="vercel/next.js"
  name="next.js"
  description="The React Framework"
  language="JavaScript"
  stars="115000"
  forks="25000"
  topics="nextjs,react,javascript,vercel,framework"
%}

## Dynamic Version (Alternative)

If you prefer to fetch repository data dynamically at runtime:

```liquid
{% raw %}{% include github-repo-card.html repo="johnlarkin1/sportradar-tennis-v3" %}{% endraw %}
```

{% include github-repo-card.html repo="johnlarkin1/sportradar-tennis-v3" %}

## Features

The GitHub repository card includes:
- Repository name and description
- Primary programming language with color indicator
- Star and fork counts
- Repository topics/tags
- Hover effects for enhanced interactivity
- Beautiful GitHub-styled design

### Static vs Dynamic

- **Static version**: Data is provided at build time, loads instantly, no API rate limits
- **Dynamic version**: Fetches latest data from GitHub API, may hit rate limits, requires JavaScript

Both versions use the same CSS styling for a consistent, visually appealing appearance that matches GitHub's design language.