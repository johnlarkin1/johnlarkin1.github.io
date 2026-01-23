# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Jekyll-based personal blog and portfolio website for John Larkin. The site uses the jekyll-sleek theme and is hosted on GitHub Pages at https://johnlarkin1.github.io.

## Architecture

The site follows standard Jekyll conventions with a custom build pipeline that combines Ruby/Jekyll with Node.js tools:

- **Content**: Blog posts in `_posts/` using Markdown with YAML front matter
- **Layouts**: HTML templates in `_layouts/` (default, post, page, compress)
- **Includes**: Reusable components in `_includes/` (header, footer, modals, cards, etc.)
- **Styling**: SCSS files in `_sass/`, compiled to `assets/css/main.css`
- **JavaScript**: Source in `_js/scripts.js`, bundled with Browserify to `assets/js/bundle.js`
- **Images**: Source images in `_img/posts/`, optimized versions in `assets/img/posts/`
- **Generated Site**: Built to `_site/` directory

## Common Commands

> [!IMPORTANT]
> **You are not to write a single word of any post's content. You are solely meant to help with administrative tasks like project organization, new css styles, interactive demos, etc.**

### Local Development

```bash
# Build the Jekyll site
bundle exec jekyll build

# Serve locally with live reload
bundle exec jekyll serve

# Compile SCSS to CSS
node generate_sass.js

# Bundle JavaScript (starts browser-sync by default)
node build_js.js

# Bundle JavaScript without starting server
node build_js.js --no-serve

# Optimize images (generates multiple sizes for responsive images)
node sharp_img.js
```

### Post Management

Blog posts go in `_posts/` with filename format `YYYY-MM-DD-title.md`. Required YAML front matter:

```yaml
---
layout: post
title: "Post Title"
featured-img: image-name  # Without extension, from assets/img/posts/
categories: [Category1, Category2]
---
```

Optional front matter fields:
- `summary`: Brief description for meta tags
- `mathjax: true`: Enable LaTeX math rendering
- `toc: true`: Enable table of contents

Draft posts can be placed in:
- `_posts/_hidden/` - Hidden drafts
- `_posts/_in_progress/` - Work in progress

### Image Processing

Place source images (`.jpg`, `.png`) in `_img/posts/` and run `node sharp_img.js`. This generates 8 sizes for responsive loading:
- `_placehold` (230px), `_thumb` (535px), `_thumb@2x` (1070px)
- `_xs` (575px), `_sm` (767px), `_md` (991px), `_lg` (1999px)
- Full size (1920px max-width)

### Encrypting Posts

```bash
node gulp_encrypt.js
# Prompts for: markdown filename and passphrase
```

## Key Files

- `_config.yml` - Jekyll configuration, navigation, social links
- `jekyll-sleek.gemspec` - Theme specification (Jekyll 3.9.2)
- `_sass/jekyll-sleek.scss` - Main SCSS entry point
- `_js/scripts.js` - Main JavaScript entry point
- `_js/vector-search.js` - Semantic search functionality
- `_js/pyodide-runner.js` - In-browser Python execution

## JavaScript Features

The site includes several interactive features initialized in `_js/scripts.js`:
- Mobile navigation with accordion dropdowns
- Lightbox for images
- Mermaid diagram rendering with pan/zoom
- Table of contents generation
- Code toggle tabs
- Knowledge check quizzes
- Pinned posts carousel
- Contact hub modal
- Hybrid search (keyword + semantic via vector embeddings)
- GitHub repository cards (dynamic API fetching)
- Keyboard shortcuts (vim-style: `g+h` home, `g+c` categories, `/` search, `?` help)

## SCSS Organization

```
_sass/
├── jekyll-sleek.scss    # Main entry (imports all partials)
├── abstracts/           # Variables, mixins
├── base/                # Base styles, typography, helpers
├── components/          # UI components (buttons, cards, modals, search, etc.)
├── layout/              # Grid, nav, footer
├── pages/               # Page and post specific styles
└── vendor/              # Normalize.css
```

## Development Notes

- Uses Jekyll 3.9.2 (not Jekyll 4)
- Main branch is `main` for pull requests
- Service worker is disabled (`service_worker: false` in `_config.yml`)
- The site uses `jemoji` for emoji support in posts

> [!IMPORTANT]
> **You are not to write a single word of any post's content. You are solely meant to help with administrative tasks like project organization, new css styles, interactive demos, etc.**
