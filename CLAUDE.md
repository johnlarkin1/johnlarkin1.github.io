# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Jekyll-based personal blog and portfolio website for John Larkin. The site uses the jekyll-sleek theme and is hosted on GitHub Pages at https://johnlarkin1.github.io.

## Architecture

The site follows standard Jekyll conventions with a custom build pipeline that combines Ruby/Jekyll with Node.js tools:

- **Content**: Blog posts in `_posts/` using Markdown with YAML front matter
- **Layouts**: HTML templates in `_layouts/`
- **Styling**: SCSS files in `_sass/`, compiled to CSS
- **JavaScript**: Source in `_js/`, bundled with Browserify
- **Images**: Source images in `_img/`, optimized versions in `assets/img/`
- **Generated Site**: Built to `_site/` directory

## Common Commands

> [!IMPORTANT]  
> **You are not to write a single word of any post's content. You are solely meant to help with administrative tasks like project organization, new css styles, interactive demos, etc.**

### Local Development

```bash
# Install dependencies
bundle install
npm install

# Build and serve locally with live reload
npm start

# Just build the site
npm run build

# Just serve the built site
npm run serve
```

### Jekyll Commands

```bash
# Build the Jekyll site
bundle exec jekyll build

# Serve Jekyll locally (without npm pipeline)
bundle exec jekyll serve
```

### Individual Build Tasks

```bash
# Compile SCSS to CSS
npm run sass

# Bundle JavaScript
npm run js

# Optimize images
npm run img

# Generate critical CSS
npm run critical-css
```

### Creating New Posts

Blog posts go in `_posts/` with filename format `YYYY-MM-DD-title.md`. Include YAML front matter:

```yaml
---
layout: post
title: "Post Title"
summary: "Brief description"
featured-img: image-name # Without extension, from assets/img/posts/
categories: [Category1, Category2]
---
```

### Image Processing

The site uses Sharp to generate multiple sizes for responsive images. Place source images in `_img/posts/` and run `npm run img` to generate optimized versions.

### Special Features

1. **Encrypted Posts**: Some posts support encryption using `staticrypt`. Related files: `gulp_encrypt.js`
2. **Google Tag Manager**: Configured in `_config.yml` with ID `GTM-KDHGNV6`
3. **Service Worker**: Currently disabled but configuration exists in `workbox-config.js`

## Key Files

- `_config.yml`: Main Jekyll configuration
- `package.json`: Node.js dependencies and build scripts
- `jekyll-sleek.gemspec`: Theme specification
- `build_js.js`, `generate_sass.js`, `sharp_img.js`: Custom build scripts

## Development Notes

- The site uses Jekyll 3.9.2 (not Jekyll 4)
- Images should be optimized before committing
- Draft posts can be placed in `_posts/_hidden/` or `_posts/_in_progress/`
- The main branch is `main` for pull requests

> [!IMPORTANT]  
> **You are not to write a single word of any post's content. You are solely meant to help with administrative tasks like project organization, new css styles, interactive demos, etc.**
