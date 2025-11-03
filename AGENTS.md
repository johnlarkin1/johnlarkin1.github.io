# Repository Guidelines

## Project Structure & Module Organization
This site is a Jekyll build that keeps editable sources in the root and emits static assets to `_site/`.
Blog content lives in `_posts/` using the `YYYY-MM-DD-title.md` pattern; layouts and partials sit under `_layouts/` and `_includes/`, while Sass sources live in `_sass/` with the main entry `_sass/jekyll-sleek.scss`.
Raw JavaScript modules live in `_js/`, images in `_img/`, and processed assets are committed to `assets/`; never edit `_site/` or `assets/js/bundle.js` by hand because the npm pipeline overwrites them.

## Build, Test, and Development Commands
Install dependencies with `bundle install` for Ruby gems and `npm install` for the asset toolchain.
Run `npm run build` to clean `_site/`, compile Sass, bundle JavaScript, optimize media, execute `jekyll build`, inline critical CSS, and refresh the service worker manifest.
For live development, use `npm run start`, which builds once and then runs BrowserSync plus file watchers; `bundle exec jekyll serve --livereload` is a lighter alternative when you only need the Jekyll server.

## Coding Style & Naming Conventions
Follow two-space indentation in Liquid templates, HTML, and SCSS, mirroring the existing layouts, and prefer BEM-style class names such as `hero__title`.
New posts should include YAML front matter fields used elsewhere (`title`, `date`, `description`, `pinned`, etc.), and assets referenced in posts go under `assets/img/posts/`.
JavaScript follows ES2015 modules transpiled via Browserify/Babel; lint with `npx eslint _js` if you touch scripts, and keep filenames kebab-cased.

## Testing Guidelines
No automated test suite exists, so treat `npm run build` as the regression gate: it fails on Sass, JS, or Jekyll errors.
After each change, run `npm run start` and confirm pages render, navigation anchors scroll correctly, and lazyloaded images resolve.
For SEO-sensitive updates, regenerate `critical-css` with `npm run critical-css` and spot-check the generated `_includes/critical.css`.

## Commit & Pull Request Guidelines
Past commits use the pattern `[type] short description` (for example `[fix] horizontal scrolling in pinned section (#75)`); keep following that style and reference issue numbers in parentheses.
Each PR should summarize user-facing changes, list the verification commands you ran, and attach before/after screenshots for visual tweaks.
Do not commit `_site/` outputs when collaborating; push only source changes and let GitHub Pages build from them.

## Configuration Tips
Site-wide settings reside in `_config.yml`; keep production URLs accurate to avoid broken asset links.
Store API keys or tokens using environment variables referenced via `JEKYLL_ENV=production` rather than committing secrets, and validate service worker settings in `workbox-config.js` before regenerating.
