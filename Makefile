.PHONY: build-css build-img build-js build serve serve-static clean jekyll pagefind

# i need to update this
build-css:
	node generate_sass.js

# build jank img
build-img:
	node sharp_img.js

# build js
build-js:
	node build_js.js --no-serve

# the whole kit and kabootle
jekyll:
	bundle exec jekyll build

# pagefind so that users can actually search (req: _site is built)
pagefind:
	npx pagefind --site _site

# quick build
build-quick: jekyll pagefind
	@echo "Build complete! _site/ ready for deployment"

# full build
build-full: build-css build-img build-js jekyll pagefind
	@echo "Build complete! _site/ ready for deployment"

# local dev server (rebuilds, no search)
serve:
	bundle exec jekyll serve

# serve pre-built _site/ with search (run 'make build' first)
serve-static:
	@echo "Serving _site/ at http://localhost:4000 (Ctrl+C to stop)"
	@cd _site && python3 -m http.server 4000

# clean
clean:
	rm -rf _site
