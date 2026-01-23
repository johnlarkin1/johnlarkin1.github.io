/**
 * @module SearchModal
 * @description Search modal with keyword, semantic, and hybrid search modes
 *
 * @requires jquery - Global jQuery ($)
 * @requires ../core/utils - trapFocus utility
 * @requires ../core/state - search state
 *
 * @exports {Function} init - Initialize search modal
 * @exports {Function} open - Open search modal
 * @exports {Function} close - Close search modal
 */

var Utils = require('../core/utils');
var State = require('../core/state');

var SearchModal = (function () {
  'use strict';

  var _isInitialized = false;
  var vectorSearchDebounceTimer = null;
  var hybridSearchDebounceTimer = null;

  function init() {
    if (_isInitialized) return;

    var $modal = $('.search-modal');
    var $trigger = $('.js-search-trigger');
    var $close = $('.js-search-close');
    var $modeToggle = $('.search-mode-toggle__btn');

    if ($modal.length === 0) return;

    // Click trigger to open
    $trigger.on('click', function (e) {
      e.preventDefault();
      open();
    });

    // Click close button or overlay to close
    $close.on('click', function (e) {
      e.preventDefault();
      close();
    });

    // Mode toggle buttons
    $modeToggle.on('click', function (e) {
      e.preventDefault();
      var mode = $(this).data('mode');
      switchSearchMode(mode);
    });

    // Vector search input handler
    var $vectorInput = $('.vector-search__input');
    $vectorInput.on('input', function () {
      clearTimeout(vectorSearchDebounceTimer);
      var query = $(this).val();

      vectorSearchDebounceTimer = setTimeout(function () {
        performVectorSearch(query);
      }, 300);
    });

    // Handle Enter key in vector search input
    $vectorInput.on('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        clearTimeout(vectorSearchDebounceTimer);
        performVectorSearch($(this).val());
      }
    });

    // Hybrid search input handler
    var $hybridInput = $('.hybrid-search__input');
    $hybridInput.on('input', function () {
      clearTimeout(hybridSearchDebounceTimer);
      var query = $(this).val();

      hybridSearchDebounceTimer = setTimeout(function () {
        performHybridSearch(query);
      }, 300);
    });

    // Handle Enter key in hybrid search input
    $hybridInput.on('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        clearTimeout(hybridSearchDebounceTimer);
        performHybridSearch($(this).val());
      }
    });

    // Keyboard shortcut: Cmd+K (Mac) / Ctrl+K (Win/Linux)
    $(document).on('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if ($modal.hasClass('is-active')) {
          close();
        } else {
          open();
        }
      }
    });

    _isInitialized = true;
  }

  function switchSearchMode(mode, forceInit) {
    var currentMode = State.getSearchMode();
    if (mode === currentMode && !forceInit) return;

    State.setSearchMode(mode);

    // Update toggle buttons
    $('.search-mode-toggle__btn').each(function () {
      var $btn = $(this);
      var isActive = $btn.data('mode') === mode;
      $btn.toggleClass('search-mode-toggle__btn--active', isActive);
      $btn.attr('aria-checked', isActive ? 'true' : 'false');
    });

    // Switch visible content
    if (mode === 'keyword') {
      $('#pagefind-container').show();
      $('#vector-search-container').hide();
      $('#hybrid-search-container').hide();

      // Focus Pagefind input
      setTimeout(function () {
        $('.pagefind-ui__search-input').focus();
      }, 100);
    } else if (mode === 'semantic') {
      $('#pagefind-container').hide();
      $('#vector-search-container').show();
      $('#hybrid-search-container').hide();

      // Initialize vector search if needed
      initVectorSearch();
    } else if (mode === 'hybrid') {
      $('#pagefind-container').hide();
      $('#vector-search-container').hide();
      $('#hybrid-search-container').show();

      // Initialize hybrid search
      initHybridSearch();
    }
  }

  function initVectorSearch() {
    if (window.VectorSearch.isAvailable()) {
      $('.vector-search__loading').hide();
      $('.vector-search__input-wrapper').show();
      $('.vector-search__input').focus();
      return;
    }

    if (window.VectorSearch.isLoading()) {
      return;
    }

    // Show loading state
    $('.vector-search__loading').show();
    $('.vector-search__input-wrapper').hide();
    $('.vector-search__results').empty();
    $('.vector-search__empty').hide();

    // Initialize with progress callback
    window.VectorSearch.initialize(function (info) {
      $('.vector-search__loading-message').text(info.message);
      $('.vector-search__loading-progress').css('width', info.progress + '%');

      if (info.stage === 'ready') {
        $('.vector-search__loading').hide();
        $('.vector-search__input-wrapper').show();
        $('.vector-search__input').focus();
      } else if (info.stage === 'error') {
        $('.vector-search__loading-message').text(
          'Failed to load: ' + info.message
        );
        $('.vector-search__loading-spinner').hide();
      }
    });
  }

  function performVectorSearch(query) {
    if (!window.VectorSearch.isAvailable()) return;

    var $results = $('.vector-search__results');
    var $empty = $('.vector-search__empty');

    if (!query || query.trim().length < 2) {
      $results.empty();
      $empty.hide();
      return;
    }

    window.VectorSearch.search(query, 10, 0.3)
      .then(function (results) {
        renderVectorResults(results);
      })
      .catch(function (err) {
        console.error('Vector search error:', err);
        $results.empty();
        $empty.show().find('p').text('Search error: ' + err.message);
      });
  }

  function renderVectorResults(results) {
    var $container = $('.vector-search__results');
    var $empty = $('.vector-search__empty');

    $container.empty();

    if (results.length === 0) {
      $empty.show();
      return;
    }

    $empty.hide();

    results.forEach(function (result, index) {
      var chunk = result.chunk;
      var similarity = (result.similarity * 100).toFixed(0);

      var $result = $(
        '<a class="vector-search__result" href="' + chunk.url + '"></a>'
      );
      $result.css('animation-delay', index * 0.04 + 's');

      // Header row: title + score
      var $header = $('<div class="vector-search__result-header"></div>');
      var $title = $(
        '<span class="vector-search__result-title"></span>'
      ).text(chunk.title);

      // Score with visual indicator
      var $score = $('<div class="vector-search__result-score"></div>');
      var scoreClass =
        similarity >= 70 ? 'high' : similarity >= 50 ? 'medium' : 'low';
      $score.addClass('vector-search__result-score--' + scoreClass);
      $score.html(
        '<span class="score-value">' +
          similarity +
          '</span><span class="score-percent">%</span>'
      );

      $header.append($title, $score);

      // Meta row: section tag
      var $meta = $('<div class="vector-search__result-meta"></div>');
      if (chunk.section && chunk.section !== chunk.title) {
        var $section = $('<span class="vector-search__result-section"></span>');
        $section.html(
          '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h7"/></svg>' +
            chunk.section
        );
        $meta.append($section);
      }

      // Excerpt with truncation
      var excerptText =
        chunk.text.length > 150
          ? chunk.text.substring(0, 150) + '...'
          : chunk.text;
      var $excerpt = $(
        '<p class="vector-search__result-excerpt"></p>'
      ).text(excerptText);

      // Categories footer
      var $footer = $('<div class="vector-search__result-footer"></div>');
      if (chunk.categories && chunk.categories.length > 0) {
        var $categories = $(
          '<div class="vector-search__result-categories"></div>'
        );
        chunk.categories.slice(0, 3).forEach(function (cat) {
          $categories.append(
            $('<span class="vector-search__result-category"></span>').text(cat)
          );
        });
        $footer.append($categories);
      }

      $result.append($header, $meta, $excerpt, $footer);
      $container.append($result);
    });
  }

  function initHybridSearch() {
    var $status = $('.hybrid-search__status');
    var $inputWrapper = $('.hybrid-search__input').closest(
      '.vector-search__input-wrapper'
    );
    var $loading = $('.hybrid-search__loading');

    if (window.VectorSearch.isAvailable()) {
      $loading.hide();
      $inputWrapper.show();
      $status.hide();
      $('.hybrid-search__input').focus();
      return;
    }

    // Always show input immediately
    $loading.hide();
    $inputWrapper.show();
    $('.hybrid-search__results').empty();
    $('.hybrid-search__empty').hide();
    $('.hybrid-search__input').focus();

    // Show status badge
    $status
      .html(
        '<span class="hybrid-search__status-spinner"></span> Loading semantic model...'
      )
      .show();

    if (window.VectorSearch.isLoading()) {
      return;
    }

    // Initialize with progress callback
    window.VectorSearch.initialize(function (info) {
      if (info.stage === 'ready') {
        $status.hide();
      } else if (info.stage === 'error') {
        $status.html('Keyword only (semantic unavailable)').show();
      }
    });
  }

  function loadPagefindAPI() {
    if (window.pagefindAPI) {
      return Promise.resolve(window.pagefindAPI);
    }

    return import('/pagefind/pagefind.js')
      .then(function (pagefind) {
        return pagefind.init().then(function () {
          window.pagefindAPI = pagefind;
          return pagefind;
        });
      })
      .catch(function (err) {
        console.error('[HybridSearch] Failed to load Pagefind:', err);
        return null;
      });
  }

  function performHybridSearch(query) {
    var $results = $('.hybrid-search__results');
    var $empty = $('.hybrid-search__empty');

    if (!query || query.trim().length < 2) {
      $results.empty();
      $empty.hide();
      return;
    }

    // Run both searches in parallel
    var keywordPromise = loadPagefindAPI().then(function (pagefind) {
      if (!pagefind) {
        return [];
      }

      return pagefind
        .search(query)
        .then(function (searchResults) {
          if (!searchResults || !searchResults.results) {
            return [];
          }

          var promises = searchResults.results
            .slice(0, 10)
            .map(function (result, index) {
              return result.data().then(function (data) {
                return {
                  url: data.url,
                  title:
                    data.meta && data.meta.title ? data.meta.title : 'Untitled',
                  excerpt: data.excerpt || '',
                  rank: index + 1,
                  source: 'keyword'
                };
              });
            });

          return Promise.all(promises);
        })
        .catch(function () {
          return [];
        });
    });

    var semanticPromise = new Promise(function (resolve) {
      if (!window.VectorSearch.isAvailable()) {
        resolve([]);
        return;
      }

      window.VectorSearch.search(query, 10, 0.25)
        .then(function (results) {
          resolve(
            results.map(function (result, index) {
              return {
                url: result.chunk.url,
                title: result.chunk.title,
                section: result.chunk.section,
                excerpt: result.chunk.text,
                categories: result.chunk.categories,
                similarity: result.similarity,
                rank: index + 1,
                source: 'semantic'
              };
            })
          );
        })
        .catch(function () {
          resolve([]);
        });
    });

    // Combine results using Reciprocal Rank Fusion
    Promise.all([keywordPromise, semanticPromise]).then(function (allResults) {
      var keywordResults = allResults[0];
      var semanticResults = allResults[1];

      var combined = combineResultsRRF(keywordResults, semanticResults);
      renderHybridResults(combined);
    });
  }

  function combineResultsRRF(keywordResults, semanticResults) {
    var k = 60; // RRF constant
    var scores = {};
    var dataByUrl = {};

    // Score keyword results
    keywordResults.forEach(function (result) {
      var url = result.url;
      if (!scores[url]) scores[url] = 0;
      scores[url] += 1 / (k + result.rank);

      if (!dataByUrl[url]) {
        dataByUrl[url] = {
          url: url,
          title: result.title,
          excerpt: result.excerpt,
          sources: []
        };
      }
      dataByUrl[url].sources.push('keyword');
      dataByUrl[url].keywordRank = result.rank;
    });

    // Score semantic results
    semanticResults.forEach(function (result) {
      var url = result.url;
      if (!scores[url]) scores[url] = 0;
      scores[url] += 1 / (k + result.rank);

      if (!dataByUrl[url]) {
        dataByUrl[url] = {
          url: url,
          title: result.title,
          excerpt: result.excerpt,
          sources: []
        };
      }
      dataByUrl[url].sources.push('semantic');
      dataByUrl[url].semanticRank = result.rank;
      dataByUrl[url].similarity = result.similarity;
      dataByUrl[url].section = result.section;
      dataByUrl[url].categories = result.categories;

      if (result.excerpt && result.excerpt.length > 0) {
        dataByUrl[url].excerpt = result.excerpt;
      }
    });

    // Convert to sorted array
    var combined = Object.keys(scores).map(function (url) {
      var data = dataByUrl[url];
      data.rrfScore = scores[url];
      return data;
    });

    combined.sort(function (a, b) {
      return b.rrfScore - a.rrfScore;
    });

    return combined.slice(0, 10);
  }

  function renderHybridResults(results) {
    var $container = $('.hybrid-search__results');
    var $empty = $('.hybrid-search__empty');

    $container.empty();

    if (results.length === 0) {
      $empty.show();
      return;
    }

    $empty.hide();

    results.forEach(function (result, index) {
      var displayScore = Math.min(99, Math.round(result.rrfScore * 3000));

      var $result = $(
        '<a class="vector-search__result" href="' + result.url + '"></a>'
      );
      $result.css('animation-delay', index * 0.04 + 's');

      // Header row
      var $header = $('<div class="vector-search__result-header"></div>');
      var $title = $(
        '<span class="vector-search__result-title"></span>'
      ).text(result.title);

      var $score = $('<div class="vector-search__result-score"></div>');
      var scoreClass =
        displayScore >= 70 ? 'high' : displayScore >= 50 ? 'medium' : 'low';
      $score.addClass('vector-search__result-score--' + scoreClass);
      $score.html(
        '<span class="score-value">' +
          displayScore +
          '</span><span class="score-percent">%</span>'
      );

      $header.append($title, $score);

      // Meta row
      var $meta = $('<div class="vector-search__result-meta"></div>');

      if (result.section && result.section !== result.title) {
        var $section = $('<span class="vector-search__result-section"></span>');
        $section.html(
          '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h7"/></svg>' +
            result.section
        );
        $meta.append($section);
      }

      // Source badges
      var $sources = $('<div class="hybrid-search__sources"></div>');
      if (result.sources.indexOf('keyword') !== -1) {
        $sources.append(
          '<span class="hybrid-search__source hybrid-search__source--keyword">K</span>'
        );
      }
      if (result.sources.indexOf('semantic') !== -1) {
        $sources.append(
          '<span class="hybrid-search__source hybrid-search__source--semantic">S</span>'
        );
      }
      $meta.append($sources);

      // Excerpt
      var excerptText = result.excerpt || '';
      excerptText = excerptText.replace(/<[^>]*>/g, '');
      excerptText =
        excerptText.length > 150
          ? excerptText.substring(0, 150) + '...'
          : excerptText;
      var $excerpt = $(
        '<p class="vector-search__result-excerpt"></p>'
      ).text(excerptText);

      // Categories footer
      var $footer = $('<div class="vector-search__result-footer"></div>');
      if (result.categories && result.categories.length > 0) {
        var $categories = $(
          '<div class="vector-search__result-categories"></div>'
        );
        result.categories.slice(0, 3).forEach(function (cat) {
          $categories.append(
            $('<span class="vector-search__result-category"></span>').text(cat)
          );
        });
        $footer.append($categories);
      }

      $result.append($header, $meta, $excerpt, $footer);
      $container.append($result);
    });
  }

  function open() {
    var $modal = $('.search-modal');
    if ($modal.length === 0 || $modal.hasClass('is-active')) return;

    // Prevent body scroll
    $('body').addClass('search-modal--open');

    // Show modal
    $modal.addClass('is-active');

    // Initialize Pagefind UI on first open
    if (!window.pagefindInitialized && typeof PagefindUI !== 'undefined') {
      new PagefindUI({
        element: '#pagefind-container',
        showSubResults: true,
        showImages: false,
        excerptLength: 20
      });
      window.pagefindInitialized = true;
    }

    // Ensure correct search mode is active
    switchSearchMode(State.getSearchMode(), true);

    // Trap focus within modal
    Utils.trapFocus($modal[0]);
  }

  function close() {
    var $modal = $('.search-modal');
    if (!$modal.hasClass('is-active')) return;

    // Add closing animation class
    $modal.addClass('is-closing');

    // Restore body scroll
    $('body').removeClass('search-modal--open');

    // Remove classes after animation
    setTimeout(function () {
      $modal.removeClass('is-active is-closing');

      // Clean up focus trap
      if ($modal[0]._trapFocusCleanup) {
        $modal[0]._trapFocusCleanup();
      }
    }, 250);
  }

  return {
    init: init,
    open: open,
    close: close
  };
})();

module.exports = SearchModal;
