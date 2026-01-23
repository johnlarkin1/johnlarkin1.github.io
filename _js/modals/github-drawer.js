/**
 * @module GitHubDrawer
 * @description GitHub repositories drawer modal
 *
 * @requires jquery - Global jQuery ($)
 * @requires ../core/utils - trapFocus utility
 * @requires ../core/constants - FAVORITE_REPOS, LANGUAGE_COLORS
 * @requires ../core/state - repos cache
 *
 * @exports {Function} init - Initialize GitHub drawer
 * @exports {Function} open - Open GitHub drawer
 * @exports {Function} close - Close GitHub drawer
 */

var Utils = require('../core/utils');
var Constants = require('../core/constants');
var State = require('../core/state');

var GitHubDrawer = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $drawer = $('.gh-repos-drawer');
    var $trigger = $('.js-gh-repos-trigger');
    var $close = $('.js-gh-repos-drawer-close');
    var $retry = $('.js-gh-repos-retry');

    if ($drawer.length === 0) return;

    // Open drawer on trigger click
    $trigger.on('click', function (e) {
      e.preventDefault();
      e.stopPropagation(); // Prevent dropdown from closing
      open();
    });

    // Close drawer
    $close.on('click', function (e) {
      e.preventDefault();
      close();
    });

    // Retry button
    $retry.on('click', function (e) {
      e.preventDefault();
      fetchGitHubRepos(true); // Force refresh
    });

    _isInitialized = true;
  }

  function open() {
    var $drawer = $('.gh-repos-drawer');
    if ($drawer.length === 0 || $drawer.hasClass('is-active')) return;

    // Prevent body scroll
    $('body').addClass('gh-repos-drawer--open');

    // Show drawer
    $drawer.addClass('is-active');

    // Fetch repos if needed
    fetchGitHubRepos();

    // Set up focus trapping
    Utils.trapFocus($drawer[0]);
  }

  function close() {
    var $drawer = $('.gh-repos-drawer');
    if (!$drawer.hasClass('is-active')) return;

    // Add closing class for exit animation
    $drawer.addClass('is-closing');

    // Wait for animation then hide
    setTimeout(function () {
      $drawer.removeClass('is-active is-closing');
      $('body').removeClass('gh-repos-drawer--open');

      // Clean up focus trap
      if ($drawer[0]._trapFocusCleanup) {
        $drawer[0]._trapFocusCleanup();
      }

      // Return focus to trigger
      $('.js-gh-repos-trigger').first().focus();
    }, 250);
  }

  function fetchGitHubRepos(forceRefresh) {
    var $loading = $('.gh-repos-drawer__loading');
    var $error = $('.gh-repos-drawer__error');
    var $list = $('.gh-repos-drawer__list');

    // Check cache first
    if (!forceRefresh) {
      var cachedRepos = State.getCachedRepos();
      if (cachedRepos) {
        $loading.hide();
        renderGitHubRepos(cachedRepos);
        return;
      }
    }

    // Show loading, hide others
    $loading.show();
    $error.hide();
    $list.empty();

    // Fetch all repos in parallel
    var promises = Constants.FAVORITE_REPOS.map(function (repo) {
      return fetch('https://api.github.com/repos/' + repo)
        .then(function (response) {
          if (!response.ok) {
            throw new Error('HTTP ' + response.status);
          }
          return response.json();
        })
        .catch(function (err) {
          console.warn('Failed to fetch ' + repo + ':', err);
          return null; // Return null for failed requests
        });
    });

    Promise.all(promises)
      .then(function (results) {
        // Filter out failed requests
        var repos = results.filter(function (repo) {
          return repo !== null && !repo.message;
        });

        if (repos.length === 0) {
          throw new Error('No repos could be fetched');
        }

        // Cache the results
        State.setCachedRepos(repos);

        $loading.hide();
        renderGitHubRepos(repos);
      })
      .catch(function (err) {
        console.error('GitHub API error:', err);
        $loading.hide();
        $error.show();
      });
  }

  function renderGitHubRepos(repos) {
    var $list = $('.gh-repos-drawer__list');
    $list.empty();

    repos.forEach(function (repo) {
      var starsFormatted = (repo.stargazers_count || 0).toLocaleString();
      var languageColor = Constants.LANGUAGE_COLORS[repo.language] || '#ccc';

      var $card = $(
        '<a class="gh-repos-drawer__repo-card" href="' +
          repo.html_url +
          '" target="_blank" rel="noopener noreferrer"></a>'
      );

      // Header with name and stars
      var $header = $('<div class="gh-repos-drawer__repo-header"></div>');
      var $name = $(
        '<h3 class="gh-repos-drawer__repo-name">' + repo.name + '</h3>'
      );
      var $stars = $(
        '<span class="gh-repos-drawer__repo-stars">' +
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>' +
          starsFormatted +
          '</span>'
      );
      $header.append($name, $stars);

      // Description
      var $description = $(
        '<p class="gh-repos-drawer__repo-description">' +
          (repo.description || 'No description available') +
          '</p>'
      );

      // Meta (language)
      var $meta = $('<div class="gh-repos-drawer__repo-meta"></div>');
      if (repo.language) {
        var $language = $(
          '<span class="gh-repos-drawer__repo-language">' +
            '<span class="gh-repos-drawer__language-dot" style="background-color: ' +
            languageColor +
            '"></span>' +
            repo.language +
            '</span>'
        );
        $meta.append($language);
      }

      // Topics
      var $topics = $('<div class="gh-repos-drawer__repo-topics"></div>');
      if (repo.topics && repo.topics.length > 0) {
        repo.topics.slice(0, 4).forEach(function (topic) {
          $topics.append(
            '<span class="gh-repos-drawer__repo-topic">' + topic + '</span>'
          );
        });
      }

      $card.append($header, $description, $meta);
      if (repo.topics && repo.topics.length > 0) {
        $card.append($topics);
      }

      $list.append($card);
    });
  }

  return {
    init: init,
    open: open,
    close: close
  };
})();

module.exports = GitHubDrawer;
