/**
 * @module GitHubRepoCards
 * @description Fetch and display GitHub repository information in cards
 *
 * @requires jquery - Global jQuery ($)
 * @requires ../core/constants - LANGUAGE_COLORS
 *
 * @exports {Function} init - Initialize GitHub repo cards
 */

var Constants = require('../core/constants');

var GitHubRepoCards = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    $('.github-repo-card').each(function () {
      var $card = $(this);
      var repo = $card.data('repo');

      if (!repo) {
        console.warn('GitHub repo card missing data-repo attribute');
        return;
      }

      var $loading = $card.find('.github-repo-loading');
      var $content = $card.find('.github-repo-content');
      var $error = $card.find('.github-repo-error');

      fetch('https://api.github.com/repos/' + repo)
        .then(function (response) {
          if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
          }
          return response.json();
        })
        .then(function (data) {
          if (data.message) {
            console.error('GitHub API error message:', data.message);
            $loading.hide();
            $error.show();
            $error.find('p').text('Error: ' + data.message);
            return;
          }

          // Update repository name and link
          var $nameLink = $content.find('.github-repo-name a');
          $nameLink.text(data.name);
          $nameLink.attr('href', data.html_url);

          // Update author information
          var $authorLink = $content.find('.github-repo-author a');
          $authorLink.text(data.owner.login);
          $authorLink.attr('href', data.owner.html_url);

          // Update description
          var $description = $content.find('.github-repo-description');
          $description.text(data.description || 'No description available');

          // Update language
          var $languageSpan = $content.find('.github-repo-language');
          if (data.language) {
            var $colorDot = $languageSpan.find('.language-color');
            var $langName = $languageSpan.find('.language-name');
            $colorDot.css(
              'background-color',
              Constants.LANGUAGE_COLORS[data.language] || '#ccc'
            );
            $langName.text(data.language);
            $languageSpan.show();
          } else {
            $languageSpan.hide();
          }

          // Update stars and forks
          $content
            .find('.stars-count')
            .text((data.stargazers_count || 0).toLocaleString());
          $content
            .find('.forks-count')
            .text((data.forks_count || 0).toLocaleString());

          // Update topics
          var $topicsContainer = $content.find('.github-repo-topics');
          if (data.topics && data.topics.length > 0) {
            var topicsHtml = data.topics
              .map(function (topic) {
                return '<span class="github-repo-topic">' + topic + '</span>';
              })
              .join('');
            $topicsContainer.html(topicsHtml);
            $topicsContainer.show();
          }

          // Show content, hide loading
          $loading.hide();
          $content.show();
        })
        .catch(function (err) {
          console.error('Error fetching GitHub repository:', err);
          $loading.hide();
          $error.show();
          $error.find('p').text('Error: ' + err.message);
        });
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = GitHubRepoCards;
