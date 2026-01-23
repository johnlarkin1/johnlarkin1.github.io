/**
 * @module TableOfContents
 * @description Generate and manage table of contents for posts
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize table of contents
 */

var TableOfContents = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $toc = $('.toc');
    if ($toc.length === 0) return;

    var $toggle = $toc.find('.toc__toggle');
    var $nav = $toc.find('.toc__nav');

    // Generate TOC from headers (h1, h2, h3, h4) - skip "Table of Contents" header
    var $headers = $('.post-content')
      .find('h1, h2, h3, h4')
      .filter(function () {
        var text = $(this).text().trim();
        // For comparison, strip emojis/special chars but check original text isn't empty
        var textForComparison = text
          .toLowerCase()
          .replace(/[^\w\s]/g, '')
          .trim();
        return textForComparison !== 'table of contents' && text.length > 0;
      });

    if ($headers.length === 0) {
      $toc.hide();
      return;
    }

    var $list = $('<ul class="toc__list"></ul>');
    $headers.each(function () {
      var $h = $(this);
      // Get the full text content including emojis
      var text = $h.text().trim();
      var level = this.tagName.toLowerCase();
      var id = $h.attr('id');
      if (!id) {
        // Generate ID from text, removing emojis and special chars for valid ID
        id =
          'toc-' +
          text
            .toLowerCase()
            .replace(/[\u{1F300}-\u{1FAF8}]/gu, '') // Remove emojis from ID
            .replace(/[^\w\s-]/g, '')
            .trim()
            .replace(/\s+/g, '-')
            .substring(0, 50);
        $h.attr('id', id);
      }

      var $item = $('<li class="toc__item toc__item--' + level + '"></li>');
      var $link = $('<a class="toc__link" href="#' + id + '"></a>');
      $link.text(text); // Preserves emojis
      $item.append($link);
      $list.append($item);
    });
    $nav.append($list);

    // Toggle functionality - starts collapsed on all devices
    $toggle.on('click', function () {
      var expanded = $(this).attr('aria-expanded') === 'true';
      $(this).attr('aria-expanded', !expanded);
      $nav.toggleClass('is-open');
    });

    // Scroll spy
    var $links = $toc.find('.toc__link');
    var scrollTimer;
    $(window).on('scroll.toc', function () {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(function () {
        var scrollPos = $(window).scrollTop() + 100;
        var activeId = null;
        $headers.each(function () {
          if (scrollPos >= $(this).offset().top) {
            activeId = $(this).attr('id');
          }
        });
        $links.removeClass('is-active');
        if (activeId) {
          $links.filter('[href="#' + activeId + '"]').addClass('is-active');
        }
      }, 50);
    });

    // Smooth scroll on click
    $links.on('click', function (e) {
      e.preventDefault();
      var $target = $($(this).attr('href'));
      if ($target.length) {
        $('html, body').animate({ scrollTop: $target.offset().top - 80 }, 400);
        // Close on mobile after click
        if ($(window).width() < 992) {
          $toggle.attr('aria-expanded', 'false');
          $nav.removeClass('is-open');
        }
      }
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = TableOfContents;
