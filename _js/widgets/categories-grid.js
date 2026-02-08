/**
 * @module CategoriesGrid
 * @description Expandable row list for the /categories/ page
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize the categories grid
 */

var CategoriesGrid = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $grid = $('#categories-grid');
    if ($grid.length === 0) return;

    var $rows = $grid.find('.categories-grid__row');

    $rows.each(function () {
      var $row = $(this);
      var $header = $row.find('.categories-grid__row-header');

      // Click handler on the row header
      $header.on('click', function (e) {
        if ($(e.target).closest('.categories-grid__post-link').length) return;

        var isExpanded = $row.hasClass('is-expanded');

        if (isExpanded) {
          collapse($row);
        } else {
          expand($row);

          setTimeout(function () {
            var rowTop = $row.offset().top - 80;
            if ($(window).scrollTop() > rowTop) {
              $('html, body').animate({ scrollTop: rowTop }, 300);
            }
          }, 100);
        }
      });

      // Keyboard support
      $header.on('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          $header.trigger('click');
        }
      });
    });

    // Expand Favorites by default
    var $favorites = $rows.filter('.categories-grid__row--favorites');
    if ($favorites.length) {
      expand($favorites);
    }

    _isInitialized = true;
  }

  function expand($row) {
    $row.addClass('is-expanded');
    $row.find('.categories-grid__row-header').attr('aria-expanded', 'true');
    $row.find('.categories-grid__posts').attr('aria-hidden', 'false');
  }

  function collapse($row) {
    $row.removeClass('is-expanded');
    $row.find('.categories-grid__row-header').attr('aria-expanded', 'false');
    $row.find('.categories-grid__posts').attr('aria-hidden', 'true');
  }

  return {
    init: init
  };
})();

module.exports = CategoriesGrid;
