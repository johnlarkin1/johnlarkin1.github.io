/**
 * @module CodeToggle
 * @description Tab toggle functionality for code blocks
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize code toggle tabs
 */

var CodeToggle = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    $('.code-toggle__tab').on('click', function () {
      var $tab = $(this);
      var $toggle = $tab.closest('.code-toggle');
      var targetTab = $tab.data('tab');

      // Update tab active states
      $toggle.find('.code-toggle__tab').removeClass('code-toggle__tab--active');
      $tab.addClass('code-toggle__tab--active');

      // Update pane active states
      $toggle
        .find('.code-toggle__pane')
        .removeClass('code-toggle__pane--active');
      $toggle
        .find('.code-toggle__pane[data-pane="' + targetTab + '"]')
        .addClass('code-toggle__pane--active');
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = CodeToggle;
