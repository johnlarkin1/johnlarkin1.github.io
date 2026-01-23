/**
 * @module ShortcutsModal
 * @description Keyboard shortcuts help modal
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize shortcuts modal
 * @exports {Function} open - Open shortcuts modal
 * @exports {Function} close - Close shortcuts modal
 */

var ShortcutsModal = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $modal = $('.shortcuts-modal');
    var $trigger = $('.js-shortcuts-trigger');
    var $close = $('.js-shortcuts-close');

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

    _isInitialized = true;
  }

  function open() {
    var $modal = $('.shortcuts-modal');
    if ($modal.length === 0 || $modal.hasClass('is-active')) return;

    $('body').addClass('shortcuts-modal--open');
    $modal.addClass('is-active');
  }

  function close() {
    var $modal = $('.shortcuts-modal');
    if (!$modal.hasClass('is-active')) return;

    $modal.addClass('is-closing');
    $('body').removeClass('shortcuts-modal--open');

    setTimeout(function () {
      $modal.removeClass('is-active is-closing');
    }, 200);
  }

  return {
    init: init,
    open: open,
    close: close
  };
})();

module.exports = ShortcutsModal;
