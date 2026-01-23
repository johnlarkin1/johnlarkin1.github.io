/**
 * @module ContactHub
 * @description Contact hub modal with form validation
 *
 * @requires jquery - Global jQuery ($)
 * @requires ../core/utils - trapFocus utility
 *
 * @exports {Function} init - Initialize contact hub
 * @exports {Function} open - Open contact hub modal
 * @exports {Function} close - Close contact hub modal
 */

var Utils = require('../core/utils');

var ContactHub = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $hub = $('.contact-hub');
    var $trigger = $('.js-contact-trigger');
    var $close = $('.js-contact-hub-close');

    if ($hub.length === 0) return;

    // Open contact hub
    $trigger.on('click', function (e) {
      e.preventDefault();
      open();
    });

    // Close contact hub
    $close.on('click', function (e) {
      e.preventDefault();
      close();
    });

    // Form validation for contact hub
    initContactHubForm();

    _isInitialized = true;
  }

  function open() {
    var $hub = $('.contact-hub');
    if ($hub.length === 0) return;

    // Prevent body scroll
    $('body').css('overflow', 'hidden');

    // Show and animate
    $hub.addClass('is-active');

    // Focus the first input after animation
    setTimeout(function () {
      $hub.find('input, textarea').first().focus();
    }, 600);

    // Set up focus trapping
    Utils.trapFocus($hub[0]);
  }

  function close() {
    var $hub = $('.contact-hub');
    if (!$hub.hasClass('is-active')) return;

    // Add closing class for exit animation
    $hub.addClass('is-closing');

    // Wait for animation then hide
    setTimeout(function () {
      $hub.removeClass('is-active is-closing');
      $('body').css('overflow', '');

      // Return focus to trigger button
      $('.js-contact-trigger').first().focus();
    }, 300);
  }

  function initContactHubForm() {
    var $form = $('#contactHubForm');
    if ($form.length === 0) return;

    $form.on('submit', function (e) {
      var $fields = $form.find('.contact-hub__field');
      var hasError = false;

      // Clear previous errors
      $fields.removeClass('is-error');

      // Validate required fields
      $fields.each(function () {
        var $field = $(this);
        var $input = $field.find('input, textarea');

        if ($input.prop('required') && !$input.val().trim()) {
          $field.addClass('is-error');
          hasError = true;
        }

        // Validate email format
        if ($input.attr('type') === 'email' && $input.val().trim()) {
          var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (!emailRegex.test($input.val().trim())) {
            $field.addClass('is-error');
            hasError = true;
          }
        }
      });

      if (hasError) {
        e.preventDefault();
        // Focus first error field
        $fields.filter('.is-error').first().find('input, textarea').focus();
      }
    });

    // Clear error on input
    $form.find('input, textarea').on('input', function () {
      $(this).closest('.contact-hub__field').removeClass('is-error');
    });
  }

  return {
    init: init,
    open: open,
    close: close
  };
})();

module.exports = ContactHub;
