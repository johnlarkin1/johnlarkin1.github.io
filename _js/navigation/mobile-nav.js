/**
 * @module MobileNav
 * @description Mobile navigation toggle and accordion functionality
 *
 * @requires jquery - Global jQuery ($)
 * @requires velocity-animate - Velocity.js for animations
 * @requires ../core/constants - LG_BREAKPOINT
 * @requires ./profile-dropdown - Profile dropdown module
 *
 * @exports {Function} init - Initialize mobile navigation
 * @exports {Function} resetAccordion - Reset mobile accordion state
 */

var Constants = require('../core/constants');
var ProfileDropdown = require('./profile-dropdown');

var MobileNav = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    $('.header__toggle').click(function () {
      if (!$('.header__links').is('.velocity-animating')) {
        if ($('.header__links').hasClass('js--open')) {
          hideMobileNav();
        } else {
          openMobileNav();
        }
      }
    });

    $('body').on('click', function (e) {
      if (e.target.classList.contains('header__overlay')) {
        hideMobileNav();
      }
    });

    // Initialize mobile accordion for Projects dropdown
    initMobileAccordion();

    // Initialize profile dropdown
    ProfileDropdown.init();

    _isInitialized = true;
  }

  function initMobileAccordion() {
    var $trigger = $('.header__dropdown-trigger');

    $trigger.on('click', function (e) {
      // Only handle accordion on mobile (below $lg breakpoint)
      if ($(window).width() < Constants.LG_BREAKPOINT) {
        e.preventDefault();
        e.stopPropagation();

        var $dropdown = $(this).closest('.header__dropdown');
        var isOpen = $dropdown.hasClass('is-open');

        // Toggle accordion state
        $dropdown.toggleClass('is-open', !isOpen);

        // Update aria-expanded for accessibility
        $(this).attr('aria-expanded', !isOpen);
      }
    });
  }

  function resetAccordion() {
    // Reset accordion state when closing menu or resizing
    $('.header__dropdown').removeClass('is-open');
    $('.header__dropdown-trigger').attr('aria-expanded', 'false');
  }

  function openMobileNav() {
    $('.header__links').velocity('slideDown', {
      duration: 300,
      easing: 'ease-out',
      display: 'block',
      visibility: 'visible',
      begin: function () {
        $('.header__toggle').addClass('--open');
        $('body').append("<div class='header__overlay'></div>");
        // Add animating class for staggered reveal
        $(this).addClass('js--animating');
      },
      progress: function () {
        $('.header__overlay').addClass('--open');
      },
      complete: function () {
        $(this).addClass('js--open');
        // Remove animating class after animations complete (500ms buffer)
        var $links = $(this);
        setTimeout(function () {
          $links.removeClass('js--animating');
        }, 500);
      }
    });
  }

  function hideMobileNav() {
    $('.header__overlay').remove();
    $('.header__links').velocity('slideUp', {
      duration: 300,
      easing: 'ease-out',
      display: 'none',
      visibility: 'hidden',
      begin: function () {
        $('.header__toggle').removeClass('--open');
        // Reset accordion when menu closes
        resetAccordion();
      },
      progress: function () {
        $('.header__overlay').removeClass('--open');
      },
      complete: function () {
        $(this).removeClass('js--open js--animating');
        $('.header__toggle, .header__overlay').removeClass('--open');
      }
    });
  }

  return {
    init: init,
    resetAccordion: resetAccordion
  };
})();

module.exports = MobileNav;
