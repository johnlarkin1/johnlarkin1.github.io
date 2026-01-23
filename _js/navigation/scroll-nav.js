/**
 * @module ScrollNav
 * @description Show/hide navigation based on scroll direction
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize scroll-based nav behavior
 */

var ScrollNav = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var previousScroll = 0;
    var $header = $('.header');
    var navHeight = $header.outerHeight();
    var detachPoint = 576 + 60;
    var hideShowOffset = 6;

    $(window).scroll(function () {
      var wW = 1024;

      // Only apply show/hide on larger screens
      if ($(window).width() >= wW) {
        if (!$header.hasClass('fixed')) {
          var currentScroll = $(this).scrollTop();
          var scrollDifference = Math.abs(currentScroll - previousScroll);

          // If scrolled past nav
          if (currentScroll > navHeight) {
            // If scrolled past detach point -> show nav
            if (currentScroll > detachPoint) {
              if (!$header.hasClass('fix-nav')) {
                $header.addClass('fix-nav');
              }
            }

            if (scrollDifference >= hideShowOffset) {
              if (currentScroll > previousScroll) {
                // Scroll down -> hide nav
                if (!$header.hasClass('hide-nav')) {
                  $header.addClass('hide-nav');
                }
              } else {
                // Scroll up -> show nav
                if ($header.hasClass('hide-nav')) {
                  $header.removeClass('hide-nav');
                }
              }
            }
          } else {
            // At the top
            if (currentScroll <= 0) {
              $header.removeClass('hide-nav show-nav');
              $header.addClass('top');
            }
          }
        }

        // Scrolled to the bottom -> show nav
        if (window.innerHeight + window.scrollY >= document.body.offsetHeight) {
          $header.removeClass('hide-nav');
        }
        previousScroll = currentScroll;
      } else {
        $header.addClass('fix-nav');
      }
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = ScrollNav;
