/**
 * @module PinnedCarousel
 * @description Horizontal carousel for pinned posts
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize pinned carousel
 */

var PinnedCarousel = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $track = $('#pinned-carousel-track');
    if ($track.length === 0) return;

    var trackEl = $track[0];
    var $left = $('#carousel-arrow-left');
    var $right = $('#carousel-arrow-right');

    // Prevent duplicate bindings if this is re-initialized
    $left.off('click.pinned');
    $right.off('click.pinned');
    $track.off('scroll.pinned');

    // Helper to calculate max scroll distance
    var maxScroll = function () {
      return Math.max(0, trackEl.scrollWidth - trackEl.clientWidth);
    };

    // Update UI state (arrow buttons, fade indicators)
    var ticking = false;
    function updateUI() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var sl = $track.scrollLeft();
        var ms = maxScroll();

        $track.toggleClass('has-scroll-left', sl > 10);
        $track.toggleClass('has-scroll-right', sl < ms - 10);
        $left.prop('disabled', sl <= 10);
        $right.prop('disabled', sl >= ms - 10);

        ticking = false;
      });
    }

    // Arrow button click handler - scroll by 2 cards
    function scrollByCards(dir) {
      dir = dir || 1;
      var $card = $track.find('.pinned-section__card').first();
      var cardW = $card.length
        ? Math.round($card.outerWidth(true))
        : Math.round(trackEl.clientWidth * 0.8);
      var delta = dir < 0 ? -(cardW * 2) : cardW * 2;

      if (trackEl.scrollBy) {
        trackEl.scrollBy({ left: delta, behavior: 'smooth' });
      } else {
        var target = Math.max(
          0,
          Math.min(maxScroll(), $track.scrollLeft() + delta)
        );
        $track.animate({ scrollLeft: target }, 300);
      }
    }

    // Bind arrow button clicks
    $left.on('click.pinned', function (e) {
      e.preventDefault();
      scrollByCards(-1);
    });
    $right.on('click.pinned', function (e) {
      e.preventDefault();
      scrollByCards(1);
    });

    // Update UI on scroll and resize
    $track.on('scroll.pinned', updateUI);
    $(window).on('resize.pinned', updateUI);

    // Initial state
    updateUI();

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = PinnedCarousel;
