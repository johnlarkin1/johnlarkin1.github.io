/**
 * @module ScrollIndicator
 * @description Section scroll indicator for pinned/year sections on homepage
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize scroll indicator
 */

var ScrollIndicator = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $indicator = $('.scroll-indicator');
    if ($indicator.length === 0) return;

    var $items = $indicator.find('.scroll-indicator__item');
    if ($items.length === 0) return;

    // Find all post cards and organize by section
    var $allPosts = $('.post-card');
    if ($allPosts.length === 0) return;

    var sections = [];

    // Check for pinned posts
    var $pinnedPosts = $allPosts.filter('[data-pinned="true"]');
    if ($pinnedPosts.length > 0) {
      var $indicatorItem = $items.filter('[data-section="pinned"]');
      if ($indicatorItem.length > 0) {
        sections.push({
          name: 'pinned',
          $item: $indicatorItem,
          $firstPost: $pinnedPosts.first()
        });
      }
    }

    // Group regular posts by year and find first post of each year
    var yearMap = new Map();
    $allPosts.not('[data-pinned="true"]').each(function () {
      var year = $(this).data('year');
      if (year && !yearMap.has(year)) {
        yearMap.set(year, $(this));
      }
    });

    // Add year sections in order
    yearMap.forEach(function (firstPost, year) {
      var $indicatorItem = $items.filter('[data-section="year-' + year + '"]');
      if ($indicatorItem.length > 0) {
        sections.push({
          name: 'year-' + year,
          $item: $indicatorItem,
          $firstPost: firstPost
        });
      }
    });

    if (sections.length === 0) return;

    // Update active section based on scroll position
    function updateActiveSection() {
      var scrollTop = $(window).scrollTop();
      var windowHeight = $(window).height();
      var scrollMid = scrollTop + windowHeight / 3;

      // Check if we're past the hero section
      var hero = $('.hero');
      var heroHeight =
        hero.length > 0 ? hero.offset().top + hero.outerHeight() : 0;

      if (scrollTop + 200 > heroHeight) {
        $indicator.addClass('visible');
      } else {
        $indicator.removeClass('visible');
      }

      var activeSection = sections[0];

      // Find which section we're in
      for (var i = 0; i < sections.length; i++) {
        var section = sections[i];
        if (section.$firstPost.length > 0) {
          var postTop = section.$firstPost.offset().top;
          if (scrollMid >= postTop) {
            activeSection = section;
          }
        }
      }

      // Update active state
      $items.removeClass('active');
      if (activeSection) {
        activeSection.$item.addClass('active');
      }
    }

    // Throttle scroll events
    var scrollTimeout;
    $(window).on('scroll', function () {
      if (scrollTimeout) clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(updateActiveSection, 50);
    });

    // Initial update
    updateActiveSection();

    // Click to scroll to section
    $items.on('click', function (e) {
      e.preventDefault();
      var sectionName = $(this).data('section');
      var section = sections.find(function (s) {
        return s.name === sectionName;
      });

      if (section && section.$firstPost.length > 0) {
        $('html, body').animate(
          {
            scrollTop: section.$firstPost.offset().top - 100
          },
          600
        );
      }
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = ScrollIndicator;
