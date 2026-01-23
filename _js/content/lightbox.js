/**
 * @module Lightbox
 * @description Image lightbox functionality for viewing images in full screen
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize lightbox
 * @exports {Function} open - Open lightbox with an image
 * @exports {Function} close - Close the lightbox
 */

var Lightbox = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    // Create lightbox overlay HTML
    var lightboxHTML =
      '<div class="lightbox-overlay" id="lightbox-overlay">' +
      '<div class="lightbox-content">' +
      '<button class="lightbox-close" id="lightbox-close">&times;</button>' +
      '<div class="lightbox-image-container">' +
      '<img id="lightbox-image" src="" alt="">' +
      '</div>' +
      '<div class="lightbox-caption" id="lightbox-caption"></div>' +
      '</div>' +
      '</div>';

    // Add lightbox to body
    $('body').append(lightboxHTML);

    // Click handlers for lightbox images
    $(document).on('click', '.lightbox-image', function (e) {
      e.preventDefault();
      openFromElement(this);
    });

    // Click handler for hero images
    $(document).on('click', '.hero--clickable', function (e) {
      // Don't trigger if clicking on a link inside the hero
      if ($(e.target).closest('a').length) return;

      var src = $(this).data('lightbox-src');
      var caption = $(this).data('lightbox-caption') || '';

      if (src) {
        openHero(src, caption);
      }
    });

    // Keyboard support for hero lightbox
    $(document).on('keydown', '.hero--clickable', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        $(this).trigger('click');
      }
    });

    // Close lightbox handlers
    $('#lightbox-close').click(close);
    $('#lightbox-overlay').click(function (e) {
      if (e.target === this) {
        close();
      }
    });

    _isInitialized = true;
  }

  function openFromElement(imageElement) {
    var $img = $(imageElement);
    var src = $img.attr('src');
    var alt = $img.attr('alt') || '';

    // Find caption from next sibling with .image-caption class
    var caption = '';
    var $nextElement = $img.parent().next('.image-caption');
    if ($nextElement.length) {
      caption = $nextElement.text();
    }

    $('#lightbox-image').attr('src', src).attr('alt', alt);
    $('#lightbox-caption').text(caption);

    $('body').css('overflow', 'hidden');
    $('.lightbox-overlay').css('display', 'flex').hide().fadeIn(300);
  }

  function openHero(src, caption) {
    $('#lightbox-image').attr('src', src).attr('alt', caption);
    $('#lightbox-caption').text(caption);

    $('body').css('overflow', 'hidden');
    $('.lightbox-overlay').css('display', 'flex').hide().fadeIn(300);
  }

  function close() {
    $('.lightbox-overlay').fadeOut(300, function () {
      $('body').css('overflow', 'visible');
      $('#lightbox-image').attr('src', '');
      $('#lightbox-caption').text('');
    });
  }

  return {
    init: init,
    open: openFromElement,
    openHero: openHero,
    close: close
  };
})();

module.exports = Lightbox;
