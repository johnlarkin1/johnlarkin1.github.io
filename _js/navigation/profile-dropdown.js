/**
 * @module ProfileDropdown
 * @description Profile dropdown in header (desktop only)
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize profile dropdown
 * @exports {Function} close - Close the profile dropdown
 */

var ProfileDropdown = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var $profile = $('.header__profile');

    if ($profile.length === 0) return;

    // Initialize profile image fallback (for desktop)
    initProfileImageFallback();

    // Close profile dropdown when clicking contact trigger inside it (desktop)
    $profile.find('.js-contact-trigger').on('click', function () {
      close();
    });

    // Close profile dropdown when clicking any link inside it (desktop)
    $profile.find('.header__profile-link').on('click', function () {
      close();
    });

    _isInitialized = true;
  }

  function initProfileImageFallback() {
    // Handle trigger image
    var $triggerImg = $('.header__profile-trigger .header__profile-img');
    $triggerImg.on('error', function () {
      $(this).closest('.header__profile-trigger').addClass('has-fallback');
    });

    // Handle avatar image in dropdown
    var $avatarImg = $(
      '.header__profile-avatar-wrapper .header__profile-avatar'
    );
    $avatarImg.on('error', function () {
      $(this).closest('.header__profile-avatar-wrapper').addClass('has-fallback');
    });
  }

  function close() {
    var $profile = $('.header__profile');
    var $trigger = $('.js-profile-trigger');
    $profile.removeClass('is-open');
    $trigger.attr('aria-expanded', 'false');
  }

  return {
    init: init,
    close: close
  };
})();

module.exports = ProfileDropdown;
