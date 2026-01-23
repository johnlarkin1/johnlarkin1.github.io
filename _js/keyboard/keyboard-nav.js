/**
 * @module KeyboardNav
 * @description Vim/GitHub style keyboard navigation (g + key sequences)
 *
 * @requires jquery - Global jQuery ($)
 * @requires ../modals/search-modal - Search modal
 * @requires ../modals/contact-hub - Contact hub modal
 * @requires ../modals/shortcuts-modal - Shortcuts modal
 *
 * @exports {Function} init - Initialize keyboard navigation
 */

var SearchModal = require('../modals/search-modal');
var ContactHub = require('../modals/contact-hub');
var ShortcutsModal = require('../modals/shortcuts-modal');

var KeyboardNav = (function () {
  'use strict';

  var _isInitialized = false;
  var gKeyPressed = false;
  var gKeyTimeout = null;

  function init() {
    if (_isInitialized) return;

    $(document).on('keydown', function (e) {
      // Ignore if user is typing in an input/textarea
      var tag = e.target.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) {
        return;
      }

      // Ignore if any modifier keys are held (except shift)
      if (e.metaKey || e.ctrlKey || e.altKey) {
        return;
      }

      var key = e.key.toLowerCase();

      // First key: 'g' starts the sequence
      if (key === 'g' && !gKeyPressed) {
        gKeyPressed = true;

        // Reset after 1.5 seconds if no second key
        clearTimeout(gKeyTimeout);
        gKeyTimeout = setTimeout(function () {
          gKeyPressed = false;
        }, 1500);

        return;
      }

      // Second key: execute action if 'g' was pressed
      if (gKeyPressed) {
        gKeyPressed = false;
        clearTimeout(gKeyTimeout);

        switch (key) {
          case 'h': // g h → Home
            e.preventDefault();
            window.location.href = '/';
            break;

          case 'c': // g c → Categories
            e.preventDefault();
            window.location.href = '/categories/';
            break;

          case 'a': // g a → About
            e.preventDefault();
            window.location.href = '/about/';
            break;

          case 'o': // g o → Open contact modal
            e.preventDefault();
            ContactHub.open();
            break;

          case 's': // g s → Open search
            e.preventDefault();
            SearchModal.open();
            break;
        }
      }

      // Standalone shortcuts (no 'g' prefix needed)
      if (!gKeyPressed) {
        // '/' opens search (common pattern)
        if (key === '/' && !$('.search-modal').hasClass('is-active')) {
          e.preventDefault();
          SearchModal.open();
        }

        // '?' opens shortcuts help
        if (
          (key === '?' || (e.shiftKey && key === '/')) &&
          !$('.shortcuts-modal').hasClass('is-active')
        ) {
          e.preventDefault();
          ShortcutsModal.open();
        }
      }
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = KeyboardNav;
