/**
 * @module BaseModal
 * @description Base modal open/close utilities
 *
 * @requires jquery - Global jQuery ($)
 * @requires velocity-animate - Velocity.js for animations
 *
 * @exports {Function} open - Open the base modal
 * @exports {Function} close - Close the base modal
 * @exports {Function} init - Initialize modal close handlers
 */

var BaseModal = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    $('.js-modal-close').click(function () {
      close();
    });

    $('.modal__overlay').click(function () {
      close();
    });

    _isInitialized = true;
  }

  function open() {
    $('body').css('overflow', 'hidden');
    $('.modal, .modal__overlay').show().css('display', 'flex');
    $('.modal__inner').velocity({ translateY: 0, opacity: 1 });
    $('.modal__overlay').velocity({ opacity: 1 }, 100);
  }

  function close() {
    $('body').css({ overflow: 'visible' });
    $('.modal, .modal__overlay, .modal__inner').velocity(
      { opacity: 0 },
      function () {
        $('.modal').css({ opacity: 1 });
        $('.modal__inner').css({
          '-webkit-transform': 'translateY(200px)',
          '-ms-transform': 'translateY(200px)',
          transform: 'translateY(200px)'
        });
        $('.modal, .modal__overlay').hide();
        $('.modal__body').empty();
      }
    );
  }

  return {
    init: init,
    open: open,
    close: close
  };
})();

module.exports = BaseModal;
