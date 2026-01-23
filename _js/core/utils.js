/**
 * @module Utils
 * @description Core utility functions used across modules
 *
 * @exports {Function} trapFocus - Trap keyboard focus within an element
 * @exports {Function} debounce - Debounce a function call
 * @exports {Function} throttle - Throttle a function call
 */

var Utils = (function () {
  'use strict';

  /**
   * Trap keyboard focus within an element for accessibility
   * @param {HTMLElement} element - The container to trap focus within
   */
  function trapFocus(element) {
    var focusableElements = element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    var firstFocusable = focusableElements[0];
    var lastFocusable = focusableElements[focusableElements.length - 1];

    function handleTabKey(e) {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable.focus();
        }
      }
    }

    // Add listener
    element.addEventListener('keydown', handleTabKey);

    // Store cleanup function
    element._trapFocusCleanup = function () {
      element.removeEventListener('keydown', handleTabKey);
    };
  }

  /**
   * Debounce a function call
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} Debounced function
   */
  function debounce(func, wait) {
    var timeout;
    return function () {
      var context = this;
      var args = arguments;
      clearTimeout(timeout);
      timeout = setTimeout(function () {
        func.apply(context, args);
      }, wait);
    };
  }

  /**
   * Throttle a function call
   * @param {Function} func - Function to throttle
   * @param {number} limit - Minimum time between calls in milliseconds
   * @returns {Function} Throttled function
   */
  function throttle(func, limit) {
    var inThrottle;
    return function () {
      var context = this;
      var args = arguments;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(function () {
          inThrottle = false;
        }, limit);
      }
    };
  }

  return {
    trapFocus: trapFocus,
    debounce: debounce,
    throttle: throttle
  };
})();

module.exports = Utils;
