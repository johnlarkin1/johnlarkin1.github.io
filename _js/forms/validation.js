/**
 * @module FormValidation
 * @description Form validation utilities
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize form validation
 * @exports {Function} validateEmail - Validate email format
 * @exports {Function} validateRequired - Check if field is not empty
 */

var FormValidation = (function () {
  'use strict';

  var _isInitialized = false;

  /**
   * Validate if the input is not empty
   * @param {string} value - Value to check
   * @returns {boolean} True if not empty
   */
  function validateRequired(value) {
    return value !== '';
  }

  /**
   * Validate if the email is using correct format
   * @param {string} value - Email to validate
   * @returns {boolean} True if valid or empty
   */
  function validateEmail(value) {
    if (value !== '') {
      return /[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/i.test(
        value
      );
    }
    return true;
  }

  /**
   * Add error message to the input
   * @param {jQuery} element - Input element
   * @param {string} error - Error message
   */
  function addErrorData(element, error) {
    element.parent().addClass('error');
    element.after("<span class='error-data'>" + error + '</span>');
  }

  function init() {
    if (_isInitialized) return;

    $('.js-submit').click(function (e) {
      e.preventDefault();

      var $inputs = $('.form__input input');
      var textarea = $('.form__input textarea');
      var isError = false;

      $('.form__input').removeClass('error');
      $('.error-data').remove();

      for (var i = 0; i < $inputs.length; i++) {
        var input = $inputs[i];
        if (
          $(input).attr('required', true) &&
          !validateRequired($(input).val())
        ) {
          addErrorData($(input), 'This field is required');
          isError = true;
        }
        if (
          $(input).attr('required', true) &&
          $(input).attr('type') === 'email' &&
          !validateEmail($(input).val())
        ) {
          addErrorData($(input), 'Email address is invalid');
          isError = true;
        }
        if (
          $(textarea).attr('required', true) &&
          !validateRequired($(textarea).val())
        ) {
          addErrorData(
            $(textarea),
            'This field is required - is this change getting detected'
          );
          isError = true;
        }
      }
      if (isError === false) {
        $('#contactForm').submit();
      }
    });

    _isInitialized = true;
  }

  return {
    init: init,
    validateEmail: validateEmail,
    validateRequired: validateRequired
  };
})();

module.exports = FormValidation;
