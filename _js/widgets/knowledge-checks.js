/**
 * @module KnowledgeChecks
 * @description Interactive quiz/knowledge check functionality
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize knowledge check quizzes
 */

var KnowledgeChecks = (function () {
  'use strict';

  var _isInitialized = false;

  function parseBoolean(value) {
    if (typeof value === 'string') {
      return value.toLowerCase() === 'true';
    }
    return Boolean(value);
  }

  function init() {
    if (_isInitialized) return;

    var $quizzes = $('.markdown-quiz');

    if ($quizzes.length === 0) return;

    $quizzes.each(function (index) {
      var $quiz = $(this);
      var quizId = $quiz.data('quiz-id');

      if (!quizId) {
        quizId = 'markdown-quiz-' + (index + 1);
        $quiz.attr('data-quiz-id', quizId);
      }

      var $options = $quiz.find('.markdown-quiz__option');
      var $feedback = $quiz.find('.markdown-quiz__feedback');
      var $submit = $quiz.find('.markdown-quiz__submit');
      var $reset = $quiz.find('.markdown-quiz__reset');
      var $explanation = $quiz.find('.markdown-quiz__explanation');

      if ($options.length === 0 || $submit.length === 0) return;

      function clearState() {
        $quiz.removeClass('is-correct is-incorrect has-feedback');
        $options.removeClass('is-selected is-answer is-wrong');
        $feedback.text('');
        if ($explanation.length) {
          $explanation.attr('hidden', true);
        }
      }

      clearState();

      $options.each(function () {
        var $option = $(this);
        var $input = $option.find('input[type="radio"]');

        if (quizId && $input.length && !$input.attr('name')) {
          $input.attr('name', quizId);
        }

        $input.on('change', function () {
          $options.removeClass('is-selected');
          if ($input.is(':checked')) {
            $option.addClass('is-selected');
          }
          $quiz.removeClass('is-correct is-incorrect has-feedback');
          $options.removeClass('is-answer is-wrong');
          $feedback.text('');
          if ($explanation.length) {
            $explanation.attr('hidden', true);
          }
        });

        $option.on('click', function (event) {
          var tag = event.target.tagName.toLowerCase();
          if (tag !== 'input' && tag !== 'label') {
            $input.prop('checked', true).trigger('change');
          }
        });
      });

      $submit.on('click', function (event) {
        event.preventDefault();

        var $selected = $options.filter('.is-selected');
        if ($selected.length === 0) {
          $feedback.text('Select an option before checking the answer.');
          $quiz.addClass('has-feedback');
          return;
        }

        var isCorrect = parseBoolean($selected.data('correct'));
        var customFeedback = $selected.data('feedback');
        var $correctOption = $options.filter(function () {
          return parseBoolean($(this).data('correct'));
        });

        $options.removeClass('is-answer is-wrong');

        if (isCorrect) {
          $quiz.addClass('is-correct').removeClass('is-incorrect');
          $selected.addClass('is-answer');
          $feedback.text(customFeedback || "Great job! That's correct.");
        } else {
          $quiz.addClass('is-incorrect').removeClass('is-correct');
          $selected.addClass('is-wrong');
          if ($correctOption.length > 0) {
            $correctOption.first().addClass('is-answer');
          }
          $feedback.text(
            customFeedback || 'Not quite. Review the explanation and try again.'
          );
        }

        if ($explanation.length) {
          $explanation.attr('hidden', false);
        }

        $quiz.addClass('has-feedback');
      });

      if ($reset.length) {
        $reset.on('click', function (event) {
          event.preventDefault();
          clearState();
          $quiz.find('input[type="radio"]').prop('checked', false);
        });
      }
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = KnowledgeChecks;
