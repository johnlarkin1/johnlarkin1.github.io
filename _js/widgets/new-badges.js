/**
 * @module NewBadges
 * @description Add "NEW" badges to recent posts
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize new badges
 */

var NewBadges = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    var twoWeeksAgo = new Date();
    twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);

    $('.post-card').each(function () {
      var $card = $(this);
      var dateStr = $card.data('date');

      if (!dateStr) return;

      var postDate = new Date(dateStr);

      if (postDate >= twoWeeksAgo) {
        $card.addClass('new');
        var $placeholder = $card.find('.post-card__new-badge-placeholder');
        if ($placeholder.length > 0) {
          $placeholder.replaceWith(
            '<span class="label post-card__new-badge">✨ NEW</span>'
          );
        }
      }
    });

    _isInitialized = true;
  }

  return {
    init: init
  };
})();

module.exports = NewBadges;
