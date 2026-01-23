/**
 * @module Events
 * @description Simple pub/sub event system for cross-module communication
 *
 * @exports {Function} on - Subscribe to an event
 * @exports {Function} off - Unsubscribe from an event
 * @exports {Function} emit - Emit an event
 * @exports {Object} EVENTS - Event name constants
 */

var Events = (function () {
  'use strict';

  // Event name constants
  var EVENTS = {
    ESCAPE_PRESSED: 'escape:pressed',
    MODAL_OPEN: 'modal:open',
    MODAL_CLOSE: 'modal:close',
    SEARCH_OPEN: 'search:open',
    SEARCH_CLOSE: 'search:close',
    CONTACT_OPEN: 'contact:open',
    CONTACT_CLOSE: 'contact:close',
    SHORTCUTS_OPEN: 'shortcuts:open',
    SHORTCUTS_CLOSE: 'shortcuts:close',
    GITHUB_DRAWER_OPEN: 'github-drawer:open',
    GITHUB_DRAWER_CLOSE: 'github-drawer:close'
  };

  // Event listeners storage
  var _listeners = {};

  /**
   * Subscribe to an event
   * @param {string} event - Event name
   * @param {Function} callback - Handler function
   */
  function on(event, callback) {
    if (!_listeners[event]) {
      _listeners[event] = [];
    }
    _listeners[event].push(callback);
  }

  /**
   * Unsubscribe from an event
   * @param {string} event - Event name
   * @param {Function} callback - Handler to remove
   */
  function off(event, callback) {
    if (!_listeners[event]) return;

    var index = _listeners[event].indexOf(callback);
    if (index > -1) {
      _listeners[event].splice(index, 1);
    }
  }

  /**
   * Emit an event
   * @param {string} event - Event name
   * @param {*} data - Data to pass to handlers
   */
  function emit(event, data) {
    if (!_listeners[event]) return;

    _listeners[event].forEach(function (callback) {
      try {
        callback(data);
      } catch (e) {
        console.error('[Events] Error in handler for ' + event + ':', e);
      }
    });
  }

  return {
    EVENTS: EVENTS,
    on: on,
    off: off,
    emit: emit
  };
})();

module.exports = Events;
