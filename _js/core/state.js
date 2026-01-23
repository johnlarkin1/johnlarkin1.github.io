/**
 * @module State
 * @description Shared state management for cross-module data
 *
 * @exports {Object} search - Search-related state
 * @exports {Object} repos - GitHub repos cache
 */

var State = (function () {
  'use strict';

  // Mobile detection for default search mode
  var isMobile = window.matchMedia('(max-width: 576px)').matches;

  // Search state
  var search = {
    mode: isMobile ? 'keyword' : 'hybrid', // 'keyword', 'semantic', or 'hybrid'
    vectorDebounceTimer: null,
    hybridDebounceTimer: null
  };

  // GitHub repos cache
  var repos = {
    data: null,
    timestamp: null,
    ttl: 5 * 60 * 1000 // 5 minutes
  };

  /**
   * Get the current search mode
   * @returns {string} Current search mode
   */
  function getSearchMode() {
    return search.mode;
  }

  /**
   * Set the search mode
   * @param {string} mode - New search mode
   */
  function setSearchMode(mode) {
    search.mode = mode;
  }

  /**
   * Check if repos cache is valid
   * @returns {boolean} True if cache is still valid
   */
  function isReposCacheValid() {
    if (!repos.data || !repos.timestamp) return false;
    var age = Date.now() - repos.timestamp;
    return age < repos.ttl;
  }

  /**
   * Get cached repos data
   * @returns {Array|null} Cached repos or null
   */
  function getCachedRepos() {
    return isReposCacheValid() ? repos.data : null;
  }

  /**
   * Set repos cache
   * @param {Array} data - Repos data to cache
   */
  function setCachedRepos(data) {
    repos.data = data;
    repos.timestamp = Date.now();
  }

  return {
    search: search,
    repos: repos,
    getSearchMode: getSearchMode,
    setSearchMode: setSearchMode,
    isReposCacheValid: isReposCacheValid,
    getCachedRepos: getCachedRepos,
    setCachedRepos: setCachedRepos
  };
})();

module.exports = State;
