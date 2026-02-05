/**
 * @module ContribChart
 * @description GitHub 3D Contribution Chart theme selector and display
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize contribution chart controls
 */

var ContribChart = (function () {
  'use strict';

  var _isInitialized = false;

  // Base URL for contribution chart SVGs
  var BASE_URL =
    'https://raw.githubusercontent.com/johnlarkin1/johnlarkin1/main/profile-3d-contrib/';

  // Theme name mapping
  var THEMES = {
    default: { day: 'day.svg', night: 'night.svg' },
    'burnt-copper': {
      day: 'burnt-copper-day.svg',
      night: 'burnt-copper-night.svg',
    },
    'midnight-bloom': {
      day: 'midnight-bloom-day.svg',
      night: 'midnight-bloom-night.svg',
    },
    'moss-stone': { day: 'moss-stone-day.svg', night: 'moss-stone-night.svg' },
    'neon-sunset': {
      day: 'neon-sunset-day.svg',
      night: 'neon-sunset-night.svg',
    },
    'obsidian-aurora': {
      day: 'obsidian-aurora-day.svg',
      night: 'obsidian-aurora-night.svg',
    },
    'polar-dawn': { day: 'polar-dawn-day.svg', night: 'polar-dawn-night.svg' },
  };

  // Current state
  var _currentTheme = 'default';
  var _currentMode = 'day';

  function init() {
    if (_isInitialized) return;

    // Detect system preference for initial mode
    if (
      window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches
    ) {
      _currentMode = 'night';
    }

    // Update all mode buttons to reflect initial state
    updateModeButtons();

    // Update all images to reflect initial state
    updateAllImages();

    // Theme dropdown handlers (desktop)
    $(document).on('change', '.js-contrib-theme', function () {
      _currentTheme = $(this).val();
      syncThemeSelectors();
      updateAllImages();
    });

    // Theme dropdown handlers (mobile)
    $(document).on('change', '.js-contrib-theme-mobile', function () {
      _currentTheme = $(this).val();
      syncThemeSelectors();
      updateAllImages();
    });

    // Day/night toggle handlers (desktop)
    $(document).on('click', '.js-contrib-mode', function () {
      toggleMode();
    });

    // Day/night toggle handlers (mobile)
    $(document).on('click', '.js-contrib-mode-mobile', function () {
      toggleMode();
    });

    // Listen for system color scheme changes
    if (window.matchMedia) {
      window
        .matchMedia('(prefers-color-scheme: dark)')
        .addEventListener('change', function (e) {
          _currentMode = e.matches ? 'night' : 'day';
          updateModeButtons();
          updateAllImages();
        });
    }

    _isInitialized = true;
  }

  function toggleMode() {
    _currentMode = _currentMode === 'day' ? 'night' : 'day';
    updateModeButtons();
    updateAllImages();
  }

  function updateModeButtons() {
    // Update all mode toggle buttons
    $('.js-contrib-mode, .js-contrib-mode-mobile').attr(
      'data-mode',
      _currentMode
    );
  }

  function syncThemeSelectors() {
    // Sync all theme dropdowns
    $('.js-contrib-theme, .js-contrib-theme-mobile').val(_currentTheme);
  }

  function updateAllImages() {
    var themeFiles = THEMES[_currentTheme];
    var filename = themeFiles[_currentMode];
    var src = BASE_URL + filename;

    // Update all contribution chart images
    $('.js-contrib-img, .js-contrib-img-mobile').attr('src', src);
  }

  function getCurrentSrc() {
    var themeFiles = THEMES[_currentTheme];
    var filename = themeFiles[_currentMode];
    return BASE_URL + filename;
  }

  return {
    init: init,
    getCurrentSrc: getCurrentSrc,
  };
})();

module.exports = ContribChart;
