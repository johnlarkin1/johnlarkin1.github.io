/**
 * @module Main
 * @description Main entry point for the site's JavaScript
 *
 * This file imports and initializes all modules in the correct order.
 * It replaces the monolithic scripts.js with a modular architecture.
 */

// External dependencies
window.jQuery = window.$ = require('jquery');
require('velocity-animate/velocity.js');
require('lazysizes');
require('lazysizes/plugins/unveilhooks/ls.unveilhooks.js');

// Existing standalone modules (unchanged)
require('./pyodide-runner.js');
require('./vector-search.js');

// Core modules
var Events = require('./core/events');

// Navigation modules
var MobileNav = require('./navigation/mobile-nav');
var ScrollNav = require('./navigation/scroll-nav');
var ProfileDropdown = require('./navigation/profile-dropdown');

// Modal modules
var BaseModal = require('./modals/base-modal');
var ContactHub = require('./modals/contact-hub');
var SearchModal = require('./modals/search-modal');
var ShortcutsModal = require('./modals/shortcuts-modal');
var GitHubDrawer = require('./modals/github-drawer');

// Content modules
var Lightbox = require('./content/lightbox');
var Mermaid = require('./content/mermaid');
var GitHubRepoCards = require('./content/github-repo-cards');
var TableOfContents = require('./content/table-of-contents');
var CodeToggle = require('./content/code-toggle');

// Widget modules
var ScrollIndicator = require('./widgets/scroll-indicator');
var PinnedCarousel = require('./widgets/pinned-carousel');
var KnowledgeChecks = require('./widgets/knowledge-checks');
var NewBadges = require('./widgets/new-badges');
var BetaDistributionViz = require('./widgets/beta-distribution-viz');
var ContribChart = require('./widgets/contrib-chart');

// Form modules
var FormValidation = require('./forms/validation');

// Keyboard modules
var KeyboardNav = require('./keyboard/keyboard-nav');

// -------------------------------------------------------------------------
// Document Ready - Initialize All Modules
// -------------------------------------------------------------------------

$(document).ready(function () {
  // Reveal header once DOM is ready (prevents FOUC)
  $('.header').css('opacity', '1');

  // Navigation
  MobileNav.init();
  ScrollNav.init();

  // Forms
  FormValidation.init();
  filterBlogCategories();

  // Content
  GitHubRepoCards.init();
  Lightbox.init();
  Mermaid.init();
  TableOfContents.init();
  CodeToggle.init();

  // Widgets
  ScrollIndicator.init();
  KnowledgeChecks.init();
  NewBadges.init();
  PinnedCarousel.init();
  BetaDistributionViz.init();
  ContribChart.init();

  // Modals
  BaseModal.init();
  ContactHub.init();
  SearchModal.init();
  ShortcutsModal.init();
  GitHubDrawer.init();

  // Keyboard navigation
  KeyboardNav.init();

  // Initialize Python runners if present
  if ($('.interactive-python').length > 0) {
    initPythonRunners();
  }
});

// -------------------------------------------------------------------------
// Global Event Handlers
// -------------------------------------------------------------------------

// Close all modals if ESC is pressed
$(document).keyup(function (e) {
  if (e.keyCode === 27) {
    BaseModal.close();
    Lightbox.close();
    ContactHub.close();
    SearchModal.close();
    ShortcutsModal.close();
    GitHubDrawer.close();
    ProfileDropdown.close();
  }
});

// Handle window resize
$(window).resize(function () {
  $('.header').removeClass('hide-nav');
  $('.header__toggle').removeClass('--open');
  $('.header__links').removeClass('js--open');
  $('.header__links').removeAttr('style');
  $('.header__overlay').remove();
  // Reset accordion state on resize
  MobileNav.resetAccordion();
});

// -------------------------------------------------------------------------
// Legacy Helper Functions (kept for compatibility)
// -------------------------------------------------------------------------

// Filter navigation (legacy)
function filterBlogCategories() {
  $('.filter-click').click(function (e) {
    e.preventDefault();
    var button = $('.filter-click');
    var button_val = button.val();
    return button_val;
  });
}

// Footnotes header injection
document.addEventListener('DOMContentLoaded', function () {
  var footnotesDiv = document.querySelector('.footnotes');
  if (footnotesDiv) {
    var header = document.createElement('h2');
    header.className = 'footnotes-header';
    header.textContent = 'Footnotes';
    footnotesDiv.insertBefore(header, footnotesDiv.firstChild);
  }
});

// F1 score cell coloring
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('td.f1-score').forEach(function (cell) {
    if (cell.textContent !== 'N/A') {
      var score = parseFloat(cell.textContent);
      if (score >= 0.8) {
        cell.style.backgroundColor = '#4CAF50'; // Green
      } else if (score >= 0.6) {
        cell.style.backgroundColor = '#FFEB3B'; // Yellow
      } else {
        cell.style.backgroundColor = '#F44336'; // Red
      }
    }
  });
});

// Cluster time cell coloring
document.addEventListener('DOMContentLoaded', function () {
  var cells = Array.from(document.querySelectorAll('td.cluster-time')).filter(
    function (cell) {
      return cell.textContent !== 'N/A';
    }
  );
  var scores = cells.map(function (cell) {
    return parseFloat(cell.textContent);
  });
  var logScores = scores.map(function (score) {
    return Math.log1p(score);
  });
  var minLogScore = Math.min.apply(null, logScores);
  var maxLogScore = Math.max.apply(null, logScores);

  function interpolateColor(logScore, minLogScore, maxLogScore) {
    var fraction = (logScore - minLogScore) / (maxLogScore - minLogScore);
    var r = Math.round(255 * fraction);
    var g = Math.round(255 * (1 - fraction));
    var b = 0;
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  cells.forEach(function (cell) {
    var score = parseFloat(cell.textContent);
    var logScore = Math.log1p(score);
    var color = interpolateColor(logScore, minLogScore, maxLogScore);
    cell.style.backgroundColor = color;
  });
});
