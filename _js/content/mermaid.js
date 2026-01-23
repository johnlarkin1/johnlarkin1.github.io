/**
 * @module Mermaid
 * @description Initialize and render Mermaid diagrams with pan/zoom controls
 *
 * @requires jquery - Global jQuery ($)
 *
 * @exports {Function} init - Initialize Mermaid diagrams
 */

var Mermaid = (function () {
  'use strict';

  var _isInitialized = false;

  function init() {
    if (_isInitialized) return;

    // Check if there are any mermaid code blocks or elements on the page
    var hasMermaidContent =
      $('code.language-mermaid').length > 0 || $('.mermaid').length > 0;

    if (!hasMermaidContent) {
      // No Mermaid content found, skip initialization
      return;
    }

    // Wait for Mermaid library to be available
    waitForMermaid();

    _isInitialized = true;
  }

  function waitForMermaid() {
    if (typeof window.mermaid !== 'undefined') {
      console.log('Mermaid library loaded, initializing...');
      setupMermaid();
    } else {
      console.log('Waiting for Mermaid library...');
      setTimeout(waitForMermaid, 100);
    }
  }

  function setupMermaid() {
    // Convert Jekyll syntax-highlighted code blocks to Mermaid divs
    $('code.language-mermaid').each(function () {
      var $code = $(this);
      var $pre = $code.closest('pre');
      var $div = $('<div></div>');

      $div.addClass('mermaid');
      var content = $code.text().trim();
      $div.text(content);

      console.log('Converting mermaid code block:', content);
      $pre.replaceWith($div);
    });

    // Initialize Mermaid - always use dark theme
    console.log('Theme detected:', 'dark (forced for mindmaps)');

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: 'dark',
      themeVariables: {
        primaryColor: '#ff6b6b',
        primaryTextColor: '#fff',
        primaryBorderColor: '#444',
        lineColor: '#888',
        mindmapLabelBackgroundColor: 'transparent',
        mindmapNodeBackgroundColor: '#374151',
        mindmapNodeTextColor: '#fff'
      }
    });

    // Find and render Mermaid diagrams
    var $mermaidElements = $('.mermaid');
    console.log('Found mermaid elements:', $mermaidElements.length);

    if ($mermaidElements.length > 0) {
      window.mermaid
        .run({
          querySelector: '.mermaid'
        })
        .then(function () {
          console.log('Mermaid diagrams rendered successfully');

          // Add pan/zoom controls to each rendered diagram
          $('.mermaid').each(function () {
            addPanZoomToMermaid(this);
          });
        })
        .catch(function (error) {
          console.error('Error rendering Mermaid diagrams:', error);
        });
    }
  }

  function addPanZoomToMermaid(container) {
    var $container = $(container);
    var svg = container.querySelector('svg');

    if (!svg || $container.closest('.mermaid-wrapper').length > 0) {
      return; // Already wrapped or no SVG
    }

    console.log('Adding pan/zoom to diagram with SVG ID:', svg.id);

    // Wait for panzoom library to be available
    if (typeof window.panzoom === 'undefined') {
      console.log('Panzoom library not yet loaded, will retry...');
      setTimeout(function () {
        addPanZoomToMermaid(container);
      }, 100);
      return;
    }

    console.log('Adding pan/zoom controls to Mermaid diagram');

    // Create wrapper
    var $wrap = $('<div class="mermaid-wrapper"></div>');
    $wrap.css({
      width: '100%',
      height: 'min(75vh, 600px)',
      overflow: 'hidden',
      position: 'relative',
      border: '1px solid #444',
      borderRadius: '8px',
      background: '#1a1a1a',
      margin: '1.5rem 0'
    });

    $container.before($wrap);
    $wrap.append($container);

    // Initialize panzoom
    var pz = window.panzoom(svg, {
      maxZoom: 4,
      minZoom: 0.5,
      smoothScroll: false
    });

    // Create control buttons
    var $controls = $('<div class="mermaid-controls"></div>');
    $controls.css({
      position: 'absolute',
      top: '.75rem',
      right: '.75rem',
      display: 'flex',
      gap: '.25rem',
      zIndex: 10
    });

    var buttons = [
      {
        text: '↺',
        action: function () {
          pz.moveTo(0, 0);
          pz.zoomAbs(0, 0, 1);
        }
      }
    ];

    buttons.forEach(function (btn) {
      var $button = $('<button class="mermaid-btn"></button>');
      $button.text(btn.text);
      $button.attr('type', 'button');
      $button.click(btn.action);
      $controls.append($button);
    });

    $wrap.append($controls);

    console.log('Pan/zoom controls added successfully');
  }

  return {
    init: init
  };
})();

module.exports = Mermaid;
