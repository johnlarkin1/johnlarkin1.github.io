/**
 * @module Constants
 * @description Shared constants used across modules
 *
 * @exports {Array} FAVORITE_REPOS - List of featured GitHub repositories
 * @exports {Object} LANGUAGE_COLORS - GitHub language color mapping
 * @exports {number} LG_BREAKPOINT - Large screen breakpoint (992px)
 */

var Constants = (function () {
  'use strict';

  // Featured GitHub repositories for the drawer
  var FAVORITE_REPOS = [
    'johnlarkin1/claude-code-extensions',
    'johnlarkin1/imessage-data-foundry',
    'johnlarkin1/larkin-mcp',
    'johnlarkin1/yourname-mcp',
    'johnlarkin1/word-hunt-solver'
  ];

  // GitHub language colors mapping
  var LANGUAGE_COLORS = {
    JavaScript: '#f1e05a',
    TypeScript: '#2b7489',
    Python: '#3572A5',
    Java: '#b07219',
    'C++': '#f34b7d',
    C: '#555555',
    Go: '#00ADD8',
    Rust: '#dea584',
    Ruby: '#701516',
    PHP: '#4F5D95',
    Swift: '#ffac45',
    Kotlin: '#A97BFF',
    Scala: '#c22d40',
    Shell: '#89e051',
    HTML: '#e34c26',
    CSS: '#563d7c',
    Vue: '#4fc08d',
    React: '#61dafb',
    Angular: '#dd0031',
    Svelte: '#ff3e00',
    'Jupyter Notebook': '#DA5B0B'
  };

  // Breakpoint constants (match SCSS variables)
  var LG_BREAKPOINT = 992;

  return {
    FAVORITE_REPOS: FAVORITE_REPOS,
    LANGUAGE_COLORS: LANGUAGE_COLORS,
    LG_BREAKPOINT: LG_BREAKPOINT
  };
})();

module.exports = Constants;
