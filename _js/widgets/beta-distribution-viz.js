/**
 * @module BetaDistributionViz
 * @description Interactive Beta Distribution visualization for Thompson Sampling
 *
 * @requires jquery - Global jQuery ($)
 * @requires chart.js - Chart library (loaded from CDN)
 *
 * @exports {Function} init - Initialize beta distribution visualizations
 * 
 * @warning This is entirely a Claude Code creation!!
 */

// Chart.js is loaded from CDN via script tag
var Chart = window.Chart;

var BetaDistributionViz = (function () {
  'use strict';

  var _isInitialized = false;

  // Arm colors (colorblind-friendly palette)
  var ARM_COLORS = [
    { bg: 'rgba(59, 130, 246, 0.2)', border: 'rgb(59, 130, 246)', name: 'Blue' },
    { bg: 'rgba(239, 68, 68, 0.2)', border: 'rgb(239, 68, 68)', name: 'Red' },
    { bg: 'rgba(34, 197, 94, 0.2)', border: 'rgb(34, 197, 94)', name: 'Green' },
    { bg: 'rgba(168, 85, 247, 0.2)', border: 'rgb(168, 85, 247)', name: 'Purple' }
  ];

  // Log-gamma function for beta PDF calculation
  function logGamma(x) {
    var cof = [
      76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
    ];
    var y = x;
    var tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    var ser = 1.000000000190015;
    for (var j = 0; j < 6; j++) {
      ser += cof[j] / ++y;
    }
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }

  // Beta PDF calculation
  function betaPDF(x, alpha, beta) {
    if (x <= 0 || x >= 1) return 0;
    var logB = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
    return Math.exp(
      (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logB
    );
  }

  // Sample from beta distribution using gamma sampling
  function sampleFromBeta(alpha, beta) {
    // Use gamma sampling to generate beta samples
    function gammaVariate(shape) {
      if (shape < 1) {
        return gammaVariate(1 + shape) * Math.pow(Math.random(), 1 / shape);
      }
      var d = shape - 1 / 3;
      var c = 1 / Math.sqrt(9 * d);
      while (true) {
        var x, v;
        do {
          x = gaussianRandom();
          v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        var u = Math.random();
        if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    }

    function gaussianRandom() {
      var u1 = Math.random();
      var u2 = Math.random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    var x = gammaVariate(alpha);
    var y = gammaVariate(beta);
    return x / (x + y);
  }

  // Generate points for beta distribution curve
  function generateBetaCurve(alpha, beta, numPoints) {
    numPoints = numPoints || 200;
    var points = [];
    for (var i = 0; i <= numPoints; i++) {
      var x = i / numPoints;
      // Avoid exact 0 and 1 for numerical stability
      if (x === 0) x = 0.001;
      if (x === 1) x = 0.999;
      points.push({ x: x, y: betaPDF(x, alpha, beta) });
    }
    return points;
  }

  function init() {
    if (_isInitialized) return;

    var $containers = $('.interactive-beta-viz');
    if ($containers.length === 0) return;

    $containers.each(function (index) {
      initVisualization(this, index);
    });

    _isInitialized = true;
  }

  function initVisualization(container, index) {
    var $container = $(container);
    var numArms = parseInt($container.data('arms'), 10) || 3;
    var trueProbsStr = $container.data('true-probs') || '0.7,0.4,0.55';
    var trueProbs = trueProbsStr.split(',').map(function (p) {
      return parseFloat(p.trim());
    });

    // Ensure we have enough true probs
    while (trueProbs.length < numArms) {
      trueProbs.push(Math.random() * 0.6 + 0.2);
    }

    // Initialize arm states
    var arms = [];
    for (var i = 0; i < numArms; i++) {
      arms.push({
        id: i,
        alpha: 1,
        beta: 1,
        trueProb: trueProbs[i],
        successes: 0,
        failures: 0
      });
    }

    // Build UI
    var vizId = 'beta-viz-' + index;
    var html = buildHTML(vizId, arms);
    $container.html(html);

    // Initialize chart
    var ctx = document.getElementById(vizId + '-chart').getContext('2d');
    var chart = createChart(ctx, arms);

    // Store state
    var state = {
      arms: arms,
      chart: chart,
      autoPlayInterval: null,
      log: []
    };

    // Bind events
    bindEvents($container, vizId, state);

    // Initial render
    updateChart(state);
    updateArmStats($container, state);
  }

  function buildHTML(vizId, arms) {
    var html = '<div class="beta-viz" id="' + vizId + '">';

    // Chart section
    html += '<div class="beta-viz__chart-container">';
    html += '<canvas id="' + vizId + '-chart"></canvas>';
    html += '</div>';

    // Controls section
    html += '<div class="beta-viz__controls">';
    html += '<button class="beta-viz__btn beta-viz__btn--primary" data-action="thompson">Run Thompson Round</button>';
    html += '<button class="beta-viz__btn beta-viz__btn--secondary" data-action="autoplay">Auto Play</button>';
    html += '<button class="beta-viz__btn beta-viz__btn--secondary" data-action="reset">Reset All</button>';
    html += '</div>';

    // Arms grid
    html += '<div class="beta-viz__arms">';
    for (var i = 0; i < arms.length; i++) {
      html += buildArmCard(i, arms[i]);
    }
    html += '</div>';

    // Add arm button (if less than 4)
    if (arms.length < 4) {
      html += '<button class="beta-viz__btn beta-viz__btn--add" data-action="add-arm">+ Add Arm</button>';
    }

    // Log section
    html += '<div class="beta-viz__log-section">';
    html += '<h4 class="beta-viz__log-title">Sampling Log</h4>';
    html += '<div class="beta-viz__log" data-log></div>';
    html += '</div>';

    html += '</div>';
    return html;
  }

  function buildArmCard(index, arm) {
    var color = ARM_COLORS[index];
    var html = '<div class="beta-viz__arm" data-arm-index="' + index + '">';
    html += '<div class="beta-viz__arm-header" style="border-color: ' + color.border + '">';
    html += '<span class="beta-viz__arm-label" style="color: ' + color.border + '">Arm ' + (index + 1) + '</span>';
    html += '<span class="beta-viz__arm-true-prob">True p = ' + arm.trueProb.toFixed(2) + '</span>';
    html += '</div>';
    html += '<div class="beta-viz__arm-stats">';
    html += '<span data-stat="alpha">&#945; = 1</span>';
    html += '<span data-stat="beta">&#946; = 1</span>';
    html += '<span data-stat="mean">Mean = 0.50</span>';
    html += '</div>';
    html += '<div class="beta-viz__arm-counts">';
    html += '<span data-stat="successes">Successes: 0</span>';
    html += '<span data-stat="failures">Failures: 0</span>';
    html += '</div>';
    html += '<div class="beta-viz__arm-buttons">';
    html += '<button class="beta-viz__btn beta-viz__btn--success" data-action="success" data-arm="' + index + '">+Success</button>';
    html += '<button class="beta-viz__btn beta-viz__btn--failure" data-action="failure" data-arm="' + index + '">+Failure</button>';
    html += '</div>';
    html += '</div>';
    return html;
  }

  function createChart(ctx, arms) {
    var datasets = [];

    // Add datasets for each arm
    for (var i = 0; i < arms.length; i++) {
      var color = ARM_COLORS[i];
      datasets.push({
        label: 'Arm ' + (i + 1) + ' Beta(' + arms[i].alpha + ',' + arms[i].beta + ')',
        data: generateBetaCurve(arms[i].alpha, arms[i].beta),
        borderColor: color.border,
        backgroundColor: color.bg,
        borderWidth: 2,
        fill: true,
        pointRadius: 0,
        tension: 0.1
      });

      // Add vertical reference line for true probability
      datasets.push({
        label: 'True p=' + arms[i].trueProb.toFixed(2),
        data: [
          { x: arms[i].trueProb, y: 0 },
          { x: arms[i].trueProb, y: 10 }
        ],
        borderColor: color.border,
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        showLine: true,
        fill: false
      });
    }

    return new Chart(ctx, {
      type: 'line',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              filter: function (item) {
                // Only show arm labels, not reference lines
                return item.text.indexOf('Beta') !== -1;
              },
              color: '#888',
              font: { size: 11 }
            }
          },
          tooltip: {
            enabled: true,
            callbacks: {
              label: function (context) {
                if (context.dataset.label.indexOf('Beta') !== -1) {
                  return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                }
                return null;
              }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: 1,
            title: {
              display: true,
              text: 'Probability',
              color: '#888'
            },
            ticks: { color: '#888' },
            grid: { color: 'rgba(255,255,255,0.1)' }
          },
          y: {
            type: 'linear',
            min: 0,
            suggestedMax: 5,
            title: {
              display: true,
              text: 'Density',
              color: '#888'
            },
            ticks: { color: '#888' },
            grid: { color: 'rgba(255,255,255,0.1)' }
          }
        }
      }
    });
  }

  function updateChart(state) {
    var chart = state.chart;
    var arms = state.arms;

    // Find max density for y-axis scaling
    var maxY = 5;
    for (var i = 0; i < arms.length; i++) {
      var data = generateBetaCurve(arms[i].alpha, arms[i].beta);
      for (var j = 0; j < data.length; j++) {
        if (data[j].y > maxY) maxY = data[j].y;
      }
    }

    // Update datasets
    for (var k = 0; k < arms.length; k++) {
      var arm = arms[k];
      var datasetIndex = k * 2; // Each arm has 2 datasets (curve + reference line)

      // Update curve
      chart.data.datasets[datasetIndex].data = generateBetaCurve(arm.alpha, arm.beta);
      chart.data.datasets[datasetIndex].label = 'Arm ' + (k + 1) + ' Beta(' + arm.alpha + ',' + arm.beta + ')';

      // Update reference line height
      chart.data.datasets[datasetIndex + 1].data = [
        { x: arm.trueProb, y: 0 },
        { x: arm.trueProb, y: maxY * 1.1 }
      ];
    }

    // Update y-axis max
    chart.options.scales.y.suggestedMax = Math.min(maxY * 1.2, 20);

    chart.update('none');
  }

  function updateArmStats($container, state) {
    var arms = state.arms;

    for (var i = 0; i < arms.length; i++) {
      var arm = arms[i];
      var $armCard = $container.find('[data-arm-index="' + i + '"]');
      var mean = arm.alpha / (arm.alpha + arm.beta);

      $armCard.find('[data-stat="alpha"]').html('&#945; = ' + arm.alpha);
      $armCard.find('[data-stat="beta"]').html('&#946; = ' + arm.beta);
      $armCard.find('[data-stat="mean"]').text('Mean = ' + mean.toFixed(2));
      $armCard.find('[data-stat="successes"]').text('Successes: ' + arm.successes);
      $armCard.find('[data-stat="failures"]').text('Failures: ' + arm.failures);
    }
  }

  function runThompsonRound(state) {
    var arms = state.arms;

    // Sample from each arm's beta distribution
    var samples = arms.map(function (arm) {
      return sampleFromBeta(arm.alpha, arm.beta);
    });

    // Find arm with highest sample
    var maxSample = -1;
    var selectedArm = 0;
    for (var i = 0; i < samples.length; i++) {
      if (samples[i] > maxSample) {
        maxSample = samples[i];
        selectedArm = i;
      }
    }

    // Pull the selected arm and observe outcome based on true probability
    var arm = arms[selectedArm];
    var success = Math.random() < arm.trueProb;

    if (success) {
      arm.alpha += 1;
      arm.successes += 1;
    } else {
      arm.beta += 1;
      arm.failures += 1;
    }

    // Add to log
    state.log.unshift({
      arm: selectedArm + 1,
      samples: samples.map(function (s) { return s.toFixed(3); }),
      success: success
    });

    // Keep only last 20 entries
    if (state.log.length > 20) {
      state.log.pop();
    }

    return { selectedArm: selectedArm, success: success, samples: samples };
  }

  function updateLog($container, state) {
    var $log = $container.find('[data-log]');
    var html = '';

    for (var i = 0; i < state.log.length; i++) {
      var entry = state.log[i];
      var color = ARM_COLORS[entry.arm - 1].border;
      var icon = entry.success ? '&#10003;' : '&#10007;';
      var iconClass = entry.success ? 'beta-viz__log-success' : 'beta-viz__log-failure';

      html += '<div class="beta-viz__log-entry">';
      html += '<span class="beta-viz__log-arm" style="color: ' + color + '">Arm ' + entry.arm + '</span>';
      html += '<span class="' + iconClass + '">' + icon + '</span>';
      html += '<span class="beta-viz__log-samples">[' + entry.samples.join(', ') + ']</span>';
      html += '</div>';
    }

    $log.html(html || '<span class="beta-viz__log-empty">No rounds played yet</span>');
  }

  function bindEvents($container, vizId, state) {
    // Success button
    $container.on('click', '[data-action="success"]', function () {
      var armIndex = parseInt($(this).data('arm'), 10);
      state.arms[armIndex].alpha += 1;
      state.arms[armIndex].successes += 1;
      updateChart(state);
      updateArmStats($container, state);
    });

    // Failure button
    $container.on('click', '[data-action="failure"]', function () {
      var armIndex = parseInt($(this).data('arm'), 10);
      state.arms[armIndex].beta += 1;
      state.arms[armIndex].failures += 1;
      updateChart(state);
      updateArmStats($container, state);
    });

    // Thompson round button
    $container.on('click', '[data-action="thompson"]', function () {
      runThompsonRound(state);
      updateChart(state);
      updateArmStats($container, state);
      updateLog($container, state);
    });

    // Auto play button
    $container.on('click', '[data-action="autoplay"]', function () {
      var $btn = $(this);
      if (state.autoPlayInterval) {
        clearInterval(state.autoPlayInterval);
        state.autoPlayInterval = null;
        $btn.text('Auto Play').removeClass('beta-viz__btn--active');
      } else {
        $btn.text('Stop').addClass('beta-viz__btn--active');
        state.autoPlayInterval = setInterval(function () {
          runThompsonRound(state);
          updateChart(state);
          updateArmStats($container, state);
          updateLog($container, state);
        }, 500);
      }
    });

    // Reset button
    $container.on('click', '[data-action="reset"]', function () {
      // Stop autoplay if running
      if (state.autoPlayInterval) {
        clearInterval(state.autoPlayInterval);
        state.autoPlayInterval = null;
        $container.find('[data-action="autoplay"]').text('Auto Play').removeClass('beta-viz__btn--active');
      }

      // Reset all arms
      for (var i = 0; i < state.arms.length; i++) {
        state.arms[i].alpha = 1;
        state.arms[i].beta = 1;
        state.arms[i].successes = 0;
        state.arms[i].failures = 0;
      }

      // Clear log
      state.log = [];

      updateChart(state);
      updateArmStats($container, state);
      updateLog($container, state);
    });

    // Add arm button
    $container.on('click', '[data-action="add-arm"]', function () {
      if (state.arms.length >= 4) return;

      var newArm = {
        id: state.arms.length,
        alpha: 1,
        beta: 1,
        trueProb: Math.random() * 0.6 + 0.2,
        successes: 0,
        failures: 0
      };
      state.arms.push(newArm);

      // Rebuild the entire visualization
      var $viz = $container.find('.beta-viz');
      var vizId = $viz.attr('id');

      // Destroy old chart
      state.chart.destroy();

      // Rebuild HTML
      var html = buildHTML(vizId, state.arms);
      $container.html(html);

      // Recreate chart
      var ctx = document.getElementById(vizId + '-chart').getContext('2d');
      state.chart = createChart(ctx, state.arms);

      // Rebind events
      bindEvents($container, vizId, state);

      updateChart(state);
      updateArmStats($container, state);
      updateLog($container, state);
    });
  }

  return {
    init: init
  };
})();

module.exports = BetaDistributionViz;
