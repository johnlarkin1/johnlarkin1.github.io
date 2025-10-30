/**
 * Interactive Python code runner using Pyodide
 * Allows running Python code in blog posts with matplotlib plotting support
 */

let pyodideReadyPromise = null;
let pyodide = null;

/**
 * Initialize Pyodide (lazy load when first needed)
 */
async function initPyodide() {
  if (pyodideReadyPromise) {
    return pyodideReadyPromise;
  }

  pyodideReadyPromise = (async () => {
    try {
      console.log("Loading Pyodide...");
      pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/",
      });

      // Load common packages for ML/scientific computing
      console.log("Loading Python packages...");
      await pyodide.loadPackage(["numpy", "matplotlib"]);

      // Setup matplotlib to work in web environment
      await pyodide.runPythonAsync(`
        import matplotlib
        matplotlib.use('AGG')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import io
        import base64

        def show_plot():
            """Capture matplotlib plot as base64 image"""
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            return img_str
      `);

      console.log("Pyodide ready!");
      return pyodide;
    } catch (error) {
      console.error("Failed to load Pyodide:", error);
      throw error;
    }
  })();

  return pyodideReadyPromise;
}

/**
 * Execute Python code and return output
 */
async function runPythonCode(code) {
  await initPyodide();

  try {
    // Redirect stdout to capture print statements
    await pyodide.runPythonAsync(`
      import sys
      from io import StringIO
      sys.stdout = StringIO()
    `);

    // Run user code
    await pyodide.runPythonAsync(code);

    // Check if there's a plot to show
    let plotImage = null;
    try {
      const hasPlot = await pyodide.runPythonAsync(`
        len(plt.get_fignums()) > 0
      `);

      if (hasPlot) {
        plotImage = await pyodide.runPythonAsync("show_plot()");
      }
    } catch (e) {
      // No plot, that's fine
    }

    // Get captured output
    const output = await pyodide.runPythonAsync(`
      sys.stdout.getvalue()
    `);

    // Reset stdout
    await pyodide.runPythonAsync(`
      sys.stdout = sys.__stdout__
    `);

    return {
      success: true,
      output: output || "",
      plot: plotImage,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Create interactive Python block UI
 */
function createPythonRunner(container) {
  const $container = $(container);

  // Extract code from either <code> tag or text content
  let code;
  const $codeTag = $container.find("code, pre > code");
  if ($codeTag.length > 0) {
    code = $codeTag.text().trim();
  } else {
    // Handle markdown code fences that Jekyll might have processed
    code = $container.text().trim();
    // Remove ```python and ``` if present
    code = code.replace(/^```python\s*/i, '').replace(/```\s*$/, '').trim();
  }

  // Create the UI structure
  const $wrapper = $('<div class="python-runner"></div>');

  // Create header with collapse button
  const $header = $('<div class="python-runner__header"></div>');
  const $collapseBtn = $(
    '<button class="python-runner__collapse-btn" type="button" aria-label="Toggle code visibility">' +
    '<span class="collapse-icon">▼</span> Show Code' +
    '</button>'
  );
  $header.append($collapseBtn);

  // Create collapsible code wrapper
  const $codeWrapper = $('<div class="python-runner__code"></div>');
  const $pre = $('<pre class="python-runner__pre"></pre>');
  const $codeBlock = $('<code class="python-runner__code-block language-python"></code>');

  // Add the code with syntax highlighting
  $codeBlock.text(code);

  // Apply syntax highlighting if Prism is available
  if (window.Prism) {
    $codeBlock.html(window.Prism.highlight(code, window.Prism.languages.python, 'python'));
  }

  $pre.append($codeBlock);
  $codeWrapper.append($pre);

  // Create controls (always visible)
  const $controls = $('<div class="python-runner__controls"></div>');
  const $runBtn = $(
    '<button class="python-runner__run-btn" type="button">▶ Run Code</button>'
  );
  const $spinner = $(
    '<span class="python-runner__spinner" style="display:none;">⏳ Running...</span>'
  );

  $controls.append($runBtn);
  $controls.append($spinner);

  // Handle collapse/expand
  $collapseBtn.on('click', function() {
    $codeWrapper.slideToggle(300);
    const $icon = $(this).find('.collapse-icon');
    const isCollapsed = !$codeWrapper.is(':visible');

    if (isCollapsed) {
      $icon.text('▶');
      $(this).html('<span class="collapse-icon">▶</span> Show Code');
      $wrapper.addClass('collapsed');
    } else {
      $icon.text('▼');
      $(this).html('<span class="collapse-icon">▼</span> Hide Code');
      $wrapper.removeClass('collapsed');
    }
  });

  // Create output area
  const $output = $('<div class="python-runner__output" style="display:none;"></div>');

  // Assemble the components
  $wrapper.append($header);
  $wrapper.append($codeWrapper);
  $wrapper.append($controls);
  $wrapper.append($output);

  // Start collapsed by default
  $codeWrapper.hide();
  $collapseBtn.html('<span class="collapse-icon">▶</span> Show Code');
  $wrapper.addClass('collapsed');

  // Replace original container with new UI
  $container.replaceWith($wrapper);

  // Handle run button click
  $runBtn.on("click", async function () {
    $runBtn.prop("disabled", true);
    $spinner.show();
    $output.empty().hide();

    try {
      const result = await runPythonCode(code);

      if (result.success) {
        // Show text output if any
        if (result.output) {
          const $textOutput = $(
            '<pre class="python-runner__text-output"></pre>'
          );
          $textOutput.text(result.output);
          $output.append($textOutput);
        }

        // Show plot if any
        if (result.plot) {
          const $plotOutput = $(
            '<div class="python-runner__plot-output"></div>'
          );
          const $img = $('<img alt="Plot output" />');
          $img.attr("src", "data:image/png;base64," + result.plot);
          $plotOutput.append($img);
          $output.append($plotOutput);
        }

        if (!result.output && !result.plot) {
          $output.append(
            '<div class="python-runner__success">✓ Code executed successfully (no output)</div>'
          );
        }
      } else {
        // Show error
        const $errorOutput = $(
          '<pre class="python-runner__error-output"></pre>'
        );
        $errorOutput.text("Error: " + result.error);
        $output.append($errorOutput);
      }

      $output.show();
    } catch (error) {
      const $errorOutput = $('<pre class="python-runner__error-output"></pre>');
      $errorOutput.text("Unexpected error: " + error.message);
      $output.append($errorOutput);
      $output.show();
    } finally {
      $runBtn.prop("disabled", false);
      $spinner.hide();
    }
  });
}

/**
 * Initialize all Python runners on the page
 */
function initPythonRunners() {
  $(".interactive-python").each(function () {
    createPythonRunner(this);
  });
}

// Export for use in main scripts
window.initPythonRunners = initPythonRunners;
