# Interactive Python Code Blocks

This repository now supports interactive Python code execution in blog posts! Users can run Python code directly in their browser with full NumPy and Matplotlib support.

## Features

- ✅ Run Python code directly in the browser (no backend needed)
- ✅ Full NumPy and Matplotlib support
- ✅ Persistent output display
- ✅ Beautiful syntax highlighting
- ✅ Error handling with clear messages
- ✅ Perfect for educational/technical content

## How to Use

### 1. Enable Python Interactive in Your Post

Add `python-interactive: true` to your post's front matter:

```yaml
---
title: "My Post with Python"
layout: post
python-interactive: true
categories: [Python, Tutorial]
---
```

### 2. Add Interactive Code Blocks

Use HTML div wrapper with class `interactive-python`:

```html
<div class="interactive-python">
```python
import numpy as np
import matplotlib.pyplot as plt

# Create some data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Plot it
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.legend()
plt.grid(True, alpha=0.3)
```
</div>
```

### 3. Example for Muon Post

Here's a perfect example for visualizing optimizer behavior:

```html
<div class="interactive-python">
```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate SGD vs Momentum optimizer on a simple function
def loss_landscape(x, y):
    """A simple 2D loss landscape"""
    return x**2 + 10*y**2

# SGD updates
def sgd_step(pos, grad, lr=0.1):
    return pos - lr * grad

# Momentum updates
def momentum_step(pos, grad, velocity, lr=0.1, beta=0.9):
    velocity = beta * velocity - lr * grad
    return pos + velocity, velocity

# Initialize
np.random.seed(42)
start_pos = np.array([3.0, 2.0])

# Run both optimizers
sgd_path = [start_pos.copy()]
mom_path = [start_pos.copy()]
velocity = np.zeros(2)

for i in range(20):
    # Compute gradients at current position
    grad_sgd = 2 * np.array([sgd_path[-1][0], 10*sgd_path[-1][1]])
    grad_mom = 2 * np.array([mom_path[-1][0], 10*mom_path[-1][1]])

    # Update positions
    sgd_path.append(sgd_step(sgd_path[-1], grad_sgd))
    new_pos, velocity = momentum_step(mom_path[-1], grad_mom, velocity)
    mom_path.append(new_pos)

# Visualize
sgd_path = np.array(sgd_path)
mom_path = np.array(mom_path)

plt.figure(figsize=(12, 5))

# Left plot: trajectories
plt.subplot(1, 2, 1)
x = np.linspace(-4, 4, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = loss_landscape(X, Y)
plt.contour(X, Y, Z, levels=20, alpha=0.3)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'o-', label='SGD', markersize=4)
plt.plot(mom_path[:, 0], mom_path[:, 1], 's-', label='Momentum', markersize=4)
plt.plot(0, 0, 'r*', markersize=15, label='Optimum')
plt.xlabel('θ₁')
plt.ylabel('θ₂')
plt.title('Optimizer Paths')
plt.legend()
plt.grid(True, alpha=0.3)

# Right plot: loss over time
plt.subplot(1, 2, 2)
sgd_loss = [loss_landscape(p[0], p[1]) for p in sgd_path]
mom_loss = [loss_landscape(p[0], p[1]) for p in mom_path]
plt.plot(sgd_loss, 'o-', label='SGD')
plt.plot(mom_loss, 's-', label='Momentum')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
print("✓ Visualization complete!")
print(f"Final SGD loss: {sgd_loss[-1]:.4f}")
print(f"Final Momentum loss: {mom_loss[-1]:.4f}")
```
</div>
```

## Technical Details

### Architecture

- **Pyodide**: WebAssembly Python runtime (loaded from CDN)
- **Packages**: NumPy, Matplotlib (pre-loaded)
- **Execution**: Isolated per code block
- **Output**: Text output + matplotlib plots as base64 PNG images

### Files Created

1. `_js/pyodide-runner.js` - Main JavaScript logic
2. `_sass/components/_python-runner.scss` - Styling
3. `_layouts/post.html` - Updated to load Pyodide conditionally

### Build Integration

The JavaScript is automatically bundled via Browserify in your existing build pipeline:

```bash
npm run js    # Bundle JavaScript
npm run sass  # Compile SCSS
npm run build # Full build
```

## Limitations

- First run may take 10-20 seconds (Pyodide needs to download ~40MB)
- Code execution is sandboxed (no file system access, no network)
- Not all Python packages are available (stick to NumPy, Matplotlib, standard library)
- Plots are static PNG images (no interactive plotly/bokeh)

## Best Practices

1. **Keep code simple**: Complex computations may be slow in browser
2. **Add print statements**: Help users understand what's happening
3. **Use clear examples**: Make the code educational
4. **Set figure size**: `plt.figure(figsize=(10, 6))` for good readability
5. **Close plots**: Pyodide auto-closes with `show_plot()` helper

## Troubleshooting

### Pyodide not loading
- Check browser console for errors
- Ensure `python-interactive: true` is in front matter
- Verify CDN is accessible

### Code not running
- Check for syntax errors
- Ensure you're using Python 3 syntax
- Try simpler code first to isolate issue

### Plots not showing
- Make sure you're using `plt.plot()` not `plt.show()`
- Check that matplotlib code is correct
- Try the example above first

## Future Enhancements

Potential improvements:
- Add more packages (pandas, scipy)
- Support editable code blocks
- Add code download feature
- Syntax highlighting within editor
- Multiple code blocks with shared state

---

**Note**: This feature uses Pyodide v0.25.0. Check the [Pyodide documentation](https://pyodide.org/) for supported packages and features.
