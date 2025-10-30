# How to Add Interactive Python to Your Muon Post

## Quick Start (3 Steps)

### Step 1: Update Front Matter
Open `_posts/2025-10-28-understanding-muon.md` and add `python-interactive: true`:

```yaml
---
title: "Exploring Muon"
layout: post
featured-img:
mathjax: true
pinned: false
python-interactive: true    # ← ADD THIS LINE
categories: [Algorithms, A.I., M.L.]
summary: Deep diving into one element of Karpathy's nanochat
---
```

### Step 2: Add Interactive Code Block
Anywhere in your markdown, wrap Python code in this HTML structure:

```html
<div class="interactive-python">
<pre><code>import numpy as np
import matplotlib.pyplot as plt

# Your Python code here
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('My Interactive Plot')
plt.grid(True)

print("✓ Code executed successfully!")
</code></pre>
</div>
```

**Important:** Use `<pre><code>` tags, NOT markdown code fences (` ```python `), because Jekyll doesn't process markdown inside HTML blocks.

### Step 3: Build and Test
```bash
# Already done! But if you make changes:
node generate_sass.js  # Rebuild CSS
node build_js.js       # Rebuild JS
bundle exec jekyll build  # Rebuild site
```

---

## 📍 Suggested Location in Your Muon Post

I recommend adding an interactive example after **line 102** (after "### Visualization"). Here's the perfect spot:

**Current content (line 102-105):**
```markdown
### Visualization

I don't have a loss function that is equivalent to the

## Standard Gradient Descent
```

**Add this right after line 104:**

```html
Let's visualize how momentum helps optimization compared to standard gradient descent:

<div class="interactive-python">
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 2D loss landscape (elongated bowl shape)
def compute_loss(x, y):
    return 0.5 * x**2 + 5 * y**2

def compute_gradient(x, y):
    return np.array([x, 10*y])

# Standard SGD
def run_sgd(start, steps=50, lr=0.05):
    path = [start.copy()]
    pos = start.copy()

    for _ in range(steps):
        grad = compute_gradient(pos[0], pos[1])
        pos = pos - lr * grad
        path.append(pos.copy())

    return np.array(path)

# SGD with Momentum
def run_momentum(start, steps=50, lr=0.05, beta=0.9):
    path = [start.copy()]
    pos = start.copy()
    velocity = np.zeros(2)

    for _ in range(steps):
        grad = compute_gradient(pos[0], pos[1])
        velocity = beta * velocity - lr * grad
        pos = pos + velocity
        path.append(pos.copy())

    return np.array(path)

# Run optimizers
start = np.array([10.0, 3.0])
sgd_path = run_sgd(start)
mom_path = run_momentum(start)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Loss landscape with paths
x = np.linspace(-12, 12, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = compute_loss(X, Y)

ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.4)
ax1.plot(sgd_path[:, 0], sgd_path[:, 1], 'o-',
         linewidth=2, markersize=3, label='SGD')
ax1.plot(mom_path[:, 0], mom_path[:, 1], 's-',
         linewidth=2, markersize=3, label='Momentum')
ax1.plot(0, 0, '*', markersize=20, label='Optimum', zorder=5)
ax1.set_xlabel('θ₁', fontsize=12)
ax1.set_ylabel('θ₂', fontsize=12)
ax1.set_title('Optimizer Paths', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss convergence
sgd_loss = [compute_loss(p[0], p[1]) for p in sgd_path]
mom_loss = [compute_loss(p[0], p[1]) for p in mom_path]

ax2.plot(sgd_loss, 'o-', linewidth=2, label='SGD')
ax2.plot(mom_loss, 's-', linewidth=2, label='Momentum')
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss Over Time', fontsize=14)
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

print(f"Final SGD loss: {sgd_loss[-1]:.6f}")
print(f"Final Momentum loss: {mom_loss[-1]:.6f}")
print(f"Speedup: {sgd_loss[-1]/mom_loss[-1]:.1f}x faster!")
```
</div>

Notice how momentum takes a smoother, more direct path to the minimum!
```

---

## 🎨 What It Looks Like

The interactive block will render as:
1. **Code display** - Syntax highlighted Python code
2. **▶ Run Code button** - Gradient purple/blue button
3. **Output area** (after clicking Run):
   - Text output (print statements)
   - Matplotlib plots (as images)
   - Error messages (if any)

---

## 💡 More Examples

### Example 1: Show Momentum Buildup
```html
<div class="interactive-python">
```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate velocity accumulation
iterations = 20
beta = 0.9
lr = 0.1
gradient = 2.0

velocity = 0
velocities = [velocity]

for i in range(iterations):
    velocity = beta * velocity - lr * gradient
    velocities.append(velocity)

plt.figure(figsize=(10, 5))
plt.plot(velocities, 'o-', linewidth=2, markersize=6)
plt.axhline(y=-(lr*gradient)/(1-beta), color='r',
            linestyle='--', label='Terminal velocity')
plt.xlabel('Iteration')
plt.ylabel('Velocity')
plt.title('Momentum Buildup (β=0.9)')
plt.grid(True, alpha=0.3)
plt.legend()

print(f"Terminal velocity: {-(lr*gradient)/(1-beta):.4f}")
```
</div>
```

### Example 2: Parameter Update Comparison
```html
<div class="interactive-python">
```python
# Compare update rules
print("Standard Gradient Descent:")
print("  θ_{t+1} = θ_t - η∇L(θ_t)")
print()
print("SGD with Momentum:")
print("  v_{t+1} = βv_t - η∇L(θ_t)")
print("  θ_{t+1} = θ_t + v_{t+1}")
print()
print("Key difference: Momentum carries 'memory' of past gradients")
print(f"With β=0.9, we remember ~90% of previous velocity")
```
</div>
```

---

## 🚀 Testing Your Changes

1. **Local development:**
   ```bash
   bundle exec jekyll serve
   # Visit http://localhost:4000
   ```

2. **Check your post:**
   - Navigate to the Muon post
   - You should see the "▶ Run Code" button
   - Click it and wait 10-20 seconds (first run loads Pyodide)
   - Subsequent runs are much faster!

3. **Verify output:**
   - Check for print statements
   - Check for plots
   - Try intentionally breaking the code to see error handling

---

## ⚠️ Important Notes

1. **First load is slow** - Pyodide downloads ~40MB on first run (then cached)
2. **Only basic packages** - NumPy, Matplotlib, standard library only
3. **No pandas/scipy** - If you need these, let me know and we can add them
4. **Static plots only** - No interactive plotly/bokeh (yet)

---

## 🐛 Troubleshooting

**Button doesn't appear:**
- Check `python-interactive: true` is in front matter
- Check the div wrapper is correct: `<div class="interactive-python">`
- Rebuild JS: `node build_js.js`

**Code doesn't run:**
- Check browser console for errors
- Verify Pyodide CDN is accessible
- Try the simple test example first

**Plots don't show:**
- Make sure you're creating the figure: `plt.figure()`
- Don't use `plt.show()` - plots are auto-captured
- Check matplotlib code is correct

---

## 📝 Summary

**You're all set!** Just:
1. Add `python-interactive: true` to front matter
2. Wrap Python code in `<div class="interactive-python">`
3. Push to GitHub Pages

The feature is production-ready and will work great for demonstrating optimizer behavior in your Muon post! 🎉

---

**Need help?** Check:
- Full docs: `PYTHON_INTERACTIVE_README.md`
- Example code: `_posts/_examples/muon-optimizer-example.html`
- Test page: `_posts/_examples/test-python-interactive.md`
