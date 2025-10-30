---
title: "Test Python Interactive"
layout: post
python-interactive: true
categories: [Test]
summary: Testing interactive Python execution
published: false
---

# Testing Interactive Python

This is a test post to verify the Python interactive feature works.

## Simple Example

<div class="interactive-python">
```python
print("Hello from Python!")
print("2 + 2 =", 2 + 2)

import numpy as np
print("NumPy version:", np.__version__)

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", arr.mean())
```
</div>

## Plot Example

<div class="interactive-python">
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Sine Wave')
plt.legend()
plt.grid(True, alpha=0.3)

print("Plot generated successfully!")
```
</div>

## Done!

If both code blocks above run successfully, the feature is working.
