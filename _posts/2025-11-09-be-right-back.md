---
title: "Summoning Ghosts"
layout: post
featured-video:
featured-poster:
featured-gif:
mathjax: false
python-interactive: false
categories: [Algorithms, A.I., M.L.]
summary: Walking through a
---

Things to talk about:

- chat.db / address_book.db
  - attribute body
  - snapshotting logic
  - memory implications
- fine-tuning woes

```
--base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
--base-model mlx-community/Llama-3.2-3B-Instruct-4bit
--base-model mlx-community/Llama-3.1-8B-Instruct-4bit
--base-model mlx-community/phi-3-mini-intruct
--base-model mlx-community/Phi-4-mini-instruct-4bit
--base-model microsoft/Phi-4-mini-instruct
```

## iMessage `sqlite3` Schemas

### `chat.db`

<div style="width: 100%; max-width: 100vw; margin: 0 auto;">
  <iframe 
    src="https://dbdiagram.io/e/6910b1e46735e11170ef0295/6910b1e76735e11170ef0330" 
    style="width: 100%; min-height: 400px; aspect-ratio: 16/9; border: none;"
    allowfullscreen
    loading="lazy"
  ></iframe>
</div>

### The Woes of `attributeBody`

### `address_book.db`

<div style="width: 100%; max-width: 100vw; margin: 0 auto;">
  <iframe 
    src="https://dbdiagram.io/e/6910b7a06735e11170ef8e37/6910b7c66735e11170ef922a" 
    style="width: 100%; min-height: 400px; aspect-ratio: 16/9; border: none;"
    allowfullscreen
    loading="lazy"
  ></iframe>
</div>
