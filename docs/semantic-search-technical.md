# Semantic Search Technical Deep Dive

This document explains the technical implementation of the semantic search feature in PR #98 (`feat-semantic-search`).

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BUILD TIME (CI)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   _posts/*.md ──► generate-embeddings.js ──► embeddings.json + chunks.json  │
│                          │                                                  │
│                          ▼                                                  │
│              ┌─────────────────────┐                                        │
│              │ Xenova/all-MiniLM-  │                                        │
│              │       L6-v2         │                                        │
│              │   (384 dimensions)  │                                        │
│              └─────────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CLIENT TIME (Browser)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Query ──► vector-search.js ──► Same Model (from CDN) ──► Embedding   │
│                                                │                            │
│                                                ▼                            │
│                                    Cosine Similarity vs                     │
│                                    Pre-computed Embeddings                  │
│                                                │                            │
│                                                ▼                            │
│                                         Top K Results                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Build-Time Embedding Generation

**File:** `_scripts/generate-embeddings.js`

### What It Does

This Node.js script runs during CI (GitHub Actions) and:
1. Reads all markdown blog posts from `_posts/`
2. Chunks them into ~400 token passages
3. Generates 384-dimensional embeddings using a sentence transformer
4. Outputs to `_site/search/embeddings.json` and `chunks.json`

### Configuration

```javascript
const CONFIG = {
  model: 'Xenova/all-MiniLM-L6-v2',  // Sentence transformer model
  dimensions: 384,                    // Output vector size
  targetTokens: 400,                  // Ideal chunk size
  minChunkTokens: 100,                // Discard chunks smaller than this
  maxChunkTokens: 500,                // Split chunks larger than this
  previewLength: 300,                 // Characters stored for preview
  postsDir: '_posts',
  outputDir: '_site/search',
};
```

### Step-by-Step Flow

#### 1. Token Estimation
```javascript
function estimateTokens(text) {
  return Math.ceil(text.length / 4);  // ~4 chars per token for English
}
```
This is a rough heuristic. Actual tokenizers like BPE produce variable results, but this is sufficient for chunking purposes.

#### 2. Markdown Stripping
The `stripMarkdown()` function converts markdown to plain text by removing:
- Code blocks (` ``` `) and inline code
- Images (`![](...)`)
- HTML tags
- Header markers (`#`, `##`, etc.)
- Bold/italic/strikethrough markers
- List markers (bullets, numbers)
- Blockquotes (`>`)
- Jekyll/Liquid tags (`{% %}`, `{{ }}`)

**Why?** Embeddings should capture semantic meaning, not formatting syntax.

#### 3. URL Building
```javascript
function buildUrl(filePath) {
  // YYYY-MM-DD-slug.md -> /YYYY/slug/
  const match = filename.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)$/);
  if (match) {
    return `/${match[1]}/${match[4]}/`;
  }
}
```
This follows your Jekyll permalink structure.

#### 4. Content Chunking Strategy

**Two-level chunking:**

1. **Section-level split:** Split by markdown headers (`# Header`)
2. **Paragraph-level split:** Within sections, combine paragraphs up to `targetTokens`

```javascript
function chunkContent(content, title) {
  // Split by headers first
  const sections = content.split(/^(#{1,6}\s+.+)$/gm);

  // For each section, create appropriately-sized chunks
  // respecting paragraph boundaries
}
```

**Why this approach?**
- Headers provide semantic boundaries (different topics)
- Paragraphs are natural thought units
- ~400 tokens balances context vs specificity
- Too small = loses context; too large = dilutes relevance

#### 5. Embedding Generation

```javascript
const extractor = await pipeline('feature-extraction', CONFIG.model, {
  quantized: true,  // Smaller model, faster inference
});

const output = await extractor(chunk.text, {
  pooling: 'mean',      // Average all token embeddings
  normalize: true,      // L2 normalize for cosine similarity
});
```

**Key settings:**
- `pooling: 'mean'` - Averages all token vectors into one document vector
- `normalize: true` - Normalizes to unit length, so dot product = cosine similarity

#### 6. Output Files

**embeddings.json:**
```json
{
  "version": "1.0",
  "model": "Xenova/all-MiniLM-L6-v2",
  "dimensions": 384,
  "generated": "2024-01-15T...",
  "count": 150,
  "embeddings": [
    [0.023, -0.045, 0.012, ...],
    ...
  ]
}
```

**chunks.json:**
```json
{
  "version": "1.0",
  "count": 150,
  "chunks": [
    {
      "id": 0,
      "url": "/2024/my-post/",
      "title": "My Post Title",
      "section": "Introduction",
      "text": "First 300 chars of chunk...",
      "categories": ["Tech", "ML"]
    },
    ...
  ]
}
```

---

## Part 2: Client-Side Vector Search

**File:** `_js/vector-search.js`

### What It Does

A browser-side module that:
1. Loads Transformers.js from CDN (~23MB)
2. Loads pre-computed embeddings from your site
3. Embeds user queries with the same model
4. Computes cosine similarity against all chunks
5. Returns ranked, deduplicated results

### Architecture: IIFE Module Pattern

```javascript
var VectorSearch = (function () {
  // Private state
  var state = {
    initialized: false,
    loading: false,
    extractor: null,
    embeddings: null,
    chunks: null,
  };

  // Public API
  return {
    initialize: initialize,
    search: search,
    isAvailable: isAvailable,
    isLoading: isLoading,
    getChunkCount: getChunkCount,
  };
})();
```

**Why IIFE?** Browserify compatibility + clean global namespace.

### CDN Loading Strategy

```javascript
function loadTransformers() {
  // Create inline ES module that imports from CDN
  var inlineCode =
    'import * as transformers from "' + TRANSFORMERS_CDN + '";\n' +
    'window.transformers = transformers;\n' +
    'window.dispatchEvent(new Event("transformers-loaded"));';

  var script = document.createElement("script");
  script.type = "module";
  script.textContent = inlineCode;
  document.head.appendChild(script);
}
```

**Why this approach?**
- Avoids bundling 23MB library into your bundle.js
- Uses browser's native ES module support
- CDN provides caching across sites
- Custom event signals completion

### Initialization Flow

```
User clicks "Semantic" mode
        │
        ▼
    initialize()
        │
        ├── 1. Fetch embeddings.json (~50-200KB)
        │
        ├── 2. Fetch chunks.json (~20-50KB)
        │
        ├── 3. Load Transformers.js from CDN (~23MB, cached)
        │
        └── 4. Create pipeline with model
                │
                ▼
         State: initialized = true
```

### Cosine Similarity Implementation

```javascript
function cosineSimilarity(a, b) {
  var dotProduct = 0;
  var normA = 0;
  var normB = 0;

  for (var i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

**Note:** Since embeddings are pre-normalized (`normalize: true`), this simplifies to just the dot product. But the full formula handles edge cases.

### Search Algorithm

```javascript
function search(query, topK, threshold) {
  // 1. Embed the query
  return state.extractor(query.trim(), {
    pooling: "mean",
    normalize: true,
  }).then(function (output) {
    var queryVector = Array.from(output.data);

    // 2. Compute similarity against ALL pre-computed embeddings
    var results = [];
    for (var i = 0; i < state.embeddings.length; i++) {
      var similarity = cosineSimilarity(queryVector, state.embeddings[i]);
      if (similarity >= threshold) {
        results.push({ chunk: state.chunks[i], similarity: similarity });
      }
    }

    // 3. Sort by similarity (descending)
    results.sort((a, b) => b.similarity - a.similarity);

    // 4. Deduplicate by URL (keep best chunk per post)
    var seen = {};
    var deduped = [];
    for (var j = 0; j < results.length; j++) {
      var url = results[j].chunk.url;
      if (!seen[url]) {
        seen[url] = true;
        deduped.push(results[j]);
        if (deduped.length >= topK) break;
      }
    }

    return deduped;
  });
}
```

**Key parameters:**
- `topK = 10` - Return top 10 results
- `threshold = 0.3` - Minimum similarity (30%) to include

---

## Part 3: Hybrid Search (Keyword + Semantic)

**File:** `_js/scripts.js` (search functions)

### Three Search Modes

| Mode | How it Works |
|------|--------------|
| **Keyword** | Pagefind (inverted index, exact matching) |
| **Semantic** | Vector similarity (meaning-based) |
| **Hybrid** | Both combined with RRF ranking |

### Reciprocal Rank Fusion (RRF)

The hybrid mode uses RRF to combine results from both systems:

```javascript
function combineResultsRRF(keywordResults, semanticResults) {
  var k = 60;  // RRF constant (standard value)
  var scores = {};

  // Score from keyword search
  keywordResults.forEach(function (result) {
    scores[result.url] = (scores[result.url] || 0) + 1 / (k + result.rank);
  });

  // Score from semantic search
  semanticResults.forEach(function (result) {
    scores[result.url] = (scores[result.url] || 0) + 1 / (k + result.rank);
  });

  // Sort by combined RRF score
  return Object.keys(scores)
    .map(url => ({ url, rrfScore: scores[url], ...metadata }))
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .slice(0, 10);
}
```

**Why RRF?**
- Rank-agnostic: doesn't need comparable scores between systems
- Handles different result set sizes gracefully
- Documents appearing in both lists get boosted
- k=60 is empirically proven effective

**Formula:** `RRF(d) = Σ 1/(k + rank(d))`

Example:
- Post ranked #1 in keyword, #3 in semantic:
  - RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
- Post ranked #1 in keyword only:
  - RRF = 1/(60+1) = 0.0164

---

## Part 4: The Embedding Model

### all-MiniLM-L6-v2

This is a **sentence transformer** from the sentence-transformers library, converted to ONNX format by Xenova for browser use.

**Properties:**
- 384-dimensional output vectors
- ~23MB model size (quantized)
- 6 transformer layers
- Based on MiniLM (distilled from BERT)
- Trained on 1B+ sentence pairs for semantic similarity

**Why this model?**
- Small enough for browser (~23MB)
- Fast inference (~50-100ms per query)
- Good quality for semantic similarity tasks
- Well-tested, widely used

### How Embeddings Work

1. **Tokenization:** Text → Token IDs
   - "How do I learn Rust?" → [101, 2129, 2079, 1045, 4553, 17682, 1029, 102]

2. **Embedding Layer:** Token IDs → Token Vectors
   - Each token → 384-dimensional vector

3. **Transformer Layers:** Contextualize
   - Each token's vector updated based on surrounding context
   - "bank" in "river bank" vs "bank account" → different vectors

4. **Pooling:** Many vectors → One vector
   - `mean` pooling: average all token vectors
   - Result: single 384-dim vector representing the whole text

5. **Normalization:** Scale to unit length
   - Enables dot product = cosine similarity
   - Makes similarity scores consistent

---

## Part 5: CI/CD Integration

**File:** `.github/workflows/deploy.yml`

```yaml
- name: Build Jekyll
  run: bundle exec jekyll build

- name: Build Pagefind search index
  run: npx pagefind --site _site

- name: Generate vector search embeddings
  run: node _scripts/generate-embeddings.js
```

**Order matters:**
1. Jekyll builds HTML to `_site/`
2. Pagefind indexes the HTML
3. Embeddings script reads `_posts/` and writes to `_site/search/`

---

## Performance Characteristics

| Operation | Time | Size |
|-----------|------|------|
| Build: Generate embeddings | ~30-60s | - |
| Client: Download embeddings.json | - | ~50-200KB |
| Client: Download chunks.json | - | ~20-50KB |
| Client: Load model (first time) | ~2-5s | ~23MB |
| Client: Load model (cached) | ~0.5-1s | - |
| Client: Search query | ~50-150ms | - |

---

## Why Semantic Search Matters

**Keyword search limitations:**
- "How do I learn Rust?" won't match "getting started with Rust programming"
- Synonyms don't match: "vehicle" vs "car"
- Typos break search

**Semantic search advantages:**
- Matches meaning, not just words
- "machine learning tutorial" matches "intro to ML"
- Natural language queries work

**Hybrid is best of both:**
- Exact matches still rank highly (keyword)
- Conceptually similar content discovered (semantic)
- RRF ensures neither dominates unfairly

---

## Summary

This implementation is a production-ready client-side semantic search:

1. **Build time:** Pre-compute embeddings for all blog chunks
2. **Runtime:** Load model from CDN, embed queries, compute similarity
3. **Hybrid:** Combine with Pagefind using RRF for best results

This is the same architectural pattern used by modern search engines, just scaled down for a static site.
