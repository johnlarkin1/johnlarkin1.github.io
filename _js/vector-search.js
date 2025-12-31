/**
 * Vector Search Module
 *
 * Client-side semantic search using pre-computed embeddings and
 * Transformers.js loaded from CDN. Compatible with Browserify.
 * Uses Promises (not async/await) to avoid regenerator-runtime dependency.
 *
 * Usage:
 *   VectorSearch.initialize(progressCallback).then(...)
 *   VectorSearch.search(query, topK, threshold).then(...)
 *   VectorSearch.isAvailable()
 *   VectorSearch.isLoading()
 */

var VectorSearch = (function () {
  "use strict";

  // CDN URL for Transformers.js
  var TRANSFORMERS_CDN =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
  var MODEL_ID = "Xenova/all-MiniLM-L6-v2";
  var DIMENSIONS = 384;

  // State
  var state = {
    initialized: false,
    loading: false,
    error: null,
    extractor: null,
    embeddings: null,
    chunks: null,
  };

  /**
   * Load Transformers.js from CDN via script tag
   * This avoids Browserify trying to resolve the import
   */
  function loadTransformers() {
    if (window.transformers) {
      console.log("[VectorSearch] Using cached Transformers module");
      return Promise.resolve(window.transformers);
    }

    console.log("[VectorSearch] Loading Transformers.js from CDN...");

    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.type = "module";

      // Create an inline module that imports and exposes transformers
      var inlineCode =
        'import * as transformers from "' + TRANSFORMERS_CDN + '";\n' +
        'window.transformers = transformers;\n' +
        'window.dispatchEvent(new Event("transformers-loaded"));';

      script.textContent = inlineCode;

      var onLoad = function () {
        window.removeEventListener("transformers-loaded", onLoad);
        if (window.transformers) {
          console.log("[VectorSearch] Transformers.js loaded successfully");
          resolve(window.transformers);
        } else {
          reject(new Error("Transformers module not found after load"));
        }
      };

      var onError = function (err) {
        console.error("[VectorSearch] Script load error:", err);
        reject(new Error("Failed to load Transformers.js script"));
      };

      window.addEventListener("transformers-loaded", onLoad);
      script.onerror = onError;

      document.head.appendChild(script);
    });
  }

  /**
   * Fetch pre-computed embeddings and chunk metadata
   */
  function loadSearchData() {
    var baseUrl = window.location.origin;
    console.log("[VectorSearch] Loading search data from:", baseUrl + "/search/");

    return Promise.all([
      fetch(baseUrl + "/search/embeddings.json"),
      fetch(baseUrl + "/search/chunks.json"),
    ]).then(function (responses) {
      var embeddingsRes = responses[0];
      var chunksRes = responses[1];

      console.log("[VectorSearch] Embeddings response:", embeddingsRes.status);
      console.log("[VectorSearch] Chunks response:", chunksRes.status);

      if (!embeddingsRes.ok) {
        throw new Error("Failed to load embeddings: " + embeddingsRes.statusText);
      }
      if (!chunksRes.ok) {
        throw new Error("Failed to load chunks: " + chunksRes.statusText);
      }

      return Promise.all([embeddingsRes.json(), chunksRes.json()]);
    }).then(function (data) {
      var embeddingsData = data[0];
      var chunksData = data[1];

      console.log("[VectorSearch] Loaded", embeddingsData.embeddings.length, "embeddings");
      console.log("[VectorSearch] Loaded", chunksData.chunks.length, "chunks");

      return {
        embeddings: embeddingsData.embeddings,
        chunks: chunksData.chunks,
      };
    });
  }

  /**
   * Initialize the vector search system.
   * Call this when user switches to semantic mode.
   *
   * @param {Function} progressCallback - Called with { stage, message, progress }
   * @returns {Promise<boolean>} - True if initialized successfully
   */
  function initialize(progressCallback) {
    console.log("[VectorSearch] Initialize called, state:", {
      initialized: state.initialized,
      loading: state.loading,
    });

    if (state.initialized) return Promise.resolve(true);
    if (state.loading) return Promise.resolve(false);

    state.loading = true;
    state.error = null;

    var onProgress = progressCallback || function () {};

    onProgress({
      stage: "data",
      message: "Loading search index...",
      progress: 0,
    });

    // Load search data first (smaller, faster)
    return loadSearchData()
      .then(function (searchData) {
        state.embeddings = searchData.embeddings;
        state.chunks = searchData.chunks;

        onProgress({
          stage: "model",
          message: "Loading AI model (~23MB)...",
          progress: 30,
        });

        // Load Transformers.js from CDN
        return loadTransformers();
      })
      .then(function (tf) {
        onProgress({
          stage: "model",
          message: "Initializing model...",
          progress: 50,
        });

        // Create feature extraction pipeline
        return tf.pipeline("feature-extraction", MODEL_ID, {
          dtype: "fp32",
          progress_callback: function (info) {
            if (info.status === "progress" && info.progress) {
              var pct = 50 + info.progress * 0.5;
              onProgress({
                stage: "model",
                message: "Downloading model...",
                progress: Math.min(pct, 99),
              });
            }
          },
        });
      })
      .then(function (extractor) {
        state.extractor = extractor;

        onProgress({
          stage: "ready",
          message: "Ready!",
          progress: 100,
        });

        state.initialized = true;
        state.loading = false;
        console.log("[VectorSearch] Initialization complete!");
        return true;
      })
      .catch(function (err) {
        state.error = err.message;
        state.loading = false;
        onProgress({
          stage: "error",
          message: err.message,
          progress: 0,
        });
        console.error("[VectorSearch] Initialization failed:", err);
        return false;
      });
  }

  /**
   * Compute cosine similarity between two vectors
   */
  function cosineSimilarity(a, b) {
    var dotProduct = 0;
    var normA = 0;
    var normB = 0;

    for (var i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Search for similar chunks
   *
   * @param {string} query - User's search query
   * @param {number} topK - Number of results to return (default 10)
   * @param {number} threshold - Minimum similarity threshold (default 0.3)
   * @returns {Promise<Array>} - Array of { chunk, similarity } objects
   */
  function search(query, topK, threshold) {
    topK = topK || 10;
    threshold = threshold || 0.3;

    if (!state.initialized) {
      return Promise.reject(new Error("Vector search not initialized"));
    }

    if (!query || query.trim().length === 0) {
      return Promise.resolve([]);
    }

    // Compute query embedding
    return state.extractor(query.trim(), {
      pooling: "mean",
      normalize: true,
    }).then(function (output) {
      var queryVector = Array.from(output.data);

      // Compute similarities against all chunks
      var results = [];
      for (var i = 0; i < state.embeddings.length; i++) {
        var similarity = cosineSimilarity(queryVector, state.embeddings[i]);

        if (similarity >= threshold) {
          results.push({
            chunk: state.chunks[i],
            similarity: similarity,
          });
        }
      }

      // Sort by similarity descending
      results.sort(function (a, b) {
        return b.similarity - a.similarity;
      });

      // Deduplicate by URL (keep highest scoring chunk per post)
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

  /**
   * Check if vector search is initialized and ready
   */
  function isAvailable() {
    return state.initialized;
  }

  /**
   * Check if currently loading
   */
  function isLoading() {
    return state.loading;
  }

  /**
   * Get error message if any
   */
  function getError() {
    return state.error;
  }

  /**
   * Get the number of indexed chunks
   */
  function getChunkCount() {
    return state.chunks ? state.chunks.length : 0;
  }

  // Public API
  return {
    initialize: initialize,
    search: search,
    isAvailable: isAvailable,
    isLoading: isLoading,
    getError: getError,
    getChunkCount: getChunkCount,
  };
})();

// Export for Browserify (CommonJS)
if (typeof module !== "undefined" && module.exports) {
  module.exports = VectorSearch;
}

// Also attach to window for direct access
if (typeof window !== "undefined") {
  window.VectorSearch = VectorSearch;
}
