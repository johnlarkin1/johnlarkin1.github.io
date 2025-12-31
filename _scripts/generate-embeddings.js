/**
 * Generate embeddings for blog posts at build time.
 *
 * This script:
 * 1. Reads all markdown posts from _posts/
 * 2. Chunks them into ~400 token passages
 * 3. Generates embeddings using all-MiniLM-L6-v2
 * 4. Outputs to _site/search/embeddings.json and chunks.json
 *
 * Run: node _scripts/generate-embeddings.js
 */

import { pipeline } from '@huggingface/transformers';
import matter from 'gray-matter';
import { glob } from 'glob';
import fs from 'fs/promises';
import path from 'path';

// Configuration
const CONFIG = {
  model: 'Xenova/all-MiniLM-L6-v2',
  dimensions: 384,
  targetTokens: 400,
  minChunkTokens: 100,
  maxChunkTokens: 500,
  previewLength: 300,
  postsDir: '_posts',
  outputDir: '_site/search',
};

/**
 * Estimate token count (rough: ~4 chars per token for English)
 */
function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

/**
 * Strip markdown/HTML to plain text
 */
function stripMarkdown(content) {
  return content
    // Remove code blocks
    .replace(/```[\s\S]*?```/g, '')
    // Remove inline code
    .replace(/`[^`]+`/g, '')
    // Remove images
    .replace(/!\[.*?\]\(.*?\)/g, '')
    // Convert links to just text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Remove HTML tags
    .replace(/<[^>]+>/g, '')
    // Remove header markers
    .replace(/^#+\s*/gm, '')
    // Remove bold/italic/strikethrough
    .replace(/[*_~]+/g, '')
    // Remove list markers
    .replace(/^\s*[-*+]\s+/gm, '')
    // Remove numbered lists
    .replace(/^\s*\d+\.\s+/gm, '')
    // Remove blockquotes
    .replace(/^\s*>\s*/gm, '')
    // Remove horizontal rules
    .replace(/^[-*_]{3,}\s*$/gm, '')
    // Collapse multiple newlines
    .replace(/\n{3,}/g, '\n\n')
    // Remove Jekyll/Liquid tags
    .replace(/\{%[\s\S]*?%\}/g, '')
    .replace(/\{\{[\s\S]*?\}\}/g, '')
    .trim();
}

/**
 * Build URL from filename and frontmatter
 */
function buildUrl(filePath) {
  const filename = path.basename(filePath, '.md');
  // Format: YYYY-MM-DD-slug.md -> /YYYY/slug/
  const match = filename.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)$/);
  if (match) {
    return `/${match[1]}/${match[4]}/`;
  }
  return `/${filename}/`;
}

/**
 * Split text into chunks, respecting section boundaries
 */
function chunkContent(content, title) {
  const chunks = [];
  const plainContent = stripMarkdown(content);

  // Split by markdown headers first
  const sections = content.split(/^(#{1,6}\s+.+)$/gm);

  let currentSection = title;
  let buffer = '';

  for (const part of sections) {
    // Check if this is a header
    if (part.match(/^#{1,6}\s+/)) {
      // Save current buffer as chunk(s)
      if (buffer.trim()) {
        chunks.push(...splitBuffer(buffer, currentSection));
      }
      currentSection = part.replace(/^#+\s*/, '').trim();
      buffer = '';
    } else {
      buffer += part;
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    chunks.push(...splitBuffer(buffer, currentSection));
  }

  return chunks;
}

/**
 * Split a text buffer into appropriately-sized chunks
 */
function splitBuffer(text, section) {
  const plainText = stripMarkdown(text);
  const tokens = estimateTokens(plainText);

  // If small enough, return as single chunk
  if (tokens <= CONFIG.maxChunkTokens) {
    if (tokens >= CONFIG.minChunkTokens) {
      return [{ section, text: plainText }];
    }
    return [];
  }

  // Split on paragraphs
  const paragraphs = plainText.split(/\n\n+/).filter(p => p.trim());
  const result = [];
  let current = '';

  for (const para of paragraphs) {
    const combined = current ? `${current}\n\n${para}` : para;
    const combinedTokens = estimateTokens(combined);

    if (combinedTokens <= CONFIG.targetTokens) {
      current = combined;
    } else {
      // Save current and start new
      if (current && estimateTokens(current) >= CONFIG.minChunkTokens) {
        result.push({ section, text: current });
      }
      current = para;
    }
  }

  // Don't forget the last chunk
  if (current && estimateTokens(current) >= CONFIG.minChunkTokens) {
    result.push({ section, text: current });
  }

  return result;
}

/**
 * Main function
 */
async function main() {
  console.log('Vector Search Embedding Generator');
  console.log('==================================\n');

  // Find all posts
  console.log('Finding posts...');
  const postFiles = await glob(`${CONFIG.postsDir}/*.md`);
  console.log(`Found ${postFiles.length} posts\n`);

  if (postFiles.length === 0) {
    console.log('No posts found, exiting.');
    return;
  }

  // Process posts into chunks
  console.log('Processing posts into chunks...');
  const allChunks = [];

  for (const file of postFiles) {
    try {
      const content = await fs.readFile(file, 'utf-8');
      const { data: frontmatter, content: body } = matter(content);

      // Skip drafts, hidden, or encrypted posts
      if (frontmatter.draft || frontmatter.hidden || frontmatter.encrypted) {
        console.log(`  Skipping: ${frontmatter.title || file} (draft/hidden/encrypted)`);
        continue;
      }

      const post = {
        title: frontmatter.title || path.basename(file, '.md'),
        url: buildUrl(file),
        categories: frontmatter.categories || [],
        summary: frontmatter.summary || '',
      };

      // Add summary as first chunk if available
      if (post.summary) {
        allChunks.push({
          ...post,
          section: 'Summary',
          text: post.summary,
        });
      }

      // Chunk the content
      const chunks = chunkContent(body, post.title);
      for (const chunk of chunks) {
        allChunks.push({
          title: post.title,
          url: post.url,
          categories: post.categories,
          section: chunk.section,
          text: chunk.text,
        });
      }

      console.log(`  ${post.title}: ${chunks.length + (post.summary ? 1 : 0)} chunks`);
    } catch (err) {
      console.error(`  Error processing ${file}:`, err.message);
    }
  }

  console.log(`\nTotal chunks: ${allChunks.length}\n`);

  if (allChunks.length === 0) {
    console.log('No chunks generated, exiting.');
    return;
  }

  // Load the embedding model
  console.log('Loading embedding model...');
  console.log(`  Model: ${CONFIG.model}`);
  const extractor = await pipeline('feature-extraction', CONFIG.model, {
    quantized: true,
  });
  console.log('  Model loaded!\n');

  // Generate embeddings
  console.log('Generating embeddings...');
  const embeddings = [];
  const chunkMeta = [];

  for (let i = 0; i < allChunks.length; i++) {
    const chunk = allChunks[i];

    try {
      const output = await extractor(chunk.text, {
        pooling: 'mean',
        normalize: true,
      });

      embeddings.push(Array.from(output.data));
      chunkMeta.push({
        id: i,
        url: chunk.url,
        title: chunk.title,
        section: chunk.section,
        text: chunk.text.slice(0, CONFIG.previewLength),
        categories: chunk.categories,
      });

      if ((i + 1) % 25 === 0 || i === allChunks.length - 1) {
        console.log(`  Progress: ${i + 1}/${allChunks.length} chunks`);
      }
    } catch (err) {
      console.error(`  Error embedding chunk ${i}:`, err.message);
    }
  }

  console.log('\nEmbeddings generated!\n');

  // Ensure output directory exists
  await fs.mkdir(CONFIG.outputDir, { recursive: true });

  // Write embeddings file
  const embeddingsData = {
    version: '1.0',
    model: CONFIG.model,
    dimensions: CONFIG.dimensions,
    generated: new Date().toISOString(),
    count: embeddings.length,
    embeddings,
  };

  const embeddingsPath = path.join(CONFIG.outputDir, 'embeddings.json');
  await fs.writeFile(embeddingsPath, JSON.stringify(embeddingsData));
  const embeddingsSize = (await fs.stat(embeddingsPath)).size;
  console.log(`Written: ${embeddingsPath} (${(embeddingsSize / 1024).toFixed(1)} KB)`);

  // Write chunks metadata file
  const chunksData = {
    version: '1.0',
    count: chunkMeta.length,
    chunks: chunkMeta,
  };

  const chunksPath = path.join(CONFIG.outputDir, 'chunks.json');
  await fs.writeFile(chunksPath, JSON.stringify(chunksData));
  const chunksSize = (await fs.stat(chunksPath)).size;
  console.log(`Written: ${chunksPath} (${(chunksSize / 1024).toFixed(1)} KB)`);

  console.log('\nDone!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
