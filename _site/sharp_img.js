import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// __dirname is not defined in ES module scope, so we create it
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Directory paths
const sourceDir = path.join(__dirname, '_img/posts/');
const destDir = path.join(__dirname, 'assets/img/posts/');

// Image formats to be generated
const formats = [
  { width: 230, suffix: '_placehold' },
  { width: 535, suffix: '_thumb' },
  { width: 535 * 2, suffix: '_thumb@2x' },
  { width: 575, suffix: '_xs' },
  { width: 767, suffix: '_sm' },
  { width: 991, suffix: '_md' },
  { width: 1999, suffix: '_lg' },
  { width: 1920, suffix: '' }, // max-width hero
];

async function modifyImages() {
  // Read the source directory for image files
  const files = fs.readdirSync(sourceDir).filter((file) => /\.(png|jpg)$/i.test(file));

  // Process each file
  for (const file of files) {
    const filePath = path.join(sourceDir, file);

    // Generate each format for current file
    for (const format of formats) {
      const outputFilePath = path.join(
        destDir,
        path.basename(file, path.extname(file)) + format.suffix + path.extname(file)
      );

      // Use Sharp to resize and apply settings
      await sharp(filePath)
        .resize({ width: format.width })
        .jpeg({ quality: 70, progressive: true })
        .withMetadata()
        .toFile(outputFilePath);
    }
  }

  console.log('Done');
}

modifyImages().catch(console.error);
