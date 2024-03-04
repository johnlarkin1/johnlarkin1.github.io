import browserify from 'browserify';
import fs from 'fs';
import path from 'path';
import babelify from 'babelify';
import browserSync from 'browser-sync';
import { pipeline } from 'stream';
import uglifyify from 'uglifyify';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const src = {
  js: "_js/scripts.js"
};
const dist = {
  js: "_site/assets/js"
};

function bundleJS() {
  const b = browserify({
    entries: path.join(__dirname, src.js),
    debug: true,
    transform: [babelify.configure({ /* your babel config here */ })]
  });

  b.transform(uglifyify, { global: true });

  b.bundle((err, buffer) => {
    if (err) {
      console.error(err);
      return;
    }

    // Ensure dist path exists
    const distPath = path.join(__dirname, dist.js);
    if (!fs.existsSync(path.dirname(distPath))) {
      fs.mkdirSync(path.dirname(distPath), { recursive: true });
    }

    // Output bundle file
    fs.writeFile(path.join(distPath, 'bundle.js'), buffer, (err) => {
      if (err) {
        console.error('Error writing bundle:', err);
      } else {
        console.log('Bundle created successfully');
      }
    });

    // Optionally, setup browserSync for live reloading
    browserSync.init({
      server: path.join(__dirname, "./_site")
    });

    browserSync.reload();
  });
}

bundleJS();
