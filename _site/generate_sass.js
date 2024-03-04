import dartSass from 'sass';
import sourcemaps from 'source-map';
import autoprefixer from 'autoprefixer';
import postcss from 'postcss';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const src = {
  css: path.join(__dirname, "_sass/jekyll-sleek.scss"),
};
const dist = {
  css: path.join(__dirname, "_site/assets/css"),
};

function regenerateSass() {
  dartSass.render({
    file: src.css,
    outputStyle: "compressed",
    includePaths: [path.join(__dirname, "scss")],
  }, (err, result) => {
    if (err) {
      console.error(err);
      return;
    }

    postcss([autoprefixer])
      .process(result.css, { from: undefined })
      .then((prefixed) => {
        const mapOptions = { includeContent: false, sourceMap: true };
        prefixed.map = prefixed.map ? prefixed.map.toString() : '';

        fs.writeFile(path.join(dist.css, "main.css"), prefixed.css, (err) => {
          if (err) {
            console.error(err);
            return;
          }
          console.log("SASS compiled and prefixed.");
        });

        if (prefixed.map) {
          fs.writeFile(path.join(dist.css, "main.css.map"), prefixed.map, (err) => {
            if (err) {
              console.error(err);
              return;
            }
            console.log("Sourcemap generated.");
          });
        }
      });
  });
}

regenerateSass();
console.log("Done");
