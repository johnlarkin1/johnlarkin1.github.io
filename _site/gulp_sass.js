
import gulp from 'gulp';
import dartSass from 'sass'
import gulpSass from 'gulp-sass'
const sass = gulpSass(dartSass)

import sourcemaps from "gulp-sourcemaps";
import browserSync from "browser-sync";
import prefix from "gulp-autoprefixer";
import rename from "gulp-rename";

const src = {
  css: "_sass/jekyll-sleek.scss",
  js: "_js/scripts.js"
};
const dist = {
  css: "_site/assets/css",
  js: "_site/assets/js"
};

function regenerateSass() {
  return gulp.src( src.css )
    .pipe( sourcemaps.init() )
    .pipe( sass( {
      outputStyle: "compressed",
      includePaths: [ "scss" ],
      onError: browserSync.notify
    } ).on( "error", sass.logError ) )
    .pipe( sourcemaps.write( { includeContent: false } ) )
    .pipe( sourcemaps.init( { loadMaps: true } ) )
    .pipe( prefix() )
    .pipe( sourcemaps.write( "./" ) )
    .pipe( rename( { basename: "main" } ) )
    .pipe( gulp.dest( dist.css ) )
    .pipe( browserSync.reload( { stream: true } ) )
    .pipe( gulp.dest( "assets/css" ) );
};
regenerateSass();
console.log("Done");

