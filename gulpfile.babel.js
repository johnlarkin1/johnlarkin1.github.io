/* eslint-env node, process */
"use strict";

// Gulp and node
import gulp from "gulp";
import cp from "child_process";
import notify from "gulp-notify";
import size from "gulp-size";

// Basic workflow plugins
import browserSync from "browser-sync";
import browserify from "browserify";
import source from "vinyl-source-stream";
import buffer from "vinyl-buffer";
import clean from "gulp-clean";
import sass from "gulp-sass";
const jekyll = process.platform === "win32" ? "jekyll.bat" : "jekyll";
const messages = {
    jekyllBuild: "<span style=\"color: grey\">Running:</span> $ jekyll build"
};

// Performance workflow plugins
import htmlmin from "gulp-htmlmin";
import prefix from "gulp-autoprefixer";
import sourcemaps from "gulp-sourcemaps";
import uglify from "gulp-uglify";
import critical from "critical";
import sw from "sw-precache";

// Image Generation
import responsive from "gulp-responsive";
import rename from "gulp-rename";
import imagemin from "gulp-imagemin";

const src = {
  css: "_sass/jekyll-sleek.scss",
  js: "_js/scripts.js"
};
const dist = {
  css: "_site/assets/css",
  js: "_site/assets/js"
};

function handleErrors() {
  var args = Array.prototype.slice.call( arguments );
  notify.onError( {
    title: "Compile Error",
    message: "<%= error.message %>"
  } ).apply( this, args );
  this.emit( "end" ); // Keep gulp from hanging on this task
}

// SASS
gulp.task( "sass", () => {
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
} );

//  JS
gulp.task( "js", () => {
  return browserify( src.js, { debug: true, extensions: [ "es6" ] } )
    .transform( "babelify", { presets: [ "es2015" ] } )
    .bundle()
    .on( "error", handleErrors )
    .pipe( source( "bundle.js" ) )
    .pipe( buffer() )
    .pipe( sourcemaps.init( { loadMaps: true } ) )
    .pipe( uglify() )
    .pipe( sourcemaps.write( "./maps" ) )
    .pipe( size() )
    .pipe( gulp.dest( dist.js ) )
    .pipe( browserSync.reload( { stream: true } ) )
    .pipe( gulp.dest( "assets/js" ) );
} );

gulp.task( "critical", done => {
  critical.generate( {
    base: "_site/",
    src: "index.html",
    css: [ "assets/css/main.css" ],
    dimensions: [ {
      width: 320,
      height: 480
    }, {
      width: 768,
      height: 1024
    }, {
      width: 1280,
      height: 960
    } ],
    dest: "../_includes/critical.css",
    minify: true,
    extract: false,
    ignore: [ "@font-face" ]
  } );
  done();
} );

// Minify HTML
gulp.task( "html", done => {
    gulp.src( "./_site/index.html" )
      .pipe( htmlmin( { collapseWhitespace: true } ) )
      .pipe( gulp.dest( "./_site" ) );
    gulp.src( "./_site/*/*html" )
      .pipe( htmlmin( { collapseWhitespace: true } ) )
      .pipe( gulp.dest( "./_site/./" ) );
    done();
} );

// Service Worker
gulp.task( "sw", () => {
  const rootDir = "./";
  const distDir = "./_site";

  return sw.write( `${rootDir}/sw.js`, {
    staticFileGlobs: [ distDir + "/**/*.{js,html,css,png,jpg,svg}" ],
    stripPrefix: distDir
  } );
} );

// Images
gulp.task( "img", () => {
  return gulp.src( "_img/posts/*.{png,jpg}" )
    .pipe( responsive( {
        "*": [ // For all the images in the posts folder
          {
            width: 230,
            rename: { suffix: "_placehold" }
          },
          { // thubmnail
            width: 535,
            rename: { suffix: "_thumb" }
          },
          { // thumbnail @2x
            width: 535 * 2,
            rename: { suffix: "_thumb@2x" }
          },
          {
            width: 575,
            rename: { suffix: "_xs" }
          },
          {
            width: 767,
            rename: { suffix: "_sm" }
          },
          {
            width: 991,
            rename: { suffix: "_md" }
          },
          {
            width: 1999,
            rename: { suffix: "_lg" }
          },
          { // max-width hero
            width: 1920
          }
        ]
      },
      {
        quality: 70,
        progressive: true,
        withMetadata: false,
        errorOnEnlargement: false,
        errorOnUnusedConfig: false,
        silent: true
      } ) )
      .pipe( imagemin() )
      .pipe( gulp.dest( "assets/img/posts/" ) );
} );

// Build the Jekyll Site
gulp.task( "jekyll-build", done =>  {
    browserSync.notify( messages.jekyllBuild );
    return cp.spawn( jekyll, [ "build" ], { stdio: "inherit" } )
        .on( "close", done );
} );

// Rebuild Jekyll & do page reload
gulp.task( "rebuild",
  gulp.series( [ "jekyll-build" ], done => {
    browserSync.reload();
    done();
  } )
);

gulp.task( "clean", () => {
  return gulp.src( "_site", { read: false, allowEmpty: true } )
    .pipe( clean() );
} );

gulp.task( "serve", function() {
  return browserSync( {
    server: {
      baseDir: "_site"
    }
  } );
} );

gulp.task( "styles", gulp.series( [ "sass", "critical" ] ) );

gulp.task( "watch", () => {
  gulp.watch( "_sass/**/*.scss", gulp.series( "styles" ) );
  gulp.watch( [
    "*.html",
    "_layouts/*.html",
    "_includes/*.html",
    "_posts/*.md",
    "pages_/*.md",
    "_include/*html"
  ], gulp.series( "rebuild" ) );
  gulp.watch( "_js/**/*.js", gulp.series( "js" ) );
} );

gulp.task( "build", gulp.series( [
  "clean",
  gulp.parallel( [ "sass", "js", "img" ] ),
  "jekyll-build",
  "critical",
  "sw"
] ) );

gulp.task( "default", gulp.series( [
  "build",
  "serve",
  "watch"
] ) );
