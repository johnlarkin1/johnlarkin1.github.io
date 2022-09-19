import gulp from 'gulp';
import browserify from "browserify";
import source from "vinyl-source-stream";
import buffer from "vinyl-buffer";
import sourcemaps from "gulp-sourcemaps";
import uglify from "gulp-uglify";
import size from "gulp-size";
import browserSync from "browser-sync";
import notify from "gulp-notify";

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
};

function regenerateJS() {
  return browserify( src.js, { debug: true, extensions: [ "es6" ] } )
    // .transform( "babelify", { presets: [ "es2015" ] } )
    .transform( "babelify" )
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
};

regenerateJS();
console.log("Done")