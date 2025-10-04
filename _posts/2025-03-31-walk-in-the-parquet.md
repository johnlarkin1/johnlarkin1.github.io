---
title: "Walk in the Parquet"
layout: post
featured-gif: walk-in-the-parquet
pinned: true
categories: [⭐️ Favorites, Dev]
summary: "Visualize, query, and analyze your Parquet files through a lightweight Tauri desktop app"
favorite: true
---

At [Mojo][mojo], we use [Parquet][parquet] files to store some of our simulation data. I - however - have been increasingly frustrated by the lack of support on MacOS to natively view them. They are (normally) compressed through a [snappy] algorithm, and Apple doesn't have a native application to open them.

So I decided to build one - to help myself out, my teammates at work out, and hopefully some other random engineers out in the wild. In the very least, this blog post will detail how you can build your own desktop application, specifically in this case using [Tauri][tauri].

There's more information (i.e. lame marketing) here: [walkintheparquet.com](https://www.walkintheparquet.com/). Here's an iframe if don't want to leave this page:

<div style="text-align: center;">
<div style="max-width: 800px; margin: 0 auto; box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05); border-radius: 4px; overflow: hidden; background-color: #1a1a1a; padding: 15px;">
<iframe src="https://www.walkintheparquet.com/" 
        width="100%" 
        height="450px" 
        frameborder="0" 
        allowfullscreen></iframe>
</div>
</div>

<br/>

<div class="markdown-alert markdown-alert-tip">
<p>Also! If you have feature requests or issues, you can head over to the Canny board for this project and leave some notes. There's a link at the bottom of the main website (/ iframe above), or you can go <a href="https://walk-in-the-parquet.canny.io/">here</a>. Obviously feel free to email me too!</p>
</div>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Driving Motivation](#driving-motivation)
  - [Are there really no other solutions?](#are-there-really-no-other-solutions)
- [What is Parquet?](#what-is-parquet)
- [What's the Problem?](#whats-the-problem)
- [Engineering + Design](#engineering--design)
  - [Desktop Decisions](#desktop-decisions)
  - [Challenges and Conquests](#challenges-and-conquests)
    - [Documentation](#documentation)
    - [Supporting Structs](#supporting-structs)
    - [App Store Annoyance](#app-store-annoyance)
- [Conclusion](#conclusion)
  - [Kudos](#kudos)

# Driving Motivation

To inspire you a little bit, here's what we've built. This is also available (again) [here][walk-in-the-parquet-site], but also available on the [App Store][walk-in-the-parquet-app].

![walk-in-the-parquet-show](/images/walk-in-the-parquet/slideshow.gif){: .center-image}

And here it is in the [App Store][walk-in-the-parquet-app]:

![app-store](/images/walk-in-the-parquet/app-store.png){: .center-small }

Again, this blog post is going to talk a bit more about the actual building process, but if you want to see more about the product and download it, head over to the [main website][walk-in-the-parquet-site].

And before we go any deeper, I know there's this question...

## Are there really no other solutions?

Yeah I mean there's this:

<div style="display: flex; justify-content: center; align-items: center;">
    <div style="margin: 10px;">
        <img src="{{ '/images/walk-in-the-parquet/other-app-store-pt1.png' | prepend: site.baseurl }}" alt="Image 1" style="width: 300px; height: auto;">
    </div>
    <div style="margin: 10px;">
        <img src="{{ '/images/walk-in-the-parquet/other-app-store-pt2.png' | prepend: site.baseurl }}" alt="Image 2" style="width: 600px; height: auto;">
    </div>
</div>
<div class="image-caption">Can't wait until people are leaving me the same reviews. But hey mine is free. And it's a v1. </div>
<br/>

Which.... yeah don't love people trying to charge for that.

The one that I've seen the most is this: [https://www.parquet-viewer.com/](https://www.parquet-viewer.com/), which I use but they _also_ have some dumb paywalling features. And to be honest, I don't really love uploading potentially sensitive files to the web.

And finally, it seems like the third alternative is a VSCode Extension, but the downside there is it's just `json` I believe. Again, totally fine - I won't be upset if you want to do that. It's not as smooth to query or see some top level analytics, but c'est la vie.

# What is Parquet?

Let's back up a little bit for those not even familiar with Parquet. I won't go into too much detail because there's enough information out there on the web, but I'll give a high level overview.

[Parquet][parquet] is a file type that is optimized and used prevalently for data processing frameworks. It was introduced by Apache and has numerous benefits, some of which I'll get into below.

The big distinguisher for Parquet is that it's a columnar storage format. This has some big wins especially in terms of compression. For example, if you were storing numerous of the same types of player-ids then you can have a vastly higher compression rate given the column is going to have a lot of redundancy.

These columns are then going to be split into row-groups. Row groups are just logical partitions of the data into separate rows.

These row groups are incredibly clutch when it comes to parallel processing because it lets them be read in parallel. Furthermore, the optimization is done so that only _relevant_ row groups are read.

So this then becomes a bit of a hyperparamter performance optimization question right? What's the ideal number to set for your row-groups? Well... yeah this is a bit of experimentation. There's surprisingly little documentation around what is best, but generally you want a trade off between compression and performance. Generally, I've seen people suggesting you want your rowgroup to be around 128MB to 512MB. AWS seems to default it to be 128MB [here](https://docs.aws.amazon.com/prescriptive-guidance/latest/apache-iceberg-on-aws/best-practices-read.html).

You can think about it like so:

| Row Group Size         | Pros                                                                 | Cons                                                       |
| ---------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Larger Row Groups**  | - Better read performance (fewer metadata reads, more sequential IO) | - Higher memory usage during write                         |
|                        | - Better compression (larger chunks compress more efficiently)       | - Slower write performance if memory is constrained        |
| **Smaller Row Groups** | - Lower memory usage during write                                    | - Slower reads (more metadata overhead and disk seeks)     |
|                        | - Faster writes in streaming or frequent-flush scenarios             | - Worse compression                                        |
|                        |                                                                      | - Less effective filtering (min/max stats less meaningful) |

There's also other fields like column chunks and pages. The best overview I've seen is actually from [CelerData][celerdata] [here][celerdata-parquet].

This image from CelerData does a good job breaking out the different parts of the underlying structure:

![parquet-file-layout](https://parquet.apache.org/images/FileLayout.gif){: .center-shrink }

<div class="image-caption">Full credit to CelerData for the image</div>
<br/>

But! If you don't like that one, noooo worries. Databricks has $62B valuation and they also wrote about it [here][databricks-parquet]. So feel free to check out some other links.

# What's the Problem?

Well, the problem that I wanted to address is that there's not a great way to open these files. I discussed some alternatives and their downsides above, but it's dumb that I couldn't have everything local (excluding a paywalled App Store app) or I'd have to upload things to the web and some dude's random server.

![app-store](/images/walk-in-the-parquet/no-default-application.png){: .center-small }

The other problem? I haven't worked with Rust in awhile, and I still desperately want to get better at it, so that was the selfish motivation. It's a borderline smooth transition into the next section.

# Engineering + Design

## Desktop Decisions

Ah the desktop application game - what a question.

Now I've worked with [Electron][electron] at [Dropbox][dropbox] so I was familiar with generally that architecture and paradigm. It has been a minute since I've dealt with [preload scripts][electron-preload] or the [ipcMain vs ipcRenderer distinction][electron-ipc].

The downside (in this case) and why I didn't choose Electron was because I didn't really want an all Typescript backend.

Truthfully, I really wanted a Python backend, both because that's what I'm best at, but also because I wanted to use [`duckdb`][duckdb] for loading in the files and doing analysis quickly, on-disk, and keeping things lightweight. I haven't loaded Parquet files in Typescript before, and I've also been seeing more about [Tauri][tauri] and figured that it was a better use case.

Additionally, I know from googling Apache Parquet documentation (we integrate in [Golang][golang], [C++][cpp], and [Python][python] all at work) that they DO have Rust support. I know this because I personally think that most of the documentation put out from Apache blows. The other noticeable benefit of using Rust and Tauri is that Tauri is a lot lighter weight of a desktop application.

[Coditation][coditation] sums it up pretty well below ([ref][coditation-ref]):

> Tauri is designed to be more lightweight and faster than Electron, as it uses less memory and CPU resources, which means that Tauri is designed to run more efficiently than Electron.
>
> Tauri uses Rust as a native layer instead of JavaScript and web technologies, which results in lower memory usage and CPU usage compared to Electron. Additionally, Tauri is also designed to be more lightweight overall, which means that it has less overhead and a smaller binary size than Electron.

In other words,

| Feature      | Electron      | Tauri (what I picked) |
| ------------ | ------------- | --------------------- |
| Backend Lang | JavaScript/TS | Rust                  |
| Binary Size  | Large         | Small                 |
| Memory Usage | Higher        | Lower                 |

## Challenges and Conquests

### Documentation

So this is a core part of it, but recently, I have been one of the many to get hit with the "yoooo how much did you vibe code". There have been [many][vibe-coding-meme1] [good][vibe-coding-meme2] [memes][vibe-coding-meme3] about this.

The thing is **I basically did vibecode the entire website. NextJS, simple lightweight static frontend is a perfect use case for it**. I'm not at all a frontend designer and so yeah, of course I'm not going to be ripping that manually or going into Figma first or anything like that. So that was lovely. Way faster and way quicker to ship.

The interesting part (at least for me) was how best to architect this with Tauri and have that handoff. The challenges were about that design, as well as the blatant lack of LLMs that are trained on Tauriv2 and the parquet versions I was using in Rust.

Specifically,

```
arrow = "54.3.0"
arrow-schema = "54.3.0"
parquet = "54.3.0"
```

these crates had virtually no LLM support (what a breath of fresh air).

### Supporting Structs

As a result, it meant using the documentation and figuring out exactly why some of my string data was being parsed as a `Utf8View` vs a `Utf8`.

In terms of code, it meant that I had in my `sql.rs` parsing engine, a match statement (one of Rust's best features imo) like this:

```rust
// ... many more types before

        DataType::Int64 => {
            let array = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| QueryError::Other("Failed to downcast to Int64Array".to_string()))?;
            serde_json::Value::Number(serde_json::Number::from(array.value(row_idx)))
        }

        DataType::UInt8 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| QueryError::Other("Failed to downcast to UInt8Array".to_string()))?;
            serde_json::Value::Number(serde_json::Number::from(array.value(row_idx) as u64))
        }

        DataType::UInt16 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| {
                    QueryError::Other("Failed to downcast to UInt16Array".to_string())
                })?;
            serde_json::Value::Number(serde_json::Number::from(array.value(row_idx) as u64))
        }

// ... many more types after
```

There are numerous `DataType`s that get pulled in with `use datafusion::arrow::datatypes::*;`. I tried to handle most, but yeah of course Parquet files can be increasingly complex so as a `v1.0.0` I am not promising to have entire support. There is basic support for nested structures as seen here:

![app-store](/images/walk-in-the-parquet/nested-structure.png){: .center-shrink }

However, handling this recursively is a bit of a challenge. I am expecting there to be some corner cases that I missed.

### App Store Annoyance

By far however, the biggest learning I had was about bundling up a package for the App Store and the numerous steps to get that going.

There's already quite a bit out there about [notarization][apple-notarization] and [code-signing][apple-codesigning], but I think the most helpful thing was putting this all in `post-build.sh` script.

So basically after running this:

```bash
╭─johnlarkin@Mac ~/Documents/coding/walk-in-the-parquet ‹main›
╰─➤  npm run tauri build -- --target universal-apple-darwin
```

You can get a [fat client][tauri-thick] from Tauri that will support both Apple silicon and Intel-based archs.

After doing that, then what I finally got set up was this type of setup (thanks Claude for some of the emojis):

```bash
#!/bin/bash
set -euo pipefail

APP_NAME="Walk in the Parquet"
ENTITLEMENTS_PATH="src-tauri/entitlements.plist"
PKG_NAME="WalkInTheParquet.pkg"

# I was trying to automatically detect the app-path / dmg-path but what was happening
# was I was occassionalyl picking the wrong app / dmg and then yeah i was too lazy to fix this
# APP_PATH=$(find src-tauri/target/universal-apple-darwin -type d -name "$APP_NAME.app" | head -n 1)
APP_PATH="src-tauri/target/universal-apple-darwin/release/bundle/macos/Walk in the Parquet.app"
DMG_PATH="src-tauri/target/universal-apple-darwin/release/bundle/dmg/Walk in the Parquet_1.0.0_universal.dmg"

# you basically need `APPLE_ISSUER_ID`, `APPLE_PARQUET_KEY_ID`
# set up in your env
if [[ -z "${APPLE_ISSUER_ID:-}" ]]; then
  echo "🚨 Error: Environment variable APPLE_ISSUER_ID is not set"
  exit 1
else
  echo "✅ Environment variable APPLE_ISSUER_ID is set"
fi

if [[ -z "${APPLE_PARQUET_KEY_ID:-}" ]]; then
  echo "🚨 Error: Environment variable APPLE_PARQUET_KEY_ID is not set"
  exit 1
else
  echo "✅ Environment variable APPLE_PARQUET_KEY_ID is set"
fi
echo "🔑 All required environment variables are set"

if [[ ! -d "$APP_PATH" ]]; then
  echo "🚨 Error: .app bundle not found at expected path: $APP_PATH"
  exit 1
else
  echo "✅ .app bundle found at: $APP_PATH"
fi
if [[ -z "$DMG_PATH" ]]; then
  echo "🚨 Error: DMG file not found!"
  exit 1
else
  echo "✅ DMG file found at: $DMG_PATH"
fi

echo "🔐 Re-signing .app with entitlements using 3rd Party Application cert..."
codesign --entitlements "$ENTITLEMENTS_PATH" --deep --force --options runtime \
  --sign <REDACTED-BUT-PUT-YOUR-KEYCHAIN-NAME-HERE> "$APP_PATH"

echo "🧳 Rebuilding and signing .pkg with 3rd Party Installer cert..."
productbuild \
  --component "$APP_PATH" /Applications \
  --sign <REDACTED-BUT-PUT-YOUR-KEYCHAIN-NAME-HERE> \
  "$PKG_NAME"

echo "🚀 Submitting DMG to notarization..."
xcrun notarytool submit "$DMG_PATH" \
  --key <THIS IS THE PATH TO YOUR p8 KEY you downloaded from APPLE> \
  --key-id "$APPLE_PARQUET_KEY_ID" \
  --issuer "$APPLE_ISSUER_ID" \
  --keychain-profile "notarytool-password" \
  --wait

# this is check the arch, staple, validate staple steps
lipo -info "$APP_PATH/Contents/MacOS/walk-in-the-parquet"
xcrun stapler staple "$DMG_PATH"
xcrun stapler validate "$DMG_PATH"
hdiutil imageinfo "$DMG_PATH" | grep Format

echo "📦 .pkg ready to be uploaded via Transporter:"
echo "   -> $PKG_NAME"
echo ""
echo "🚀 Open Transporter and upload the package manually if needed."
```

The key parts are that you'll want your `--sign` argument to be `3rd Party Mac Developer Application <etc>`. That is your 3rd party developer application that you can use for signing

# Conclusion

Anyway, I have sunk more time than allocated into this, but it was a fun project, and I'm looking forward to working on this in the future. If you have issues or feedback requests, feel free to blow up that Canny board.

Enjoy the application and I hope I've helped some random stranger out there.

## Kudos

Oh also thank you to my girlfriend for coming up with the name. Better than what I could have thought of.

[comment]: <> (Bibliography)
[parquet]: https://parquet.apache.org/
[tauri]: https://tauri.app/
[rust]: https://www.rust-lang.org/
[mojo]: https://mojo.com/
[snappy]: https://en.wikipedia.org/wiki/Snappy_(compression)
[walk-in-the-parquet-site]: https://www.walkintheparquet.com/
[walk-in-the-parquet-app]: https://apps.apple.com/us/app/walk-in-the-parquet/id6743959514?mt=12
[databricks-parquet]: https://www.databricks.com/glossary/what-is-parquet
[parquet-row-groups]: https://duckdb.org/docs/stable/data/parquet/tips.html#selecting-a-row_group_size
[celerdata-parquet]: https://celerdata.com/glossary/parquet-file-format
[celerdata]: https://celerdata.com/
[electron]: https://www.electronjs.org/
[dropbox]: https://www.dropbox.com/
[electron-preload]: https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts
[electron-ipc]: https://www.electronjs.org/docs/latest/tutorial/ipc
[duckdb]: https://duckdb.org/
[golang]: https://go.dev/
[cpp]: https://cplusplus.com/
[python]: https://www.python.org/
[coditation-ref]: https://www.coditation.com/blog/electron-vs-tauri#:~:text=and%20CPU%20resources.-,Tauri%20is%20designed%20to%20be%20more%20lightweight%20and%20faster%20than,run%20more%20efficiently%20than%20Electron.
[coditation]: https://www.coditation.com/
[vibe-coding-meme1]: https://www.reddit.com/r/ProgrammerHumor/comments/1jcjrzf/vibecoding/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
[vibe-coding-meme2]:https://preview.redd.it/viberagingnow-v0-1vjd0a87owpe1.jpeg?auto=webp&s=00830b4959b1426e6280068dd59b528257aa8c3b
[vibe-coding-meme3]:https://preview.redd.it/vibe-coding-v0-hwsv07yperre1.jpeg?auto=webp&s=004dcaea56ead53a5c453efa24d93d174865fa57
[apple-notarization]: https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution
[apple-codesigning]: https://support.apple.com/guide/security/app-code-signing-process-sec7c917bf14/web
[tauri-thick]: https://v1.tauri.app/v1/guides/building/macos#binary-targets
