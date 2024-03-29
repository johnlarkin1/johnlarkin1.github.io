---
title: A Better Way to Reference PRs
layout: post
featured-img: gh-pr-formatter-icon
categories: [Development]
summary: My first published Chrome Extension! All because I'm not a fan of Slack auto unfurling my whole PR description.
---

# 🔌 Shameless (somewhat shameful) Plug! 🔌

This fun little developer tool got approved and is on the Chrome Web Store now!

[Add it to Chrome here!][my-chrome-extension] (or go here: <https://chromewebstore.google.com/detail/github-pr-title-formatter/pdjmmincfaficadbaemdifkbaioiihob>)

Here's a preview:

[![chrome_store_preview](/images/github-pr-formatter/chrome_store_preview.png){: .center-image}][my-chrome-extension]

# Motivation

More or less, I'm sick of [Slack][slack] [auto-unfurling][unfurl] my Github PRs into a single link with basically my entire PR description pasted in the output.

Here's what I mean:

![noisy_slack_unfurl](/images/github-pr-formatter/noisy_slack_unfurl.png){: .center-image}

It just feels noisy, and when you're on a team of developers (and they have two workstreams and multiple people posting in a review channel or referencing PRs), this Github unfurling just takes up too much space.

![slack_doing_too_much](https://media1.tenor.com/m/3du_m9alZJoAAAAC/too-much.gif){: .center-shrink}

The thing is, if you're titling your PRs well, they should convey enough information that other developers on your team will be able to see that in Slack, understand from the title what the PR is roughly aiming to attack, click on it, and then have the details in the normal Github UI like we're used to.

# Solution

So perhaps a fun Chrome Extension can help!

What if we just let you pick how you wanted your resulting link to look in Slack? So here's how I like it:

![cleaner_slack_unfurl](/images/github-pr-formatter/cleaner_slack_formatting.png){: .center-shrink}

Or a little live demo:

![live_example](/images/github-pr-formatter/demo.gif){: .center-image}

Or a bigger demo here:

<p align="center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/3IMhGmX5tFc?si=cCrYYKrgaxEPbK4P" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen>
    </iframe>
</p>

And we have a nice dropdown for different configurations:

![options](/images/github-pr-formatter/options.png){: .center-super-shrink}

# Chrome Store

If you're thinking, oh this looks great, where can i download it? No problem! Chrome Extension store for the win.

It should be publically available on the [Chrome Extension Store][chrome-store], or specifically, [you can download it here][my-chrome-extension].

## Submission Process

This submission process actually wasn't too bad at all! More careful specification in the `manifest.json` for the `content_scripts` helps to speed up this process. For example, this Chrome extension will only be active here: `"matches": ["*://github.com/*/*/pull/*"],`. So on Github urls under the pull tab basically.

I ran into some trouble with my Draftkings web scraper but that was more mainly blocked for the extension store not supporting anything related to betting or gambling. You can read more about that here: **[Chrome Extension, Betting Analysis, and Kelly Criterion][dkng-post]**.

# Code

It's for sure not the prettiest thing I've written, but it is efficient and functional. The CSS and styling isn't the most mobile friendly, it works on Desktop and I'm fine with shipping it like that (given I don't think that many people will end up using it, but I like it a lot and so wanted to share).

[So here's the code][code].

Feel free to tweak and optimize as you wish (just acknowledge)! Happy hacking!

[comment]: <> (Bibliography)
[slack]: https://slack.com/
[unfurl]: https://medium.com/slack-developer-blog/everything-you-ever-wanted-to-know-about-unfurling-but-were-afraid-to-ask-or-how-to-make-your-e64b4bb9254
[code]: https://github.com/johnlarkin1/github-pr-formatter
[slack-example]: (/images/github-pr-formatter/demo.gif)
[chrome-store]: https://chromewebstore.google.com/
[my-chrome-extension]: https://chromewebstore.google.com/detail/github-pr-title-formatter/pdjmmincfaficadbaemdifkbaioiihob
[dkng-post]: {{ site.baseurl }}/2023/chrome-betting-kelly/
