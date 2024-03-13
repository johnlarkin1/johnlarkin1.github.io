---
title: A Better Way to Reference PRs
layout: post
featured-img: gh-pr-formatter-icon
categories: [Development]
summary: I'm not a fan of Slack auto unfurling my whole PR description. 
---

# Motivation

More or less, I'm sick of [Slack][slack] [auto-unfurling][unfurl] my Github PRs into a single link with basically my entire PR description pasted in the output. 

Here's what I mean: 

![noisy_slack_unfurl](/images/github-pr-formatter/noisy_slack_unfurl.png){: .center-image}

It just feels noisy, and when you're on a team of developers (and they have two workstreams and multiple people posting in a review channel or referencing PRs), this Github unfurling just takes up too much space. 

![slack_doing_too_much](https://media1.tenor.com/m/3du_m9alZJoAAAAC/too-much.gif){: .center-shrink}

The thing is, if you're titling your PRs well, they should convey enough information that other developers on your team will be able to see that in Slack, understand from the title what the PR is roughly aiming to attack, click on it, and then have the details in the normal Github UI like we're used to. 

# Solution

So perhaps a fun little Chrome Extension can help here:

![overview](/images/github-pr-formatter/overview.png){: .center-image}

What if we just let you pick how you wanted your resulting link to look in Slack? So here's how I like it: 

![cleaner_slack_unfurl](/images/github-pr-formatter/cleaner_slack_formatting.png){: .center-image}

Or a little live demo: 

![live_example](/images/github-pr-formatter/demo.gif){: .center-image}

There's this nice little dropdown for different configurations:

![options](/images/github-pr-formatter/options.png){: .center-image}

# Chrome Store

If you're thinking, oh this looks great, where can i download it? Let me spin ya a story. 

## Chrome Store Submission

This process is so painful, and rightfully so. There is a bit of a rigorous process, which you can also read about hereL:

It should be publically available on the [Chrome Extension Store][chrome-store], or specifically, from this link: 


## Chrome

# Code 

It's for sure not the prettiest thing I've written, but it is efficient and functional. The CSS and styling isn't the most mobile friendly, it works on Desktop and I'm fine with shipping it like that (given I don't think that many people will end up using it, but I like it a lot and so wanted to share). 

[So here's the code][code]. 

Feel free to tweak and optimize as you wish (just acknowledge).


[comment]: <> (Bibliography)
[slack]: https://slack.com/
[unfurl]: https://medium.com/slack-developer-blog/everything-you-ever-wanted-to-know-about-unfurling-but-were-afraid-to-ask-or-how-to-make-your-e64b4bb9254
[code]: https://github.com/johnlarkin1/github-pr-formatter
[slack-example]: (/images/github-pr-formatter/demo.gif)
[chrome-store]: https://chromewebstore.google.com/