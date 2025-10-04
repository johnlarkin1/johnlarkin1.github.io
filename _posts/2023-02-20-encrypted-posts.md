---
title: Encrypted Blog Posts
layout: post
featured-img: lock
categories: [Dev]
test-encrypted: true
encrypted: true
encrypted-hash: 3aa4cb08d481cfe2b08e4a5e31777f642263d58d
summary: I wanted to see if I could encrypt a blog post (in some way)
---

# Introduction

There's a couple of good projects out there about the best way to encrypt static web pages with passwords.

Here are a couple good sites I found:

- [Staticrypt][staticrypt]
- [Chrissy Dev's Github Protected Pages][gh-pages-protection]

# Downsides

I like both of these, but here's the thing. For specifically, Github Pages there's not really a smooth integration. Specifically, I liked Chrissy Dev's but the downside was that you could always share either the Markdown file or the link with the right `sha1` hash, and everything would immediately show. You could even send that link around and no one would even really realize it was password protected.

# My Approach

So my approach is somewhere like a blend between those two. I basically provide a secret phrase, store that hash somewhere (not going to tell where), and then upon the user entering the right secret phrase, I'll reveal the actual blog's content.

Ok and before I get absolutely torn apart. I know there are MULTIPLE ways to work around these. While it is a true `sha1` encryption, it's not even close to being a fully protected page... especially since Github Pages requires a public repo. I did think about either creating a cipher and mixing the actual page content around, but I wanted to preserve the HTML contents, and honestly, the Liquid format of `\{\{ content \}\}` being set as a Javascript variable doesn't really play nicely with a homegrown Cipher.

## Technical

This adjustment basically just includes a small modification to the post.html. I'd check out that code in the corresponding Github repo. I'm not going to link because I don't really need to make it easier than what I've given you :)

# Conclusion

This was a nice little refresher on some DOM elements and interaction. Kinda a good exploration of differnet Javascript decryption/encryption methods like `SHA1`[link][sha1] vs `AES`[link][aes] etc. Eventually I chose a basic approach, but kinda a fun quick exercise (and a good way to check off my February post).

[comment]: <> (Bibliography)
[staticrypt]: https://github.com/robinmoisson/staticrypt
[gh-pages-protection]: https://github.com/chrissy-dev/protected-github-pages
[sha1]: https://en.wikipedia.org/wiki/SHA-1#:~:text=In%20cryptography%2C%20SHA%2D1%20(,rendered%20as%2040%20hexadecimal%20digits.
[aes]: https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
