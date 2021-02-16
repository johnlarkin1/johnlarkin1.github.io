---
title: Stanford's CS193 - Developing Apps for iOS
layout: post
---

Exploring SwiftUI and iOS app development.

Introduction
============
I've been fiending to learn a little bit more about iOS development and all the hot rage behind SwiftUI Apple's very own development language. 

The class I elected to take (although there are many great resources) is Stanford's CS193p - Developing Apps for iOS. The class is - incredibly enough - free of charge and can be found over [here][stanford-class].

Overview
========
You can explore the class page for yourself but basically I'll outline the 14 lessons that I went through. 

1. Course Logistics and Introduction to SwiftUI
2. MVVM and the Swift Type System
3. Reactive UI + Protocols + Layout
4. Grid + enum + Optionals
5. ViewBuilder + Shape + ViewModifier
6. Animation
7. Multithreading EmojiArt
8. Gestures JSON
9. Data Flow
10. Navigation + TextField
11. Picker
12. Core Data
13. Persistence 
14. UIKit Integration

In addition, you can find all 27 pages of my class notes in a nice [PDF][notes].

Projects
========
There are a couple main projects you build. The class - the instructor is fantastic by the way - likes to reassert the motto, "a picture is worth a thousand words but a demo is worth tens of thousands of words." While not exactly the same thing, I'll let the short synopsis here cover some of the work. There is really four main projects you explore. 

1. [Memorize Game][memorize-wiki]
2. [Set Game][set-wiki]
3. Emoji Art (game where you can combine images and emojis to create art)
4. Enroute (Basically, an application to parse the FlightAwareAPI and show flight information)

Note, I didn't show gif demo for the Enroute application (number 4 above) because it was largely all demo code that I didn't write. I included the code itself in the repo because I made comments in the code. 


### Memorize Game

![memorizegame](/videos/developing-ios-apps/MemorizeGameRecording.gif){: .center-image }

### Set Game

![setgame](/videos/developing-ios-apps/SetGameRecording.gif){: .center-image }

### Emoji Art

#### iPad Version

![emojiartipad](/videos/developing-ios-apps/EmojiArtiPadRecording.gif){: .center-image }

#### iPhone Version

![emojiartiphone](/videos/developing-ios-apps/EmojiArtiPhoneRecording.gif){: .center-image }

Comments
========
The class was definitely a fair bit of work. It felt good to get absolutely wrecked by a couple niche bugs (I would recommend watching Lecture 11 before completing Assignment 6 as there is a lot of revelant material in there). The class took me longer than I expected and also provided a lot more late night debugging sessions. 

Also full disclosure, while the online free class is an excellent resource, I can only imagine that having access to other Stanford students as well as the Class Piazza accelerates the amount of information absorbed. 

I would highly recommend this class to anyone interested in learning more about iOS development and working with SwiftUI. It's very clear, relevant, and hopefully will be useful for my own pet projects. 

Other Helpful Links
===================
* [Swift UI Cheat Sheet][swift-cheatsheet]
* [Impossible Grids][impossible-grids]
* [Grid Trainer][grid-trainer]

Again, I'll same the same disclaimer as when I took Coursera's Computational Neuroscience class. While I've included all of the code that I used to answer the homework assignments, please do not abuse this. You're only doing yourself a disservice in terms of what you learn. I included some of the stanford code lectures (also found on the class website) because I commented the code and took notes within the code itself. 

As always feel free to email me with questions - although, I'm far from an expert! I am more than happy to help debug. 

**Once again, you can get the notes [here][notes].** 

In addition, there was one aspect I could not get... I posted on Stack Overflow about it after I kept getting it wrong and couldn't code up a solution. If you think you can help, I'd love for you to answer my question on Stack! Check it out [here][question].

[comment]: <> (Bibliography)
[stanford-class]: https://cs193p.sites.stanford.edu/
[swift-cheatsheet]: https://jaredsinclair.com/2020/05/07/swiftui-cheat-sheet.html#:~:text=Use%20%40State%20when%20your%20view,ancestor%20has%20a%20reference%20to.&text=If%20your%20view%20needs%20more,you%20are%20out%20of%20luck.
[impossible-grids]: https://swiftui-lab.com/impossible-grids/
[grid-trainer]: https://github.com/swiftui-lab/GridTrainer
[memorize-wiki]: https://en.wikipedia.org/wiki/Concentration_(card_game)
[set-wiki]: https://en.wikipedia.org/wiki/Set_(card_game)
[code]: https://github.com/johnlarkin1/developing-ios-swiftui-stanford
[notes]: {{ site.baseurl }}/pdfs/DevelopingiOSApps2020.pdf