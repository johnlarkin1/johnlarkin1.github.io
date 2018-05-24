---
title: Exploring your Options
layout: post
---

What is an option in the financial sense? We explore an example with an :apple: market example.

![options](/images/intro-to-options/option-trading.jpg){: .center-shrink }
<center> <i> Could I find a worse photo for this post? No. No I could not. </i> </center>

This post is not going to talk about how you can use options, but generally there are three types of ways to use options:
1. As protection for a current position
2. As a form of income 
3. As speculation on the underlying

Introduction
============
I work for a trading firm in Chicago. It's [Belvedere Trading][bt]. That is my life now. And I'm happy with it, _but_ one of the downsides of working a fulltime job is that it is increasingly hard to find time to code and do things with my own time. I really want to take a couple posts and share some of what I've learned (within reason @sec) at my job.

When I showed up at [Belvedere][bt] on August 21st 2017, I - for starters, barely knew how to write good code - had no idea what an option was or anything really about trading. It was an absolute gift to be hired by Belvedere and something that I am thankful for everyday. And the company has done an incredible job of teaching me about a world I had no exposure to. 

So let's dig into it. I'll organize this post in the following. 

- [History of Options](#hoo)
- [Definition of an Option](#def)
- [Extra Terminology](#term)
- [Example of an Option](#ex)
- [How You Could Buy An Option](#buy)
- [Conclusion and Up Next](#next)

History of Options {#hoo}
=========================
I figured I could write about this a bit and then it would kind of generate (hopefully) some interest about the following sections. I'm pretty much just going to summarize, but you can find a ton of the information [here][history].

The link I just tagged above talks about [future contracts][fut], but I'm not really going to focus on that at all today. Options first came about in a pretty similar manner to the one that we're going to explore as an example here today. Namely, the Greeks first came up with a similar financial instrument to an option, speculating on their olive harvests.

Going more into depth, the story goes that [Thales of Miletus][thales] was big into astronomy and mathematics. He foresaw in the stars that there was going to be a plentiful olive harvest and wanted to turn a profit from his bold astronomical prediction. He couldn't afford to own olive presses so he paid the owners of the olive presses a sum of money to use the presses in the future when it was harvest time. 

And sure enough, when harvest time came, it was a massive harvest. Thales then resold his rights to the people than needed to press their olives for a tidy profit. 

This is really just a call option as we will soon see!

You can check out more about the history of options [here][history2].

Definition of an Option {#def}
==============================
An option is the _right_ but **not** the obligation, to buy or sell the underlying asset at a set price (the strike price) by _or_ on a given date (expiration date).

Extra Terminology {#term}
=========================
The base definition doesn't really make much sense without an explanation of some of the other terminology. Let's give some context.

The "right but not the obligation" bit is pretty relevant. You have the *option* hence the name of the financial instrument. 

### Strike Price 
The strike price is simply the price at which you can buy or sell the underlying asset.

### Expiration Date
Let's clarify the expiration date bit. There is a huge difference between the words `by` and `on`. For **European** options, you can only exercise the option (meaning buy or sell the underlying asset at the strike price) **ON** the expiration date. However, for **American** options, you can exercise the option **UP UNTIL** the expiration date. This is a very important distinction. If you have an American exercise option, and you're deep in the money, you could exercise it *as long as it is before or on the expiration date*. For a European exercised option, you *have* to wait until the expiration date to exercise. So even if you're deep in the money, you still have to wait until the expiration date. They're two different types of bets and the math behind the option totally changes based on the expiration style. 

### Option Types
There are **two** types of options - **calls** and **puts**. 
**Calls** - this is when you have the right but not the obligation to **buy** the underlying asset at the strike price. 
**Puts** - this is when you have the right but not the obligation to **sell** the underlying asset at the strike price. 

### Premium 
This is how much you're paying or collecting for the option contract. So if you're selling options (either call or put), then you're going to be collecting a premium, whereas if you're buying options, then you're outlaying money.

### Summary
All of this information is packaged in the option contract. Specifically, the option contract is composed of the following information:
* name of the underlying security
* expiration month / expiration date 
* strike price 
* type of option (put / call)
* number of contracts
* premium

Example of an Option Transaction {#ex}
======================================
Let's start with a very basic fake example. 

### Example 1
Let's say it's March. You're a smooth talker and you manage to go to your local [Krogers] grocery store and strike up a deal concerning apples. Krogers thinks that the apple market is going to go spiral because they know that apple farmers had a great farming season and just crushed that apple game. But you're not so sure, you think that the price of an apple is about to become more expensive for some reason, and you trust your gut.

So you and Krogers strike a deal. Krogers says that they'll sell you coupons for $2 to buy apples on or before May for $4. Let's say the current price of an apple is $1.50. You think it's a great deal (because of your theoretical apple pricing engine... we'll get there), so you negotiate and you buy 10 of those coupons. So you outlay $20 bucks. Not cheap, but hey it's wild we're trading apples out here.

#### Questions
0. What's the underlying asset?
1. What type of contract does this resemble - call or put? 
2. How many contracts did you buy?
3. Are these European style options or American style options? 

Think about those before moving on. **Don't scroll pass this point until you've answered those questions! (At least in your head).**

***

Ok now for the answers. 

#### Answers
0. The underlying asset is what we're speculating on. It's apples :apple:
1. This example is essentially a call option right? You have the *right* but not the obligation to *buy* the underlying asset (which is the apples).
2. You bought 10 may $4 apple calls. The quantity here is 10 because you negoatiated for that many contracts.
3. These are *American* style options. This is because Kroger was dope enough to say you can buy these apples at this price *on or before* May. 

Ok cool. So let's say time goes by. It's April now. The apples get delivered to all the groceries and everyone is expecting prices to drop because the supply is through the roof and demand is relatively constant. *However*, plot twist, all of these apples have [fire blight][fire-blight]. Ohhh boy you hate to see fire blight. Everyone has to pitch all those apples and while demand is relatively constant, there just aren't that many good apples that people can buy. So the price actually increases. And let's say it's wild and apples are trading for $10. Stop... question time. 

#### Questions
0. Let's say you want to exercise. How could you generate a profit? 
1. What would that profit be?
2. What would be some reasons that you *wouldn't* want to exercise your calls?

Think about those before moving on. **Don't scroll pass this point! Again**

***

Ok now for the answers.

#### Answers
0. You could generate a profit by exercising your call option, right? Because you have your call option, or your coupon, to buy from Krogers for $4 dollars. You have ten of them and let's say you want to exercise all of your contracts. So you buy ten apples from Krogers for $40. But then apples are trading for $10, and you own ten apples, so you can instantly sell those 10 apples and make $100 dollars. 
1. So what's your net profit? Well it's not $100 - $40 = $60. Why not? Well because you originally paid $2 per contract and you bought ten of those contracts. So you actually also spent $20 there. So your net profit is actually only $40 (which is still a pretty good cut for apple trading).
2. When wouldn't you want to exercise? Well... let's say that the rise in the apple market has sparked off other apple option traders. Let's say that you think you might actually be able to trade the option contract in the secondary market yet again. Let's also say that someone is willing to pay $12 a contract for the apple contracts which you own (because they think apples are going to rise more). Well then, you could sell your contracts in the secondary market. You sell 10 contracts at $12 per, making a profit of $120 minus the cost you originally paid and now you're working with $100 of net profit solely from a liquid apple market. 

This example actually spiraled out into a slightly better example / longer example. I was originally going to use options on the [S&P 500][sp], however, I figured that might be a tricky example. Thinking about it however, the only real difference is that options on the S&P are cash-settled meaning that you don't actually acquire any shares of stock, but you simply are credited in your option account with the difference between the exercise price and the price that the S is trading at the end of the day. It generally works the same however. You can check out more information [here][index-cash].

How You Could Buy An Option {#buy}
==================================
I figure while a lot o fyou guys might appreciate theory as much as me, part of you are curious where you can actually participate in buying options and this type of trading. [Robinhood][rh] is an example I would suggest because of the modern UI and also the fact that it's free to use and there aren't any transaction fees. 

_**Warning: Trading options involves a lot more risk than buying normal stocks. There is literally unlimited risk when you sell a call or sell a put. There are multipliers in effect. I would highly encourage you to do more research if you're actively going to participate in trading options.**_

After reading that warning (read it twice), you can go [here to checkout options on robinhood][rbo]. Another interesting note about [Robinhood][rh], as of now (2018-05-23), they haven't set up the trading platform for options on their desktop portal. As a result, you'll need to first register your [Robinhood][rh] account for trading options, [heed their caution][rho], and then you can start. Robinhood has a gorgeous UI and it's very simply to start trading options. Check out the series of steps from my account. 

<div class="basic-center">
    <div class="center-super-super-shrink-sbs">
        <img src="/images/intro-to-options/rh_1.png">
    </div>
    <div class="center-super-super-shrink-sbs">
        <img src="/images/intro-to-options/rh_2.png">
    </div>
    <div class="center-super-super-shrink-sbs">
        <img src="/images/intro-to-options/rh_3.png">
    </div>
</div>

Conclusion and Up Next {#next}
==============================
And that's going to wrap up this blog post. As always, feel free to message me or email me if you have questions or comments about the blog. Belvedere has been an incredible experience and I'm happy with what I'm learning and decently satisfied at the rate at which I'm absorbing information. 

**Thanks for your time as always!**

[comment]: <> (Bibliography)
[bt]: http://www.belvederetrading.com/
[krogers]: https://www.kroger.com/
[sp]: https://www.marketwatch.com/investing/index/spx
[shar]: https://en.wikipedia.org/wiki/Sharanya_Haridas
[index-cash]: https://www.investopedia.com/study-guide/series-4/index-interest-rate-and-currency-options/index-option-settlement/
[fire-blight]: https://www.starkbros.com/growing-guide/article/got-fire-blight
[history]: https://www.investopedia.com/articles/optioninvestor/10/history-options-futures.asp
[history2]: http://www.optionstrading.org/history/
[fut]: https://www.investopedia.com/terms/f/futurescontract.asp
[thales]: https://en.wikipedia.org/wiki/Thales_of_Milet
[rh]: https://robinhood.com/
[rbo]: https://www.robinhood.com/options/
[rho]: https://support.robinhood.com/hc/en-us/articles/115005706846-Trading-Options
