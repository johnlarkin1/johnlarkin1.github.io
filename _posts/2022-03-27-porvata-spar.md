---
title: Porvata and SPAR Integration
layout: post
featured-img: porvata
categories: [⭐️ Favorites, Dev, Friends]
summary: I got to work with my best friend Will on a fun technical project!
favorite: true
---

I got to work with my best friend [Will][will] on an interesting development project. This was by far one of my favorite pieces of development work I've done.

# Background

One of my best friends, [Will Jaroszewicz][will] (see below (he's the good looking one on the left)), and his brother, [Nick Jaroszewicz][nick], started a ecommerce retailer for home and office furniture in the middle of the pandemic named [Porvata][porvata].

![will-and-i](/images/porvata-spar/will-and-i.jpg){: .center-shrink }

Also this is from longer ago, but here's a photo of Nick J and I (also Will and another Nick). Nick is on the right side, next to me.

![nick-will-nick-and-i](/images/porvata-spar/nick-will-nick-and-i.jpg){: .center-shrink }

[Porvata][porvata] helps to give consumers a better work from home setup, providing a range of great products like [72" desks][porvata-desk] to [ergonomic chairs][porvata-chair] to even very helpful [power strips][porvata-power-strip] compatable with their desks. I've got some of their products and endorse it heavily. In the very least, I'd recommend anyone reading check their [site][porvata] for a couple minutes (at least for the site traffic).

As an aside, and this is more biased obviously, but I've seen how hard both Nick and Will have worked to grow [Porvata][porvata]. They care about the quality of their products, and they care about customer happiness and satisfaction. It's been extremely rewarding and impressive to watch them grow as people and as a company throughout. I've always loved them as friends but I hadn't yet had a good glimpse into their professional tendencies. It's just another facet into what makes them unique. It's been a lot of fun to work with them for this side project, and it's been inspiring to me to keep growing as well.

So let's dive in.

<!--
# Table of Contents

- [Background](#background)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Overview of Architecture](#overview-of-architecture)
- [Sweating the Details](#sweating-the-details)
  - [Shopify Partners](#shopify-partners)
  - [Shopify Authentication](#shopify-authentication)
    - [`ngrok` and SSL Authentication](#ngrok-and-ssl-authentication)
  - [Shopify Order Events](#shopify-order-events)
    - [GraphQL](#graphql)
  - [Posting to SPAR](#posting-to-spar)
    - [Interesting Questions](#interesting-questions)
      - [Dynamic Pricing and SKU Checks](#dynamic-pricing-and-sku-checks)
      - [Zip Codes](#zip-codes)
      - [Multiple SPAR Legs](#multiple-spar-legs)
      - [Cancellations](#cancellations)
  - [Extras and Nice-to-Haves](#extras-and-nice-to-haves)
    - [Email Notifications](#email-notifications)
      - [Order Received Event](#order-received-event)
      - [Unknown Error Event](#unknown-error-event)
      - [Outside Zip Code Event](#outside-zip-code-event)
      - [Partial Order Fulfillment Event](#partial-order-fulfillment-event)
      - [Full Order Fulfillment Event](#full-order-fulfillment-event)
    - [Additional Service Checking](#additional-service-checking)
    - [Logging](#logging)
    - [Linting and GitHooks](#linting-and-githooks)
- [Conclusion](#conclusion)
-->

# Introduction

Late into 2021, maybe around November, Will broached me saying that he might soon need a bit of development help on the backend side to tie together assembly integrations into orders. The gist is that [Porvata][porvata] would talk to [SPAR][spar], a third party company that would handle the assembly portion of orders. Before this, Will and Nick would often arrange for a TaskRabbit or some Fiver task if a customer personally emailed in and requested about assembly for one of their desks or chairs. After this, the goal was that customers could simply select assembly when going through the checkout process on [Porvata][porvata] and when the order shipped, they'd get a call and email from SPAR asking when a good time for them to handle assembly was. Ok cool.

So in theory, as most development projects go, this sounded pretty standard and relatively simple. But the fun parts came in with the details. So let's get to that.

# Overview of Architecture

So my initial thoughts were basically the following. [Porvata][porvata] is a Shopify application (and we'll get into that, but I'm a huge fan of Shopify (and now a shareholder (precisely 3 months to early to buy apparently))) so I figured that the developer services / events we could listen to would be pretty accessible. I figured that we could have some persistent service sitting on an EC2 instance listening to some Google Pub/Sub events and when we heard those events we could publish to the RESTful API that SPAR provides for the different types of events.

That was, and still is, the main design of the architecture but there's a lot more that we wanted to add.

But again... this post is more wordy than even I have the attention span for, so I tried out [Lucidchart][lucidchart] (of which I'm now a big fan of), and came up with this basic design flow:

![porvata-arch-overview](/images/porvata-spar/porvata-arch-overview.png)

# Sweating the Details

In [Dropbox][dbx] fashion, there turned out to be a lot of corner cases. I'm going to walk through in more detail the approach and how I iterated on this project.

## [Shopify Partners][shopify-partner]

[Will][will] and I talked about this to some degree about whether this should just be a standalone app that sits on some server or if I should jump through the hoops of being set up as a Shopify Partner. At first, I thought I could get by with simple webhooks and tokenization for either listening to events or some type of polling mechanism, but ultimately (and I think Shopify guided me this way), it seemed way more efficient to register as a Shopify Partner. And man, am I glad I did.

Here are some of the benefits that this provides:

- It's _really_ relatively painless
  - There wasn't a big delay or like secondary approval, it's just signing up an account
- **Dev testing**
  - This was the big one. I - unfortunately - do not write perfect code. While unit tests help, it was nice to walk through with orders and simulate different customer actions.
- Hello, it's fun?!
  - I got to set up my own little Pet Store and simulate orders going through so that I could ensure we were parsing the data on the backend correctly as well as subscribing to the appropriate order events.

Alright :white_check_mark: so that portion has been good. I signed up as a Shopify Partner.

## Shopify Authentication

This was the second big hurdle I had to jump through. And also fun enough, where I got to have some front end application.

Again, if you're trying to set up your own custom app (awesome), but also I'd largely follow this content here: [Shopify OAuth Tutorial][shopify-auth]

I'd recommend watching this [Dropbox Capture][dbx-capture] video (great product, again biased, but would recommend checking it out esp if you're a Dropbox user. Think Loom but more natural integration):

Here's two videos showing the brief UI interaction (with a screenshot)

<p align="center">
    <iframe src="https://www.dropbox.com/s/e9j24elsrkn16i1/BasicFunctionalityDemoPt1.mp4?raw=1" 
        width="560" 
        height="315"
        frameborder="0" 
        allowfullscreen>
    </iframe>
</p>

<p align="center">
    <iframe src="https://www.dropbox.com/s/miwsx5yrtz9fpjv/BasicFunctionalityDemoPt2.mp4?raw=1" 
        width="560" 
        height="315"
        frameborder="0" 
        allowfullscreen>
    </iframe>
</p>

And a corresponding screenshot:

![example-success](/images/porvata-spar/example-success.png)

Note, I'm going to eventually try to share more of the code, but for now, I'm a bit protective over it because it's largely shaped around the [Porvata][porvata] needs. There are some portions (like some utils and Shopify authentication) that I can share, so I'll post that just as raw content here.

```python
def compute_hmac_from_args(api_password: str, args_dict: Dict[str, str]) -> str:
    """Computing hmac from request args. Logic is to use all of the data minus the hmac key."""
    sorted(args_dict)
    data_to_be_hashed = "&".join(
        [f"{key}={value}" for key, value in args_dict.items() if key != "hmac"]
    ).encode("utf-8")
    EventLogger.info(data_to_be_hashed)
    comparison_hmac = hmac.new(
        api_password.encode("utf-8"), data_to_be_hashed, hashlib.sha256
    )
    current_hmac = comparison_hmac.hexdigest()
    return current_hmac


def is_nonce_the_same() -> bool:
    """Confirm that the session nonce is the same as the nonce from the current request."""
    curr_nonce = request.args.get("state")
    session_nonce = session["nonce"]
    if curr_nonce != session["nonce"]:
        EventLogger.error(
            f"Nonce's do not match. From request: {curr_nonce} - from session: {session_nonce}"
        )
        return False
    return True


def is_hmac_valid() -> bool:
    """Confirm that the hmac is valid using our payload and `compute_hmac_from_args` call."""
    basis_hmac = request.args.get("hmac")
    if not SHOPIFY_API_PASSWORD:
        EventLogger.error("Did not have a SHOPIFY_API_PASSWORD")
        return False

    current_hmac = compute_hmac_from_args(SHOPIFY_API_PASSWORD, request.args)
    EventLogger.info(f"New hmac: {current_hmac} ; Old hmac: {basis_hmac}")
    if current_hmac != basis_hmac:
        EventLogger.error("hmac's do not match up. Aborting.")
        return False
    return True
```

### `ngrok` and SSL Authentication

The one annoying for me (but good from Shopify) is that you are required to provide a HTTPS endpoint for authentication so I'm guessing your token can't get sniffed or anything. That means that I had to run my flask server securely. I ended up turning to [`ngrok`][ngrok] which exposes the server behind a secure tunnel.

This was awesomel, but the downside is the free version (again, I'm cheap) of [Ngrok][ngrok] is only available for 8 hours and so I built a little helper to automatically start the `ngrok` process if one wasn't found on the VM and publish that endpoint. It does mean that the URL the Admin uses for authentication is basically only stable for 8 hours, but post authentication, they shouldn't neeed to interact with the UI anymore.

---

So great! Now we're actually authenticating. We're in the door. Now, we need to subscribe to the appropriate order events!

## Shopify Order Events

Now, this is really where I became a buyer (both literally and metaphorically) of Shopify.

The sheer volume of things you can listen, query, and ask for from a developer's perspective with Shopify is close to overwhelming. What we specifically were looking for is the [Shopify webhooks][shopify-webhooks]. Webhooks are a perfect way in order to listen to various events that occur for your shop... like an order getting placed or fulfilled.

This is where a bit of the business sense had to come into play. At first, we were emailing SPAR when an order had gotten created, but that doesn't really make sense does it? We didn't want to call the customer as soon as they actually placed the order, because it could have been a couple days before the order was fulfilled and on its way to the customer. As a result, we needed to change from the `ORDERS_CREATE` event to the `ORDERS_FULFILLED` event.

You can read more about the various webhook subscription events to tie into [here][shopify-subscription-webhooks]. There's also an example screenshot below:

![example-webhooks](/images/porvata-spar/example-webhooks.png)

For the sake of what I was doing, there were a couple that I needed to subscribe to (I can get into that later). But let's take a look at our nice pretty logs confirming we've subscribed to the various events.

![example-subscription-logs](/images/porvata-spar/example-subscription-logs.png)

So ok another great step, we're listening to the appropriate Pub/Sub events.

### GraphQL

This also touched into a bit of GraphQL knowledge. Tying into any of these webhooks was not a RESTful call, but actually a [GraphQL][graphql] post. You can read more at the [GraphQL Admin API Reference][shopify-graphql] from Shopify. The base endpoint that I used to register normally looked like this:

```python
    webhook_subscription_query = """mutation pubSubWebhookSubscriptionCreate($topic: WebhookSubscriptionTopic!,
        $webhookSubscription: PubSubWebhookSubscriptionInput!) {
    pubSubWebhookSubscriptionCreate(topic: $topic, webhookSubscription: $webhookSubscription) {
        webhookSubscription {
            id
            topic
            format
            endpoint {
                __typename
                ... on WebhookPubSubEndpoint {
                    pubSubProject
                    pubSubTopic
                }
            }
        }
    }
}"""

```

## Posting to SPAR

Ok so now we've basically got all the data that we need. The simple part was just extracting the appropriate information and sending that over to SPAR. Obviously, we only care about orders that have assembly in order to send to SPAR. So that meant that we needed some identifier. Given the UI and client facing design from [Porvata][porvata], we found that the easiest solution was just to give assembly orders their own [SKU][sku], which I then would check for. If it fell in this `set` of valid SKUs, we would then check that it was in a valid zip code (covered below), and if so, bundle up the legs and send it over to spar. We'll dive into some of these in more detail below.

### Interesting Questions

There were some interesting questions posed throughout this project.

#### Dynamic Pricing and SKU Checks

> Can you guys send us the appopriate just price associated with just the assembly SKU?

So one of the things we wanted was for high uptime of this service. Given that this is going to a customer and we're fine with adding a little bit of latency making an additional network call, I wanted [Will][will] to be able to add a SKU to the site, and have that be pulled in dynamically. We also wanted to extract specifically assembly pricing data, which because of how Shopify handled the presentation on the frontend, it was basically bundled into the total price. As a result, we decided to look at a private [Google Sheets][gsheets] integration idea (something which I had also done in this [post][crypto-craze-post]. So we are pulling a valid set of SKUs and mapping them to the corresponding broken out just assembly price that SPAR is interested in.

#### Zip Codes

> So, can we get assembly in every part of the US?

[SPAR][spar] was a blast to work with. Their developers with prompt and their business contacts informative. I would recommend working with them. They cover a lot of territory as well, but unfortunately not every zip code in the US (they cover close to 18,000 and there are ([according to USPS][usps-zip-codes]) 41,683 US zip codes at the time of writing. Interesting note, apparently the USPS metadata for Google search might be broken though because there's an inconsistency in [this search][google-zip-code-search]).

So this was a corner case that we obviously needed to support. [Porvata][porvata] services the entire US so a customer should be able to buy a desk anywhere, and the frontend (at this time) isn't configured to filter results based on zip codes.

We actually ran into this a couple weeks back. A customer ordered assembly from an outside zip code. What actually happened was (I believe due to some casting between an `int` and a `str`), my service threw an error. I investigated, saw the bug, fixed the bug, [replayed the event][google-replay-event] (again, love love love GCP Pub/Sub), and we got the appropriate email notification that the customer was actually outside of the valid zip code set. [Will][will] and [Nick][nick] then took the appropriate actions, contacted the customer quickly, and set up a TaskRabbit for them to have assembly in another manner.

#### Multiple SPAR Legs

> What should we do when a customer orders assembly on multiple items, and they're fulfilled at different times?

This was another interesting one we saw a couple weeks back. A customer had multiple assembly items and so we piecewise uploaded each fulfillment to SPAR. This was... well... working like how I expected but maybe not ideally (and I probably should have thought about this). What we wanted to do instead was only send SPAR one notification, basically saying, "Hey, we've got this customer here are all their assembly legs, the last one just got fulfilled, give them a call".

But on the flip side, let's say there's some crisis - like maybe a global pandemic or [this][suez-canal] - and there's a large delay between fulfillment orders. Let's say so large in fact that the customer **gets** one of their items that is _meant_ to have assembly with it, and SPAR hasn't called yet. You'd also be thrown off right?

So we basically just had to put in additional logic on the **initial** event that a customer with multiple assembly legs has their first order fulfilled to email [Will][will] and [Nick][nick] and just say, "We haven't posted to SPAR yet, but the first leg got fulfilled, let's reach out to the customer to inform them what's going on".

And **yes**, this **does** seem like an automated task (for all the constructive critics) **BUT!!** the obvious caveat:

> What would Porvata be without the added benefit of getting to interact with Will and Nick?!

#### Cancellations

> A customer decided to cancel their order after getting an Assembly SKU fulfilled. We've already posted to SPAR. What do we do?

This was _maybe_ a nice to have, maybe a [P0][google-p0] (as everyone says at [DBX][dbx-ticker] / in tech; also that P0 link has some good content).

SPAR wanted us to handle cancellations where SPAR assembly was an associated SKU and then post a reason to a different cancellation endpoint.

This wasn't too bad just basically required some extra validation in terms of checking the `reason` of the `OrderCancellation` event for some keywords specified by Will as well as matching on the assembly SKU.

Here's a demo of that:

<https://capture.dropbox.com/JCyXdxLMnzPGDywN>

<p align="center">
    <iframe src="https://capture.dropbox.com/JCyXdxLMnzPGDywN?raw=1" 
        width="560" 
        height="315"
        frameborder="0" 
        allowfullscreen>
    </iframe>
</p>

## Extras and Nice-to-Haves

These were some of the things that became apparent we need.

### Email Notifications

I would borderline say this is a `P0` given we need insight into when we actually upload information. As a result, on key events, we're emailing either myself, or [Porvata][porvata] depending on the event. Check out some examples below. Note, these are just emails, so I'm sending over enough to validate the order but nothing sensitive.

#### Order Received Event

![order-recieved-event](/images/porvata-spar/email-examples/order-received.png)

#### Unknown Error Event

![unknown-error](/images/porvata-spar/email-examples/unknown-error.png)

#### Outside Zip Code Event

![outside-zip-code](/images/porvata-spar/email-examples/outside-zip-code.png)

#### Partial Order Fulfillment Event

Yes I know I spelled `fulfillment` wrong :expressionless:

![partial-order-fulfillment](/images/porvata-spar/email-examples/partial-order-fulfillment.png)

#### Full Order Fulfillment Event

![success](/images/porvata-spar/email-examples/success.png)

### Additional Service Checking

I wanted confirmation that the Porvata/SPAR integration service was up daily. [Porvata][porvata] is crushing so I basically get `OrderEvent`s coming through at a good cadence to confirm that the service is still working, however... I still wanted this as a double check. The code is small enough that I figured I could just post it below:

```python
"""Send an email alert if the desired process cannot be found."""

from typing import List

import psutil

# imports for GMAIL

from utils.email_sender import EmailSender

class ProcessSearcher:
    def __init__(self) -> None:
        assert GMAIL_SENDER_ADDRESS is not None, "GMAIL_SENDER_ADDRESS is empty"
        assert GMAIL_SENDER_PASSWORD is not None, "GMAIL_SENDER_PASSWORD is empty"
        self._email_sender = EmailSender(GMAIL_SENDER_ADDRESS, GMAIL_SENDER_PASSWORD)

    def find_process_by_name_and_alert(self, target_name: str) -> None:
        is_found: bool = False
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=["pid", "name", "create_time", "cmdline"])
                # We only want to look at python processes
                if "python" in pinfo["name"].lower():
                    # We want to look for the specific application executing and make
                    # sure that it's still up
                    cmd: List[str] = pinfo["cmdline"]
                    executing_cmd = " ".join(cmd)
                    if target_name in executing_cmd:
                        print(
                            "Found target name in executing command. Sending information."
                        )
                        is_found = True

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if not is_found:
            print(
                f"Scanned through all running processes and did not find {target_name}"
            )
            print("Sending porvata.dev@gmail.com an alert.")
            self._email_sender.send_email_htmltext(
                "porvata.dev@gmail.com",
                f"🆘⚠️ Did not find {target_name} process running. ⚠️🆘",
                "Please check the EC2 instance and restart service if need be.",
            )
        else:
            self._email_sender.send_email_htmltext(
                "porvata.dev@gmail.com",
                f"✅🚀 Service {target_name} is up and running. 🚀✅",
                "All good!! Keep juicing.",
            )

```

I basically just set up a cronjob to run this bit of code. At first, it was every hour, then every four hours, now every 8 hours, soon to be once a day.

And here's the resulting email:

![service-check](/images/porvata-spar/email-examples/service-check.png)

### Logging

This is obviously important for debugging and understanding what has happened to a customer order. As you can see from the screenshot above, I have a decent logging system setup, predicated on the python `logging` module.

And once again, I know you guys are thinking... dang, well you might as well have set up a `mysql` or `sqllite` database just for easier searching with logs and events. And yeah, maybe that's true, but Shopify is so queryable and easy to integrate with that I haven't really found the need. Note, I excluded `postgres` from the suggestions of DBs because I am not too familiar with it, and also because I've heard it can be memory intensive. Given that there's already a solid amount of work being performed on my tiny EC2 instance, I don't think inundating it with more processes would be ideal. And if you're asking why not use [Amazon RDS][amazon-rds], because 1) I didn't want to pay more 2) I didn't want to go through that additional setup. Once [Porvata][porvata] becomes a unicorn, I'll scale this service in like ten different areas.

### Linting and GitHooks

I've already learned a good amount from Dropbox but I wanted to have it so whenever I pushed up a commit, it would automatically run `black` and `pylint` the code as well as run the `mypy` validation. Type hints are an absolute life saver at Dropbox and similarly, they were with this project as well. [Here's the documentation][commit-hooks] I'd use for commit hooks if you're interested in setting those up for your project.

# Conclusion

Again, this was probably one of my favorite things to work on. Learned a ton and got to work with a best friend so it doesn't really get much better than that.

This ended up touching a lot of new frameworks and packages that I hadn't really used before. Here's a summary below:

- Infrastructure
  - [EC2][amazon-ec2]
    - I had used this before, but definitely got a bit more familiar throughout this project
    - Elastic IPs
  - [Google Pub/Sub Notifications][google-pubsub]
    - Historical events saved for 10 days
    - Replaying events if issues
  - [Gmail email automation][gmail]
  - [ngrok]
  - [cronjob]s
  - [GitHub commit hooks][commit-hooks]
    - [black] formatting
    - [mypy] validation
    - [pylint] linting
- Software
  - [Flask][flask]
  - [Shopify API][shopify-api]
  - [GraphQL][graphql]
  - [Multithreading][py-multithreading]
    - Again, had some exposure through work, but this expanded
    - Better utilization of Python `logging` module
  - Process checking for service checks

And as for clear business results...

- Order placed with SPAR information
  - Email notification sent
- Order placed without SPAR information
  - Different email notification sent
- Order fulfilled with SPAR information
  - Both Will and Nick notified, as well as the appropriate information uploaded to SPAR
- Order fulfilled with SPAR information, but placed outside of the zip code
  - Email placed with corresponding zip code and Will and Nick alerted
- Order fulfilled with multiple SPAR legs
  - First time one of the legs is fulfilled, Will and Nick alerted (with action item to reach out to the customer to confirm SPAR will call once all legs fulfilled)
- Error state
  - Alert email sent out

Let me know if any questions / suggestions! Thanks for reading! I know this one was a long one.

[comment]: <> (Bibliography)
[matt]: https://mzucker.github.io/swarthmore/]
[porvata]: https://porvata.com/
[will]: https://www.linkedin.com/in/will-jaroszewicz-27313466/
[nick]: https://www.linkedin.com/in/nicholas-jaroszewicz/
[spar]: https://sparinc.com/services/assembly-installation/
[porvata-desk]: https://porvata.com/collections/all-desks/products/butcher-block-72-motorized-height-adjustable-standing-desk
[porvata-chair]: https://porvata.com/collections/office-chairs/products/ergonomic-chair
[porvata-power-strip]: https://porvata.com/collections/wire-management-power-outlets/products/clamp-mounted-power-management
[shopify-partner]: https://www.shopify.com/partners
[shopify-auth]: https://shopify.dev/apps/auth/oauth/getting-started
[shopify-webhooks]: https://shopify.dev/api/admin-rest/2022-01/resources/webhook#top
[shopify-subscription-webhooks]: https://shopify.dev/api/admin-graphql/2021-10/enums/WebhookSubscriptionTopic
[shopify-graphql]: https://shopify.dev/api/admin-graphql#top
[dbx]: https://www.dropbox.com/home
[dbx-ticker]: https://www.google.com/finance/quote/DBX:NASDAQ
[dbx-capture]: https://www.dropbox.com/capture
[dbx-demo1]: https://www.dropbox.com/s/e9j24elsrkn16i1/BasicFunctionalityDemoPt1.mp4?dl=0
[dbx-demo2]: https://www.dropbox.com/s/miwsx5yrtz9fpjv/BasicFunctionalityDemoPt2.mp4?dl=0
[sku]: https://www.investopedia.com/terms/s/stock-keeping-unit-sku.asp#:~:text=A%20stock-keeping%20unit
[google-pubsub]: https://cloud.google.com/pubsub
[google-replay-event]: https://cloud.google.com/pubsub/docs/replay-overview
[gmail]: https://developers.google.com/gmail/api/quickstart/python
[gsheets]: https://www.google.com/sheets/about/
[crypto-craze-post]: /2018/crypto-google-sheets-update/
[lucidchart]: https://www.lucidchart.com/pages/
[usps-zip-codes]: https://facts.usps.com/42000-zip-codes/
[google-zip-code-search]: https://www.google.com/search?q=how+many+zip+codes+are+in+the+US&oq=how+many+zip+codes+are+in+the+US&aqs=chrome..69i57.6923j1j7&sourceid=chrome&ie=UTF-8
[suez-canal]: https://www.bbc.com/news/business-56559073
[google-p0]: https://www.businessinsider.com/google-doubleclick-p0-failure-2014-11
[amazon-rds]: https://aws.amazon.com/rds/
[amazon-ec2]: https://aws.amazon.com/pm/ec2/
[commit-hooks]: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[ngrok]: https://ngrok.com/product
[cronjob]: https://crontab.guru/
[black]: https://black.readthedocs.io/en/stable/
[mypy]: http://mypy-lang.org/
[pylint]: https://pylint.pycqa.org/en/latest/
[flask]: https://flask.palletsprojects.com/en/2.0.x/
[shopify-api]: https://shopify.dev/api
[graphql]: https://graphql.org/
[py-multithreading]: https://www.tutorialspoint.com/python/python_multithreading.htm
