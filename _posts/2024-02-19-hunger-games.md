---
title: 'The Hunger Games: NYC Edition'
layout: post
featured-img: jules-ari-tv-dinner
categories: [Development]
summary: It's an absolute battle to get to get a reservation in NYC
---

# Introduction

There's a lot to love about New York. And there's also inevitably some things that are frustrations of the city. The big ones for me are:

1. Trying to find a coffee shop to work at when I want a change of pace
   a. Sub-point: finding a coffee shop with good wifi speeds
2. Playing tennis in the city is basically impossible and you need to go through the whole song and dance of having a parks permit and then waiting 2+ hours for a public court in Brooklyn or on West Side Highway.
3. Booking a fun / popular restaurant is basically an arms race.

We're going to focus on the third one in this post.

# Context

My girlfriend is a bit of a foodie. She's lived in New York for a long time and as a result, she knows the city better and therefore the restaurants waaaay better than me.

Many restaurants are on [Resy][resy], which is a great company and website. And some have very absurd and artificial rules about like "ok we only post 7 days before at 9am". And that's where we're going to focus in. Specifically - and I'll drop names - on [Don Angie][don-angie].

![path1](/images/hunger-games/don-angie-original.png){: .center-shrink }

<center> <i> Oh what a pleasant and cute restaurant... or is it? </i> </center>

# Approach and Kudos

Don Angie does exactly this - list reservations 7 days out at 9am. So I check at 9:10am one day for a week out and literally all the reservations from 5pm - 11pm are gone :thinking:. I don't know about to you, but that seems _slightly_ suspicious. I assume it's because there are loads of bots targeting these high profile restaurants and then selling reservations on a secondary market because people are scumbags.

I figured that's ok though because... if you can't beat 'em, join em. I'll just automate this at well and join the foray.

Now, I'm not going to lie, there's not a lot of code I wrote for this post, so I largely want this to just be an attribution of this [Github Repo](https://github.com/robertjdominguez/ez-resy). I want to give a big shoutout to Rob Dominguez for making this an open source project and not being an asshole. [Here's][rob-gh] his Github repo and [here's][rob-dev] his actual webpage. So feel free to give him some click traffic.

# Technical Solution

Again, there wasn't too much that I needed to do here. I checked out the above repo and then built a little wrapper script. I made a fork here: https://github.com/johnlarkin1/ez-resy-fork . It has a couple of helpful scripts good for targeting from cron (setting up environ, retry logic, etc).

The way the repo works is that you:

1. Populate a `.env` file (see his repo for instructions on where to get what values)

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/ez-resy ‹main*›
╰─➤  cat .env
VENUE_ID=
DATE=
EARLIEST=
LATEST=
PARTY_SIZE=
PAYMENT_ID=
AUTH_TOKEN=
```

2. I built out a wrapper so that we would have retries if there was some race condition about that. It's more or less here:

```bash
#!/bin/bash

# Maximum number of attempts
max_attempts=5
attempt=0

# Command to run
command="npm run start:today"

# Directory where your npm project is located
# TODO(@user)
project_dir=$PWD

# Change to the project directory
cd "$project_dir"

# Run the command and retry upon failure
while [ $attempt -lt $max_attempts ]; do
  echo "Attempt $((attempt+1)) of $max_attempts"
  if $command; then
    echo "Command succeeded."
    break
  else
    echo "Command failed, retrying in 20 seconds..."
    sleep 20
  fi
  attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
  echo "Command failed after $max_attempts attempts."
fi
```

For my specific setup, I also needed a wrapper script to have my PATH be setup correctly with the right version of `node` and all that.

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/ez-resy ‹main*›
╰─➤  cat wrapper_script.sh
#!/bin/zsh
# Load the zprofile
source /Users/$USER/.zprofile

# TODO(@user): Modify path here accordingly
PATH=$PATH

# Execute the original script, change path as needed
/Users/$USER/Documents/coding/ez-resy/attempt_book_and_retry.sh
```

3. Target with cronjob for exactly when the reservation becomes available.

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/ez-resy ‹main*›
╰─➤  crontab -l
0 9 18 2 * /Users/johnlarkin/Documents/coding/ez-resy/wrapper_script.sh >> /Users/johnlarkin/Documents/coding/ez-resy/cron.log 2>&1
```

So 😎💣 and then you _should_ be all set.

# Success?

But are you? Actually? Yeah, actually, no :upside_down_face:.

![path1](/images/hunger-games/es-rezy-failure.png){: .center-shrink }

Yeah. That's right. By my guess, at 9:00:01am, all of the reservations are STILL gone for Don Angie. The way I see it is there's two possibilities...

## Dystopia Reality

So two possibilities:

1. My automation is just slightly too slow (although I'd be shocked if there was an army of bots specifically targetting this restaurant each day??)

Call it hubris, but I don't think that it's the first option. I would imagine that there's not that many bots / accounts that are specifically targetting this restaurant each day. And if I'm wrong here, then perhaps it's me underestimating New York's competetive edge, which wouldn't be that surprising.

The second scenario is even more scary...

2. **Don Angie is lying and these reservations are not actually listed right at 9:00am.**

But I think the more likely scenario is that Don Angie is just lying to us.... and that means you should think twice about this cute West Village dining establishment. Perhaps... it's not what first meets the eye. And I hope I'm wrong and this blog post goes viral and Don Angie calls and offers me a reservation. But somehow I doubt that.

![path1](/images/hunger-games/don-angie-good-to-evil.gif){: .center-shrink }

<center> <i> A true glimpse at the horrors of Don Angie </i> </center>

That being said, this tool will be helpful down the road in New York - I'm sure of it.

Thanks for reading as always! If you have interest in setting this up, or need help, feel free to reach out.

[comment]: <> (Bibliography)
[Github Repo]: https://github.com/robertjdominguez/ez-resy
[resy]: https://resy.com/
[don-angie]: https://www.donangie.com/
[rob-gh]: https://github.com/robertjdominguez
[rob-dev]: https://www.dominguezdev.com/
