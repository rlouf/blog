---
title: "Growth metrics and the user lifecycle"
date: 2018-11-05T08:52+01:00
draft: true
---

Over the past years, growth has been measured against a set of metrics, the "pirate metrics": AARRR standing
for:

- Acquisition, aka the number of people who get in your application every day;
- Activation, aka the proportion of these people who perform a meaningful action. For us it would be to play
  at least one track entirely.
- Retention, the proportion of users who are still there after 1, 7, 30 days...
- Referral, The proportion of people who refer the application
- Revenue,  the proportion of people who pay for your service

This framework is useful in that it appropriately describes different steps in the **lifecyle** of your user,
*on which you should be able to work on separately*. Like any framework, it has been heavily praised and
criticised, all for various reasons. I am no different, and I will work towards a more accurate model (at
least in the context of customer apps) pointing out the following flaws:

- Understood literally, aka linearly, the model does not necessarily reflect the lifecycle as you design it in
  your application: you might monetize users way before you can call them retained, way before they refer
  other people.
- It is rigid, forces you to think in a linear fashion, when really customer applications are all about loops.
  As a result it is not always clear what you need to optimize first in your application, the biggest leverage
  that you have.

# The Engine


Let us first look at a '2D' version of the AARRR metrics. Acquisition can come from 3 categories of channels:

- Paid acquisition
- External (linear) acquisition
- Viral acquisition (word of mouth, invitations, etc.)

Ignoring paid acquisition for the moment, two channel remains. But they are not created equal. Viral
acquisition is a result of having new users who get activated and remain long enough to refer their friends,
who become new users, etc. You get the idea: this format of acquisition forms loop. We can thus redraw the
AARRR diagram as follows:

## Free applications, or the importance of viral loops

```
External --> New user --> Activated user --> Retained User
                | 
                |   |
              Referral <-------------------------+
```                

Now you see that External acquisition, while sometimes necessary to kick-start your growth. For instance, when
you don't monetize your application, paid acquisition is an external channel. Viral acquisition is the only
viable channel in this situation. But if your application is free, you better have a pretty strong engine, or
all the money you spend on paid acquisition is just a waste of VC's money.

There may be different viral acquisition channels in your application, and they may occur at different times
in the users' lifecycle. For instance, at Sounds, all of our initial growth was driven by the watermark on
the videos that people shared in Instagram. Although it is a very cheap loop, it allowed us to scale to the 
hundreds of thousands of active user without any other mean. A second viral loop was inviting users to share
their music directly to friends via messages, Instagram direct messages, WhatsApp, Messenger... a pretty
strong loop here too.

This is usually where the `hacking` from `growth hacking` comes in: crafting viral loop necessarily implies
building on existing platforms, and you often need to get creative to get there.

### Variations on the viral loops

While the arrow I drew to Referral started at `Retained User`, this does not necessarily have to be in the
case. At Sounds, we decided to move up the music sharing up on the onboarding (as most users heard of the
application for this) so more users could 'work' on the viral loop. This was a huge success, and gaining
the 40 percentage points of so by starting the loop from the onboarding led to a further increase in active
users. Another strategy we adopted was to increase the multiplication factor (number of times users shared) by
inviting users to share songs they had listend to several times. Same effect.

What is important here is to **map your viral loop** and write down the conversion rate between each steps and
the multiplier factor. It will often be obvious which levers you should pull to 

Growth hacking is all about building and tuning your engine so your application's growth self-sustains.
Several channel will work, at different stages in the life of your application. Some will be sustainable, some
will not, there are a million possible solutions to improve it, but if you don't map you;'re just navigating
in the dark, waiting for the lucky strike. But luck has nothing to do with this. Looking for growth needs to
be tactical.

Also, always be looking for the 10x improvement. You'll have time to worry about the extra 10% once your MAU
hits the 100,000,000 bar.


## Monetized application, and the possibility of paid acquisition loop

Paid acquisition has a bad rap these days: they kill startups, they don't scale, etc. This is true if your
application does not bring any money in the bank, and if it has no viral loop to kickstart. However, if you
have some sort of subscription plans, paid acquisition can be transformed into a growth loop as well. The idea
is simple:

```
Paid Acquisition ----> New User ---> Revenue
```

For this to work and be sustainable, you need to make sure that the LTV of a **new user** is way above the
acquisition cost. Then you have a money printer, and you can scale your paid acquisition, acquiring more and
more as people get in the funnel. Along with viral loops, this can be a deadly combination.

# Wisdom from the trenches

- Improving the onboarding is **always** a huge leverage.
- When A/B testing, you need to be considering the **whole** engine as well as the part-specific metrics. From
  experience, the onboarding (activation) often has a big impact on retention. An app is not the sum of its
  parts, it has a complex dynamics.
- Regarding referral, time to referral is almost more important than K-factor. A 0.2 K-factor with a one day
  time to referral will blow your K=1 app with a 7 day TTR in the water. A better measure is K/day.
