---
title: "The Engine: a framework for growth"
date: 2018-11-05T22:00:10+01:00
draft: false
---

A popular set of metrics against which growth of applications can be measured is the ["pirate
metrics"](https://www.slideshare.net/dmc500hats/startup-metrics-for-pirates-long-version),
or AARRR metrics:

- *Acquisition:* the number of users who get in your application every day;
- *Activation:* the proportion of these users who perform a meaningful action.
  For a music streaming app it could be to play at least one song entirely. For
  a food delivery service, placing an order;
- *Retention:* the proportion of users who are still there after 1, 7, 30 days,
  etc[^2];
- *Referral:* The proportion of users who refer the application to potential
  users;
- *Revenue:* the proportion of users who pay for your service.

If you interpret the framework literally, it leads to the following diagram:

```bash
New user --> Activated User --> Retained User --> Referral --> Paying User
```

This framework is useful in that it appropriately describes different **steps** in
the lifecyle of your user on which you should be able to work on
separately. However, like every framework--including the one presented here--it
has its shortcomings. Mainly:


1. The model does not necessarily reflect the lifecycle as you design it in your
   application, the order of the different steps and how they relate to each
   other. For instance, you might monetize users way before you can call them retained
   or encourage them to refer other people.

2. The framework is rigid, invites you to think in a linear fashion[^1], when
   customer applications are all about loops. As a result it is not always clear
   what you need to optimize first in your application, where your biggest
   leverage is; improving one step of the lifecycle may actually improve every
   other step.


When working on growth, it is important that you have a clear view of the
current user lifecyle in your application. To be able to find the biggest levers
to pull, you need to graduate from this linear representation.


# The Engine

When I worked on growth at my previous gig, I liked to focus instead on what I
called the "engine": the set of loops that have the potential to make the growth
of your application self-sustainable. Without an engine you are relying solely on
external supply, which will eventually die out because of churn.

You can think of the framework as a 2-dimensional version of AARRR. In the
following, I will mostly focus on acquisition and referral.

#### Free applications: the importance of viral loops

Acquisition can come from 3 channels:

- Paid acquisition
- External (linear) acquisition
- Viral acquisition (word of mouth, invitations, etc.)

External acquisition may be thanks to your website, a post on Hacker News. It
brings in a constant number of users, but does not scale. We include paid
acquisition as an external channel for free applications: it only increases with
the amount of money you have in the bank, not the number of users that you have.

Viral acquisition on the other hand is a result of having new users who get
activated and remain long enough to refer their friends, who become new users,
etc. Viral acquisition forms loops.

We can redraw the AARRR diagram for our free application as follows:


```bash
External --> New user --> Activated user --> Retained User
                |                                |
                |                                |
              Referral <-------------------------+
```                

Your engine is viral acquisition. It is the only viable channel in this
situation, the only one that loops. The one you should concentrate your efforts,
but the hardest to work on. I suspect many startup spend a lot of money in ads,
because it is much easier than spending time improving their engine. It often
doesn't pay off.

There may be different viral acquisition channels in your application, and they
may occur at different times in the users' lifecycle. For instance, at my
previous company, all of our initial growth was driven by a watermark on videos
that people shared in Instagram. Although it is a very cheap loop, it allowed us
to scale to the hundreds of thousands of active user without any acquisition
effort. A second viral loop was inviting users to share their music directly to
friends via messages, Instagram direct messages, WhatsApp, Messenger... a pretty
strong loop here too.

While the arrow I drew to `Referral` started at `Retained User`, this does not
necessarily have to be in the case. We decided to move up the music sharing up
on the onboarding (this is what most users came in for) so more users could
'work' on the viral loop. The engine roughly looked like this:

```bash
                +--------------+
                |              |
External --> New user ----Onboarding----> Activated user --> Retained User
                |                                |               |
                |                                |               |
              Referral <-------------------------+---------------+
```                

This was a huge success, and gaining the 40 percentage points or so by starting
the loop from the onboarding led to a further increase in active users. You can
see that thinking in terms of loops is way more flexible. It can map the
lifecyle of users *in your application* more accurately.

Growth is about, first and foremost, *crafting* loops in your application to
*build* a reliable engine. Without an engine you don't have a rocket, but a
leaky bucket.

#### Not all loops are created equal

There are usually several levers you can pull to improve the engine. If I take
the example of the watermark loop mentionned earlier, we could work on two
aspects:

1. Improve the conversion `song shared --> new user`. There are several things
   we could work on: how visible and appealing the watermark was, the hashtags
   that we inserted by default in the post, appealing to the users that have the
   largest number of followers, etc. 
2. Improve the multiplier by increasing the average number of track
   shared per user.

Assuming that the conversion rate `r` is not affected by the number of shares
`N` per user (not true in practice), the average number of new users coming from
this loop is given by `r * N`. This looks ridiculously obvious, but now that you
formalized this, it becomes easier to quantitatively compare your options thus
make decisions. At the time, it seemed easier for us to work on the average
number of tracks shared by user, so we made it a priority, and it served us
well.

It is therefore important to **map your viral loop** and write down the
conversion rate between each steps and the multipliers. Take your app, a
whiteboard, and go through it carefully. Pull your analytics and write down the
numbers. It will often be obvious which levers you should pull once that work is
done. Don't forget to write down timescales sa well: an ok performing loop
activated daily is always better than an incredible loop that is activated once
a week. Growth hacking is all about building and tuning your engine so your
application's growth self-sustains.


> If you don't map your engine then you're just navigating in the dark, waiting
> for the lucky strike. But luck should have nothing to do with this. Looking
> for growth needs to be tactical.

Several channel will work, at different stages in the life of your application.
Some will be sustainable, some will not, there are a million possible solutions
to improve it. So monitor the life of your engine closely. It is usually easy to
write down a formula that sums it up in one number (e.g. the average number of
users that come in the application after a referral per day per user in the
application, or time-adjusted K factor), and follow it on a dashboard. You only
need to look at the loop-level details when there is an issue.

Having said that, don't get too lost in the details of your map. Always be
looking for the 10x improvement. You'll have time to worry about the extra 10%
once your MAU hits the 100,000,000 bar.


#### Monetized application & paid acquisition loops

Paid acquisition gets a bad rap these days: they kill startups, they don't scale,
etc. This is true if your application does not bring any money in the bank, and
if it has no viral loop to kickstart. However, if you have some sort of
subscription plans, paid acquisition can be transformed into a growth loop as
well. The idea is simple:

```bash
Paid Acquisition ----> New User ---> Revenue
     |                                  |
     +----------------------------------+
```

For this to work and be sustainable, you need to make sure that the revenue you
make per **new user** is way above the acquisition cost. Then you have a money
printer, and you can scale your paid acquisition, acquiring more and more as
people get in the funnel. Along with viral loops, this can be a killer
combination.

Here is what our engine looked like with paid acquisition:

```bash
                +-------------------------------------------- Paying User
                |                                                |
External --> New user ----Onboarding----> Activated user --> Retained User
                |              |                 |               |
                |              |                 |               |
              Referral <-------------------------+---------------+
```                

Word of caution: do not compute user revenue in terms of Lifetime Value, unless
you have an unlimited amount of cash and all the time in the world. Timescales
matter when it comes to paid acquisition, and you should think hard about them.
If it takes you 5 months to repay the acquisition cost of one customer, and you
do not have much cash to spend, you are buying yourself a very slow growth loop.

We made that mistake and realized it too late. We had a very good LTV on new
users as retention on our paid plans was extremely good and the prices, well, high.
But it took several months to repay the acquisition of one customer. The cash we
gave to Facebook would have probably been better spent on someone working full
time on other loops.

So don't think in terms of LTV in early-stage, it is more useful to think in
terms of

- Growth loop: is this acquisition channel sustainable? Does it bring me more
  cash than I spend? If this is your only growth loop, it better repay itself.
- Timescale: How long does it take to repay itself? A day, a week? (worth doing)
  A month? (think hard) 6 months? (don't do it).

In summary:

> Paid acquisition is interesting if it repays itself quickly, or to kickstart an
> organic growth engine.


# Other lessons

To finish, here are a couple of things I that learned working on growth:

- Improving the onboarding is **always** a huge leverage. The further you go
  down in the onboarding funnel, the fewer users are still with you. And it can
  drop quickly. Engineering a loop very early in the funnel is always a very
  good idea.
- Think about *where* you can position your loop as much as *how* your loop is
  implemented. I've seen too many referral schemes hidden in the application.
- When you A/B test a specific part of the engine, you need to
  be considering the whole engine as well as the part-specific metrics. **A
  growth engine is a system, not just the sum of different parts.**  Crafting a
  loop in your onboarding may increase the number of referrals, but you may have
  shot yourself in the foot by killing retention. Think in systems.
- **K-factor is meaningless**. Time-to-referral (TTR) can be more important. A K-factor
  of .3 with an average one day TTR will blow any app with a K-factor of 1.1
  with 7 day TTR out of the water. **A better measure is K/day.**
- I'll say it for the 5th time because I don't see it often in discussions:
  **timescales matter more than anything else when it comes to growth.** Time is
  insanely valuable in early stage.

[^1]: I don't think it is a useful metrics in itself, usually there is a
  *reason* why you want user to be retained: more revenue, more referrals, more
  DAU?
[^2]: I do not think that the framework was meant to be interpreted linearly,
  but I have seen many people do so. It's worth repeating. 
