---
title: "How I learned to stop worrying and love A/B tests"
date: 2018-05-09T22:11:10+01:00
draft: true
---

## A/B testing and existential angst

So, you run A/B tests. Let me guess.

You often found yourself curled into a ball in anticipation of a product meeting, dreading the "are you sure
about this?" question. Because no, you're not sure. Unless, of course, the improvement is above 10%, in which case you wouldn't
have needed an A/B test anyway.

You often cannot sleep at night, thinking "what if my A/B tests are all wrong?". You read a lot of statistics
book and skim blog posts in the hope that someone has found a solution. In the meantime, most of the uplifts
that you see during the tests do not translate in production.

If not you're probably naive, overconfident, or you are already applying a version of what I am going to
discuss here.

The accuracy (and usefulness) of A/B tests have been questioned a.lot, really.a.lot. Some go as far as saying
that \\(90\%\\) of the A/B tests give wrong (Jeeeeez) conclusions. Of course, you can console yourself by thinking
that, on average, if you're only mistaken when the difference between variants is small, you're good. This
kind of hand-waving argument makes no sense to me: how about if you're always biaised towards accepting the
worst variants?

Surely, there must be something better out there.


## How decisions happen

When we talk about A/B tests, we are usually concern about modelling *what* we are trying to observe---most of
the time, conversion. So we focus on the model a lot, perfect it, perfect the inference method. What happens
past the inference is loosely defined. Perhaps you'll sit around at a product meeting and someone will ask
you: "So, which variant should we pick?". If you're Bayesian you'll probably say that the results are not significant yet, but that
they lean towards variant B. 

- "We can close it then? We really need to move forward with this." the Product Manager says. 

- "Mmm, Guess so". And there goes variant B in production. And here is you anxiously hoping the changes will
  translate in production.

And this cycle just keeps going on. Sometimes you'll ask for more time, everyone will be frustrated, but
you'll get it. Most of the time, due to shitty conversion rate the test would take forever before giving any
sort of reliable results, and you just give up. This is even more true when you're in a startup and
development cycles last typically less than a week.

Everyone get frustrated, you carry an insane amount of stress, and changes that shouldn't see the light of the day
are seen by millions. This is not good for you, the product team, and your application as a whole.

What is the point of crafting relevant models, choosing apropriate priors and diagnosing the sampler if
everything is ditched in the trash during a meeting because you are pressured to give an answer? I know, I
know, people want answers quickly? But how quickly? What price are they willing to pay to get their answer
quickly? Well, these are the questions we need to answer. These are the questions that **you** are responsible
for asking. In short:

**You can't pretend to be serious about you A/B tests if the decision process is not part of
the modeling.**

There is a better way, but it can be painful at first. It requires you to sit down with the product team and
ask yourselves the following questions:

- What uncertainty are we willing to tolerate with regards to X and Y metrics?
- How costly is it for us to wait for an extra day of data collection?
- Is there any situation in which we want to immediately interrupt an ongoing experiment?

I know, it's daunting. But, you may have not realised it yet, but *you are already having that discussion*, or
at least part of it, everytime you're in a product meeting and deciding whether you should continue the
experiment or not. At least, after writing everything down, we're sure everyone is clear about what is about
to happen. When there's a discussion, we can point at the original document and say "these are the terms. Do
you think we should review them?". 

Also, it avoids bargaining. Although we are all tempted to do it when results don't match intuition,
bargaining is bad. 

On your end, as an experimenter, this approach has several benefits. First, you can now model the decision
process mathematically. Say you built a loss function that depends on the parameters you infer, \\(L(\vec
\theta \\), which tells you how much you are losing by choosing a given variant (you should also discuss about
that with your product team, btw). It would make sense to incorporate in this function the loss incured by
waiting for another day/week/month to release the product.

$$
L^F\left(\theta, \delta, n\right) = L\left(\theta, \delta\right) + C(n)
$$

It is always tough to weigh certainty against time-related costs, but this is the real world. If you refuse to
make these choices now, you will be fooling yourself in the future. Luckily, since the decision process is
modeled, you can perform simulations of the process. Start with the scenarios you absolutely want to avoid,
and see if the model is robust against this.

As always, it was worth it to sit down and think hard about this for a couple of days. This had several
benefit, for me and the company. The first, probably the most important, is that it forced us all to sit down
and think hard about our decision process: what uncertainty are willing to tolerate? How costly is it for us
to wait for an extra day of data collection? Is there any situation is which we want to exit the experiment?
Te first one is that you dropped the monkey on someone else's shoulders: deciding when to stop, what is
acceptable, etc. is not your call anymore. Yay! The second one is: you'll be able to sleep at night with the
certainty that everything might slip, and you might make the wrong decision, *but it is one that you will not
regret*.

This may sound very nebulous right now, so I'll soon follow up with a more technical post and give practical
examples of how you may want to implement this new framework.
