---
title: "How I learned to stop worrying and love A/B tests"
date: 2018-05-09T22:11:10+01:00
draft: true
---

## A/B testing and existential angst

You run A/B tests. Let me guess:

You often found yourself anxious before product meetings, dreading the "are you
sure this version is better?" question. Because no, you're not sure. Unless, of
course, the improvement is above 10%, in which case you wouldn't have needed an
A/B test anyway.

You keep wondering "what if my A/B tests are all wrong?".  You read a lot of
statistics book and skim blog posts in the hope of an answer. In the meantime,
most of the uplifts that you see during the tests do not translate in
production.

If not you're probably naive, overconfident, or you are already applying a
version of what I am going to discuss here.

The accuracy (and usefulness) of A/B tests have been questioned a.lot,
really.a.lot. Some go as far as saying that \\(90\%\\) of the A/B tests give
wrong (Jeeeeez) conclusions. Of course, you can console yourself by thinking
that, on average, if you're only mistaken when the difference between variants
is small, you're good. This kind of hand-waving argument makes no sense to me:
how about if you're always biaised towards accepting the worst variants?

Surely, there must be something better out there.


## How decisions happen

When we talk about A/B tests, we are usually concern about modelling *what* we
are trying to observe---most of the time, conversion. So we focus on the model a
lot, perfect it, perfect the inference method until we get a satisfactoy
estimate of the parameters we are interested in.

What happens past the inference is loosely defined. Perhaps you'll sit around at
a product meeting and someone will ask you: "So, which variant should we pick?".
Several uncomfortable discussions can happen at this point:

#### The estimation of the uplift for one or more variables is not significant

If you're Bayesian you'll probably say that the results are not significant yet,
but that they lean towards, say,  variant B. 

- "We can close it then? We really need to move forward with this." the Product
  Manager says. 
- "Mmm, Guess so". And there goes variant B in production. And here is you
  anxiously hoping the changes will translate in production.

#### Some metrics are up, some are down

Let the bargaining festival begin. I mean, which metric is more important? How
much are we willing to sacrifice one for the other? You'll hastily make a
decision, and probably a different one each time, depending on the day's mood.
The discussions are *important*, but they happen quickly when they do happen at
all.

<!--will give example below:-->
<!--I have a personal example of this. The obsession of the moment was user-->
<!--retention, so we would only accept experiments that improved user retention. At-->
<!--some point we did heavy experimenting -->

And this cycle just keeps repeating. Sometimes you'll ask for more time,
everyone will be frustrated, but you'll get it. Most of the time the uplifts
don't show up in production. Sometimes the uplift in one area lead to a disaster
in others that the product managers hadn't internalized during the meeting.
This is even more true when you're in a startup and development cycles last
typically less than a week.

Everyone get frustrated, you carry an insane amount of stress, and changes that
shouldn't see the light of the day are seen by millions. This is not good for
you, the product team, and your product.

What is the point of crafting relevant models, choosing apropriate priors and
diagnosing the sampler if everything is ditched in the trash during a meeting
because you are pressured to give an answer?

I know, I know, people want answers quickly. But how quickly? What price are
they willing to pay to get their answer quickly? Well, these are the questions
we need to answer. These are the questions that **you** are responsible for
asking. In short:

>> You can't pretend to be serious about your A/B tests if the decision process
>> is not part of the modeling.

This requires you to sit down with the product team and ask yourselves the
following questions:

- What uncertainty are we willing to tolerate with regards to X and Y metrics?
- How costly is it for us to wait for an extra day of data collection?
- Is there any situation in which we want to immediately interrupt an ongoing experiment?

It's daunting, it will probably require a lot of back-and-forth. But don't give
up until you have a clear answer, something you all agree one; make sure to
write it down somewhere. 

If you still doubt the usefulness, think about it: you may have not realised it
yet, but *you are already having that discussion*, or at least part of it,
everytime you're in a product meeting and deciding whether you should continue
the experiment or not.  At least, after writing everything down, we're sure
everyone is clear about what is about to happen. When there's a discussion, we
can point at the original document and say "these are the terms. Do you think we
should review them?". 

Also, it avoids bargaining. **Although we are all tempted to bargain when results
don't match our intuition, or dispell our baby, bargaining is bad.**
Being clear from the beginning about what is and what is not a good result
avoids this.

On your end, as an experimenter, this approach has several benefits:

- You can now model the decision process mathematically. Say you built a loss
  function that depends on the parameters you infer, \\(L(\vec \theta \\), which
  tells you how much you are losing by choosing a given variant (you should also
  discuss about that with your product team, btw).
- You can reduce the cognitive load by monitoring a single variable: the loss
  function.
- You can include many externalities in the loss function: the cost of releasing
  something in production (which would biais towards status-quo), the cost of
  waiting for another day/week/month (which would biais towards riskier bets).

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
