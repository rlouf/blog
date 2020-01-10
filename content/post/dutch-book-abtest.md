---
title: "AB tests: don't Dutch Book yourself"
date: 2018-05-09T22:11:10+01:00
draft: true
---

The Dutch Book argument is one of Bayesian's favourite argument in favour of the theory. It basically says
that if your decision framework respects the axioms of probability theory, no situation can make you bet
against yourself.

When I set up an experiment, I usually choose in accord with the product team a bunch of metrics to track.
Some are estimated values (revenue), conversion (N-day retention) or frequencies (number of tracks shared). At
the very beginning, we did something that I bet most people are doing: independent Bayesian inferences on each
of the parameters, tried to estimate when the result were significant (in terms of precision) and stop the
experiment then. Then we'd just put our finger in the air and choose the variant was the best. As I explained
before, there were several problems with this approach:

* We didn't really know when to stop;
* Reaching significance\\(^{\text{TM}}\\) could take forever for low conversion rates (typically, revenue);
* The decision process wasn't clear, and led to a lot of bargaining;

We learned from this, and recognising that the problem was typically an application of decision theory, we
started to define loss functions for each experiment. Applying existing models in sequential decision theory,
we also had a consistent stopping rule that only depended on the loss, and the estimated cost of a day of
experiment. That way, we could easily implement a platform that updated the results each day, and decided
automatically to stop the experiments. The decision was easily made, and we usually performed exploratory
analysis to disantangle the loss function and understand the effect of various variables, giving precious
insights for product people. For instance, does the loss function increases because of revenue, or shares? If
it is the latter, is it because people are more likely to share, or because people who do share more often?

It felt good, theoretically and pragmatically. At the beginning, it was hard to make decisions and come up
with good loss functions. But these are decisions you have to make anyway, and it was better to lay them down
mathematically. It really eased the discussions. It sped up the process, and allowed me to sleep at night: we
did not control for false positives, but we were assured to make the best decisions possible given our
constraints.

Amazing, right? Except...

If you change your loss function over time, you can expect that after  given sequence of choices that looked
totally reasonable you are going to end up being worst off in terms of another `reasonable` loss function. The
only way you can avoid betting yourself down if by being **consistant** over the decisions you make. You need
to sit down and think about a single loss function that summarizes the company's goals. This has several
advantages:

1. You are assured to never choose an option that leaves you worse off according to the criterio you chose;
2. This forces you to **sit down** with product owners and stakeholders to determined what exactly we are
   trying to achieve **as a company**.

The choice of the loss function and the way you weigh the different factors is very structuring not only for
the A/B tests, but also for the business as a whole. Any modification to this function will impact the
business' direction in a dramatic way. It can be very simple, of the form:

$$
L(\theta_1, \theta_2, accept) = a \theta_1 + b \theta_2
$$

Or adapt to more complex situations:

$$\begin{align}
  L(\delta_{\$}) &= \infty \quad &\text{if } \delta < 0 \\\\\\
  L(\delta, \delta_1) &= a \delta_1 - b \delta_2 \quad &\text{otherwise}
\end{align}$$

which would entail to rebuke any experiment where there is even a slight chance that the revenue might
decrease, otherwise that we are ready to loose \\(b\\) in \\(\delta_2\\) for a gain \\(a\\) in \\(\delta_1\\).
This goal is unrealistic, but this was just to prove a point.

### Who does this?

To my knowledge, no one. But surely, the bigs must implement some version of this. At the time of writing of
[this article](link to airbnb), data scientists at AirBnb seemed vaguely aware of this problem, and imposed
the use of at least one core metric in the experiment. I would assume that a variant is rejected if all lights
are gree *except* for the core metric. This, indeed, is a way to prevent betting against yourself when you
perform an experiment. 

But it's not enough. What if, following the previous examples, you were monitoring, say total number of
bookings as a core metric leaving aside the return rate of users?

{{< highlight python >}}
def function(x,y):
  return x+y
{{< /highlight >}}
