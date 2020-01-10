---
title: "Iterate fast, document faster"
date: 2018-06-21T07:43:28+02:00
draft: true
---

Every respectable startup employee will tell you: iteration speed is key to success. It is actually the best
predictor of a company's success. Why iterate quickly, though? Because before finding your product-market fit
you are in a **learning phase**, and you better learn fast before the market swallows you.

At Sounds we pride ourselves in always having a version of the application in review at the App Store, which
mean a release cycle of the order of 4 days. That's fast. So fast, in fact, that the application you have in
your hands right now is very different from the one available last month. [Experimental debt]() is something we
take very seriously: every day that is not spent experimenting with something is a wasted opportunity.

But there is no point in learning if you forget everything, right? So iterating fast is not enough, you also
need to document.

When I arrived, the process was kind of messy, however. One drawback with this approach is that people can
hardly remember what happened in the application last month, let alone six months ago. As a result, I had to
spend hours reading Swift and Kotlin code bases, and worse, had to ask the developers what happened. This is
unacceptable: *Your product is code. In the production cycle, developers are your bottleneck, and you should
get everything you can out of their way.*

**The only thing that developers owe you is clear commit messages and self-explanatory release notes. Become
the most annoying thing in their life until you get that. And then disappear forever.**

# Data is good, documented data is better

Here we have had a long tradition of gathering all data we could. There are two things you need to put in
place:

- Adopt a **naming convention** and write it down somewhere. It may feel inadequate at times, but stick to it.
  Just follow the guidelines. It is also worth thinking about what you are going to do with the data: do you
  want to track intent, or the fact that an event has actually been performed? What you are going to do with
  the data should influence a lot your system. I'll write a follow-up post on the system we adopted at Sounds.
- **Document** your events. Make sure they're up-to-date. All events are described in YAML files that are
  pushed in a Git repository. This approach is ideal and solved two problems:
    + *Differences between platforms:* it wasn't uncommon that Android and iOS clients would send different
    events (one wit caps, another one without, a typo). You cannot avoid that, but you can reduce the
    probability of this happening. Each feature has its branch, and when we are done defining the events we
    make a pull request that the clients can consult. At least they both get the same instructions.
    + *Inconsistency in events, non-documented events:* See the previous point for an example. I once asked
    advice regarding this point to a more senior colleague. His response was that the only way to assure
    consistency was to have eyes on the events. The thing is, it does not need to be human eyes: we have a CI
    process that checks the events in our analytics database against the definitions: if one event is not
    recorded the daily build fails and we are informed. There will be inconsistencies, but we can catch them
    insanely quickly. It also opens the possibility to correct the inconsistencies and feed a second datalake
    with accurate data.
- Take a f-ing phone every week and go through the application with a debugger on. This will teach you how the
  analytics system really behaves. You will be surprised: often the details what you want to be recorded get
  lost in the ticketing system, and there is a slight difference between what you wanted and what the devs
  understood. Learn from this, ask for corrections, communicate your needs better.

# Document your A/B tests

People need to know what was tested, what was te rationale behind the test, what were the results, and what
you decided. **Keep the cohorts**. Bonus points if your analysis is reproducible, publicly accessible and
understandable by people outside the data team. You might want to know what people became later, or re-analyze
the data with a new methodology.

Your idea of the product will evolve, and you might consider whether rolling back an old feature might be
worth it. Indeed, your KPIs might change, and you will remember that at X time we had better value of Y than
now. Well, two options: 

- Roll back and start a new test: ~ 2 weeks. This may be inconclusive.
- Re-run the analysis on a past test with a new KPI ~ 1 hour. This will not tell you whether rolling back in
  the application as it is now will bring you the $$$, but at least you'll rule out stupid ideas.

# Document your product

[Link to twitter thread from VC]()

Work in coordination with whoever is responsible of the product. Put all the features on a timeline ("test",
"production") to know exactly what the application looked like at any point in time. The documentation should
include sketched of the features, the rationale behind them, the tests associated with them, all the relevant
usage analytics. If you have a data scientist it will be highly beneficial for them ('embrace your inner
product personality'); more generally making product and data sit together fosters a **truly data-driven
culture** (Hint: having a data scientist is not enough).

Also, investors will love this.

# Work properly

1. Version your code. But do I really need to tell you that?
2. Document your methodology. You know why you took a standard deviation of 0.75 in this prior, but the next
   person won't --> wasted time.
3. Your analyzes should be reproducible. This means I don't want to have to type more than 5 commands to get
   the results (ideally, I don't want to have to use my mouse).

# Things that I would love having

- A versioned design where I can go back in time to how the app looked like;
- Backward-compatible backend API changes so I can build an old version of the app and use it.


The complaint I often hear is that there is "no time for documentation". That's making excuses: all I've
needed was 3 hours a week, and after 6 months I caught up with the existing system. 3 hours is nothing. If you
don't do this, you will spend weeks trying to understand what the hell happened 3 weeks ago when the retention
started dropping EVERY 2 WEEKS. Another reason to do that is turnover: you don't want to loose all knowledge about the
product's history when a key employee leaves.h
