---
title: "Why I am moving to finance"
date: 2020-09-08T08:50:46+02:00
draft: true
---

Back when I was a student in theoretical physics it wasn't unusual to receive an
email from recruiters for large head funds or banks (mostly based in London). At
first I remember finding the idea completely absurd. After all, I had chose this
path to understand the universe, and not merely look at numbers moving up and
down to make my managers' go up.

But the it planted a seed. I knew nothing about the field of quantitative
finance and became curious about it. A well-known physicist in France,
Jean-Philippe Bouchaud, had started his own hedge fund, quite successfully. He
also wrote a book about it, so I bought it and started reading. It taught me
that the number were not merely moving randomly, there was structure to that
randomness. Structure we could try to understand, to our advantage of course. So
I considered doing my PhD thesis with him, but backed off last minute to work on
[something else]().

At the same time, I started building a trading engine in my free time. But I
gave up due to my lack of programming skills, and the huge time commitment that
my MSc was.

I was happy to be where I was as a PhD student and postdoc. I liked my topic,
the novelty and the potential impact it could have---towards the end I'd started
working on measures of residential segregation; the other ones are terrible, do
not reflect reality and are not actionable. I was going to get a permanent
position, my research project got a lot of interest among committee members and
I had to defend it orally. I didn't go, having realized that doing research
while living in Paris with a newborn and a low salary (30k/year) was the
privilege of a social class I did not belong to. That reality had never hit me
in the face before; I was able to go from scholarship to a school where students
are paid (Ecole Normale Supérieure, true story) another scholarship to go to
Oxford, another one to do my PhD and another one to do my postdoc. I flew over
social inequality by being lucky enough to get the scholarship so many of us
were after. But there it stopped, and crushed my dreams. I was bitter. I still
am to some extent. I would like to be able to say that I was crushed by
academia's toxicity and the administrative burden. But I wasn't, I learned to
deal with toxicity and I loved it.

Now that great endeavors were off the table, I started applying for jobs. I got
yet another email from a hedge fund recruiter and finally obliged. But when I
got an offer I backed down. Finance, at the time, was evil. 2008 was still in
everyone's mind. 

At the same time I was also applying for Palantir and was willing to go for it.
I learned later from a friend that I was rejected because I seemed reluctant to
some of the company's deployments. It was probably true, but it was certainly
not conscious. I remember feeling some cognitive dissonance at the time, but
this feeling had never translated into a conscious state.

It was interesting that throughout these experiences, whenever I got bored I
would always pick mathematical finance papers and try to understand what there
was to it. I bought books, read blog posts, to understand the whole chain from
making signal out of time series to placing orders. The field is very secretive,
which made it very interesting.

Here are a few things I learned over the years. I am convinced that finance is
one of the most interesting field to be in as a data scientist. 

First, the signal to noise ratio seem to be incredibly low. It seems challenging
to find paterns, find information. More that in, say, understand the behaviour
of users in an application. The movement of prices, after all, can be
succesfully modeled by a random walk. What is fascinating, when you go beyond
that, is that this seemingly random behaviour is the combination of more or less
rational behaviours made by different types of actors. If you can identify this
behaviour, you can extract a lot more signal from this noise.  There is of
course the temptation as a DS to throw trees and LSTMs at the problem to do the
work for you. But there is some satisfaction in identifying a generative process
that captures part of the reality.

Second, market data are *clean*. There are caveats to be aware of (splits,
dividends, market opening and closing times among other things), but what you
get is the hard truth, not a proxy. After years in analytics it is truly a joy
to not have to worry about this sometimes.

Third, there is so much more data you can use. What we call *alternative* data.
Satellite data (to track those supertrankers or parking lots), grocery shop
ticket data, etc. And so many data science technique you can apply to them.

Fourth, time series are *hard*, harder that deep learning in my opinion. Having
worked a bit with them and read a ton of literature it always feels that
something is conceptually wrong. For instance, take the obsession of quants with
making time series stationary. They often proceed by studying returns, which are
first order derivatives. But computing returns destroys some signal. Having
stationary time series is a necessity to apply traditional ML algorithms, but I
think it reflects more a problem with our tools that with the data itself.

Fifth, it is a perfect playing ground for bayesians. In the end, everything we
do in finance is geared towards making a decision. Bayesian decision theory is
the optimal tool to make decisions

Uncertainty on correlation matrices is huge. Frequentists say they are noisy,
and use many clever tricks (like random matrix theory) to regularize them so
they can be used in portfolio optimization. But Bayesians will say we're just
very uncertain about their values, and take this uncertainty into account when
making decisions. I think it is fair to say that I have seen more research on
applying Bayesian method to finance than I have seen to any other field I have
worked in.
