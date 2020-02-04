---
title: "Introducing Birdland"
date: 2018-06-04T22:11:10+01:00
draft: true
---

This post is a long overdue post on a recommendation algorithm that I developed
and implemented with colleagues in my previous job: [Birdland].

## Content

- [Bye implicitMF]()
- [Requirements]()
- [Birdland]()

## Bye, implicitMF

When I was asked nearly 2 years ago to design a recommendation algorithm, my
first incline was to look into implicitMF, and its social variations.

For the past 10 year or so it has been the go-to recommendation method for
implicit feedback contexts. In ways it worked, and it has amply contributed to
what the world of recommendation is like today. If, like me, you have as Spotify
account you have probably been served some of its results and anjoyed it
(shoutout to [Erik Berhnarhdsson], [Chris Johnson] and the talented people we
don't hear about who created this fantastic product!).

However, despite its past success, we ended up deciding it was just not going to
cut it. There are two things that I try to steer away from when I work on a
machine learning problem: complexity and theory smell. Unfortunately, ImplicitMF
is both dubious theoretically speaking, and harder to implement that it seems.

### Play counts are not normally distributed

Let us go into the implications of implicit MF for a second. What you do,
basically, is optimizing the L2 norm of the following function:

Which is equivalent to finding the maximum likelihood of a Gaussian
distribution. Not quite true, it actually follows a Poisson distribution.

### The hidden cost of Implicit MF

The result of implicit Matrix Factorization is two matrices song-vector and
user-song; it is an embedding of songs and users in a K-dimensional space. But
this embedding is the first part of the process. What the tutorials you read on
the web usually don't tell you is that once your implicitMF implemented and
connected to the play database you need to:

- Store these vectors in a database and recompute these vectors in batch.
- Given a user and its past plays, recommend new songs.

Without getting into details, you already see that what seemed to be a simple
recommendation service actually ends up being a 3 services ordeal: a server to
compute the vectors, a database to store these vectors, and a server to serve
recommendations based on user data. Not that simple, is it?

What is also omitted is that the recommendation part of the algorithm is
complicated, way complicated than the implicitMF part: the N-nearest neibour
problem.

The idea is to find the "song" vector that are the closest to your "user"
vector. For each user there are N such posibility. In real settings, this simply
does not scale. There are very clever approximation method that make the
computation scalable, but their details are much harder than implicitMF's.

See, had you started to work on implicitMF and went as far as factorizing the
plays matrix, you would have ended up in front of a much harder problem.


## Our requirements

Birdland was born in a particular context. At the time I was working as Chief
Scientist (whatever that means) at a small Startup that built a social app
around music. To be completely honest, the goal of this project was more to show
off our capabilities (and spend funding we'd earned to develop it) rather than
answering a real need. Actually, the first version of "recommendation" was
simply to recomend to users songs they listened to in the past. Which made for a
great baseline for A/B testing.

- Data was not the product. My main role was in support of the product:
  analytics, A/B testing, modeling behavior, etc.
- The team moved fast, really fast. Development cycles were of about one week;
  this meant a new version of the application was released every week. We had
  maximum 2 development cycles to implement something new.
- Not many backend resources allocated to the project;

Sure, I had read a lot about (music) recommendation; I'd even implemented
Poisson Factorization (try it!) on my free time. But fancy stuff was not going
to cut it given the tight deadline. So the requirements were:

- Two weeks from first draft to production;
- Single server; no preprocessing allowed;
- We'd sold social recommendation to get funding, and the social aspect was the
  only edge we had on bigger services such as Spotify.

## Birdland
