---
title: "Don't use implicitMF for music recommendation"
date: 2018-02-27T00:27:07+01:00
draft: true
---

ImplicitMF has paid its dues: for the past 10 year or so it has been the go-to recommendation method for
implicit feedback datasets. In ways it worked, and it has amply contributed to what the world of
recommendation is like today. If, like me, you have as Spotify account you have probably been served some of
its results and anjoyed it (shoutout to Erik Berhnarhdsson, Chris Johnson and the talented people we don't
hear about who created this fantastic product!).

However, despite its past successes, it's a bad idea to use it for recommendations. Because we can find a
method that beats it on three accounts:

1. **Theory smell:** The method is unprincipled, and doesn't fit music recommendation well. 
2. **Relevance:** We can find a method that produces more relevant recommendations.
3. **Performance:** We can actually find a principled method that is more computationally efficient.

### Theory smell

I have to admit, I am a bit of the new kid in the block here. The little experience I have with implicitMF is
recoding the ALS in Python and then Go when it was too slow. It worked-ish, the recommendations were not too
crazy, but here is what I didn't like about it:

- As I'll show you below, optimizing for \\(L_2\\) norm is equivalent to assuming that the play counts are
  Gaussian-distributed. Not such a great assumption, when you ask me.
- The idea of giving more weight to the non-zero factors is just strange.

These two are indicators of what I call **theory smell**. And like with codes, when your theory smells, it's
time to refactor! Forget about performance consideration for a minute (but just one): wouldn't it be nice to
have a principled approach more realistic and that doesn't need extra assumption? 

Yes.

What we need is a *model* of listening behaviours. The first thing we need is a probabilistic representation
of the number of times a track is listened to. This will be the building block of the algorithm. I do have an
idea of what it may be, but there is nothing better than looking at data to have a quick idea.

[Show here for each song the number of time it's been listened to, and normalize by the average]

Ok, but can't the Poisson distribution be approximated by a normal distribution? Indeed, if \\(x \sim
Poisson(\lambda)\\) then \\(x\\) is approximately distributed according to a normal distribution of mean and
variance \\(\lambda\\) for \\(\lambda\\) **sufficiently large**.  
In pratice you would need \\(\lambda >
1000\\). \\(\lambda\\) represents the average number of times a track is listened to. For the normal
approximation to be correct, you would need people who listen to each track on average for more than 2
consecutive days. I mean, I do like to listen to *Comfortably Numb* on repeat (best guitar solo in history),
but 2 days?! In short, unless your users are fanatics, this will never hold.

### Why matrix factorization to begin with?

### Hierarchical Poisson Factorization

ImplicitMF was a rather *ad-hoc* model, like many models in machine learning: we identify the problem to solve
(reduce the dimensionality), specify an objective function (the \\(L_2\\) norm), and we're good to go for
optimization. In contrast, Poisson factorization is a *generative model*, it tells a story about how the data
were generated.

#### Time budget and popularity

First, it assumes that each user has a finite budget \\(\xi_u\\) in terms of how many songs she is going to
listen to. All budgets are assumed to be drawn from the same \\(Gamma\\) distribution:

$$\begin{align}
\xi_u &\sim Gamma(a', a'/b') \\\\\\
\theta_u &\sim Gamma(\xi_u, a) \\\\\\
\end{align}$$

The \\(Gamma\\) distribution is interesting because [xxxxx].

Then, it assumes that each song is charaterized by its popularity \\(\eta_i\\). All popularities are assumed
to be drawn from the same \\(Gamma\\) distribution:

$$\begin{align}
\eta_i &\sim Gamma(c', c'/d') \\\\\\
\beta_i &\sim Gamma(\eta_i, c)
\end{align}$$

#### Play counts

It then models the distribution of play counts with a \\(Poisson\\) distribution, which makes a lot of sense,
as the number of plays is indeed distributed according to a Poisson:

$$
y_{ui} \sim Poisson(\beta_i^T \theta_u) \\ 
$$

- [ ] intuitive explanation of the algorithm
- [ ] Special cases
- [ ] How the different paramters depend on each other
- [ ] How 0 is taken into account
- [ ] How inference works

### Shortcomings of Poisson Factorization

Theoretical shortcomings:

- The need to specify the number \\(K\\) of latent factors. In practice we scan a widee-ish range of values of
  \\(K\\) until we find an optimum. In theory, we could use Bayesian non-parametric methods;
- Time is not taken into account. Tastes evolve over time and the method only uses an aggregate number of
  plays. You could use some hacky tricks, like assigning a weight that is proportional to the age of a play
  when building the plays matrix, or, simpler (and equivalent if your weighing function is an exponential)
  only counting the plays of the last X days. It probably works in practice, but it's hacky.
- The parameter of the Poisson is a linear function of \\(\theta_u\\)  and \\(\beta_i\\). You should probably
  be able to learn a more general relation using auto-encoders.

Pratical shortcomings:

- Variational inference is sort-of slow. Although I would argue (1) these algorithms don't need to be run very
  often (once a week) (2) it is very easy to scale the computation horizontally, and fairly big inference can
  be achieved in a few hours.
- It's more complicated. Most of the data scientist you will recruit won't have any experience with Bayesian
  statistics, and throwing variational inference at them on their first day is probably going to scare them.
  However, I'd assume you recruit smart people, and smart people catch up quickly. Plus, recruiting a little army
  of bayesians is going to pay in the long run.

Of course we could talk for hours---preferably around a good Bourbon---about all the data the model is not taking
into account (genre, album cover and lead singer haircuts), but these are not shortcomings of the model per
se. At any rate, the model with the additional information should reduce to Poisson Factorization in the limit
where these data are not available.

### Dance off

We judge them on two dimensions: sheer speed, and accuracy of recommendations.

- Performance of ALS on Implicit MF
- Performance of Variational Inference on Poisson Factorization

### Conclusion

[[I]] often hear that variational inference is problematic, as it's not guaranteed that we found th eglobal
minimum. Well, it's not even guarateed with ALS either. But the problem with ImplicitMF is not just about the
inference method, it is that the inference is flawed from the beginning on.
