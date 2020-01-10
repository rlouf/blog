---
title: "5 minutes recommendation"
date: 2018-06-04T22:11:10+01:00
draft: true
---

I hate complex solutions. I really do. I don't know if this comes form my background in physics where
simplicity is hailed as the greatest virtue, my background in philosophy of science, or my job as chief
scientist in a small startup. I live by the mantra

> A simple model in production tomorrow is better than a deep learning solution in 6 months.

This might not apply if you work for a company that has the luxury to hire 100 data scientist (lucky you!),
but it definitely applies to people working in a small startups. Just to give you some context about Sounds:

- Data is not the product. What we do mostly happens behind the scene: A/B testing, modeling of user
  behaviours, ect. We are here in support of the product.
- We move fast, very fast. We release on average once every week on iOS, and twice on Android. If it wasn't
  for the logs I wouldn't be able to tell you what we did a month ago. Seriously.
- The necessity to iterate crazy fast is because, despite a comfortable user base, we are still looking for
  THE THING. And to find THE THING, sitting at a table making some drawings can only go so far. You need to
  roll in production, and roll it NOW. Every day you're not releasing you're loosing time that could be used
  testing new ideas. I call that experimental debt, and I'll probablyt write a couple of blog posts about that
  soon.

So when I was asked to devise a recommendation algorithm that is good enough to not deter users in less than a
week. Well, I panicked. I'd prepared for that for months. I read about collaborative filtering, delved into
the reasons why it may not be appropriate for music recommendation. Read about Poisson Factorization, and
variational inference (even implemented it). Song2Vec, sequence learning, LDA. So on and so forth. But nothing
prepared me to the CEO standing at the board on Monday and saying: "We're releasing this on Friday". 

Now, the irony is that I'd been asking this question at interviews for the past 6 months. I found it was a
great way to assess how pragmatic a data scientist can get, and how creative they get when resource is
limited.

My thought process was:

- Fancy stuff is not going to make the cut. Assuming we're going with something I've already coded locally,
  like PMF, it will require too much work on the backend, even if we join effots
  with the backend.
- OH MY GOD WHAT AM I GONNA DO?????

Then sat down, reformulated the problem as a question (always do that!):

> Given what the user has listened to in the past, what is she going to do next?

You can turn this in an inference problem. Given a song \\(s\\) and the listening history \\(\mathcal(S)\\)
what is the value of the probability:

$$
P(s|\mathcal{S})
$$

traditional factorization approaches suppose that probabilities are a function of the dot product of two
vectors, one proper to the user \\(\theta_u\\) and one linked to the song \\(\beta_i\\). In the case of
poisson factorization you would write that the number of plays of the song \\(i\\) by the user \\(u\\) is
given by

$$
y_{ui} \sim \text{Poisson}(\theta_u^T\;\beta_i)
$$

In the case of traditional NMF you would assume it's a Gaussian. If you really want to be fancy, you can
assume it's a non-linear function of the two vectors and use Variational Auto-Encoders to do the job (good
luck scaling that thing in less than a week!).

Too complicated for us. What is we drop the personalized aspect for a second a do a "mean-field" (one of these
statistical physics terms that mean nothing and everything) recommendation? Then I just have to write the
probability to observe a `transition` between \\(i\\) and  \\(j\\). This follows a binomial distribution:

$$
N_{ij} \sim \text{Binomial}(N_i, \theta_{ij})
$$

where $\theta_{ij}$ is the probability of a transition \\(i\rightarrow j\\) to happen (given you are at
\\(j\\)). These probabilities are fairly easy to estimate: you just have to project the user-song bipartite
network into a song-song network, and look at the number of times a song is connected to another. We loose the
information that was contained in the bipartite network, but heh, we only have one week.

And this, my friends, can be written as a SQL query. I can feel the backend love already. In practice, though,
we ended up re-writing this part in C to bind it with Go and Pfew! Real-time.

One problem with this approach, especially if you don't take time into account (don't have time! PS: pun not
intended) is that you will end up recommending Justin Bieber to everyone. Not because I don't like his music,
but because you want to offer people more diversity (otherwise you might as well recommend the most famous
songs), more relevant recommendations. My first impulse was to write:

$$
score(i \rightarrow j) = \frac{\theta_{ij}}{\theta_j}
$$

where \\(\theta_j\\) is the probability for the average user to listen to \\(j\\) and can be infered from:

$$
N_i \sim \text{Binomial}(N, \theta_j)
$$

Mmm. That feels probably right. But if there is anything that makes me cringe more than complex models, is
hacky tricks (unless the perpetrator takes full ownership). Non-principled methods. 

> Build an approximate method where you state all the assumptions, you get the more complex model for free
> (devs will disagree). Build a hacky model, and all you've got is a hacky model.

Re-reading the formula, it occured to me that this is very similar to the tf-idf trick in language retrieval.
Think about the transitions starting from \\(i\\) as the terms from the same document \\(i\\) and the number
of times \\(j\)) appears as the frequence of the term \\(j\\). Neat. This is probably linked to what we mean y
`relevant` 

> I learned very early the difference between knowing the name of something and knowing something.


# Markov chain

The information we have originally is a bipartite graph user-track, represented by the matrix \\(M\\) where
\\(M_{ij}\\) is the number of times the user \\(i\\) has listened to the track \\(j\\). To simplify the
problem, we choose to project the bipartite graph on the songs, where the weight of the link between \\(s\\)
and \\(t\\) is given by the number of times \\(s\\) and \\(t\\) co-occur the users' listening history.

Let us now imagine a random user that would have to navigate through this graph by choosing at each step a new
song to listened to. The user needs a rule to choose a song over another. We would like here to behave like a
`typical', average user. If we want her to do so, we require that the distribution of the number of times she
has listened to a song needs to match the global distribution.

We are transition between songs \\(i\\), and each of the songs have a probability of appearing \\(P(i)\\).

\\(p(i)\\)  is the function that associates to each song \\(i\\) the relative number of times it has been
listened to by all users. It defines a probability measure on the space of songs since

$$
\sum_j p(j) = 1
$$

Knowing a users' probability measure on the space of songs, which one should we recommend?

If we choose the kernel \\(p(j|i)\\) then a random walk on this space keeps the distribution intact: if we put
a person at either of the nodes then it will reproduce the exact distribution \\(p(j)\\) after a long enough
time. Therefore, the strategy thats consists in, at each step, to recommend a song starting from what the
person has already listened to is the global distribution. This is not personalized, and probably not a great
distribution: one will end up listening to the most popular songs.


$$
\begin{align*}
\omega'(A) = \sum_i \frac{r(j|i)}{r(j)} r(i) \\
          &= \sum_i \frac{N^{*}_{ij} N}{N_i N_j} \frac{N^*_i}{N}\\
          &= \sum_i uniform 
\end{align*}
$$

A plausible transition function wrt to the global measure is:

$$
T(i,j) = \frac{r(j|i)}{r(j)}
$$

That is not a Markov kernel.



## The network perspective

One of the most basic models in graph theory is the Erdos-Renyi model where, at each step you draw two nodes
at random and draw an edge between them. This gives you a nice baseline to which you can compare all the
models you can come up with. In this situation, this assumes that people are listening to songs at random in
this case the probability of having an edge between \\(i\\) and \\(j\\) is constant

$$
P_{ij} = p
$$

The degree distribution (number of times a song has been listened to is then given by a binomial distribution
that is well-approximated by a Poisson distribution when \\(n\\) is large.

The Erdos-Renyi model is very simple, because it draws on very little information. In reality, the data give
us an information: the degree distribution is a given. The problem with the baseline is that, in reality,
the draws are not completely random in the sense that the degree of nodes are fixed. Some songs are just more
popular than others, songs are not exchangeable. A more reasonable model is thus a model in which the
"popularity" of the songs are fixed and the edges are drawn at random. In other words, people still listen to
tracks the same number of times on average, but the listening patterns are mixed. Say you listened 10 times to
Justin Bieber, 5 to Coltrane and your neighbour has listened to Britney Spears 13 times and Radiohead 3 times.
It might well have been the same you were the one who had listened to BS and JB. We can still freely swap the
plays around and get the same distribution. With this model, the probability of having a connection between
two nodes is:

$$
P_{ij} \propto k_i k_j
$$

in other words, if we drew the number of co-occurences of two tracks at random we would get something of the
order of \\(k_i k_j\\). This amounts to assuming that people listen to things at random while keeping the
total number of counts per track fixed. This, due to the
popularity of tracks alone! A good way to measure the latent tendency for songs to co-occur is therefore to
write that on average the number of connections between $i$ and $j$ is given by:

$$
N_{ij} \propto \frac{\theta_{ij}\:k_i k_j }{N_{TOT}}
$$

and $\theta_{ij}$ is the quantity that interests us! We can thus write:

$$
\theta_{ij} = \frac{N_{ij} N_TOT}{N_i N_j}
$$

What is interesting is that this quantity varies between $[0, +\infty]$ where the value $1$ has a particular
meaning: if $\theta > 1$ it means that the two songs are *more* connect than if they'd been at random,
otherwise less. You obviously don't want to consider the latter option when making recommendations.

Here is what we knowingly left out:

- Sequences. This would make the recommender personalized.
- The bipartiteness of the user-songs network. Here we're technically recommending to the `average` (not less
  awesome) user.
- Time!
- Contextural information about music, music signal, etc. But I believe this is useful for the last 10 inches
  of the road.
  
  
\begin{equation}
  P(j|i) = P(k|i) * P(k has listened to j) * P(select j)
         = \frac{N_ij}{N_i} * \frac{1}{N_j} * \frac{1}{\sum_{n=1}^{K} 1/N_k}
\end{equation}


Select one person that has listened to i & j at random can be written as:

\begin{equation}
  K(i,j) = \frac{N_ij}{N_i}
\end{equation}

The issue with this is that, if listens were atttributed at random, transition $j$ would be higher just
becasue $j$ is more popular. This is what you might to have, but not ideal in terms of 'surprise'.

As we have currently defined, N_ij = K(i,j) N_i; yet, in a random setting, N_ij \propto N_i N_j. We want a
kernel K' from which we have removed the random part to show more meaningful relationship:

N_ij = K'(i,j) N_j N_i


\begin{equation}
  \sum_i \frac{N_i}{N_TOT} K(i,j) = \frac{N_j}{N_TOT}
\end{equation}

On the other hand if

\begin{equation}
  K(i,j) = \frac{N_ij}{N_j}
\end{equation}

Then 

\begin{equation}
  \sum_i \frac{1}{|S|} K(i,j) = \frac{1}{|S|} \sum_i \frac{N_{ij}}{N_j} = \frac{1}{|S|}
\end{equation}

How does that related with the MCMC explorer?

P(j|i) = \frac{N_ij}{N_i} * \frac{1/N_j}


Good kernel if \sum_j K(i,j) = 1
In this case, K(i,j) = N_{ij} / N_i works great.

But this is not what we want. we do not want to respect the current situation. We want the uniform
distirbution to be invariant i.e. we want to markov chain to produce the uniform distribution as the
invartiant distribution. The rationale for this is that the popularity of $N_j$ is included in N_ij, which
makes popular items to be more recommended, although they might not be the most relevant.


N'_ij = N_ij / N_j

so \sum j \frac{N'}{N_i} = \frac{1/Ni} \sum_j \frac{N_ij}{N_j}

Instead we want K(i,j) such that:

\sum_i \frac{1}{|S|} K(i,j) = \frac{1}{|S|}

K(i,j) = \frac{N_ij}{N_j} / \left( \sum_k \frac{N_ik/N_k} \right)

\sum_i K(i,j) = 1


## Real implementation

Starting from item $i$

1. Choose user at random with probability 1/N_i
2. Choose $k$ with probability \frac{1}{N_k} (1/\sum_n \frac{1}{N_n})

So we can write:

P(j|i) = \frac{1}{N_j} \sum_{p=1}^N_ij \mathcal{N}(p)

If we sum this over $i$
