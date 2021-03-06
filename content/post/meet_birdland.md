---
title: "Introducing Birdland"
date: 2018-06-04T22:11:10+01:00
draft: true
---

This post is a long overdue post on a recommendation algorithm that I developed
and implemented with Etienne Duschene in my previous job:
[Birdland](https://github.com/rlouf/birdland) (checkout the code on Github). I
will explain the meat of the algorithm and what you can do with it. And most
importantly, how we got there.

This is a long post, so I added a little table of contents.

## Content

- [Bye implicitMF]()
- [Requirements]()
- [Birdland]()

## You have 2 weeks to come up with something

I was asked a little less than 2 years ago to design a music recommendation
algorithm for the application I was working on. The application was a social
network built around music; users had a profile where they curated artists and
song, and could share music with their friends. In app speech, think instagram
meets LastFM; the app did pretty well until the adventure abruptly ended, but
that's another story.

At the time, Spotify had [a full battery]() of recommendation algorithms in
production and it was obvious we could not compete. What we had that Spotify did
not, however, was a social graph. We had to do social recommendation. Well, in
fact we had promised social recommendation to a funding agency and got money to
do it, so we REALLY had to do it.

For the past 10 year or so, [implicitMF]() had been the go-to recommendation
method for implicit feedback contexts---and am sure still is. In ways it worked,
and it has amply contributed to what the world of recommendation is like today.
If, like me, you have as Spotify account you have probably been served some of
its results and anjoyed it (shoutout to [Erik Berhnarhdsson], [Chris Johnson]
and the talented people we don't hear about who created this fantastic
product!).

However, despite its past success, I quickly decided it was not going to cut it.
There are two things that I try to steer away from when I work on a machine
learning problem: complexity and conceptual smell. Unfortunately, ImplicitMF is
both dubious theoretically speaking, and harder to implement that it seems.

### Why implicitMF sucks

#### Conceptual smell

Conceptually, in the context of music recommendation, implicitMF sucks. I've
said it.

The MF part in `implicit MF` stands for implicit matrix factorization. Roughly
speaking, it starts with a matrix $R$ (user x song) where $R_{i,j}$ is the
number of times the user $i$ has played the song $j$. The $R_{i,j}$s constitue
\textit{implicit} feedback; historically recommendation started with rated
items. Here we don't have ratings, but data about interactions.

The method is essentially a dimensionality reduction method. For each user $i$
and song $j$ you want to find vectors $u_i$ and $v_j$ of dimension $K$ such that
the scalar product $u_i v_j$ is as close as $R_{i,j}$ as possible. I don't have
an issue with that. The problem is with how we define close: implicit MF
basically says the 2 are close when following $L_2$ norm is minimum:

$$L = \sum_{i,j} \left(R_{i,j} - u_i v_j\right)^2$$

Plus a regularization term. This can be solved approximately with [alternating
least square](). But let's take a step back here and think about the
signification of the $L_2$ norm. It may seem innocent to most, but it makes a 
strong assumption about the distribution of plays.

> Implicit MF makes the implicit (!) assumption that the number of plays
> follows a normal distribution.

Indeed, if we write this:

$$
r_{i,j} \sim Normal(u_i v_j)
$$

Then finding the $\left\{u_i\right\}_{i=1..N}$ and $\left\{v_j\right\}_{j=1..S}$
that maximize the likelihood of the data amounts to... minimizing the $L_2$
norm!

It's bad. Think about it:

1. The normal distribution's support is the real line; it can give negative
   number. And unless you're feeling lucky or hacky, you may get negative
   counts;
2. If you look the $r_{i,j}$ over all users, getting the number of plays per
   user for each song you also get a Gaussian. Trust me on this, but the
   distribution of the number of plays does not look like a Gaussian at all:
   there is no "typical" number of plays that all values lie around. Actually, 
   there are many users who do not listen to most songs.

The distribution looks more like a Poisson distirbution, actually. If this is
the only thing that bothers you, there is a great paper on [Poisson
factorization] that on top of using an underlying Poisson distirbution instead
of normal.

But as we will see, there is another problem with matrix factorization: the
hidden engineering costs.

#### Hidden costs

The result of implicit Matrix Factorization is two matrices song-vector and
user-song; it is an embedding of songs and users in a K-dimensional space. But
this embedding is the first part of the process. What the tutorials you read on
the web usually don't tell you is that once your implicit MF implemented and
connected to the database you need to:

- Store these vectors in a database and recompute these vectors in batch.
- Given a user and its past plays, recommend new songs.

Without getting into details, you already see that what seemed to be a simple
recommendation service actually ends up being a 3 services ordeal: a server to
compute the vectors, a database to store these vectors, and a server to serve
recommendations based on user data. Not that simple, is it?

What is also omitted is that the recommendation part of the algorithm is
complicated, way complicated than the factorization part: the N-nearest neibour
problem.

The idea is to find the "song" vector that are the closest to your "user"
vector. For each user there are N such posibility. In real settings, this simply
does not scale. There are very clever approximation method that make the
computation scalable, but their details are much harder than implicitMF's.

See, had you started to work on implicitMF and went as far as factorizing the
plays matrix, you would have ended up in front of a much harder problem. I have
seen a lot of data scientists falling into the matrix factorization trap, and
taking their whole engineering team with them. Don't be that data scientist.

Birdland was born in a particular context. At the time I was working as Chief
Scientist (whatever that means) at a small Startup that built a social app
around music. To be completely honest, the goal of this project was more to show
off our capabilities (and spend funding we'd earned to develop it) rather than
answering a real need. Actually, the first version of "recommendation" was
simply to recomend to users songs they listened to in the past. This is a
great baseline for A/B testing, by the way.

And we had 3 weeks, and no engineering resources allocated. Needless to say, we
had to ditch matrix factorization early on.


## Here comes Birdland

So we sat down, and reformulated the problem as a question:

> Given what a person has listened to in the past, who she interacts with,
> what is she going to do next?

Now, this sounds a lot like a stochastic process. So we model the recommendation
process as a stochastic process over the global set of song: given the songs
that someone has listened to in the past, we have to find a process them to
songs that they haven't listened to but might like.

Building such a kernel is insanely complicated, it presupposes a model of the
user's behavior, understand how songs are related, and playing history. The best
thing we can do is to learn this kernel from the behavior of users themselves.


### A naive approach


### A network perspective

Although this behaviour only holds in the infinite time limit, and
recommendations would initially be based on the local neighbourhood, this can be
considered as ``bad'' recommendation. Can we do better? 

Yes, by switching to a network representation!

Let us consider the network defined by $A_{ij}$ with nodes $\{1, \dots,
S\right\} of degrees $\left\{k_1, \dots, k_S\right\}$, the number of times
they have each been listened to.

When no self loops are allowed (which is the case here), the random baseline for
such network is a network where edges are added at random under the constraints
that degrees need to stay the same. It can be shown that his model assigns to each edge the
probability:

$$
p_{ij}^{rand} = \frac{k_i k_j}{\sum_n k_n}
$$

On average, the number of edges in our random model is given by

$$
N_{ij}^{rand} = k_i k_j \frac{N}{\sum_n k_n}
$$

What is interesting for recommendations is the *deviation* from this random
model. Assuming the existence of a latent parameter $\theta_{ij}$ that
characterizes how "special" the connection between $i$ and $j$ is, we can write:

$$
p_{ij}^{rand} = \the_{ij} \frac{k_i k_j}{\sum_n k_n}
$$

And we can compute the latent "attraction" factor

$$
\theta_{ij} \propto \frac{N_{ij}}{k_i k_j} \left(\frac{\sum_n k_n}{N}\right)
$$

If $\theta_{ij} < 1$, songs are less connected that they would have been had the
connections been drawn randomly; they probably should not be recommended one
after another. If $\theta_{ij}$ > 1$ they are more connected than they would
have been if people were playing things randomly. The connection between these
two songs is probably meaningful, irrespective of the item's relative
popularity.


### Issues with this naive approach

Beyond the obvious criticisms that all recommendation schemes face, we had very
good reason to leave this method aside:

1. Projecting the user-songs bipartite graph onto the songs leads to an
   intractable transition matrix. In typical settings we have $10^7$ songs; ew
   would need to compute and keep track of a matrix of size $10^{14}$. The
   bipartite graph, on the other hand, only requires storing $N_{users}\;
   \overline{\ell}$ elements where $\overline{\ell}$ is the average number of
   distinct songs listened to by a user.
2. The information contained in the user-song graph is partially washed out by
   the projection onto the items. Yet, it may contain interesting information

For these reasons, we turned to a simpler---yet surprisingly
powerful---approach: random walks on the bipartite graph.


### Recommendations using random walks.

For those of you who are old enough to have known LastFM, you probably used the
method I did when looking for new music to listen to. What I usually did was
finding the profile of people who had roughly similar tastes and explore they
history to find new things to listened to. Sometimes work, sometimes did not.
But still, it was pretty good. 

For some of you who had friends, you probably ask for recommendations to people
you know have similar tastes to you, not to the others.

Both are local explorations of a bipartite graph. This is tedious work, but
imagine if you could do that a million times faster. Wouldn't you get
interesting results? Well, let's figure it out.

#### Simple procedure

If I formalize and simplify the motivating examples I get the following
procedure:

1. Pick a song $i$ uniformly at random in my listen history;
2. Find a user $\mu$ who has listen to this song too;
3. Pick a song $j$ that this user has listened to uniformly at random and store
   to it.

The process is overly simplified: all songs that I listen to are equal in my
eyes. Some I like more and listen to heavily, some I like less. We'll see more
about that below.

We can write the probability $p_{ij}$ to pick the song $j$ starting from $i$ and
going through $\mu$ under this procedure:

$$
p_{ij}^{\mu} = \frac{1}{N_i} \frac{1}{k_{\mu}}
$$

where $N_i$ is the number of people who have listened to $i$ and $k_{\mu}$ the
number of songs the user $\mu$ has listened to. $p_{ij}^{\mu}$ is a transition
kernel. Indeed:

$$
\sum_{\mu, j} p_{ij}^{\mu} = \frac{1}{N_i} \sum_{\mu, j} \frac{1}{k_{\mu}} = 1
$$

The probability measure $\pi_i = N_i / N$ defined above is invariant under this
kernel:

$$
\sum_i \frac{N_i}{N} p_{ij}^{\mu} = \frac{1}{N} \sum_{i,\mu} \frac{1}{k_{\mu}} =
\frac{N_j}{N}
$$

Indeed, for each song $j$ the user $\mu$ has interacted with, all the other
items are included in the dum over $i$ so that the sum over $i$ and $\mu$ can be
reduced to a sum of $j$ times 1.

Don't be thrown off by these technical details and calculations: this algorithm
is simpler and easier to interpret than the naive one above, let alone implicit
matrix factorization.

### The long tail problem

[Explain the long-tail problem]

An issue that remains is that we leave the popularity distribution of songs
invariant under this transition kernel. The nice thing about this framework is
that it can be modified very easily. Let us imagine a procedure that could
diminish the "long-tail" problem:


1. Pick a song $i$ uniformly at random from your listening history;
2. Pick a user $\mu$ who has listened to this song uniformly at random;
3. Pick a song $j$ this user has listened to with probability inversely
   proportional to this item popularity.

The probability to walk from $i$ to $j$ via $\mu$ now reads:

$$
p_{ij}^\mu = \frac{\mathcal{N}_\mu}{N_i N_j}
$$

where 

$$
\mathcal{N} = \frac{1}{\sum_{n=1}^{N_\mu} 1 / N_j}
$$

We now show that the uniform probability measure is left invariant under this
kernel:

$$
\sum_i \frac{1}{N} p_{ij}^{\mu} = \frac{1}{N N_j} \sum_{\mu, j}
\frac{\mathcak{N}_\mu}{N_i} = \frac{1/N}
$$

In layman speak, this means that, following this procedure, an infinitely long
random walk would go through each song the same number of times, disrepective of
its original popularity. **In other words, this algorithm addresses the
long-tail problem, or at least does not make it worse. This is intersting and,
to our knowledge, unique.**

Now, whether this is a good thing is another debate.

#### Other variants

##### Social recommendation

You can customize this algorithm as you wish. In our case, the goal was to get 
a recommendation algorithm that could be used for social recommendation. Easy:

1. Pick a song $i$ uniformly at random from your listening history;
2. Pick a user $\mu$ with a probability $\delta$ that depends on how close
   you are in the network.
3. Pick a song $j$ this user has listened to with probability inversely
   proportional to this item popularity.

In the limit $\delta = 1/N$ we recover the previous case. In the extreme limit
$\delta = 1/f$ where f is your number of friend, 0 otherwise we are in a
situation where you get most of your recommendations from your friends.

If you want to be fancy, you can say that you projected the multi-layer graph
user-user-song onto the user-song bipartite graph. Knowing that you can do even
fancier stuff by walking the social graph at the same time.

##### Not all songs are created equal

You don't listen to songs the same number of times, so it is pretty stupid to
pick songs uniformly at random in your history. Simple, just pick it at random
with a probability proportional to the number of times you have listened to this
song.

### Recommending items from a list of songs

To recommend songs to users you would perform many, many random walks starting
from that same user. Those walks will likely start from many different songs in
the listening history, and it is also likely that some songs will be traversed
by several walks. How do we combine these walks to provide recommendations.

There are several schemes you can think of:

*Most visited first:* We count the number of times each songs has been visited
and sort recommendations accordingly. This performs poorly in practice as you
end up recommending popular songs.

*Consensus:* Recommend the songs that have been listening to the largest number
of users first. This is equivalent to the first scheme.

*Trust:* Weigh each user by the number of times they have been visited. Weight
each item by the weight of the user who recommend it it. Sum the item weights
and sort them accordingly. This performs really well in practice, and what we
ended up doing.

*Relevance:* For each item visited, count the number of different items in my
listening history that lead to it. Sort according to this number. This performs
very well in practice too.


### Birdland in practice

We implemented Birdland in Go both for speed (the algorithm had to run online),
and because the backend was in Go. 

#### Recommending from walks.

1. The algorithm is randomized. You have guaranteed novelty every time you
   refresh the page.
2. You can precompute many, many things when you load the data; as a result
   inference is really fast.
