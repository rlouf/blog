---
title: "Massively parallel MCMC with JAX"
date: 2019-01-09T13:19:11+01:00
draft: false
---

## TL;DR

JAX blows everyone out of the water, up to a factor of 20x in extreme cases
(1,000 samples with 1,000,000 chains). Numpy wins in the small number of
samples, small number of chains regime due to JAX's JIT compilation.  TFP with
XLA compilation shines in the small number of chains, large number of samples
regime. I must be doing something wrong with Tensorflow Probability as the
numbers I get otherwise are very underwhelming.

## Introduction

I recently got very excited about the possibility to generate thousands,
millions of sampling chains with Monte Carlo algorithms. I vaguely followed
several discussions around the topic 

I have seen here and there discussion about vectorized sampling algorithms that
were able to generate hundreds, thousands of chains in one pass. 

Colin Caroll posted an interesting [blog
post](https://colindcarroll.com/2019/08/18/very-parallel-mcmc-sampling/) that
uses numpy and a vectorized version of the random walk metropolis-hastings
(RWMH) algorithm to generate a large number of samples.

Around the same time I stumbled upon [JAX](http://github.com/jax/jax).  Jax
offers Just-in-Time compilation using [XLA](), inherits from autograd's autodiff
functionality. It favours a functional style of programming which makes sense in
a world where we are combining mathematical functions. I fell in love (there is
some place left for stickers on my laptop, btw!).

So, for future reference I decided to benchmark different backends:

- Numpy
- Jax
- Tensorflow Probability (TFP)
- Tensorflow Probability with XLA compilation

I am sampling an arbitrary Gaussian mixture with 4 components. Using Numpy:

```python
import numpy as np
from scipy.stats import norm

def mixture_logpdf(x):
    """Log probability distribution function of a gaussian mixture model.

    x: np.ndarray (4, n_chains)
        Position at which to evaluate the probability density function.
    """
    log_probs = np.array(
        [
            norm(-2.0, 1.2).logpdf(x[0]),
            norm(0, 1).logpdf(x[1]),
            norm(3.2, 5).logpdf(x[2]),
            norm(2.5, 2.8).logpdf(x[3]),
        ]
    )
    weights = np.repeat(np.array([[0.2, 0.3, 0.1, 0.4]]).T, x.shape[1], axis=1)
    return -logsumexp(np.log(weights) - log_probs, axis=0)
```

The code for all 3 frameworks is available in this [gist]().

## Notes about benchmarking

Before giving the results, a few words of caution:

1. The reported times are the average of 10 runs on my laptop, with nothing
   other than the terminal open. For all but the post-compilation JAX runs,
   the times were measure by the `hyperfine` command line tool.
2. My code is probably not optimal, especially for Numpy and TFP. I would
   appreciate tips to make the codes faster.
3. The experiments are performed on CPU. Since JAX and TFP can run computations
   on GPU, I would expect TFP to beat Numpy in the large number of samples,
   number of shapes regime.
4. For Numpy and JAX the sampler is a generator and the samples are not kept in
   memory. This is not the case for TFP, thus (1) the computer runs out of
   memory during big experiment (2) this might affect the performance.


So far, running multiple chains at once was reserved to performing posterior
checks on the convergence of algorithms, or reducing the variance (not the
biais) of the results of Monte Carlo sampling. It was traditionally achieved by
running one chain per thread on a multithreaded machine, in Python using joblib
or a custom backend. It did the job.

But then, multiple people started talking about vectorized sampling algorithms
that were able to generate hundreds, thousands of chains in one pass. It turns
out that I have recently gotten ridiculously obsessed with Sequential Markov
Chain Monte Carlo, and part of the requirement is to be able to sample many
chains at once. You will probably read a lot about SMCMC here in the near
future.

I am not going to repeat what Colin said about why we would like to have many,
many chains, so you can read the reasons here.

The idea was that if you want to build a reasonable PPL around this idea of 
vectorized sampling, you do need a framework that performs autodiff. Tensorflow
Probability and Pyro allow you to do that already, but JAX seemed interesting to
me for several reasons:

- It is in most cases a drop-in replacement for numpy, and numpy is known for its
  simple, clean interface (in most cases, don't jump on me here);
- Autodiff is plain simple;
- Its forward differentiation mode allows to easily compute higher-order
  derivatives;
- It performs JIT compilation, accelerating your code even on CPU;
- Using GPU is straightforward;
- Matter of taste, but I really like its functional style.

Colin's MiniMC is an exercise in style, the simplest and most readable
implementation of HMC I have seen. Probably thanks to the numpy backend.
My numpy is implementation is an iteration upon his

## Setup and results

### Numpy

We implement the Random Walk Metropolis algorithm in the following,
uncontroversial way:

```python
import numpy as np

def rw_metropolis_sampler(logpdf, initial_position):
    """Generate samples using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    logpdf: function
      Returns the log-probability of the model given a position.
    initial_position: np.ndarray, shape (n_dims, n_chains)
      The initial position for each chain.

    Yields
    ------
    np.ndarray, shape (,n_chains)
      The next sample generated by the random walk metropolis algorithm.
    """
    position = initial_position
    log_prob = logpdf(initial_position)
    yield position

    while True:
        move_proposals = np.random.normal(0, 0.1, size=initial_position.shape)
        proposal = position + move_proposals
        proposal_log_prob = logpdf(proposal)

        log_unif = np.log(np.random.rand(initial_position.shape[0], initial_position.shape[1]))
        accept = log_unif < proposal_log_prob - log_prob

        position = np.where(accept, proposal, position)
        log_prob = np.where(accept, proposal_log_prob, log_prob)
        yield position
```

### JAX

```python
from functools import partial

import jax
import jax.numpy as np

@partial(jax.jit, static_argnums=(0, 1))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob):
    move_proposals = jax.random.normal(rng_key, shape=position.shape) * 0.1
    proposal = position + move_proposals
    proposal_log_prob = logpdf(proposal)

    log_unif = np.log(jax.random.uniform(rng_key, shape=position.shape))
    accept = log_unif < proposal_log_prob - log_prob

    position = np.where(accept, proposal, position)
    log_prob = np.where(accept, proposal_log_prob, log_prob)
    return position, log_prob


def rw_metropolis_sampler(rng_key, logpdf, initial_position):
    """Vectorized Metropolis-Hastings.
    """
    position = initial_position
    log_prob = logpdf(initial_position)
    yield position

    while True:
        position, log_prob = rw_metropolis_kernel(rng_key, logpdf, position, log_prob)
        yield position
```

If you are familiar with Numpy, the syntax of the version using JAX should feel
very familiar to you. There a couple of things to note:

1. `jax.numpy` acts as a drop-in replacement to numpy. For codes only involving
   array operations, replacing `import numpy as np` by `import jax.numpy as np`
   should already give you performance benefits.
2. JAX handle random number generation differently from other python package,
   for [very good reasons](). Every distribution takes a RNG key as an input.
3. We extracted the kernel from the sampler because Jax cannot compile
   generators (or can it?). So we extract and JIT the function that does all the
   heavy lifting: `rw_metropolis_kernel`.
4. We need to help Jax's compiler a little bit by indicating which arguments are
   not going to change when the function is run several times:
   `@partial(jax.jit, argnums=(0, 1))`. This is compulsory if you pass a
   function as an argument, and can enable further compile-time optimizations.


### Tensorflow Probability

For TFP we use the Random Walk Metropolis algorithm implemented in the library:

```python
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def run_raw_metropolis(n_dims, n_samples, n_chains, target):
    dtype = np.float32
    samples, _ = tfp.mcmc.sample_chain(
        num_results=n_samples,
        current_state=np.zeros((n_chains, n_dims), dtype=dtype),
        kernel=tfp.mcmc.RandomWalkMetropolis(target.log_prob, seed=42),
        num_burnin_steps=0,
        parallel_iterations=8,
    )

    return samples

run_mcm = partial(run_tfp_mcmc, n_dims, n_samples, n_chains, target)

## Without XLA
run_mcm()

## With XLA compilation
tf.xla.experimental.compile(run_mcm)
```

Numpy was a serious contender of Jax for RW Metropolis. However, there is one
place where it cannot compete, situations where the gradient of the function is
needed. Hamiltonian Monte Carlo!

TFP must be doing some sort of step-size adaptation which may affect the 
performance in terms on time (but not sample quality).

Better measure is Effective Sample / s but here we are just comaring raw
performance. The numbers may shift, but the comparisons should still hold.
