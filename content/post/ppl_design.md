---
title: "PPLs that scale"
author: "Remi Louf"
date: 2019-12-25T09:24:11+01:00
draft: true
---

## Bayesian computations that scale?

Probabilistic programming, and its python incarnations, have made tremendous progress. While Machine Learning has always been content with the existing constructs of the language, probabilistic programming has struggled to push the boundaries of the programming language to get something as simple as a *function* to represent a model. But at last, we are getting there.

What do we need?

1. The ability to write probabilistic programs as procedures, or at least as functions. These should be composable to create complex algorithms from simpler routines. This is the basics of machine learning. But it’s not ad obvious as it seems with probabilistic programming.
2. Inference built on a framework that leverages modern hardware, at the very least parallel computations.
3. Inference algorithms that scale. First that leverage modern hardware and its specificities.
4. Principled algorithms that can handle out-of-memory datasets;
5. Finally, a Bayesian promise that remains unfulfilled: the possibility to update one’s knowledge with new information.

### Why would you want to make them scale?

Aren’t Bayesian models best fitted to problems with small models, small data? True, they tremendously outperform "classical" methods in these areas. But

- As Gelman rightly points out, and I think most practitioners can relate to that, big data always eventually becomes small due to the specific questions we’re asking;
- Prior knowledge is great. Classical methods make assumptions that are often unstated. Bayesians have to be more honest about them.
- Classical methods do not have a framework to make decisions, while it is embedded in Bayesian methods. The purpose of analysis is to make a decisions, in the end.
- Bayesian updating.
- More fine grained control over the specifics of the model.



\\[ P(\theta|\mathcal{D}_1) = \frac{P(\mathcal{D}_1|\theta)
P(\theta)}{P(\mathcal{D}_1} \propto P(\mathcal{D}_1|\theta) \\]

Much of the practical & theoretical debate has been focusing on the choice of
priors, which for me is a *non-debate*. Traditional ML does have priors, but
does not state them explicitly. When the results are not satisfactory, they do
use priors, without stating it. They lie in the application of the probabilistic
program, not the program itself.

```math
P(\theta|\mathcal{D}_1, \mathcal{D}_2) = P(\mathcal{D}_2|\theta) P(\theta|\mathcal{D}_1)
```

```python
p = ProbabilisticProgram()
p.observe(data)
p.query(fn_query)
```

with a sequence of data:

```python
p = ProbabilisticProgram()
p.observe(data1)
p.query(fn_query)
p.observe(data2)
p.query(fn_query)
```

But what we currently do is:

```python
p = ProbabilisticProgram()
p.observe(data1)
p.query(fn_query)
p.observe(data1 + data2)
```

# SMCMC to the rescue?

- Sampling should be iterative by default; We should be able to log the 
  progress deep-learning style.
- Ability to modify the graph in place to perform re-parametrization
- Update the observed nodes: there can be less, more, etc.
- Composite inference -> discrete / continuous handling (or continuous version
  of the distributions)
- Graph coloring / ability to identify conditionally independant variable
- Bayesian updating that reduces to HMC when prior distribution
- Minibatch sampling that reduces to HMC for full batch
- Prior PS
- Posterior PS
- BNN sampling
- Prefetching to accelerate
- How do we do this on GPU?
- Ability to build BNNs
- Support variational inference
- Ability to observe data

Framework = JAX because of neural tangent and its performance/simplicity + I
like the functional style.
