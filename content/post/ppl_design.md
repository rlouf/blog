---
title: "PPLs that scale"
author: "Remi Louf"
date: 2019-12-25T09:24:11+01:00
draft: true
---

Let me state two things:

1. I am a Bayesian at heart. I have been using Bayesian methods professionally
   for 5 years now;
2. I do not claim to be an expert in the field, and there are a lot of people
   smarter than me who have been thinking about these problems for a lot longer
   than I have.

A traditional program may look like this:

```python
```


# Where there's improvement

1. Make use of multi-core computations / GPU computation
2. Bayesian knowledge updating
3. Mini-batch updates


```math
P(\theta|\mathcal{D}_1) = \frac{\P(\mathcal{D}_1|\theta)
P(\theta)}{P(\mathcal{D}_1} \propto P(\mathcal{D}_1|\theta)
```

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
