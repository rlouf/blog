---
title: "Introducing MCX"
date: 2020-09-25T19:09:52+02:00
draft: false
---

# A simple API

The difficulty of designing a user-friendly API for a Probabilistic Programming
Language stems for the deceiving simplicity of the mathematical representation
of graphical models:

```
a ~ Beta(1, 2)
b ~ Binomial(a)
```

From this representation we need to extract at least two objects: one that
produces samples form the prior distribution, and the multivariate
log-probability density conditioned on the observe data. We have a one-to-many
relationship between the notation and what we need from it in practice, and this
one-to-many relationship does not have related constructs in programming
languages.

Of course we can implement these two objects separately. This is actually what
we do with "elementary" distributions like the Beta, Dirichlet, Multvariate
Normal, etc. distributions. In many PPLs they are implemented as a single object
that implements the `sample` and `logpdf` distribution. This is however tedious
with complex graphical models. But it should not be in the sense that these
models are composed of elementary distibutions. We would like a language that
allows us to write the combination of such elementary distributions and returns
its logpdf and sampling distribution.

Programming language are different: one set of instructions does one thing and
only one. There is thus a tension between the above notations and the
programming language. This is a problem of two languages: how do you express in
a programming language an abstract concept?

PPLs have different ways to resolve this tension: some build a graph, some use
effect handlers, others use a non-standard intepretation of the programming
language (PyMC3's use of the context manager for instance). MCX's stance is much
like Stan's: what if we created our own language, close to the mathematical
expression, that can then be compiled in a logpdf and a sampling function?

Unlike Stan, we would like this language to be embedded in python. To benefit
from its autodiff, acceleration and array manipulation libraries, and the
integration with the rest of the machine learning ecosystem. MCX should thus
be a superset a python: allow every operation python allows, with an added
construct. Having posed the problem in these terms there is only one solution:
defining a new syntax and modify this syntax at runtime using Python's
introspection abilities.


```
@mcx.model
def beta_binomial():
  a <~ Beta(1, 2)
  b <~ Binomial(a)
  return b
```

which is syntactically correct python, but wouldn't do anything good without the
`@mcx.model` decorator. What happens when you call the function is that its
content is being parsed: random variables and distributions are identified as
well as their dependencies. And this is the second important thing: they are
parsed into a directed graph, another representation of the equations above. The
graph is stored in the model (which is actually a class).

Models are distributions, so they implement `sample` and `logpdf` functions.
When you call either of these, two different **compilers** are called and
traverse the graph to build the necessary functions.


# A tale of 4 distributions 

The duality between multivariate distribution and generative function usually
implies the existence of 4 different distribution in PPLs. It is very important
to understand their difference when implementing the API.

Those related to the model interpreted as a multivariate distribution:

- The prior distribution is related to the multivariate distribution
  interpretation. It is the distribution parametrized by the model's
  random variables, given the model's parameters.
- The posterior distribution is the multivariate distribution of the
  model's parameters conditioned on the value of some of the model's
  variables.

And those relatied to the model interpreted as a generative function:

- The prior predictive distribution, which is the distribution of the
  returned value of the model when the random variables' values are 
  drawn from their prior distribution.
- The posterior predictive distribution, which is the distribution of
  the returned value assuming that the random variables' values are 
  drawn from their posterior distribution.
  
In MCX it is assumed that both are predictive distributions of different
objects: the model in the first case, the evaluated model in the second model.
While the model is both a multivariate distribution and a generative function it
can be evaluated. A generative function is only associated with a predictive
distribution.

Writing a model as a function is intuitive: most bayesian models are generative
models. Given input values and parameters they return other values that can be
observed. While MCX models also represent a distribution, in the API we treat
them first as generative functions.

Consider the following linear regression model:

```
@mcx.model
def linear_regression(x, lmba=1.):
    scale <~ Exponential(lmbda)
    coef <~ Normal(np.zeros(x.shape[-1]), 1)
    y = np.dot(x, coef)
    preds <~ Normal(y, scale)
    return preds
```

Calling the generative function should return a different value each time it is
called with a different value of `rng_key`:

```
>>> linear_regression(rng_key, x)
2.3
```

Note the apparition of `rng_key` between the definition and the call here,
necessary because of JAX's pseudo-random number generation system. It can be
cumbersome to specify a different `rng_key` at each call so we can handle the
splitting automatically using:

```
>>> fn = mcx.seed(linear_regression, rng_key)
>>> fn(10)
34.5
>>> fn(10)
52.1
```

`linear_regression` is a regular function so we can use JAX's vmap construct to
obtain a fixed number of samples from the prior predictive distribution.

```
>>> keys = jax.random.split(rng_key, num_samples)
>>> jax.vmap(linear_regression, in_axes=(0, None))(keys, x_data)
```

Again, for convenience, we provide a `sample_predictive` function, which draws
samples from the function's generative distribution.

```
>>> mcx.sample_predictive(linear_regression, (x_data,), num_samples=1000)
```

**Interacting with the distribution**

The generative function implicitly defines a multivariate distribution over the
model's random variables. We include utilities to sample from this distribution.
To sample from the prior distribution, we implicitly assume the evaluator is the
forward sampler and thus:

```
>>> sampler = mcx.sampler(rng_key, linear_regression, (x_data,))
>>> samples = sampler.run(1000)
```

Since forward sampling can be an efficient way to debug a model, we also
introduce a convenient `forward` method to the model:

```
>>> linear_regression.forward(rng_key, x_data)
```

To sample from the posterior distribution we need to specify which variables
we are conditioning the distribution on (the observed variables) and the 
kernel we use to sample from the posterior:

```
>>> sampler = mcx.sampler(
...               rng_key,
...               linear_regression, 
...               (x_data,),
...               {'preds': y_data},
...               HMC(100),
...           )
... sampler.run(1000)
```

**Posterior predictive**

Once the model's posterior distribution has been sampled we can define a new
generative function that is the original function evaluated at the samples from
the posterior distribution.

```
>>> evaluated_model = mcx.evaluate(linear_regression, trace)
```

When sampling from the predictive distribution, instead of drawing a value for
each variable from its prior distribution, we sample one position of the chains
and compute the function's output. Apart from this we can draw samples from
the generative distribution like we would the model:

```
>>> evaluated_model(rng_key, x_data)
>>> seeded = mcx.seed(evaluated_model, rng_key)
>>> mcx.sample_predictive(seeded, (x_data,), num_samples=100)
```

Unlike the original model, however, the evaluated program is not a distribution.


# Models are programs and samplers evaluators

Evaluators are free to modify the graph to their liking. Hamiltonian Monte Carlo
only performs well with random variables that have an unconstrained support. It
may thus want to apply transformations to these variables so it is able to
sample from their posterior.

This allows other possibilities. When training bayesian neural networks, we may
actually want to train the network with gradient descent while the
inputs/outputs are random variables that can be sampled. We can thus imagine
having evaluators that are mere optimizers. In this case it would change the
neural network from a bayesian one to a regular neural network.

When I started MCX in February my main goal was learning: learning Google's
framework JAX and learn how to code a PPL. JAX looked appealing with its
numpy-style API, no fuss jit-compilation, autodiff and vectorization. On the
other hand, I had been using Stan and PyMC3 for years but did not have a good
grasp of how inference was happening under the hood. And I have this default of
not feeling comfortable using tools I do not fully understand.

MCX was by no means intended to be a PPL at the time, more like Colin Carrol's
[MiniMC]() but using JAX instead of Numpy.


Here were the requirements:

1. A simple API. We all love PyMC3's API. It has all what we need, and just what
   we need. Except having to repeat the names of variables.
2. A graph representation. Having a graph representation allows you to do a lot
   of cool things: introspection and manipulation. And we call them *graphical
   models*, after all?
3. Fast inference, for many chains. What's worse than waiting in front of your
   computer for hours to get samples. As for the chain, they are mostly cool
   right now, but I have an idea of how to exploit many, many chains.
4. Modularity.



# 'Graph' as in Graphical model 

The models' graph can be accessed interactively. It can be changed in place. It
is possible to set the value of one node and see how it impacts the others, very
useful to debug without re-writing the whole in scipy!

```
new_graph = simplify_conjugacy(graph)
```

Having a graph is wonderful: it means that you can symbolically manipulate your
model. You can detect conjugacies and using conjugate distibution to optimize
sampling, reparametrization is trivial to do, etc. Manipulating the graph is
pretty much akin to manipulating the mathematical object.


````
                                       +----> logpdf
  @mcx.model                           |
  def my_model(X):   ----->   Graph  -------> ....
      .....                            |
      return y                         +----> forward_sampler 

````

All this happens in *pure python*, there is no framework involved. We do use
NetworkX to build and manipulate graphs for convenience, but could do without.

Currently the graph we compile is a static graph. It only contains the random
variables and transformation. As such it can only handle a fixed number of
random variables. This, however, is a strong

The advantage of compiling pure python function is that it nicely decouples the
modeling language from inference. Any inference library that accepts python
functions (with jax constructs) could use the functions used by the DSL. So far
the entire code only relies on functions in JAX that are present in numpy/scipy.
So you could very well consider this as a numpy/scipy function. And if you were
introduce JAX-specific constructs such as control flow, you could still specify
a different compiler for each backend since the graph representation is
framework-agnostic. Hell, you could even write, without too much effort, an
edward2, pymc3 or pyro compiler!

```
Example with control flow and different
```

Is it crazy to do AST manipulation? It might be harder to do it right than in
language with a full-fledged macro system such as, say, Julia or Lisp, but done
correctly it actually gives us nice benefits: a nice API with a powerful
intermediate representation. Corner cases can also be tested as it is possible
to output the code of the logpdfs from the model.

```
Example model.source_logpdf
```


# Inference

I'll never repeat enough: the modeling language and the inference module are
completely separate. But they need  

The philosophy is that inference in traditional PPLs can be divided according to
three different levels of abstraction:

1. The building blocks (or routines) of the algorithms: integrators, metrics, proposals, ...
   which do only one thing and do it well.
2. Programs like the HMC algorithm are a particular assembly of these building
   blocks. They form a transition kernel.
3. Runtimes, that tie the data, the model and the kernel together and then make
   the chains move forward following an execution plan.
   
```   
Runtime (Batch sampler)
-------------------------------------------------
Programs (HMC)
-------------------------------------------------
Routines (velocity Verlet, dynamic proposal, etc.)
-------------------------------------------------
```

Most users will interact with the pre-defined programs (HMC or NUTS with the
Stan warmup) and runtimes. But it is possible to create custom inference, it can
be as simple as overriding HMC's warmup by subclassing its class, or as
complicated as implementing your own transition kernel using the available
blocks or blocks you have programmed.

MCX comes with sane defaults (runtimes and pre-defined programs), but has many
trap doors that allow you to tinker with the lower level.

### Carving MCMC at its joints

In order to provide composable inference you need to tear existing algorithms
apart, extract the specific from the general until you have independent pieces
that work well together. This has two advantages:

1. The biggest is development time. It is not rare to see a lot of duplicated
   code in libraries that implement HMC and NUTS. In MCX it is "just" a matter
   of creating program that replaces the HMC proposal with the NUTS proposal.
   You can even add empirical HMC by replacing the proposal. Implementing
   Riemannian HMC is just a matter of switching the metric and integrators. You
   only need to write the parts that are specific to the algorithms.
2. It allows advanced users to take all these parts and create a program you
   would have never thought of.

## Give me a million chains
