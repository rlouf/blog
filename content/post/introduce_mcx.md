---
title: "Introducing MCX"
date: 2020-09-25T19:09:52+02:00
draft: false
---

# A simple API

Designing a user-friendly API for a Probabilistic Programming
Language (PPLs) is cursed by the deceiving simplicity of the mathematical representation
of graphical models:

```
a ~ Beta(1, 2)
b ~ Binomial(a)
```

Why should it have to be harder than this? 


Well. First, to manipulate these models in a useful way we need to extract at least two different representations of this object: one that
draws samples form the prior joint distribution, and the
log-probability density function. We have a one-to-many
relationship between the notation and what we need from it in practice, and this
one-to-many relationship is not trivial to implement in programming
languages.

Of course we can implement these two objects separately. This is simple to do with "elementary" distributions like the Beta, Dirichlet, Multivariate
Normal, etc. distributions. In many PPLs they are implemented as a single object
that implements the `sample` and `logpdf` functions. So you can implement distributions as classes so that when you write:

```python
x = Beta(1., 1.)
```

You can draw samples from the distribution

`x.sample()`

Or compute the log-probability as a point `a`:

`x.logpdf(a)`

The notation mirrors that of `x ~ Beta(1., 1.)`. But it becomes tedious
with complex graphical models.

Of course you could implement your model as a class with a `logpdf` and `sample` method, but it wouldn’t mirror the mathematical expression. Take the beta-binomial model above

`a ~ Beta(1, 2)
b ~ Binomial(a)`

Shouldn’t we be able to express the fact that the model is the combination of two probability distribution? We would like a language that
allows us to combine such elementary distributions, just like above, into an object that gives us
its logpdf and sampling distribution.

But programming languages are different: one set of instructions does one thing and
only one. I can probably write a programme like the one above that would give me predictive samples. But then I wouldn’t have the logpdf.

There is thus a tension between the mathematical notations and the
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

Programming languages do not have a notion of random variable and the ability to reason about them.

# MCX models

Here’s how you would express the beta-binomial model in MCX:

```python
@mcx.model
def beta_binomial():
  a <~ Beta(1, 2)
  b <~ Binomial(a)
  return b
```

which is syntactically correct python, but wouldn't do anything good without the
`@mcx.model` decorator. Notice the `<~` operator, which is not standard python notation. It stands for "random variable" assignment in MCX.

What happens when you call this function? Under the hood MCX parses the content of your model into an intermediate representation, which is a mix between a graphical model (which is the mathematical object your model describes) and an abstract syntax tree (to maintain control flow). Random variables are identified as well as their distributions and the other variables they depend on.

Models are funny objects. First, they are generative functions. If you call `beta_binomial` with a Prngkey it will return a sample from the predictive distribution. They are also distributions and implement `sample` that returns samples from the joint distribution implicitly defined, as well as `logpdf` which returns the logprobabiliy of points in the sample space.


What happens when you call the function is that its
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

- The prior distribution is related to the joint distribution interpretation. It is the distribution parametrized by the model's random variables, given the model's parameters.
- The posterior distribution is the multivariate distribution of the model's parameters conditioned on the value of some of the model's variables.

And those related to the model interpreted as a generative function:

- The prior predictive distribution, which is the distribution of the returned value of the model when the random variables' values are  drawn from their prior distribution.
- The posterior predictive distribution, which is the distribution of the returned value assuming that the random variables' values are drawn from their posterior distribution.
  
In MCX it is assumed that both are predictive distributions of different objects: the model in the first case, the evaluated model in the second model. To get prior predictive distribution:

```python
samples = mcx.sample_predictive(rng_key, model, args)
```


While the model is both a multivariate distribution and a generative function it
can be evaluated. A generative function is only associated with a predictive
distribution.

```python
model = mcx.evaluate(modeling, trace)
samples = mcx.sample_predictive(rng_key, model, args)
```

Writing a model as a function is intuitive: most bayesian models are generative models. Given input values and parameters they return other values that can be
observed. While MCX models also represent a distribution, in the API we treat them first as generative functions.

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


# 'Graph' as in Graphical model 

The models' graph can be accessed interactively. It can be changed in place. It is possible to set the value of one node and see how it impacts the others, very useful to debug without re-writing the whole in scipy!

```
new_graph = simplify_conjugacy(graph)
```

Having a graph is wonderful: it means that you can symbolically manipulate your model. You can detect conjugacies and using conjugate distibution to optimize
sampling, reparametrization is trivial to do, etc. Manipulating the graph is pretty much akin to manipulating the mathematical object.


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