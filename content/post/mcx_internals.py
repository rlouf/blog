---
title: "Designing MCX's core"
date: 2020-09-25T19:09:52+02:00
draft: false
---

# A tale of two languages

The difficulty of designing a user-friendly API for a Probabilistic Programming
Language stems for the deceiving simplicity of the language we use to represent
graphical models:

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

<<<<<<< HEAD:content/post/introduce_mcx.md
Programming languages do not have a notion of random variable and the ability to reason about them.
=======
The Beta-Binomial model shown in mathematical notations above translates to:
>>>>>>> 426d9e7 (update, add grief):content/post/internals_mcx.md

# MCX models

Here’s how you would express the beta-binomial model in MCX:

```python
@mcx.model
def beta_binomial(beta=2):
  a <~ Beta(1, beta)
  b <~ Binomial(a)
  return b
```

Which is syntactically correct python but would not do anything good without the
`@mcx.model` decorator.


## Behind the scenes

The modeling language was settled early on in the project. The internals have
evolved a little since, and will keep evolving. What happens at the beggining
is always the same: when initializing the model it is sent to a parser that
reads the abstract syntax tree that converts the code into a graph.

The first version consisted in registering transformations and distributions
line by line, with a graph that contains constant nodes, transformation nodes
and distributions. This graph would later be compiled into sampling functions or
the logpdf.

There were three main issues with this:

1. It was impossible to use anything other than numbers and names to initialize
   distribution. For instance, `Normal(np.ones(5), 1)` was forbidden. That was
   because I did not parse the children of the nodes that define random
   variables.
2. There was a lot of custom AST re-writing, which is prone to errors. The less
   I have to change python's code the better it is.
3. When a model was called inside a model, the strategy was to merge recursively
   the imported graphs into the current one. This can be problematic with
   complex models.
   
For instance, here is the original implementation of `RandVar`
```
class RandVar(object):
    def __init__(
        self,
        name: str,
        distribution: Distribution,
        args: Optional[List[Union[int, float]]],
        is_returned: bool,
    ):
        self.name = name
        self.distribution = distribution
        self.args = args

    def to_logpdf(self):
        """Returns the AST corresponding to the expression
        logpdf_{name} = {distribution}.logpdf_sum(*{arg_names})
        """
        return ast.Assign(
            targets=[ast.Name(id=f"logpdf_{self.name}", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution,
                    attr="logpdf_sum",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=self.name, ctx=ast.Load())],
                keywords=[],
            ),
        )

    def to_sampler(self, graph):
        args = [ast.Name(id="rng_key", ctx=ast.Load())]

        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution,
                    attr="forward",
                    ctx=ast.Load(),
                ),
                args=args,
                keywords=[],
            ),
        )
```

As you can see, a `RandVar` node carries its name, the name of the distribution
as a string (e.g. "dist.Normal") and its args. It has two methods `to_logpdf`
and `to_sampler` which are later called by the compiler depending on the
context. Each node implements these methods, for instance for an argument:

```
class Argument(object):
    def __init__(self, name: str, default: Optional[ast.expr] = None):
        self.name = name
        self.default_value = default

    def to_logpdf(self):
        return ast.arg(arg=self.name, annotation=None)

    def to_sampler(self):
        return ast.arg(arg=self.name, annotation=None)
```

There is a lot of AST writing that probably should not be happening. For
instance, `dist.Normal(0, 1)` is something that will never be changed, why can't
I transpose it directly from the original AST? I am not showing code from the
compiler here (see [here](https://github.com/rlouf/mcx/blob/e906ee0d0b6c85269cafc0f6246afe6b15a3e7bb/mcx/compiler/compiler.py)) but it gets even messier).

This did not feel either efficient, nor elegant, nor robust. Time for something
else.

My problems boiled down to two things:
1. My graph representation was not granular enough, which led to problem (1)
   above;
2. I did not store the AST in a format that minimizes rewriting. Hence problem
   (2);
3. I used the compiler to do all the work instead of the graph, which made
   problem (2) worse.
   
I took some inspiration from libraries that build symbolic graph, such as
[Theano]() or [SymJAX](). To simplify, they have three kinds of nodes:

1. Placeholders that represent the inputs to the graph who value is unknown when
   the graph is built. For instance, in the beta-binomial example above the 
   argument `beta` would be represented as a Placeholder.
2. Constants, which represent numerical constants.
3. Ops, which represent transformations from one node to the other. They can be
   names (in which case they represent variables), or not. For instance, if
   I parse `a = 2 * 3 + 4` I will have a first Op that represents the
   multiplication of 2 by 3, and an Op named `a` that represents the sum
   of the previous Op with 4.
   
What is intersting is that the AST can be easily translated into a graph like
this. Take `a = 2 * 3 + 4` which python parses into the following AST:

```
Assign(
  targets=[Name(id='a', ctx=Store())],
  value=BinOp(
    left=BinOp(
          left=Constant(value=2, kind=None),
          op=Mult(),
          right=Constant(value=3, kind=None),
    op=Add(),
    right=Constant(value=4, kind=None))
  )]
```

In MCX, we would first add the two constants $2$ and $3$ the graph as constants,
then create an unnamed Op with a `to_ast` function defined as:

```
def to_ast(left, right):
    return ast.Binop(left, ast.Mult(), right)
```

which is copied from the original ast. When we add this Op to the graph we
specify the constants as its arguments, `graph.node(op, constant2, constant3)`.
This function adds two edges between the constants and this op, and anotates the
order of the arguments.

Then we add an Op named "a" with the following `to_ast` function:

```
def to_ast(left, right):
  return ast.Binop(left, ast.Add(), right)
```

And add to the graph specifying that the previous Op and the constant 4 are
arguments. This is the simplified story. In reality the named Op is encountered
first by the parser. What we do then is to recursively parse its arguments until
we encounter a constant or a name and then add them to the graph backwards.
While this may looks tedious, there are only 9 types of nodes that need to be
handled during the recursion.

Additionally, since we are building a PPL we would like to be able to reason
about random variables and how they are connected to each other. So I add
a `SampleOp` which is mostly syntactic sugar, except it also carries the
distribution object.

Compilation is fairly straightforwward: we need to traverse the graph in 
topological order, and whenever a named Op is encountered compile it
recursively. The only bits of AST that need to be added are the assignment
statements and the function definition statement.

How do we create the logpdf and samplers now? We create a new graph and
manipulate it: add placeholder, change names, call methods on distributions
and then compile this graph. As a result the compiler is kept fairly generic.
Different functions share the sample core manipulations, and are specialized.

We achieved most of what we wanted!
1. There are less chances to do something wrong with the AST;
2. The code is more generic;
3. We can use any valid python expression as parameters of the distribution;

And we can do more:
1. Identify whether there is a direct link between two distributions. This
   is important to identify conjugacies. Collapsing conjugacies is a matter
   of traversing the graph to identify directly connected random variables,
   check if there are conjugate and if so modify the graph.
2. Identify whether a variable is a transformed random variable.




The model is syntactically close to both the mathematical notation and that of a
standard python function. When you write a MCX model such as the one above you
implicitely define 2 distributions. First the joint probability distribution of
the associated graphical model. You can sample from this distribution with the
`mcx.sample_joint(beta_binomial)` function. You can also the log-probability for
a set of values of is random variables with `mcx.log_prob(model)`.

Second, the `return` statement of the function defines the model's `predictive
distribution`. It is usually associated with prior predictive and posterior
predictive sampling.

Like any function, MCX model can be parametrized by passing arguments to the
model. `beta_binomial` takes `beta` as an argument, which can
later be used to parametrize the model.

A MCX model is a generative function that implicitely defines a joint
distribution on its random variables.

which is syntactically correct python, but wouldn't do anything good without the
`@mcx.model` decorator. Notice the `<~` operator, which is not standard python notation. It stands for "random variable" assignment in MCX.

What happens when you call this function? Under the hood MCX parses the content of your model into an intermediate representation, which is a mix between a graphical model (which is the mathematical object your model describes) and an abstract syntax tree (to maintain control flow). Random variables are identified as well as their distributions and the other variables they depend on.

Models are funny objects. First, they are generative functions. If you call `beta_binomial` with a Prngkey it will return a sample from the predictive distribution. They are also distributions and implement `sample` that returns samples from the joint distribution implicitly defined, as well as `logpdf` which returns the logprobabiliy of points in the sample space.


What happens when you call the function is that its
content is being parsed: random variables and distributions are identified as
well as their dependencies. And this is the second important thing: they are
parsed into a directed graph, another representation of the equations above. The
graph is stored in the model (which is actually a class).


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
>>> mcx.sample_predictive(rng_key, evaluate_model, (x_data,), num_samples=100)
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
