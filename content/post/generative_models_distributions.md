---
title: "Of generative models and distributions"
date: 2020-01-29T09:45:48+01:00
draft: true
---

I am currently developping yet another probabilistic programming library in
Python, based on source code generation. This is not a new idea, and it is
basically how Stan works: models are written in a DSL and interpreted by the C++
library. Most other libraries rely on a non-standard intepretation of the
language and while they manage to bend python well-enough to build complete
probabilistic programming libraries, there is a sense of complication that
probably should not exist:

- A probabilistic language is akin to a regular language, with an additional
  construct that is the idea of a random variable. This changes the behaviour
  of the language at runtime, behaviour which can be encoded in the intepreter
  or compiler.
- Generative models are parametrizable functions: from data $\left\{X\right\}$ and
  parameters $\left(\alpha_i\right\)$ they can generate values.
- They are not your regular kind of functions; as in standard Machine Learning,
  they can be learned from data.

One of the ideas of `mcx` (pronounce "mix") is to modify the python syntax only
the slightlest to be able to express a wide range of probabilistic models. A
`mcx` program should *feel* like regular python, with and additional construct.

But to get things straights, here are a couple of questions:

- Are generative models distributions?
- If not, what are they?

Let us take the simplified example of the normal distribution.

```python
class Normal(Distribution):
  parameters = {'mu': constraints.real, 'sigma': constraints.positive}
  domain = constraints.real

  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma
  
  def sample(self, num_samples: int) -> float:
    return self.mu + random.norm() * self.sigma
  
  @limit_to_domain
  def logpdf(self, num_samples: int) -> float:
    return stats.norm.logpdf(x, self.mu, self.sigma)
```

Things to note:

- A distribution ascribes a measure on the random variable's domain. The measure
  is given by `logpdf`.
- We can generate values at random that follow this measure using `sample`
- We reified the measure to a class that also holds the sampling function and 
  properties such as the domain of the random variable, and the constraints
  on the parameters' values.

So that when I write `x ~ Normal(0, 1)` what I am really writing is "x is a
random variable and the probability of its values are given by a Normal
distribution parametrized by a zero mean and unit variance". So, really, the `~`
symbol means `has` and not `is`. What `x` is, is potentially any number on its
domain. It feels rather strange that we are ascribing the domain the the
probability density function, but hey. The distribution defines a measure that
takes a single variable as an input. That variable that have several dimensions,
e.g. multivariate normal.

Now let us compare this with a simple model, a linear regression:

```python
def linear_regression(X):
  weights ~ Normal(0, 1)
  sigma ~ Normal(0, 1)
  y ~ Normal(x * weight, sigma)
  return y
```

We can define a generative model as such: `linear_regression` is a
function that returns a random value of `y` everytime we pass a number `x`. I
now have a question:

But if `linear_regression` is a function, what does it mean to write (and should
it be valid)?

```python
w ~ linear_regression(rng_key, 10)
```

We can intuitively think that sampling values makes straightforward sense:

```python
w = linear_regression(10).sample(rng_key, num_samples=1_000)
```

But there is a cognitive conflict between the intepretations `linear_regression(10)` is the result of a function and `linear_regression(10)` is a distribution.

So, are generative models distributions? There is a sense that they should, but
it is not clear to me.

Let us write the `logpdf` function:

```python
w = linear_regression.logpdf(10, weights, sigma, y)
```

Defining the random variable through its forward sampling behaviour.

Can we define other distributions like this:

```python
def Exponential(lmbda):
    U ~ Uniform(0, 1)
    t = - np.log(U) / lmbda
    return t
```

The function *generates* numbers drawn from the exponential distribution. What
is the associated logpdf?

```
def Exponential_logpdf(lmbda, t):
    U = np.exp(- lmbda * t)
    return U.logpdf(np.exp(- lmbda * t))
```

Now another slightly more complicated distribution, the multivariate normal:

```python
def MvNormal(mu, sigma):
  z ~ Normal(0, 1, size=2)
  a = Cholesky(sigma)
  x = mu + a * z
  return x
```

All these use independent draws from the same distribution. Models are a
generalization of this concept where random number can be generated from draws
from different distributions. Note that the logpdfs here are functions of
deterministic variables. This is what we are interested in.

```
+---+      +---+
| z | ---> | x |
+---+      +---+
```

The logpdf can be a function of any random variable or function of a random
variable. How do we indicate this is what we care about: we retun it!
 
generative definition -> Distribution (sample, logpdf)

A generative model generated as above *is not* a distribution but it implicitly
defines a multivariate distribution through its generative process. It is a
probability distribution augmented by a `forward`, or `call` function.

```python
def linear_regression(X):
  weights ~ Normal(0, 1)
  sigma ~ Normal(0, 1)
  y ~ Normal(x * weight, sigma)
  return y
```

```python
def linear_logpdf(X, weight, sigma, y):
  logpdf = 0
  logpdf += Normal(0 ,1).logpdf(weight)
  logpdf += Normal(0, 1).logpdf(sigma)
  logpdf += Normal(X * weight, sigma).logpdf(y)
  return logpdf
```

So the `LinearRegression` logpdf could be defined as:

```python
class LinearRegression(Distribution):
    def __call__(self, X):
      weights, sigma, y = self.sample(X)
      return y

    def sample(self, X):
      weights = Normal(0, 1).sample()
      sigma = Normal(0, 1).sample()
      y = Normal(self.X * weights, sigma).sample()
      return weights, sigma, y

    def logpdf(self, X, weight, sigma, y):
      logpdf = 0
      logpdf += Normal(0 ,1).logpdf(weight)
      logpdf += Normal(0, 1).logpdf(sigma)
      logpdf += Normal(X * weight, sigma).logpdf(y)
      return logpdf
```

So a slight modification of the distributions can make them equivalent:

```python
class Normal(distribution):
  def __call__(self, mu, sigma):
      return self.sample(mu, sigma)

  def sample(self, mu, sigma):
      return mu + random.norm() * sigma

  def logpdf(self, x, mu, sigma):
      return stats.logpdf(x, mu, sigma)
```

```python
def normal(mu, sigma):
  event_shape = ()
  batch_shape = lax.broadcast_shape(mu, sigma)
  domain = constraints.real

  def sample(rng_key, sample_shape):
    return mu + random.norm(rng_key, sample_shape) * sigma
  
  @limit_to(domain)
  def logpdf(x):
    return stats.norm(x, mu, sigma)
```

```python
def linear_model(X):
    def call(rng_key, sample_sample):
      # something

    def sample(rng_key, sample_shape):
      w = normal(0, 1)[0](rng_key, sample_shape)
      s = normal(0, 1)[0](rng_key, sample_shape)
      y = normal(X*w)[0](rng_key)
      return y

    def logpdf(x, w, s, y):
      logpdf = 0
      logpdf += normal(0, 1)[1](w)
      logpdf += normal(0, 1)[1](s)
      logpdf += normal(X*w)[1](y)
      return logpdf
```

So when I do

```
model = mcx.model(linear_regression)
model(X) gives a random sample from distribution -> "call" = forward sampling
model.sample(X, ...) gives an aribitrary number of samples from distribution
model.logpdf(x, w, s, y) gives the logpdf
```

## The "return" is really only there to indicate which variables are interesting
## during forward calls.

before the model has been learned:

```
def forward(X):
    w = normal(0, 1).sample()
    s = normal(0, 1).sample()
    return normal(x*w).sample()
```


Once the model has been learned:

```
def forward(X, trace):
    w = empirical(trace['x']).sample()
    s = empirical(trace['s']).sample()
    return Normal(x*w).sample() <-- important to have the "return" !!!
```

```python
sampler, logpdf = model(linear_regression)
```


```python
class model():
  def __init__(self, X):
    self.X = x
    print(sample(X))

  def __call__(X):
    w, s, y = self.sample_forward(X)
    return y

  def sample_forward():

  def sample_posterior():

  def logpdf():
```

```python
Normal(0, 1)
sample_normal(0, 1)
```

So a generative model can be transformed into a sample generating process as
follows:

```python
def linear_model(X):
    weights = Normal(0, 1)
    sigma = Normal(0, 1) 
    y = Normal(weights*x, sigma)
    return y
```

Model as distribution?

```python
@as_distribution
def HorseshoePrior(a):
  return Normal(0, 1)

def complicated_model(X):
    a @ HorseshoePrior(0)
    x @ Normal(a)
    return x
```

`Model` inherits from distribution. When asking for model logpdf -> instantiates
Model.

Ok but how do we define a model?

```
model = model(complicated_model)  # telling it's a model
```
