---
title: "GMC: Probabilistic programming in Golang"
date: 2018-03-10T22:11:10+01:00
draft: true
---

Probabilistic programming is amazing. Seriously. Bayesian analysis is the most useful stuff I have learnt over
the past couple of years. We use it on a daily basis at Sounds to perform A/B tests, measure the numer of
sessions and sessions lengths, detecting trends in timeseries, etc.

It is fair to say that Bayesian data analysis wouldn't have the success that it currently has if libraries
like Stan, PyMC3 (which I heavily used) or Edward and many others hadn't existed.

Most libraries are written for R or Python. Although I do enjoy Python a lot for prototyping, I prefer to use 
Go to write code that is production-ready. Because we intend to use Bayesian analysis in production, it would
be convenient to have a library that is written in Go.

The second reasons is that I am new to Go (a couple of months), and I wanted to pick up a project that went
beyond writing web servers to learn to write idiomatic Go. I want to go how far the language can go despite
the usual complaint about the lack of generics.

Finally, I just like to understand the tools I am using, especially when they are performing non-trivial
operations.

I started with 3 requirements:

1. The language should be expressive; 
2. The code should be simple;
3. The inference should be fast. 

(3) was not really a concern: Go is naturally a fast language, and spawning goroutines can help reduce the
computation time without a great complexity cost. (1) wasn't either: if PyMC3 could do it in Python, we should
probably be able to do something similar.  I was more worried about (2). Chaining functions seems more
adapated to functional languages than procedural ones. This is why libraries such as Edward and PyMC3 have to
resort to Tensorflow and Theano respectively, that compile a computation graph before inference is performed.
These tools are not available in Go, so I had to find another way.

That's when I remembered about closure.

A variable hasa minimum of 3 properties that we need: its name, its id in the graph, its value and. Here is
how they are defined:

```golang
type Variable struct {
  id     int
  name   string
  value  func() float64
}
```

The thing that should strike you is that *the value is a function*. To see why that is necessary, consider the
case where $x$ and $y$ are variables, and we want to define $z = x + y$. The thing is, during inference we are
going to set $x$ and $y$ to some random values, and the value of $z$ should vary accordingly. We thus define
the function `Add` as :

```golang
func Add(x, y *Variable) *Variable {
  fn := func() float64 {
    return x.Value() + y.Value()
  }
  return &Variable{value: fn}
}
z := Add(x, y)
```

What we did is that we used a **closure** to keep a reference to X's and Y's value in Z's Value() functions.
Because the value of X and Y are obtained via a function, it doesn't matter whether they are pure variables,
or transformed variables themselves: the implementation is transparent to that.

Now, a random variable is a variable to which a logPdf is attached.

```go
type RandomVariable struct {
  id     int
  name   string
  value  func() float64
  logPdf func(float64) float64
}
```

We create a new normally distributed random variable using `gonum/stats` implementation of probability
distributions:

```go
func Normal(mu, sigma *Variable) *Variable {
  rfn := func(x float64) float64 {
    normal := distuv.Normal{Mu: mu.Value(), Sigma: sigma.Value()}
    return normal.LogProb(x)
  }
  return NewRV(rfn)
}
```

We used the same  
