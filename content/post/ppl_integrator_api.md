---
title: "Designing modular inference engines: API for the HMC kernel"
date: 2020-02-13T09:13:38+01:00
draft: false
---

I have been working on a probabilistic programming library,
[MCX](https://github.com/rlouf/mcx) (don't use it yet, most of the inference
engine is in API prototype stage) for the past few weeks. I will write more
about it soon, but the library is based on source code transformation: you
express the model as a python function, and a compiler reads the function,
applies the necessary transformations and outputs either a function that
generates samples from this distribution, or its logpdf. The logpdf can be
JIT-compiled with JAX and used for batched inference.

Despite the black magic involved, this scheme is convenient; it neatly separates
the process of defining the model and that of performing inference, with python
functions as the bridge between the two. Algorithm-specific optimizations are
done at compilation time; as a result, inference algorithms need not be aware of
how the logpdf was generated and can execute it "as is".

It is even possible to use inference engines without using MCX's DSL: users can
import the `inference` module in their projects and use custom-built functions
as logpdfs. *Designed appropriately, MCX could be used both for its DSL and
convenient interface with common algorithms, but also for the composability of
its inference elements.*

As a result, there is no excuse to not make the inference engine as modular as
is possible and its parts re-usable. So here I am, going down the rabbit hole
inside the rabbit hole.


## Hamiltonian Monte Carlo

Hamiltonian Monte Carlo (HMC) methods are the cornerstone of most PPLs:
[Anglican](https://probprog.github.io/anglican/),
[Numpyro](http://pyro.ai/numpyro/), [PyMC3](https://docs.pymc.io/),
[Pyro](https://pyro.ai/), [Stan](https://mc-stan.org/), [Tensorflow
Probability](https://www.tensorflow.org/probability),
[Turing.jl](https://turing.ml/dev/), [Rainier](https://rainier.fit/),
[Soss.jl](https://github.com/cscherrer/Soss.jl), and many others I forgot
(someone should write about the current Cambrian explosion of PPLs!) are all
built around variants of Hamiltonian Monte Carlo. If you don't already know why
everyone is using HMC, have a look at [Betancourt's article](https://arxiv.org/abs/1410.5110) for a theoretical explanation. 

The first expression I came up with was the following, which probably looks a
lot like what you are used to. I use a closure to compose the different elements
into a self-contained step function:

```python
class HMCState(NamedTuple):
    """State on which the HMC kernel acts.

    We can (and probably should) add more information which can be useful
    higher up:
      - State of the proposal
      - Whether the move was accepted
      - Acceptance ratio
      - State of the integrator
    """
    position: Array
    log_prob:float
    log_prob_grad: float
    energy: float  # here for convenience


def hmc_kernel(
    logpdf: Callable,
    integrator: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
    path_length: float,
    step_size: float,
):
    
    def step(rng_key, state: HMCState) -> HMCState:
        """Moves the chain by one step using the Hamiltonian dynamics.
        """
        key_momentum, key_uniform = jax.random.split(rng_key)

        momentum = momentum_generator(key_momentum)
        position, momentum, log_prob, log_prob_grad = integrator(
            logpdf,
            state.position,
            momentum,
            state.log_prob_grad,
            path_length,
            step_size,
        )
        new_energy = log_prob + kinetic_energy(momentum)
        new_state = HMCState(position, log_prob, log_prob_grad, new_energy)

        log_uniform = np.log(jax.random.uniform(rng_key))
        do_accept = log_uniform < new_energy - state.energy
        if do_accept:
            return new_state

        return state

    return step
```

where `integrator` can be the often-used leapfrog integrator, or any other
symplectic integrator. Separating `momentum_generator`, `kinetic_energy` and
`integrator` allows you to easily generalize to Riemannian HMC, without having
to re-define a kernel for every possibe combination.

There are, however, problems with this:

1. What about empirical HMC, when path lengths are drawn from an empirical
   distribution?
2. What about [NUTS](https://arxiv.org/abs/1111.4246) where path length is adaptively computed at each iteration?
3. What about adaptive schemes for the step size?

It is temptin to pass `step_size_generator` and `path_length_generator` to
`hmc`. While it works for (1) and (3), it fails to accomodate more complex
schemes like NUTS.

With hindsight, what I am about to write feels like stating the obvious: this
design smells because it is conceptually incorrect; *the step size and path
length are properties of the path integration*. 

For empirical HMC where the path length is drawn from an empirical distribution,
this would look like:

```python
def ehmc_integrator(rng_key, state: IntegratorState):
    step_size = epsilon_0  # constant
    path_length = path_length_generator(rng_key)  # returns a path length
    state = leapfrog(state, step_size, path_length)
    return state
```

And a extremely simplified version of NUTS:

```python
def nuts_integrator(rng_key, state: IntegratorState):
    step_size = epsilon_0  # constant
    state = nuts_integrator(state, step_size)  # performs one leapfrog step
    return state
```

We can now accomodate for many adaptive schemes for the step size and path
length. Notice that the integrators are now only a function of a PRNG key and an
integrator state; in other words they are also kernels. I saw yesterday that
the developers [FunMC](https://arxiv.org/abs/2001.05035) use a similar approach.

The HMC kernel now looks like:

```python
def hmc_kernel(
    logpdf: Callable,
    integrator: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
):
    
    def step(rng_key, state: HMCState) -> HMCState:
        """Moves the chain by one step using the Hamiltonian dynamics.
        """
        key_momentum, key_uniform = jax.random.split(rng_key)

        momentum = momentum_generator(key_momentum)
        position, momentum, log_prob, log_prob_grad = integrator(
            logpdf,
            state.position,
            momentum,
            state.log_prob,
            state.log_prob_grad,
        )
        new_energy = log_prob + kinetic_energy(new_momentum)
        new_state = HMCState(position, log_prob, log_prob_grad, new_energy)

        log_uniform = np.log(jax.random.uniform(rng_key))
        do_accept = log_uniform < new_energy - state.energy
        if do_accept:
            return new_state

        return state

    return step
```

Decomposing the HMC kernel in meaningful blocks and specializing using a closure
brings a lot of benefits:

- The logic of the kernel is more transparent;
- The code is more modular; we can separate the implementations of the dynamics
  (momentum and kinetic energy) and trajectory integration. We are free to use
  any symplectic integrator (2nd, 3rd, etc. order)[^symplectic], use any metric (euclidean,
  riemannian, etc.) and any variant in implementation. 
- One can initialize a kernel with algorithms that are not in the library;
- We can now free to use many adaptive schemes for the parameters of the
  integrator.

I am particularly excited at the perspective of having a module with inference
engines decoupled to any particular PPL implementation. There is a lot of
duplicated effort in the community (I am guilty of that too), and a central
inference library would help in many ways:

1. More pairs of eyes on the code; less bugs.
2. People can focus on implementing different things rather than re-implementing
   NUTS over and over again;
3. More time could be allocated to more interesting problems: designing a flexible
   interface, implementing BNNs, stochastic processes, etc.

Wouldn't that be nice?

[^symplectic]: I am particularly excited about being able to implement and use
  algorithms mentioned in [this review paper](https://arxiv.org/abs/1711.05337).
