---
title: "Designing modular inference engines: API for the HMC kernel"
date: 2020-02-13T09:13:38+01:00
draft: false
---

*Update (20/12/2020): MCX’s inference core is being moved to [BlackJAX](https://github.com/blackjax-devs/blackjax), a joint project with the Numpyro and PyMC devs.*

I have been working on a probabilistic programming library,
[MCX](https://github.com/rlouf/mcx) (don't use it yet, most of the inference
engine is in API prototype stage) for the past few weeks. The library is based
on source code transformation: you express the model as a python function, and a
compiler reads the function, applies the necessary transformations and outputs
either a function that generates samples from this distribution, or its logpdf.
The logpdf can be JIT-compiled with JAX and used for batched inference.

Despite the black magic involved, this scheme is convenient; it neatly separates
the process of defining the model and that of performing inference, with python
functions as the bridge between the two. Algorithm-specific optimizations are
done at compilation time; as a result, inference algorithms need not be aware of
how the logpdf was generated and can execute it "as is".

It is even possible to use inference engines without using MCX's DSL: users can
import the `inference` module in their projects and use custom-built functions
as logpdfs. As long at the logpdf is compatible with JAX's `jit` and `grad` the
inference with work out of the box. *Designed appropriately, MCX could be used
both for its DSL and convenient interface with common algorithms, but also for
the composability of its inference elements.*

Since we wish to expose the internals to experienced users there is a strong
incentive to make the inference engine as modular as is possible and its parts
re-usable. This way, users can come up with combinations that are not yet
implemented in MCX or even build their own parts.


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

When it comes to implementation, here is the implementation of HMC that you
often find (I omitted the implementation of the integrator for simplicity):

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
    potential: Callable,
    integrator: Callable,  # integrates the trajectory
    momentum_generator: Callable,  # generates a momentum value
    kinetic_energy: Callable,  # computes kinetic energy from momentum value
    num_integration_steps: int,
    step_size: float,
):
    
    def step(rng_key, state: HMCState) -> HMCState:
        """Moves the chain by one step using the Hamiltonian dynamics.
        """
        key_momentum, key_uniform = jax.random.split(rng_key)

        momentum = momentum_generator(key_momentum)
        position, momentum, log_prob, log_prob_grad = integrator(
            potential,
            state.position,
            momentum,
            state.log_prob_grad,
            num_integration_steps,
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

But this implementation has its limits, which often cause PPLs to copy-paste a
good portion of the code whenever a new member of the HMC family is introduced.
In particular empirical HMC (eHMC) and NUTS where the number of integration steps
changes between iterations. It is temptin to pass `step_size_generator` and
`path_length_generator` to `hmc`. While this works for eHMC and algorithms with
an adaptive step size, it fails to accomodate more complex schemes like NUTS.

With hindsight, what I am about to write feels like stating the obvious: this
design smells because it is conceptually incorrect; *the step size and path
length are not properties of the kernel, but of the path integration.* We can
even go further, and separate proposals from integration. After all, both eHMC
and NUTS do call the same integrators and there is no reason they should be
"trapped" in a specific proposal.

For empirical HMC where the path length is drawn from an empirical distribution
with `num_steps_generator`, this would look like:

```python
def ehmc_proposal(
    step_size: float,
    num_steps_generator: Callable,  # returns a number of steps
    integrator: Callable = leapfrog,
):

    def propose(rng_key, state):
        num_integration_steps = num_steps_generator(rng_key)
        for _ in range(num_integration_steps): 
            state = integrator(state, step_size)
        return state
      
    return propose
```

Note that by using a closure we can make every proposal depend only on a PRNG key
and the chain state. In other words, they are kernels and all provide the same
interface to the kernel. This way we can accomodate virtually any variation on
HMC while using the same kernel. The HMC kernel now looks like:

```python
def hmc_kernel(
    logpdf: Callable,
    proposal: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
):
    
    def step(rng_key, state: HMCState) -> HMCState:
        """Moves the chain by one step using the Hamiltonian dynamics.
        """
        key_momentum, key_uniform = jax.random.split(rng_key)

        momentum = momentum_generator(key_momentum)
        position, momentum, log_prob, log_prob_grad = proposal(
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

And here is a summary of the decomposition of the algorithms into independent
parts:

```

          +--> Momentum generator      
          |         |              step size (generator)
          |         v                    v
Metric ---+      Kernel <----------- Proposal  <---------- Integrator
          |         ^                    ^
          |         |               num steps (generator)
          +--> Kinetic Energy             


```

Decomposing the HMC kernel in meaningful blocks and specializing using a closure
brings a lot of benefits:

- The logic of the kernel is more transparent;
- The code is more modular; we can separate the implementations of the dynamics
  (momentum and kinetic energy) and trajectory integration. We are free to use
  any symplectic integrator (2nd, 3rd, etc. order)[^symplectic], use any metric (euclidean,
  riemannian, etc.) and any variant in implementation. 
- One can initialize a kernel with algorithms that are not in the library. These
  algorithm only need to expose the same API as the others.
  
A couple of examples:
- Vanilla HMC uses a Euclidean metric, a fixed step size and a fixed number of
  steps. We can use any integrator we want.
- empirical HMC uses a distribution for the number of integration steps, a fixed
  step size.
- NUTS uses a fixed step size, but a complex internal logic to determine the
  number of integration steps.
- For all the above you can switch the metric to a Riemannian metric and use the
  appropriate integrator.

This makes development easier and less error-prone.


[^symplectic]: I am particularly excited about being able to implement and use
  algorithms mentioned in [this review paper](https://arxiv.org/abs/1711.05337).
