---
title: "txt: a sequence generation library"
date: 2019-12-25T09:24:11+01:00
draft: true
---

Sequence generation is the most fascinating aspect of the recent advances
in Natural Language Processing. They also potentially have the furthest-reaching
applications:

- Dialogs
- Abstractive Summarization
- Translation

It is however hard for one to keep up with the advances in the field, so I built
this library as a way for me to keep up with the literature but also to
democratize the access to natural language generation methods.

The library is aimed at researchers and practioners alike.

Due to the lack of modularity of the available libraries, researchers often roll
their own version of existing algorithms, or copy/paste the code of other
libraries due to the lack of a composable API. `txt` ensures a clear separation
of concern, and thus a minimal coupling between decoders and the generation
algorithms. This results in modular, general-purpose writers that can be used
with any underlying decoder, with just a few lines of code. This ensures
reproducibility of results and fast comparison of generation methods.

`txt` can perform generation on both CPU an GPU.

`txt` is not tied to a specific implementation of decoder (be it `fairseq`'s ,
`AllenNLP`'s or `HuggingFace`'s) but can easily use every one of them.

# A simple API

APIs should read like prose and require little cognitive effort to use. I
strived to hide the writers' complexity as much as I possibly could, and to
provide a unified interface to each algorithm.

We provide the `generate` and `generate_until` wrapper method as simple
interfaces to generate sequences of text. These should be enough to cover most
use cases:


```python
generated_text = writer.generate(10)
generated_text = writer.generate_until(".")
```

Beware that `generate` takes a number of tokens as an argument, and that the
number of words thus obtained will be lower.

If you prefer to work with tokens ids, we provide the `generate_ids` and
`generate_ids_until` wrapper methods:

```python
generated_ids = writer.generate_ids(10)
generated_ids = writer.generate_ids_until(0)
```

At the core of every algorithms lies a generator function that returns token ids
(for greedy search or sampling) or search state (for beam search). The user can,
but doesn't need to, interact with this generator for more complex workflows. 
```python
for token_id in writer:  # infinite iterator
  print(token_id)
```

## Setup the writer

To start generating sequences, one needs a model and a writer. Let us use
GPT2 medium with greedy search.

```python
from txt.models import GPT2
from txt.writers import greedy_search
```

We can load pre-trained models using an AllenNLP and HuggingFace like API:

```python
model = GPT2.from_pretrained('gpt2-medium')
```

We then initiate a writer with the model simply as:

```python
writer = greedy_search().using(GPT2)
```

The model and writer are loaded on CPU by default. If you want to load
everything on GPU, write instead:

```python
writer = greedy_search().using(GPT2).on("cuda")
```

The library provides a variety of methods to generate samples. From the simplest
to the most customizable.

## Prompt the writer

You may want to generate text from a prompt. You can prompt directly from text or
from token ids:

```python
writer = writer.prompt_with("this was the best of times")
writer = writer.prompt_with_ids([1, 234, 20394, 2])
```

## Generate sequences

### Fixed size

```python
text = writer.generate(10)
token_ids = writer.generate(10)
```

# (Some) batteries included

I included as many transformer architecture as I possibly could to make the
library "batteries included": one can instantiate a decoder, pass it to an
algorithm and simply start writing text. It doesn't get any easier than this!

Note that some of the algorithms are bleeding-edge research and, as such, have
not been integrated to many models.


# re-usable algorithms

Although we provide model implementations for ease of use, every algorithm can
be easily used with any model that outputs logits on its last layer. To be
compatible with the writer's API, your model class needs to implement the
`decode`, `ids_from_text` and `text_from_ids` methods. If you cannot or don't
want to modify you model class, you can write a thin wrapper around it as
follows:


```python
class CustomModel(Decoder):
  def __init__(self, tokenizer, decoder):
    self.tokenizer = tokenizer
    self.decoder = decoder

  def decode(self, input_ids: torch.LongTensor) -> Torch.FloatTensor:
      output = self.decoder(input_ids)
      return output[0][:, -1, :]

  def ids_from_text(self, text: str) -> List[int]:
      return self.tokenizer.encode(text)

  def text_from_ids(self, ids: List[int]) -> str:
      return self.tokenizer.decode(ids)

my_model = CustomModel()
writer = GreedySearch().using(my_model)
token_ids = writer.generate_ids(10)
```

As a result, the writers' API make no assumption about your model's implementation.


# To come

- Stochastic beam search as a nice interpolation between sampling and beam
  search;
- Diverse beam search;
- PPLM fo conditional generation;
- Non-autoregressive writers.
