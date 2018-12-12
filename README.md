Conditional Generation from a Sentence Variational Autoencoder
--------------------------------------------------------------

**based on**

+ [Generating Sentences from a Continuous Space Bowman et al 2015](https://arxiv.org/abs/1511.06349)
+ [Latent Constraints: Learning to Generate Conditionally from Unconditional Generative Models Jesse et al 2018](https://openreview.net/forum?id=Sy8XvGb0-)
+ [VAE Model Architecture](https://github.com/timbmg/Sentence-VAE/)
+ [Latent Constraints Model](www.alejandro.com)

**scripts**

+ `train_vae.py`: trains a VAE to build a continuous latent space out of sentences.
+ `train_ac.py`: trains an actor critic pair to apply the realism constraint on the VAE's latent space, as well as conditionally generate using the phrase-level parse tags defined in `utils.py`.
+ `make_parsers.py` makes `.pkl` files:
    + `parsers/grammar.pkl`: a PCFG of the entire penn treebank parses
    + `parsers/viterbi_parser.pkl`: a cubic-time parser trained on this grammar.
    + `parsers/shift_reduce_parser.pkl`: a linear-time parser trained on this grammar.
+ `inference.py`: generates samples from a saved-model's latent space, optionally using conditional generation via the actor.
+ `data/download_data.sh` downloads treebank data.
+ `run_scripts.sh`: wrapper to run all experiments.

**parser performance**

Parser performance degrades rapidly for the viterbi algorithm when using longer
sentences. We might need to switch to the `shift_reduce` parser for longer
sentences.

```
viterbi      = 11.49  sec for 12 words
shift reduce = 0.55   sec for 12 words
viterbi      = 24.59  sec for 15 words
shift reduce = 0.98   sec for 15 words
viterbi      = 59.02  sec for 21 words
shift reduce = 1.17   sec for 21 words
viterbi      = 113.06 sec for 26 words
shift reduce = 1.46   sec for 26 words
viterbi      = 923.90 sec for 48 words
shift reduce = 2.54   sec for 48 words
```

**dataset interaction**

`datasets['train'][0]` returns a dict of sentence 0 from 'train' (vs 'valid')
with the following fields:

+ `input`: numpy array of words in int form,
+ `input_str`: list of the words (preprocessed)
+ `input_tag`: list of the POS tags (maybe not required),
+ `target`: numpt array of the words in int form,
+ `target_str`: list of the words (preprocessed),
+ `target_tag`: list of the POS tags (maybe not required),
+ `length`: length of the input sentence in tokens
+ `phrase_tags`: binary vector of the phrase-level tags `['SBAR', 'PRT', 'PNP', 'INTJ', 'ADJP']`

**penn treebank tags**

[See here for all the details.](http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html#X)

+ `SBAR Clause introduced by subordinating conjugation n=21612`
+ `PP   Prepositional phrase                           n=36143`
+ `ADJP Adjective phrase                               n=11738`
+ `QP   Quantifier phrase                              n=7043`
+ `WHNP Wh-noun phrase                                 n=8429`
+ `ADVP Adverb phrase                                  n=17321`


Abstract / Introduction
-----------------------

Previous work has shown that, using variational autoencoders (VAE), one can
build a continuous latent space from which one can sample novel sentences. Other
recent work has shown that, given a latent space trained in an unsupervised way,
one can use an actor-critic approach to learn how to A) sharpen the
representations in this latent space (i.e., improve their quality), and B)
conditionally generate samples using categorical tags.

In this work, we propose to incorperate phrase-level parse tags to generate
sentences of a particular syntactic structure from a previously-trained VAE code
layer. We believe that the incorperation of these parse tags will allow us to
fine-tune the latent space such that it produces more syntactically-correct
(and therefore, more grammatical) sentences.

- [/] 3-5 references of previous work. Include some stuff on efficient parsing
      using large CFGs.
- [ ] Novel contribution.


Methods
-------

- [X] VAE construction, training (notes on KL-divergence annealing and word
      dropout).
- [X] Actor-critic construction.
- [X] Dataset (Penn Treebank, full parse trees.)
- [ ] Evaluation of realism (PCFGs / Parse Accuracy Measure)

    -- sample from neihbourhood, show in continuity in phrase level information
    -- evaluation should not be internal to the theory of the model -- BLEU
    -- paraphrasing, sentence similarity -- BLEU


Results
-------

- [X] VAE training curves
- [ ] Actor-critic results -- samples pre/post realism constraint, samples with
      conditional generation.
- [ ] Quantitative measures of sample quality. **SUGGESTIONS**


Discussion
----------

- [ ] Conclusion
- [ ] Limitations
- [ ] Future work


Contributions
-------------

This was a Posada & Viviano Joint.

