
You must write and submit a report describing your project. There is no minimum
number of citations required, but the report should be structured like a typical
conference paper:


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

- [ ] 3-5 references of previous work. Include some stuff on efficient parsing
      using large CFGs.
- [ ] Novel contribution.


Methods
-------

- [ ] VAE construction, training (notes on KL-divergence annealing and word
      dropout).
- [ ] Actor-critic construction.
- [ ] Dataset (Penn Treebank, full parse trees.)
- [ ] Evaluation of realism (PCFGs / Parse Accuracy Measure)

??? WHAT SHOULD WE USE TO EVALUATE REALISM?
??? SHOULD WE USE PHRASE TAGS? -- get away from 'grammatical' and move towards
    'phrase types'
??? PARSERS -- Shift Reduce vs. Viterbi, UTILITY of PCFG??
??? CFG -- DEFINED ON ENTIRE PENN TREEBANK, SMARTER WAY TO DO THIS???
    -- HOW TO REDUCE SIZE OF GRAMMAR??


Results
-------

- [ ] VAE training curves
- [ ] Actor-critic results -- samples pre/post realism constraint, samples with
      conditional generation.
- [ ] Quantitative measures of sample quality. **SUGGESTIONS**

??? EVALUATION OF MODEL -- WHAT METRICS?


Discussion
----------

- [ ] Conclusion
- [ ] Limitations
- [ ] Future work


Contributions
-------------

This was a Posada & Viviano Joint.

