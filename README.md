### EP Structured Variational Auto-Encoders

This code is submitted as part requirement for the MSc Degree in Computational Statistics and Machine Learning at University College London.

The code is based on, and forked from https://github.com/mattjj/svae.  We would like to express our thanks to Matt Johnson et al. for making the original SVAE code publicly available.  Our new contributions are primarily the addition of EP inference routines to the [GMM model](svae/models/gmm.py), as well as the new [cycle GMM model](svae/models/gmm_cycle.py) with [corresponding experiments](experiments/gmm_cycle_2d.py).

A summary of new contributions can be seen at https://github.com/jonny-so/svae/compare/master...epsvae
