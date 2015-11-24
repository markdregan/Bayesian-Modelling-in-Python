# [Bayesian Modelling in Python](https://github.com/markdregan/Bayesian-Modelling-in-Python)

![Bayesian Modelling in Python](/graphics/cover.png)

Welcome to "Bayesian Modelling in Python" - a tutorial for those interested in learning how to apply bayesian modelling techniques in python ([PYMC3](https://github.com/pymc-devs/pymc3)). This tutorial doesn't aim to be a bayesian statistics tutorial - but rather a programming cookbook for those who understand the fundamental of bayesian statistics and want to learn how to build bayesian models using python. The tutorial sections and topics can be seen below.

### Contents
- [**Introduction**](http://nbviewer.ipython.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%200.%20Introduction.ipynb)
    - Motivation for learning bayesian statistics
    - Loading and parsing Hangout chat data
    
- [**Section 1: Estimating model parameters**](http://nbviewer.ipython.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%201.%20Estimating%20model%20parameters.ipynb)
    - Frequentist technique for estimating parameters of a poisson model (Optimization routine)
    - Bayesian technique for estimating parameters of a poisson model (MCMC)

- [**Section 2: Model checking & comparison**](http://nbviewer.ipython.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%202.%20Model%20checking.ipynb)
    - Posterior predictive check
    - Bayes factor
    
- [**Section 3: Hierarchal modeling**](http://nbviewer.ipython.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%203.%20Hierarchical%20modelling.ipynb)
    - Model pooling (separate models)
    - Partial pooling (hierarchal models)
    - Shrinkage effect of partial pooling
    
- [**Section 4: Bayesian regression**](http://nbviewer.ipython.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%204.%20Bayesian%20regression.ipynb)
    - Bayesian fixed effects poisson regression
    - Bayesian mixed effects poisson regression
    
- **Section 5: Bayesian survival analysis**
    - Survival model theory
    - Cox proportional hazard model
    - Aalen's additive hazard model
    
- **Section 6: Bayesian A/B tests**
    - Bayesian test of proportions
    - Bayesian t-test (BEST)

### Contributions
- All contributions are more than welcome. They can be minor (spelling, better explanations, improved code/charts) or major (contribute a full section).
- If you would like to contribute, please create a pull request in GitHub. Happy to discuss ideas before you begin working on the addition.
- I would especially welcome any contributions that address: survival analysis, mixture models, time series models or A/B experiments.
- If you're not familiar with GitHub - please email me at mark@thinkvein.com.

### Motivation for learning bayesian statistics
Statistics is a topic that never resonated with me throughout university. The frequentist techniques that we were taught (p-values etc) felt contrived and ultimately I turned my back on statistics as a topic that I wasn't interested in.

That was until I stumbled upon Bayesian statistics - a branch to statistics quite different from the traditional frequentist statistics that most universities teach. I was inspired by a number of different publications, blogs & videos that I would highly recommend any newbies to bayesian stats to begin with. They include:
- [Doing Bayesian Data Analysis](http://www.amazon.com/Doing-Bayesian-Analysis-Second-Edition/dp/0124058884/ref=dp_ob_title_bk) by John Kruschke
- [Python port](https://github.com/aloctavodia/Doing_bayesian_data_analysis) of John Kruschke's examples by Osvaldo Martin
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) provided me with a great source of inspiration to learn bayesian stats. In recognition of this influence, I've adopted the same visual styles as BMH.
- [While My MCMC Gently Samples](http://twiecki.github.io/) blog by Thomas Wiecki
- [Healthy Algorithms](http://healthyalgorithms.com/tag/pymc/) blog by Abraham Flaxman
- [Scipy Tutorial 2014](https://github.com/fonnesbeck/scipy2014_tutorial) by Chris Fonnesbeck

I created this tutorial in the hope that others find it useful and it helps them learn Bayesian techniques just like the above resources helped me. I hope you find it useful and I'd welcome any corrections/comments/contributions from the community.

### Note
This tutorial is actively being worked on. I'm keen to get feedback and welcome ideas/contributions.
