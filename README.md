# Expectation Maximization using Kmeans Algorithm
This repository performs Expectation Maximization (EM) algorithm initializing the parameters with K-Means algorithm.

# Dataset Description
The file Binomial_20_flips.txt contains the outcomes of coin tosses with 3 different coins. A coin is
selected based on an unknown prior probability from a set of 3 coins. The selected coin is tossed 20 times and
the number of heads is noted. This procedure is repeated 1000 times to give rise to the 1000 entries in the
Binomial_20_flips.txt file. The task of this repository is to determine the parameters (probability of heads) for the 3 coins and the prior probabilities of picking a coin.

# Code Description
1. k_means.py file implements K-Means algorithm with 3 clusters to determine the parameters and the prior probabilities.
2. em.py file implements Expectation Maximization (EM) algorithm to determine the parameters and the prior probabilities.
3. em_kmeans.py file repeats Expectation Maximization (EM) algorithm, but it initializes parameters and prior probabilities with the outcomes of K-Means algorithm.
