import numpy as np
from math import factorial
import random
from matplotlib import pyplot as plt
from k_means import determine_cluster_parameters

def calculate_combination(n, k):
	return factorial(n) / (factorial(k) * factorial(n - k))

def calculate_binomial(prob, n, k):
	return calculate_combination(n, k) * (prob ** k) * ((1 - prob) ** (n - k))

def calculate_mixture(prob_initials, prob_heads, n, k):
	mixture_prob = 0
	for i in range(len(prob_heads)):
		mixture_prob += prob_initials[i] * calculate_binomial(prob_heads[i], n, k)
	return mixture_prob

def calculate_log_likelihood(data, prob_initials, prob_heads, n):
	log_likelihood = 0
	for instance in data:
		mixture_prob = calculate_mixture(prob_initials, prob_heads, n, instance)
		if mixture_prob <= 0:
			mixture_prob = 0.00000000000000000000000001
		log_likelihood += np.log(mixture_prob)
	return log_likelihood

def maximize_expectation(data, prob_initials, prob_heads, n):
	#Evaluating log-likelihood with initial parameters
	log_likelihood = calculate_log_likelihood(data, prob_initials, prob_heads, n)

	#starting iteration for E-step, M-step and log-likelihood evaluation
	iter = 1
	likelihoods = []
	iterations = []
	while True:
		#E-step: calculating responsibilities
		sum_responsibilities = [0, 0, 0]
		sum_responsibilities_numerator = [0, 0, 0]
		for instance in data:
			mixture_prob = calculate_mixture(prob_initials, prob_heads, n, instance)
			for j in range(len(prob_heads)):
				responsibilities = (prob_initials[j] * calculate_binomial(prob_heads[j], n, instance))/mixture_prob
				sum_responsibilities[j] += responsibilities
				sum_responsibilities_numerator[j] += responsibilities * instance

		#M-step: re-estimating parameters
		for j in range(len(prob_heads)):
			prob_initials[j] = sum_responsibilities[j]/len(data)
			prob_heads[j] = sum_responsibilities_numerator[j]/sum_responsibilities[j]

		#Evaluating log-likelihood with new parameters
		log_likelihood_new = calculate_log_likelihood(data, prob_initials, prob_heads, n)

		if abs(log_likelihood_new - log_likelihood) < 0.01:
			break
		else:
			likelihoods.append(log_likelihood_new)
			iterations.append(iter)
			log_likelihood = log_likelihood_new
			iter += 1

	return iterations, likelihoods, prob_initials, prob_heads


if __name__== "__main__":
	#loading dataset
	data_file = 'Binomial_20_flips.txt'
	data = np.loadtxt(data_file)

	prob_initials, prob_heads = determine_cluster_parameters(data, 3)
	print("Initial values of parameters")
	print("Prior Probabilities:")
	print(prob_initials)
	print("Probabilities of Heads:")
	print(prob_heads)

	iterations, likelihoods, prob_initials, prob_heads = maximize_expectation(data, prob_initials, prob_heads, 20)

	print("Resulting values of parameters")
	print("Prior Probabilities:")
	print(prob_initials)
	print("Probabilities of Heads:")
	print(prob_heads)

	plt.plot(iterations, likelihoods)
	plt.xlabel("Iterations")
	plt.ylabel("Log-likelihood")
	plt.title("Log-likelihood vs iterations")
	plt.show()

