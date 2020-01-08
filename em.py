import numpy as np
from math import factorial
import random
from matplotlib import pyplot as plt

def generate_prob_initials():
	probs = []
	for i in range(3):
		probs.append(random.randint(1, 10))
	probs = np.array(probs)
	factor = 1/np.sum(probs)
	probs = probs * factor
	return probs

def generate_prob_heads():
	probs = []
	for i in range(3):
		probs.append(random.uniform(0, 1))
	return np.array(probs)


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

		if log_likelihood_new == log_likelihood:
			break
		else:
			likelihoods.append(log_likelihood_new)
			iterations.append(iter)
			log_likelihood = log_likelihood_new
			iter += 1
		'''print("After iteration: " + str(iter))
		print(prob_initials)
		print(prob_heads)'''


	return iterations, likelihoods, prob_initials, prob_heads


if __name__== "__main__":
	#loading dataset
	data_file = 'Binomial_20_flips.txt'
	data = np.loadtxt(data_file)

	prob_initials = generate_prob_initials()
	prob_heads = generate_prob_heads()

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

