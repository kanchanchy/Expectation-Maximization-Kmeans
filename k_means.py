import numpy as np
import random

#Method for selecting initial centers randomly
def init_center_randomly(data, k):
	center_list = []
	index_list = []
	for i in range(k):
		while True:
			random_index = random.randrange(len(data) - 1)
			if random_index not in index_list:
				index_list.append(random_index)
				center_list.append(data[random_index])
				break
	return np.array(center_list)

# Method for finding the closest cluster for each data sample
def find_closest_cluster(instance, center_list):
	distances = []
	for center in center_list:
		distances.append((instance - center)**2)
	distances = np.array(distances)
	min_distance = np.amin(distances)
	return (np.where(distances == min_distance))[0][0]

def kmeans_clustering(data, no_cluster):
	#finding centers of each cluster
	center_list = init_center_randomly(data, no_cluster)

	#finding clusters
	cluster_final = {}
	while True:
		clusters = {}
		for instance in data:
			cluster_index = find_closest_cluster(instance, center_list)
			if cluster_index in clusters:
				clusters[cluster_index].append(instance)
			else:
				clusters[cluster_index] = [instance]
		new_center_list = []
		for i in range(no_cluster):
			if i in clusters:
				np_cluster = np.array(clusters[i])
				new_center_list.append(np.mean(np_cluster, axis = 0))
			else:
				new_center_list.append(center_list[i])
		if np.array_equal(np.array(center_list), np.array(new_center_list)):
			cluster_final = clusters
			break
		else:
			center_list = new_center_list
	return center_list, cluster_final

def determine_cluster_parameters(data, no_cluster):
	center_list, clusters = kmeans_clustering(data, no_cluster)
	total_point = len(data)
	prob_initial = []
	prob_head = []
	for i in range(no_cluster):
		cluster_points = clusters[i]
		prob_initial.append(len(cluster_points)/total_point)
		prob_head.append(np.sum(cluster_points)/(len(cluster_points)*20))
	return np.array(prob_initial), np.array(prob_head)


if __name__== "__main__":
	#loading dataset
	data_file = 'Binomial_20_flips.txt'
	data = np.loadtxt(data_file)

	no_cluster = 3

	prob_initial, prob_head = determine_cluster_parameters(data, no_cluster)
	print("Prior Probabilities:")
	print(prob_initial)
	print("Probabilities of Heads:")
	print(prob_head)



