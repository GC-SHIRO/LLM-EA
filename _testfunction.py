import networkx as nx
import numpy as np
from utils import Dataset_Loader
from utils import sample_by_degree_distribution
from scipy.sparse import load_npz
import multiprocessing
import igraph as ig
from score import ScoringModule
from utils import check_code
from prepared_algorithm import pre_algorithms

from sklearn.cluster import AgglomerativeClustering
number_population = 10
pool_size = 10
number_of_init_algorithms = 10
number_for_crossover = 4
target_score = 1
target_epoch = 3
prob_mutation = 0.3
similarity_lower_threshold = 0.93
similarity_upper_threshold = 0.99
file_path = './results/algorithms_pool.pkl'
dataset_path = f"./datasets/dolphins.txt"
model_name = 'bert-base-uncased'
Total_Cost = 0

algorithm = '''import numpy as np
def label_nodes(edge_matrix, k=2):
    n = len(edge_matrix)
    labels = np.random.randint(0, k, size=n)
    def compute_centroids():
        centroids = np.zeros((k, n))
        for i in range(n):
            centroids[labels[i]] += edge_matrix[i]
        for i in range(k):
            centroids[i] /= np.sum(labels == i)
        return centroids
    for _ in range(100):
        centroids = compute_centroids()
        new_labels = np.argmin(np.linalg.norm(centroids[:, np.newaxis] - edge_matrix, axis=2), axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
    return {i: labels[i] for i in range(n)}'''

scoring_module = ScoringModule(
    file_path=dataset_path,
    truthpath='./datasets/dolphins_groundtruth.txt'
)
def get_truth(path):
    result_dict = {}
    with open(path, 'r') as file:
        for index, line in enumerate(file):
            value = int(line.strip())
            result_dict[index] = value
    return result_dict



def main():
    # ground_truth = get_truth('./datasets/dolphins_groundtruth.txt')
    # print(ground_truth)
    score , nmi = scoring_module.evaluate_algorithm(algorithm)
    print(score)
    print(nmi)
    # for i in pre_algorithms:
    #     print(scoring_module.evaluate_algorithm(i))
    #print(check_code(algorithm))
if __name__ == "__main__":
    main()
