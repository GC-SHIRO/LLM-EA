import random
import multiprocessing
import time
import networkx as nx
import igraph as ig
from utils import Dataset_Loader
from utils import sample_by_degree_distribution, sorted_by_value
from scipy.sparse import load_npz
from sklearn.metrics import normalized_mutual_info_score

class ScoringModule:
    def __init__(self, file_path, truthpath):
        self.edge_matrix = Dataset_Loader(path=file_path)
        self.ground_truth_labels = {}
        with open(truthpath, 'r') as file:
            for index, line in enumerate(file):
                value = int(line.strip())
                self.ground_truth_labels[index] = value
        
    def modularity(self,edge_matrix, community_dict):
        community_membership = [community_dict[i] for i in range(len(edge_matrix))]
        g = ig.Graph.Adjacency(edge_matrix, mode='UNDIRECTED')
        modularity_value = g.modularity(community_membership)
        return modularity_value

    def run_algorithm(self, result_queue, edge_matrix, algorithm, globals_dict, ground_truth_labels):
        try:
            exec(algorithm, globals_dict)
            result_dict = globals_dict['label_nodes'](edge_matrix)
            score = self.modularity(edge_matrix, result_dict)

            gd_labels = [ground_truth_labels[i] for i in sorted(ground_truth_labels.keys())]
            predicted_labels = [result_dict[i] for i in sorted(result_dict.keys())]
            nmi = normalized_mutual_info_score(gd_labels, predicted_labels)
            result_queue.put((score, nmi))

        except Exception as e:
            print("Scoring evaluate:",end='')
            print(e)
            print("This code can not be evaluated!")
            result_queue.put((0,0))

    def score_nodes_with_timeout(self, algorithm, timeout=60):
        globals_dict = {}
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self.run_algorithm,
            args=(result_queue, self.edge_matrix, algorithm, globals_dict, self.ground_truth_labels)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            print("Algorithm execution exceeded timeout. Terminating...")
            process.terminate()
            process.join()
            return (0,0)
        else:
            return result_queue.get()
    def evaluate_algorithm(self, algorithm):
        res = self.score_nodes_with_timeout(algorithm, timeout=60)
        score = res[0]
        nmi = res[1]
        return score, nmi


if __name__ == '__main__':
    print('Score Module')
