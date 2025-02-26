import numpy as np
import networkx as nx 
import openai
import re
import threading
import multiprocessing
import time
import igraph as ig
import random
from scipy.sparse import load_npz

#write apikey and model here
KEY = "   "
model = '  '
url = "   " 
#此处使用的是旧版的openai库，新版的openai库无法使用




Input_35 = 0.001
Output_35 = 0.002

def extract_function(text):
    pattern = r'(?<=def\s\w+\(.*\):)|#.*'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# 定义超时处理函数
# class TimeoutException(Exception):
#     pass

def check_code(code_str):
    try:
        edge_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        all_results = [{0:0,1:0,2:0},
                        {0:1,1:0,2:0},{0:0,1:1,2:1},{0:1,1:0,2:1},{0:0,1:1,2:0},{0:1,1:1,2:0},
                        {0:0,1:0,2:1},{0:0,1:1,2:2},{0:0,1:2,2:1},{0:1,1:0,2:2},{0:2,1:0,2:1},
                        {0:1,1:2,2:0},{0:2,1:1,2:0}]
        globals_dict = {}
        exec(code_str, globals_dict)
        result_dict = globals_dict['label_nodes'](edge_matrix)
        if any(result_dict == d for d in all_results):
            return True
        else:
            print("Code output is incorrect")
            return False
    except Exception as e:
        print(f"Code can not run: {e}")
        return False

def calculate_pairwise_connectivity(Graph):
    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)] 
    element_of_pc  = [size*(size - 1)/2 for size in size_of_connected_components] 
    pairwise_connectivity = sum(element_of_pc)
    return pairwise_connectivity

def LLM_generate_algorithm(prompt, stochasticity = 1):#prompt ——> LLMs ——> res , cost($)
    scoring_module = ScoringModule(file_path="./datasets/dolphins.txt")#用于验证可运行性

    openai.api_key = KEY
    openai.api_base = url
    while True:
        for retry in range(10):
            try:
                completion = openai.ChatCompletion.create(                               
                    model= model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=stochasticity
                )
                algorithm = completion.choices[0].message["content"]
                break
            except Exception as e:
                print("API Failed:",end='')
                print(e)
                print(f"retry Time:{retry}")

        Input_tokens = completion.usage.prompt_tokens
        Output_tokens = completion.usage.completion_tokens
        all_cost = Input_tokens / 1000 * Input_35 + Output_tokens / 1000 * Output_35
        score = scoring_module.evaluate_algorithm(algorithm)
        if score != None and score != 0.0 and score != 0:
            print('Generated Algorithm Succeeded!')
            break
        else:
            print('Generated Algorithm Failed!')
    return algorithm, all_cost

def sorted_by_value(my_dict):
    sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_dict.keys())

def sample_by_degree_distribution(adj_matrix, sample_size_ratio):
    degrees = adj_matrix.sum(axis=1)
    selection_probability = degrees / degrees.sum()
    num_nodes = len(degrees)
    sample_size = int(num_nodes * sample_size_ratio)
    sampled_nodes = np.random.choice(num_nodes, size=sample_size, replace=False, p=selection_probability)
    sampled_adj_matrix = adj_matrix[sampled_nodes, :][:, sampled_nodes]
    return sampled_adj_matrix
    
def Dataset_Loader(path):
    edges = []
    with open(path, 'r') as f:
        for line in f:
            u, v = map(int, line.split(' '))
            # u -= 1 #针对dataset从1开始计数的情况
            # v -= 1
            edges.append((u,v))
    max_node = max(max(u,v) for u, v in edges)
    adj_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)
    for u, v in edges:
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    return adj_matrix

class ScoringModule:
    def __init__(self, file_path):
        self.edge_matrix = Dataset_Loader(path=file_path)
    def modularity(self,edge_matrix, community_dict):
        community_membership = [community_dict[i] for i in range(len(edge_matrix))]
        g = ig.Graph.Adjacency(edge_matrix, mode='UNDIRECTED')
        modularity_value = g.modularity(community_membership)
        return modularity_value

    def run_algorithm(self, result_queue, edge_matrix, algorithm, globals_dict):
        try: 
            exec(algorithm, globals_dict)
            result_dict = globals_dict['label_nodes'](edge_matrix)
            score = self.modularity(edge_matrix, result_dict)
            result_queue.put(score)
        except Exception as e:
            print("Generate evaluate:",end='')
            print(e)
            #print(f"algorithm:{algorithm}")
            print("This code can not be evaluated!")
            result_queue.put(0)
    def score_nodes_with_timeout(self, algorithm, timeout=60):
        globals_dict = {}
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self.run_algorithm,
            args=(result_queue, self.edge_matrix, algorithm, globals_dict)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            print("Algorithm execution exceeded timeout. Terminating...")
            process.terminate()
            process.join()
            return 0
        else:
            return result_queue.get()
    def evaluate_algorithm(self, algorithm):
        score = self.score_nodes_with_timeout(algorithm, timeout=60)
        return score

if __name__ == '__main__':
    print("Utils")
