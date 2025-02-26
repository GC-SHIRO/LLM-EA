import random
import pickle
import csv
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from initialization import InitializationModule
from crossover import CrossoverModule
from mutation import MutationModule
from score import ScoringModule
from pool import AlgorithmPool
from NMI import NMI_Module
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM for Combinatorial Optimization")
    parser.add_argument("--train_type", type=str, default='new', help="Training type: new or continue")
    parser.add_argument("--pool_path", type=str, default='algorithms_25.pkl', help="File path of gained algorithm pool")
    parser.add_argument("--dataset", type=str, default='dolphins', help="Dataset")
    parser.add_argument("--ratio", type=int, default=20, help="Fraction*100 of removed nodes")
    parser.add_argument("--calculate", type=str, default='pc', help="pc of gcc")
    args = parser.parse_args()

    number_population = 10
    pool_size = 10
    number_of_init_algorithms = 10
    number_for_crossover = 4
    target_score = 1
    target_epoch = 100
    prob_mutation = 0.3
    similarity_lower_threshold = 0.93
    similarity_upper_threshold = 0.99
    file_path = './results/algorithms_pool.pkl'
    dataset_path = f"./datasets/{args.dataset}.txt"
    truth_path = f'./datasets/{args.dataset}_groundtruth.txt'
    model_name = 'bert-base-uncased'
    Total_Cost = 0


    task = "Given a edge matrix of a network, you need to find the communities at each node.And give each node a label of which community they at.\n"
    prompt_crossover = "I have two codes as follows:\n"
    prompt_initial = "Please provide a new algorithm.\n"
    prompt_mutation = "Without changing the input and output format of this code,modify this code to make node classification more reasonable and have higher modularity score:\n"
    prompt_code_request = "Mix the two algorithms above, and create a completely different Python function which has higher modularity score.The function called \"label_nodes\" that accepts an \"edge_matrix\" as input and returns \"labeled_nodes\" as output. \"edge_matrix\" should be a adjacency matrix in the form of a NumPy array, and \"scored_nodes\" should be a dictionary where the keys are node IDs and the values are the label of the node.(start count from 0)"
    extra_prompt = "Provide only one Python function, not any explanation, just the function in text;dont't use format like '''python\n code\n'''.Use from_numpy_array() instead of from_numpy_matrix()."


    algorithm_pool = AlgorithmPool(
        number_population=number_population,
        capacity=pool_size,
        model_name=model_name,
        lower_threshold=similarity_lower_threshold,
        upper_threshold=similarity_upper_threshold
    )

    initialization_module = InitializationModule(
        task=task,
        prompt_init=prompt_initial,
        prompt_code_request=prompt_code_request,
        extra_prompt=extra_prompt,
        handmade=True#True:手动代码 False:自动生成（结果质量低）c
    )

    crossover_module = CrossoverModule(
        task=task,
        prompt_crossover=prompt_crossover,
        prompt_code_request=prompt_code_request,
        extra_prompt=extra_prompt
    )

    mutation_module = MutationModule(
        task=task,
        prompt_mutation=prompt_mutation,
        extra_prompt=extra_prompt
    )

    scoring_module = ScoringModule(
        file_path=dataset_path,
        truthpath=truth_path
    )


    if args.train_type == 'new':#新运行
        initial_algorithms, cost = initialization_module.generate_initial_algorithms(count=number_of_init_algorithms)
        Total_Cost += cost#LLM消费，目前有计算问题，后续考虑删除
        print('Initialization!')
        for algorithm in tqdm(initial_algorithms, desc='Scoring Initial Codes'):
            score , nmi = scoring_module.evaluate_algorithm(algorithm)
            algorithm_pool.add_algorithm(algorithm, score, f"Population_{len(algorithm_pool.pool)}", 0, nmi)#初始化打分后加入pool
        print("Evaluate Initial Algorithms")
    else:#准备好的Algorithms（文件路径）
        algorithm_pool.load_algorithm(args.pool_path)

    average_scores = []
    best_scores = []
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d_%H_%M")
    folder_path = f'./results/algorithms_{args.dataset}_{args.ratio}_{args.train_type}_{formatted_time}'
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    population_statistics = []

    #开始生成
    for epoch in tqdm(range(target_epoch), desc='Epoch'):
        #crossover 在pool中sample
        print("Begin Crossover!")
        sampled_algorithms = algorithm_pool.sample_algorithms(number_for_crossover)
        best_algorithm = algorithm_pool.get_best_algorithm()
        if best_algorithm not in sampled_algorithms:
            sampled_algorithms.append(best_algorithm)
        
        crossed_algorithms, cost = crossover_module.crossover_algorithms(sampled_algorithms)
        Total_Cost += cost

        for algorithm in crossed_algorithms:
            
            if random.uniform(0, 1) < prob_mutation:#随机变异
                print("begin mutation")
                algorithm, cost = mutation_module.mutate_algorithm(algorithm)
                Total_Cost += cost
            
            score , nmi = scoring_module.evaluate_algorithm(algorithm)
            print("score :", score, 'nmi :', nmi)
            population_label, similarity = algorithm_pool.algorithm_classification(algorithm)
            algorithm_pool.add_algorithm(algorithm, score, population_label, similarity, nmi)
        
        #evolution 在population中sample
        print("Begin Self-Evolution!")
        epoch_stats = {
            "epoch": epoch,
            "population_stats": []
        }
        for population_label, population in algorithm_pool.pool.items():
            if len(algorithm_pool.pool[population_label]) > 1:
                sampled_algorithms = algorithm_pool.sample_algorithms_population(population_label, 2)
                crossed_algorithms, cost = crossover_module.crossover_algorithms(sampled_algorithms)
                Total_Cost += cost

                for algorithm in crossed_algorithms:

                    if random.uniform(0, 1) < prob_mutation:
                        print("begin mutation")
                        algorithm, cost = mutation_module.mutate_algorithm(algorithm)
                        Total_Cost += cost
                        
                    score, nmi = scoring_module.evaluate_algorithm(algorithm)
                    algorithm_pool.add_algorithm(algorithm, score, population_label, 1, nmi)

            population_size = len(population)
            ave_score_population = algorithm_pool.calculate_average_score(population_label)
            best_score_population = algorithm_pool.get_highest_score(population_label)

            population_stats = {
                "label": population_label,
                "size": population_size,
                "average_score": ave_score_population,
                "highest_score": best_score_population
            }

            epoch_stats["population_stats"].append(population_stats)

        population_statistics.append(epoch_stats)
        ave_score = algorithm_pool.calculate_average_score()
        best_score = algorithm_pool.get_highest_score()
        ave_nmi = algorithm_pool.calculate_average_nmi()
        average_scores.append(ave_score)
        best_scores.append(best_score)

        print(f"Average Score: {ave_score:.4f} Best Score: {best_score:.4f} Average NMI: {ave_nmi:.4f} Population Number: {len(algorithm_pool.pool)}, Pool Size: {algorithm_pool.__len__()} Total Cost: {Total_Cost:.4f}$")

        algorithm_pool.save_algorithms(folder_path + f'/algorithms_{epoch}.pkl')
        score_to_save = {
            "ave": average_scores,
            "best": best_scores
        }
        with open(folder_path + '/score_data.pkl', 'wb') as file:
            pickle.dump(score_to_save, file)
        with open(folder_path + '/population_data.pkl', 'wb') as file:
            pickle.dump(population_statistics, file)

        epoch += 1
        print('Data saved!')

        if best_score >= target_score:
            print('Success!')
            break

    with open('./results/API_cost.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([formatted_time, Total_Cost])
