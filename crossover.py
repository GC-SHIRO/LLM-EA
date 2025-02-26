from utils import LLM_generate_algorithm

class CrossoverModule:
    def __init__(self, task, prompt_crossover, prompt_code_request, extra_prompt):
        self.task = task
        self.prompt_crossover = prompt_crossover
        self.prompt_code_request = prompt_code_request
        self.extra_prompt = extra_prompt

    def crossover_algorithms(self, parent_algorithms):
        crossed_algorithms = []
        all_cost = 0
        
        for i in range(len(parent_algorithms)):
            for j in range(i+1, len(parent_algorithms)):
                if parent_algorithms[i][1] < parent_algorithms[j][1]:
                    all_prompt = self.prompt_crossover + f"Algorithm 1: " + parent_algorithms[i][0] + '\n' + f"Algorithm 2: " + parent_algorithms[j][0] + '\n' + "Algorithm 2 is the better one." + self.prompt_code_request + self.extra_prompt
                else:
                    all_prompt = self.prompt_crossover + f"Algorithm 1: " + parent_algorithms[j][0] + '\n' + f"Algorithm 2: " + parent_algorithms[i][0] + '\n' + "Algorithm 2 is the better one." + self.prompt_code_request + self.extra_prompt
                algorithm, cost = LLM_generate_algorithm(all_prompt)
                crossed_algorithms.append(algorithm)
                all_cost += cost

        return crossed_algorithms, all_cost

if __name__ == '__main__':
    print('Crossover Module')
