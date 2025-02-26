import random
from prepared_algorithm import pre_algorithms
from utils import LLM_generate_algorithm

class InitializationModule:
    def __init__(self, task, prompt_init, prompt_code_request, extra_prompt, handmade=False):
        self.all_prompt = task + prompt_init + prompt_code_request + extra_prompt
        self.handmade = handmade

    def generate_initial_algorithms(self, count):#LLM生成初始个体或直接手动获取
        if not self.handmade:
            algorithms = [LLM_generate_algorithm(self.all_prompt) for i in range(count)]
            cost = sum(cost for _, cost in algorithms)
        else:
            algorithms = self.handmade_algorithms()
            cost = 0
        return algorithms[:count], cost

    def handmade_algorithms(self):
        return pre_algorithms

if __name__ == '__main__':
    print('Initialization Module')