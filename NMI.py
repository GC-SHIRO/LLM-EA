from utils import Dataset_Loader
from sklearn.metrics import normalized_mutual_info_score

class NMI_Module():
    def __init__(self,truthpath):
        self.ground_truth_labels = {}
        with open(truthpath, 'r') as file:
            for index, line in enumerate(file):
                value = int(line.strip())
                result_dict[index] = value

    def getNMI(self, labeled_nodes, algorithm):
        nmi = normalized_mutual_info_score(ground_truth_labels, labeled_nodes)
        print(f"NMI:{nmi}")
        return nmi


if __name__ == '__main__':
    print("NMI Module")