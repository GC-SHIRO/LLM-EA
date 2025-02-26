#------------------------------------------------------------
# 存放初始个体10个，算法来源分别是
# 1 Louvain算法 0.4097569310948781
# 2 Girvan-Newman算法 0.4952137969225902
# 3 label Propagation算法 0.37348206162730896
# 4 Spectral Clustering算法 0.3787033740753926
# 5 K-means算法 0.30993631581029235
# 6 Walktrap算法 0.37348206162730896
# 7 来自生成 0.38137336339543526（random）
# 8 来自生成 0.3363988766267156
# 9 LFM 0.355741465923025
# 10 来自生成 0.3205569399944623
#------------------------------------------------------------

pre_algorithms = [
        """import numpy as np
def label_nodes(edge_matrix):
    def calculate_betweenness(edge_matrix):
        num_nodes = len(edge_matrix)
        betweenness = np.zeros_like(edge_matrix, dtype=float)
        for source in range(num_nodes):
            dist = [-1] * num_nodes
            dist[source] = 0
            queue = [source]
            paths = [[] for _ in range(num_nodes)]
            paths[source] = [source]
            while queue:
                u = queue.pop(0)
                for v in range(num_nodes):
                    if edge_matrix[u][v] > 0 and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    if edge_matrix[u][v] > 0 and dist[v] == dist[u] + 1:
                        paths[v].append(u)
            dependency = np.zeros_like(edge_matrix, dtype=float)
            for target in range(num_nodes):
                if target == source:
                    continue
                stack = [target]
                while stack:
                    node = stack.pop()
                    for prev in paths[node]:
                        if prev != source:
                            dependency[prev][node] += 1
                            stack.append(prev)
            betweenness += dependency
        return betweenness
    def get_connected_components(edge_matrix):
        visited = [False] * len(edge_matrix)
        components = []
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(len(edge_matrix)):
                if edge_matrix[node][neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor, component)
        for node in range(len(edge_matrix)):
            if not visited[node]:
                component = []
                dfs(node, component)
                components.append(component)
        return components
    def girvan_newman_algorithm(edge_matrix):
        components = get_connected_components(edge_matrix)
        if len(components) > 1:
            return components
        betweenness = calculate_betweenness(edge_matrix)
        max_betweenness = np.max(betweenness)
        u, v = np.unravel_index(np.argmax(betweenness), edge_matrix.shape)
        edge_matrix[u][v] = edge_matrix[v][u] = 0
        return girvan_newman_algorithm(edge_matrix)
    components = girvan_newman_algorithm(edge_matrix)
    labeled_nodes = {}
    for label, component in enumerate(components):
        for node in component:
            labeled_nodes[node] = label
    return labeled_nodes
""","""import numpy as np
def label_nodes(edge_matrix):
    def compute_modularity(edge_matrix, communities):
        m = np.sum(edge_matrix) / 2
        modularity = 0.0
        for community in set(communities):
            nodes_in_community = [i for i, c in enumerate(communities) if c == community]
            internal_edges = sum(edge_matrix[i][j] for i in nodes_in_community for j in nodes_in_community)
            degree_sum = sum(np.sum(edge_matrix[i]) for i in nodes_in_community)
            expected_edges = degree_sum * degree_sum / (2 * m)
            modularity += internal_edges / (2 * m) - expected_edges / (2 * m)
        return modularity
    def louvain_algorithm(edge_matrix):
        num_nodes = len(edge_matrix)
        communities = list(range(num_nodes))
        modularity = -1
        while True:
            current_modularity = modularity
            for node in range(num_nodes):
                best_community = communities[node]
                best_modularity_increase = 0
                for neighbor in range(num_nodes):
                    if edge_matrix[node][neighbor] > 0:
                        original_community = communities[node]
                        communities[node] = communities[neighbor]
                        new_modularity = compute_modularity(edge_matrix, communities)
                        modularity_increase = new_modularity - current_modularity
                        if modularity_increase > best_modularity_increase:
                            best_community = communities[node]
                            best_modularity_increase = modularity_increase
                        communities[node] = original_community
                communities[node] = best_community
            modularity = compute_modularity(edge_matrix, communities)
            if modularity == current_modularity:
                break
        return communities
    communities = louvain_algorithm(edge_matrix)
    unique_communities = list(set(communities))
    community_mapping = {old: new for new, old in enumerate(unique_communities)}
    communities = [community_mapping[community] for community in communities]
    labeled_nodes = {node: label for node, label in enumerate(communities)}
    return labeled_nodes""","""import numpy as np
def label_nodes(edge_matrix):
    n = len(edge_matrix)
    labels = np.arange(n)
    scores = {i: i for i in range(n)}
    while True:
        new_labels = labels.copy()
        for i in range(n):
            neighbors = np.where(edge_matrix[i] == 1)[0]
            neighbor_labels = labels[neighbors]
            most_common_label = np.bincount(neighbor_labels).argmax()
            new_labels[i] = most_common_label
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return {i: labels[i] for i in range(n)}""","""import numpy as np
def label_nodes(edge_matrix):
    n = len(edge_matrix)
    D = np.diag(np.sum(edge_matrix, axis=1))
    L = D - edge_matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler_vector = eigvecs[:, 1]
    labels = (fiedler_vector > 0).astype(int)
    return {i: labels[i] for i in range(n)}""",
    """import numpy as np
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
    return {i: labels[i] for i in range(n)}""",
    """import numpy as np

def label_nodes(edge_matrix):
    n = len(edge_matrix)
    P = edge_matrix / np.sum(edge_matrix, axis=1, keepdims=True)
    labels = np.arange(n)
    while True:
        new_labels = labels.copy()
        for i in range(n):
            neighbors = np.where(edge_matrix[i] == 1)[0]
            neighbor_labels = labels[neighbors]
            label_counts = np.bincount(neighbor_labels)
            new_labels[i] = np.argmax(label_counts)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return {i: labels[i] for i in range(n)}""",
    """import numpy as np
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
    return {i: labels[i] for i in range(n)}""",
    """import numpy as np
def label_nodes(edge_matrix):
    n = len(edge_matrix)
    degrees = np.sum(edge_matrix, axis=1)
    total_edges = np.sum(degrees) / 2
    modularity_matrix = edge_matrix - np.outer(degrees, degrees) / (2 * total_edges)
    labels = np.arange(n)
    best_modularity = 0
    best_labels = labels.copy()
    def calculate_modularity(labels):
        modularity = 0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    modularity += modularity_matrix[i][j]
        return modularity / (2 * total_edges)
    def find_cliques():
        cliques = []
        for i in range(n):
            for j in range(i+1, n):
                if edge_matrix[i][j] == 1:
                    cliques.append([i, j])
        return cliques
    def expand_clique(clique):
        new_cliques = set(clique)
        for node in range(n):
            if all(edge_matrix[node][i] == 1 for i in clique):
                new_cliques.add(node)
        return list(new_cliques)
    def merge_cliques(cliques):
        merged = []
        visited = []
        for clique in cliques:
            added = False
            for i, merged_clique in enumerate(merged):
                if not set(clique).isdisjoint(merged_clique):
                    merged[i] = list(set(merged_clique).union(clique))
                    added = True
                    break
            if not added:
                merged.append(clique)
        return merged
    def merge_communities(labels, community1, community2):
        new_labels = labels.copy()
        new_labels[labels == community2] = community1
        return new_labels
    def get_best_merge(labels):
        best_merge_modularity = best_modularity
        best_community1, best_community2 = -1, -1
        for community1 in range(n):
            for community2 in range(community1 + 1, n):
                new_labels = merge_communities(labels, community1, community2)
                new_modularity = calculate_modularity(new_labels)
                if new_modularity > best_merge_modularity:
                    best_merge_modularity = new_modularity
                    best_community1, best_community2 = community1, community2
        return best_community1, best_community2, best_merge_modularity
    cliques = find_cliques()
    expanded_cliques = [expand_clique(clique) for clique in cliques]
    merged_cliques = merge_cliques(expanded_cliques)

    for i, clique in enumerate(merged_cliques):
        for node in clique:
            labels[node] = i
    best_modularity = calculate_modularity(labels)
    while True:
        community1, community2, new_modularity = get_best_merge(labels)
        if new_modularity > best_modularity:
            labels = merge_communities(labels, community1, community2)
            best_modularity = new_modularity
        else:
            break
    labeled_nodes = {i: labels[i] for i in range(n)}
    return labeled_nodes""",
    """import numpy as np
def label_nodes(edge_matrix, num_labels=3, max_iter=1000, learning_rate=0.01, regularization=0.1):
    n = len(edge_matrix)
    labels = np.zeros((n, num_labels))
    for i in range(n):
        labels[i, np.random.randint(0, num_labels)] = 1
    def calculate_loss(labels, W, H):
        predicted = np.dot(W, H.T)
        error = edge_matrix - predicted
        loss = np.sum(error ** 2) + regularization * (np.sum(W ** 2) + np.sum(H ** 2))
        return loss
    W = np.random.rand(n, num_labels)
    H = np.random.rand(n, num_labels)
    for _ in range(max_iter):
        for i in range(n):
            for j in range(n):
                if edge_matrix[i, j] > 0:
                    predicted_ij = np.dot(W[i, :], H[j, :].T)
                    error_ij = edge_matrix[i, j] - predicted_ij
                    for k in range(num_labels):
                        W[i, k] += learning_rate * (2 * error_ij * H[j, k] - 2 * regularization * W[i, k])
                        H[j, k] += learning_rate * (2 * error_ij * W[i, k] - 2 * regularization * H[j, k])
        loss = calculate_loss(labels, W, H)
        if loss < 1e-6:
            break
    label_ids = np.argmax(W, axis=1)
    return {i: label_ids[i] for i in range(n)}
""",
    """import numpy as np
def label_nodes(edge_matrix):
    n = len(edge_matrix)
    # Create Laplacian matrix
    D = np.diag(np.sum(edge_matrix, axis=1))
    L = D - edge_matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler_vector = eigvecs[:, 1]
    initial_labels = (fiedler_vector > 0).astype(int)
    def expand_community(community):
        new_community = set(community)
        while True:
            neighbors = set()
            for node in new_community:
                neighbors = neighbors.union(set(np.where(edge_matrix[node] == 1)[0]))
            neighbors = neighbors.difference(new_community)
            if not neighbors:
                break
            new_community = new_community.union(neighbors)
        return list(new_community)
    communities = [expand_community(np.where(initial_labels == i)[0]) for i in set(initial_labels)]
    def louvain_algorithm(edge_matrix, nodes):
        num_nodes = len(nodes)
        communities = list(range(num_nodes))
        modularity = -1
        while True:
            current_modularity = modularity
            for node in range(num_nodes):
                best_community = communities[node]
                best_modularity_increase = 0
                for neighbor in range(num_nodes):
                    if edge_matrix[node][neighbor] > 0:
                        original_community = communities[node]
                        communities[node] = communities[neighbor]
                        new_modularity = compute_modularity(edge_matrix, communities, nodes)
                        modularity_increase = new_modularity - current_modularity
                        if modularity_increase > best_modularity_increase:
                            best_community = communities[node]
                            best_modularity_increase = modularity_increase
                        communities[node] = original_community
                communities[node] = best_community
            modularity = compute_modularity(edge_matrix, communities, nodes)
            if modularity == current_modularity:
                break
        return communities
    def compute_modularity(edge_matrix, communities, nodes):
        m = np.sum(edge_matrix) / 2
        modularity = 0.0
        for community in set(communities):
            nodes_in_community = nodes[communities == community]
            internal_edges = sum(edge_matrix[i][j] for i in nodes_in_community for j in nodes_in_community)
            degree_sum = sum(np.sum(edge_matrix[i]) for i in nodes_in_community)
            expected_edges = degree_sum * degree_sum / (2 * m)
            modularity += internal_edges / (2 * m) - expected_edges / (2 * m)
        return modularity
    labeled_nodes = {}
    for i, community in enumerate(communities):
        if len(community) > 2:
            subgraph = edge_matrix[np.ix_(community, community)]
            subgraph_labels = louvain_algorithm(subgraph, np.arange(len(community)))
            for j, label in enumerate(subgraph_labels):
                labeled_nodes[community[j]] = i * 10 + label
        else:
            for node in community:
                labeled_nodes[node] = i
    return labeled_nodes""",
        ]