import networkx as nx
import numpy as np


class NodeFeatureExtractor:
    def __init__(self, graph):
        """
        Initializes the NodeFeatureExtractor with a given graph.

        Computes and normalizes the following features:
        - Node Degree
        - Clustering Coefficient
        - Betweenness Centrality
        - Closeness Centrality
        - Eigenvector Centrality

        Parameters:
        - graph: A NetworkX graph object.
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.node_list = list(graph.nodes())

        # Compute static features
        self.degree = self.compute_degree()
        self.clustering = self.compute_clustering()
        self.betweenness = self.compute_betweenness()
        self.closeness = self.compute_closeness()
        self.eigenvector = self.compute_eigenvector()

        # Normalize features
        self.degree = self.normalize_feature(self.degree)
        self.clustering = self.normalize_feature(self.clustering)
        self.betweenness = self.normalize_feature(self.betweenness)
        self.closeness = self.normalize_feature(self.closeness)
        self.eigenvector = self.normalize_feature(self.eigenvector)

        self.get_static_feature_matrix()

    def compute_degree(self):
        """
        Computes the degree for each node.

        Returns:
        - degree: A dictionary mapping node to degree.
        """
        degree_dict = dict(self.graph.degree())
        return degree_dict

    def compute_clustering(self):
        """
        Computes the clustering coefficient for each node.

        Returns:
        - clustering: A dictionary mapping node to clustering coefficient.
        """
        clustering_dict = nx.clustering(self.graph)
        return clustering_dict

    def compute_betweenness(self):
        """
        Computes the betweenness centrality for each node.

        Returns:
        - betweenness: A dictionary mapping node to betweenness centrality.
        """
        betweenness_dict = nx.betweenness_centrality(self.graph, normalized=True)
        return betweenness_dict

    def compute_closeness(self):
        """
        Computes the closeness centrality for each node.

        Returns:
        - closeness: A dictionary mapping node to closeness centrality.
        """
        closeness_dict = nx.closeness_centrality(self.graph)
        return closeness_dict

    def compute_eigenvector(self):
        """
        Computes the eigenvector centrality for each node.

        Returns:
        - eigenvector: A dictionary mapping node to eigenvector centrality.
        """
        eigenvector_dict = nx.eigenvector_centrality(self.graph, max_iter=1000)
        return eigenvector_dict

    def normalize_feature(self, feature_dict):
        """
        Normalizes a feature dictionary to have values between 0 and 1.

        Parameters:
        - feature_dict: A dictionary mapping node to feature value.

        Returns:
        - normalized_dict: A dictionary mapping node to normalized feature value between 0 and 1.
        """
        values = np.array(list(feature_dict.values()))
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val
        if range_val == 0:
            # All values are the same
            normalized_dict = {node: 0.0 for node in feature_dict}
        else:
            normalized_dict = {
                node: (value - min_val) / range_val
                for node, value in feature_dict.items()
            }
        return normalized_dict

    def get_static_feature_matrix(self):
        """
        Returns a numpy array of shape (num_nodes, num_static_features) containing the static features.

        Returns:
        - static_features: A numpy array of shape (num_nodes, num_static_features)
        """
        static_features = []
        for node in self.node_list:
            feature_vector = [
                self.degree[node],
                self.clustering[node],
                self.betweenness[node],
                self.closeness[node],
                self.eigenvector[node],
            ]
            static_features.append(feature_vector)
        self.static_features = np.array(
            static_features
        )  # Shape (num_nodes, num_static_features)

    def get_features(
        self,
    ):
        """
        Given a batch of evidence node lists, returns the features for nodes as a numpy array.

        Parameters:
        - evidence_batch: A list of lists, where each inner list contains the evidence node indices for an example.

        Returns:
        - features_batch: A numpy array of shape (num_examples, num_nodes, num_features)
        """
        num_nodes = self.num_nodes
        num_static_features = self.static_features.shape[
            1
        ]  # degree, clustering, betweenness, closeness, eigenvector
        num_features = num_static_features

        static_features = self.static_features  # Shape (num_nodes, num_static_features)

        # Prepare the features_batch array
        features_batch = np.zeros((num_nodes, num_features))
        features_batch[:, :num_static_features] = static_features
        return features_batch
