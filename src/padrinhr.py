from io import StringIO
import json
from random import randint
import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MatchMaker:
    def __init__(self):
        self.columns = [
            {
                'name': 'id',
                'type': 'uuid',
            }, 
            {
                'name': 'role',
                'type': 'role',
            },
            {
                'name': 'course',
                'type': 'binary',
                'weight': 3
            },
            # outros atributos nesse formato foram usados no apadrinhamento 
        ]

        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.df = None
        self.topic_to_cols = None
        self.role_col = None
        self.curso_col = None
        self.nome_col = None
        self.par_idx = None
        self.chi_idx = None
        self.edges = None
        self.col_to_weights = None
        self.max_connections = 2

    def import_data(self, data):
        self.df = pd.read_json(data)
                
        self.par_idx = self.df[self.df['role'] == 'veterane'].index.to_list()
        self.chi_idx = self.df.index.difference(self.par_idx).to_list()

    def normalize_func(self, row):
        if np.sum(row) == 0:
            return row
        return row / np.sum(row)

    def normalize_rows(self, mat):
        normalized_mat = np.apply_along_axis(self.normalize_func, axis=1, arr=mat)
        return normalized_mat

    def get_embed_weight(self, chi_resp, par_resp):
        par_embed = self.embedder.encode(par_resp)
        chi_embed = self.embedder.encode(chi_resp)
        weight_mat = cosine_similarity(chi_embed, par_embed)
    
        return weight_mat

    def num_func(self, x1, x2):
        return 1 - (x1 - x2)**2

    def get_num_weight(self, chi_resp, par_resp, scale=10):
        par_numeric = np.array([int(x) for x in par_resp])
        chi_numeric = np.array([int(x) for x in chi_resp])
        par_numeric = par_numeric / scale
        chi_numeric = chi_numeric / scale

        weight_mat = np.zeros((len(chi_numeric), len(par_numeric)))
        for i, x1 in enumerate(chi_numeric):
            weight_mat[i, :] = np.vectorize(self.num_func)(x1, par_numeric)
        return weight_mat

    def get_binary_weight(self, chi_resp, par_resp):
        weight_mat = np.zeros((len(chi_resp), len(par_resp)))
        for i, chi in enumerate(chi_resp):
            for j, par in enumerate(par_resp):
                weight_mat[i, j] = chi == par
        return weight_mat

    def get_multiple_choice_weight(self, chi_resp, par_resp): 
        weight_mat = np.zeros((len(chi_resp), len(par_resp)))
    
        par_resp = [x for x in par_resp]
        chi_resp = [x for x in chi_resp]
        
        for i, chi in enumerate(chi_resp):
            for j, par in enumerate(par_resp):
                weight_mat[i, j] = any(element in chi for element in par)
        return weight_mat

    def handle_topic_weights(self):
        col_to_weights = {}
        for col in self.columns:
            if col['type'] in ['role', 'uuid']:
                continue
            
            if col['type'] == 'binary':
                weight_func = self.get_binary_weight
            elif col['type'] == 'embedding':
                weight_func = self.get_embed_weight
            elif col['type'] == 'numeric':
                weight_func = self.get_num_weight
            elif col['type'] == 'multiple_choice':
                weight_func = self.get_multiple_choice_weight
  
            par_resp = self.df.loc[self.par_idx, col['name']].to_numpy()
            chi_resp = self.df.loc[self.chi_idx, col['name']].to_numpy()

            weight_mat = weight_func(chi_resp, par_resp)
            weight_mat = self.normalize_rows(weight_mat)
            col_to_weights[col['name']] = weight_mat

        return col_to_weights

    def distribute_random(self, n, max_n, max_chi_par):
        distribution = np.ones(n, dtype=int)
        diff = max_n - n
        available_indices = set(range(n))
        while diff > 0:
            idx = np.random.choice(list(available_indices))
            distribution[idx] += 1
            diff -= 1
            if distribution[idx] == max_chi_par:
                available_indices.remove(idx)
        return distribution

    def build_graph_and_match(self):
        self.col_to_weights = self.handle_topic_weights()

        edges = np.zeros((len(self.chi_idx), len(self.par_idx)))
        weight_sum = 0
        for col in self.columns:
            if col['type'] in ['role', 'uuid']:
                continue

            weight_sum += int(col['weight'])
            edges += self.col_to_weights[col['name']] * int(col['weight'])

        self.edges = edges / weight_sum

        par_to_mat = {idx: i for i, idx in enumerate(self.par_idx)}
        chi_to_mat = {idx: i for i, idx in enumerate(self.chi_idx)}

        n_chi = len(self.chi_idx)
        n_par = len(self.par_idx)
        max_matches = min(n_chi, n_par) * self.max_connections

        chi_dist = self.distribute_random(n_chi, max_matches, self.max_connections)
        par_dist = self.distribute_random(n_par, max_matches, self.max_connections)

        chi_ids = np.array([f"{str(self.chi_idx[i])}_{j}"
                            for i in range(n_chi)
                            for j in range(chi_dist[i])])
        par_ids = np.array([f"{str(self.par_idx[i])}_{j}"
                            for i in range(n_par)
                            for j in range(par_dist[i])])

        G = nx.Graph()
        G.add_nodes_from(chi_ids, bipartite=0)
        G.add_nodes_from(par_ids, bipartite=1)

        for chi in range(n_chi):
            for par in range(n_par):
                i = chi_to_mat[self.chi_idx[chi]]
                j = par_to_mat[self.par_idx[par]]
                weight = self.edges[i, j]

                rand_chi = randint(0, chi_dist[chi] - 1)
                rand_par = randint(0, par_dist[par] - 1)

                for k in range(chi_dist[chi]):
                    if k == rand_chi:
                        continue
                    for l in range(par_dist[par]):
                        if l == rand_par:
                            continue
                        G.add_edge(f"{self.chi_idx[chi]}_{k}", f"{self.par_idx[par]}_{l}", weight=0)

                G.add_edge(f"{self.chi_idx[chi]}_{rand_chi}", f"{self.par_idx[par]}_{rand_par}", weight=weight)

        matching = nx.matching.max_weight_matching(G, maxcardinality=True)
        return matching

    def perform_matching(self):
        matching = self.build_graph_and_match()

        matched = {}

        for p1_idx, p2_idx in matching:
            p1_uuid = self.df.loc[int(p1_idx.split('_')[0]), 'id']
            p2_uuid = self.df.loc[int(p2_idx.split('_')[0]), 'id']

            if self.df.loc[int(p1_idx.split('_')[0]), 'role'] == 'bixe':
                p1_uuid, p2_uuid = p2_uuid, p1_uuid

            if not matched.get(f"{p2_uuid}"):
                matched[f"{p2_uuid}"] = [f"{p1_uuid}"]
            else:
                matched[f"{p2_uuid}"].append(f"{p1_uuid}")

        return json.dumps(matched)

if __name__ == "__main__":
    with open('data.json', 'r') as f:
        data = StringIO(f.read())
    
    matcher = MatchMaker()
    matcher.import_data(data)
    with open('matches.json', 'w') as f:
        f.write(matcher.perform_matching())