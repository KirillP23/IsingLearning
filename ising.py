import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

RANDOM_SEED = 44
data_path = 'data/'


def load_data(dataset_str):
    """
    Load data.
    Two graphs available: 'cora' and 'pubmed'.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))

    return {'adj': adj, 'features': features, 'labels': labels}


def normalize_adj(adj):
    """Normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype=float)
    d_inv = np.power(rowsum, -1.).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return adj.dot(d_mat_inv).transpose().tocoo()


def prepare_labels(labels, class_mask):
    """
    Reduce the number of classes to two ('-1' and '+1') by masking.
    Mask is a 1d boolean array of the same dimension as labels.
    """
    mask = [1 if elt else -1 for elt in class_mask]
    mask = np.array(mask)
    return labels.dot(mask)


def prepare_features(features, labels, adj, symmetrize=True,
                     max_neigh_sample=None):
    """
    Append neighborhood features to each vertex.
    If symmetrize is set to True, neighborhood features are:
        symmetric sum = sum(|X_0 + X_i|) / (number of neighbors)
        symmetric diviation = sum(|X_0 - X_i|) / (number of neighbors)
    where the sum is taken over all neighbors of v_0 if 'max_neigh_sample'
    is None, and over a random neighborhood sample of size 'k' if
    'max_neigh_sample' is 'k'.
    If symmetrize is set to True, the output features have form
    (features, symmetric sum, symmetric diviation) corresponding to each
    vertex.
    Labels must be '+1' or '-1' if known and '0' if unknown.
    """
    assert features.shape[0] == labels.shape[0] == adj.shape[0]

    if not symmetrize:
        adj = normalize_adj(adj)
        label_diag = sp.diags(labels)
        neigh_feat = label_diag * adj * features
        return (features, neigh_feat)

    samples_num = features.shape[0]
    feature_num = features.shape[1]
    neigh_feat_list = []

    for vert in range(samples_num):
        neighs = adj[vert].nonzero()[1]
        neighs = [neigh for neigh in neighs if labels[neigh] != 0]
        if len(neighs) == 0:
            neigh_feat_all = sp.coo_matrix((1, feature_num*2),
                                           dtype=np.float32)
            neigh_feat_list.append(neigh_feat_all)
            continue
        if max_neigh_sample and len(neighs) > max_neigh_sample:
            neighs = np.random.choice(neighs, max_neigh_sample)
        vert_feat = features[vert].toarray()[0]

        neigh_feat_sum = sp.coo_matrix(
            np.sum(
                len(neighs)**(-1) * labels[neighs]
                * np.abs(features[neighs] + vert_feat),
                axis=0)
        )
        neigh_feat_div = sp.coo_matrix(
            np.sum(
                len(neighs)**(-1) * labels[neighs]
                * np.abs(features[neighs] - vert_feat),
                axis=0)
        )

        neigh_feat_all = sp.hstack([neigh_feat_sum, neigh_feat_div])
        neigh_feat_list.append(neigh_feat_all)
    neigh_feat = sp.vstack(neigh_feat_list)
    return sp.hstack([features, neigh_feat])


class IsingLogReg():
    """
    Class for producing an Ising model weight matrix given vertex features,
    known labels and adjacency matrix.
    Parameters:
        -- "positive"- forces the ising weights to be positive by dropping all
        negative coefficients in logistic regression
        -- "ising"- if False, doesn't use any neighborhood features to fit
        -- "symmetrize"- constructs symmetrized neighborhood features
        -- "max_neigh_sample"- the maximum number of neighbors used to construct
        neighborhood features
    """

    def __init__(self, features, labels, adj,
                 ising=True, symmetrize=True,
                 max_neigh_sample=None, positive=True):
        """
        Label vector has values +1 or -1 if label is in the training set,
        and 0 if label is in the test set.
        """
        self.features = features
        self.labels = labels
        self.adj = adj
        self.ising = ising
        self.symmetrize = symmetrize
        self.max_neigh_sample = max_neigh_sample
        self.positive = positive

    def train(self, train_split=0.5, C_grid=2.**np.arange(6, 16)):
        """
        If 'ising' set to True, prepare the features and fit neighborhood
        features to the known labels using logistic regression with lasso
        penalty. If 'ising' set to False, use only the vertex features to fit
        the regression.
        Set attribute 'self.coeffs' to find ising weights in
        'self.generate_ising_weights_()', and set attribute
        'self.ising_weights' to be used in Ising model.
        'train_split' determines the proportion of known labels fo be used for
        fitting.
        """
        if self.ising:
            features = prepare_features(
                self.features, self.labels, self.adj, symmetrize=self.symmetrize,
                max_neigh_sample=self.max_neigh_sample).tolil()
        else:
            features = self.features

        train_set = (self.labels != 0)
        labels_train = self.labels[train_set]
        features_train = features[train_set]
        split = train_test_split(features_train, labels_train,
                                 test_size=1-train_split,
                                 random_state=RANDOM_SEED)
        feat_train, feat_test, labels_train, labels_test = split

        logreg = LogisticRegression(penalty='l1')
        param_grid = [{'C': C_grid}]

        logreg_search = GridSearchCV(logreg, param_grid, cv=5,
                                     scoring='neg_log_loss')
        logreg_search.fit(feat_train, labels_train)

        logreg_best = logreg_search.best_estimator_
        coeffs = logreg_best.coef_[0]
        intercept = logreg_best.intercept_
        feat_num = self.features.shape[1]
        if not self.ising:
            assert len(coeffs) == feat_num
            self.coeffs = {'raw': coeffs, 'intercept': intercept}
            weight_vertices = (self.features.dot(self.coeffs['raw'])
                               + self.coeffs['intercept'])
            self.ising_weights = {'vertices': weight_vertices}
            return self

        assert len(coeffs) == feat_num*3
        raw_cfs = coeffs[:feat_num]
        sum_cfs = coeffs[feat_num:feat_num*2]
        div_cfs = coeffs[feat_num*2:]

        if self.positive:
            sum_cfs = np.where(sum_cfs > 0, sum_cfs, 0)
            div_cfs = np.where(sum_cfs > 0, div_cfs, 0)

        self.coeffs = {'raw': raw_cfs,
                       'sum': sum_cfs,
                       'div': div_cfs,
                       'intercept': intercept}
        self.ising_weights = self.generate_ising_weights_()
        return self

    def generate_ising_weights_(self):
        vert_num = self.adj.shape[0]
        weight_edges = sp.lil_matrix((vert_num, vert_num))
        sum_cfs = self.coeffs['sum']
        div_cfs = self.coeffs['div']
        for v1, v2 in np.vstack(adj.nonzero()).transpose():
            weight_edges[v1, v2] = (
                np.abs(self.features[v1]+self.features[v2]).dot(sum_cfs)
                + np.abs(self.features[v1]-self.features[v2]).dot(div_cfs)
            )
        weight_vertices = (self.features.dot(self.coeffs['raw'])
                           + self.coeffs['intercept'])
        return {'edges': weight_edges, 'vertices': weight_vertices}

    def predict(self):
        """
        Use pseudo-likelihood and ising weights to predict the labels.
        """
        train_array = np.array(self.labels != 0, dtype=float)
        if not self.ising:
            labels_logit = self.ising_weights['vertices']
        else:
            neigh_num = self.adj.dot(train_array)
            neigh_num = np.where(neigh_num == 0, 1, neigh_num)
            neigh_weights = self.ising_weights['edges'] * self.labels
            labels_logit = (np.multiply(neigh_weights, neigh_num**(-1))
                            + self.ising_weights['vertices'])
        self.prediction = np.where(labels_logit > 0, 1, -1)
        return self

    def score(self, true_labels):
        test_set = (self.labels == 0)
        test_size = sum(test_set)
        if test_size == 0:
            return 1
        self.predict()
        test_matches = sum(true_labels[test_set] == self.prediction[test_set])
        return test_matches * float(test_size)**(-1)


features, labels_raw, adj = tuple(load_data('cora')[name]
                                  for name in ['features', 'labels', 'adj'])

class_mask = [0, 0, 0, 1, 1, 0, 0]
labels = prepare_labels(labels_raw, class_mask)

test_size = 0.5
random_array = np.random.rand(len(labels))
train_labels = np.where(random_array > test_size, labels, 0)

fitter = IsingLogReg(features, train_labels, adj, positive=True, ising=True)
fitter.train(train_split=0.3)
print(fitter.score(labels))
