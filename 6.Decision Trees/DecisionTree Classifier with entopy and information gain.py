import math
import numpy as np
from graphviz import Digraph

class DecisionTreeClassifier:
    def __init__(self, max_depth = 5):
        self.max_depth = max_depth
        self.root = None
    
    # Class to represent the node of decision tree
    class Node:
        def __init__(self, feature=None, feature_index=None, thresholds = None, children=None, label=None):
            # decision node
            self.feature = feature  # Feature name used for splitting
            self.feature_index = feature_index # Feature index
            self.threshold = thresholds # label of the decision node
            self.children = children # children nodes 
            # leaf node
            self.label = label # class label

        def is_leaf(self):
            return self.label is not None
        
    # implement entropy function from scratch 
    def entropy(self, classes_values): 
        Entropy_value, length = 0, np.sum(classes_values)
        # Find proportion of positive and negative classes
        for class_value in classes_values:
            P = class_value/(length)
            Entropy_value += -P * math.log2(P)
        return Entropy_value
    # implement average entropy function from scratch 
    def average_entropy(self, feature_name, X, Y):
        S_length = len(Y)
        feature_values = X[feature_name].unique() # all different values
        average_entropy = 0
        for value in feature_values:
            classes_values = Y[X[feature_name] == value].value_counts() # yes_values, No_values
            value_entropy = self.entropy(classes_values)
            value_length = classes_values.sum()
            average_entropy += (value_entropy * value_length / S_length)
        return average_entropy
    # implement information gain function from scratch 
    def information_gain(self, S_entropy,average_entropy):
        return S_entropy - average_entropy
    
    # Find best feature with max gain
    def find_best_feature(self, X, Y):
        max_gain = 0
        best_feature = None
        S_values = Y.value_counts() # yes_values: k , No_values: g
        S_gain = self.entropy(S_values)
        for feature_name in X.columns:
            average_entropy = self.average_entropy(feature_name, X, Y)
            info_gain = self.information_gain(S_gain, average_entropy)
            if info_gain > max_gain:
                max_gain = info_gain
                best_feature = feature_name
        return best_feature
    # implement build_tree function from scratch
    def build_tree(self, X, Y, depth=0):
        # We reached the max depth or pure Node 
        if depth == self.max_depth or len(Y.unique()) == 1: 
            return self.Node(label = Y.value_counts().idxmax()) # Create leaf Node
        # Find the best feature
        best_feature = self.find_best_feature(X, Y) 
        best_feature_thresholds = X[best_feature].unique()
        children = []
        for threshold in best_feature_thresholds:
            child_x = X[X[best_feature] == threshold] # Find X values of the child
            child_y = Y[X[best_feature] == threshold] # Find Y values of the child
            child = self.build_tree(child_x, child_y, depth + 1) # Create child tree
            children.append(child) # Add child to the parent's list

        return self.Node(feature=best_feature, 
                        feature_index=X.columns.get_loc(best_feature),
                        thresholds = best_feature_thresholds, 
                        children=children)

    def fit(self, x, y):
        self.root = self.build_tree(x,y)
    
    # Make predictions
    def predict(self, X):
        predictions = []
        for x in X:
            child = self.root
            while(not child.is_leaf()):
                feature_index = child.feature_index
                value = x[feature_index]
                for threshold_index in range(len(child.threshold)):
                    if  value == child.threshold[threshold_index]:
                        child = child.children[threshold_index]
                        break
            predictions.append(child.label)
        return predictions

    def print(self, Node=None):
        if not Node:
            Node  = self.root
        if Node.is_leaf():
            print(Node.label)
        else:
            print("the feature name :", Node.feature,"the thresholds :", Node.threshold)
            for node in Node.children:
                self.print(node)

    # finally plot the tree you have implemented
    def plot(self):
        dot = Digraph()
        def add_nodes_edges(node):
            if node.label:
                dot.node(str(node), str(node.label))
            else:
                dot.node(str(node), str(node.feature))
                for child_index in range(len(node.children)):
                    dot.edge(str(node), str(node.children[child_index]), label=node.threshold[child_index])
                    add_nodes_edges(node.children[child_index])
        add_nodes_edges(self.root)
        dot.render('decision_tree', format='png', cleanup=True)
    


