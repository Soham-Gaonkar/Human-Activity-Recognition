"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union
from graphviz import Digraph
from IPython.display import Image, display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionNode:
    attribute: str
    value: float
    left: "DecisionNode"
    right: "DecisionNode"
    is_leaf: bool
    output: Union[str, float]
    criterion_pair: tuple
    gain: float

    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, criterion_pair=None, gain=0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.criterion_pair = criterion_pair
        self.gain = gain
    
    def is_leaf_node(self):
        return self.is_leaf



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None


    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        def build_tree(X: pd.DataFrame, y: pd.Series, depth: int) -> DecisionNode:

            my_criterion, criterion_func = check_criteria(y, self.criterion)
            criterion_value = criterion_func(y)
            criterion_pair = (my_criterion, criterion_value)     

            # If the depth exceeds max_depth or all the targe valaues are same, create a leaf node.
            if depth >= self.max_depth or y.nunique() == 1:
                if check_ifreal(y):
                    return DecisionNode(is_leaf=True, output=np.round(y.mean(),4), criterion_pair=criterion_pair)
                else:
                    return DecisionNode(is_leaf=True, output=y.mode()[0], criterion_pair=criterion_pair)
            
            best_attribute = opt_split_attribute(X, y, X.columns, self.criterion)
            # If no attribute is found to split upon, create a leaf node.
            if best_attribute is None:
                if check_ifreal(y):
                    return DecisionNode(is_leaf=True, output=np.round(y.mean(),4), criterion_pair=criterion_pair)
                else:
                    return DecisionNode(is_leaf=True, output=y.mode()[0], criterion_pair=criterion_pair)

            if check_ifreal(X[best_attribute]):
                best_value = find_optimal_threshold(y, X[best_attribute], self.criterion)
                # If no valid threshold is found, create a leaf node.
                if best_value is None:
                    if check_ifreal(y):
                        return DecisionNode(is_leaf=True, output=np.round(y.mean(),4), criterion_pair=criterion_pair)
                    else:
                        return DecisionNode(is_leaf=True, output=y.mode()[0], criterion_pair=criterion_pair)
            else:
                best_value = X[best_attribute].mode()[0]

            X_left, y_left, X_right, y_right = split_data(X, y, best_attribute, best_value)
            # If a valid split is not possible, create a leaf node.
            if X_left.empty or X_right.empty:
                if check_ifreal(y):
                    return DecisionNode(is_leaf=True, output=np.round(y.mean(),4), criterion_pair=criterion_pair)
                else:
                    return DecisionNode(is_leaf=True, output=y.mode()[0], criterion_pair=criterion_pair)
                
            left_subtree = build_tree(X_left, y_left, depth + 1)
            right_subtree = build_tree(X_right, y_right, depth + 1)

            return DecisionNode(attribute=best_attribute, value=best_value, left=left_subtree, right=right_subtree, criterion_pair=criterion_pair, gain=information_gain(y, X[best_attribute], self.criterion))

        self.tree = build_tree(X, y, depth)
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        def predict_row(x: pd.Series) -> float:
            """
            Function to predict the output for a single row of input
            """
            current_node = self.tree
            while not current_node.is_leaf_node():
                if check_ifreal(x[current_node.attribute]):
                    if x[current_node.attribute] <= current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                else:
                    if x[current_node.attribute] == current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
            return current_node.output 
        return pd.Series([predict_row(x) for _, x in X.iterrows()])


    def plot(self, path=None) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        
        if not self.tree:
            print("Tree not trained yet")
            return
        
        dot = Digraph()

        def add_node(node: DecisionNode, parent_name: str = None, edge_label: str = None) -> None:
            node_id = str(id(node))
            if node.is_leaf:
                node_label = f"Prediction: {node.output}\n {node.criterion_pair[0]} = {node.criterion_pair[1]:.4f}"
            else:
                node_label = f"?(attr {node.attribute} <= {node.value:.2f})\n {node.criterion_pair[0]} = {node.criterion_pair[1]:.4f}"
            dot.node(node_id, label=node_label, shape='box' if node.is_leaf else 'ellipse')
            if parent_name:
                dot.edge(parent_name, node_id, label=edge_label)

            if node.left:
                add_node(node.left, node_id, 'Yes')
            if node.right:
                add_node(node.right, node_id, 'No')

        add_node(self.tree)

        print("\nDecision Tree Structure:")
        print(self.print_tree())
        if path:
            dot.render(path, format="png", view=False, cleanup=True)
            display(Image(filename=f"{path}.png"))  
        else:
            png_data = dot.pipe(format='png')
            display(Image(data=png_data))
            
    
    def print_tree(self) -> str:
        def print_node(node: DecisionNode, indent: str = '') -> str:
            output = ''
            if node.is_leaf:
                output += f'Class: {node.output}\n'
            else:
                output += f'?(attr {node.attribute} <= {node.value:.2f})\n'
                output += indent + '    Yes: '
                output += print_node(node.left, indent + '    ')
                output += indent + '    No: '
                output += print_node(node.right, indent + '    ')
            return output

        if not self.tree:
            return "Tree not trained yet"
        else:
            return print_node(self.tree)


    def __repr__(self):
        return f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth})\n\nTree Structure:\n{self.print_tree()}"