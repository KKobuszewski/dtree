#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source: http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""

from __future__ import print_function

import math
import numpy as np


def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini
 
# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('|%s[X%d < %.3f]' % ((depth*'--', (node['index']), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*'  ', node)))



if __name__ == '__main__': 
    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.812813570, 0],
               [3.961043357, 2.619950320, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.002203260, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]
    
    data = np.loadtxt('data_banknote/data_banknote_authentication.txt',delimiter=',')
    
    print(data.shape)
    print(data)
    
    index = 1
    print( data[index,0] )
    groups = test_split(index, data[index,0], data)
    #gini = gini_index(groups, class_values)
    
    
    tree = build_tree(data, 4, 1)
    print_tree(tree)