import numpy as np
from binarytree import build


class Node():
    def __init__(self, name=None, feature=None, threshold=None, left=None, right=None, value=None):
        # decision node
        self.name = name
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # leaf node
        self.value = value

    def __str__(self):
        return f'Node {self.name}, feature {self.feature}, threshold {self.threshold}'

class OptimalTree():
    def __init__(self, non_empty_nodes,splitting_nodes,depth,SplitType='Parallel',ModelTree=True,classes=[]):
        self.non_emtpy_nodes = non_empty_nodes
        self.splitting_nodes = splitting_nodes
        self.depth = depth
        self.SplitType = SplitType
        self.ModelTree = ModelTree
        self.nodes = [i for i in range(2 ** (depth + 1) - 1)]
        self.classes = classes

        self.complete_tree = build(self.nodes)
        self.T_L = [i.value for i in self.complete_tree.leaves]  # leave nodes
        self.T_B = [i for i in self.complete_tree.values if i not in self.T_L]
        self.root = self.complete_tree.levels[0][0]

    def build_tree(self, current_node):

        if current_node in self.splitting_nodes:
            left_subtree = self.build_tree( Children(self.root,current_node)[0].value)
            right_subtree = self.build_tree(Children(self.root, current_node)[1].value)
            return Node(
                current_node,
                self.splitting_nodes[current_node]['a'],
                self.splitting_nodes[current_node]['b'],
                left_subtree,
                right_subtree
            )
        elif current_node in self.non_emtpy_nodes:
            return Node(
                current_node,
                value = self.non_emtpy_nodes[current_node]
            )
        else:
            descendants = [i for i in self.non_emtpy_nodes if current_node in Ancestors(self.root,i)]
            if len(descendants) > 0:
                return Node(
                    current_node,
                    value=self.non_emtpy_nodes[descendants[0]]
                )
            else:
                raise ValueError('THE TREE HAS TO MANY SPLITS')

    def print_tree(self,tree, indent=" "):

        if tree is not None:
            print('Node',tree.name)
            if tree.value is not None:
                print(tree.value)
            else:
                print(f'{str(tree.feature)} < {tree.threshold} ')
                print(f'{indent}left:', end="")
                self.print_tree(tree.left, indent + indent)
                print('%sright:' % (indent), end="")
                self.print_tree(tree.right, indent + indent)
        else:
            print('No Node')

    def predict_class(self, X, tree):
        '''function to predict_regr a new dataset'''
        predictions = [self.make_classification(value, tree) for key,value in X.items()]
        return predictions

    def make_classification(self, x, tree):
        '''function to make a single prediction'''
        if self.ModelTree:
            if tree.value != None:
                scores = [
                    1 if sum([tree.value[c]['Beta'][f] * x[f] for f in x]) + tree.value[c]['Delta'] > 0 else 0
                    for c in self.classes
                ]
                return scores

            if tree.value != None:
                return tree.value
        if self.SplitType == 'Parallel':
            if x[tree.feature] < tree.threshold:
                return self.make_classification(x, tree.left)
            else:
                return self.make_classification(x, tree.right)
        elif self.SplitType == 'Oblique':
            if sum([x[key] * value for key, value in tree.feature.items()]) < tree.threshold:
                return self.make_classification(x, tree.left)
            else:
                return self.make_classification(x, tree.right)

# Some additional functions needed to deal with binary trees

def Ancestors(root, target):
    ancestors = []

    def findAncestors(root, target):
        # Base case
        if root == None:
            return False

        if root.value == target:
            return True

        # If target is present in either left or right subtree
        # of this node, then print this node
        if (findAncestors(root.left, target) or
                findAncestors(root.right, target)):
            ancestors.append(root.value)
            # print(root.value,end=' ')
            return True

        # Else return False
        return False

    findAncestors(root, target)
    return ancestors

def Parent(node, val):
    the_parent = []

    def findParent(node, val, parent=None):
        if (node is None):
            return

            # If current node is the required node
        if (node.value == val):
            # assign its parent
            the_parent.append(parent)

        else:
            # Recursive calls for the children of the current node. current node is now the new parent
            findParent(node.left, val, node.value)
            findParent(node.right, val, node.value)

    findParent(node, val)
    return the_parent[0]

def Children(node, val):
    children = {}
    def findChildren(node, val):
        if (node is None):
            return

        # # If current node is the required node
        if (node.value == val):
            # assign its parent
            children.update({'left':node.left,'right':node.right})

        else:
            # Recursive calls for the children of the current node. current node is now the new parent
            findChildren(node.left, val)
            findChildren(node.right, val)

    findChildren(node, val)
    return children['left'],children['right']

def RAE(Y_labels,Y_predicted):
    mean_Y = np.average(Y_labels)
    numerator = sum([ abs(i-j) for i,j in zip(Y_predicted,Y_labels)])
    denominator = sum([abs(mean_Y - j) for j in Y_labels])
    return round(numerator/denominator,2)

def RRSE(Y_labels,Y_predicted):
    mean_Y = np.average(Y_labels)
    numerator = sum([ (i-j)**2 for i,j in zip(Y_predicted,Y_labels)])
    denominator = sum([ (mean_Y - j)**2 for j in Y_labels])
    return round(np.sqrt(numerator/denominator),2)

def Multiplier(vector):
    theVec = []
    for i in vector:
        if '.' in str(i):
            theVec.append(len(str(i).split('.')[1]))
        else:
            theVec.append(0)
    return 10**max(theVec)

def MultiLabelAcc(y_labels,y_predicted):
    correct = {
        d:[y_labels[d][i] == y_predicted[d][i] for i in y_predicted[d]]
        for d in y_labels
    }
    final_count = 0
    for i in correct.values():
        if all(i):
            final_count += 1

    return round(final_count/len(y_predicted)*100,2)