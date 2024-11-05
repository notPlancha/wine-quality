from abc import ABC, abstractmethod

class Model(ABC):
  def __init__(self):
    self.trained: bool = False
  @abstractmethod
  def train(self):
    self.trained = True
  @abstractmethod
  def fit(self):
    assert self.trained


"""
Tree structured classifiers, or, more correctly, binary tree structured classifiers, are constructed by
repeated splits of subsets of X into two descendant subsets, beginning with X itself.

The entire construction of a tree, then, revolves around three elements:
1. The selection of the splits
2. The decisions when to declare a node terminal or to continue splitting it
3. The assignment of each terminal node to a class

1. 
This idea of finding splits of nodes so as to give “purer” descendant nodes was implemented in this
way: (for example 3 classes) (t is the node)
1. Define the node proportions of the target p(1|t), p(2|t), p(3|t)
2. Define a measure i(t) of the impurity of the node as a function φ
such that φ(1/3,1/3,1/3) = maximum, φ (1, 0, 0) = 0, φ (0, 1, 0) = 0, φ (0,0, 1) = 0 (for 3 classes for example).
How pure a node is is defined by how little i(t) is
3. Define a candidate set S of binary splits s at each node.


2. 
To terminate the tree growing, a heuristic rule was designed. When a node t was reached such that
no significant decrease in impurity was possible, then t was not split and became a terminal node.

3. 
The class character of a terminal node was determined by the plurality rule.

tldr:
The four elements needed in the initial tree growing procedure were
1. A set Q of binary questions of the form {Is x ∈ A?}, A ⊂ X
2. A goodness of split criterion φ(s, t) that can be evaluated for any split s of any node t
3. A stop-splitting rule
4. A rule for assigning every terminal node to a class

the split selected is the one that maximazes φ.

The standardized set of questions Q is defined as follows:
- Each split depends on the value of only a single variable m.
- If the variable m is ordered or numerical, Q includes all questions in the form "Is x_m <= c" 
for all values c in the variable m 
(for example, if pH are 1, 2, 3, 4, the questions in Q are "is pH <= 1", "is pH <= 2", etc)
- If the variable is categorical, the Q includes all questions in "is it 1, is it 2, etc"

stopped at 2.4.2 The Splitting and Stop-Splitting Rule (o livro esta no droå)
"""
class CART(Model):
  pass