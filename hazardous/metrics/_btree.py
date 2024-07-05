# -*- coding: utf-8 -*-
from collections import de

import numpy as np


class _BTree:

    """A simple balanced binary order statistic tree to help compute the concordance.

    When computing the concordance, we know all the values the tree will ever contain. That
    condition simplifies this tree a lot. It means that instead of crazy AVL/red-black shenanigans
    we can simply do the following:

    - Store the final tree in flattened form in an array (so node i's children are 2i+1, 2i+2)
    - Additionally, store the current size of each subtree in another array with the same indices
    - To insert a value, just find its index, increment the size of the subtree at that index and
      propagate
    - To get the rank of an element, you add up a bunch of subtree counts
    """

    def __init__(self, nodes, weights):
        """
        Parameters
        ----------
        values: list
            List of sorted (ascending), unique values that will be inserted.
        """
        self._tree = self._treeify(nodes)
        # self._counts = np.zeros_like(self._tree, dtype=int)
        self._weight_indices = [[]] * len(self._tree)
        self._weights = weights

    @staticmethod
    def _treeify(values):
        """Convert the np.ndarray `values` into a complete balanced tree.

        Assumes `values` is sorted ascending. Returns a list `t` of the same length in which t[i] >
        t[2i+1] and t[i] < t[2i+2] for all i."""
        if len(values) == 1:  # this case causes problems later
            return values
        tree = np.empty_like(values)
        # Tree indices work as follows:
        # 0 is the root
        # 2n+1 is the left child of n
        # 2n+2 is the right child of n
        # So we now rearrange `values` into that format...

        # The first step is to remove the bottom row of leaves, which might not be exactly full
        last_full_row = int(np.log2(len(values) + 1) - 1)
        len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
        if len_ragged_row > 0:
            bottom_row_ix = np.s_[: 2 * len_ragged_row : 2]
            tree[-len_ragged_row:] = values[bottom_row_ix]
            values = np.delete(values, bottom_row_ix)

        # Now `values` is length 2**n - 1, so can be packed efficiently into a tree
        # Last row of nodes is indices 0, 2, ..., 2**n - 2
        # Second-last row is indices 1, 5, ..., 2**n - 3
        # nth-last row is indices (2**n - 1)::(2**(n+1))
        values_start = 0
        values_space = 2
        values_len = 2**last_full_row
        while values_start < len(values):
            tree[values_len - 1 : 2 * values_len - 1] = values[
                values_start::values_space
            ]
            values_start += int(values_space / 2)
            values_space *= 2
            values_len = int(values_len / 2)
        return tree

    def insert(self, value, weight_index):
        """Insert an occurrence of `value` into the btree."""
        idx_node = 0
        while idx_node < len(self._tree):
            current = self._tree[idx_node]
            # self._counts[i] += 1
            self._weight_indices[idx_node].append(weight_index)
            if value < current:
                idx_node = 2 * idx_node + 1
            elif value > current:
                idx_node = 2 * idx_node + 2
            else:
                return
        raise ValueError(
            f"Value {value} not contained in tree. Also, the counts are now messed up."
        )

    def total_counts(self, idx_value):
        return self._get_counts(idx_node=0, idx_value=idx_value)

    def rank(self, value, idx_value):
        """Returns the rank and count of the value in the btree."""
        idx_node = 0
        n = len(self._tree)
        rank = count = 0
        
        while idx_node < n:

            current = self._tree[idx_node]

            if value < current:
                idx_node = 2 * idx_node + 1

            elif value > current:
                # Since the input value is higher than all the values from the
                # left subtree, we add to the rank the weighted sum of items that
                # were inserted in the left subtree:
                # rank += counts_current_tree - counts_right_subtree
                rank += self._get_counts(idx_node, idx_value)

                # Subtract off the right subtree if exists
                idx_node = 2 * idx_node + 2
                if idx_node < n:
                    rank -= self._get_counts(idx_node, idx_value)

            else:
                # We have found the node corresponding to our input value.
                # We now add to the rank the weighted sum of items inserted into
                # the left subtree.
                # 'count' represent the weighted sum of items inserted at
                # the current node.
                # counts_current_node = counts_current_tree
                # - counts_left_subtree - counts_right_subtree
                count = self._get_counts(idx_node, idx_value)
    
                idx_node = 2 * idx_node + 1
                if idx_node < n:
                    count_left = self._get_counts(idx_node, idx_value)
                    count -= count_left
                    rank += count_left
                    
                    # Remove the counts of the right subtree
                    idx_node += 1
                    if idx_node < n:
                        count -= self._get_counts(idx_node, idx_value)
                
                return (rank, count)
        
        return (rank, count)
    
    def _get_counts(self, idx_node, idx_value):
        indices = self._weight_indices[idx_node]
        left_weight = self._weights[idx_value]
        return sum([
            left_weight * self._weights[jdx] for jdx in indices])
