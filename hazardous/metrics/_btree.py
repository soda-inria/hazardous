"""
The original implementation of this balanced tree belongs to lifelines.
This class extends the original one by enabling weighted counts of inserted
elements instead of simple counts.
"""

# %%
import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType, ListType, unicode_type

# See https://stackoverflow.com/questions/58637802/how-to-declare-a-typed-dict-whose-value-is-a-nested-list-in-numba  # noqa
# for nested types.
spec = [
    ("_tree", float64[:]),
    ("_left_weights", float64[:]),
    ("_right_weights", float64[:]),
    ("_left_weight_indices", DictType(int64, ListType(int64))),
]


@jitclass(spec)
class _BTree:
    """A balanced binary order statistic tree to help compute the concordance.

    When computing the concordance, we know all the values the tree will ever
    contain.
    That condition simplifies this tree a lot, because instead of complex
    AVL/red-black mechanisms, we do the following:

    - Store the final tree in flattened form in an array (so node i's children
    are 2i+1, 2i+2).
    - Store the current size of each subtree in another array with the same indices.
    - Insert a value by finding its index, then incrementing the size of the subtree
    at that index and propagating.
    - Rank an element by adding up subtree weighted counts.
    """

    def __init__(self, nodes, left_weights=None, right_weights=None):
        """
        Parameters
        ----------
        nodes: array of float
            Iterable of sorted (ascending), unique values that will be inserted.

        left_weights : array of float, default=None
            The inverse weight to apply for each inserted node.

        right_weights : array of float, default=None
            The inverse weight to apply for each node to rank.
        """
        self._tree = self._treeify(nodes)
        self._left_weights = left_weights
        self._right_weights = right_weights
        self._left_weight_indices = Dict.empty(
            key_type=int64,
            value_type=List.empty_list(int64),
        )

    def _treeify(self, values):
        """Convert the array of values into a complete balanced tree.

        Tree indices work as follows:
        * 0 is the root
        * 2n+1 is the left child of n
        * 2n+2 is the right child of n

        Parameters
        ----------
        values : array of float, sorted ascending.
            The values to order into a balanced tree

        Returns
        -------
        tree : array of shape (n_values,)
            Array in which t[i] > t[2i+1] and t[i] < t[2i+2] for all i.
        """
        n_values = len(values)
        if n_values == 1:
            return values

        tree = np.empty_like(values)

        # The first step is to remove the bottom row of leaves,
        # which might not be exactly full.
        last_full_row = int(np.log2(n_values + 1) - 1)
        len_ragged_row = n_values - (2 ** (last_full_row + 1) - 1)
        if len_ragged_row > 0:
            bottom_row_ix = slice(None, 2 * len_ragged_row, 2)
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

    def insert(self, value, left_weight_index):
        """Insert an occurrence of `value` into the btree.

        Parameters
        ----------
        value : float,
            The value to insert, i.e. to place in the tree
            and append its weight index to the traversed nodes.

        left_weight_index : int,
            The index to append to all traversed node. It matches a value
            in self._left_weights.
        """
        idx_node = 0
        while idx_node < len(self._tree):
            current = self._tree[idx_node]

            if idx_node not in self._left_weight_indices:
                self._left_weight_indices[idx_node] = List.empty_list(int64)
            self._left_weight_indices[idx_node].append(left_weight_index)

            if value < current:
                idx_node = 2 * idx_node + 1
            elif value > current:
                idx_node = 2 * idx_node + 2
            else:
                return
        raise ValueError(
            f"Value {value} not contained in tree. Also, the counts are now messed up."
        )

    def rank(
        self,
        value,
        jdx_right_weight=None,
        use_left_weight_only=True,
        return_weighted=False,
    ):
        """Returns the rank and count of the value in the btree.

        Parameters
        ----------
        value : float
            The value to rank. It may or may not belong to the tree.

        jdx_right_weight : int, default=None
            The weight index of the value to rank. It matches a value in
            self._right_index.

        use_left_weight_only : bool, default=True
            If set to True, the weighting will only consider left weights, i.e.
            weight of inserted nodes, and won't use the weight of the value
            to rank.
            Unused if return_weight=False or use_left_weight_only=True

        return_weighted : bool, default=False
            If set to True, the rank will use weighted count of subtrees.
            Otherwise, the rank use a regular count with uniform weights.

        Returns
        -------
        rank : Counter with keys:
            * num_pairs : int
                The number of concordant pairs, unweighted.
            * weighted_pairs : float
                The number of concordant pairs, weighted.
                0. if return_weighted is False

        count : Counter with keys:
            * num_pairs : int
                The number of prediction ties, unweighted.
            * weighted_pairs : float
                The number of prediction ties, weighted.
                0. if return_weighted is False
        """
        idx_node = 0
        n = len(self._tree)
        rank = self._init_counter()
        count = self._init_counter()

        while idx_node < n:
            current = self._tree[idx_node]

            if value < current:
                idx_node = 2 * idx_node + 1

            elif value > current:
                # Since the input value is higher than all the values from the
                # left subtree, we incremente the rank by the weighted sum of items
                # that were inserted in the left subtree:
                # rank += counts_current_tree - counts_right_subtree
                # self._add(rank, idx_node, **count_params)
                rank = self._add(
                    rank,
                    idx_node,
                    jdx_right_weight,
                    use_left_weight_only,
                    return_weighted,
                )

                # Subtract off the right subtree if exists
                idx_node = 2 * idx_node + 2
                if idx_node < n:
                    rank = self._sub(
                        rank,
                        idx_node,
                        jdx_right_weight,
                        use_left_weight_only,
                        return_weighted,
                    )

            else:
                # We have found the node corresponding to our input value.
                # We now add to the rank the weighted sum of items inserted into
                # the left subtree.
                # 'count' represent the weighted sum of items inserted at
                # the current node.
                # counts_current_node = counts_current_tree
                # - counts_left_subtree
                # - counts_right_subtree
                count = self._counts(
                    idx_node,
                    jdx_right_weight,
                    use_left_weight_only,
                    return_weighted,
                )

                idx_node = 2 * idx_node + 1
                if idx_node < n:
                    count = self._sub(
                        count,
                        idx_node,
                        jdx_right_weight,
                        use_left_weight_only,
                        return_weighted,
                    )
                    rank = self._add(
                        rank,
                        idx_node,
                        jdx_right_weight,
                        use_left_weight_only,
                        return_weighted,
                    )

                    # Remove the counts of the right subtree
                    idx_node += 1
                    if idx_node < n:
                        count = self._sub(
                            count,
                            idx_node,
                            jdx_right_weight,
                            use_left_weight_only,
                            return_weighted,
                        )
                break

        return rank, count

    def _init_counter(self):
        counter = Dict.empty(unicode_type, float64)
        counter["num_pairs"] = 0.0
        counter["weighted_pairs"] = 0.0
        return counter

    def _add(
        self,
        counter,
        idx_node,
        jdx_right_weight,
        use_left_weight_only,
        return_weighted,
    ):
        counter_ = self._counts(
            idx_node,
            jdx_right_weight,
            use_left_weight_only,
            return_weighted,
        )
        counter["num_pairs"] += counter_["num_pairs"]
        counter["weighted_pairs"] += counter_["weighted_pairs"]
        return counter

    def _sub(
        self,
        counter,
        idx_node,
        jdx_right_weight,
        use_left_weight_only,
        return_weighted,
    ):
        counter_ = self._counts(
            idx_node,
            jdx_right_weight,
            use_left_weight_only,
            return_weighted,
        )
        counter["num_pairs"] -= counter_["num_pairs"]
        counter["weighted_pairs"] -= counter_["weighted_pairs"]
        return counter

    def total_counts(
        self,
        jdx_right_weight=None,
        use_left_weight_only=True,
        return_weighted=False,
    ):
        """Compute the weighted total number of inserted values, i.e. the
        inserted values that traversed the root node.
        """
        return self._counts(
            idx_node=0,
            jdx_right_weight=jdx_right_weight,
            use_left_weight_only=use_left_weight_only,
            return_weighted=return_weighted,
        )

    def _counts(
        self,
        idx_node,
        jdx_right_weight=None,
        use_left_weight_only=True,
        return_weighted=False,
    ):
        """Compute the weighted total number of inserted values at
        the node whose index is idx_node.

        Parameters
        ----------
        idx_node : int
            The index of the tree node where to count inserted values.

        jdx_right_weight : int, default=None
            The index of the right inverse weight to apply to the count operation.
            Unused if return_weight=False or use_left_weight_only=True.

        use_left_weight_only : bool, default=True
            If set to True, the weighting count will only consider left weights, i.e.
            weight of inserted nodes, and won't use the weight of the value to rank.
            Unused if return_weight=False or use_left_weight_only=True

        return_weighted : bool, default=False
            If set to True, the operation will use weighted count of subtrees.
            Otherwise, a regular count with uniform weights is performed.

        Returns
        -------
        stats : Counter with keys:
            * num_pairs : int
                The total number of inserted nodes at idx_node, unweighted.
            * weighted_pairs : float
                The total number of inserted ndoes at idx_node, weighted.
        """
        stats = self._init_counter()

        if idx_node in self._left_weight_indices:
            indices = self._left_weight_indices[idx_node]
            stats["num_pairs"] = float64(len(indices))

            if return_weighted:
                if use_left_weight_only:
                    for idx in indices:
                        stats["weighted_pairs"] += 1 / (self._left_weights[idx] ** 2)
                else:
                    right_weight = self._right_weights[jdx_right_weight]
                    for idx in indices:
                        stats["weighted_pairs"] += 1 / (
                            self._left_weights[idx] * right_weight
                        )

        return stats
