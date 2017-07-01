import itertools
import random
from collections import defaultdict

import pandas as pd
import numpy as np


def bipartite_shuffle_binary_matrix(df, n_permutations=1, return_matrices=True):
    # Todo: write a generator version, since large matrices take up lots of space
    set_dict = binary_matrix_to_set_dict(df)
    shuffled_set_dicts = permute_sets_preserving_bipartite_degree(set_dict, n_permutations=n_permutations)
    if return_matrices:
        return [set_dict_to_binary_matrix(shuffled_set_dict, df.index, df.columns) for shuffled_set_dict in shuffled_set_dicts]
    else:
        return shuffled_set_dicts


def set_dict_to_binary_matrix(set_dict, set_order=None, item_order=None):
    if item_order is None:
        items = itertools.chain.from_iterable(set_dict.values())
        item_order = list(set(items))
    if set_order is None:
        set_order = set_dict.keys()
    df = pd.DataFrame(index=set_order, columns=item_order)
    for s in set_order:
        if s in set_dict:
            gs = set_dict[s]
            df.loc[s, gs] = 1
    return df.fillna(0)


def binary_matrix_to_set_dict(binary_matrix):
    set_dict = dict()
    for s in binary_matrix.index:
        set_dict[s] = set(binary_matrix.columns[binary_matrix.loc[s].nonzero()])
    return set_dict


def reverse_item2set_dict(p2g):
    g2p = defaultdict(set)
    for p, gs in p2g.items():
        for g in gs:
            g2p[g].add(p)
    return g2p


def bipartite_degree_preserved(original_sets, permuted_sets, verbose=False):
    o_sets, o_itemss = zip(*original_sets.items())
    original_set_sizes = [len(o_items) for o_items in o_itemss]
    permuted_set_sizes = [len(permuted_sets[s]) for s in o_sets]
    ro_items, ro_setss = zip(*reverse_item2set_dict(original_sets).items())
    original_item_sizes = [len(ro_sets) for ro_sets in ro_setss]
    perm_reverse = reverse_item2set_dict(permuted_sets)
    permuted_item_sizes = [len(perm_reverse[ro_item]) for ro_item in ro_items]
    if verbose:
        print("Set sizes:")
        print(original_set_sizes)
        print(permuted_set_sizes)
        print("Item sizes:")
        print(ro_items)
        print(original_item_sizes)
        print(permuted_item_sizes)
    return original_set_sizes == permuted_set_sizes and original_item_sizes == permuted_item_sizes


def list_duplicate_indices(seq):
    unique_items = set()
    #duplicates = set()
    duplicate_indices = set()
    #original_indices = dict()
    for i, item in enumerate(seq):
        if item in unique_items:
            #duplicates.add(item)
            duplicate_indices.add(i)
        else:
            unique_items.add(item)
            #original_indices[item] = i
    #original_indices = {dup: original_indices[dup] for dup in duplicates}
    #print "Duplicates:", duplicates
    return duplicate_indices, unique_items#, original_indices


def resolve_collisions(perm_edge_list, verbose=False):
    #perm_edge_set = set(perm_edge_list) # this is duplicating work of finding duplicates, but may be faster than my method
    n_edges = len(perm_edge_list)
    duplicates_indices, perm_edge_set = list_duplicate_indices(perm_edge_list)
    if verbose:
        print (len(duplicates_indices), "collisions resolved of {} edges".format(n_edges))
    while duplicates_indices:
        dup_index = duplicates_indices.pop()
        a, b = perm_edge_list[dup_index]
        resolved = False
        #print a, b
        while not resolved:
            random_index = random.randint(0, n_edges - 1)
            if random_index != dup_index:
                c, d = perm_edge_list[random_index]
                #print '    ', c, d
                if all([b != c, a != d, a != b, c != d, a != c, b != d]):
                    if (a, d) not in perm_edge_set and (c, b) not in perm_edge_set:
                        perm_edge_list[dup_index] = (a, d)
                        perm_edge_list[random_index] = (c, b)
                        # what if you remove cd and it was the original of a duplicate?
                        if random_index in duplicates_indices:
                            duplicates_indices.remove(random_index)
                        else:
                            perm_edge_set.remove((c, d))
                        perm_edge_set.add((a, d))
                        perm_edge_set.add((c, b))
                        resolved = True
    return perm_edge_set


def permute_sets_preserving_bipartite_degree(sets_to_items, n_permutations=1, verbose=False):
    # Todo: it is a small innovation, but would it be worth showing this is faster than birewire and still correct?
    # i.e. should we publish this? Would have to re-write in C++ to make comparable
    # Matan should get credit, too.
    edge_list = dict_to_edge_list(sets_to_items)
    sets, items = zip(*edge_list)
    rewired = []
    for i in range(n_permutations):
        # random_sets = np.random.permutation(sets)
        # np profiles as being faster but overall runtime becomes slower for some reason...
        # maybe because it has to be cast as a list for zip?
        random_sets = random.sample(sets, len(sets))
        shuffled_tups = list(zip(random_sets, items))
        resolved_shuffled_tups = resolve_collisions(shuffled_tups, verbose=verbose)
        shuffled_sets_to_items = defaultdict(set)
        for s, item in resolved_shuffled_tups:
            shuffled_sets_to_items[s].add(item)
        rewired.append(dict(shuffled_sets_to_items))
    return rewired


def permute_sets_preserving_bipartite_degree_gen(sets_to_items, n_permutations=1, verbose=False):
    edge_list = dict_to_edge_list(sets_to_items)
    sets, items = zip(*edge_list)
    i = 0
    while i < n_permutations:
        random_sets = random.sample(sets, len(sets))
        shuffled_tups = list(zip(random_sets, items))
        resolved_shuffled_tups = resolve_collisions(shuffled_tups, verbose=verbose)
        shuffled_sets_to_items = defaultdict(set)
        for s, item in resolved_shuffled_tups:
            shuffled_sets_to_items[s].add(item)
        yield dict(shuffled_sets_to_items)
        i += 1


def dict_to_edge_list(d):
    tups = d.items()
    edge_list = list(itertools.chain.from_iterable(itertools.product([tup[0]], tup[1]) for tup in tups))
    return edge_list


def permute_sets_approx_preserving_bipartite_degree(sets_to_items, n_permutations=1, verbose=False):
    # the idea is that a little unbiased random variation around the original degree will not hurt results
    # and will allow a significant speedup. Remains to be tested and optimized.
    # turns out to be only barely faster here.
    # with current implementation, exact is hard to beat by much
    edge_list = dict_to_edge_list(sets_to_items)
    sets, items = zip(*edge_list)
    n_edges = len(edge_list)
    rewired = []
    for i in range(n_permutations):
        perm_edge_set = set()
        """
        while n_edges_added < n_edges: # could probably speed up by sampling many and removing dups?
            random_set = random.sample(sets, 1)[0]
            random_item = random.sample(items, 1)[0]
            edge = (random_set, random_item)
            if edge not in perm_edge_set: # is this checking twice whether it's in the set?
                perm_edge_set.add(edge)
                n_edges_added += 1
        """
        """
        n_random_edges = int(round(n_edges * 1.25))
        random_sets = np.random.choice(sets, size=n_random_edges, replace=True)
        random_items = np.random.choice(items, size=n_random_edges, replace=True)
        perm_edge_set = set(zip(random_sets, random_items))
        perm_edge_set = random.sample(perm_edge_set, n_edges)
        """
        # random_sets = np.random.permutation(sets)
        # np profiles as being faster but overall runtime becomes slower for some reason...
        # maybe because it has to be cast as a list for zip?
        random_sets = random.sample(sets, len(sets))
        shuffled_tups = zip(random_sets, items)
        perm_edge_set = set(shuffled_tups)
        n_edges_added = len(perm_edge_set)
        while n_edges_added < n_edges: # could probably speed up by sampling many and removing dups?
            random_set = random.sample(sets, 1)[0]
            random_item = random.sample(items, 1)[0]
            edge = (random_set, random_item)
            if edge not in perm_edge_set: # is this checking twice whether it's in the set?
                perm_edge_set.add(edge)
                n_edges_added += 1
        perm_sets_to_items = defaultdict(set)
        for s, item in perm_edge_set:
            perm_sets_to_items[s].add(item)
        rewired.append(perm_sets_to_items)
    return rewired


def monte_carlo_permute_sets_preserving_bipartite_degree(sets_to_items, n_permutations=1, verbose=False):
    tups = sets_to_items.items()
    sets, items = zip(*tups)
    edge_list = list(itertools.chain.from_iterable(itertools.product([tup[0]], tup[1]) for tup in tups))
    n_edges = len(edge_list)
    n_sets = len(sets)
    unique_items = set()
    for item in items:
        unique_items.update(item)
    n_items = len(unique_items)
    edge_idxs = range(n_edges)
    n_switches = calculate_n_switches(n_edges, n_sets, n_items)
    if verbose:
        print("Performing {} switches for each permutation".format(n_switches))
    rewired = []
    for i in range(n_permutations):
        #print i
        perm_edge_list = list(edge_list)
        perm_edge_set = set(perm_edge_list)
        switches_left = n_switches
        while switches_left > 0:
            m, n = np.random.choice(edge_idxs, size=2, replace=False)
            a, b = perm_edge_list[m]
            c, d = perm_edge_list[n]
            if all([b != c, a != d, a != b, c != d]):
                if (a, d) not in perm_edge_set and (c, b) not in perm_edge_set:
                    perm_edge_list[n] = (a, d)
                    perm_edge_list[m] = (c, b)
                    perm_edge_set.remove((a, b))
                    perm_edge_set.remove((c, d))
                    perm_edge_set.add((a, d))
                    perm_edge_set.add((c, b))
            switches_left -= 1  # Todo: should decrement be done only if conditions met, or for every proposed switch?
        perm_sets_to_items = defaultdict(set)
        for s, item in perm_edge_set:
            perm_sets_to_items[s].add(item)
        rewired.append(perm_sets_to_items)
    return rewired


def calculate_n_switches(n_edges, n_sets, n_items):
    # Todo: allow for specifying error (to allow fewer switches--otherwise can be too slow on large problems)
    d = n_edges / float(n_sets * n_items)
    n = n_edges / (2 * (1 - d))
    n *= np.log((1 - d) * n_edges)
    return n


def birewire(binary_df, n_permutations=1):
    import rpy2.robjects as ro
    from rpy2.robjects.numpy2ri import numpy2ri
    ro.conversion.py2ri = numpy2ri
    from rpy2.robjects.packages import importr
    birewire = importr("BiRewire")
    rewired_dfs = []
    for i in range(n_permutations):
        rewired = np.array(birewire.birewire_rewire_bipartite(binary_df.values))
        rewired_df = pd.DataFrame(rewired, index=binary_df.index, columns=binary_df.columns)
        rewired_dfs.append(rewired_df)
    return rewired_dfs