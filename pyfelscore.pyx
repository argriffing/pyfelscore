"""
Speed up log likelihood calculations.

The implementation is in Cython for speed
and uses python numpy arrays for speed and convenience.
For compilation instructions see
http://docs.cython.org/src/reference/compilation.html
For example:
$ cython -a pyfelscore.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/include/python2.7 -o pyfelscore.so pyfelscore.c
"""

from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp, sqrt

np.import_array()

__all__ = [
        'site_fels',
        'align_fels',
        'get_mmpp_block',
        'get_mmpp_block_zero_off_rate',
        'get_mmpp_frechet_all_positive',
        'get_mmpp_frechet_diagonalizable_w_zero',
        'mcy_get_node_to_pset',
        'get_node_to_set',
        'mcy_esd_get_node_to_pset',
        'esd_get_node_to_set',
        'mcy_esd_get_node_to_pmap',
        ]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t ddot(
        np.float64_t [:] a,
        np.float64_t [:] b,
        ) nogil:
    """
    This is slower than doing the dot product manually.
    """
    cdef double accum = 0.0
    cdef long n = a.shape[0]
    cdef long i
    for i in range(n):
        accum += a[i] * b[i]
    return accum


@cython.boundscheck(False)
@cython.wraparound(False)
def align_fels(
        np.ndarray[np.int_t, ndim=1] OV,
        np.ndarray[np.int_t, ndim=2] DE,
        np.ndarray[np.int_t, ndim=2] patterns,
        np.ndarray[np.float64_t, ndim=1] pattern_weights,
        np.ndarray[np.float64_t, ndim=3] multi_P,
        np.ndarray[np.float64_t, ndim=1] root_prior,
        ):
    """
    @param OV: ordered vertices with child vertices before parent vertices
    @param DE: array of directed edges
    @param patterns: each pattern maps each vertex to a state or to -1
    @param pattern_weights: pattern multiplicites
    @param multi_P: a transition matrix for each directed edge
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
    """
    cdef long nvertices = OV.shape[0]
    cdef long nstates = root_prior.shape[0]
    cdef long npatterns = pattern_weights.shape[0]
    cdef double ll_pattern
    cdef double ll_total

    # allocate the ndarray that maps vertices to subtree likelihoods
    cdef np.ndarray[np.float64_t, ndim=2] likelihoods = np.empty(
            (nvertices, nstates), dtype=float)

    # sum over all of the patterns
    return align_fels_experimental(
            OV, DE, patterns, pattern_weights,
            multi_P, root_prior, likelihoods)


@cython.boundscheck(False)
@cython.wraparound(False)
def site_fels(
        np.ndarray[np.int_t, ndim=1] OV,
        np.ndarray[np.int_t, ndim=2] DE,
        np.ndarray[np.int_t, ndim=1] pattern,
        np.ndarray[np.float64_t, ndim=3] multi_P,
        np.ndarray[np.float64_t, ndim=1] root_prior,
        ):
    """
    @param OV: ordered vertices with child vertices before parent vertices
    @param DE: array of directed edges
    @param pattern: maps vertex to state, or to -1 if internal
    @param multi_P: a transition matrix for each directed edge
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
    """

    # init counts and indices
    cdef long nvertices = OV.shape[0]
    cdef long nstates = root_prior.shape[0]

    # initialize the map from vertices to subtree likelihoods
    cdef np.ndarray[np.float64_t, ndim=2] likelihoods = np.empty(
            (nvertices, nstates), dtype=float)

    # return the log likelihood
    return site_fels_experimental(
            OV, DE, pattern, multi_P, root_prior, likelihoods)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double site_fels_experimental(
        np.int_t [:] OV,
        np.int_t [:, :] DE,
        np.int_t [:] pattern,
        np.float64_t [:, :, :] multi_P,
        np.float64_t [:] root_prior,
        np.float64_t [:, :] likelihoods,
        ) nogil:
    
    # init counts and indices
    cdef long nvertices = OV.shape[0]
    cdef long nstates = root_prior.shape[0]
    cdef long nedges = DE.shape[0]
    cdef long root = OV[nvertices - 1]

    # declare more variables
    cdef long parent_vertex
    cdef long child_vertex
    cdef long parent_state
    cdef long child_state
    cdef long pattern_state
    cdef long child_pat_state
    cdef double log_likelihood

    # utility variables for computing a dot product manually
    cdef long i
    cdef double accum
    cdef double a_i, b_i

    # Compute the subtree likelihoods using dynamic programming.
    for parent_vertex_index in range(nvertices):
        parent_vertex = OV[parent_vertex_index]
        pattern_state = pattern[parent_vertex]
        for parent_state in range(nstates):
            if pattern_state != -1 and parent_state != pattern_state:
                likelihoods[parent_vertex, parent_state] = 0.0
            else:
                likelihoods[parent_vertex, parent_state] = 1.0
                for edge_index in range(nedges):
                    if DE[edge_index, 0] == parent_vertex:
                        child_vertex = DE[edge_index, 1]
                        child_pat_state = pattern[child_vertex]
                        if child_pat_state == -1:
                            accum = 0.0
                            for i in range(nstates):
                                a_i = multi_P[edge_index, parent_state, i]
                                b_i = likelihoods[child_vertex, i]
                                accum += a_i * b_i
                        else:
                            i = child_pat_state
                            a_i = multi_P[edge_index, parent_state, i]
                            b_i = likelihoods[child_vertex, i]
                            accum = a_i * b_i
                        likelihoods[parent_vertex, parent_state] *= accum

    # Get the log likelihood by summing over equilibrium states at the root.
    #return log(ddot(root_prior, likelihoods[root]))
    accum = 0.0
    for i in range(nstates):
        a_i = root_prior[i]
        b_i = likelihoods[root, i]
        accum += a_i * b_i
    return log(accum)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double align_fels_experimental(
        np.int_t [:] OV,
        np.int_t [:, :] DE,
        np.int_t [:, :] patterns,
        np.float64_t [:] pattern_weights,
        np.float64_t [:, :, :] multi_P,
        np.float64_t [:] root_prior,
        np.float64_t [:, :] likelihoods,
        ) nogil:
    """
    @param OV: ordered vertices with child vertices before parent vertices
    @param DE: array of directed edges
    @param patterns: each pattern maps each vertex to a state or to -1
    @param pattern_weights: pattern multiplicites
    @param multi_P: a transition matrix for each directed edge
    @param root_prior: equilibrium distribution at the root
    @param likelihoods: an (nvertices, nstates) array
    @return: log likelihood
    """

    # things required for alignment but not for just one pattern
    cdef long npatterns = pattern_weights.shape[0]
    cdef long pat_index
    cdef double ll_pattern
    cdef double ll_total

    # This should represent a view of a row of the pattern matrix.
    cdef np.int_t [:] pattern

    # Compute the subtree likelihoods using dynamic programming.
    ll_total = 0.0
    for pat_index in range(npatterns):
        pattern = patterns[pat_index]
        ll_pattern = site_fels_experimental(
                OV, DE, pattern, multi_P, root_prior, likelihoods)
        ll_total += ll_pattern * pattern_weights[pat_index]
    
    # Return the total log likelihood summed over all patterns.
    return ll_total


@cython.boundscheck(False)
@cython.wraparound(False)
def align_rooted_star_tree(
        np.int_t [:, :] patterns,
        np.float64_t [:] pattern_weights,
        np.float64_t [:, :, :] multi_P,
        np.float64_t [:] root_prior,
        ):
    """
    This is a specialization to rooted star trees.
    The assumptions are that the state is unknown at the root,
    but that the state at all tips of the tree are known.
    We assume that we have a prior distribution at the root
    and that the transition matrices from the root state to the leaf
    states are given for each terminal branch (and all branches are terminal).
    @param patterns: each pattern maps each leaf to a nonnegative state
    @param pattern_weights: pattern multiplicites
    @param multi_P: a transition matrix associated to each terminal branch
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
    """

    # Read some dimensions of the input.
    cdef long nleaves = multi_P.shape[0]
    cdef long npatterns = pattern_weights.shape[0]
    cdef long nstates = root_prior.shape[0]

    # For each pattern, sum over all possible root states.
    cdef double ll_accum = 0
    cdef double plike, p
    cdef long pattern_index, leaf_index
    cdef long root_state, leaf_state
    for pattern_index in range(npatterns):
        plike = 0
        for root_state in range(nstates):
            p = root_prior[root_state]
            for leaf_index in range(nleaves):
                leaf_state = patterns[pattern_index, leaf_index]
                p *= multi_P[leaf_index, root_state, leaf_state]
            plike += p
        ll_accum += pattern_weights[pattern_index] * log(plike)
    return ll_accum




###############################################################################
# More recent attempts at speedups related to likelihood functions.
#
# Terminology like 'csr', 'csc', 'indices', and 'indptr' is intended
# to be compatible with the corresponding sparse matrix jargon.
# For example, 'csr' is 'compressed sparse rows',
# and 'csc' is 'comparessed sparse columns'.
#
# The 'mcy' means 'markov chain with observation type y' where 'type y'
# means that for each node of the tree,
# a set of set of states is possible.
# This is observation type is more general than allowing the state
# to be only either known or unknown at each node,
# but it is less general than a full hidden-Markov framework
# for which each state at each node is associated with some
# emission probability.
#
# The 'esd' terminology means 'edge specific dense transition matrix'.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mcy_esd_get_node_to_pmap(
        np.int_t [:] tree_csr_indices,
        np.int_t [:] tree_csr_indptr,
        #
        np.float_t[:, :, :] esd_transitions, # (nnodes, nstates, nstates)
        #
        np.int_t [:, :] state_mask, # (nnodes, nstates)
        np.float_t[:, :] subtree_probability, # (nnodes, nstates)
        ):
    """
    Map each node to the set of states for which a subtree is feasible.

    The first group of args defines the structure of the rooted tree.
    Its nodes should be indexed in preorder from the root,
    so node 0 is the root and the rest of the nodes are ordered
    according to a preorder traversal.

    The second group of args is a single arg that defines the edge-specific
    dense transition matrices.
    For a rooted tree, the non-root nodes are in a bijective
    correspondence to the edges of the tree.
    For notational purposes the root node
    (necessarily node 0 because the nodes are sorted in preorder from the root)
    will also be associated with an edge,
    but the transition matrix on this edge will be ignored.

    The third arg group has information per state per node.
    An entry in the state_mask is 0 to represent a state that is impossible
    for whatever reason.
    This state_mask is an input ndarray which will not be modified.
    The subtree_probability ndarray is for output only.
    An entry in subtree_probability will be 0 if the corresponding
    entry in the state_mask is 0, otherwise it will contain
    the subtree probability given the state at the node.

    """
    cdef int nnodes = state_mask.shape[0]
    cdef int nstates = state_mask.shape[1]
    cdef int node_ind_start, node_ind_stop
    cdef double multiplicative_prob
    cdef double additive_prob
    cdef int na, nb
    cdef int i, j
    for i in range(nnodes):

        # Define the current node.
        na = (nnodes - 1) - i
        node_ind_start = tree_csr_indptr[na]
        node_ind_stop = tree_csr_indptr[na+1]

        # Compute the subtree probability for each possible state.
        for sa in range(nstates):
            subtree_probability[na, sa] = 0
            if not state_mask[na, sa]:
                continue
            multiplicative_prob = 1
            for j in range(node_ind_start, node_ind_stop):
                nb = tree_csr_indices[j]
                additive_prob = 0 
                for sb in range(nstates):
                    if state_mask[nb, sb]:
                        additive_prob += (
                                esd_transitions[nb, sa, sb] *
                                subtree_probability[nb, sb])
                multiplicative_prob *= additive_prob
            subtree_probability[na, sa] = multiplicative_prob

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mcy_esd_get_node_to_pset(
        np.int_t [:] tree_csr_indices,
        np.int_t [:] tree_csr_indptr,
        #
        np.float_t[:, :, :] esd_transitions, # (nnodes, nstates, nstates)
        #
        np.int_t [:, :] state_mask,
        ):
    """
    Map each node to the set of states for which a subtree is feasible.

    The first group of args defines the structure of the rooted tree.
    Its nodes should be indexed in preorder from the root,
    so node 0 is the root and the rest of the nodes are ordered
    according to a preorder traversal.

    The second group of args is a single arg that defines the edge-specific
    dense transition matrices.
    For a rooted tree, the non-root nodes are in a bijective
    correspondence to the edges of the tree.
    For notational purposes the root node
    (necessarily node 0 because the nodes are sorted in preorder from the root)
    will also be associated with an edge,
    but the transition matrix on this edge will be ignored.

    The third arg group comprises the single dense mutable binary ndarray
    which defines the set of allowed states for each node.
    This ndarray will be used for both input and output;
    as input, it will contain some zeros and some ones,
    and upon termination of this function, some of the ones will possibly
    have been changed to zeros.

    """
    cdef int nnodes = state_mask.shape[0]
    cdef int nstates = state_mask.shape[1]
    cdef int node_ind_start, node_ind_stop
    cdef int na, nb
    cdef int good_state_flag
    cdef int bad_node_flag
    cdef int i, j
    for i in range(nnodes):

        # Define the current node.
        na = (nnodes - 1) - i
        node_ind_start = tree_csr_indptr[na]
        node_ind_stop = tree_csr_indptr[na+1]

        # Query each potentially allowed state of the current node
        # by checking whether for each child node,
        # a valid state of the child node can be reached
        # from the putative state of the current node.
        for sa in range(nstates):
            if not state_mask[na, sa]:
                continue
            bad_node_flag = 0
            for j in range(node_ind_start, node_ind_stop):
                nb = tree_csr_indices[j]
                good_state_flag = 0
                for sb in range(nstates):
                    if esd_transitions[nb, sa, sb] and state_mask[nb, sb]:
                        good_state_flag = 1
                        break
                if not good_state_flag:
                    bad_node_flag = 1
                    break

            # If, for one of the child nodes nb,
            # no allowed state can be reached from the
            # current state sa of the current node na,
            # then mark the state sa as forbidden in the state mask of na.
            if bad_node_flag:
                state_mask[na, sa] = 0

    return 0



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mcy_get_node_to_pset(
        np.int_t [:] tree_csr_indices,
        np.int_t [:] tree_csr_indptr,
        #
        np.int_t [:] trans_csr_indices,
        np.int_t [:] trans_csr_indptr,
        #
        np.int_t [:, :] state_mask,
        ):
    """
    Map each node to the set of states for which a subtree is feasible.

    The first group of args defines the structure of the rooted tree.
    Its nodes should be indexed in preorder from the root,
    so node 0 is the root and the rest of the nodes are ordered
    according to a preorder traversal.

    The second group of args defines the structure of the transition matrix.
    The same transition matrix is used on each edge of the tree.
    Its structure is defined in a way that
    efficiently gives the set of states that can be reached in one step
    from a given state.

    The third arg group comprises the single dense mutable binary ndarray
    which defines the set of allowed states for each node.
    This ndarray will be used for both input and output;
    as input, it will contain some zeros and some ones,
    and upon termination of this function, some of the ones will possibly
    have been changed to zeros.

    """
    cdef int nnodes = state_mask.shape[0]
    cdef int nstates = state_mask.shape[1]
    cdef int na, nb
    cdef int sa, sb
    cdef int node_ind_start, node_ind_stop
    cdef int state_ind_start, state_ind_stop
    cdef int good_state_flag
    cdef int bad_node_flag
    cdef int i, j, k
    for i in range(nnodes):

        # Define the current node.
        na = (nnodes - 1) - i
        node_ind_start = tree_csr_indptr[na]
        node_ind_stop = tree_csr_indptr[na+1]

        # Query each potentially allowed state of the current node
        # by checking whether for each child node,
        # a valid state of the child node can be reached
        # from the putative state of the current node.
        for sa in range(nstates):
            if not state_mask[na, sa]:
                continue
            state_ind_start = trans_csr_indptr[sa]
            state_ind_stop = trans_csr_indptr[sa+1]
            bad_node_flag = 0
            for j in range(node_ind_start, node_ind_stop):
                nb = tree_csr_indices[j]
                good_state_flag = 0
                for k in range(state_ind_start, state_ind_stop):
                    sb = trans_csr_indices[k]
                    if state_mask[nb, sb]:
                        good_state_flag = 1
                        break
                if not good_state_flag:
                    bad_node_flag = 1
                    break

            # If, for one of the child nodes nb,
            # no allowed state can be reached from the
            # current state sa of the current node na,
            # then mark the state sa as forbidden in the state mask of na.
            if bad_node_flag:
                state_mask[na, sa] = 0

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def esd_get_node_to_set(
        np.int_t [:] tree_csr_indices,
        np.int_t [:] tree_csr_indptr,
        #
        np.float_t[:, :, :] esd_transitions, # (nnodes, nstates, nstates)
        #
        np.int_t [:, :] state_mask, # (nnodes, nstates)
        ):
    """
    This function should be called after calling mcy_esd_get_node_to_pset.

    Whereas mcy_get_node_to_pset is a backward pass,
    from the leaves to the root,
    the function get_node_to_set is a forward pass
    from the root to the leaves.
    Both of these functions use only the sparsity structure
    of the tree and the transition matrix and the node state constraints.
    In other words, they only care about whether or not a given event
    is possible, rather than caring about whether the event has
    a high vs. a low probability.

    For each pre-order parent node,
    constrain the set of allowed states of each child node.
    Do this by constructing the set of child node states
    which can be reached by starting in one of the allowed parent states
    and transitioning to the child state.
    This constructed set will act as a mask which can be applied
    to all child node state masks.

    """
    cdef int nnodes = state_mask.shape[0]
    cdef int nstates = state_mask.shape[1]
    cdef int na, nb
    cdef int sa, sb
    cdef int node_ind_start, node_ind_stop
    cdef int i
    cdef int reachable_flag

    # For each parent node, restrict the set of possible child node states.
    for na in range(nnodes):

        # Get the info about the child nodes implied by the tree shape.
        node_ind_start = tree_csr_indptr[na]
        node_ind_stop = tree_csr_indptr[na+1]

        # If there are no child nodes then skip.
        if node_ind_start == node_ind_stop:
            continue

        # For each child node,
        # restrict the set of allowed states,
        # by allowing only the states which can be reached in one step
        # from a state that is allowed at the parent node.
        for i in range(node_ind_start, node_ind_stop):
            nb = tree_csr_indices[i]
            for sb in range(nstates):
                reachable_flag = 0
                for sa in range(nstates):
                    if state_mask[na, sa] and esd_transitions[nb, sa, sb]:
                        reachable_flag = 1
                        break
                if not reachable_flag:
                    state_mask[nb, sb] = 0

    return 0



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_node_to_set(
        np.int_t [:] tree_csr_indices,
        np.int_t [:] tree_csr_indptr,
        #
        np.int_t [:] trans_csr_indices,
        np.int_t [:] trans_csr_indptr,
        #
        np.int_t [:, :] state_mask, # (nnodes, nstates)
        np.int_t [:] tmp_state_mask, # (nstates,)
        ):
    """
    This function should be called after having called mcy_get_node_to_pset.

    Whereas mcy_get_node_to_pset is a backward pass,
    from the leaves to the root,
    the function get_node_to_set is a forward pass
    from the root to the leaves.
    Both of these functions use only the sparsity structure
    of the tree and the transition matrix and the node state constraints.
    In other words, they only care about whether or not a given event
    is possible, rather than caring about whether the event has
    a high vs. a low probability.

    For each pre-order parent node,
    constrain the set of allowed states of each child node.
    Do this by constructing the set of child node states
    which can be reached by starting in one of the allowed parent states
    and transitioning to the child state.
    This constructed set will act as a mask which can be applied
    to all child node state masks.

    """
    cdef int nnodes = state_mask.shape[0]
    cdef int nstates = state_mask.shape[1]
    cdef int na, nb
    cdef int sa, sb
    cdef int node_ind_start, node_ind_stop
    cdef int state_ind_start, state_ind_stop
    cdef int i

    # For each parent node, restrict the set of possible child node states.
    for na in range(nnodes):

        # Get the info about the child nodes implied by the tree shape.
        node_ind_start = tree_csr_indptr[na]
        node_ind_stop = tree_csr_indptr[na+1]

        # If there are no child nodes then skip.
        if node_ind_start == node_ind_stop:
            continue

        # Construct the state mask defining the set of states
        # that can be reached in one step from a state
        # that is allowed at the parent node.
        for sa in range(nstates):
            tmp_state_mask[sa] = 0
        for sa in range(nstates):
            if not state_mask[na, sa]:
                continue
            state_ind_start = trans_csr_indptr[sa]
            state_ind_stop = trans_csr_indptr[sa+1]
            for i in range(state_ind_start, state_ind_stop):
                sb = trans_csr_indices[i]
                tmp_state_mask[sb] = 1

        # For each child node,
        # restrict the set of allowed states,
        # by allowing only the states which can be reached in one step
        # from a state that is allowed at the parent node.
        for i in range(node_ind_start, node_ind_stop):
            nb = tree_csr_indices[i]
            for sb in range(nstates):
                state_mask[nb, sb] &= tmp_state_mask[sb]

    return 0



###############################################################################
# Special cases of the exponential of a rate matrix.



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_mmpp_block(double a, double w, double r, double t):
    """
    Compute differential equations evaluated at the given time.
    @param a: rate from off to on
    @param w: rate from on to off
    @param r: poisson event rate
    @param t: elapsed time
    @return: P
    """
    cdef double p, q
    cdef np.ndarray[np.float64_t, ndim=2] P = np.empty(
            (2, 2), dtype=np.float64)

    # initialize some common variables
    cdef double x = sqrt((a + r + w)*(a + r + w) - 4*a*r)
    cdef double denom = 2 * x * exp(t * (x + a + r + w) / 2)

    # first row of the output ndarray
    p = (exp(t*x)*(x + r + w - a) + (x - r - w + a)) / denom
    q = (2 * a * (exp(t * x) - 1)) / denom
    P[0, 0] = p
    P[0, 1] = q

    # second row of the output ndarray
    p = (2 * w * (exp(t * x) - 1)) / denom
    q = (exp(t*x)*(x - r - w + a) + (x + r + w - a)) / denom
    P[1, 0] = p
    P[1, 1] = q

    # return the probability ndarray
    return P


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_mmpp_block_zero_off_rate(double a, double r, double t):
    """
    Compute differential equations evaluated at the given time.
    @param a: rate from off to on
    @param r: poisson event rate
    @param t: elapsed time
    @return: P
    """
    cdef np.ndarray[np.float64_t, ndim=2] P = np.empty(
            (2, 2), dtype=np.float64)

    # lower left
    P[1, 0] = 0

    # diagonal
    P[0, 0] = exp(-a*t)
    P[1, 1] = exp(-r*t)
    
    # upper right
    if a == r:
        P[0, 1] = a * t * exp(-a*t)
    else:
        P[0, 1] = a*exp(-(r+a)*t)*(exp(r*t) - exp(a*t)) / (r - a)

    # return the probability ndarray
    return P




###############################################################################
# Special cases of the Frechet derivative of the exponential of a rate matrix.



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_mmpp_frechet_defective_w_zero(
        double a, double t,
        int ai, int bi, int ci, int di,
        ):
    """
    The rates a and r are positive, and w is zero.
    The a and r values are equal to each other.
    The index args must each be 0 or 1.
    A double precision floating point number is returned.
    This is computed using a symbolic math package
    together with the EXPM method mentioned in Tataru and Hobolth (2011).
    @param a: rate from off to on
    @param t: elapsed time
    @param ai: index
    @param bi: index
    @param ci: index
    @param di: index
    @return: double
    """

    # declare matrix exponential storage
    cdef double M[2][2][2][2]

    # construct scaled variable
    cdef double b = a*t
    cdef double et = exp(t)

    # the matrix has only a few unique entries
    cdef double alpha = exp(-b-t)*(et-1)
    cdef double beta = a * exp(-b-t)*(et*(t-1) + 1)
    cdef double gamma = -a*a*exp(-b-t)*(-et - 0.5*t*t*et + t*et + 1)

    # Fill the matrix exponential entries.
    # This could be done more cleverly.
    #
    M[0][0][0][0] = alpha
    M[0][1][0][0] = beta
    M[1][0][0][0] = 0
    M[1][1][0][0] = 0
    #
    M[0][0][0][1] = 0
    M[0][1][0][1] = alpha
    M[1][0][0][1] = 0
    M[1][1][0][1] = 0
    #
    M[0][0][1][0] = beta
    M[0][1][1][0] = gamma
    M[1][0][1][0] = alpha
    M[1][1][1][0] = beta
    #
    M[0][0][1][1] = 0
    M[0][1][1][1] = beta
    M[1][0][1][1] = 0
    M[1][1][1][1] = alpha

    # return the requested entry
    return M[ai][bi][ci][di]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_mmpp_frechet_diagonalizable_w_zero(
        double a, double r, double t,
        int ai, int bi, int ci, int di,
        ):
    """
    The rates a and r are positive, and w is zero.
    The a and r values are not equal to each other.
    The index args must each be 0 or 1.
    A double precision floating point number is returned.
    @param a: rate from off to on
    @param r: poisson event rate
    @param t: elapsed time
    @param ai: index
    @param bi: index
    @param ci: index
    @param di: index
    @return: double
    """

    # declare diagonalization storage
    cdef double U[2][2] 
    cdef double L[2]
    cdef double V[2][2] 
    cdef double J[2][2] 

    # eigendecomposition
    U[0][0] = 1
    U[0][1] = a / (a-r)
    U[1][0] = 0
    U[1][1] = 1
    L[0] = -a
    L[1] = -r
    V[0][0] = 1
    V[0][1] = -a / (a-r)
    V[1][0] = 0
    V[1][1] = 1

    # function of eigenvalues
    J[0][0] = t*exp(L[0]*t)
    J[0][1] = (exp(L[0]*t) - exp(L[1]*t)) / (L[0] - L[1])
    J[1][0] = (exp(L[1]*t) - exp(L[0]*t)) / (L[1] - L[0])
    J[1][1] = t*exp(L[1]*t)

    # construct the return value
    cdef int i, j
    cdef double isum
    cdef double jsum
    isum = 0
    for i in range(2):
        jsum = 0
        for j in range(2):
            jsum += U[di][j] * V[j][bi] * J[i][j]
        isum += U[ai][i] * V[i][ci] * jsum
    return isum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_mmpp_frechet_all_positive(
        double a, double w, double r, double t,
        int ai, int bi, int ci, int di,
        ):
    """
    All rates are positive.  So not zero.
    The index args must each be 0 or 1.
    A double precision floating point number is returned.
    @param a: rate from off to on
    @param w: rate from on to off
    @param r: poisson event rate
    @param t: elapsed time
    @param ai: index
    @param bi: index
    @param ci: index
    @param di: index
    @return: double
    """

    # precompute some things
    cdef double x = sqrt((a+r+w)*(a+r+w) - 4*a*r)
    cdef double xa = (-a + r + w - x) / (2 * w)
    cdef double xb = (-a + r + w + x) / (2 * w)
    cdef double det = 1 / (xa - xb)

    # declare diagonalization storage
    cdef double U[2][2] 
    cdef double L[2]
    cdef double V[2][2] 
    cdef double J[2][2] 

    # eigendecomposition
    U[0][0] = xa
    U[0][1] = xb
    U[1][0] = 1
    U[1][1] = 1
    L[0] = 0.5 * (-a - r - w - x)
    L[1] = 0.5 * (-a - r - w + x)
    V[0][0] = det
    V[0][1] = -xb*det
    V[1][0] = -det
    V[1][1] = xa*det

    # function of eigenvalues
    J[0][0] = t*exp(L[0]*t)
    J[0][1] = (exp(L[0]*t) - exp(L[1]*t)) / (L[0] - L[1])
    J[1][0] = (exp(L[1]*t) - exp(L[0]*t)) / (L[1] - L[0])
    J[1][1] = t*exp(L[1]*t)

    # construct the return value
    cdef int i, j
    cdef double isum
    cdef double jsum
    isum = 0
    for i in range(2):
        jsum = 0
        for j in range(2):
            jsum += U[di][j] * V[j][bi] * J[i][j]
        isum += U[ai][i] * V[i][ci] * jsum
    return isum

