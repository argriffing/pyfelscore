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

