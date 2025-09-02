from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from motmetrics import math_util
from collections import OrderedDict
import itertools
import pandas as pd
_INDEX_FIELDS = ['FrameId', 'Event']
_EVENT_FIELDS = ['Type', 'OId', 'HId', 'D']
from configparser import ConfigParser
import logging
import time
from contextlib import contextmanager
import warnings
from itertools import count

def _module_is_available_py2(name):
    try:
        imp.find_module(name)
        return True
    except ImportError:
        return False


def _module_is_available_py3(name):
    return importlib.util.find_spec(name) is not None


try:
    import importlib.util
except ImportError:
    import imp
    _module_is_available = _module_is_available_py2
else:
    _module_is_available = _module_is_available_py3


def linear_sum_assignment(costs, solver=None):
    """Solve a linear sum assignment problem (LSA).

    For large datasets solving the minimum cost assignment becomes the dominant runtime part.
    We therefore support various solvers out of the box (currently lapsolver, scipy, ortools, munkres)

    Params
    ------
    costs : np.array
        numpy matrix containing costs. Use NaN/Inf values for unassignable
        row/column pairs.

    Kwargs
    ------
    solver : callable or str, optional
        When str: name of solver to use.
        When callable: function to invoke
        When None: uses first available solver
    """
    costs = np.asarray(costs)
    if not costs.size:
        return np.array([], dtype=int), np.array([], dtype=int)

    solver = solver or default_solver

    if isinstance(solver, str):
        # Try resolve from string
        solver = solver_map.get(solver, None)

    assert callable(solver), 'Invalid LAP solver.'
    rids, cids = solver(costs)
    rids = np.asarray(rids).astype(int)
    cids = np.asarray(cids).astype(int)
    return rids, cids


def add_expensive_edges(costs):
    """Replaces non-edge costs (nan, inf) with large number.

    If the optimal solution includes one of these edges,
    then the original problem was infeasible.

    Parameters
    ----------
    costs : np.ndarray
    """
    # The graph is probably already dense if we are doing this.
    assert isinstance(costs, np.ndarray)
    # The linear_sum_assignment function in scipy does not support missing edges.
    # Replace nan with a large constant that ensures it is not chosen.
    # If it is chosen, that means the problem was infeasible.
    valid = np.isfinite(costs)
    if valid.all():
        return costs.copy()
    if not valid.any():
        return np.zeros_like(costs)
    r = min(costs.shape)
    # Assume all edges costs are within [-c, c], c >= 0.
    # The cost of an invalid edge must be such that...
    # choosing this edge once and the best-possible edge (r - 1) times
    # is worse than choosing the worst-possible edge r times.
    # l + (r - 1) (-c) > r c
    # l > r c + (r - 1) c
    # l > (2 r - 1) c
    # Choose l = 2 r c + 1 > (2 r - 1) c.
    c = np.abs(costs[valid]).max() + 1  # Doesn't hurt to add 1 here.
    large_constant = 2 * r * c + 1
    return np.where(valid, costs, large_constant)


def _exclude_missing_edges(costs, rids, cids):
    subset = [
        index for index, (i, j) in enumerate(zip(rids, cids))
        if np.isfinite(costs[i, j])
    ]
    return rids[subset], cids[subset]


def lsa_solve_scipy(costs):
    """Solves the LSA problem using the scipy library."""

    from scipy.optimize import linear_sum_assignment as scipy_solve

    # scipy (1.3.3) does not support nan or inf values
    finite_costs = add_expensive_edges(costs)
    rids, cids = scipy_solve(finite_costs)
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def lsa_solve_lapsolver(costs):
    """Solves the LSA problem using the lapsolver library."""
    from lapsolver import solve_dense

    # Note that lapsolver will add expensive finite edges internally.
    # However, older versions did not add a large enough edge.
    finite_costs = add_expensive_edges(costs)
    rids, cids = solve_dense(finite_costs)
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def lsa_solve_munkres(costs):
    """Solves the LSA problem using the Munkres library."""
    from munkres import Munkres

    m = Munkres()
    # The munkres package may hang if the problem is not feasible.
    # Therefore, add expensive edges instead of using munkres.DISALLOWED.
    finite_costs = add_expensive_edges(costs)
    # Ensure that matrix is square.
    finite_costs = _zero_pad_to_square(finite_costs)
    indices = np.array(m.compute(finite_costs), dtype=int)
    # Exclude extra matches from extension to square matrix.
    indices = indices[(indices[:, 0] < costs.shape[0])
                      & (indices[:, 1] < costs.shape[1])]
    rids, cids = indices[:, 0], indices[:, 1]
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def _zero_pad_to_square(costs):
    num_rows, num_cols = costs.shape
    if num_rows == num_cols:
        return costs
    n = max(num_rows, num_cols)
    padded = np.zeros((n, n), dtype=costs.dtype)
    padded[:num_rows, :num_cols] = costs
    return padded


def lsa_solve_ortools(costs):
    """Solves the LSA problem using Google's optimization tools. """
    from ortools.graph import pywrapgraph

    if costs.shape[0] != costs.shape[1]:
        # ortools assumes that the problem is square.
        # Non-square problem will be infeasible.
        # Default to scipy solver rather than add extra zeros.
        # (This maintains the same behaviour as previous versions.)
        return linear_sum_assignment(costs, solver='scipy')

    rs, cs = np.isfinite(costs).nonzero()  # pylint: disable=unbalanced-tuple-unpacking
    finite_costs = costs[rs, cs]
    scale = find_scale_for_integer_approximation(finite_costs)
    if scale != 1:
        warnings.warn('costs are not integers; using approximation')
    int_costs = np.round(scale * finite_costs).astype(int)

    assignment = pywrapgraph.LinearSumAssignment()
    # OR-Tools does not like to receive indices of type np.int64.
    rs = rs.tolist()  # pylint: disable=no-member
    cs = cs.tolist()
    int_costs = int_costs.tolist()
    for r, c, int_cost in zip(rs, cs, int_costs):
        assignment.AddArcWithCost(r, c, int_cost)

    status = assignment.Solve()
    try:
        _ortools_assert_is_optimal(pywrapgraph, status)
    except AssertionError:
        # Default to scipy solver rather than add finite edges.
        # (This maintains the same behaviour as previous versions.)
        return linear_sum_assignment(costs, solver='scipy')

    return _ortools_extract_solution(assignment)


def find_scale_for_integer_approximation(costs, base=10, log_max_scale=8, log_safety=2):
    """Returns a multiplicative factor to use before rounding to integers.

    Tries to find scale = base ** j (for j integer) such that:
        abs(diff(unique(costs))) <= 1 / (scale * safety)
    where safety = base ** log_safety.

    Logs a warning if the desired resolution could not be achieved.
    """
    costs = np.asarray(costs)
    costs = costs[np.isfinite(costs)]  # Exclude non-edges (nan, inf) and -inf.
    if np.size(costs) == 0:
        # No edges with numeric value. Scale does not matter.
        return 1
    unique = np.unique(costs)
    if np.size(unique) == 1:
        # All costs have equal values. Scale does not matter.
        return 1
    try:
        _assert_integer(costs)
    except AssertionError:
        pass
    else:
        # The costs are already integers.
        return 1

    # Find scale = base ** e such that:
    # 1 / scale <= tol, or
    # e = log(scale) >= -log(tol)
    # where tol = min(diff(unique(costs)))
    min_diff = np.diff(unique).min()
    e = np.ceil(np.log(min_diff) / np.log(base)).astype(int).item()
    # Add optional non-negative safety factor to reduce quantization noise.
    e += max(log_safety, 0)
    # Ensure that we do not reduce the magnitude of the costs.
    e = max(e, 0)
    # Ensure that the scale is not too large.
    if e > log_max_scale:
        warnings.warn('could not achieve desired resolution for approximation: '
                      'want exponent %d but max is %d', e, log_max_scale)
        e = log_max_scale
    scale = base ** e
    return scale


def _assert_integer(costs):
    # Check that costs are not changed by rounding.
    # Note: Elements of cost matrix may be nan, inf, -inf.
    np.testing.assert_equal(np.round(costs), costs)


def _ortools_assert_is_optimal(pywrapgraph, status):
    if status == pywrapgraph.LinearSumAssignment.OPTIMAL:
        pass
    elif status == pywrapgraph.LinearSumAssignment.INFEASIBLE:
        raise AssertionError('ortools: infeasible assignment problem')
    elif status == pywrapgraph.LinearSumAssignment.POSSIBLE_OVERFLOW:
        raise AssertionError('ortools: possible overflow in assignment problem')
    else:
        raise AssertionError('ortools: unknown status')


def _ortools_extract_solution(assignment):
    if assignment.NumNodes() == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    pairings = []
    for i in range(assignment.NumNodes()):
        pairings.append([i, assignment.RightMate(i)])

    indices = np.array(pairings, dtype=int)
    return indices[:, 0], indices[:, 1]


def lsa_solve_lapjv(costs):
    """Solves the LSA problem using lap.lapjv()."""

    from lap import lapjv

    # The lap.lapjv function supports +inf edges but there are some issues.
    # https://github.com/gatagat/lap/issues/20
    # Therefore, replace nans with large finite cost.
    finite_costs = add_expensive_edges(costs)
    row_to_col, _ = lapjv(finite_costs, return_cost=False, extend_cost=True)
    indices = np.array([np.arange(costs.shape[0]), row_to_col], dtype=int).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]  # pylint: disable=unsubscriptable-object
    rids, cids = indices[:, 0], indices[:, 1]
    # Ensure that no missing edges were chosen.
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


available_solvers = None
default_solver = None
solver_map = None


def _init_standard_solvers():
    global available_solvers, default_solver, solver_map  # pylint: disable=global-statement

    solvers = [
        ('lapsolver', lsa_solve_lapsolver),
        ('lap', lsa_solve_lapjv),
        ('scipy', lsa_solve_scipy),
        ('munkres', lsa_solve_munkres),
        ('ortools', lsa_solve_ortools),
    ]

    solver_map = dict(solvers)

    available_solvers = [s[0] for s in solvers if _module_is_available(s[0])]
    if len(available_solvers) == 0:
        default_solver = None
        warnings.warn('No standard LAP solvers found. Consider `pip install lapsolver` or `pip install scipy`', category=RuntimeWarning)
    else:
        default_solver = available_solvers[0]


_init_standard_solvers()


@contextmanager
def set_default_solver(newsolver):
    """Change the default solver within context.

    Intended usage

        costs = ...
        mysolver = lambda x: ... # solver code that returns pairings

        with lap.set_default_solver(mysolver):
            rids, cids = lap.linear_sum_assignment(costs)

    Params
    ------
    newsolver : callable or str
        new solver function
    """

    global default_solver  # pylint: disable=global-statement

    oldsolver = default_solver
    try:
        default_solver = newsolver
        yield
    finally:
        default_solver = oldsolver

def preprocessResult(res, gt, inifile):
    """Preprocesses data for utils.CLEAR_MOT_M.

    Returns a subset of the predictions.
    """
    # pylint: disable=too-many-locals
    st = time.time()
    labels = [
        'ped',               # 1
        'person_on_vhcl',    # 2
        'car',               # 3
        'bicycle',           # 4
        'mbike',             # 5
        'non_mot_vhcl',      # 6
        'static_person',     # 7
        'distractor',        # 8
        'occluder',          # 9
        'occluder_on_grnd',  # 10
        'occluder_full',     # 11
        'reflection',        # 12
        'crowd',             # 13
    ]
    distractors = ['person_on_vhcl', 'static_person', 'distractor', 'reflection']
    is_distractor = {i + 1: x in distractors for i, x in enumerate(labels)}
    for i in distractors:
        is_distractor[i] = 1
    seqIni = ConfigParser()
    seqIni.read(inifile, encoding='utf8')
    F = int(seqIni['Sequence']['seqLength'])
    todrop = []
    for t in range(1, F + 1):
        if t not in res.index or t not in gt.index:
            continue
        resInFrame = res.loc[t]

        GTInFrame = gt.loc[t]
        A = GTInFrame[['X', 'Y', 'Width', 'Height']].values
        B = resInFrame[['X', 'Y', 'Width', 'Height']].values
        disM = iou_matrix(A, B, max_iou=0.5)
        le, ri = linear_sum_assignment(disM)
        flags = [
            1 if is_distractor[it['ClassId']] or it['Visibility'] < 0. else 0
            for i, (k, it) in enumerate(GTInFrame.iterrows())
        ]
        hid = [k for k, it in resInFrame.iterrows()]
        for i, j in zip(le, ri):
            if not np.isfinite(disM[i, j]):
                continue
            if flags[i]:
                todrop.append((t, hid[j]))
    ret = res.drop(labels=todrop)
    logging.info('Preprocess take %.3f seconds and remove %d boxes.',
                 time.time() - st, len(todrop))
    return ret

class MOTAccumulator(object):
    """Manage tracking events.

    This class computes per-frame tracking events from a given set of object / hypothesis
    ids and pairwise distances. Indended usage

        import motmetrics as mm
        acc = mm.MOTAccumulator()
        acc.update(['a', 'b'], [0, 1, 2], dists, frameid=0)
        ...
        acc.update(['d'], [6,10], other_dists, frameid=76)
        summary = mm.metrics.summarize(acc)
        print(mm.io.render_summary(summary))

    Update is called once per frame and takes objects / hypothesis ids and a pairwise distance
    matrix between those (see distances module for support). Per frame max(len(objects), len(hypothesis))
    events are generated. Each event type is one of the following
        - `'MATCH'` a match between a object and hypothesis was found
        - `'SWITCH'` a match between a object and hypothesis was found but differs from previous assignment (hypothesisid != previous)
        - `'MISS'` no match for an object was found
        - `'FP'` no match for an hypothesis was found (spurious detections)
        - `'RAW'` events corresponding to raw input
        - `'TRANSFER'` a match between a object and hypothesis was found but differs from previous assignment (objectid != previous)
        - `'ASCEND'` a match between a object and hypothesis was found but differs from previous assignment  (hypothesisid is new)
        - `'MIGRATE'` a match between a object and hypothesis was found but differs from previous assignment  (objectid is new)

    Events are tracked in a pandas Dataframe. The dataframe is hierarchically indexed by (`FrameId`, `EventId`),
    where `FrameId` is either provided during the call to `update` or auto-incremented when `auto_id` is set
    true during construction of MOTAccumulator. `EventId` is auto-incremented. The dataframe has the following
    columns
        - `Type` one of `('MATCH', 'SWITCH', 'MISS', 'FP', 'RAW')`
        - `OId` object id or np.nan when `'FP'` or `'RAW'` and object is not present
        - `HId` hypothesis id or np.nan when `'MISS'` or `'RAW'` and hypothesis is not present
        - `D` distance or np.nan when `'FP'` or `'MISS'` or `'RAW'` and either object/hypothesis is absent

    From the events and associated fields the entire tracking history can be recovered. Once the accumulator
    has been populated with per-frame data use `metrics.summarize` to compute statistics. See `metrics.compute_metrics`
    for a list of metrics computed.

    References
    ----------
    1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
    EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
    2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
    3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene."
    Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.
    """

    def __init__(self, auto_id=False, max_switch_time=float('inf')):
        """Create a MOTAccumulator.

        Params
        ------
        auto_id : bool, optional
            Whether or not frame indices are auto-incremented or provided upon
            updating. Defaults to false. Not specifying a frame-id when this value
            is true results in an error. Specifying a frame-id when this value is
            false also results in an error.

        max_switch_time : scalar, optional
            Allows specifying an upper bound on the timespan an unobserved but
            tracked object is allowed to generate track switch events. Useful if groundtruth
            objects leaving the field of view keep their ID when they reappear,
            but your tracker is not capable of recognizing this (resulting in
            track switch events). The default is that there is no upper bound
            on the timespan. In units of frame timestamps. When using auto_id
            in units of count.
        """

        # Parameters of the accumulator.
        self.auto_id = auto_id
        self.max_switch_time = max_switch_time

        # Accumulator state.
        self._events = None
        self._indices = None
        self.m = None
        self.res_m = None
        self.last_occurrence = None
        self.last_match = None
        self.hypHistory = None
        self.dirty_events = None
        self.cached_events_df = None

        self.reset()

    def reset(self):
        """Reset the accumulator to empty state."""

        self._events = {field: [] for field in _EVENT_FIELDS}
        self._indices = {field: [] for field in _INDEX_FIELDS}
        self.m = {}  # Pairings up to current timestamp
        self.res_m = {}  # Result pairings up to now
        self.last_occurrence = {}  # Tracks most recent occurance of object
        self.last_match = {}  # Tracks most recent match of object
        self.hypHistory = {}
        self.dirty_events = True
        self.cached_events_df = None

    def _append_to_indices(self, frameid, eid):
        self._indices['FrameId'].append(frameid)
        self._indices['Event'].append(eid)

    def _append_to_events(self, typestr, oid, hid, distance):
        self._events['Type'].append(typestr)
        self._events['OId'].append(oid)
        self._events['HId'].append(hid)
        self._events['D'].append(distance)

    def update(self, oids, hids, dists, frameid=None, vf=''):
        """Updates the accumulator with frame specific objects/detections.

        This method generates events based on the following algorithm [1]:
        1. Try to carry forward already established tracks. If any paired object / hypothesis
        from previous timestamps are still visible in the current frame, create a 'MATCH'
        event between them.
        2. For the remaining constellations minimize the total object / hypothesis distance
        error (Kuhn-Munkres algorithm). If a correspondence made contradicts a previous
        match create a 'SWITCH' else a 'MATCH' event.
        3. Create 'MISS' events for all remaining unassigned objects.
        4. Create 'FP' events for all remaining unassigned hypotheses.

        Params
        ------
        oids : N array
            Array of object ids.
        hids : M array
            Array of hypothesis ids.
        dists: NxM array
            Distance matrix. np.nan values to signal do-not-pair constellations.
            See `distances` module for support methods.

        Kwargs
        ------
        frameId : id
            Unique frame id. Optional when MOTAccumulator.auto_id is specified during
            construction.
        vf: file to log details
        Returns
        -------
        frame_events : pd.DataFrame
            Dataframe containing generated events

        References
        ----------
        1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics."
        EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
        """
        # pylint: disable=too-many-locals, too-many-statements

        self.dirty_events = True
        oids = np.asarray(oids)
        oids_masked = np.zeros_like(oids, dtype=np.bool_)
        hids = np.asarray(hids)
        hids_masked = np.zeros_like(hids, dtype=np.bool_)
        dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0]).copy()

        if frameid is None:
            assert self.auto_id, 'auto-id is not enabled'
            if len(self._indices['FrameId']) > 0:
                frameid = self._indices['FrameId'][-1] + 1
            else:
                frameid = 0
        else:
            assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'

        eid = itertools.count()

        # 0. Record raw events

        no = len(oids)
        nh = len(hids)

        # Add a RAW event simply to ensure the frame is counted.
        self._append_to_indices(frameid, next(eid))
        self._append_to_events('RAW', np.nan, np.nan, np.nan)

        # There must be at least one RAW event per object and hypothesis.
        # Record all finite distances as RAW events.
        valid_i, valid_j = np.where(np.isfinite(dists))
        valid_dists = dists[valid_i, valid_j]
        for i, j, dist_ij in zip(valid_i, valid_j, valid_dists):
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', oids[i], hids[j], dist_ij)
        # Add a RAW event for objects and hypotheses that were present but did
        # not overlap with anything.
        used_i = np.unique(valid_i)
        used_j = np.unique(valid_j)
        unused_i = np.setdiff1d(np.arange(no), used_i)
        unused_j = np.setdiff1d(np.arange(nh), used_j)
        for oid in oids[unused_i]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', oid, np.nan, np.nan)
        for hid in hids[unused_j]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', np.nan, hid, np.nan)

        if oids.size * hids.size > 0:
            # 1. Try to re-establish tracks from previous correspondences
            for i in range(oids.shape[0]):
                # No need to check oids_masked[i] here.
                if oids[i] not in self.m:
                    continue

                hprev = self.m[oids[i]]
                j, = np.where(~hids_masked & (hids == hprev))
                if j.shape[0] == 0:
                    continue
                j = j[0]

                if np.isfinite(dists[i, j]):
                    o = oids[i]
                    h = hids[j]
                    oids_masked[i] = True
                    hids_masked[j] = True
                    self.m[oids[i]] = hids[j]

                    self._append_to_indices(frameid, next(eid))
                    self._append_to_events('MATCH', oids[i], hids[j], dists[i, j])
                    self.last_match[o] = frameid
                    self.hypHistory[h] = frameid

            # 2. Try to remaining objects/hypotheses
            dists[oids_masked, :] = np.nan
            dists[:, hids_masked] = np.nan

            rids, cids = linear_sum_assignment(dists)

            for i, j in zip(rids, cids):
                if not np.isfinite(dists[i, j]):
                    continue

                o = oids[i]
                h = hids[j]
                is_switch = (o in self.m and
                             self.m[o] != h and
                             abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
                cat1 = 'SWITCH' if is_switch else 'MATCH'
                if cat1 == 'SWITCH':
                    if h not in self.hypHistory:
                        subcat = 'ASCEND'
                        self._append_to_indices(frameid, next(eid))
                        self._append_to_events(subcat, oids[i], hids[j], dists[i, j])
                # ignore the last condition temporarily
                is_transfer = (h in self.res_m and
                               self.res_m[h] != o)
                # is_transfer = (h in self.res_m and
                #                self.res_m[h] != o and
                #                abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
                cat2 = 'TRANSFER' if is_transfer else 'MATCH'
                if cat2 == 'TRANSFER':
                    if o not in self.last_match:
                        subcat = 'MIGRATE'
                        self._append_to_indices(frameid, next(eid))
                        self._append_to_events(subcat, oids[i], hids[j], dists[i, j])
                    self._append_to_indices(frameid, next(eid))
                    self._append_to_events(cat2, oids[i], hids[j], dists[i, j])
                if vf != '' and (cat1 != 'MATCH' or cat2 != 'MATCH'):
                    if cat1 == 'SWITCH':
                        vf.write('%s %d %d %d %d %d\n' % (subcat[:2], o, self.last_match[o], self.m[o], frameid, h))
                    if cat2 == 'TRANSFER':
                        vf.write('%s %d %d %d %d %d\n' % (subcat[:2], h, self.hypHistory[h], self.res_m[h], frameid, o))
                self.hypHistory[h] = frameid
                self.last_match[o] = frameid
                self._append_to_indices(frameid, next(eid))
                self._append_to_events(cat1, oids[i], hids[j], dists[i, j])
                oids_masked[i] = True
                hids_masked[j] = True
                self.m[o] = h
                self.res_m[h] = o

        # 3. All remaining objects are missed
        for o in oids[~oids_masked]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('MISS', o, np.nan, np.nan)
            if vf != '':
                vf.write('FN %d %d\n' % (frameid, o))

        # 4. All remaining hypotheses are false alarms
        for h in hids[~hids_masked]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('FP', np.nan, h, np.nan)
            if vf != '':
                vf.write('FP %d %d\n' % (frameid, h))

        # 5. Update occurance state
        for o in oids:
            self.last_occurrence[o] = frameid

        return frameid

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulator.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @property
    def mot_events(self):
        df = self.events
        return df[df.Type != 'RAW']

    @staticmethod
    def new_event_dataframe():
        """Create a new DataFrame for event tracking."""
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),          # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=float)),      # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=float)),      # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),      # Distance or NaN when FP or MISS
            ]),
            index=idx
        )
        return df

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """Create a new DataFrame filled with data.

        Params
        ------
        indices: dict
            dict of lists with fields 'FrameId' and 'Event'
        events: dict
            dict of lists with fields 'Type', 'OId', 'HId', 'D'
        """

        if len(events) == 0:
            return MOTAccumulator.new_event_dataframe()

        raw_type = pd.Categorical(
            events['Type'],
            categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'],
            ordered=False)
        series = [
            pd.Series(raw_type, name='Type'),
            pd.Series(events['OId'], dtype=float, name='OId'),
            pd.Series(events['HId'], dtype=float, name='HId'),
            pd.Series(events['D'], dtype=float, name='D')
        ]

        idx = pd.MultiIndex.from_arrays(
            [indices[field] for field in _INDEX_FIELDS],
            names=_INDEX_FIELDS)
        df = pd.concat(series, axis=1)
        df.index = idx
        return df

    @staticmethod
    def merge_analysis(anas, infomap):
        # pylint: disable=missing-function-docstring
        res = {'hyp': {}, 'obj': {}}
        mapp = {'hyp': 'hid_map', 'obj': 'oid_map'}
        for ana, infom in zip(anas, infomap):
            if ana is None:
                return None
            for t in ana.keys():
                which = mapp[t]
                if np.nan in infom[which]:
                    res[t][int(infom[which][np.nan])] = 0
                if 'nan' in infom[which]:
                    res[t][int(infom[which]['nan'])] = 0
                for _id, cnt in ana[t].items():
                    if _id not in infom[which]:
                        _id = str(_id)
                    res[t][int(infom[which][_id])] = cnt
        return res

    @staticmethod
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True, return_mappings=False):
        """Merge dataframes.

        Params
        ------
        dfs : list of pandas.DataFrame or MotAccumulator
            A list of event containers to merge

        Kwargs
        ------
        update_frame_indices : boolean, optional
            Ensure that frame indices are unique in the merged container
        update_oids : boolean, unique
            Ensure that object ids are unique in the merged container
        update_hids : boolean, unique
            Ensure that hypothesis ids are unique in the merged container
        return_mappings : boolean, unique
            Whether or not to return mapping information

        Returns
        -------
        df : pandas.DataFrame
            Merged event data frame
        """

        mapping_infos = []
        new_oid = itertools.count()
        new_hid = itertools.count()

        r = MOTAccumulator.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulator):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                # pylint: disable=cell-var-from-loop
                next_frame_id = max(r.index.get_level_values(0).max() + 1, r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                # pylint: disable=cell-var-from-loop
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                # pylint: disable=cell-var-from-loop
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = r.append(copy)
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r

def norm2squared_matrix(objs, hyps, max_d2=float('inf')):
    """Computes the squared Euclidean distance matrix between object and hypothesis points.

    Params
    ------
    objs : NxM array
        Object points of dim M in rows
    hyps : KxM array
        Hypothesis points of dim M in rows

    Kwargs
    ------
    max_d2 : float
        Maximum tolerable squared Euclidean distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to +inf

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0, 0))

    assert hyps.shape[1] == objs.shape[1], "Dimension mismatch"

    delta = objs[:, np.newaxis] - hyps[np.newaxis, :]
    C = np.sum(delta ** 2, axis=-1)

    C[C > max_d2] = np.nan
    return C


def rect_min_max(r):
    min_pt = r[..., :2]
    size = r[..., 2:]
    max_pt = min_pt + size
    return min_pt, max_pt


def boxiou(a, b):
    """Computes IOU of two rectangles."""
    a_min, a_max = rect_min_max(a)
    b_min, b_max = rect_min_max(b)
    # Compute intersection.
    i_min = np.maximum(a_min, b_min)
    i_max = np.minimum(a_max, b_max)
    i_size = np.maximum(i_max - i_min, 0)
    i_vol = np.prod(i_size, axis=-1)
    # Get volume of union.
    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = np.prod(a_size, axis=-1)
    b_vol = np.prod(b_size, axis=-1)
    u_vol = a_vol + b_vol - i_vol
    return np.where(i_vol == 0, np.zeros_like(i_vol, dtype=np.float64),
                    math_util.quiet_divide(i_vol, u_vol))


def iou_matrix(objs, hyps, max_iou=1.):
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.

    The IoU is computed as

        IoU(a,b) = 1. - isect(a, b) / union(a, b)

    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.

    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asfarray(objs)
    hyps = np.asfarray(hyps)
    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4
    iou = boxiou(objs[:, None], hyps[None, :])
    dist = 1 - iou
    return np.where(dist > max_iou, np.nan, dist)

def compare_to_groundtruth(gt, dt, dist='iou', distfields=None, distth=0.5):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union ('iou'). Euclidean
        distance ('euc') and squared euclidean distance ('seuc') are also supported.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return np.sqrt(norm2squared_matrix(a, b, max_d2=distth**2))

    def compute_seuc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    if dist.upper() == 'IOU':
        compute_dist = compute_iou
    elif dist.upper() == 'EUC':
        compute_dist = compute_euc
    elif dist.upper() == 'SEUC':
        compute_dist = compute_seuc
    else:
        raise f'Unknown distance metric {dist}. Use "IOU", "EUC" or "SEUC"'

    acc = MOTAccumulator()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]

    gt = gt[distfields]
    dt = dt[distfields]
    fid_to_fgt = dict(iter(gt.groupby('FrameId')))
    fid_to_fdt = dict(iter(dt.groupby('FrameId')))

    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))
        if fid in fid_to_fgt:
            fgt = fid_to_fgt[fid]
            oids = fgt.index.get_level_values('Id')
        if fid in fid_to_fdt:
            fdt = fid_to_fdt[fid]
            hids = fdt.index.get_level_values('Id')
        if len(oids) > 0 and len(hids) > 0:
            dists = compute_dist(fgt.values, fdt.values)
        acc.update(oids, hids, dists, frameid=fid)

    return acc


def CLEAR_MOT_M(gt, dt, inifile, dist='iou', distfields=None, distth=0.5, include_all=False, vflag=''):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulatorCustom()
    dt = preprocessResult(dt, gt, inifile)
    if include_all:
        gt = gt[gt['Confidence'] >= 0.99]
    else:
        gt = gt[(gt['Confidence'] >= 0.99) & (gt['ClassId'] == 1)]
    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]
    analysis = {'hyp': {}, 'obj': {}}
    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))

        if fid in gt.index:
            fgt = gt.loc[fid]
            oids = fgt.index.values
            for oid in oids:
                oid = int(oid)
                if oid not in analysis['obj']:
                    analysis['obj'][oid] = 0
                analysis['obj'][oid] += 1

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values
            for hid in hids:
                hid = int(hid)
                if hid not in analysis['hyp']:
                    analysis['hyp'][hid] = 0
                analysis['hyp'][hid] += 1

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)

        acc.update(oids, hids, dists, frameid=fid, vf=vflag)

    return acc, analysis


class MOTAccumulatorCustom(MOTAccumulator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """
        Create a new DataFrame filled with data.
        This version overwrites the original in MOTAccumulator achieves about 2x speedups.

        Params
        ------
        indices: list
            list of tuples (frameid, eventid)
        events: list
            list of events where each event is a list containing
            'Type', 'OId', HId', 'D'
        """
        idx = pd.MultiIndex.from_tuples(indices, names=['FrameId', 'Event'])
        df = pd.DataFrame(events, index=idx, columns=['Type', 'OId', 'HId', 'D'])
        return df

    @staticmethod
    def new_event_dataframe():
        """ Create a new DataFrame for event tracking. """
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),  # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=object)),
                # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=object)),
                # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),  # Distance or NaN when FP or MISS
            ]),
            index=idx
        )
        return df

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @staticmethod
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True,
                               return_mappings=False):
        """Merge dataframes.

        Params
        ------
        dfs : list of pandas.DataFrame or MotAccumulator
            A list of event containers to merge

        Kwargs
        ------
        update_frame_indices : boolean, optional
            Ensure that frame indices are unique in the merged container
        update_oids : boolean, unique
            Ensure that object ids are unique in the merged container
        update_hids : boolean, unique
            Ensure that hypothesis ids are unique in the merged container
        return_mappings : boolean, unique
            Whether or not to return mapping information

        Returns
        -------
        df : pandas.DataFrame
            Merged event data frame
        """

        mapping_infos = []
        new_oid = count()
        new_hid = count()

        r = MOTAccumulatorCustom.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulatorCustom):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                next_frame_id = max(r.index.get_level_values(0).max() + 1,
                                    r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = pd.concat((r, copy))
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r
