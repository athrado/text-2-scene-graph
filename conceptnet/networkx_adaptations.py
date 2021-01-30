# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# networkx_adapations.py
# ----------------
# Adapt networkx function for all_shortest_paths 
# including a cutoff to reduce processing time.


# Imports
import networkx as nx

def all_shortest_paths(G, source, target, weight=None, method='dijkstra', cutoff=None):
    """Compute all shortest paths in the graph.

    Modified to include cutoff - but only for method='unweighted';
    TODO: Add cutoff to other methods or remove other methods from this function.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    target : node
       Ending node for path.

    weight : None or string, optional (default = None)
       If None, every edge has weight/distance/cost 1.
       If a string, use this edge attribute as the edge weight.
       Any edge attribute not present defaults to 1.

    method : string, optional (default = 'dijkstra')
       The algorithm to use to compute the path lengths.
       Supported options: 'dijkstra', 'bellman-ford'.
       Other inputs produce a ValueError.
       If `weight` is None, unweighted graph methods are used, and this
       suggestion is ignored.

    cutoff : TO DOCUMENT!

    Returns
    -------
    paths : generator of lists
        A generator of all paths between source and target.

    Raises
    ------
    ValueError
        If `method` is not among the supported options.

    NetworkXNoPath
        If `target` cannot be reached from `source`.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2])
    >>> nx.add_path(G, [0, 10, 2])
    >>> print([p for p in nx.all_shortest_paths(G, source=0, target=2)])
    [[0, 1, 2], [0, 10, 2]]

    Notes
    -----
    There may be many shortest paths between the source and target.

    See Also
    --------
    shortest_path()
    single_source_shortest_path()
    all_pairs_shortest_path()
    """
    method = 'unweighted' if weight is None else method
    if method == 'unweighted':
        pred = nx.predecessor(G, source, cutoff=cutoff if cutoff is None else cutoff-1)
    elif method == 'dijkstra':
        pred, dist = nx.dijkstra_predecessor_and_distance(G, source,
                                                          weight=weight)
    elif method == 'bellman-ford':
        pred, dist = nx.bellman_ford_predecessor_and_distance(G, source,
                                                              weight=weight)
    else:
        raise ValueError('method not supported: {}'.format(method))

    if target not in pred:
        raise nx.NetworkXNoPath('Target {} cannot be reached'
                                'from Source {}'.format(target, source))

    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node == source:
            yield [p for p, n in reversed(stack[:top + 1])]
        if len(pred[node]) > i:
            top += 1
            if top == len(stack):
                stack.append([pred[node][i], 0])
            else:
                stack[top] = [pred[node][i], 0]
        else:
            stack[top - 1][1] += 1
            top -= 1
