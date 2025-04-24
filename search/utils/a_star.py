from collections.abc import Callable
from typing import Any, Optional
import time

try:
    from .abstract import WeightedGraph, PriorityQueue, Node
except:
    from utils.abstract import WeightedGraph, PriorityQueue, Node
import random


def a_star_search(
    graph: WeightedGraph,
    start: Node,
    goal: Node,
    heuristic: Callable[[Node, Node], float],
    max_time : float = 3.,
    ):
        
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: dict[Node, Optional[Node]] = {}
    cost_so_far: dict[Node, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    start = time.time()
    
    while not frontier.empty():
        if time.time() - start > max_time:
            return {}, {}
            
        current: Node = frontier.get()
        
        if current == goal:
            break
        
        for next_hash in graph.neighbors(current):
            next = graph.revert_hash(next_hash)
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def reconstruct_path(came_from: dict[Node, Node],
                     start: Node, goal: Node) -> list[Node]:

    current: Node = goal
    path: list[Node] = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path