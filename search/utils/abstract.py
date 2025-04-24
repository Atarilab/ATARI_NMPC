from abc import ABC, abstractmethod
from typing import Any
import heapq

Node = Any   

class Graph(ABC):
    def __init__(self):
        self.edges: dict[Node, list[Node]] = {}
        
    def neighbors(self, node: Node) -> list[Node]:
        if node not in self.edges:
            neighbors = self.get_neighbors(node)
            self.edges[node] = neighbors
            return neighbors

        return self.edges[node]
    
    @abstractmethod
    def get_neighbors(self, node: Node) -> list[Node]:
        pass

class WeightedGraph(Graph):
    @abstractmethod
    def cost(self, node_from: Node, node_to: Node) -> float: 
        pass
    
class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, Any]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: Any, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> Any:
        return heapq.heappop(self.elements)[1]