import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from collections import defaultdict

class TreeNode:
    """Represents a node in the tree."""
    def __init__(self, node_id: int, properties: Dict[str, Union[np.ndarray, str]]):
        self.id = node_id
        self.properties = properties
        self.parent: Optional['TreeNode'] = None
        self.children: List['TreeNode'] = []

class TreeEdge:
    """Represents an edge in the tree."""
    def __init__(self, start_node: TreeNode, end_node: TreeNode, 
                 properties: Dict[str, Union[np.ndarray, str]]):
        self.start = start_node
        self.end = end_node
        self.properties = properties

class Tree:
    """
    A tree data structure where all nodes and edges have uniform property schemas.
    
    Node data is saved both in a dictionary for id-based access and in pre-allocated
    numpy arrays for efficient concatenated access.

    Features:
    - All nodes have the same properties (numpy arrays or strings)
    - All edges have the same properties  
    - Efficient concatenated property access without copying
    - Pre-allocated arrays that grow automatically
    - Full tree traversal and query functionality
    """
    
    def __init__(self, node_property_schema: Dict[str, type], 
                 edge_property_schema: Dict[str, type], 
                 initial_capacity: int = 1000, growth_factor: float = 1.5):
        """
        Initialize tree with property schemas.
        
        Args:
            node_property_schema: Dict mapping property names to types (np.ndarray or str)
            edge_property_schema: Dict mapping property names to types (np.ndarray or str)
            initial_capacity: Initial capacity for pre-allocated arrays
            growth_factor: Factor by which to grow arrays when capacity is exceeded
        """
        self.node_property_schema = node_property_schema
        self.edge_property_schema = edge_property_schema
        self.nodes: Dict[int, TreeNode] = {}
        self.edges: List[TreeEdge] = []
        self.edge_map: Dict[Tuple[int, int], TreeEdge] = {}  # (parent_id, child_id) -> edge
        self.root: Optional[TreeNode] = None
        
        # Pre-allocated array management
        self.initial_capacity = initial_capacity
        self.growth_factor = growth_factor
        self.current_capacity = initial_capacity
        self.node_count = 0
        
        # Pre-allocated arrays for concatenated properties (no-copy access)
        self._node_property_arrays: Dict[str, np.ndarray] = {}
        self._node_id_to_index: Dict[int, int] = {}  # Maps node_id to array index
        self._index_to_node_id: Dict[int, int] = {}  # Maps array index to node_id
        
        # Initialize arrays - will be set up on first node addition
        self._arrays_initialized = False
    
    def add_node(self, node_id: int, properties: Dict[str, Union[np.ndarray, str]], 
                 parent_id: Optional[int] = None) -> TreeNode:
        """
        Add a node to the tree.
        
        Args:
            node_id: Unique identifier for the node
            properties: Dict of node properties matching the schema
            parent_id: ID of parent node (None for root)
            
        Returns:
            The created TreeNode
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        
        # Validate properties match schema
        self._validate_node_properties(properties)
        
        # Initialize arrays on first node
        if not self._arrays_initialized:
            self._initialize_arrays(properties)
        
        # Check if we need to grow arrays
        if self.node_count >= self.current_capacity:
            self._grow_arrays()
        
        # Create node
        node = TreeNode(node_id, properties)
        self.nodes[node_id] = node
        
        # Set up parent-child relationships
        if parent_id is None:
            if self.root is not None:
                raise ValueError("Tree already has a root node")
            self.root = node
        else:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent node {parent_id} does not exist")
            parent = self.nodes[parent_id]
            node.parent = parent
            parent.children.append(node)
        
        # Store node properties in pre-allocated arrays
        array_index = self.node_count
        self._node_id_to_index[node_id] = array_index
        self._index_to_node_id[array_index] = node_id
        
        for prop_name, value in properties.items():
            if self.node_property_schema[prop_name] == np.ndarray:
                self._node_property_arrays[prop_name][array_index] = value
        
        self.node_count += 1
        return node
    
    def add_edge_properties(self, parent_id: int, child_id: int, 
                           properties: Dict[str, Union[np.ndarray, str]]) -> TreeEdge:
        """
        Add properties to the edge between parent and child nodes.
        
        Args:
            parent_id: ID of parent node
            child_id: ID of child node  
            properties: Dict of edge properties matching the schema
            
        Returns:
            The created TreeEdge
        """
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError("Both nodes must exist before adding edge properties")
        
        parent_node = self.nodes[parent_id]
        child_node = self.nodes[child_id]
        
        # Verify parent-child relationship exists
        if child_node not in parent_node.children:
            raise ValueError(f"No parent-child relationship between {parent_id} and {child_id}")
        
        # Validate properties match schema
        self._validate_edge_properties(properties)
        
        edge = TreeEdge(parent_node, child_node, properties)
        self.edges.append(edge)
        self.edge_map[(parent_id, child_id)] = edge
        
        return edge
    
    def get_node_property_concatenated(self, property_name: str) -> np.ndarray:
        """
        Get concatenated property values for all nodes without making copies.
        
        Args:
            property_name: Name of the property to concatenate
            
        Returns:
            Numpy array view of concatenated property values (only valid entries)
        """
        if property_name not in self.node_property_schema:
            raise ValueError(f"Property '{property_name}' not in node schema")
        
        if self.node_property_schema[property_name] != np.ndarray:
            raise ValueError(f"Property '{property_name}' is not a numpy array")
        
        if not self._arrays_initialized:
            raise ValueError("No nodes added yet")
        
        # Return view of only the valid entries (up to node_count)
        return self._node_property_arrays[property_name][:self.node_count]
    
    def get_children(self, node_id: int) -> List[TreeNode]:
        """Get all children of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        return self.nodes[node_id].children.copy()
    
    def get_parent(self, node_id: int) -> Optional[TreeNode]:
        """Get parent of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        return self.nodes[node_id].parent
    
    def get_ancestors(self, node_id: int) -> List[TreeNode]:
        """Get all ancestors of a node (from parent to root)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        ancestors = []
        current = self.nodes[node_id].parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self, node_id: int) -> List[TreeNode]:
        """Get all descendants of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        descendants = []
        self._collect_descendants(self.nodes[node_id], descendants)
        return descendants
    
    def get_edge(self, parent_id: int, child_id: int) -> Optional[TreeEdge]:
        """Get edge between two nodes."""
        return self.edge_map.get((parent_id, child_id))
    
    def get_edge_start(self, parent_id: int, child_id: int) -> Optional[TreeNode]:
        """Get start node of an edge."""
        edge = self.get_edge(parent_id, child_id)
        return edge.start if edge else None
    
    def get_edge_end(self, parent_id: int, child_id: int) -> Optional[TreeNode]:
        """Get end node of an edge."""
        edge = self.get_edge(parent_id, child_id)
        return edge.end if edge else None
    
    def get_path_to_root(self, node_id: int) -> List[TreeNode]:
        """Get path from node to root."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        path = []
        current = self.nodes[node_id]
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def get_depth(self, node_id: int) -> int:
        """Get depth of a node (root has depth 0)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        depth = 0
        current = self.nodes[node_id].parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def is_leaf(self, node_id: int) -> bool:
        """Check if node is a leaf."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        return len(self.nodes[node_id].children) == 0
    
    def get_node_property_by_id(self, node_id: int, property_name: str) -> Union[np.ndarray, str]:
        """
        Get a specific property of a specific node from the pre-allocated arrays.
        
        Args:
            node_id: ID of the node
            property_name: Name of the property
            
        Returns:
            Property value (numpy array or string)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        if property_name not in self.node_property_schema:
            raise ValueError(f"Property '{property_name}' not in node schema")
        
        if self.node_property_schema[property_name] == np.ndarray:
            if not self._arrays_initialized:
                raise ValueError("Arrays not initialized")
            array_index = self._node_id_to_index[node_id]
            return self._node_property_arrays[property_name][array_index].copy()
        else:
            # For string properties, get from node object
            return self.nodes[node_id].properties[property_name]
    
    def update_node_property(self, node_id: int, property_name: str, 
                           value: Union[np.ndarray, str]):
        """
        Update a property of a specific node.
        
        Args:
            node_id: ID of the node
            property_name: Name of the property
            value: New property value
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        if property_name not in self.node_property_schema:
            raise ValueError(f"Property '{property_name}' not in node schema")
        
        # Validate property type
        expected_type = self.node_property_schema[property_name]
        if expected_type == np.ndarray and not isinstance(value, np.ndarray):
            raise ValueError(f"Property '{property_name}' should be numpy array")
        elif expected_type == str and not isinstance(value, str):
            raise ValueError(f"Property '{property_name}' should be string")
        
        # Update in node object
        self.nodes[node_id].properties[property_name] = value
        
        # Update in pre-allocated array if it's a numpy array
        if expected_type == np.ndarray and self._arrays_initialized:
            array_index = self._node_id_to_index[node_id]
            self._node_property_arrays[property_name][array_index] = value
    
    def _validate_node_properties(self, properties: Dict[str, Union[np.ndarray, str]]):
        """Validate node properties match schema."""
        if set(properties.keys()) != set(self.node_property_schema.keys()):
            raise ValueError("Node properties don't match schema")
        
        for name, value in properties.items():
            expected_type = self.node_property_schema[name]
            if expected_type == np.ndarray and not isinstance(value, np.ndarray):
                raise ValueError(f"Property '{name}' should be numpy array")
            elif expected_type == str and not isinstance(value, str):
                raise ValueError(f"Property '{name}' should be string")
    
    def _validate_edge_properties(self, properties: Dict[str, Union[np.ndarray, str]]):
        """Validate edge properties match schema."""
        if set(properties.keys()) != set(self.edge_property_schema.keys()):
            raise ValueError("Edge properties don't match schema")
        
        for name, value in properties.items():
            expected_type = self.edge_property_schema[name]
            if expected_type == np.ndarray and not isinstance(value, np.ndarray):
                raise ValueError(f"Property '{name}' should be numpy array")
            elif expected_type == str and not isinstance(value, str):
                raise ValueError(f"Property '{name}' should be string")
    
    def _collect_descendants(self, node: TreeNode, descendants: List[TreeNode]):
        """Recursively collect all descendants."""
        for child in node.children:
            descendants.append(child)
            self._collect_descendants(child, descendants)
    
    def _initialize_arrays(self, sample_properties: Dict[str, Union[np.ndarray, str]]):
        """Initialize pre-allocated arrays based on the first node's properties."""
        for prop_name, prop_type in self.node_property_schema.items():
            if prop_type == np.ndarray:
                sample_array = sample_properties[prop_name]
                # Create pre-allocated array with the same shape and dtype
                if sample_array.ndim == 1:
                    # For 1D arrays, pre-allocate as 2D: (capacity, original_size)
                    array_shape = (self.current_capacity, sample_array.shape[0])
                else:
                    # For multi-dimensional arrays, add capacity as first dimension
                    array_shape = (self.current_capacity,) + sample_array.shape
                
                self._node_property_arrays[prop_name] = np.empty(
                    array_shape, dtype=sample_array.dtype
                )
        
        self._arrays_initialized = True
    
    def _grow_arrays(self):
        """Grow pre-allocated arrays when capacity is exceeded."""
        new_capacity = int(self.current_capacity * self.growth_factor)
        
        for prop_name, old_array in self._node_property_arrays.items():
            # Create new larger array
            new_shape = (new_capacity,) + old_array.shape[1:]
            new_array = np.empty(new_shape, dtype=old_array.dtype)
            
            # Copy existing data
            new_array[:self.node_count] = old_array[:self.node_count]
            
            # Replace old array
            self._node_property_arrays[prop_name] = new_array
        
        self.current_capacity = new_capacity
        print(f"Arrays grown to capacity: {new_capacity}")  # Optional: remove in production


# Example usage and testing
if __name__ == "__main__":
    # Define schemas
    node_schema = {
        'position': np.ndarray,
        'velocity': np.ndarray, 
        'name': str
    }
    
    edge_schema = {
        'weight': np.ndarray,
        'label': str
    }
    
    # Create tree with small initial capacity to test growth
    tree = Tree(node_schema, edge_schema, initial_capacity=2, growth_factor=2.0)
    
    # Add nodes
    root = tree.add_node(0, {
        'position': np.array([0.0, 0.0, 0.0]),
        'velocity': np.array([1.0, 0.0]),
        'name': 'root'
    })
    
    child1 = tree.add_node(1, {
        'position': np.array([1.0, 1.0, 1.0]),
        'velocity': np.array([0.5, 1.0]), 
        'name': 'child1'
    }, parent_id=0)
    
    child2 = tree.add_node(2, {
        'position': np.array([2.0, 2.0, 2.0]),
        'velocity': np.array([0.0, 0.5]),
        'name': 'child2'  
    }, parent_id=0)
    
    # This should trigger array growth (capacity was 2, now adding 4th node)
    grandchild = tree.add_node(3, {
        'position': np.array([3.0, 3.0, 3.0]),
        'velocity': np.array([1.5, 1.5]),
        'name': 'grandchild'
    }, parent_id=1)
    
    # Add more nodes to test continued growth
    for i in range(4, 8):
        tree.add_node(i, {
            'position': np.array([float(i), float(i), float(i)]),
            'velocity': np.array([float(i)*0.1, float(i)*0.1]),
            'name': f'node_{i}'
        }, parent_id=0)
    
    # Add edge properties
    tree.add_edge_properties(0, 1, {
        'weight': np.array([0.8]),
        'label': 'strong'
    })
    
    tree.add_edge_properties(0, 2, {
        'weight': np.array([0.3]),
        'label': 'weak'
    })
    
    # Test functionality
    print("All positions concatenated (zero-copy):")
    positions = tree.get_node_property_concatenated('position')
    print(positions)
    print(f"Shape: {positions.shape}")
    
    print("\nAll velocities concatenated (zero-copy):")
    velocities = tree.get_node_property_concatenated('velocity')
    print(velocities)
    print(f"Shape: {velocities.shape}")
    
    # Test property access and updates
    print(f"\nOriginal position of node 1: {tree.get_node_property_by_id(1, 'position')}")
    tree.update_node_property(1, 'position', np.array([10.0, 10.0, 10.0]))
    print(f"Updated position of node 1: {tree.get_node_property_by_id(1, 'position')}")
    print(f"Position in concatenated array: {tree.get_node_property_concatenated('position')[1]}")
    
    print(f"\nChildren of root: {[child.id for child in tree.get_children(0)]}")
    print(f"Parent of child1: {tree.get_parent(1).id if tree.get_parent(1) else None}")
    print(f"Ancestors of grandchild: {[a.id for a in tree.get_ancestors(3)]}")
    print(f"Descendants of root: {[d.id for d in tree.get_descendants(0)]}")
    print(f"Path to root from grandchild: {[n.id for n in tree.get_path_to_root(3)]}")
    print(f"Depth of grandchild: {tree.get_depth(3)}")
    print(f"Is child2 a leaf? {tree.is_leaf(2)}")
    
    # Test edge queries
    edge = tree.get_edge(0, 1)
    if edge:
        print(f"\nEdge (0->1) weight: {edge.properties['weight']}")
        print(f"Edge (0->1) label: {edge.properties['label']}")
    
    print(f"\nTotal nodes: {tree.node_count}")
    print(f"Current array capacity: {tree.current_capacity}")