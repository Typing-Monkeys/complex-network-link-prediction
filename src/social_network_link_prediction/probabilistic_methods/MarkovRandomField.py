# DA RICONTROLLARE SE è GIUSTO

class MarkovRandomField:
    def __init__(self):
        self.nodes = set() # Insieme dei nodi del grafo
        self.constraints = set() # Insieme dei vincoli tra i nodi del grafo

    def add_node(self, node):
        self.nodes.add(node)

    def add_constraint(self, constraint):
        self.constraints.add(constraint)

    def enforce_constraint(self, constraint):
        # Impone il vincolo specificato, ossia che i valori delle variabili coinvolte siano consistenti tra loro
        for node in constraint:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph")
        for value in constraint.values():
            if value not in node.values():
                raise ValueError(f"Value {value} is not allowed for this constraint")
        # Trova tutte le coppie di nodi coinvolte nel vincolo
        node_pairs = set()
        for node in constraint:
            for neighbor in self.get_neighbors(node):
                if neighbor in constraint:
                    node_pairs.add(frozenset((node, neighbor)))
        # Calcola il supporto del vincolo (cioè quante volte è soddisfatto)
        support = 0
        for assignment in self.enumerate_assignments(constraint):
            if all(self.constraint_satisfied(pair, assignment) for pair in node_pairs):
                support += 1
        # Se il vincolo è impossibile da soddisfare, solleva un'eccezione
        if support == 0:
            raise ValueError("Constraint is unsatisfiable")
        # Aggiorna i pesi degli archi tra i nodi coinvolte nel vincolo
        for pair in node_pairs:
            weight = support / len(self.enumerate_assignments(constraint))
            self.set_edge_weight(pair, weight)

    def constraints_satisfied(self):
        # Verifica se tutti i vincoli del grafo sono soddisfatti
        return all(self.constraint_satisfied(constraint) for constraint in self.constraints)

    def get_edge_weight(self, node1, node2):
        # Restituisce il peso dell'arco tra due nodi
        if not self.has_edge(node1, node2):
            raise ValueError(f"No edge between nodes {node1} and {node2}")
        return self.weights[frozenset((node1, node2))]

    def set_edge_weight(self, pair, weight):
        # Imposta il peso dell'arco tra due nodi
        if not all(node in self.nodes for node in pair):
            raise ValueError(f"Nodes {pair} are not both in the graph")
        self.weights[pair] = weight

    def has_edge(self, node1, node2):
        # Verifica se esiste un arco tra due nodi
        return frozenset((node1, node2)) in self.weights

    def get_neighbors(self, node):
        # Restituisce gli archi uscenti dal nodo
        return {neighbor for neighbor in self.nodes if self.has_edge(node, neighbor)}
