class Node:
    def __init__(self, data, left=None, right=None, parent=None):

        """
            The Node class defines a tree node, which has a data attribute and two child nodes: left and right.
        """

        self.data = data
        self.left = left
        self.right = right
        self.parent = parent
    
    def is_leaf(self):

        """
            The is_leaf method returns True if the node has no child nodes.
        """

        return self.left is None and self.right is None

def build_tree(subtree, parent=None):

    """
        The build_tree function takes a subtree, which can be a single integer (in the case of a leaf node)
        or a tuple of two subtrees and a link count (in the case of an internal node).
    """

    if isinstance(subtree, int):
        # base case: leaf node
        return Node(subtree, parent=parent)
    else:
        # recursive case: internal node
        left_subtree, right_subtree, link_count = subtree
        left_node = build_tree(left_subtree, parent=link_count)
        right_node = build_tree(right_subtree, parent=link_count)

        return Node(link_count, left=left_node, right=right_node, parent=parent)
    
def likelihood(D, p):

    """
        Calculates the likelihood of a given tree topology, D, with a set of branch lengths, p.
    """

    # Inizialize likelihood
    likelihood = 1.0
    
    # Iterate over the nodes in the tree
    for r in D:
        # Count the sizes of the subtrees rooted at the node
        left_size, right_size = count_subtree_sizes(r)
        
        # Count the number of links in the subtree rooted at the node
        E_r = count_links_in_subtree(r)
        
        # Calculate the probability of observing the subtree rooted at r given the branch length p[r]
        likelihood *= p[r] ** E_r * (1 - p[r]) ** (left_size * right_size - E_r)
    
    return likelihood

def count_links_in_subtree(node):

    """
        Counts the number of links in the subtree rooted at `node`.
    """
    
    stack = [node]  # initialize the stack with the root node
    links = {}  # create an empty dictionary to store the link counts
    
    while stack:
        curr = stack.pop()  # get the current node from the stack
        
        if curr.is_leaf():  # if the current node is a leaf, its link count is 0
            links[curr] = 0
        
        else:
            left_links = links.get(curr.left, None)  # get the link count of the left child, if it exists
            right_links = links.get(curr.right, None)  # get the link count of the right child, if it exists
            
            if left_links is None:  # if the left child's link count has not been computed yet, add it to the stack
                stack.append(curr)
                stack.append(curr.left)
            
            elif right_links is None:  # if the right child's link count has not been computed yet, add it to the stack
                stack.append(curr)
                stack.append(curr.right)
            
            else:  # if both children's link counts have been computed, compute the link count of the current node and store it in the dictionary
                links[curr] = left_links + right_links + curr.data
    
    return links[node]  # return the link count of the root node

def count_subtree_sizes(node):

    """
        Returns the sizes of the left and right subtrees rooted at `node`.
    """

    stack = [node]  # Initialize stack with the root node
    sizes = {}  # Create a dictionary to store subtree sizes

    while stack:
        curr = stack.pop()  # Pop the current node from the stack

        if curr.is_leaf():
            # If the current node is a leaf node, set its subtree size to (0, 0)
            sizes[curr] = (0, 0)
        
        else:
            # Check if the subtree sizes of the left and right child nodes have been computed
            left_size = sizes.get(curr.left, None)
            right_size = sizes.get(curr.right, None)
            
            if left_size is None:
                # If the size of the left subtree has not been computed, push the current node and its left child onto the stack
                stack.append(curr)
                stack.append(curr.left)
            
            elif right_size is None:
                # If the size of the right subtree has not been computed, push the current node and its right child onto the stack
                stack.append(curr)
                stack.append(curr.right)
            
            else:
                # If the sizes of both subtrees have been computed, compute the size of the current subtree and store it in the dictionary
                sizes[curr] = (left_size[0] + left_size[1] + 1, right_size[0] + right_size[1] + 1)
    
    return sizes[node]  # Return the size of the subtree rooted at the input node


def count_nodes(node):

    """
        Counts the number of nodes in the subtree rooted at `node`.
    """

    if node is None:
        return 0

    stack = [node]  # Initialize stack with the root node
    count = 0  # Initialize node count to 0

    while stack:
        current = stack.pop()  # Pop the current node from the stack
        count += 1  # Increment the node count

        if current.left is not None:
            stack.append(current.left)  # Push the left child onto the stack

        if current.right is not None:
            stack.append(current.right)  # Push the right child onto the stack

    return count  # Return the node count

def main():

    # Set the dendogram 
    D = [(1, (2, 3, 2), 1), ((4, 5, 3), 6, 2)]

    # Convert to tree structure with Node objects
    D = [build_tree(subtree) for subtree in D]

    # Set probability dictionary for all node
    p = {node: 0.7 for node in D}

    # Sets the probability of the root node to 0.5 and the probability of all other nodes to 0.8.
    p2 = {node: 0.5 if node.parent is None else 0.8 for node in D}

    # Compute likelihood
    likelihood2 = likelihood(D, p2)
    print(likelihood2)


if __name__ == '__main__':
    main()