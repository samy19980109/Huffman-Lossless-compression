"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}  # the frequency dictionary that we'll return
    for byte in text:
        if byte in d:
            d[byte] += 1
        else:
            d[byte] = 1
    return d


def insert_in_priority_queue(q, frequency, node):
    """A utility function that inserts the element (frequency, node)
    the the priority queue q. The element is inserted so that after
    the insertion, the queue is order along the increasing frequencies.

    @param list q: a priority queue
    @param int frequency: the frequency of the symbol of the node
    @param HuffmanNode|None node: a node
    @rtype: NoneType

    >>> q = []
    >>> insert_in_priority_queue(q, 10, None)
    >>> q
    [(10, None)]
    >>> q = [(10, None), (20, None), (40, None)]
    >>> insert_in_priority_queue(q, 30, None)
    >>> q
    [(10, None), (20, None), (30, None), (40, None)]
    >>> q = [(10, None), (20, None), (40, None)]
    >>> insert_in_priority_queue(q, 50, None)
    >>> q
    [(10, None), (20, None), (40, None), (50, None)]
    """
    # find the index where to insert
    index = 0
    while index < len(q) and frequency >= q[index][0]:
        index += 1

    # insert the element after a right shift
    q.append(None)  # add a new position in the queue
    q[index + 1:] = q[index:-1]  # right shift
    q[index] = (frequency, node)  # insert the element

    return None


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    # create a priority queue (as a list) used to construct the tree
    q = []

    # create a leaf node for each symbol and add it to q
    for symbol in freq_dict:
        node = HuffmanNode(symbol=symbol, left=None, right=None)
        insert_in_priority_queue(q, freq_dict[symbol], node)

    # while there is more than one element in q, we keep constructing the tree
    while len(q) > 1:
        # remove the 2 first nodes from q
        freq1, node1 = q.pop(0)
        freq2, node2 = q.pop(0)

        # create an internal node with these two as its children
        internal_node = HuffmanNode(symbol=None, left=node1, right=node2)

        # add the internal node to the queue
        insert_in_priority_queue(q, freq1 + freq2, internal_node)

    # Here, there is only one node remaining in q,
    # it's the root of the Huffman tree
    return q.pop(0)[1]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    # we use a recursive algorithm. If tree is a
    # leaf node, we return "" for its symbol
    if tree.is_leaf():
        return {tree.symbol: ""}
    else:
        # We are in an internal node. First, we get the codes of to the
        # left and right children. Then, we append "0" at the beginning of every
        # code in the left child, and a "1" to the codes of the right child
        d = {}  # the dict to  return
        left_dict = get_codes(tree.left)  # symbols from the left child
        right_dict = get_codes(tree.right)  # symbols from the right child
        for symbol in left_dict:
            # append 0 to the beginning of each symbol in the left child
            d[symbol] = "0" + left_dict[symbol]
        for symbol in right_dict:
            # append 1 to the beginning of each symbol in the right child
            d[symbol] = "1" + right_dict[symbol]
        return d


def number_the_nodes(tree, number=0):
    """
    A helper function
    Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @param number: the number from where we start the numbering.
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_the_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # if there is only one symbol
    if tree.is_leaf():
        return None

    # Since it's a postorder traversal, we number the left child, then the right
    # child, then the current node
    # we first check if the left child is a leaf
    if tree.left.is_leaf():
        # we go number the right child. we check if it's a leaf
        if tree.right.is_leaf():
            tree.number = number
        else:
            # right child is not a leaf, we number it recursivly
            number_the_nodes(tree.right, number)
            # then we number the parent node
            tree.number = tree.right.number + 1
    else:
        # left child is not a leaf, we number it recursivly
        number_the_nodes(tree.left, number)
        # we update the next number we'll use for numbering
        number = tree.left.number + 1
        # we check if the right child is a leaf
        if tree.right.is_leaf():
            # we number directly this node
            tree.number = number
        else:
            # right child is not a leaf, we number it recursivly
            number_the_nodes(tree.right, number)
            # then we number the parent node
            tree.number = tree.right.number + 1
    return None


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    number = 0
    number_the_nodes(tree, number)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.
    the average number of bits per symbol

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    # we first get the codes corresponding to the tree
    codes = get_codes(tree)

    # we calculate the average length taking into account
    # the length of a symbol and its frequency
    s = 0
    d = 0
    for symbol in codes:
        s += len(codes[symbol]) * freq_dict[symbol]
        d += freq_dict[symbol]
    return s / d


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    result = bytes()  # the compressed form to return

    # we first get a string representation of the compressed text (as 0s and 1s)
    s = ""
    for byte in text:
        s += codes[byte]

    # we extract the bytes from s.
    # s_byte will contain a string representation of the current byte
    s_byte = s[0:8]
    index = 0
    # while we didn't reach the last byte
    while len(s_byte) == 8:
        # we concatenate the byte to the final result
        result += bytes([bits_to_byte(s_byte)])
        # we get the next byte
        index += 1
        s_byte = s[8 * index: 8 * (index + 1)]

    # we fill the last "byte" with 0s (because its length < 8)
    # except if its empty (the case when the text's length is a multiple of 8)
    if len(s_byte) > 0:
        s_byte += (8 - len(s_byte)) * "0"
        # we add the last byte to the result
        result += bytes([bits_to_byte(s_byte)])

    # the end. We return the result
    return result


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    # we also suppose that the root is not a leaf, which is the
    # case if there are more than 2 symbols
    bytes_representation = bytes()  # the bytes representation to return

    # we do a postorder traversal (left, right, node) recursivly
    # we start with the left child. If it's not a leaf
    # the left child is the first to write its bytes (if not leaf)
    if not tree.left.is_leaf():
        bytes_representation += tree_to_bytes(tree.left)

    # then the right child writes its bytes (if not leaf)
    if not tree.right.is_leaf():
        bytes_representation += tree_to_bytes(tree.right)

    # finally, we write the bytes of the current node
    # first, the left child
    if tree.left.is_leaf():
        bytes_representation += bytes([0])
        bytes_representation += bytes([tree.left.symbol])
    else:
        bytes_representation += bytes([1])
        bytes_representation += bytes([tree.left.number])
    # then, the right child
    if tree.right.is_leaf():
        bytes_representation += bytes([0])
        bytes_representation += bytes([tree.right.symbol])
    else:
        bytes_representation += bytes([1])
        bytes_representation += bytes([tree.right.number])

    # we return the result
    return bytes_representation


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    # root = ReadNode(1, left_node_index, 1, right_node_index)
    # we create a list of HuffmanNode nodes which are the nodes of the tree.
    # we initialize every element to None
    huffman_nodes = [None for _ in range(len(node_lst))]

    # We transform each ReadNode into an internal HuffmanNode, and we generate
    # the corresponding children
    for index, r_node in enumerate(node_lst):
        # enumerate(L) = [(index, L(index)]
        # we generate the huffman tree node whose number is index. It might have
        # already been generated by its parent so we check for that
        if huffman_nodes[index]:
            node = huffman_nodes[index]
        else:
            # we must generate it now
            node = HuffmanNode(None, None, None)
            # we set its number to index
            node.number = index
            # we add it to the list of huffman tree nodes, at index position
            huffman_nodes[index] = node

        # we create the left child of the node depending
        # on whether it's a leaf or not
        if r_node.l_type == 0:
            # it's a leaf node. l_data is the symbol if the left child
            child = HuffmanNode(r_node.l_data, None, None)
        else:
            # internal node, we get it from the list of
            # huffman nodes or we create it.
            # l_data is the index (and number) of the child
            if huffman_nodes[r_node.l_data]:
                child = huffman_nodes[r_node.l_data]
            else:
                # we create the left child and add it to the list
                child = HuffmanNode(None, None, None)
                child.number = r_node.l_data
                huffman_nodes[r_node.l_data] = child
            # now we set the child as a left child of the node
        node.left = child

        # we do the same thing for the right child.
        # we create the right child of the node
        # depending on whether it's a leaf or not
        if r_node.r_type == 0:
            # it's a leaf node. r_data is the symbol if the right child
            child = HuffmanNode(r_node.r_data, None, None)
        else:
            # it's an internal node, we get it from the
            # list of huffman nodes or we create it.
            # r_data is the index (and number) of the child
            if huffman_nodes[r_node.r_data]:
                child = huffman_nodes[r_node.r_data]
            else:
                # we create the right child and add it to the list
                child = HuffmanNode(None, None, None)
                child.number = r_node.r_data
                huffman_nodes[r_node.r_data] = child
            # now we set the child as a right child of the node
        node.right = child

    # the root node of the huffman is at root_index of the huffman_nodes list
    return huffman_nodes[root_index]


def internal_nodes_count(tree):
    """A utility function that returns the number of internal nodes
    in a tree.
    @param HuffmanNode tree: the tree
    @rtype: int

    >>> tree = HuffmanNode(5, HuffmanNode(3), HuffmanNode(4))
    >>> print(internal_nodes_count(tree))
    1
    """
    # base case of the recursive algorithm
    if tree.is_leaf():
        return 0
    else:
        return 1 +\
               internal_nodes_count(tree.left) +\
               internal_nodes_count(tree.right)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    Note that the parameter root_index is useless because since the
    tree is represented in postorder, the root node is the last
    element of node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12),\
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    # we create a list of HuffmanNode nodes which are the nodes of the tree.
    huffman_nodes = []
    root_index += 1

    # since we have a postorder traversal, children are always generated before
    # parents. So, we go through the read nodes and transform them to internal
    # huffman nodes. We don't need to check if children have
    # already been generated, we know they were. We add the nodes to the
    # huffman nodes list in a postorder way, meaning that if we were to create
    # the list by traversing the huffman tree in postorder, we would obtain
    # the same list.
    for index, read_node in enumerate(node_lst):
        # create the internal node
        node = HuffmanNode(None, None, None)

        # Right child
        # if right child is a leaf
        if read_node.r_type == 0:
            # r_data is the symbol of the right child
            node.right = HuffmanNode(read_node.r_data, None, None)
        else:
            # right child is an internal leaf
            # index of right child is index - 1
            # we know (because postorder) that it has already
            # been created.
            node.right = huffman_nodes[index - 1]

        # Left child
        # if the left child is a leaf
        if read_node.l_type == 0:
            # l_data is the symbol of the left child
            node.left = HuffmanNode(read_node.l_data, None, None)
        else:
            # left child is an internal leaf. We must find its index.
            # if right child is a leaf then left index = index - 1
            if read_node.r_type == 0:
                # left is not leaf and right is leaf
                node.left = huffman_nodes[index - 1]
            else:
                # left and right both are not leafs
                # index of left child = index - number of internal nodes in
                # right child tree - 1
                right_internal_nodes_count = internal_nodes_count(node.right)
                node.left = huffman_nodes[index-right_internal_nodes_count-1]

        # Add the current node to the list of nodes
        huffman_nodes.append(node)

    # At the end, the last element in the list of nodes
    # is the root of the huffman tree
    return huffman_nodes[-1]


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    # 2nd approach: traverse the tree and discovering a
    # symbol whenever a leaf is reached.
    # first, we convert the text into string bits
    s = ""  # will contain the text to decompresse as a string of bits
    for byte in text:
        s += byte_to_bits(byte)

    result = bytes()  # will contain the decompressed text
    p = tree  # will be used to navigate the tree

    # we go through s bit by bit
    for b in s:
        if b == "0":
            p = p.left
        else:  # b == "1"
            p = p.right

        # if we reached a leaf, we add its symbol to the result
        if p.is_leaf():
            result += bytes([p.symbol])
            p = tree  # go back to the root
            if len(result) >= size:  # we've reached to limit size
                break

    return result


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def depth(tree, depths, current_depth=0):
    """Calculates the depths of all the internal nodes
    of the tree and saves them as tuples (depth, node)
    in the depths list. current_depth represents the
    depth of the current node.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param list depths: list of tuples (depth, node)
    @param int current_depth: the depth of the current node (tree)
    @rtype: NoneType
    """
    depths.append((current_depth, tree))
    if tree.left:
        depth(tree.left, depths, current_depth + 1)
    if tree.right:
        depth(tree.right, depths, current_depth + 1)
    return None


def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # we order the symbols by frequencies decreasing
    freq_sym = [(freq_dict[symbol], symbol) for symbol in freq_dict]
    freq_sym.sort(reverse=True)
    symbols = [x[1] for x in freq_sym]
    # symbols = [97, 98, 99, 100, 101]
    # we calculate the depth of each leaf node, and sort the nodes according to
    # their depths (from the highest to the deepest)
    depths = []
    depth(tree, depths, current_depth=0)
    depths.sort()
    # smallest to biggest
    nodes = [x[1] for x in depths]
    # keep only leaf nodes
    nodes = [node for node in nodes.copy() if node.is_leaf()]

    # put the most frequent symbols in the heighest nodes
    for i, node in enumerate(nodes):
        node.symbol = symbols[i]

    return None
    # therefore smallest depth gets most frequently used symbol

if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds.".format(fname,
                                                      time.time() - start))
