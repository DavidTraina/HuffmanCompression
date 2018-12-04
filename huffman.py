"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# Resources used: https://en.wikipedia.org/wiki/Huffman_coding, https://en.wikipedia.org/wiki/Breadth-first_search
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
    >>> get_bit(0b010000101, 7)
    1
    >>> get_bit(0b00000101, 7)
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
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for byte in text:
        freq_dict.setdefault(byte, 0)
        freq_dict[byte] += 1
    return freq_dict


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

    nodes = sorted(list(freq_dict.items()), key=lambda x: x[1],
                   reverse=True)  # list of tuples of (symbol, freq) sorted by freq in descending order
    for i in range(len(nodes)):  # converts each tuple of (symbol, freq) into tuple of (freq, HuffmanNode())
        freq = nodes[i][1]
        nodes[i] = (freq, HuffmanNode(nodes[i][0], None, None))

    while len(nodes) > 1:
        lowest = nodes.pop()
        sec_lowest = nodes.pop()
        new_node = (lowest[0] + sec_lowest[0], HuffmanNode(None, lowest[1], sec_lowest[1]))
        # inserts new_node into nodes, maintaining order. If another node has the same frequency, it is inserted to the right.
        low_in = 0
        high_ex = len(nodes)
        while low_in < high_ex:
            mid_index = (low_in + high_ex) // 2
            if new_node[0] > nodes[mid_index][0]:
                high_ex = mid_index
            else:
                low_in = mid_index + 1
        nodes.insert(low_in, new_node)
    return nodes[0][1]


def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    def generate_code_list(tree1, code=''):
        if tree1.is_leaf():
            if code == '':
                code = '1'
            return [
                (tree1.symbol, code)]  # if current node is a leaf, returns the symbol and its code as a list of tupple
        return generate_code_list(tree1.left, code + '0') + generate_code_list(tree1.right,
                                                                               code + '1')  # returns the list of tuples of (symbol, code) pairs for left and right subtree and concatenated. (0 is added to the code for left, and 1 for right)

    return dict(generate_code_list(tree))


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

    def postorder(tree1):
        if tree1 is None:
            return []
        return postorder(tree1.left) + postorder(tree1.right) + [tree1]

    postorder_nodes = postorder(tree)
    if len(postorder_nodes) == 1:
        postorder_nodes[0].number = 0
        return
    count = 0
    for node in postorder_nodes:
        if node.symbol is None:
            node.number = count
            count += 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

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
    the_codes = get_codes(tree)
    total = 0
    for item in the_codes.items(): total += len(item[1]) * freq_dict[item[0]]
    return total / sum(freq_dict.values())


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
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
    bits, result1 = '', []
    for i in range(len(text)):
        bits += codes[text[i]]  # for each byte in text, appends the
        while len(
                bits) >= 8:  # nested loops because slicing a long string is linear and becomes very slow for long strings since you have to do it over and over
            result1.append(bits_to_byte(bits[:8]))
            bits = bits[8:]
    while bits:
        result1.append(bits_to_byte(bits[:8]))
        bits = bits[8:]
    return bytes(result1)
    # Personal Note: bits_to_byte('101') = bits_to_byte('10100000') = 160 != bits_to_byte('00000101') = 5. Thus when we compress the last few bits '101' and '1010' will compress as the same thing, they will compress to the byte represented by the int 160. So when we decompress the byte 160, it will convert to '10100000'. We don't know how many of the trailing 0's are valid, so we need to use the size parameter to know how many bytes to parse from the bits. Once the number of bytes we've parsed reaches size (which is the original number of bytes in the file) we know to stop converting the bits to bytes and so we know the remaining trailing 0's (if any) are not valid.


def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

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

    def postorder(tree1):
        if tree1 is None:
            return []
        return postorder(tree1.left) + postorder(tree1.right) + [tree1]

    postorder_nodes = postorder(tree)
    if len(postorder_nodes) == 1: return bytes([postorder_nodes[0].symbol, 0, 2,
                                                0])  # special case of single node tree, just return the symbol contained in the node.
    result = []
    for node in postorder_nodes:
        if not node.is_leaf():
            if node.left.is_leaf():
                result.append(0)
                result.append(node.left.symbol)
            else:
                result.append(1)
                result.append(node.left.number)
            if node.right.is_leaf():
                result.append(0)
                result.append(node.right.symbol)
            else:
                result.append(1)
                result.append(node.right.number)
    return bytes(result)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
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

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    if len(node_lst) == 1 and node_lst[0].r_type == 2: return HuffmanNode(node_lst[0].l_type, None,
                                                                          None)  # r_type will normally be only 0 or 1, we use 2 to indicate that the whole tree is just one node (special case). If we didnt check a single node tree and a 3 node tree would look the same since node lst is only supposed to be the internal nodes. (only one internal node and two leaf nodes in 3 node tree)
    if node_lst[root_index].l_type == 0:  # if node left of root is a leaf
        left = HuffmanNode(node_lst[root_index].l_data, None, None)
    else:  # if node left of root is not a leaf
        left = generate_tree_general(node_lst, node_lst[
            root_index].l_data)  # generate the subtree with the left child of the root of the overall tree as the root of the subtree
    if node_lst[root_index].r_type == 0:  # if node right of root is a leaf
        right = HuffmanNode(node_lst[root_index].r_data, None, None)
    else:  # if node right of root is not a leaf
        right = generate_tree_general(node_lst, node_lst[
            root_index].r_data)  # generate the subtree with the left child of the root of the overall tree as the root of the subtree
    return HuffmanNode(None, left, right)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    if len(node_lst) == 1 and node_lst[0].r_type == 2: return HuffmanNode(node_lst[0].l_type, None,
                                                                          None)  # r_type will normally be only 0 or 1, we use 2 to indicate that the whole tree is just one node (special case). If we didnt check a single node tree and a 3 node tree would look the same since node lst is only supposed to be the internal nodes.(only one internal node and two leaf nodes in 3 node tree)
    huff_nodes = []  # used as a stack with top at right and bottom at left
    for node in node_lst:
        if node.r_type == 0:  # if node right of root is a leaf
            right = HuffmanNode(node.r_data, None, None)
        else:  # since the right child is not a leaf then we must have already constructed the right subtree since nodes are in postorder; children always come before parents
            right = huff_nodes.pop()  # pop the most recently built subtree because right subtrees are built after left since the nodes are in postorder so it will be on the top of the stack.
        if node.l_type == 0:  # if node left of root is a leaf
            left = HuffmanNode(node.l_data, None, None)
        else:  # since the left child is not a leaf then we must have already constructed the left subtree since nodes are in postorder; children always come before parents. While admittedly left nodes are visited before right nodes in postorder, the right subtree would be on the top of the stack, if there was a right subtree it would have already been popped so we know the left subtree is actually at the top
            left = huff_nodes.pop()
        huff_nodes.append(HuffmanNode(None, left,
                                      right))  # add the newly constructed tree to the top of the stack. note this will not get in the way once we start constructing the right subtree because we will start from the bottom due to postorder so we will put other smaller trees on top of this one.
    return huff_nodes[-1]  # could also use index 0

    # #------------------------------------------------------------------------------------------
    # def recurse(node_lst, root_index):
    # internal_nodes_passed = 0 # keeps track of how many internal nodes have been fully contructed
    # if node_lst[root_index].r_type == 0: #if node right of root is a leaf
    # right = HuffmanNode(node_lst[root_index].r_data, None, None) # does not "consume" any internal nodes, only leaf nodes which are not in node_lst
    # else:
    # right, temp = recurse(node_lst, root_index -1) # returns a tupple with the right subtree construction, and how many internal nodes the subtree contains
    # internal_nodes_passed += temp # increment by the number of internal nodes that were "consumed" in the making of the right subtree.
    # if node_lst[root_index].l_type == 0: #if node right of root is a leaf
    # left = HuffmanNode(node_lst[root_index].l_data, None, None) # does not "consume" any internal nodes, only leaf nodes which are not in node_lst
    # else:
    # left, temp = recurse(node_lst, root_index - internal_nodes_passed -1)# returns a tupple with the left subtree construction, and how many internal nodes the subtree contains. we recursively call recurse with root_index - internal_nodes_passed -1 because we know that the index of the left child of the current root will be one index left of (before) the index of the first internal node visited in the right subtree of the current root when traversing the tree in postorder (node_lst is in postorder). So, since we kept track of the number of internal nodes "consumed" to construct the right subtree, we know how many indices left we need to move to get to the left child of the current root. That is, move internal_nodes_passed + 1 left of the index of the current root, this call recurse with root_index - internal_nodes_passed - 1.
    # internal_nodes_passed += temp # increment by the number of internal nodes "consumed" to construct the left subtree of the current root because the current root could be the right child larger tree, in which case to construct the left subtree of that larger tree we would need to know how many internal nodes were "consumed" to create the right subtree. and since our current root could be the right child of a larger tree, we would be part of that right subtree
    # internal_nodes_passed += 1 # if we've constructed the left and the right then we've constructed the current root node so we increment because we are done with the current root node; we've "passed it"
    # return (HuffmanNode(None, left, right), internal_nodes_passed)
    # return recurse(node_lst, root_index)[0] # index 0 to return only the tree and not the whole tupple


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """
    # size is the number of bytes we should be returning (number of original bytes in the file), so once we have that many byte in uncompressed we stop interpretting the remaining bits (trailing 0s) as valid
    # see comment of compress for further explaination
    if tree.is_leaf(): return bytes([tree.symbol] * size)
    uncompressed = []
    node = tree
    for byte in text:  # nestedloops because slicing a long string is linear and becomes very slow for long strings since you have to do it over and over
        for bit in byte_to_bits(byte):
            if bit == '0':
                node = node.left
            else:
                node = node.right
            if node.is_leaf():
                uncompressed.append(node.symbol)
                if len(uncompressed) == size: break  # we have uncompressed all we need
                node = tree
        if len(uncompressed) == size: break  # we have

    return bytes(uncompressed)


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
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
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
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    def level_order_traversal():
        nodes = []
        q = [tree]
        while q:
            front_node = q.pop(0)
            if front_node.symbol is not None: nodes.append(front_node)
            if front_node.left: q.append(front_node.left)
            if front_node.right: q.append(front_node.right)
        return nodes

    level_order_nodes = level_order_traversal()  # level order nodes should also be in order of frequency of each node's symbol (only leaf nodes included in list)
    freq_sorted = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)  # sorted greatest to least frequency
    for i in range(len(level_order_nodes)):
        level_order_nodes[i].symbol = freq_sorted[i][0]
    # todo


if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
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
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
