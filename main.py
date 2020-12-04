class Tree:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def binaryPaths(root):
    xds = []
    def go(t, res):
        if not t:
            # print("res ", res)
            return res

        tmp = res + [t.val]
        go(t.left, tmp)
        go(t.right, tmp)

        xds.append(tmp)

    go(root, [])
    return xds


if __name__ == '__main__':
    root = Tree(1)
    root.left = Tree(2)
    root.right = Tree(3)
    root.left.right = Tree(5)
    # print(root.left.right.right)
    # print(binaryPaths(root))
    # root.right.left = Tree(10)
    # root.left.right.right = Tree(5)
    res = binaryPaths(root)
    xd = {}
    print([*res])
