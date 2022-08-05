# from collections import deque
#
#
# def bfs(orders, warehouses, paths):
#     if not orders:
#         print("Finalized", paths)
#         return paths
#     next_order = orders.popleft()
#     print("next order", next_order)
#     # find the children
#     children = [(next_order, w) for w in warehouses]
#     print("paths", paths)
#     print("children", children)
#     # Add each children to the known paths.
#     new_paths = []
#     for p in paths:  # paths:=[[(o,w),(o,w)], [(o,w),(o,w)], ...]
#         for c in children:  # (o,w)
#             new_paths.append(p + [c])
#     return bfs(orders, warehouses, new_paths)
#
#
# def find_paths(orders, warehouses):
#     o = orders.popleft()
#     all_paths = []
#     for w in warehouses:
#         print("Starting from ", (o, w))
#         all_paths.append(bfs(orders.copy(), warehouses, paths=[[(o, w)]]))
#     return [item for sublist in all_paths for item in sublist]
#
#
# result = find_paths(deque([0, 1, 2]), deque([0, 1, 2]))
# print(len(result))
# print(result)

# itertools
from itertools import product

choices = [0, 1, 2]
num_orders = 3
l = [choices] * num_orders  # you have 3 choices for each of the three orders
print(list(product(*l)))
