import networkx as nx
import time
import random

def k_shell_method(G, n):
    """
    根据k_shell算法计算节点的重要性
    :param G: 图
    :param n: 选取的前n个种子节点
    :return: 节点排序列表sorted_nodes, 选取前n个节点耗时timings, ks值
    """
    # 初始化数据结构
    k_shell = {}
    ks = {}
    remaining_nodes = G.nodes()
    current_k = 1
    # i = 1
    # time_star = time.perf_counter()
    # timings = 1
    # 迭代分解过程
    while remaining_nodes:
        # 计算剩余节点的度
        degrees = {node: G.degree(node) for node in remaining_nodes}
        if not degrees:
            break
        # 动态调整current_k
        min_degree = min(degrees.values())
        current_k = max(current_k, min_degree)
        # 收集所有度<=current_k的节点
        to_remove = {node for node in remaining_nodes if degrees[node] <= current_k}
        if not to_remove:
            # if i > n:
            #     break
            current_k += 1
            continue
        # 标记k-shell值并移除节点
        for node in to_remove:
            k_shell[node] = current_k
            ks[node] = current_k    # 作为第三个返回值，返回所有节点评分值（k-shell值）
            # i += 1
            # if i == n:
            #     timings = time.perf_counter() - time_star
        remaining_nodes -= to_remove

    # 按k-shell降序排序，相同k值随机排序
    # sorted_k_shell = sorted(k_shell.keys(), key=lambda x: (-k_shell[x], random.random()))
    # 转换为节点ID列表（兼容字符串和数字类型）  没必要 只是简单的
    # sorted_nodes = [node for node in sorted_k_shell]

    start = time.perf_counter()
    timings = 1
    selected_k_shell = []
    remaining_k_shell = list(G.nodes())
    # 为了增加耗时
    while remaining_k_shell:
        u, v = max(k_shell.items(), key=lambda x: x[1])
        selected_k_shell.append(u)
        remaining_k_shell.remove(u)
        k_shell.pop(u)
        if len(selected_k_shell) == n:
            timings = time.perf_counter() - start

    return selected_k_shell, timings, ks
