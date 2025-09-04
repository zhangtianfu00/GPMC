import math
from collections import deque
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import copy
from sympy.physics.control.control_plots import matplotlib
from tqdm import tqdm
import random
import heapq
from collections import defaultdict
from scipy.stats import kendalltau
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
from multiprocessing import Pool
import k_shell_method
import KBKNR
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
import seaborn as sns
from matplotlib.cm import ScalarMappable
from scipy import stats
from collections import Counter


def k_shell(G):
    """
    手动实现k-shell分解算法
    奖励值β更新了衰减公式，先移除的节点赋予其邻居的β值更大
    """
    G = G.copy()
    k_s = {}
    K_max = 1  # 初始化为1，后续动态更新
    remaining_nodes = set(G.nodes())
    current_k = 1
    removal_round = {}  # 记录节点被移除的轮次
    current_round = 1  # 轮次计数器，每轮current_k递增或节点移除后+1

    while remaining_nodes:
        degrees = {node: G.degree(node) for node in remaining_nodes}
        min_degree = min(degrees.values())
        current_k = max(current_k, min_degree)

        to_remove = {node for node in remaining_nodes if degrees[node] <= current_k}
        if not to_remove:
            current_k += 1      # 记录当前层的k_shell值
            current_round += 1  # 记录移除轮次
            continue

        for node in to_remove:
            k_s[node] = current_k  # k-shell值为被移除时的current_k
            removal_round[node] = current_round

        remaining_nodes -= to_remove
        G.remove_nodes_from(to_remove)  # k-shell移除过程节点度下降
        current_round += 1  # 每处理完一轮节点移除，轮次+1

    return k_s, removal_round


def k_shell_decomposition(G):
    """
    手动实现k-shell分解算法
    奖励值β更新了衰减公式，先移除的节点赋予其邻居的β值更大
    """
    G = G.copy()
    ks = {}
    K_max = 1  # 初始化为1，后续动态更新
    remaining_nodes = set(G.nodes())
    current_k = 1
    beta = {u: 0.0 for u in G.nodes()}
    removal_round = {}  # 记录节点被移除的轮次
    current_round = 1  # 轮次计数器，每轮current_k递增或节点移除后+1
    D_max = max([G.degree(u) for u in G.nodes()])
    degrees = {node: G.degree(node) for node in G.nodes()}

    kk_s, kk_round = k_shell(G)
    kk_max = max(kk_s.values())

    while remaining_nodes:
        degrees = {node: G.degree(node) for node in remaining_nodes}
        min_degree = min(degrees.values())
        current_k = max(current_k, min_degree)
        to_remove = {node for node in remaining_nodes if degrees[node] <= current_k}

        if not to_remove:
            current_k += 1
            current_round += 1
            continue

        # 预处理邻居平均k-shell值和全图最大k-shell
        K_max = max(K_max, current_k)
        beta_updates = defaultdict(float)
        for node in to_remove:
            ks[node] = current_k  # k-shell值为被移除时的current_k
            removal_round[node] = current_round

            d_u = G.degree(node)
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                ks_avg = 0.0
            else:
                ks_avg = sum(ks.get(n, 0) for n in neighbors) / len(neighbors)
            weight = d_u / D_max * (1 + ks_avg / K_max)
            # weight = (1 + ks_avg / kk_max)

            # 更新β值
            for neighbor in neighbors:
                if neighbor in remaining_nodes:
                    delta = current_round - removal_round.get(node, 0)
                    # beta_updates[neighbor] += weight + kk_s[neighbor]  #/ (1 + delta)
                    beta_updates[neighbor] += weight #/ (1 + delta)

        # 批量更新β值
        for neighbor, increment in beta_updates.items():
            beta[neighbor] += increment

        remaining_nodes -= to_remove
        G.remove_nodes_from(to_remove)  # k-shell移除过程节点度下降
        current_round += 1  # 每处理完一轮节点移除，轮次+1

    return ks, beta


def precompute_neighbors(G):
    """优化邻居预处理，增加三跳信息和局部密度"""
    cache = {
        'one_hop': {},
        'two_hop': {},
        'three_hop': {},
        'clustering': nx.clustering(G)
    }

    for u in G.nodes():
        # 一跳邻居
        one_hop = set(G.neighbors(u))
        cache['one_hop'][u] = len(one_hop)

        # 二跳邻居
        two_hop = set()
        for v in one_hop:
            two_hop.update(G.neighbors(v))
        two_hop -= one_hop
        two_hop.discard(u)
        cache['two_hop'][u] = len(two_hop)

        # 三跳邻居
        three_hop = set()
        for w in two_hop:
            three_hop.update(G.neighbors(w))
        three_hop -= one_hop | two_hop
        three_hop.discard(u)
        cache['three_hop'][u] = len(three_hop)

    return cache


def select_seeds_method(G, n):
    """
    执行完整的种子节点选择算法并记录时间
    时间复杂度O(n log n)
    - 整合所有计算步骤（μ、kss、α、φ、γ、Φ）。
    - 首轮使用一重综合度φ选择第一个种子。
    - 后续迭代使用二重综合度Φ选择种子，并动态更新惩罚值γ。
    :param G: 邻接表表示的图
    :return:（种子节点选择顺序列表，时间戳列表）
    时间戳列表记录每个节点被选中时的累计时间
    """
    n_seed_time = 0  # 记录选中第n个节点的时间
    # 预处理阶段
    # ks_max = max(ks.values())
    # 预计算指标
    nodes = list(G.nodes())
    mu = {}
    d = {u: len(G[u]) for u in nodes}
    # kss = {}
    alpha = {}  # 存储桶值α
    psi = {}  # 存储一重综合度ψ
    upsilon = {}  # 存储二重综合度
    # features = {}  # 计算节点特征
    start_time = time.perf_counter()  # 记录选择n个节点的耗时
    ks, beta = k_shell_decomposition(G)  # 执行k-shell分解，获取每个节点的k-shell值和β奖励值
    print(f"节点奖励值：{beta}")
    neighbor_cache = precompute_neighbors(G)  # 预处理所有节点的两跳、三跳邻居数及聚集系数
    for u in nodes:
        one_hop = neighbor_cache['one_hop'][u]
        two_hop = neighbor_cache['two_hop'][u]
        three_hop = neighbor_cache['three_hop'][u]  # 理论依据：真实社交网络中，影响力传播通常具有3度效应
        # clustering = neighbor_cache['clustering'][u]

        # 改进的影响系数μ（公式：μ = 两跳 / (一跳 + 二跳 + 三跳））
        denominator = one_hop + two_hop + three_hop
        mu[u] = two_hop / denominator if denominator else 0.0

        # 动态传播权重（公式：kss = ks + μ * d * (1 + sqrt(clustering)) )在紧密社区中优先选择枢纽节点
        # kss = ks[u] + mu[u] * d[u]  # * (1 + clustering)  # math.sqrt(clustering)
        # kss = mu[u] * math.log(1 + d[u], 10)  # * (1 + clustering)  # math.sqrt(clustering)
        kss = (1 + mu[u]) * d[u]  # * (1 + clustering)  # math.sqrt(clustering)
        alpha[u] = math.floor(kss) + 1

        # 一重综合度，综合传播力（公式：φ = α + μ * β  )
        psi[u] = alpha[u] + mu[u] * beta[u]  #* (1 + ks[u] / ks_max)

        # # 堆排序元组
        # features[u] = (-phi[u], -alpha[u], u)

    # 初始化堆和全局变量
    heap = [(-psi[u], -alpha[u], u) for u in G.nodes()]
    heapq.heapify(heap)
    remaining = set(G.nodes())
    selected = []  # 记录种子及其选中时间步
    time_step = 0  # 全局时间步计数器
    gamma = defaultdict(float)  # 惩罚值 浮点型以支持衰减
    scores = {}  # 记录每个节点的评分值
    while remaining:
        # 弹出有效节点
        while heap:
            p, a, u = heapq.heappop(heap)  # psi, alpha, u
            if u in remaining:
                scores[u] = -p  # 记录评分值（取反以恢复原始综合度）
                break
        else:
            break
        # 记录选中时间和更新gamma
        selected.append((u, time_step))
        remaining.remove(u)
        time_step += 1

        # 记录第n个节点的时间
        if len(selected) == n:
            n_seed_time = time.perf_counter() - start_time

        # 更新邻居的gamma值
        for v in G.neighbors(u):
            if v in remaining:
                delta = time_step - selected[-1][1]  # 当前时间 - 种子选中时间
                gamma[v] += 1 / (1 + delta)
                # 二重综合度upsilon（更新堆中的phi值）
                upsilon[v] = psi[v] - mu[v] * gamma[v] * math.exp(-1 * delta / 15)  # netscience下T取15效果更好
                heapq.heappush(heap, (-upsilon[v], -alpha[v], v))

    return [u for (u, _) in selected], n_seed_time, scores


def compare_centrality(G, n):
    """
    对比各中心性方法前n个节点耗时并输出排序结果
    :param G: 网络图
    :param n: 要比较的前n个节点
    :return: 包含各方法的整体排序列表
    """

    print("比较不同方法...")
    # 记录各方法计算时间
    timing_data = {}    # 新方法选择前n个节点的耗时
    orders = {}     # 新方法的节点排序列表
    centrality_scores = {}  # 存储各方法评分字典

    # 新方法
    seed_order, new_method_time, new_method_scores = select_seeds_method(G, n)
    timing_data['new_method'] = new_method_time
    orders['Seeds_new_method'] = seed_order
    centrality_scores['New'] = new_method_scores

    # 度中心性
    start = time.perf_counter()
    timing_degree = 1
    original_degree = nx.degree_centrality(G)
    degree_centrality = original_degree.copy()
    selected_degree = []
    remaining_degree = list(G.nodes())
    while remaining_degree:
        u, v = max(degree_centrality.items(), key=lambda x: x[1])
        selected_degree.append(u)
        remaining_degree.remove(u)
        degree_centrality.pop(u)
        if len(selected_degree) == n:
            timing_degree = time.perf_counter() - start
    timing_data['Degree'] = timing_degree
    orders['Seeds_degree'] = selected_degree
    centrality_scores['DC'] = original_degree

    # 介数中心性
    start = time.perf_counter()
    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(G.nodes(), key=lambda x: -betweenness_centrality[x])
    timing_data['Betweenness'] = time.perf_counter() - start
    orders['Seeds_betweenness'] = sorted_betweenness
    centrality_scores['BC'] = betweenness_centrality

    # 接近中心性
    start = time.perf_counter()
    closeness_centrality = nx.closeness_centrality(G)
    sorted_closeness = sorted(G.nodes(), key=lambda x: -closeness_centrality[x])
    timing_data['Closeness'] = time.perf_counter() - start
    orders['Seeds_closeness'] = sorted_closeness
    centrality_scores['CC'] = closeness_centrality

    # 特征向量中心性
    start = time.perf_counter()
    timing_eigenvector = 1
    original_eigen = nx.eigenvector_centrality(G, max_iter=5000)
    eigenvector_centrality = original_eigen.copy()
    selected_eigenvector = []
    remaining_eigenvector = list(G.nodes())
    while remaining_eigenvector:
        u, v = max(eigenvector_centrality.items(), key=lambda x: x[1])
        selected_eigenvector.append(u)
        remaining_eigenvector.remove(u)
        eigenvector_centrality.pop(u)
        if len(selected_eigenvector) == n:
            timing_eigenvector = time.perf_counter() - start
    timing_data['Eigenvector'] = timing_eigenvector
    orders['Seeds_eigenvector'] = selected_eigenvector
    centrality_scores['EC'] = original_eigen

    # # PageRank
    # start = time.perf_counter()
    # pagerank = nx.pagerank(G)
    # sorted_pagerank = sorted(G.nodes(), key=lambda x: -pagerank[x])
    # timing_data['PageRank'] = time.perf_counter() - start
    # orders['Seeds_pagerank'] = sorted_pagerank

    # k-shell, 系数n统计前n个节点耗时
    orders['Seeds_k_shell'], timing_data['K_shell'], k_shell_centrality = k_shell_method.k_shell_method(G, n)
    centrality_scores['K-shell'] = k_shell_centrality

    # kbknr
    # orders['Seeds_KBKNR'], timing_data['Kbkrn'] = KBKNR.kbknr_algorithm(G)

    # 随机方法
    start = time.perf_counter()
    timing_random = 1
    selected_random = []
    remaining_random = list(G.nodes())
    # 方法一:
    # selected_random = random.sample(G.nodes(), n)
    while remaining_random:
        # 方法二:
        # random.shuffle(remaining_random)
        # selected_nodes = remaining_random.pop(0)
        # selected_random.append(selected_nodes)
        # 方法三:
        u = random.sample(remaining_random, 1)[-1]
        selected_random.append(u)
        remaining_random.remove(u)
        if len(selected_random) == n:
            timing_random = time.perf_counter() - start
    timing_data['Random'] = timing_random
    orders['Seeds_random'] = selected_random
    # orders['Seeds_random'] = ['34','35', '36', '37', '38', '39', '40', '41']#['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                              # '12', '13', '14', '15', '16', '17', '18', '19','20',
                              # '21', '22', '23', '24', '25', '26', '27', '28', '29',
                              # '25', '26', '27', '28', '29', '30', '31', '32', '33',
                              # , '42',
                              # '43', '44','45', '46', '47', '48', '49', '50', '51',
                              # '52', '53', '54', '55']

    for method in orders:
        print(f'{method}: {len(orders[method])}')

    print(f"\n输出各类度量方法的选取前{n}个种子节点所需的时间及具体的节点列表:")
    # 打印对比结果
    print(f"\n{'Metric':<15} | {'Top {n} Time (s)':<10} |          Top {n} Nodes")
    print("-" * 90)
    print(f"{'New Method':<15} | {timing_data['new_method']:.6f}  | {orders['Seeds_new_method'][:n]}")
    print(f"{'Degree':<15} | {timing_data['Degree']:.6f}  | {orders['Seeds_degree'][:n]}")
    print(f"{'Betweenness':<15} | {timing_data['Betweenness']:.6f}  | {orders['Seeds_betweenness'][:n]}")
    print(f"{'Closeness':<15} | {timing_data['Closeness']:.6f}  | {orders['Seeds_closeness'][:n]}")
    print(f"{'Eigenvector':<15} | {timing_data['Eigenvector']:.6f}  | {orders['Seeds_eigenvector'][:n]}")
    # print(f"{'PageRank':<15} | {timing_data['PageRank']:.6f}  | {orders['Seeds_pagerank']}")
    print(f"{'K-shell':<15} | {timing_data['K_shell']:.6f}  | {orders['Seeds_k_shell'][:n]}")
    # print(f"{'kbknr':<15} | {timing_data['Kbkrn']:.6f}  | {orders['Seeds_KBKNR']}")
    print(f"{'Random':<15} | {timing_data['Random']:.6f}  | {orders['Seeds_random'][:n]}")

    return orders, timing_data, centrality_scores


def SIR_model(G, seed_nodes, T, beta, gama):
    """
    执行SIR模型，模拟传播过程
    :param G: 图G
    :param seed_nodes: 种子集
    :param T: 执行的时间步
    :param beta: 感染概率
    :param gama: 恢复概率
    :return: 每一个时间步感染节点列表A和恢复节点列表B及所影响的节点数量
    """
    nx.set_node_attributes(G, 'S', name='state')  # 节点状态初始化：所有节点初始化为易感
    for sn in seed_nodes:
        if sn in G.nodes():  # 简单有效性检查
            G.nodes[sn]['state'] = 'I'  # 设置种子节点

    infected = []  # 感染节点(包括初始种子集)
    infected.extend(seed_nodes)  # 将初始种子节点存入A中
    recovered = []  # 恢复节点

    for t in range(1, T + 1):
        for node in list(infected):  # 转换为列表避免迭代时修改
            # 感染邻居
            for neighbor in nx.neighbors(G, node):
                if G.nodes[neighbor]['state'] == 'S' and np.random.rand() < beta:
                    G.nodes[neighbor]['state'] = 'I'
                    infected.append(neighbor)  #将被感染的邻居暂时存入TtoS

            # # 恢复自身
            # eventp2 = np.random.random_sample()  # 随机恢复概率eventp2,或可以写成 np.random.rand()
            # if eventp2 < gama:
            #     G.nodes[node]['state'] = 'R'
            #     recovered.append(node)
            #     infected.remove(node)

        # 输出每个时间步的感染节点A和恢复节点B(累计恢复节点)
        # print(A, B)
    # 输出数量和列表（只输出最终结果，即时间步T时的结果）
    # print(f"Time {T}: Infected={len(A)}, Recovered={len(B)}, count={len(A + B)}")

    # 返回统计结果（可选）
    return len(infected) + len(recovered)


def plot_results(trials, seed_nums, seed_groups, labels, beta, gamma, T):
    """
    重复多次SIR模型传播，并可视化结果
    :param trials: 重复实验次数
    :param seed_nums:
    :param seed_groups:
    :param labels: 标签，即方法名
    :param beta: 感染率
    :param gamma: 恢复率
    :param T: 传播周期，轮次
    :return:
    """
    # 不同种子组在不同数量种子下总受影响节点数的变化，多次实验取平均
    results = np.zeros((len(seed_groups), len(seed_nums)))  # 结果矩阵
    print("\n评估传播效果...")
    # for num1 in tqdm(seed_nums, desc='Processing seed groups'):     # 实时进度显示
    for group_idx, group in enumerate(seed_groups):  # 遍历7种方法,group_idx对应种子选择方法的索引;group为当前种子选择方法
        for num_idx, num in enumerate(tqdm(seed_nums, desc=f'Processing seeds for group {labels[group_idx]}')):  # 遍历种子数量,num_idx为对应种子数量的索引;截取其前num个节点
            select_seeds = group[:num]  # 直接截取前num个节点
            trial_results = []
            for _ in range(trials):  # 重复实验
                total = SIR_model(G, select_seeds, T, beta, gamma)
                trial_results.append(total)
            results[group_idx, num_idx] = np.mean(trial_results)  # 存储平均值

    # 创建画布
    plt.figure(figsize=(10, 8), dpi=600)
    ax = plt.gca()  # 获取当前axes
    colors = plt.cm.tab10(np.linspace(0, 1, len(seed_groups)))
    # markers = random.randint(1, len(list(plt.Line2D.markers.keys())))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', '*', '.', 's']

    # --------------------------
    # 主图绘制
    # --------------------------
    for group_idx in range(len(seed_groups)):
        plt.plot(
            seed_nums, results[group_idx, :],
            label=labels[group_idx],  # 使用预定义标签
            color=colors[group_idx],
            marker=markers[group_idx],
            markersize=8,
            linewidth=2
        )

        # # 计算趋势线
        # z = np.polyfit(seed_nums, results[group_idx, :], 1)  # 一次多项式拟合（线性拟合）
        # p = np.poly1d(z)
        # # 绘制趋势线
        # plt.plot(seed_nums, p(seed_nums), linestyle='--', color=colors[group_idx], alpha=0.7)

    # 添加图例和标签
    # ax.set_title(f"Propagation Performance of Different Seed Selection Methods", fontsize=14)  # 标题
    ax.set_xlabel('Number of Seeds', fontsize=12)
    ax.set_ylabel('Total Affected', fontsize=12)
    # ax.grid(ls='--', alpha=0.5)
    ax.legend(loc='best')  # upper left bbox_to_anchor=(1.05, 1)
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 600
    plt.tight_layout()
    # 保存与显示
    plt.savefig("sin_wave.png", dpi=600, bbox_inches='tight')  # PNG格式，300 DPI
    plt.savefig("sin_wave.tif", dpi=600, bbox_inches='tight')  # TIF格式，300 DPI
    plt.savefig("sin_wave.svg", dpi=600, bbox_inches='tight')  # SVG格式，300 DPI
    plt.savefig("sin_wave.pdf", bbox_inches='tight')  # PDF格式（矢量图）
    plt.show()

    # # --------------------------
    # # 添加局部放大视图（聚焦关键区域）
    # # --------------------------
    # # 创建inset axes（嵌入主图内部）
    # axins = inset_axes(
    #     ax,
    #     width="80%",
    #     height="60%",
    #     loc='upper right',
    #     borderpad=1
    # )
    # # 在放大图中绘制相同数据
    # for group_idx in range(len(seed_groups)):
    #     plt.plot(
    #         seed_nums, results[group_idx, :],
    #         label=labels[group_idx],  # 使用预定义标签
    #         color=colors[group_idx],
    #         marker=markers[group_idx],
    #         markersize=8,
    #         linewidth=2
    #     )
    # # 设置放大区域
    # axins.set_xlim(10, 30)  # 放大种子数量
    # axins.set_ylim(400, 420)  # 放大总受影响数
    # axins.grid(ls=':', alpha=0.5)
    #
    # # 添加连接指示
    # ax.indicate_inset_zoom(axins, edgecolor="gray")
    #
    # plt.tight_layout()
    # # plt.savefig('奖励值β引入衰减因子.png', dpi=300, bbox_inches='tight')
    # plt.show()


def plot_kendall(correlations):
    # 打印排序相关性（Kendall's Tau）
    print("\n排序相关性（Kendall's Tau）:")
    for method, tau in correlations.items():
        print(f"{method:<12}: {tau:.4f}")

    # 准备绘图数据
    methods = list(correlations.keys())
    taus = list(correlations.values())

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    ax2 = plt.gca()
    bars = ax2.bar(methods, taus, color=[('r' if tau < 0 else 'g') for tau in taus])
    ax2.set_title(f"Kendall coefficient correlation of different methods", fontsize=14)
    ax2.set_xlabel("method", fontsize=12)
    ax2.set_ylabel("Kendall's Tau", fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='--')

    # 添加数据标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.annotate(f'{height:.2f}',
    #                  xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3),  # 3 points vertical offset
    #                  textcoords="offset points",
    #                  ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_repeated_experiments(G, n, repetitions):
    """运行重复实验并返回平均耗时

    参数：
    G : 网络图
    n : 选取的种子节点数量
    repetitions : 实验重复次数
    new_method : 新方法的计算函数，返回值为(seed_order, time_cost)

    返回：
    avg_times : 各方法的平均耗时字典
    """
    # 初始化耗时记录容器
    time_records = defaultdict(list)

    for _ in range(repetitions):
        # 计算包含新方法的所有方法耗时
        _, timing_data, _ = compare_centrality(G, n)

        # 收集各方法耗时
        for method, t in timing_data.items():
            time_records[method].append(t)

    # 计算平均耗时
    avg_times = {method: np.mean(times) for method, times in time_records.items()}
    return avg_times


def run_repeated_experiments1(G, seed_counts, repetitions):
    """运行重复实验并返回各方法在不同种子数量下的平均耗时
    参数：
    G : 网络图
    seed_counts : 可视化时图像的横坐标，即节点范围
    repetitions : 实验重复次数
    返回：
    avg_times : 各方法的平均耗时字典，格式为：
        {
            'method1': [t1, t2, ...],  # 不同种子数量对应的耗时
            'method2': [t1, t2, ...],
            ...
        }
    """
    # 固定种子节点参数范围,
    # seed_counts = list(range(5, 45, 5))  # [5, 10, ..., 40]

    # 初始化结果存储结构
    results = defaultdict(lambda: [[] for _ in seed_counts])

    for rep in range(repetitions):
        for idx, n in enumerate(seed_counts):
            # 计算包含新方法在内的所有方法耗时
            _, timing_data, _ = compare_centrality(G, n)

            # 收集各方法耗时
            for method, t in timing_data.items():
                results[method][idx].append(t)

    # 计算各方法不同种子数量的平均耗时
    avg_times = {}
    for method, time_lists in results.items():
        avg_times[method] = [np.mean(times) for times in time_lists]

    return avg_times


# 七个算法在六个数据集上的耗时结果，统一在一张图上显示
def plot_timing_comparison(timings, datasets, labels):
    """绘制带对数坐标的耗时对比柱状图（微秒单位）

    参数：
    timings (list of lists): 二维耗时数据（秒单位），外层列表对应对比组，内层列表对应各数据集耗时
    datasets (list): 数据集名称列表
    """
    # 可配置参数
    bar_width = 0.1  # 单柱宽度
    group_spacing = 0.02  # 组内间距

    # 计算布局参数
    n_groups = len(timings)
    n_datasets = len(datasets)
    x = np.arange(n_datasets)  # 数据集基准位置
    offsets = (np.arange(n_groups) - n_groups // 2) * (bar_width + group_spacing)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 使用颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

    # 转换为微秒 (1秒 = 1e6微秒)
    timings_micro = [[t * 1e6 for t in group] for group in timings]

    # 绘制柱状图
    for group_idx in range(n_groups):
        plt.bar(x + offsets[group_idx],
                timings_micro[group_idx],
                width=bar_width,
                color=colors[group_idx],
                label=labels[group_idx])

    # 配置对数坐标轴
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

    # 设置坐标轴范围（1-100000微秒，可根据数据调整）
    min_val = min(min(group) for group in timings_micro)
    max_val = max(max(group) for group in timings_micro)
    plt.ylim(max(1, min_val * 0.8), max_val * 1.2)

    # 添加图表元素
    # plt.title('Running Time', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Running Time (μs)', fontsize=12)  # 单位为微秒
    plt.xticks(x, datasets)
    # plt.grid(axis='y', alpha=0.4)  # 显示网格线
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left')
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 保存与显示
    plt.savefig('myplot.png', dpi=300, bbox_inches='tight')
    # 自动调整布局
    plt.tight_layout()
    plt.show()


# 单个数据集下不同种子数量对应耗时
def plot_timing_for_dataset(timings, seed_counts, algorithm_names, dataset_name):
    """绘制单个数据集下不同种子节点数量的耗时对比柱状图（微秒单位）
    参数：
    timings (list of lists): 二维耗时数据（秒单位），外层列表对应不同算法，内层列表对应各种子节点数量的耗时
    seed_counts (list): 种子节点数量列表（如[5,10,15,...,40]）
    algorithm_names (list): 算法名称列表
    dataset_name (str): 数据集名称（用于标题和文件名）
    """
    # 可配置参数
    bar_width = 0.1  # 单柱宽度
    group_spacing = 0.02  # 组内间距
    # seed_counts = list(range(5, 45, 5))  # [5,10,15,20,25,30,35,40]
    # seed_counts = [5, 10, 15, 20, 25, 30, 35, 40]
    # 检查数据维度是否正确
    n_algorithms = len(timings)
    n_seed_counts = len(seed_counts)
    # 转换为微秒 (1秒 = 1e6微秒)
    timings_micro = [[t * 1e6 for t in algorithm] for algorithm in timings]

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 使用颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, n_algorithms))

    # 计算柱状图位置
    x = np.arange(n_seed_counts)  # 种子节点数量的基准位置
    offsets = (np.arange(n_algorithms) - n_algorithms // 2) * (bar_width + group_spacing)

    # 绘制柱状图
    for alg_idx in range(n_algorithms):
        plt.bar(x + offsets[alg_idx],
                timings_micro[alg_idx],
                width=bar_width,
                color=colors[alg_idx],
                label=algorithm_names[alg_idx])

    # 配置对数坐标轴
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

    # 设置坐标轴范围
    min_val = min(min(algorithm) for algorithm in timings_micro)
    max_val = max(max(algorithm) for algorithm in timings_micro)
    plt.ylim(max(1, min_val * 0.8), max_val * 1.2)

    # 添加图表元素
    plt.title(f'Running Time on {dataset_name}', fontsize=12)
    plt.xlabel('Seed Number', fontsize=12)
    plt.ylabel('Running Time (μs)', fontsize=12)
    plt.xticks(x, seed_counts)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left')

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 自动调整布局
    plt.tight_layout()

    # 保存与显示
    filename = f'timing_{dataset_name.replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison(new_method_scores, centrality_scores, node_influence):
    """
    绘制多维对比图
    :param new_method_scores: 新方法评分字典 {node: score}
    :param centrality_scores: 各对比方法评分字典 {method: {node: score}}
    :param node_influence: 节点传播能力字典 {node: influence_count}
    return: 相关性图
    """
    methods = ['DC', 'BC', 'CC', 'EC', 'K-shell']
    # 配色方案(viridis(蓝-黄渐变) / magma(岩浆色系) / plasma / inferno)
    cmap = plt.get_cmap('viridis')  # 使用 Viridis 配色
    # cmap = sns.color_palette("YlGnBu", as_cmap=True)  # 使用seaborn的配色方案（蓝->青->黄渐变）

    # 样式参数
    plot_params = {
        'figure.dpi': 300,          # 分辨率
        'font.size': 10,            # 统一字体大小
        'axes.titlesize': 12,       # 标题加粗
        'axes.labelsize': 10,       # 坐标轴标签加粗
        'xtick.labelsize': 8,       # 刻度标签缩小
        'ytick.labelsize': 8
    }
    plt.rcParams.update(plot_params)

    # 放在一张图上
    # plt.figure(figsize=(15, 10))
    # for i, method in enumerate(methods, 1):
    #     ax = plt.subplot(2, 3, i)

    # 为每个方法生成独立图表
    for method in methods:
        # 创建新画布
        fig = plt.figure(figsize=(6, 5), dpi=600)
        ax = fig.add_subplot(111)

        # 准备数据
        nodes = list(new_method_scores.keys())
        x = [new_method_scores[n] for n in nodes]
        y = [centrality_scores[method][n] for n in nodes]
        c = [node_influence[n] for n in nodes]

        # 绘制散点图（alpha：透明度，edgecolors：白色边缘增强区分度）
        scatter = ax.scatter(x, y, c=c, cmap=cmap, alpha=0.8, edgecolors='w', linewidth=0.3, s=40, marker='o')

        # 添加趋势线--方法一
        sns.regplot(x=x, y=y, scatter=False, color='red', ax=ax)
        # 添加皮尔逊相关系数标注
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Trendline (R={r:.2f})', transform=ax.transAxes, ha='left')

        # 添加趋势线（多项式拟合）及相关系数--方法二
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # ax.plot(x, p(x),
        #         color='#2F4F4F',  # 深石板灰
        #         linestyle='--',
        #         linewidth=1.5,
        #         label=f'Trendline (R²={np.corrcoef(x, y)[0, 1] ** 2:.2f})'
        #         )

        # 美化图形
        ax.set_xlabel('New Method', fontsize=9, fontweight='bold')
        ax.set_ylabel(f'{method}', fontsize=9, fontweight='bold')
        # ax.set_title(f'VS {method}', fontsize=12)
        # ax.grid(True, alpha=0.3)

        # 添加颜色条
        norm = plt.Normalize(min(c), max(c))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        # cbar.set_label('Infection Capacity', rotation=270, labelpad=15)

        # 图例优化
        # ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', shadow=False)

        # 保存为独立文件
        plt.savefig(f'Comparison_{method}.png', bbox_inches='tight', pad_inches=0.1)

    plt.tight_layout()
    plt.show()


def calculate_monotonicity98(new_method_scores, centrality_scores):
    """
    评估所有方法的单调性
    :param G: 网络图
    :param centrality_scores: 各方法评分字典（含新方法）
    :return: 单调性结果字典
    """
    monotonicity_results = {}
    # 合并所有方法（包括新方法）
    all_scores = {'New Method': new_method_scores, **centrality_scores}
    # 计算各方法单调性
    for method in all_scores:
        unique_scores = set(all_scores[method].values())
        monotonicity_results[method] = len(unique_scores) / len(all_scores[method])

    # 计算相对于度中心性的提升比例（若无Degree则用第一个方法作为基线）
    baseline_method = 'Degree' if 'Degree' in centrality_scores else list(centrality_scores.keys())[0]
    baseline = monotonicity_results[baseline_method]
    improvement_rates = {
        method: (M - baseline) / baseline if baseline != 0 else float('nan')
        for method, M in monotonicity_results.items()
    }

    # 打印完整结果
    print("\n{:<15} | {:<12} | {:<15}".format("Method", "Monotonicity", "Improvement"))
    print("-" * 50)
    for method in sorted(monotonicity_results, key=lambda x: -monotonicity_results[x]):
        print("{:<15} | {:<12.4f} | {:<+15.1%}".format(
            method,
            monotonicity_results[method],
            improvement_rates[method]))

    return monotonicity_results, improvement_rates


def plot_monotonicity_comparison(mono_results, improvement_rates):
    """
    绘制双面板对比图（左：单调性绝对值，右：提升比例）
    """
    methods = list(mono_results.keys())

    # 创建画布
    plt.figure(figsize=(14, 6))

    # 左图：单调性绝对值
    plt.subplot(1, 1, 1)
    colors = ['#FF6B6B' if m == 'New Method' else '#4C72B0' for m in methods]
    bars = plt.bar(methods, [mono_results[m] for m in methods], color=colors)
    plt.title('Monotonicity Comparison')
    plt.ylabel('Monotonicity')
    plt.ylim(0, 1.1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f'{height:.3f}', ha='center')

    # 右图：提升比例
    # plt.subplot(1, 2, 2)
    # colors = ['#FF6B6B' if imp >= 0 else '#DD8452' for imp in improvement_rates.values()]
    # bars = plt.bar(methods, list(improvement_rates.values()), color=colors)
    # plt.title('Improvement Rate vs Baseline')
    # plt.ylabel('Improvement Rate')
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2,
    #              height + (0.01 if height >= 0 else -0.03),
    #              f"{height:+.1%}",
    #              ha='center',
    #              va='bottom' if height >= 0 else 'top')

    # 美化设置
    plt.suptitle('Node Ranking Monotonicity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    for ax in plt.gcf().axes:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig('monotonicity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 计算单调性的函数
def calculate_monotonicity_dange(scores):
    """
    根据提供的评分列表计算单调性
    :param scores: 评分值列表
    :return: 单调性值 (M(R))
    """
    n = len(scores)

    # 处理特殊情况
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0  # 只有一个节点，完全可区分

    # 统计每个评分值的出现次数
    score_counts = Counter(scores)

    # 计算 ∑[n_r(n_r - 1)]
    sum_nr = 0
    for count in score_counts.values():
        if count > 1:
            sum_nr += count * (count - 1)

    # 计算分母 n(n-1)
    denominator = n * (n - 1)

    # 计算单调性
    ratio = sum_nr / denominator
    monotonicity = (1 - ratio) ** 2

    return monotonicity


def caclulate_motonicity_zuizhong(data):
    # 示例输入数据（替换为您的7个方法的结果）
    # method_results = [
    #     {'32': 7.6, '29': 7.066666666666666, '3': 6.933333333333334, '25': 6.9, '28': 6.9},
    #     # 添加其他6个方法的结果字典...
    #     # 示例: 这里复制了第一个方法作为占位符（实际使用时替换为真实数据）
    #     {'32': 7.5, '29': 7.2, '3': 6.8, '25': 6.7, '28': 6.6},
    #     {'32': 7.7, '29': 7.0, '3': 6.9, '25': 6.8, '28': 6.7},
    #     {'32': 7.4, '29': 7.1, '3': 6.95, '25': 6.85, '28': 6.75},
    #     {'32': 7.55, '29': 7.15, '3': 6.85, '25': 6.8, '28': 6.7},
    #     {'32': 7.62, '29': 7.05, '3': 6.92, '25': 6.88, '28': 6.82},
    #     {'32': 7.58, '29': 7.08, '3': 6.94, '25': 6.89, '28': 6.83}
    # ]

    # 定义要计算的比例
    # 节点比例
    percentages = [0.05, 0.1, 0.15, 0.2, 1]
    # 存储所有方法的单调性结果
    results = {}

    # 处理每种方法
    for method_name, node_scores in data.items():
        # 1. 将字典转换为列表并按评分值降序排序
        sorted_items = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        N = len(sorted_items)

        method_results = {}

        for p in percentages:
            # 2. 计算当前比例下的节点数量（向上取整）
            k = max(1, math.ceil(p * N))

            # 3. 取前k个节点的评分值
            top_k_scores = [item[1] for item in sorted_items[:k]]

            # 4. 计算单调性
            monotonicity = calculate_monotonicity_dange(top_k_scores)
            method_results[f'Top {int(p * 100)}%'] = monotonicity

        results[method_name] = method_results

    # 打印结果表格
    print("各方法在不同比例下的单调性结果:")
    print("方法\t\tTop 5%\t\tTop 10%\t\tTop 15%\t\tTop 20%\t\tTop 100%")
    print("-" * 70)

    # 方法名称排序（按原顺序）
    method_names = list(data.keys())

    for method in method_names:
        res = results[method]
        print(
            f"{method.ljust(10)}\t{res['Top 5%']:.6f}\t{res['Top 10%']:.6f}\t{res['Top 15%']:.6f}\t{res['Top 20%']:.6f}\t{res['Top 100%']:.6f}")

    # # 输出为CSV格式
    # print("\nCSV格式:")
    # header = "Method," + ",".join([f"Top{int(p * 100)}%" for p in percentages])
    # print(header)
    #
    # for method in method_names:
    #     res = results[method]
    #     row = f"{method}," + ",".join([f"{res[f'Top {int(p * 100)}%']:.6f}" for p in percentages])
    #     print(row)


# 计算Jaccard相似度的函数
def Jaccard_similarity(set1, set2):
    """计算两个集合之间的Jaccard相似度"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


# 计算并存储所有方法在不同比例下的Jaccard相似度
def compute_jaccard_similarity(node_influence, data):
    # 节点比例
    percentages = [0.05, 0.1, 0.15, 0.2]
    # percentages = list(range(0, 60, 10))

    # 对SIR数据按传播能力降序排序
    sorted_sir = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)
    N = len(sorted_sir)

    correlations = {}
    Jaccard_results = {}
    # 处理每种方法
    for method_name, node_scores in data.items():
        # 1. 将字典转换为列表并按评分值降序排序
        sorted_items = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

        method_results = {}
        method_results1 = {}

        for p in percentages:
            # 2. 计算当前比例下的节点数量（向上取整）
            k = max(1, math.ceil(p * N))

            # 3. 取前k个节点的ID集合
            sir_sets = set(item[0] for item in sorted_sir[:k])      # 取SIR模型下的前p*N个节点
            top_k_nodes = set(item[0] for item in sorted_items[:k])      # 取各类方法的前p*N个节点

            # 4. 计算与SIR模型对应节点集合的Jaccard相似度、kendalltau相似度
            Jaccard = Jaccard_similarity(top_k_nodes, sir_sets)
            method_results[p] = Jaccard

            tau, _ = kendalltau(top_k_nodes, sir_sets)
            method_results1[p] = tau

        Jaccard_results[method_name] = method_results
        correlations[method_name] = method_results1

    # 打印结果表格
    print("各方法在不同比例下与SIR模型的Jaccard相似度:")
    print("方法\t\tTop 5%\t\tTop 10%\t\tTop 15%\t\tTop 20%")
    print("-" * 70)

    # 方法名称排序（按原顺序）
    method_names = list(data.keys())

    for method in method_names:
        res = Jaccard_results[method]
        res1 = correlations[method]
        print(f"{method.ljust(10)}\t{res[0.05]:.4f}\t\t{res[0.1]:.4f}\t\t{res[0.15]:.4f}\t\t{res[0.2]:.4f}")
        print(f"{method.ljust(10)}\t{res1[0.05]:.4f}\t\t{res1[0.1]:.4f}\t\t{res1[0.15]:.4f}\t\t{res1[0.2]:.4f}")

    # # 输出为CSV格式
    # print("\nCSV格式:")
    # header = "Method," + ",".join([f"Top{int(p * 100)}%" for p in percentages])
    # print(header)
    #
    # for method in method_names:
    #     res = Jaccard_results[method]
    #     row = f"{method}," + ",".join([f"{res[p]:.6f}" for p in percentages])
    #     print(row)

    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 设置颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # 绘制每种方法的折线
    for i, method in enumerate(method_names):
        x = percentages
        y = [Jaccard_results[method][p] for p in percentages]
        plt.plot(x, y,
                 label=method,
                 marker=markers[i],
                 markersize=10,
                 linewidth=3,
                 color=colors[i])

    # 添加标题和标签
    # plt.title('Jaccard Similarity with SIR Model', fontsize=16, fontweight='bold')
    plt.xlabel('Top Percentage of Nodes', fontsize=14)
    plt.ylabel('Jaccard Similarity', fontsize=14)

    # 设置横纵坐标范围和刻度
    plt.xticks(percentages, [f'{int(p*100)}%' for p in percentages], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.05)  # Jaccard值在0-1之间

    # 添加图例和网格
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加数据标签
    for method in method_names:
        for p in percentages:
            value = Jaccard_results[method][p]
            plt.text(p, value + 0.02, f'{value:.2f}',
                     fontsize=9, ha='center', va='bottom')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig('jaccard_similarity.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    G = nx.Graph()
    with open(
            'D:\\文件\\论文\\未来的SCI\\数据集\\SW-100-3-0.1-trial3.txt') as f:
        for line in f:
            # u, v, *_ = map(int, line.split())
            u, v, *_ = line.split()  # split(',')：按逗号对字符串进行分割
            if u == v: continue  # 跳过自环边
            try:
                # 如果边已经存在，增加其权重
                G[u][v]['weight'] += 1
            except:
                # 如果边不存在，添加边并设置权重为 1
                G.add_edge(u, v, weight=1)
    # G.add_edges_from([(1, 2),(1,3), (1,4), (1,5), (1,8), (1,9), (2, 3), (2,4), (4, 3), (3, 6), (4,5), (4,6), (5,7), (6,7), (8,9), (8, 11), (9,10), (9,14), (10, 12), (11,12), (12, 13)])
    print("Built graph G")
    print(f"节点数：{len(G)}，边数：{len(G.edges())}")
    # 参数
    n_time = 5  # 选择前n个节点计算耗时
    beta, gamma, T = 0.1, 0.05, 10  # 感染率β, 恢复率γ, 传播轮次
    seed_nums = list(range(5, 40, 1))  # 测试的种子数量   list(range(10))
    trials_SIR = 30  # 每组实验重复次数

    # 调用新方法（输出为：选择全体种子列表, 选择前n个节点耗时, 全体节点评分）
    # seed_order, new_method_time, new_method_scores = select_seeds_method(G, n_time)
    # print(f"选择前n个节点耗时:{new_method_time}")  # 新方法选择n_sort个节点的耗时
    # print(f"所有节点的评分值：{new_method_scores}")    # 新方法对所有节点的评分值

    # 与传统方法比较（输出中包含新方法，输出格式为：种子集 前n个节点耗时 节点评分）
    orders, timings, centrality_scores = compare_centrality(G, n_time)

    # SIR传播并可视化感染曲线，orders.values()对应种子组，orders.keys()对应标签，即不同方法名
    plot_results(trials_SIR, seed_nums, list(orders.values()), list(orders.keys()), beta, gamma, T)

    # 计算单调性
    # caclulate_motonicity_zuizhong(centrality_scores)

    # 计算每个节点单独传播的能力
    # node_influence = {}
    # for node in G.nodes():
    #     node_influence[node] = SIR_model(G, [node], T, beta, gamma)  # 单节点传播

    # 计算不同方法识别出的前k个节点与SIR模型传播能力前k个节点的jaccard相似度
    # compute_jaccard_similarity(node_influence, centrality_scores)

    # # 绘制散点图: new method 与对比算法的皮尔逊相关性及节点影响力颜色区分
    # plot_comparison(new_method_scores, centrality_scores, node_influence)

    # 耗时数据（秒单位）
    times1 = [
        [0.001925, 0.027301, 0.045846, 0.027494, 0.042119, 0.233231],  # New method
        [0.000250, 0.060965, 0.096143, 0.065337, 0.180874, 1.645490],  # Degree
        [0.006631, 2.381488, 1.008413, 2.642306, 3.557883, 114.691085],  # Betweenness
        [0.001860, 0.507100, 0.089505, 0.539438, 0.503202, 23.787558],  # Closeness
        [0.013917, 0.409667, 0.125255, 0.455488, 0.245080, 5.538660],  # Eigenvector
        [0.000199, 0.057520, 0.093783, 0.063211, 0.168284, 1.592530],  # K_shell
        [0.000109, 0.006559, 0.009560, 0.006810, 0.014882, 0.156055],  # Random
    ]
    times2 = [
        [0.021278, 0.022312, 0.022601, 0.023064],
        [0.001366, 0.00229, 0.003416, 0.004243],
        [2.452936, 2.459003, 2.445459, 2.480114],
        [0.524451, 0.534966, 0.526521, 0.532722],
        [0.353405, 0.364492, 0.359767, 0.373923],
        [0.00098, 0.001894, 0.002942, 0.003824],
        [0.000145, 0.000236, 0.000334, 0.000446]
    ]
    datasets = ['ENZYMES_g116', 'SW-100-3-0d1-trial3', 'net science', 'Euro roads', 'ca-CSphd', 'road-minnesota', 'delaunay_n12', 'power-bcspwr10']
    labels_timing = ['new_method', 'degree', 'betweenness', 'closeness', 'eigenvector', 'k_shell', 'random']
    # 重新多次实验，计算各方法平均耗时
    seed_counts = list(range(10, 41, 10))
    # # avg_times = run_repeated_experiments(G, 25, 100)
    # avg_times = run_repeated_experiments1(G, seed_counts, 100)
    # for method, times in avg_times.items():
    #     print(f"{method:12}: {[round(t, 6) for t in times]}")

    # 耗时柱状图
    # # plot_timing_comparison(times1, datasets, labels_timing)
    # plot_timing_for_dataset(list(avg_times.values()), seed_counts, labels_timing, datasets[5])

    # 可视化Kendall系数
    # plot_kendall(correlations)

    # 单调性比较并可视化
    # mono_results, improvement = calculate_monotonicity(new_method_scores, centrality_scores)
    # plot_monotonicity_comparison(mono_results, improvement)

