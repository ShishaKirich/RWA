from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional


def solve(graph_file: str, requests_file: str) -> Dict[Tuple[int, int], dict]:
    graph = defaultdict(list)

    # Построение ориентированного графа
    with open(graph_file, 'r') as f:
        n_vertices, n_edges = map(int, f.readline().split())
        for _ in range(n_edges):
            u, v = map(int, f.readline().split())
            graph[u].append(v)

    # Чтение запросов
    with open(requests_file, 'r') as f:
        n_wavelengths = int(f.readline())
        n_requests = int(f.readline())
        requests = [tuple(map(int, f.readline().split())) for _ in range(n_requests)]

    return optimized_balanced_strategy(graph, requests, n_wavelengths)


def optimized_balanced_strategy(graph, requests, n_wavelengths):
    """Оптимизированная сбалансированная стратегия"""
    result = {}
    edge_usage = defaultdict(set)

    # Многоуровневая обработка запросов
    processing_stages = [
        process_immediate_requests,  # Очень простые запросы
        process_short_requests,  # Короткие пути
        process_medium_requests,  # Средней сложности
        process_challenging_requests  # Сложные запросы
    ]

    remaining_requests = requests.copy()

    for stage in processing_stages:
        stage_result = stage(graph, remaining_requests, edge_usage, n_wavelengths)
        result.update(stage_result)
        # Обновляем оставшиеся запросы
        remaining_requests = [req for req in remaining_requests if req not in stage_result]

    return result


def process_immediate_requests(graph, requests, edge_usage, n_wavelengths):
    """Обработка мгновенных запросов (u->u и очень короткие)"""
    result = {}

    for u, v in requests:
        if (u, v) in result:
            continue

        if u == v:
            # Запросы к самому себе
            wl = find_available_wavelength(set(), edge_usage, n_wavelengths)
            if wl is not None:
                result[(u, v)] = {
                    'primary': {'wavelength': wl, 'path': [u]},
                    'backup': {'wavelength': wl, 'path': [u]}
                }
                reserve_path([u], wl, edge_usage)

        else:
            # Очень короткие пути (1 ребро)
            if v in graph[u]:
                backup_path = find_one_edge_backup(graph, u, v, {(u, v)})
                if backup_path:
                    primary_edges = {(u, v)}
                    backup_edges = path_to_edges(backup_path)

                    if are_edges_disjoint(primary_edges, backup_edges):
                        common_wl = find_available_wavelength(primary_edges | backup_edges, edge_usage, n_wavelengths)
                        if common_wl is not None:
                            result[(u, v)] = {
                                'primary': {'wavelength': common_wl, 'path': [u, v]},
                                'backup': {'wavelength': common_wl, 'path': backup_path}
                            }
                            reserve_path([u, v], common_wl, edge_usage)
                            reserve_path(backup_path, common_wl, edge_usage)

    return result


def process_short_requests(graph, requests, edge_usage, n_wavelengths):
    """Обработка коротких запросов (пути 2-3 ребра)"""
    result = {}

    for u, v in requests:
        if (u, v) in result:
            continue

        # Ищем пути длиной 2-4 вершины (1-3 ребра)
        short_paths = [p for p in find_shortest_paths_safe(graph, u, v, 8) if 2 <= len(p) <= 4]

        for primary_path in short_paths:
            primary_edges = path_to_edges(primary_path)

            # Ищем короткий резервный путь
            backup_path = find_short_disjoint_backup(graph, u, v, primary_edges, max_length=4)
            if not backup_path:
                continue

            backup_edges = path_to_edges(backup_path)

            if are_edges_disjoint(primary_edges, backup_edges):
                # Предпочитаем одну длину волны
                common_wl = find_available_wavelength(primary_edges | backup_edges, edge_usage, n_wavelengths)
                if common_wl is not None:
                    result[(u, v)] = {
                        'primary': {'wavelength': common_wl, 'path': primary_path},
                        'backup': {'wavelength': common_wl, 'path': backup_path}
                    }
                    reserve_path(primary_path, common_wl, edge_usage)
                    reserve_path(backup_path, common_wl, edge_usage)
                    break

                # Пробуем разные длины волн
                primary_wl = find_available_wavelength(primary_edges, edge_usage, n_wavelengths)
                backup_wl = find_available_wavelength(backup_edges, edge_usage, n_wavelengths)
                if primary_wl is not None and backup_wl is not None:
                    result[(u, v)] = {
                        'primary': {'wavelength': primary_wl, 'path': primary_path},
                        'backup': {'wavelength': backup_wl, 'path': backup_path}
                    }
                    reserve_path(primary_path, primary_wl, edge_usage)
                    reserve_path(backup_path, backup_wl, edge_usage)
                    break

    return result


def process_medium_requests(graph, requests, edge_usage, n_wavelengths):
    """Обработка запросов средней сложности"""
    result = {}

    for u, v in requests:
        if (u, v) in result:
            continue

        solution = find_balanced_solution(graph, u, v, edge_usage, n_wavelengths)
        if solution:
            primary_path, primary_wl, backup_path, backup_wl = solution

            if (is_path_valid(graph, primary_path) and
                    is_path_valid(graph, backup_path) and
                    are_paths_strictly_disjoint(primary_path, backup_path)):
                result[(u, v)] = {
                    'primary': {'wavelength': primary_wl, 'path': primary_path},
                    'backup': {'wavelength': backup_wl, 'path': backup_path}
                }
                reserve_path(primary_path, primary_wl, edge_usage)
                reserve_path(backup_path, backup_wl, edge_usage)

    return result


def process_challenging_requests(graph, requests, edge_usage, n_wavelengths):
    """Обработка сложных запросов (последний шанс)"""
    result = {}

    for u, v in requests:
        if (u, v) in result:
            continue

        # Используем более агрессивный поиск для сложных случаев
        solution = find_aggressive_solution(graph, u, v, edge_usage, n_wavelengths)
        if solution:
            primary_path, primary_wl, backup_path, backup_wl = solution

            if (is_path_valid(graph, primary_path) and
                    is_path_valid(graph, backup_path) and
                    are_paths_strictly_disjoint(primary_path, backup_path)):
                result[(u, v)] = {
                    'primary': {'wavelength': primary_wl, 'path': primary_path},
                    'backup': {'wavelength': backup_wl, 'path': backup_path}
                }
                reserve_path(primary_path, primary_wl, edge_usage)
                reserve_path(backup_path, backup_wl, edge_usage)

    return result


def find_one_edge_backup(graph, u, v, forbidden_edges):
    """Находит резервный путь для однореберного основного пути"""
    if u == v:
        return [u]

    # Ищем путь длиной 2 ребра (3 вершины)
    for neighbor in graph[u]:
        if neighbor != v and (u, neighbor) not in forbidden_edges:
            if v in graph[neighbor] and (neighbor, v) not in forbidden_edges:
                return [u, neighbor, v]

    # Ищем любой альтернативный путь
    return find_disjoint_backup_path(graph, u, v, forbidden_edges)


def find_short_disjoint_backup(graph, start, end, forbidden_edges, max_length=4):
    """Находит короткий непересекающийся резервный путь"""
    if start == end:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        curr, path = queue.popleft()

        if len(path) > max_length:
            continue

        if curr == end:
            return path

        if curr not in graph:
            continue

        for neigh in graph[curr]:
            edge = (curr, neigh)
            if neigh not in visited and edge not in forbidden_edges:
                visited.add(neigh)
                queue.append((neigh, path + [neigh]))

    return None


def find_balanced_solution(graph, start, end, edge_usage, n_wavelengths):
    """Сбалансированный поиск решения"""
    if start == end:
        wl = find_available_wavelength(set(), edge_usage, n_wavelengths)
        if wl is not None:
            return [start], wl, [start], wl
        return None

    # Ищем непересекающиеся пути среди кратчайших
    paths = find_shortest_paths_safe(graph, start, end, 6)

    for primary_path in paths:
        primary_edges = path_to_edges(primary_path)

        backup_path = find_disjoint_backup_path(graph, start, end, primary_edges)
        if not backup_path:
            continue

        backup_edges = path_to_edges(backup_path)

        if are_edges_disjoint(primary_edges, backup_edges):
            common_wl = find_available_wavelength(primary_edges | backup_edges, edge_usage, n_wavelengths)
            if common_wl is not None:
                return primary_path, common_wl, backup_path, common_wl

            primary_wl = find_available_wavelength(primary_edges, edge_usage, n_wavelengths)
            backup_wl = find_available_wavelength(backup_edges, edge_usage, n_wavelengths)
            if primary_wl is not None and backup_wl is not None:
                return primary_path, primary_wl, backup_path, backup_wl

    return None


def find_aggressive_solution(graph, start, end, edge_usage, n_wavelengths):
    """Агрессивный поиск решения для сложных случаев"""
    if start == end:
        wl = find_available_wavelength(set(), edge_usage, n_wavelengths)
        if wl is not None:
            return [start], wl, [start], wl
        return None

    # Ищем все возможные пути
    all_paths = find_shortest_paths_safe(graph, start, end, 15)
    if not all_paths:
        return None

    # Пробуем все комбинации для нахождения непересекающейся пары
    for i, path1 in enumerate(all_paths):
        for j, path2 in enumerate(all_paths):
            if i == j:
                continue

            edges1 = path_to_edges(path1)
            edges2 = path_to_edges(path2)

            if not are_edges_disjoint(edges1, edges2):
                continue

            # Пробуем одну длину волны
            common_wl = find_available_wavelength(edges1 | edges2, edge_usage, n_wavelengths)
            if common_wl is not None:
                return path1, common_wl, path2, common_wl

            # Пробуем разные длины волн
            wl1 = find_available_wavelength(edges1, edge_usage, n_wavelengths)
            wl2 = find_available_wavelength(edges2, edge_usage, n_wavelengths)
            if wl1 is not None and wl2 is not None:
                return path1, wl1, path2, wl2

    return None


# Вспомогательные функции остаются без изменений
def find_disjoint_backup_path(graph, start, end, forbidden_edges):
    if start == end:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        curr, path = queue.popleft()
        if curr == end:
            return path

        if curr not in graph:
            continue

        for neigh in graph[curr]:
            edge = (curr, neigh)
            if neigh not in visited and edge not in forbidden_edges:
                visited.add(neigh)
                queue.append((neigh, path + [neigh]))

    return None


def are_paths_strictly_disjoint(path1, path2):
    edges1 = path_to_edges(path1)
    edges2 = path_to_edges(path2)
    return are_edges_disjoint(edges1, edges2)


def are_edges_disjoint(edges1, edges2):
    return len(edges1 & edges2) == 0


def find_shortest_paths_safe(graph, start, end, k: int) -> List[List[int]]:
    if start not in graph or end not in graph:
        return []

    if start == end:
        return [[start]]

    first_path = bfs_shortest_path(graph, start, end)
    if not first_path:
        return []

    paths = [first_path]
    if k == 1:
        return paths

    candidates = []
    graph_copy = {node: neighbors[:] for node, neighbors in graph.items()}

    for _ in range(min(k - 1, 10)):
        if not paths:
            break

        last_path = paths[-1]

        for i in range(len(last_path) - 1):
            spur_node = last_path[i]
            root_path = last_path[:i + 1]

            removed_edges = []
            for path in paths:
                if len(path) > i and path[:i + 1] == root_path:
                    u, v = path[i], path[i + 1]
                    if v in graph_copy[u]:
                        graph_copy[u].remove(v)
                        removed_edges.append((u, v))

            spur_path = bfs_shortest_path(graph_copy, spur_node, end)

            for u, v in removed_edges:
                graph_copy[u].append(v)

            if spur_path:
                total_path = root_path[:-1] + spur_path
                if total_path not in candidates and total_path not in paths:
                    candidates.append(total_path)

        if not candidates:
            break

        candidates.sort(key=len)
        paths.append(candidates.pop(0))

    return paths


def bfs_shortest_path(graph, start, end) -> Optional[List[int]]:
    if start == end:
        return [start]

    if start not in graph or end not in graph:
        return None

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        curr, path = queue.popleft()
        if curr == end:
            return path

        if curr not in graph:
            continue

        for neigh in graph[curr]:
            if neigh not in visited:
                visited.add(neigh)
                queue.append((neigh, path + [neigh]))

    return None


def find_available_wavelength(edges: Set[Tuple[int, int]],
                              usage: Dict[Tuple[int, int], Set[int]],
                              max_wl: int) -> Optional[int]:
    for wl in range(max_wl):
        if all(wl not in usage[edge] for edge in edges):
            return wl
    return None


def reserve_path(path: List[int], wl: int,
                 usage: Dict[Tuple[int, int], Set[int]]) -> None:
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        usage[(u, v)].add(wl)


def is_path_valid(graph: Dict[int, List[int]], path: List[int]) -> bool:
    if not path:
        return False
    if len(path) == 1:
        return True

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u not in graph or v not in graph[u]:
            return False

    return True


def path_to_edges(path: List[int]) -> Set[Tuple[int, int]]:
    edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges.add((u, v))
    return edges