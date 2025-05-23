import torch
import numpy as np
import random
from typing import List, Tuple, Dict


class ClusteringSolution:
    """
    클러스터링 솔루션을 표현하는 클래스
    
    각 솔루션은 여러 클러스터의 집합이며, 각 노드는 여러 클러스터에 속할 수 있음 (중복 허용)
    """
    
    def __init__(self, 
                 num_nodes: int, 
                 num_clusters: int, 
                 num_items: int,
                 membership_matrix: torch.Tensor = None):
        """
        Args:
            num_nodes: 노드(도시) 수
            num_clusters: 클러스터 수
            num_items: 품목 수
            membership_matrix: 멤버십 행렬 (Optional). None인 경우 랜덤 초기화.
                               Shape는 (num_nodes, num_clusters)이고, 값은 0~1사이 확률
        """
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.num_items = num_items
        
        if membership_matrix is None:
            # 0~1 사이의 랜덤한 멤버십 값으로 초기화 (소프트 클러스터링)
            self.membership_matrix = torch.rand(num_nodes, num_clusters)
            
            # 초기화 시 각 노드가 적어도 하나의 클러스터에 강하게 속하도록 보장
            # 이렇게 하면 모든 클러스터에 최소한 일부 노드가 할당될 가능성이 높아짐
            for node_idx in range(num_nodes):
                # 각 노드에 대해 랜덤한 클러스터를 선택하여 높은 멤버십 값 할당
                cluster_idx = torch.randint(0, num_clusters, (1,)).item()
                self.membership_matrix[node_idx, cluster_idx] = 0.8 + 0.2 * torch.rand(1).item()  # 0.8~1.0 사이의 값
        else:
            self.membership_matrix = membership_matrix
            
        self.fitness = None  # 적합도 (낮을수록 좋음)
    
    def ensure_all_nodes_assigned(self, threshold: float = 0.5):
        """
        모든 노드가 적어도 하나의 클러스터에 강하게 할당되도록 보장
        """
        # 각 노드가 속한 클러스터의 최대 멤버십 값 확인
        for node_idx in range(self.num_nodes):
            max_membership = torch.max(self.membership_matrix[node_idx]).item()

            # 노드가 어떤 클러스터에도 강하게 할당되지 않은 경우(최대 멤버십이 임계값 미만)
            if max_membership < threshold:
                # 랜덤 클러스터 선택하여 강한 멤버십 할당
                cluster_idx = torch.randint(0, self.num_clusters, (1,)).item()
                self.membership_matrix[node_idx, cluster_idx] = 0.8 + 0.2 * torch.rand(1).item()
    
    def mutate(self, mutation_rate: float = 0.1, mutation_amount: float = 0.2):
        """
        멤버십 값을 확률적으로 변경하여 돌연변이 발생시킴
        """
        new_membership = self.membership_matrix.clone()
        
        # 각 멤버십 값에 대해 돌연변이 적용
        mask = torch.rand_like(new_membership) < mutation_rate
        delta = (torch.rand_like(new_membership) * 2 - 1) * mutation_amount
        new_membership[mask] += delta[mask]
        
        # 0~1 범위로 클리핑
        new_membership = torch.clamp(new_membership, 0.0, 1.0)
        
        mutated = ClusteringSolution(self.num_nodes, self.num_clusters, self.num_items, new_membership)
        
        # 모든 노드가 할당되도록 보장
        mutated.ensure_all_nodes_assigned()
        
        return mutated
    
    def crossover(self, other: 'ClusteringSolution') -> Tuple['ClusteringSolution', 'ClusteringSolution']:
        """
        다른 솔루션과 교차 연산 수행
        """
        # 기존 교차 연산 코드
        crossover_point = random.randint(1, self.num_nodes - 1)

        child1_membership = torch.zeros_like(self.membership_matrix)
        child2_membership = torch.zeros_like(self.membership_matrix)

        child1_membership[:crossover_point] = self.membership_matrix[:crossover_point].clone()
        child1_membership[crossover_point:] = other.membership_matrix[crossover_point:].clone()

        child2_membership[:crossover_point] = other.membership_matrix[:crossover_point].clone()
        child2_membership[crossover_point:] = self.membership_matrix[crossover_point:].clone()

        child1 = ClusteringSolution(self.num_nodes, self.num_clusters, self.num_items, child1_membership)
        child2 = ClusteringSolution(self.num_nodes, self.num_clusters, self.num_items, child2_membership)

        # 모든 노드가 할당되도록 보장
        child1.ensure_all_nodes_assigned()
        child2.ensure_all_nodes_assigned()

        return child1, child2
    
    def get_crisp_clusters(self, threshold: float = 0.5) -> List[List[int]]:
        """
        멤버십 행렬을 바탕으로 명확한 클러스터 할당 결과 반환
        
        Args:
            threshold: 클러스터에 포함시킬 멤버십 임계값
            
        Returns:
            List[List[int]]: 각 클러스터에 포함된 노드 인덱스 리스트
        """
        try:
            # 임계값 검증
            if threshold <= 0 or threshold > 1:
                print(f"Warning: Invalid threshold {threshold}, using default 0.5")
                threshold = 0.5
                
            clusters = [[] for _ in range(self.num_clusters)]
            
            for node_idx in range(self.num_nodes):
                for cluster_idx in range(self.num_clusters):
                    try:
                        # 각 요소에 안전하게 접근
                        membership_value = self.membership_matrix[node_idx, cluster_idx].item()
                        if membership_value >= threshold:
                            clusters[cluster_idx].append(node_idx)
                    except Exception as e:
                        print(f"Error accessing membership matrix at ({node_idx}, {cluster_idx}): {str(e)}")
                        continue
            
            return clusters
        except Exception as e:
            print(f"Error in get_crisp_clusters: {str(e)}")
            # 오류 발생 시 빈 클러스터 반환
            return [[] for _ in range(self.num_clusters)]


class GeneticClusteringAlgorithm:
    """
    중복 허용 클러스터링을 위한 유전 알고리즘 클래스
    """
    
    def __init__(self,
                 net_demand: torch.Tensor,
                 dist_matrix: torch.Tensor,
                 num_clusters: int = 3,
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 elite_size: int = 5,
                 distance_weight: float = 0.5,
                 demand_balance_weight: float = 0.5,
                 min_nodes_per_cluster: int = 2):
        """
        Args:
            net_demand: 각 노드별 시간대별 수요(-)/공급(+) 데이터, shape (num_nodes, num_items)
            dist_matrix: 노드 간 거리 행렬, shape (num_nodes, num_nodes)
            num_clusters: 클러스터링할 클러스터 수
            population_size: 유전 알고리즘의 인구 크기
            mutation_rate: 돌연변이 발생 확률
            elite_size: 다음 세대로 그대로 넘어갈 엘리트 솔루션 수
            distance_weight: 거리 비용의 가중치 (0~1)
            demand_balance_weight: 수요/공급 균형 비용의 가중치 (0~1)
            min_nodes_per_cluster: 각 클러스터에 필요한 최소 노드 수
        """
        self.net_demand = net_demand
        self.dist_matrix = dist_matrix
        self.num_nodes = net_demand.shape[0]
        self.num_items = net_demand.shape[1]
        self.num_clusters = num_clusters
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        self.distance_weight = distance_weight
        self.demand_balance_weight = demand_balance_weight
        self.min_nodes_per_cluster = min_nodes_per_cluster
        
        # 유효한 클러스터 제약 확인
        if self.num_nodes < self.num_clusters * self.min_nodes_per_cluster:
            print(f"경고: 노드 수({self.num_nodes})가 클러스터당 최소 노드 수 요구사항을 만족시키기에 충분하지 않습니다.")
            print(f"필요한 최소 노드 수: {self.num_clusters} 클러스터 x {self.min_nodes_per_cluster} 노드 = {self.num_clusters * self.min_nodes_per_cluster} 노드")
            print(f"최소 노드 수 요구사항을 {min(self.min_nodes_per_cluster, 1)}로 낮춥니다.")
            self.min_nodes_per_cluster = min(self.min_nodes_per_cluster, 1)
        
        # 초기 인구 생성
        self.population = []
        for _ in range(population_size):
            solution = self._create_balance_focused_solution()
            # 솔루션 적합도 초기화
            solution.fitness = self.evaluate_fitness(solution)
            self.population.append(solution)
    
    def _create_balance_focused_solution(self) -> ClusteringSolution:
        """균형에 특화된 초기 솔루션 생성"""
        solution = ClusteringSolution(self.num_nodes, self.num_clusters, self.num_items)

        # 전체 수요/공급 계산
        total_supply_nodes = []
        total_demand_nodes = []
        neutral_nodes = []

        for node_idx in range(self.num_nodes):
            node_total = sum(self.net_demand[node_idx, t].item() for t in range(self.num_items))
            if node_total > 0:
                total_supply_nodes.append((node_idx, node_total))
            elif node_total < 0:
                total_demand_nodes.append((node_idx, abs(node_total)))
            else:
                neutral_nodes.append(node_idx)

        # 공급/수요 크기별 정렬
        total_supply_nodes.sort(key=lambda x: x[1], reverse=True)
        total_demand_nodes.sort(key=lambda x: x[1], reverse=True)

        # 각 클러스터에 균형잡힌 노드 할당
        for cluster_idx in range(self.num_clusters):
            cluster_supply_target = 0
            cluster_demand_target = 0

            # 큰 공급 노드부터 할당
            if total_supply_nodes:
                supply_node, supply_amount = total_supply_nodes.pop(0)
                solution.membership_matrix[supply_node, cluster_idx] = 0.9
                cluster_supply_target += supply_amount

            # 균형을 맞출 수요 노드들 찾기
            remaining_demand = cluster_supply_target
            assigned_demand_nodes = []

            for demand_node, demand_amount in total_demand_nodes[:]:
                if remaining_demand <= 0:
                    break
                if demand_amount <= remaining_demand * 1.2:  # 20% 여유
                    assigned_demand_nodes.append((demand_node, demand_amount))
                    total_demand_nodes.remove((demand_node, demand_amount))
                    remaining_demand -= demand_amount

            # 수요 노드들 할당
            for demand_node, _ in assigned_demand_nodes:
                solution.membership_matrix[demand_node, cluster_idx] = 0.9

        return solution

    def split_node_demands(self, solution: ClusteringSolution) -> Dict[Tuple[int, int], float]:
        """
        중복 노드의 수요/공급을 클러스터 간에 분할합니다.

        Args:
            solution: 클러스터링 솔루션

        Returns:
            Dict[Tuple[int, int], float]: (노드 인덱스, 클러스터 인덱스) -> 분할 비율(0~1) 매핑
        """
        # 임계값 이상의 멤버십을 가진 클러스터 확인
        threshold = 0.5  # 클러스터 소속 임계값
        belongs_to = {}  # 노드별 소속 클러스터

        for node_idx in range(self.num_nodes):
            belongs_to[node_idx] = []
            for cluster_idx in range(self.num_clusters):
                if solution.membership_matrix[node_idx, cluster_idx] >= threshold:
                    belongs_to[node_idx].append(cluster_idx)

        # 노드별 수요/공급 분할 비율 계산
        split_ratios = {}

        for node_idx, clusters in belongs_to.items():
            if len(clusters) > 1:  # 중복 노드인 경우
                # 멤버십 값에 비례하여 분할
                membership_sum = sum(solution.membership_matrix[node_idx, c].item() for c in clusters)

                for cluster_idx in clusters:
                    ratio = solution.membership_matrix[node_idx, cluster_idx].item() / membership_sum
                    split_ratios[(node_idx, cluster_idx)] = ratio
            elif len(clusters) == 1:  # 하나의 클러스터에만 속한 경우
                split_ratios[(node_idx, clusters[0])] = 1.0
            # 어떤 클러스터에도 속하지 않는 경우 분할 비율 없음 (evaluate_fitness에서 처리)

        return split_ratios

    def balance_aware_crossover(self, other: 'ClusteringSolution') -> Tuple['ClusteringSolution', 'ClusteringSolution']:
        """균형을 고려한 교배"""
        # 각 클러스터별 균형 점수 계산
        self_balance_scores = self.calculate_cluster_balance_scores()
        other_balance_scores = other.calculate_cluster_balance_scores()
        
        child1_membership = torch.zeros_like(self.membership_matrix)
        child2_membership = torch.zeros_like(self.membership_matrix)
        
        # 클러스터별로 더 균형잡힌 부모에서 상속
        for cluster_idx in range(self.num_clusters):
            if self_balance_scores[cluster_idx] < other_balance_scores[cluster_idx]:
                # self가 더 균형잡힘
                child1_membership[:, cluster_idx] = self.membership_matrix[:, cluster_idx]
                child2_membership[:, cluster_idx] = other.membership_matrix[:, cluster_idx]
            else:
                # other가 더 균형잡힘
                child1_membership[:, cluster_idx] = other.membership_matrix[:, cluster_idx]
                child2_membership[:, cluster_idx] = self.membership_matrix[:, cluster_idx]
        
        return ClusteringSolution(self.num_nodes, self.num_clusters, self.num_items, child1_membership), \
               ClusteringSolution(self.num_nodes, self.num_clusters, self.num_items, child2_membership)

    def calculate_cluster_balance_scores(self) -> List[float]:
        """각 클러스터의 균형 점수 계산 (낮을수록 좋음)"""
        clusters = self.get_crisp_clusters()
        balance_scores = []

        for cluster in clusters:
            if not cluster:
                balance_scores.append(float('inf'))
                continue

            total_imbalance = 0.0
            for item_idx in range(self.num_items):
                cluster_sum = sum(self.net_demand[node_idx, item_idx].item() for node_idx in cluster)
                total_imbalance += abs(cluster_sum)

            balance_scores.append(total_imbalance)

        return balance_scores


    def evaluate_fitness(self, solution: ClusteringSolution) -> torch.Tensor:
        """
        솔루션의 적합도를 평가 (낮을수록 좋음)

        Args:
            solution: 평가할 솔루션

        Returns:
            torch.Tensor: 적합도 점수 (낮을수록 좋음)
        """
        try:
            # 크리스프 클러스터 할당 (threshold=0.5)
            clusters = solution.get_crisp_clusters(threshold=0.5)

            # 모든 클러스터가 비어있는 경우 처리
            if all(len(cluster) == 0 for cluster in clusters):
                # print("Returning penalty: All clusters are empty")
                return torch.tensor(float('inf'))

            # 노드 수요/공급 분할 비율 계산 - split_node_demands 메서드 사용
            split_ratios = self.split_node_demands(solution)

            # 각 페널티 초기화
            cluster_size_penalty = 0.0
            empty_cluster_penalty = 0.0
            unassigned_nodes_penalty = 0.0

            # 모든 할당된 노드 추적 (집합으로 중복 제거)
            assigned_nodes = set()
            for cluster in clusters:
                for node_idx in cluster:
                    assigned_nodes.add(node_idx)

            # 미할당 노드 수 계산 및 페널티 부여
            num_unassigned = self.num_nodes - len(assigned_nodes)
            if num_unassigned > 0:
                unassigned_nodes_penalty = 5000.0 * num_unassigned
                # 디버깅용 - 어떤 노드가 할당되지 않았는지 확인
                # unassigned_list = [i for i in range(self.num_nodes) if i not in assigned_nodes]
                # print(f"Unassigned nodes: {unassigned_list}")

            # 클러스터 크기 제약 페널티
            for cluster_idx, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    empty_cluster_penalty += 5000.0  # 비어있는 클러스터에 페널티
                elif len(cluster) < self.min_nodes_per_cluster:
                    cluster_size_penalty += 1000.0 * (self.min_nodes_per_cluster - len(cluster))  # 부족한 노드 수에 비례

            # 클러스터별 수요/공급 균형 계산
            balance_penalty = 0.0
            valid_clusters = 0
            partial_valid_clusters = 0

            # 거리 비용 계산
            distance_cost = 0.0
            clusters_with_distance = 0

            for cluster_idx, cluster in enumerate(clusters):
                if len(cluster) == 0:  # 빈 클러스터 건너뛰기
                    continue

                # 각 품목별로 수요/공급 균형 확인
                items_balance = True
                cluster_imbalance = 0.0
                balanced_items = 0

                for item_idx in range(self.num_items):
                    # 클러스터 내 모든 노드의 수요/공급 합산 (노드 분할 적용)
                    cluster_net_demand = 0.0

                    for node_idx in cluster:
                        # split_node_demands에서 계산한 분할 비율 가져오기
                        ratio = split_ratios.get((node_idx, cluster_idx), 0.0)
                        if ratio == 0.0:
                            # 비율이 0이면 경고 출력하고 기본값 설정
                            # print(f"Warning: Node {node_idx} in cluster {cluster_idx} has zero ratio")
                            ratio = 1.0  # 기본값으로 1.0 사용

                        # 비율에 따라 수요/공급 값 분할
                        original_demand = self.net_demand[node_idx, item_idx].item()
                        node_demand = original_demand * ratio
                        cluster_net_demand += node_demand

                    # 허용 오차 범위
                    tolerance = 0.1
                    is_balanced = abs(cluster_net_demand) <= tolerance

                    if is_balanced:
                        balanced_items += 1
                    else:
                        items_balance = False
                        cluster_imbalance += abs(cluster_net_demand)

                # 불균형 정도에 비례한 페널티
                balance_penalty += 100.0 * cluster_imbalance

                # 완전히 유효한 클러스터 (모든 품목이 균형)
                if items_balance:
                    valid_clusters += 1

                # 부분적으로 유효한 클러스터 (적어도 하나의 품목이 균형)
                if balanced_items > 0:
                    partial_valid_clusters += 1

                # 클러스터 내 노드 간 거리 비용 계산
                if len(cluster) > 1:
                    cluster_distance = 0.0
                    edge_count = 0

                    for i in range(len(cluster)):
                        for j in range(i+1, len(cluster)):
                            node_i, node_j = cluster[i], cluster[j]
                            cluster_distance += self.dist_matrix[node_i, node_j].item()
                            edge_count += 1

                    if edge_count > 0:
                        # 엣지 수로 정규화
                        distance_cost += cluster_distance / edge_count
                        clusters_with_distance += 1

            # 거리 비용 정규화
            if clusters_with_distance > 0:
                distance_cost /= clusters_with_distance

            # 유효한 클러스터가 하나도 없는 경우
            if valid_clusters == 0:
                if partial_valid_clusters == 0:
                    # 부분적으로도 유효한 클러스터가 없으면 큰 페널티
                    return torch.tensor(10000.0 + empty_cluster_penalty + cluster_size_penalty + unassigned_nodes_penalty)
                else:
                    # 부분적으로 유효한 클러스터가 있으면 개선 가능성 있음
                    balance_penalty *= 0.5  # 균형 페널티 감소

            # 총 적합도 계산
            total_fitness = (
                self.distance_weight * distance_cost + 
                cluster_size_penalty +
                empty_cluster_penalty +
                unassigned_nodes_penalty +  # 추가된 페널티
                self.demand_balance_weight * balance_penalty
            )

            # 음수 또는 NaN 값 방지
            if torch.isnan(torch.tensor(total_fitness)) or total_fitness < 0:
                return torch.tensor(float('inf'))

            return torch.tensor(total_fitness)

        except Exception as e:
            # 오류 발생 시 큰 값 반환
            print(f"Fitness evaluation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return torch.tensor(float('inf'))
    
    def evaluate_population(self):
        """모든 솔루션의 적합도 평가"""
        for solution in self.population:
            try:
                solution.fitness = self.evaluate_fitness(solution)
                # 확인: 적합도가 None이면 오류 로그 출력하고 재계산
                if solution.fitness is None:
                    print("Warning: Fitness calculated as None, recalculating...")
                    solution.fitness = self.evaluate_fitness(solution)
                    if solution.fitness is None:
                        # 여전히 None이면 큰 값 할당 (나쁜 솔루션으로 간주)
                        print("Error: Fitness still None after recalculation, setting to large value")
                        solution.fitness = torch.tensor(float('inf'))
            except Exception as e:
                print(f"Error evaluating fitness: {str(e)}")
                # 오류 발생 시 큰 값 할당 (나쁜 솔루션으로 간주)
                solution.fitness = torch.tensor(float('inf'))
    
    def select_parents(self) -> List[ClusteringSolution]:
        """
        토너먼트 선택 방식으로 부모 선택
        
        Returns:
            List[ClusteringSolution]: 선택된 부모 솔루션들
        """
        tournament_size = 3
        selected_parents = []
        
        # 모든 솔루션에 적합도가 있는지 확인
        valid_population = []
        for solution in self.population:
            if solution.fitness is None:
                solution.fitness = self.evaluate_fitness(solution)
            valid_population.append(solution)
        
        for _ in range(self.population_size):
            # 토너먼트 참가자 무작위 선택
            tournament = random.sample(valid_population, min(tournament_size, len(valid_population)))
            # 가장 적합도가 높은(값이 낮은) 참가자 선택
            winner = min(tournament, key=lambda x: float('inf') if x.fitness is None else x.fitness)
            selected_parents.append(winner)
            
        return selected_parents
    
    def create_next_generation(self):
        """다음 세대 생성"""
        # 현재 인구 적합도 평가 - 모든 솔루션에 대해 확실히 적합도 계산
        for solution in self.population:
            if solution.fitness is None:
                solution.fitness = self.evaluate_fitness(solution)
        
        # 엘리트 솔루션 선택 (적합도가 가장 좋은 솔루션들)
        self.population.sort(key=lambda x: x.fitness)
        elites = self.population[:self.elite_size]
        
        # 부모 선택
        parents = self.select_parents()
        
        # 다음 세대 생성
        next_generation = elites.copy()  # 엘리트는 그대로 다음 세대로
        
        # 교차와 돌연변이를 통해 나머지 자식 생성
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = parent1.crossover(parent2)
            
            # 돌연변이 적용
            child1 = child1.mutate(self.mutation_rate)
            child2 = child2.mutate(self.mutation_rate)
            
            # 자식 적합도 즉시 계산
            child1.fitness = self.evaluate_fitness(child1)
            child2.fitness = self.evaluate_fitness(child2)
            
            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)
                
        self.population = next_generation
    
    def run(self, num_generations: int = 100) -> ClusteringSolution:
        """
        유전 알고리즘 실행
        
        Args:
            num_generations: 실행할 세대 수
            
        Returns:
            ClusteringSolution: 최적의 솔루션
        """
        best_fitness_history = []
        
        # 초기 인구의 적합도 평가
        self.evaluate_population()
        
        # 안전하게 최고 솔루션 찾기
        def get_best_solution():
            # None 값을 가진 솔루션 제외
            valid_solutions = [s for s in self.population if s.fitness is not None]
            if not valid_solutions:
                # 만약 유효한 솔루션이 없다면, 모든 솔루션의 적합도 다시 계산
                for s in self.population:
                    s.fitness = self.evaluate_fitness(s)
                valid_solutions = self.population
            return min(valid_solutions, key=lambda x: x.fitness)
        
        for generation in range(num_generations):
            # 다음 세대 생성 (내부에서 적합도 평가 수행)
            self.create_next_generation()
            
            # 현재 세대의 최고 솔루션 (안전하게 검색)
            best_solution = get_best_solution()
            best_fitness_history.append(best_solution.fitness)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_solution.fitness:.4f}")
        
        # 최종 최적 솔루션 반환 (안전하게 검색)
        return get_best_solution(), best_fitness_history
