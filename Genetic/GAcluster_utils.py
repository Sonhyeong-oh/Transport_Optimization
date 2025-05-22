import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib as mpl

# 한글 폰트 설정
def set_korean_font():
    # 기본 폰트로 나눔고딕 설정 시도 (설치된 경우)
    try:
        import platform
        system_name = platform.system()
        
        if system_name == "Windows":
            # Windows인 경우
            font_name = "Malgun Gothic"  # 맑은 고딕
            plt.rc('font', family=font_name)
        elif system_name == "Darwin":
            # macOS인 경우
            font_name = "AppleGothic"
            plt.rc('font', family=font_name)
        else:
            # Linux 등 기타 시스템
            font_name = "NanumGothic"
            plt.rc('font', family=font_name)
            
        # 폰트 경로 직접 설정 방법
        # font_path = 'path/to/your/korean/font.ttf'  # 적절한 폰트 경로로 변경
        # font_prop = fm.FontProperties(fname=font_path)
        # plt.rcParams['font.family'] = font_prop.get_name()
        
        # 마이너스 기호 깨짐 방지
        mpl.rcParams['axes.unicode_minus'] = False
        
        return True
    except:
        print("한글 폰트 설정에 실패했습니다. 영문으로 표시됩니다.")
        return False
        
# 프로그램 시작 시 한글 폰트 설정
set_korean_font()

def visualize_clusters(
    locations: List[str],
    dist_matrix: torch.Tensor,
    clusters: List[List[int]],
    net_demand: torch.Tensor,
    time_period: int = 0,
    min_nodes_per_cluster: int = 2,
    balance_tolerance: float = 0.1
) -> plt.Figure:
    """
    클러스터링 결과를 시각화하는 함수
    
    Args:
        locations: 위치 이름 리스트
        dist_matrix: 거리 행렬
        clusters: 클러스터 할당 결과 (노드 인덱스의 리스트들)
        net_demand: 노드별 수요/공급 데이터
        time_period: 시각화할 시간대 인덱스
        min_nodes_per_cluster: 각 클러스터의 최소 노드 수
        balance_tolerance: 수요/공급 균형 허용 오차
        
    Returns:
        plt.Figure: 시각화 결과 그림
    """
    # 한글 폰트 설정 확인
    set_korean_font()
    
    # 영어 대체 지명 만들기 (한글 폰트 문제 대비)
    english_locations = []
    for i, loc in enumerate(locations):
        if loc == '춘천': english_locations.append('Chuncheon')
        elif loc == '원주': english_locations.append('Wonju')
        elif loc == '강릉': english_locations.append('Gangneung')
        elif loc == '동해': english_locations.append('Donghae')
        elif loc == '태백': english_locations.append('Taebaek')
        elif loc == '속초': english_locations.append('Sokcho')
        elif loc == '삼척': english_locations.append('Samcheok')
        elif loc == '홍천': english_locations.append('Hongcheon')
        elif loc == '횡성': english_locations.append('Hoengseong')
        elif loc == '영월': english_locations.append('Yeongwol')
        elif loc == '평창': english_locations.append('Pyeongchang')
        elif loc == '정선': english_locations.append('Jeongseon')
        elif loc == '철원': english_locations.append('Cheorwon')
        elif loc == '화천': english_locations.append('Hwacheon')
        elif loc == '양구': english_locations.append('Yanggu')
        elif loc == '인제': english_locations.append('Inje')
        elif loc == '고성': english_locations.append('Goseong')
        elif loc == '양양': english_locations.append('Yangyang')
        else: english_locations.append(f'Node{i}')
    
    # 2D MDS를 사용하여 거리 행렬을 2차원 좌표로 변환
    from sklearn.manifold import MDS
    
    # PyTorch 텐서를 NumPy 배열로 변환
    dist_matrix_np = dist_matrix.numpy()
    
    # MDS 적용
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos = mds.fit_transform(dist_matrix_np)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 노드 색상: 수요는 빨간색, 공급은 파란색, 중립은 회색
    node_colors = []
    for i in range(len(locations)):
        demand = net_demand[i, time_period].item()
        if demand < 0:  # 수요 (음수)
            node_colors.append('red')
        elif demand > 0:  # 공급 (양수)
            node_colors.append('blue')
        else:  # 중립
            node_colors.append('gray')
    
    # 노드 크기: 수요/공급 절대값에 비례
    node_sizes = [300 + 100 * abs(net_demand[i, time_period].item()) for i in range(len(locations))]
    
    # 노드 그리기
    for i, (x, y) in enumerate(pos):
        ax.scatter(x, y, color=node_colors[i], s=node_sizes[i], edgecolor='black', linewidth=1, alpha=0.7)
        # 한글 폰트 문제가 없으면 한글 지명 사용, 있으면 영문 지명 사용
        try:
            ax.text(x, y, locations[i], fontsize=10, ha='center', va='center')
        except:
            ax.text(x, y, english_locations[i], fontsize=10, ha='center', va='center')
    
    # 유효한 클러스터 필터링 (최소 노드 수 및 수요/공급 균형 조건 만족)
    # valid_clusters = []
    # for cluster in clusters:
    #     # 최소 노드 수 조건 확인
    #     if len(cluster) < min_nodes_per_cluster:
    #         continue
            
    #     # 수요/공급 균형 조건 확인
    #     cluster_demand = sum(net_demand[node_idx, time_period].item() for node_idx in cluster)
    #     if abs(cluster_demand) > balance_tolerance:
    #         continue
            
    #     valid_clusters.append(cluster)
    
    # 범례 요소
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='수요 노드 (음수)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='공급 노드 (양수)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='중립 노드')
    ]
    
    valid_clusters = [cluster for cluster in clusters if len(cluster) > 0]

    # 클러스터가 있는 경우에만 표시
    if valid_clusters:
        # 클러스터별 색상 맵 생성
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(valid_clusters)))
        
        # 클러스터 연결선 그리기
        for cluster_idx, cluster in enumerate(valid_clusters):
            cluster_color = cluster_colors[cluster_idx]
            
            # 클러스터 내 모든 노드 쌍을 연결
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    node_i, node_j = cluster[i], cluster[j]
                    ax.plot([pos[node_i, 0], pos[node_j, 0]], 
                           [pos[node_i, 1], pos[node_j, 1]], 
                           '-', color=cluster_color, alpha=0.5, linewidth=1.5)
            
            # 클러스터별 범례 추가
            cluster_demand = sum(net_demand[node_idx, time_period].item() for node_idx in cluster)
            try:
                label = f'클러스터 {cluster_idx+1} (수요/공급 합계: {cluster_demand:.1f})'
            except:
                label = f'Cluster {cluster_idx+1} (Demand/Supply Sum: {cluster_demand:.1f})'
                
            legend_elements.append(
                plt.Line2D([0], [0], color=cluster_color, lw=4, label=label)
            )
    
    # 범례 표시
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 제목 및 레이블
    try:
        title = f'강원도 노드 클러스터링 결과 (시간대 {time_period+1}) - 유효한 클러스터만 표시'
        xlabel = 'MDS 좌표 X'
        ylabel = 'MDS 좌표 Y'
    except:
        title = f'Gangwon Province Node Clustering Result (Time Period {time_period+1}) - Valid Clusters Only'
        xlabel = 'MDS Coordinate X'
        ylabel = 'MDS Coordinate Y'
        
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 축 제거
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 유효한 클러스터가 없는 경우 메시지 표시
    if not valid_clusters:
        try:
            message = '유효한 클러스터가 없습니다 (수요/공급 균형 및 최소 노드 수 조건 만족 필요)'
        except:
            message = 'No valid clusters (demand/supply balance and minimum node number conditions required)\nNote: Negative values = Demand, Positive values = Supply'
            
        ax.text(0.5, 0.5, message,
               ha='center', va='center', fontsize=14, color='red',
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_fitness_history(fitness_history: List[float]) -> plt.Figure:
    """
    적합도 역사를 시각화하는 함수
    
    Args:
        fitness_history: 각 세대별 최고 적합도 값 리스트
        
    Returns:
        plt.Figure: 시각화 결과 그림
    """
    # 한글 폰트 설정 확인
    set_korean_font()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(fitness_history, '-o', markersize=4, alpha=0.7)
    
    # 한글 폰트 문제가 있을 수 있으므로 영어 제목도 준비
    try:
        title = '유전 알고리즘 적합도 변화'
        xlabel = '세대'
        ylabel = '최고 적합도 (낮을수록 좋음)'
    except:
        title = 'Genetic Algorithm Fitness History'
        xlabel = 'Generation'
        ylabel = 'Best Fitness (Lower is Better)'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def analyze_clusters(
    clusters: List[List[int]],
    locations: List[str],
    net_demand: torch.Tensor,
    dist_matrix: torch.Tensor,
    min_nodes_per_cluster: int = 2,
    balance_tolerance: float = 0.1
) -> Dict:
    """
    클러스터링 결과 분석 함수
    
    Args:
        clusters: 클러스터 할당 결과 (노드 인덱스의 리스트들)
        locations: 위치 이름 리스트
        net_demand: 노드별 수요/공급 데이터
        dist_matrix: 거리 행렬
        min_nodes_per_cluster: 각 클러스터의 최소 노드 수
        balance_tolerance: 수요/공급 균형 허용 오차
        
    Returns:
        Dict: 분석 결과
    """
    analysis = {
        "cluster_sizes": [],
        "cluster_demands": [],
        "cluster_supplies": [],
        "cluster_balances": [],
        "avg_distances": [],
        "node_memberships": [0] * len(locations),
        "overlapping_nodes": [],
        "valid_clusters": [],  # 유효한 클러스터 (최소 노드 수 및 균형 조건 만족)
        "balance_status": []   # 각 클러스터의 균형 상태
    }
    
    # 각 노드가 속한 클러스터 수 계산
    for cluster in clusters:
        for node_idx in cluster:
            analysis["node_memberships"][node_idx] += 1
    
    # 여러 클러스터에 속한 노드 식별
    for node_idx, count in enumerate(analysis["node_memberships"]):
        if count > 1:
            analysis["overlapping_nodes"].append((node_idx, locations[node_idx], count))
    
    num_time_periods = net_demand.shape[1]
    
    # 클러스터별 분석
    for cluster_idx, cluster in enumerate(clusters):
        # 클러스터 크기
        analysis["cluster_sizes"].append(len(cluster))
        
        # 시간대별 수요/공급 분석
        demands_by_time = []
        supplies_by_time = []
        balances_by_time = []
        
        for time_idx in range(num_time_periods):
            demand = 0
            supply = 0
            
            for node_idx in cluster:
                value = net_demand[node_idx, time_idx].item()
                if value < 0:
                    demand += abs(value)  # 수요는 음수, 절대값으로 변환
                elif value > 0:
                    supply += value  # 공급은 양수
            
            # 균형은 수요 - 공급 (0에 가까울수록 균형이 맞음)
            balance = demand - supply
            demands_by_time.append(demand)
            supplies_by_time.append(supply)
            balances_by_time.append(balance)
        
        analysis["cluster_demands"].append(demands_by_time)
        analysis["cluster_supplies"].append(supplies_by_time)
        analysis["cluster_balances"].append(balances_by_time)
        
        # 클러스터 내 평균 거리 계산
        if len(cluster) > 1:
            total_dist = 0
            pair_count = 0
            
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    node_i, node_j = cluster[i], cluster[j]
                    total_dist += dist_matrix[node_i, node_j].item()
                    pair_count += 1
                    
            avg_dist = total_dist / pair_count if pair_count > 0 else 0
        else:
            avg_dist = 0
            
        analysis["avg_distances"].append(avg_dist)
        
        # 클러스터 유효성 확인 (최소 노드 수 및 균형 조건)
        is_valid_size = len(cluster) >= min_nodes_per_cluster
        is_balanced = all(abs(balance) <= balance_tolerance for balance in balances_by_time)
        
        cluster_status = {
            "cluster_idx": cluster_idx,
            "is_valid_size": is_valid_size,
            "is_balanced": is_balanced,
            "is_valid": is_valid_size and is_balanced
        }
        
        analysis["balance_status"].append(cluster_status)
        
        if is_valid_size and is_balanced:
            analysis["valid_clusters"].append(cluster_idx)
    
    return analysis


def print_cluster_report(
    clusters: List[List[int]],
    locations: List[str],
    net_demand: torch.Tensor,
    analysis: Dict,
    split_ratios: Dict[Tuple[int, int], float] = None,  # 노드 분할 비율 추가
    min_nodes_per_cluster: int = 2,
    balance_tolerance: float = 0.1,
    use_english: bool = False
) -> None:
    """
    클러스터링 결과 리포트 출력 (중복 노드의 수요/공급이 분할되어 표시됨)
    
    Args:
        clusters: 클러스터 할당 결과 (노드 인덱스의 리스트들)
        locations: 위치 이름 리스트
        net_demand: 노드별 수요/공급 데이터
        analysis: analyze_clusters 함수의 분석 결과
        split_ratios: 노드별, 클러스터별 분할 비율 (split_node_demands 함수의 결과)
        min_nodes_per_cluster: 각 클러스터에 필요한 최소 노드 수
        balance_tolerance: 수요/공급 균형 허용 오차
        use_english: 영어로 출력할지 여부
    """
    # 영어 대체 지명 만들기 (한글 출력 문제 대비)
    english_locations = []
    for i, loc in enumerate(locations):
        if loc == '춘천': english_locations.append('Chuncheon')
        elif loc == '원주': english_locations.append('Wonju')
        elif loc == '강릉': english_locations.append('Gangneung')
        elif loc == '동해': english_locations.append('Donghae')
        elif loc == '태백': english_locations.append('Taebaek')
        elif loc == '속초': english_locations.append('Sokcho')
        elif loc == '삼척': english_locations.append('Samcheok')
        elif loc == '홍천': english_locations.append('Hongcheon')
        elif loc == '횡성': english_locations.append('Hoengseong')
        elif loc == '영월': english_locations.append('Yeongwol')
        elif loc == '평창': english_locations.append('Pyeongchang')
        elif loc == '정선': english_locations.append('Jeongseon')
        elif loc == '철원': english_locations.append('Cheorwon')
        elif loc == '화천': english_locations.append('Hwacheon')
        elif loc == '양구': english_locations.append('Yanggu')
        elif loc == '인제': english_locations.append('Inje')
        elif loc == '고성': english_locations.append('Goseong')
        elif loc == '양양': english_locations.append('Yangyang')
        else: english_locations.append(f'Node{i}')
    
    # 한글 표시 문제가 있거나 영어 사용 옵션이 켜진 경우 영어 사용
    try:
        if use_english:
            raise Exception("Use English")
        print("\n===== 클러스터링 결과 보고서 (수요/공급 분할 적용) =====")
        print("\n[모든 클러스터 구성]")
    except:
        print("\n===== Clustering Result Report (With Demand/Supply Split) =====")
        print("\n[All Cluster Compositions]")
    
    valid_cluster_count = 0
    non_empty_cluster_count = 0
    
    for cluster_idx, cluster in enumerate(clusters):
        # 비어있는 클러스터는 출력하지 않음
        if not cluster:
            continue
            
        non_empty_cluster_count += 1
        
        # 클러스터의 제약 조건 확인
        is_valid_size = len(cluster) >= min_nodes_per_cluster
        balances = analysis["cluster_balances"][cluster_idx]
        is_balanced = all(abs(bal) <= balance_tolerance for bal in balances)
        is_valid = is_valid_size and is_balanced
        
        if is_valid:
            valid_cluster_count += 1
            
        # 클러스터 헤더 출력 (유효성 표시 포함)
        try:
            if use_english:
                raise Exception("Use English")
                
            if is_valid:
                print(f"\n클러스터 {cluster_idx+1} ({len(cluster)}개 노드) [유효]:")
            else:
                invalid_reasons = []
                if not is_valid_size:
                    invalid_reasons.append(f"최소 노드 수({min_nodes_per_cluster}) 미만")
                if not is_balanced:
                    invalid_reasons.append("수요/공급 불균형")
                reason_str = ", ".join(invalid_reasons)
                print(f"\n클러스터 {cluster_idx+1} ({len(cluster)}개 노드) [유효하지 않음: {reason_str}]:")
        except:
            if is_valid:
                print(f"\nCluster {cluster_idx+1} ({len(cluster)} nodes) [VALID]:")
            else:
                invalid_reasons = []
                if not is_valid_size:
                    invalid_reasons.append(f"below min nodes({min_nodes_per_cluster})")
                if not is_balanced:
                    invalid_reasons.append("demand/supply imbalance")
                reason_str = ", ".join(invalid_reasons)
                print(f"\nCluster {cluster_idx+1} ({len(cluster)} nodes) [INVALID: {reason_str}]:")
            
        # 클러스터의 노드 정보 출력
        for node_idx in cluster:
            node_name = english_locations[node_idx] if use_english else locations[node_idx]
            
            # 분할 비율이 제공된 경우, 해당 비율로 수요/공급 분할
            if split_ratios and (node_idx, cluster_idx) in split_ratios:
                ratio = split_ratios[(node_idx, cluster_idx)]
                split_demand_values = [net_demand[node_idx, t].item() * ratio for t in range(net_demand.shape[1])]
                demand_str = ", ".join([f"{val:.2f}" for val in split_demand_values])
                
                # 원래 값과 분할 비율 표시
                original_values = [net_demand[node_idx, t].item() for t in range(net_demand.shape[1])]
                original_str = ", ".join([f"{val:.1f}" for val in original_values])
                
                # 중복 노드 표시
                overlapping = node_idx in [n for n, _, _ in analysis["overlapping_nodes"]]
                
                if overlapping:
                    if use_english:
                        print(f"  - {node_name} (Allocated Demand/Supply: [{demand_str}], {ratio:.2f} of [{original_str}])")
                    else:
                        try:
                            print(f"  - {node_name} (할당된 수요/공급: [{demand_str}], 원래의 {ratio:.2f}배 [{original_str}])")
                        except:
                            print(f"  - {node_name} (Allocated Demand/Supply: [{demand_str}], {ratio:.2f} of [{original_str}])")
                else:
                    if use_english:
                        print(f"  - {node_name} (Demand/Supply: [{demand_str}])")
                    else:
                        try:
                            print(f"  - {node_name} (수요/공급: [{demand_str}])")
                        except:
                            print(f"  - {node_name} (Demand/Supply: [{demand_str}])")
            else:
                # 분할 비율이 없는 경우 원래 값 표시
                demand_values = [net_demand[node_idx, t].item() for t in range(net_demand.shape[1])]
                demand_str = ", ".join([f"{val:.1f}" for val in demand_values])
                
                if use_english:
                    print(f"  - {node_name} (Demand/Supply: [{demand_str}]) (Negative values=Demand, Positive values=Supply)")
                else:
                    try:
                        print(f"  - {node_name} (수요/공급: [{demand_str}])")
                    except:
                        print(f"  - {node_name} (Demand/Supply: [{demand_str}])")
        
        # 클러스터 내 수요/공급 균형 - 분할된 값 기준으로 재계산
        if split_ratios:
            # 분할된 수요/공급으로 균형 재계산
            recalculated_balances = []
            for time_idx in range(net_demand.shape[1]):
                cluster_net_demand = 0.0
                for node_idx in cluster:
                    ratio = split_ratios.get((node_idx, cluster_idx), 1.0)
                    value = net_demand[node_idx, time_idx].item() * ratio
                    cluster_net_demand += value
                recalculated_balances.append(cluster_net_demand)
            
            balance_str = ", ".join([f"{bal:.2f}" for bal in recalculated_balances])
            
            try:
                if use_english:
                    raise Exception("Use English")
                print(f"  * 재계산된 시간대별 수요/공급 균형: [{balance_str}]")
            except:
                print(f"  * Recalculated Time-period Demand/Supply Balance: [{balance_str}]")
        else:
            # 원래 값으로 계산된 균형 표시
            balance_str = ", ".join([f"{bal:.1f}" for bal in balances])
            try:
                if use_english:
                    raise Exception("Use English")
                print(f"  * 시간대별 수요/공급 균형: [{balance_str}]")
            except:
                print(f"  * Time-period Demand/Supply Balance: [{balance_str}]")
        
        try:
            if use_english:
                raise Exception("Use English")
            print(f"  * 평균 노드 간 거리: {analysis['avg_distances'][cluster_idx]:.1f}km")
        except:
            print(f"  * Average Distance Between Nodes: {analysis['avg_distances'][cluster_idx]:.1f}km")
    
    # 클러스터 통계 출력
    try:
        if use_english:
            raise Exception("Use English")
        print(f"\n총 클러스터 수: {len(clusters)}, 비어있지 않은 클러스터: {non_empty_cluster_count}, 유효한 클러스터: {valid_cluster_count}")
    except:
        print(f"\nTotal clusters: {len(clusters)}, Non-empty clusters: {non_empty_cluster_count}, Valid clusters: {valid_cluster_count}")
    
    # 할당되지 않은 노드 확인 및 출력 (추가된 부분)
    # 모든 할당된 노드 추적 (집합으로 중복 제거)
    assigned_nodes = set()
    for cluster in clusters:
        for node_idx in cluster:
            assigned_nodes.add(node_idx)
    
    # 할당되지 않은 노드 찾기
    unassigned_nodes = []
    for node_idx in range(len(locations)):
        if node_idx not in assigned_nodes:
            unassigned_nodes.append(node_idx)
    
    # 할당되지 않은 노드 출력
    try:
        if use_english:
            raise Exception("Use English")
        print("\n[할당되지 않은 노드]")
        if unassigned_nodes:
            print(f"총 {len(unassigned_nodes)}개의 노드가 어떤 클러스터에도 할당되지 않았습니다.")
            print("\n할당되지 않은 노드 목록:")
            for idx, node_idx in enumerate(unassigned_nodes):
                node_name = locations[node_idx]
                demand_values = [net_demand[node_idx, t].item() for t in range(net_demand.shape[1])]
                demand_str = ", ".join([f"{val:.1f}" for val in demand_values])
                print(f"{idx+1}. {node_name} (인덱스: {node_idx}, 수요/공급: [{demand_str}])")
        else:
            print("모든 노드가 적어도 하나의 클러스터에 할당되었습니다.")
    except:
        print("\n[Unassigned Nodes]")
        if unassigned_nodes:
            print(f"Total {len(unassigned_nodes)} nodes are not assigned to any cluster.")
            print("\nList of unassigned nodes:")
            for idx, node_idx in enumerate(unassigned_nodes):
                node_name = english_locations[node_idx]
                demand_values = [net_demand[node_idx, t].item() for t in range(net_demand.shape[1])]
                demand_str = ", ".join([f"{val:.1f}" for val in demand_values])
                print(f"{idx+1}. {node_name} (index: {node_idx}, Demand/Supply: [{demand_str}])")
        else:
            print("All nodes are assigned to at least one cluster.")
    
    # 중복 노드 분석 등 다른 출력 부분...

# utils.py 파일에 추가할 클러스터별 시각화 함수

def visualize_clusters_by_cluster(
    locations: List[str],
    dist_matrix: torch.Tensor,
    clusters: List[List[int]],
    net_demand: torch.Tensor,
    analysis: Dict = None,
    balance_tolerance: float = 0.1
) -> plt.Figure:
    """
    클러스터링 결과를 클러스터별로 시각화하는 함수
    
    Args:
        locations: 위치 이름 리스트
        dist_matrix: 거리 행렬
        clusters: 클러스터 할당 결과 (노드 인덱스의 리스트들)
        net_demand: 노드별 수요/공급 데이터
        analysis: analyze_clusters 함수의 분석 결과
        balance_tolerance: 수요/공급 균형 허용 오차
        
    Returns:
        plt.Figure: 시각화 결과 그림
    """
    # 한글 폰트 설정 확인
    set_korean_font()
    
    # 영어 대체 지명 만들기 (한글 폰트 문제 대비)
    english_locations = []
    for i, loc in enumerate(locations):
        if loc == '춘천': english_locations.append('Chuncheon')
        elif loc == '원주': english_locations.append('Wonju')
        elif loc == '강릉': english_locations.append('Gangneung')
        elif loc == '동해': english_locations.append('Donghae')
        elif loc == '태백': english_locations.append('Taebaek')
        elif loc == '속초': english_locations.append('Sokcho')
        elif loc == '삼척': english_locations.append('Samcheok')
        elif loc == '홍천': english_locations.append('Hongcheon')
        elif loc == '횡성': english_locations.append('Hoengseong')
        elif loc == '영월': english_locations.append('Yeongwol')
        elif loc == '평창': english_locations.append('Pyeongchang')
        elif loc == '정선': english_locations.append('Jeongseon')
        elif loc == '철원': english_locations.append('Cheorwon')
        elif loc == '화천': english_locations.append('Hwacheon')
        elif loc == '양구': english_locations.append('Yanggu')
        elif loc == '인제': english_locations.append('Inje')
        elif loc == '고성': english_locations.append('Goseong')
        elif loc == '양양': english_locations.append('Yangyang')
        else: english_locations.append(f'Node{i}')
    
    # 2D MDS를 사용하여 거리 행렬을 2차원 좌표로 변환
    from sklearn.manifold import MDS
    
    # PyTorch 텐서를 NumPy 배열로 변환
    dist_matrix_np = dist_matrix.numpy()
    
    # MDS 적용
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos = mds.fit_transform(dist_matrix_np)
    
    # 모든 비어있지 않은 클러스터 사용
    all_clusters = [(idx, cluster) for idx, cluster in enumerate(clusters) if len(cluster) > 0]

    # 클러스터가 없는 경우
    if not all_clusters:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, '클러스터가 없습니다',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes)
        ax.set_title('강원도 클러스터링 결과 - 클러스터 없음')
        return fig
    
    # 그래프 크기 설정 (클러스터 수에 따라 서브플롯 크기 조정)
    n_clusters = len(all_clusters)
    
    if n_clusters == 1:
        # 클러스터가 1개인 경우, 단일 Axes 객체만 생성
        fig, ax = plt.subplots(figsize=(12, 10))
        axes = np.array([[ax]])  # 2D 배열로 변환
    elif n_clusters == 2:
        # 클러스터가 2개인 경우, 1행 2열 구조
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        axes = np.array([axes])  # 2D 배열로 변환
    else:
        # 클러스터가 3개 이상인 경우
        n_cols = min(2, n_clusters)  # 최대 2열
        n_rows = (n_clusters + n_cols - 1) // n_cols  # 필요한 행 수 계산
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 10 * n_rows))
        # 1D 배열이면 2D로 변환
        if n_rows == 1:
            axes = np.array([axes])
    
    # 클러스터별 시각화
    for i, (cluster_idx, cluster) in enumerate(all_clusters):  # 수정: 튜플 언패킹
        row, col = i // n_cols, i % n_cols
        ax = axes[row][col]
        
        # 클러스터 유효성 검사 추가
        is_valid_size = len(cluster) >= 2  # 최소 노드 수
        is_balanced = True
        for time_idx in range(net_demand.shape[1]):
            cluster_net_demand = sum(net_demand[node_idx, time_idx].item() for node_idx in cluster)
            if abs(cluster_net_demand) > balance_tolerance:
                is_balanced = False
                break
        is_valid = is_valid_size and is_balanced
        
        # 모든 노드 그리기 (회색, 작게)
        for j in range(len(locations)):  # 수정: enumerate 제거
            if j not in cluster:  # 클러스터에 속하지 않는 노드
                x, y = pos[j, 0], pos[j, 1]  # 수정: pos 인덱싱
                ax.scatter(x, y, color='lightgray', s=200, alpha=0.3, edgecolor='gray')
                try:
                    ax.text(x, y, locations[j], fontsize=8, ha='center', va='center', color='gray')
                except:
                    ax.text(x, y, english_locations[j], fontsize=8, ha='center', va='center', color='gray')
        
        # 클러스터에 속한 노드 그리기 (품목별 수요/공급 정보 텍스트로 표시)
        for j in cluster:
            x, y = pos[j, 0], pos[j, 1]
            
            # 노드의 전체 품목 수요/공급을 고려하여 색상 결정 (수정: .item() 추가)
            total_demand = sum(1 for t in range(net_demand.shape[1]) if net_demand[j, t].item() < 0)
            total_supply = sum(1 for t in range(net_demand.shape[1]) if net_demand[j, t].item() > 0)
            
            if total_demand > total_supply:
                node_color = 'red'  # 주로 수요 노드
            elif total_supply > total_demand:
                node_color = 'blue'  # 주로 공급 노드
            else:
                node_color = 'purple'  # 수요와 공급이 비슷한 노드
            
            # 유효하지 않은 클러스터의 노드는 다른 스타일로 표시
            edge_color = 'black' if is_valid else 'red'
            edge_width = 2 if is_valid else 3
            ax.scatter(x, y, color=node_color, s=500, edgecolor=edge_color, 
                      linewidth=edge_width, alpha=0.7)
            
            # 노드 이름 표시
            try:
                ax.text(x, y, locations[j], fontsize=12, ha='center', va='center', weight='bold')
            except:
                ax.text(x, y, english_locations[j], fontsize=12, ha='center', va='center', weight='bold')
            
            # 품목별 수요/공급 정보 표시
            demand_info = []
            for t in range(net_demand.shape[1]):
                value = net_demand[j, t].item()
                if value < 0:
                    demand_info.append(f"품목{t+1}: 수요 {abs(value):.1f}")
                elif value > 0:
                    demand_info.append(f"품목{t+1}: 공급 {value:.1f}")
                else:
                    demand_info.append(f"품목{t+1}: 중립")
            
            # 노드 아래에 수요/공급 정보 텍스트 표시
            info_text = "\n".join(demand_info)
            ax.text(x, y - 0.1, info_text, fontsize=8, ha='center', va='top', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # 클러스터 내 노드 간 연결선 그리기 (유효성에 따라 스타일 변경)
        line_style = '-' if is_valid else '--'
        line_color = 'green' if is_valid else 'orange'
        line_alpha = 0.7 if is_valid else 0.5
        
        for j in range(len(cluster)):
            for k in range(j+1, len(cluster)):
                node_j, node_k = cluster[j], cluster[k]
                ax.plot([pos[node_j, 0], pos[node_k, 0]], 
                      [pos[node_j, 1], pos[node_k, 1]], 
                      line_style, color=line_color, alpha=line_alpha, linewidth=2)
        
        # 클러스터 균형 정보
        balance_info = []
        for t in range(net_demand.shape[1]):
            cluster_demand = sum(net_demand[node_idx, t].item() for node_idx in cluster)
            balance_info.append(f"품목{t+1} 균형: {cluster_demand:.1f}")
        
        balance_text = "\n".join(balance_info)
        ax.text(0.02, 0.02, balance_text, fontsize=12, ha='left', va='bottom', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # 클러스터 제목 (유효성 표시 포함)
        status = "유효" if is_valid else "유효하지 않음"
        ax.set_title(f'클러스터 {cluster_idx+1} ({len(cluster)}개 노드) [{status}]', fontsize=14)
        
        # 축 제거
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # 빈 서브플롯 제거
    for i in range(n_clusters, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        fig.delaxes(axes[row][col])
    
    plt.suptitle('노드 클러스터링 결과 - 모든 클러스터 시각화', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle을 위한 공간 확보
    return fig
