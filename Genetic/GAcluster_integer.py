import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

# 모델 및 유틸리티 임포트
from GAcluster_model_integer import GeneticClusteringAlgorithm, ClusteringSolution
from GAcluster_utils_integer import visualize_clusters, visualize_clusters_by_cluster, plot_fitness_history, analyze_clusters, print_cluster_report

# 이미 실행 여부를 추적하는 전역 변수
_algorithm_executed = False

def safe_format_tensor(tensor_value, format_spec=".4f"):
    """Tensor 값을 안전하게 포맷팅하는 유틸리티 함수"""
    try:
        if isinstance(tensor_value, torch.Tensor):
            if tensor_value.numel() == 1:  # 스칼라 텐서인 경우
                return f"{tensor_value.item():{format_spec}}"
            else:
                return str(tensor_value)  # 다차원 텐서는 그대로 문자열로
        else:
            return f"{tensor_value:{format_spec}}"
    except:
        return str(tensor_value)

def main(args):
    global _algorithm_executed
    
    # 중복 실행 방지
    if _algorithm_executed:
        print("알고리즘이 이미 실행되었습니다. 중복 실행을 방지합니다.")
        return
    
    # 디버그 모드 확인
    debug_mode = args.debug
    use_english = args.use_english
    
    # 영어로 출력하는 경우
    if use_english:
        print("==== Gangwon Province Overlapping Clustering - Genetic Algorithm ====")
        print(f"Number of Clusters: {args.num_clusters}")
        print(f"Number of Generations: {args.generations}")
        print(f"Population Size: {args.population_size}")
        print(f"Mutation Rate: {args.mutation_rate}")
        print(f"Distance Weight: {args.distance_weight}")
        print(f"Demand/Supply Balance Weight: {args.demand_balance_weight}")
        print(f"Cluster Membership Threshold: {args.threshold}")
        print(f"Minimum Nodes per Cluster: {args.min_nodes_per_cluster}")
        print(f"Demand/Supply Balance Tolerance: {args.balance_tolerance}")
        print(f"Debug Mode: {debug_mode}")
    else:
        # 한글로 출력
        print("==== 중복 허용 클러스터링 - 유전 알고리즘 ====")
        print(f"클러스터 수: {args.num_clusters}")
        print(f"세대 수: {args.generations}")
        print(f"인구 크기: {args.population_size}")
        print(f"돌연변이 확률: {args.mutation_rate}")
        print(f"거리 가중치: {args.distance_weight}")
        print(f"수요/공급 균형 가중치: {args.demand_balance_weight}")
        print(f"클러스터 포함 임계값: {args.threshold}")
        print(f"클러스터당 최소 노드 수: {args.min_nodes_per_cluster}")
        print(f"수요/공급 균형 허용 오차: {args.balance_tolerance}")
        print(f"디버그 모드: {debug_mode}")
        
    print("=" * 50)

    # 데이터 설정
    locations = ['춘천', '원주', '강릉', '동해', '태백', '속초', '삼척', '홍천', '횡성', '영월',
                '평창', '정선', '철원', '화천', '양구', '인제', '고성', '양양']

    # 노드별 수요/공급 데이터
    fixed_net_demand = torch.tensor([
        [-5, -4,  1], [-5, -5,  5], [-1,  5,  3],
        [-5,  5, -5], [ 5, -5, -5], [ 1, -1,  2],
        [ 1, -2, -3], [-5, -3, -5], [-1, -2,  2],
        [ 3,  5, -1], [-2, -1, -5], [ 1, -3,  0],
        [ 2,  5,  3], [ 3, -5,  3], [-1,  5,  2],
        [ 4,  5,  1], [ 0,  5,  2], [ 5, -4,  0]
    ], dtype=torch.float32)

    # 노드 간 거리 행렬
    dist_matrix = torch.tensor([
        #춘천 원주  강릉  동해  태백 속초  삼척 홍천  횡성  영월 평창  정선  철원 화천  양구 인제  고성  양양
        [0,   84,  160, 200, 222, 108, 209, 38,  61,  150, 119, 148, 78,  30,  44,  82,  116, 114], # 춘천
        [84,  0,   127, 166, 141, 181, 175, 60,  20,  72,  70,  123, 156, 112, 122, 112, 202, 158], # 원주
        [160, 127, 0,   47,  99,  65,  59,  134, 107, 117, 90,  68,  243, 187, 128, 110, 96,  53 ], # 강릉
        [200, 166, 47,  0,   54,  104, 14,  173, 146, 114, 129, 76,  282, 227, 168, 150, 136, 89 ], # 동해
        [222, 141, 99,  54,  0,   164, 47,  189, 159, 64,  84,  52,  287, 249, 216, 198, 185, 142], # 태백
        [108, 181, 65,  104, 164, 0,   118, 105, 138, 171, 144, 138, 221, 118, 67,  50,  25,  17 ], # 속초
        [209, 175, 59,  14,  47,  118, 0,   184, 159, 108, 107, 78,  293, 237, 178, 160, 147, 102], # 삼척
        [38,  60,  134, 173, 189, 105, 184, 0,   34,  124, 87,  116, 123, 65,  64,  54,  109, 90 ], # 홍천
        [61,  20,  107, 146, 159, 138, 159, 34,  0,   95,  55,  85,  145, 87,  97,  83,  178, 119], # 횡성
        [150, 72,  117, 114, 64,  171, 108, 124, 95,  0,   29,  52,  224, 187, 197, 186, 199, 151], # 영월
        [119, 70,  90,  129, 84,  144, 107, 87,  55,  29,  0,   31,  211, 143, 150, 128, 171, 124], # 평창
        [148, 123, 68,  76,  52,  138, 78,  116, 85,  52,  31,  0,   262, 220, 197, 142, 166, 118], # 정선
        [78,  156, 243, 282, 287, 221, 293, 123, 145, 224, 211, 262, 0,   61,  119, 149, 192, 211], # 철원
        [30,  112, 187, 227, 249, 118, 237, 65,  87,  187, 143, 220, 61,  0,   44,  73,  155, 143], # 화천
        [44,  122, 128, 168, 216, 67,  178, 64,  97,  197, 150, 197, 119, 44,  0,   31,  75,  74 ], # 양구
        [82,  112, 110, 150, 198, 50,  160, 54,  83,  186, 128, 142, 149, 73,  31,  0,   57,  55 ], # 인제
        [116, 202, 96,  136, 185, 25,  147, 109, 178, 199, 171, 166, 192, 155, 75,  57,  0,   43 ], # 고성
        [114, 158, 53,  89,  142, 17,  102, 90,  119, 151, 124, 118, 211, 143, 74,  55,  43,  0  ]  # 양양
    ], dtype=torch.float32)

    # 유전 알고리즘 인스턴스 생성
    ga = GeneticClusteringAlgorithm(
        net_demand=fixed_net_demand,
        dist_matrix=dist_matrix,
        num_clusters=args.num_clusters,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        distance_weight=args.distance_weight,
        demand_balance_weight=args.demand_balance_weight,
        min_nodes_per_cluster=args.min_nodes_per_cluster
    )

    # 디버그 모드에서는 초기 인구 적합도 확인
    if debug_mode:
        print("\n초기 인구 적합도 확인:")
        for i, solution in enumerate(ga.population):
            fitness_str = safe_format_tensor(solution.fitness)
            print(f"솔루션 {i}: 적합도 = {fitness_str}")
    
    # 알고리즘 실행
    try:
        print("\n유전 알고리즘 실행 중...")
        best_solution, fitness_history = ga.run(num_generations=args.generations)
        
        # Tensor 값을 안전하게 포맷팅
        final_fitness = safe_format_tensor(best_solution.fitness)
        print(f"완료! 최종 적합도: {final_fitness}")
        
        # 실행 완료 플래그 설정
        _algorithm_executed = True
        
    except Exception as e:
        import traceback
        print(f"알고리즘 실행 중 오류: {str(e)}")
        traceback.print_exc()
        
        # 오류 발생 시 가장 좋은 솔루션 선택
        valid_solutions = [s for s in ga.population if s.fitness is not None]
        if valid_solutions:
            best_solution = min(valid_solutions, key=lambda x: x.fitness)
            fitness_history = [best_solution.fitness]
            recovery_fitness = safe_format_tensor(best_solution.fitness)
            print(f"오류 후 복구된 최적 솔루션 적합도: {recovery_fitness}")
        else:
            print("유효한 솔루션을 찾을 수 없음")
            return

    # 결과 분석
    clusters = best_solution.get_crisp_clusters(threshold=args.threshold)
    print(f"\n클러스터 수: {args.num_clusters}, 실제로 비어있지 않은 클러스터: {sum(1 for c in clusters if c)}")

     # 결과 분석 및 보고서 생성
    try:
        # 노드 분할 비율 계산 - 수정된 메서드 호출
        split_ratios = ga.split_node_demands_integer(best_solution)  # net_demand 매개변수 제거

        analysis = analyze_clusters(
            clusters, 
            locations, 
            fixed_net_demand, 
            dist_matrix, 
            min_nodes_per_cluster=args.min_nodes_per_cluster,
            balance_tolerance=args.balance_tolerance
        )

        print_cluster_report(
            clusters, 
            locations, 
            fixed_net_demand, 
            analysis,
            split_ratios,  # 분할 비율 추가
            min_nodes_per_cluster=args.min_nodes_per_cluster,
            balance_tolerance=args.balance_tolerance,
            use_english=args.use_english
        )
        
        # 유효한 클러스터 수 출력
        if args.use_english:
            print(f"\nNumber of valid clusters (satisfying minimum node count and balance conditions): {len(analysis['valid_clusters'])}")
            if analysis['valid_clusters']:
                print(f"Valid cluster indices: {[idx+1 for idx in analysis['valid_clusters']]}")
        else:
            print(f"\n유효한 클러스터 수 (최소 노드 수 및 균형 조건 만족): {len(analysis['valid_clusters'])}")
            if analysis['valid_clusters']:
                print(f"유효한 클러스터 인덱스: {[idx+1 for idx in analysis['valid_clusters']]}")
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        print("기본 클러스터 정보만 출력합니다.")
        for i, cluster in enumerate(clusters):
            if cluster:
                print(f"클러스터 {i+1}: {[locations[node] for node in cluster]}")
        analysis = None

    # 결과 저장 디렉토리 생성
    results_dir = None  # 초기화 추가
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"\n결과 저장 경로: {results_dir}")

        # 적합도 그래프 저장
        fitness_fig = plot_fitness_history(fitness_history)
        fitness_fig.savefig(f"{results_dir}/fitness_history.png")
        plt.close(fitness_fig)  # 메모리 절약을 위해 닫기

        # 텍스트 보고서 저장
        with open(f"{results_dir}/clustering_report.txt", "w", encoding="utf-8") as f:
            # 기본 정보
            f.write("==== 중복 허용 클러스터링 - 유전 알고리즘 결과 ====\n")
            f.write(f"클러스터 수: {args.num_clusters}\n")
            
            # Tensor 값을 안전하게 포맷팅하여 파일에 저장
            final_fitness_str = safe_format_tensor(best_solution.fitness)
            f.write(f"최종 적합도: {final_fitness_str}\n")
            f.write(f"실제 클러스터 수: {sum(1 for c in clusters if c)}\n\n")

            # 클러스터 구성 (수요/공급 분할 적용)
            f.write("[클러스터 구성 (수요/공급 분할 적용)]\n")
            for cluster_idx, cluster in enumerate(clusters):
                if not cluster:
                    continue

                f.write(f"클러스터 {cluster_idx+1} ({len(cluster)}개 노드):\n")
                for node_idx in cluster:
                    node_name = locations[node_idx]

                    # 분할 비율이 제공된 경우, 해당 비율로 수요/공급 분할
                    if split_ratios and (node_idx, cluster_idx) in split_ratios:
                        ratio = split_ratios[(node_idx, cluster_idx)]
                        split_demand_values = [fixed_net_demand[node_idx, t].item() * ratio for t in range(fixed_net_demand.shape[1])]
                        demand_str = ", ".join([f"{val:.2f}" for val in split_demand_values])

                        # 원래 값과 분할 비율 표시
                        original_values = [fixed_net_demand[node_idx, t].item() for t in range(fixed_net_demand.shape[1])]
                        original_str = ", ".join([f"{val:.1f}" for val in original_values])

                        # 중복 노드 표시 여부 확인
                        overlapping = analysis and node_idx in [n for n, _, _ in analysis["overlapping_nodes"]]

                        if overlapping:
                            f.write(f"  - {node_name} (할당된 수요/공급: [{demand_str}], 원래의 {ratio:.2f}배 [{original_str}])\n")
                        else:
                            f.write(f"  - {node_name} (수요/공급: [{demand_str}])\n")
                    else:
                        # 분할 비율이 없는 경우 원래 값 표시
                        demand_values = [fixed_net_demand[node_idx, t].item() for t in range(fixed_net_demand.shape[1])]
                        demand_str = ", ".join([f"{val:.1f}" for val in demand_values])
                        f.write(f"  - {node_name} (수요/공급: [{demand_str}])\n")

                # 클러스터 내 수요/공급 균형 (클러스터 단위로 출력)
                if analysis:
                    balances = analysis["cluster_balances"][cluster_idx]
                    balance_str = ", ".join([f"{bal:.1f}" for bal in balances])
                    f.write(f"  * 품목별 수요/공급 균형: [{balance_str}]\n")
                    f.write(f"  * 평균 노드 간 거리: {analysis['avg_distances'][cluster_idx]:.1f}km\n\n")
                else:
                    f.write("\n")

            # 중복 노드 분석
            if analysis:
                f.write("\n[중복 노드 분석]\n")
                overlapping = analysis["overlapping_nodes"]
                if overlapping:
                    for node_idx, node_name, count in sorted(overlapping, key=lambda x: x[2], reverse=True):
                        cluster_indices = [idx+1 for idx, cluster in enumerate(clusters) if node_idx in cluster]
                        f.write(f"{node_name}: {count}개 클러스터에 속함 (클러스터: {cluster_indices})\n")
                else:
                    f.write("중복 노드 없음\n")

        print(f"결과가 {results_dir} 디렉토리에 저장되었습니다.")
    else:
        # 적합도 그래프 표시 (저장하지 않을 경우)
        plot_fitness_history(fitness_history)

    # 클러스터별 시각화 (저장 여부와 관계없이 실행)
    try:
        # 클러스터별 시각화
        cluster_fig = visualize_clusters_by_cluster(
            locations, 
            dist_matrix, 
            clusters, 
            fixed_net_demand, 
            analysis=analysis,
            balance_tolerance=args.balance_tolerance
        )

        # 저장하는 경우
        if args.save_results and results_dir:
            cluster_fig.savefig(f"{results_dir}/clusters_by_cluster.png", dpi=300, bbox_inches='tight')
            print(f"클러스터별 시각화가 {results_dir}/clusters_by_cluster.png에 저장되었습니다.")

        # 화면에 표시 (저장 여부와 관계없이)
        plt.show()

        # 메모리 정리를 위해 figure 닫기
        plt.close(cluster_fig)

    except Exception as e:
        print(f"클러스터별 시각화 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()

    # 클러스터 시각화 및 저장 (각 품목별) - 저장하는 경우에만 실행
    if args.save_results and results_dir:
        for item_idx in range(fixed_net_demand.shape[1]):
            try:
                # visualize_clusters_by_cluster 함수에 items 매개변수가 있는지 확인 필요
                # 만약 없다면 visualize_clusters 함수 사용
                cluster_fig = visualize_clusters(
                    locations, 
                    dist_matrix, 
                    clusters, 
                    fixed_net_demand,
                    time_period=item_idx,  # items 대신 time_period 사용
                    min_nodes_per_cluster=args.min_nodes_per_cluster,
                    balance_tolerance=args.balance_tolerance
                )
                cluster_fig.savefig(f"{results_dir}/clusters_time_{item_idx+1}.png", dpi=300, bbox_inches='tight')
                plt.close(cluster_fig)  # 메모리 절약을 위해 닫기
                print(f"품목 {item_idx+1} 시각화가 저장되었습니다.")
            except Exception as e:
                print(f"품목 {item_idx+1} 시각화 중 오류: {str(e)}")



# 중요: 이 블록이 직접 실행될 때만 main 함수를 호출하도록 합니다
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="중복 허용 클러스터링 - 유전 알고리즘")
    
    parser.add_argument("--num_clusters", type=int, default=4, 
                        help="클러스터 수")
    parser.add_argument("--generations", type=int, default=100, 
                        help="실행할 유전 알고리즘 세대 수")
    parser.add_argument("--population_size", type=int, default=1000, 
                        help="유전 알고리즘 인구 크기")
    parser.add_argument("--mutation_rate", type=float, default=0.8, 
                        help="돌연변이 확률")
    parser.add_argument("--elite_size", type=int, default=10, 
                        help="엘리트 솔루션 수")
    parser.add_argument("--distance_weight", type=float, default=0.0, 
                        help="거리 비용의 가중치 (0~1)")
    parser.add_argument("--demand_balance_weight", type=float, default=1.0, 
                        help="수요/공급 균형 비용의 가중치 (0~1)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="클러스터 포함 멤버십 임계값")
    parser.add_argument("--min_nodes_per_cluster", type=int, default=2, 
                        help="각 클러스터에 필요한 최소 노드 수")
    parser.add_argument("--balance_tolerance", type=float, default=0.0, 
                        help="수요/공급 균형 허용 오차")
    parser.add_argument("--save_results", action="store_true", 
                        help="결과 파일 저장 여부")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 모드 활성화")
    parser.add_argument("--use_english", action="store_true",
                        help="영어로 출력 (한글 폰트 문제 해결)")
    
    args = parser.parse_args()
    main(args)