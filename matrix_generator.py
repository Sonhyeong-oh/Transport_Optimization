import numpy as np

def generate_zero_sum_matrix(rows, cols, scale=10, seed=None):
    """
    각 열의 합이 0이 되는 랜덤 행렬 생성

    Args:
        rows (int): 행 개수
        cols (int): 열 개수
        scale (int): 숫자의 크기 범위 (e.g., 10이면 ±10 이하)
        seed (int, optional): 랜덤 시드

    Returns:
        np.ndarray: (rows x cols) 크기의 행렬
    """
    if seed is not None:
        np.random.seed(seed)

    matrix = np.zeros((rows, cols), dtype=int)

    for c in range(cols):
        # 앞의 rows-1개를 무작위로 생성
        values = np.random.randint(-scale, scale + 1, size=rows - 1)
        # 마지막 값을 조정해서 합이 0이 되게 함
        last_value = -np.sum(values)
        column = np.append(values, last_value)
        # 순서를 랜덤하게 섞기
        np.random.shuffle(column)
        matrix[:, c] = column

    return matrix

# 예시 사용
matrix = generate_zero_sum_matrix(rows=18, cols=3, scale=100, seed=42)
print(matrix)
print("열별 합:", matrix.sum(axis=0))
