import itertools

def kendall_tau_distance(a, b):
    n = len(a)
    distance = 0

    for pair_a, pair_b in itertools.combinations(range(n), 2):
        # a에서의 순서 쌍과 b에서의 순서 쌍 간의 비교
        a_i, a_j = pair_a, pair_b
        b_i, b_j = b.index(a[a_i]), b.index(a[a_j])

        # 순서 쌍 간의 차이 계산
        if (a_i < a_j and b_i > b_j) or (a_i > a_j and b_i < b_j):
            distance += 1

    return distance

# 예시 수열 A와 B
A = [[0, 0, 0, 1, 2, 1, 1, 1, 6, 1], [0, 1, 0, 0, 1, 2, 0, 2, 1, 5], [0, 1, 0, 0, 1, 3, 2, 4, 2, 3], [3, 2, 2, 4, 2, 3, 4, 7, 7, 9], [3, 2, 5, 8, 4, 4, 8, 8, 9, 9], [2, 3, 3, 5, 3, 6, 5, 5, 7, 9], [4, 3, 4, 7, 6, 5, 6, 6, 7, 9], [6, 9, 7, 7, 8, 6, 7, 8, 8, 9], [4, 5, 4, 6, 7, 4, 6, 8, 9, 9], [5, 6, 3, 5, 5, 8, 7, 8, 8, 9]]
B = [[0, 0, 0, 1, 1, 1, 1, 1, 2, 6], [0, 0, 0, 0, 1, 1 ,1, 2, 2, 5], [0, 0 ,0, 1 ,1, 2 ,2, 3 ,3 ,4], [2 ,2 ,2, 3 ,3 ,4 ,4, 7, 7 ,9],
     [2, 3, 4, 4, 5, 8, 8, 8, 9, 9], [2, 3, 3, 3, 5, 5 ,5 ,6 ,7 ,9], [3, 4, 4, 5, 6, 6, 6, 7, 7, 9], [6, 6, 7, 7, 7, 8, 8, 8, 9, 9], [4, 4, 4, 5, 6, 6, 7, 8, 9, 9], [3, 5 ,5 ,5 ,6, 7, 8, 8, 8, 9]]
# 각 하위 리스트에 대한 Kendall Tau Distance를 계산하여 더함
total_distance = sum(kendall_tau_distance(a, b) for a, b in zip(A, B))

# 결과 출력
print("Total Kendall Tau Distance:", total_distance)
