import numpy as np
import matplotlib.pyplot as plt


# 미로의 크기
ROWS = 4
COLS = 3

# 보상 (rewards) 설정
rewards = np.zeros((ROWS, COLS))
rewards[3, 2] = 1   # (3, 4) 위치에 있는 보상은 +1
rewards[3, 1] = -1  # (2, 4) 위치에 있는 보상은 -1

# (2,2) 위치는 갈 수 없으므로 해당 위치의 보상은 0으로 미설정 -> 후에 계산 하지 않음

# 초기 values 설정
values = np.zeros((ROWS, COLS))

#discount factor
discount = 0.9

# 이동 확률
transition_prob = 0.25

# 상 하 좌 우 이동
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# 벨만 방정식을 이용하여 가치 계산
def calculate_values(values, rewards, discount, transition_prob, iterations):
    for _ in range(iterations):
        new_values = np.zeros_like(values)
        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) == (1, 1):
                    # (2, 2) 위치는 갈 수 없으므로 0으로 유지
                    new_values[i, j] = 0
                elif (i, j) == (3, 2) or (i, j) == (3, 1):
                    # 설정한 reward 할당.
                    new_values[i, j] = rewards[i, j]
                else:
                    value = 0
                    for action in actions:
                        # 모든 가능한 행동에 대해 가치를 계산(상,하,좌,우)
                        next_i, next_j = i + action[0], j + action[1]
                        if 0 <= next_i < ROWS and 0 <= next_j < COLS and (next_i, next_j) != (1, 1):
                            value += transition_prob * (rewards[next_i, next_j] + discount * values[next_i, next_j])
                        #(2,2) 또는 벽을 넘어간 경우 그 상태는 고려하지 않으므로 이전 value값과 동일하게 계산
                        else:
                            value += transition_prob * (rewards[i, j] + discount * values[i, j])
                    new_values[i, j] = value
        values = new_values
    return values

# value 계산
values = calculate_values(values, rewards, discount, transition_prob, iterations=100)

# 결과 출력
values=np.rot90(values)
#결과 print
print(values)
#결과 plot
plt.imshow(values, cmap='PuBuGn', origin='upper')
plt.colorbar(label='Value')
plt.title('Value Function')
plt.xlabel('Column')
plt.ylabel('Row')
# 각 셀에 숫자 표시
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        plt.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='white')
plt.show()