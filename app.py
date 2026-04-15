import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 0. Streamlit 頁面設定
# ==========================================
st.set_page_config(page_title="RL: Q-learning vs SARSA", layout="wide")
st.title(" Cliff Walking 強化學習演算法比較")
st.markdown("本面板展示 **Q-learning (Off-policy)** 與 **SARSA (On-policy)** 在經典懸崖漫步環境中的路徑選擇與學習成效。")

# ==========================================
# 1. 環境設定
# ==========================================
ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 上, 右, 下, 左
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

def step(state, action_idx):
    r, c = state
    dr, dc = ACTIONS[action_idx]
    next_r, next_c = max(0, min(ROWS - 1, r + dr)), max(0, min(COLS - 1, c + dc))
    next_state = (next_r, next_c)
    
    if next_r == 3 and 1 <= next_c <= 10:
        return START, -100, False  # 掉入懸崖
    if next_state == GOAL:
        return next_state, -1, True # 抵達終點
    return next_state, -1, False

def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(4)
    values = q_table[state[0], state[1], :]
    return np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])

# ==========================================
# 2. 演算法訓練 (使用 st.cache_data 避免重複運算)
# ==========================================
@st.cache_data(show_spinner=False)
def train_agents(episodes=500, runs=50, alpha=0.4, gamma=0.9, epsilon=0.1):
    all_sarsa_rewards = np.zeros(episodes)
    all_q_rewards = np.zeros(episodes)
    final_sarsa_q, final_q_q = None, None
    
    for r in range(runs):
        # SARSA
        s_q = np.zeros((ROWS, COLS, 4))
        s_rewards = []
        for _ in range(episodes):
            state = START
            action = choose_action(state, s_q, epsilon)
            tot_r = 0
            done = False
            while not done:
                next_state, reward, done = step(state, action)
                next_action = choose_action(next_state, s_q, epsilon)
                target = reward + gamma * s_q[next_state[0], next_state[1], next_action] * (not done)
                s_q[state[0], state[1], action] += alpha * (target - s_q[state[0], state[1], action])
                state, action = next_state, next_action
                tot_r += reward
            s_rewards.append(tot_r)
            
        # Q-learning
        q_q = np.zeros((ROWS, COLS, 4))
        q_rewards = []
        for _ in range(episodes):
            state = START
            tot_r = 0
            done = False
            while not done:
                action = choose_action(state, q_q, epsilon)
                next_state, reward, done = step(state, action)
                best_next_a = np.argmax(q_q[next_state[0], next_state[1], :])
                target = reward + gamma * q_q[next_state[0], next_state[1], best_next_a] * (not done)
                q_q[state[0], state[1], action] += alpha * (target - q_q[state[0], state[1], action])
                state = next_state
                tot_r += reward
            q_rewards.append(tot_r)
            
        all_sarsa_rewards += np.array(s_rewards)
        all_q_rewards += np.array(q_rewards)
        
        if r == runs - 1:
            final_sarsa_q, final_q_q = s_q, q_q
            
    return final_sarsa_q, final_q_q, all_sarsa_rewards/runs, all_q_rewards/runs

# ==========================================
# 3. 視覺化繪圖函數
# ==========================================
def plot_policy_grid(q_table, title, path_color):
    """繪製網格與策略方向，並標示出實際走過的軌跡"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.invert_yaxis() # 讓 (0,0) 在左上角，(3,0) 在左下角
    
    # 畫格子
    ax.set_xticks(np.arange(COLS+1))
    ax.set_yticks(np.arange(ROWS+1))
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # 標示特殊區塊 (懸崖、起點、終點)
    for c in range(1, 11):
        ax.add_patch(patches.Rectangle((c, 3), 1, 1, facecolor='lightgray'))
        ax.text(c+0.5, 3.5, 'Cliff', ha='center', va='center', color='black', fontsize=10)
        
    ax.text(0.5, 3.5, 'Start', ha='center', va='center', fontweight='bold', color='green')
    ax.text(11.5, 3.5, 'Goal', ha='center', va='center', fontweight='bold', color='green')

    # 畫策略箭頭 & 追蹤路徑
    path = [START]
    curr_state = START
    # 追蹤 Greedy 路線 (避免無限迴圈，設個上限)
    for _ in range(50):
        if curr_state == GOAL or (curr_state[0] == 3 and 1 <= curr_state[1] <= 10):
            break
        best_a = np.argmax(q_table[curr_state[0], curr_state[1], :])
        next_state, _, _ = step(curr_state, best_a)
        path.append(next_state)
        curr_state = next_state

    # 填入所有格子的最佳動作箭頭
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == GOAL or (r == 3 and 1 <= c <= 10):
                continue
            best_a = np.argmax(q_table[r, c, :])
            ax.text(c+0.5, r+0.5, ACTION_SYMBOLS[best_a], ha='center', va='center', fontsize=16)

    # 畫出最終路徑的虛線
    if len(path) > 1:
        path_x = [p[1] + 0.5 for p in path]
        path_y = [p[0] + 0.5 for p in path]
        ax.plot(path_x, path_y, color=path_color, linestyle='--', linewidth=3, alpha=0.7)

    return fig

# ==========================================
# 4. Streamlit 渲染邏輯
# ==========================================
with st.spinner('正在執行 50 次訓練取平均... (僅首次需等待)'):
    sarsa_q, q_learning_q, sarsa_rewards, q_rewards = train_agents()

st.header("1. 最終策略路徑比較")
st.write("藍色虛線代表 Agent 在 $\\epsilon=0$ (完全貪婪) 時實際行走的最優路線。")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_policy_grid(sarsa_q, "SARSA (On-policy): 安全路徑", "blue"))
with col2:
    st.pyplot(plot_policy_grid(q_learning_q, "Q-learning (Off-policy): 最佳/危險路徑", "red"))

st.divider()

st.header("2. 學習收斂曲線")
st.write("平均 50 次訓練的每回合累積獎勵 (Smoothed)。可以看出 SARSA 在訓練過程中的穩定性。")

# 繪製平滑曲線
window = 10
avg_sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
avg_q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')

fig_curve, ax_curve = plt.subplots(figsize=(10, 4))
ax_curve.plot(avg_sarsa_smooth, label='SARSA', color='c')
ax_curve.plot(avg_q_smooth, label='Q-learning', color='red')
ax_curve.set_xlabel('Episodes')
ax_curve.set_ylabel('Reward Sum per Episode')
ax_curve.set_ylim(-100, 0)
ax_curve.legend()
ax_curve.grid(True)

st.pyplot(fig_curve)

st.success("🎉 運算完成！")