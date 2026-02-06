import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
from torch.optim.lr_scheduler import StepLR

class ActorCritic(nn.Module):
    def __init__(self, L_num, input_dim=3, hidden_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)

        # ===== MAPPO 新增：centralized critic（CTDE，用全局 L×L 状态） =====
        global_input_dim = L_num * L_num   # flatten 后的维度
        self.central_value = nn.Sequential(
            nn.Linear(global_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # with torch.autocast(device_type='cuda', dtype=torch.float16):  # 混合精度
        shared = self.shared(x)
        action_logits = self.actor(shared)

        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(shared)
        state_value = state_value.squeeze()
        return action_probs, state_value

    # ===== MAPPO 新增：centralized critic 的前向（输入全局 L×L 状态） =====
    def forward_central_value(self, global_state):
        """
        global_state: [B, L, L] 的 0/1 策略矩阵
        返回: [B] 的 centralized value
        """
        x = global_state.float().view(global_state.shape[0], -1)  # 展成 B × (L*L)
        v = self.central_value(x)
        return v.squeeze(-1)


class SPGG(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs, 
                    now_time, question, ppo_epochs, batch_size, gae_lambda,
                    output_path, delta, rho, zeta):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time
        
        # PPO超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta = delta  # w_cl
        self.rho = rho      # w_ent

        # Local Cooperation Reward
        self.zeta = zeta   # 建议 0.3 - 1.0，lcr_lambda, 可以先写死

        self.output_path = output_path
        
        # 神经网络
        self.policy = ActorCritic(L_num=self.L_num).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)  # 每1000步学习率降低10%
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # ===== MAPPO 新增：centralized critic 的全局状态 buffer =====
        self.global_states = []
        self.global_next_states = []

    def _init_state(self, question):
        if question == 1: # question 1: 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2: # question 2: 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:  # question 3: 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            # 创建一个 L x L 的零矩阵
            state = torch.zeros((self.L_num, self.L_num))
            # 填充交替的 0 和 1
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state(self, state_matrix):
        """将 2D 网格转换为 4D 张量后填充"""
        # 添加 batch 和 channel 维度 [B, C, H, W]
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 使用正确的填充参数格式 (padding_left, padding_right, padding_top, padding_bottom)
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')  # 四周各填充1
        
        # 计算邻域合作数
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L, L]
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)

    def calculate_reward(self, state_matrix):
        """计算每个智能体参与的5组博弈的总收益"""
        # 1. 对状态矩阵进行padding处理（环形边界）
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 2. 计算每个位置的邻域合作者数量（4邻居）
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        
        # 3. 计算中心智能体作为合作者时的单组收益 (r*n_C/5 - 1)
        c_single_profit = (self.r * neighbor_coop / 5) - 1
        
        # 4. 计算中心智能体作为背叛者时的单组收益 (r*n_C/5)
        d_single_profit = (self.r * neighbor_coop / 5)
        
        # 5. 对单组收益矩阵进行padding处理
        padded_c_profit = F.pad(c_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        padded_d_profit = F.pad(d_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 6. 计算每个智能体参与的5组博弈总收益
        # 中心智能体参与的5组博弈：自身作为中心的1组 + 作为邻居参与的4组
        c_total_profit = F.conv2d(padded_c_profit, self.neibor_kernel).squeeze() + c_single_profit
        d_total_profit = F.conv2d(padded_d_profit, self.neibor_kernel).squeeze() + d_single_profit
        
        # 7. 根据当前策略选择对应的总收益
        reward_matrix = torch.where(state_matrix.bool(), c_total_profit, d_total_profit)

        # ===== 加入 Local Cooperation Reward（方式 A） =====
        lcr = self.compute_lcr(state_matrix)    # [L, L]
        reward_matrix = reward_matrix + self.zeta * lcr
        
        return reward_matrix

    def compute_lcr(self, state_matrix):
        """
        Local Cooperation Reward:
        lcr(i,j) = neighbor_coop(i,j) / 4
        """

        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0),
                       (1,1,1,1), mode='circular')

        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L,L]

        lcr = neighbor_coop / 4.0   # 每个 agent 的局部合作率（0~1）

        return lcr

    def ppo_update(self):
        # 堆叠 buffer
        states = torch.stack(self.states).to(self.device)               # [T, L, L, 3]
        actions = torch.stack(self.actions).to(self.device)             # [T, L, L]
        old_log_probs = torch.stack(self.log_probs).to(self.device)     # [T, L, L]
        rewards = torch.stack(self.rewards).to(self.device)             # [T, L, L]
        next_states = torch.stack(self.next_states).to(self.device)     # [T, L, L, 3]
        dones = torch.stack(self.dones).to(self.device)                 # [T, L, L]

        global_states = torch.stack(self.global_states).to(self.device)         # [T, L, L]
        global_next_states = torch.stack(self.global_next_states).to(self.device) # [T, L, L]

        # ===== 使用 MAPPO centralized critic 计算 V(s), V(s') =====
        with torch.no_grad():
            values_scalar = self.policy.forward_central_value(global_states)          # [T]
            next_values_scalar = self.policy.forward_central_value(global_next_states) # [T]

        # 广播到每个格点，与你原来的 reward/advantage 形状一致
        values = values_scalar.view(-1, 1, 1).expand_as(rewards)         # [T, L, L]
        next_values = next_values_scalar.view(-1, 1, 1).expand_as(rewards)

        # ===== 保持你原来的 GAE & advantage 逻辑不变，只是换了 centralized value =====
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(rewards[-1])

        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values  # [T, L, L]

        # ===== PPO 更新：Actor 用局部特征，Critic 用 centralized critic =====
        for _ in range(self.ppo_epochs):
            for batch in self._make_batch(states, actions, old_log_probs, advantages, returns, global_states):
                state_b, action_b, old_log_b, adv_b, ret_b, gstate_b = batch
                if ret_b.shape[0] == 1:
                    ret_b = ret_b.squeeze()

                # 策略仍用原来的局部特征
                probs, _ = self.policy(state_b)                    # ignore local value
                dist = Categorical(probs)
                log_probs = dist.log_prob(action_b).view_as(action_b)
                entropy = dist.entropy().mean()
                
                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # centralized critic：拟合 team-level return（平均），先保证 ret_b 是 [B, L, L]

                # === 强制将 ret_b 转成 3D 张量 ===
                if ret_b.dim() == 0:
                    # scalar → [1,1,1]
                    ret_b = ret_b.view(1, 1, 1)
                elif ret_b.dim() == 1:
                    # [B] → [B,1,1]
                    ret_b = ret_b.view(-1, 1, 1)
                elif ret_b.dim() == 2:
                    # [L,L] → [1,L,L]
                    ret_b = ret_b.unsqueeze(0)
                # 如果 dim==3 就不用管

                # 现在一定是 [B, L, L]
                target_team = ret_b.mean(dim=(1, 2)).detach()     # [B]

                # centralized critic loss
                value_pred = self.policy.forward_central_value(gstate_b)  # [B]
                critic_loss = F.mse_loss(value_pred, target_team)

                delta = self.delta  # 0.5
                rho = self.rho      # 0.001

                loss = actor_loss + delta * critic_loss - rho * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()  # 更新学习率

    def _make_batch(self, states, actions, old_log_probs, advantages, returns, global_states):
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i+self.batch_size]
            yield (
                states[idx],
                actions[idx],
                old_log_probs[idx],
                advantages[idx],
                returns[idx],
                global_states[idx]
            )

    def run(self):
        coop_rates = []
        defect_rates = []
        total_values = []
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch
            action, log_prob = self.choose_action(self.current_state)
            next_state = action
            reward = self.calculate_reward(next_state)
            done = torch.zeros_like(next_state, dtype=torch.bool)
            
            self.states.append(self.encode_state(self.current_state).view(self.L_num, self.L_num, 3))
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.next_states.append(self.encode_state(next_state).view(self.L_num, self.L_num, 3))
            self.dones.append(done)

            # ===== MAPPO 新增：存 centralized critic 的全局状态 =====
            self.global_states.append(self.current_state.detach().cpu())
            self.global_next_states.append(next_state.detach().cpu())
            
            if epoch == 0:
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch, self.r, profit_matrix))
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = reward.sum().item()
                
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)

            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.ppo_update()      # PPO 更新
                self.current_state = next_state
                self._reset_buffer()   # 清空缓冲区
            else:
                self.current_state = next_state           

            # 在关键时间点保存快照（与原Q-learning相同）
            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch+1, self.r, profit_matrix))

            if epoch % 1000 == 0:
                self.save_checkpoint()

            coop_rate = self.current_state.float().mean().item()
            defect_rate = 1 - coop_rate
            total_value = reward.sum().item()
            
            coop_rates.append(coop_rate)
            defect_rates.append(defect_rate)
            total_values.append(total_value)
        
        self.save_checkpoint(is_final=True)

        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)
    
    async def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保存策略矩阵快照与数据文件（与原Q-learning代码相同格式）"""
        plt.clf()
        plt.close("all")
        
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # =============================================
        # 1. 保存策略矩阵图
        # =============================================
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        ax1.axis('off')
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)
        
        color_map = {
            0: [0, 0, 0],    # 黑色
            1: [1, 1, 1]     # 白色
        }
        
        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color
        
        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)
            
        fig1.savefig(f'{img_dir}/t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)
        
        # =============================================
        # 2. 保存收益热图
        # =============================================
        if isinstance(profit_data, tuple):
            combined_reward, _, team_utility = profit_data
            profit_matrix = combined_reward
        else:
            profit_matrix = profit_data
        
        if not isinstance(profit_matrix, torch.Tensor):
            profit_matrix = torch.tensor(profit_matrix, device=self.device)
        profit_matrix = profit_matrix.cpu().numpy()

        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)

        vmin = 0
        vmax = np.ceil(np.maximum(5*(r-1),4*r))
        im = ax2.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')
        
        cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=28)
        cbar2.set_ticks(np.ceil(np.linspace(vmin, vmax, 5)).astype(int))
        
        ax2.set_xticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.set_yticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.grid(False)

        ax2.axis('off')
        
        fig2.savefig(f'{img_dir}/profit_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                    type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                    profit_matrix, fmt='%.4f')
        return 0

    def _reset_buffer(self):
        """显式释放显存"""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        # ===== MAPPO 新增：清空 centralized critic 的全局状态 buffer =====
        del self.global_states[:]
        del self.global_next_states[:]
        torch.cuda.empty_cache()  # 立即释放未使用的显存
    
    # 修改经验存储逻辑（目前 run() 没用到它，保持原样）
    def _store_transition(self, state, action, log_prob, reward, next_state, done):
        """存储时分离梯度"""
        self.states.append(state.detach().cpu())  # 转移到CPU
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.next_states.append(next_state.detach().cpu())
        self.dones.append(done.detach().cpu())

    def save_checkpoint(self, is_final=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'r': self.r,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
        }
        model_dir = f"{self.output_path}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"model_r{self.r}_final.pth" if is_final else f"model_r{self.r}_epoch{self.epoch}.pth"
        torch.save(checkpoint, f"{model_dir}/{filename}")

    def choose_action(self, state_matrix):
        with torch.no_grad():
            features = self.encode_state(state_matrix)
            probs, _ = self.policy(features)
            dist = Categorical(probs)
            actions = dist.sample()
        return actions.view_as(state_matrix), dist.log_prob(actions).view_as(state_matrix)