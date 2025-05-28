
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        # 拳击手步态：更慢、更稳定的步频
        period = 1.5  # 增加步态周期，从0.8增加到1.2，让步态更稳；进一步增加步态周期到1.5
        offset = 0.6  # 调整相位偏移，让步态更不对称
        
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        """改进的接触奖励 - 适应拳击手步态"""
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            # 拳击手步态的接触相位更长
            is_stance = self.leg_phase[:, i] < 0.7  # 增加站立相比例
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _reward_stance_width(self):
        """改进的步宽奖励 - 更激进地鼓励宽站姿"""
        """改进的步宽奖励 - 增加上限约束"""
        left_foot_pos = self.feet_pos[:, 0, :2]
        right_foot_pos = self.feet_pos[:, 1, :2]
        foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        
        # 目标宽度和容许范围
        target_width = 0.25
        tolerance = 0.1
        
        # 使用高斯分布奖励函数，有明确的峰值
        width_reward = torch.exp(-((foot_distance - target_width) / tolerance) ** 2)
        
        # 对极端宽度进行强惩罚
        extreme_penalty = torch.where(foot_distance > 0.4, 
                                    (foot_distance - 0.4) * 10.0, 
                                    torch.zeros_like(foot_distance))
        
        return width_reward * torch.exp(-extreme_penalty)
    
    # def _reward_low_posture(self):
    #     """奖励保持低重心姿态"""
    #     current_height = self.root_states[:, 2]
    #     target_low_height = 0.7
    #     height_reward = torch.exp(-(current_height - target_low_height)**2 / 0.01)
    #     return height_reward

    def _reward_stable_base(self):
        """奖励躯干稳定性"""
        # 限制躯干的横滚和俯仰角度
        roll_pitch = self.base_euler[:, :2]  # roll和pitch角
        stability_penalty = torch.sum(torch.square(roll_pitch), dim=1)
        return torch.exp(-stability_penalty * 10)
    
    def _reward_boxer_stability(self):
        """奖励拳击手式的稳定性 - 低重心+宽站姿组合"""
        """改进的拳击手稳定性 - 更强调宽站姿"""
        # 重心高度因子
        current_height = self.root_states[:, 2]
        height_factor = torch.exp(-(current_height - 0.73)**2 / 0.02)
        
        # 步宽因子 - 增加目标宽度
        left_foot_pos = self.feet_pos[:, 0, :2]
        right_foot_pos = self.feet_pos[:, 1, :2]
        foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        width_factor = torch.exp(-(foot_distance - 0.25)**2 / 0.01)  # 目标宽度增加到0.4
        
        # 髋关节外展因子
        hip_rolls = self.dof_pos[:, [1, 7]]
        target_rolls = torch.tensor([0.15, -0.15], device=self.device)
        hip_errors = torch.abs(hip_rolls - target_rolls)
        hip_factor = torch.mean(torch.exp(-hip_errors * 2.0), dim=1)
        
        # 三重组合奖励
        return height_factor * width_factor #* hip_factor

    # def _reward_forward_lean(self):
    #     """奖励轻微前倾姿态（拳击手特征）"""
    #     # 从四元数计算pitch角
    #     quat = self.root_states[:, 3:7]  # 四元数 [x, y, z, w]
        
    #     # 计算pitch角（绕y轴旋转）
    #     # pitch = arcsin(2 * (w*y - z*x))
    #     w, x, y, z = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]
    #     pitch_angle = torch.asin(2 * (w*y - z*x))
        
    #     target_lean = 0.1  # 轻微前倾约5.7度
    #     lean_reward = torch.exp(-(pitch_angle - target_lean)**2 / 0.05)
    #     return lean_reward

    def _reward_foot_placement(self):
        """奖励合理的脚步位置（拳击手步态）"""
        # 左右脚的前后位置差异（拳击手通常一脚略微在前）
        left_foot_x = self.feet_pos[:, 0, 0]
        right_foot_x = self.feet_pos[:, 1, 0]
        
        # 期望左脚稍微在前（或根据需要调整）
        x_diff = left_foot_x - right_foot_x
        target_x_diff = 0.1  # 左脚在前10cm
        
        placement_reward = torch.exp(-(x_diff - target_x_diff)**2 / 0.02)
        return placement_reward

    def _reward_knee_bend(self):
        """改进的膝关节弯曲奖励"""
        # 确保膝关节索引正确
        knee_joints = [3, 9]  # 根据URDF调整
        knee_angles = self.dof_pos[:, knee_joints]
        
        # 拳击手姿态的理想膝关节角度
        target_knee_angles = torch.tensor([0.65, 0.65], device=self.device)
        
        knee_errors = torch.abs(knee_angles - target_knee_angles)
        knee_rewards = torch.exp(-knee_errors * 3.0)
        
        return torch.mean(knee_rewards, dim=1)

    # def _reward_hip_stability(self):
    #     """髋关节稳定性 - 保持拳击手宽髋姿态"""
    #     hip_roll_joints = [1, 7]  # 髋关节roll索引
    #     hip_rolls = self.dof_pos[:, hip_roll_joints]
        
    #     # 期望的髋关节外展角度
    #     target_rolls = torch.tensor([0.15, -0.15], device=self.device)
        
    #     roll_errors = torch.abs(hip_rolls - target_rolls)
    #     return torch.sum(torch.exp(-roll_errors * 5.0), dim=1)
    
    def _reward_hip_abduction(self):
        """专门奖励髋关节外展（拳击手宽站姿核心）"""
        """改进的髋关节外展奖励 - 增加上限约束"""
        hip_roll_left = 1   
        hip_roll_right = 7  
        
        hip_rolls = self.dof_pos[:, [hip_roll_left, hip_roll_right]]
        
        # 目标外展角度（适中）
        target_rolls = torch.tensor([0.2, -0.2], device=self.device)
        
        # 基础奖励
        roll_errors = torch.abs(hip_rolls - target_rolls)
        base_rewards = torch.exp(-roll_errors * 3.0)
        
        # 对极端外展进行惩罚
        max_safe_abduction = 0.35
        extreme_penalty = torch.sum(
            torch.clamp(torch.abs(hip_rolls) - max_safe_abduction, min=0) ** 2, 
            dim=1
        )
        
        return torch.sum(base_rewards, dim=1) * torch.exp(-extreme_penalty * 10.0)
    
    # def _reward_wide_stance_bonus(self):
    #     """额外的宽站姿奖励"""
    #     # 计算脚间距离
    #     left_foot_pos = self.feet_pos[:, 0, :2]
    #     right_foot_pos = self.feet_pos[:, 1, :2]
    #     foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        
    #     # 多层次奖励不同的宽度
    #     width_thresholds = [0.3, 0.35, 0.4, 0.45]
    #     bonus = torch.zeros_like(foot_distance)
        
    #     for i, threshold in enumerate(width_thresholds):
    #         bonus += torch.where(foot_distance > threshold, 
    #                         torch.ones_like(foot_distance) * (i + 1) * 0.2,
    #                         torch.zeros_like(foot_distance))
        
    #     return bonus

    def _reward_gait_naturalness(self):
        """奖励自然的步态模式，防止极端姿态"""
        # 限制步宽在合理范围内
        left_foot_pos = self.feet_pos[:, 0, :2]
        right_foot_pos = self.feet_pos[:, 1, :2]
        foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        
        # 合理的步宽范围：0.15-0.35m
        ideal_range_min = 0.15
        ideal_range_max = 0.35
        
        # 在合理范围内给予奖励，超出范围给予惩罚
        in_range_reward = torch.where(
            (foot_distance >= ideal_range_min) & (foot_distance <= ideal_range_max),
            torch.ones_like(foot_distance),
            torch.exp(-(torch.clamp(foot_distance - ideal_range_max, min=0) + 
                    torch.clamp(ideal_range_min - foot_distance, min=0)) * 3.0)
        )
        
        # 限制髋关节外展角度在合理范围内
        hip_rolls = self.dof_pos[:, [1, 7]]
        max_abduction = 0.4  # 最大外展角度限制
        hip_penalty = torch.sum(torch.clamp(torch.abs(hip_rolls) - max_abduction, min=0) ** 2, dim=1)
        
        return in_range_reward * torch.exp(-hip_penalty * 5.0)

    def _reward_movement_efficiency(self):
        """奖励运动效率，防止过度晃动"""
        # 基于躯干稳定性和能耗的综合评估
        base_stability = torch.exp(-torch.sum(self.base_ang_vel[:, :2] ** 2, dim=1) * 2.0)
        
        # 关节速度的平滑性
        joint_vel_penalty = torch.sum(torch.abs(self.dof_vel), dim=1)
        efficiency = torch.exp(-joint_vel_penalty * 0.01)
        
        return base_stability * efficiency
    
    def _reward_hip_yaw_penalty(self):
        """惩罚髋关节偏航过度，防止翘二郎腿"""
        # 髋关节偏航索引
        hip_yaw_left = 0   # left_hip_yaw_joint
        hip_yaw_right = 6  # right_hip_yaw_joint
        
        hip_yaws = self.dof_pos[:, [hip_yaw_left, hip_yaw_right]]
        
        # 期望的髋关节偏航角度（接近中性位置）
        target_yaws = torch.tensor([0.0, 0.0], device=self.device)
        
        # 计算偏差
        yaw_errors = torch.abs(hip_yaws - target_yaws)
        
        # 对过度偏航进行强惩罚
        penalty = torch.sum(yaw_errors ** 2, dim=1)
        
        # 特别惩罚右腿内收（翘二郎腿姿态）
        right_hip_yaw = hip_yaws[:, 1]  # 右髋偏航
        cross_leg_penalty = torch.clamp(-right_hip_yaw, min=0) ** 2  # 惩罚负值（内收）
        
        return torch.exp(-penalty * 5.0) * torch.exp(-cross_leg_penalty * 10.0)

    def _reward_leg_separation(self):
        """奖励双腿分开，防止交叉"""
        # 获取髋关节偏航角度
        left_hip_yaw = self.dof_pos[:, 0]
        right_hip_yaw = self.dof_pos[:, 6]
        
        # 计算双腿是否有交叉趋势
        leg_crossing = left_hip_yaw + right_hip_yaw  # 如果都向内，和会是负值
        
        # 惩罚交叉，奖励分开
        separation_reward = torch.exp(-torch.clamp(-leg_crossing, min=0) * 8.0)
        
        return separation_reward