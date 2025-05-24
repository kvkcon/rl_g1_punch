
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
        """奖励保持拳击手宽站姿"""
        # 计算左右脚之间的距离
        left_foot_pos = self.feet_pos[:, 0, :2]   # 左脚XY位置
        right_foot_pos = self.feet_pos[:, 1, :2]  # 右脚XY位置
        foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        
        # 理想的步宽（根据机器人尺寸调整）
        target_width = 0.3
        width_error = torch.abs(foot_distance - target_width)
        return torch.exp(-width_error * 10)
    
    def _reward_low_posture(self):
        """奖励保持低重心姿态"""
        current_height = self.root_states[:, 2]
        target_low_height = 0.6
        height_reward = torch.exp(-(current_height - target_low_height)**2 / 0.01)
        return height_reward

    def _reward_stable_base(self):
        """奖励躯干稳定性"""
        # 限制躯干的横滚和俯仰角度
        roll_pitch = self.base_euler[:, :2]  # roll和pitch角
        stability_penalty = torch.sum(torch.square(roll_pitch), dim=1)
        return torch.exp(-stability_penalty * 10)
    
    def _reward_boxer_stability(self):
        """奖励拳击手式的稳定性 - 低重心+宽站姿组合"""
        # 重心高度因子
        current_height = self.root_states[:, 2]
        height_factor = torch.exp(-(current_height - 0.5)**2 / 0.02)
        
        # 步宽因子
        left_foot_pos = self.feet_pos[:, 0, :2]
        right_foot_pos = self.feet_pos[:, 1, :2]
        foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        width_factor = torch.exp(-(foot_distance - 0.35)**2 / 0.01)
        
        # 组合奖励
        return height_factor * width_factor

    def _reward_forward_lean(self):
        """奖励轻微前倾姿态（拳击手特征）"""
        # 从四元数计算pitch角
        quat = self.root_states[:, 3:7]  # 四元数 [x, y, z, w]
        
        # 计算pitch角（绕y轴旋转）
        # pitch = arcsin(2 * (w*y - z*x))
        w, x, y, z = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]
        pitch_angle = torch.asin(2 * (w*y - z*x))
        
        target_lean = 0.1  # 轻微前倾约5.7度
        lean_reward = torch.exp(-(pitch_angle - target_lean)**2 / 0.05)
        return lean_reward

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
        target_knee_angles = torch.tensor([0.7, 0.7], device=self.device)
        
        knee_errors = torch.abs(knee_angles - target_knee_angles)
        knee_rewards = torch.exp(-knee_errors * 3.0)
        
        return torch.mean(knee_rewards, dim=1)

    def _reward_hip_stability(self):
        """髋关节稳定性 - 保持拳击手宽髋姿态"""
        hip_roll_joints = [1, 7]  # 髋关节roll索引
        hip_rolls = self.dof_pos[:, hip_roll_joints]
        
        # 期望的髋关节外展角度
        target_rolls = torch.tensor([0.15, -0.15], device=self.device)
        
        roll_errors = torch.abs(hip_rolls - target_rolls)
        return torch.sum(torch.exp(-roll_errors * 5.0), dim=1)