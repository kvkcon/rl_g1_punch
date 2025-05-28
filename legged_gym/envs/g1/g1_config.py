from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.65] # x,y,z [m] # 降低初始高度，从0.8降到0.65；再改到0.55
        default_joint_angles = {
            'left_hip_yaw_joint' : 0.078,    # From image
            'left_hip_roll_joint' : 0.3,     # From image
            'left_hip_pitch_joint' : -0.56,  # From image
            'left_knee_joint' : 0.58,        # From image
            'left_ankle_pitch_joint' : 0.0,  # From image
            'left_ankle_roll_joint' : -0.092, # From image
            'right_hip_yaw_joint' : -0.15,   # From image
            'right_hip_roll_joint' : -0.28,  # From image
            'right_hip_pitch_joint' : 0.21,  # From image                                     
            'right_knee_joint' : 0.12,       # From image                                           
            'right_ankle_pitch_joint': -0.23, # From image                            
            'right_ankle_roll_joint' : 0.26, # From image       
            'torso_joint' : -0.25            # From image (waist_yaw_joint)
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      
    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        
        class ranges:
            lin_vel_x = [-0.8, 0.8]  # 降低最大速度
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {
            'hip_yaw': 150,    # 增加髋关节刚度
            'hip_roll': 120,   # 降低hip_roll刚度，允许更大外展
            'hip_pitch': 180,
            'knee': 200,       # 增加膝关节刚度
            'ankle': 60,
        } # [N*m/rad]
        damping = {
            'hip_yaw': 3,      # 增加阻尼
            'hip_roll': 2,
            'hip_pitch': 3,
            'knee': 6,
            'ankle': 3,
        } # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.3   # 增加动作幅度
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof_punch.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.73  # 降低目标高度，从0.78降到0.6；再降低到0.5
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5     
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -1.0         
            base_height = -10.0        
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 1.0        # 重要：促进正常步态
            collision = -1.0
            action_rate = -0.01        
            dof_pos_limits = -10.0
            alive = 2.5
            contact_no_vel = -0.1
            feet_swing_height = -5.0
            contact = 1.0              # 重要：接触奖励
            
            # 简化的拳击手特征 - 只保留关键的4个
            stance_width = 1.0         # 使用新的有约束版本
            hip_abduction = 1.0        # 使用新的有约束版本  
            gait_naturalness = 0.3     # 新增：防止极端姿态
            movement_efficiency = 1.0   # 新增：运动效率
            
            # 暂时注释掉这些容易冲突的奖励
            # low_posture = 1.5
            boxer_stability = 0.7  
            # forward_lean = 0.5
            foot_placement = 1.0
            knee_bend = 1.0
            # hip_stability = 0.0
            # hip_pos = 0.0  # 也暂时禁用，因为它可能与hip_abduction冲突

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  
