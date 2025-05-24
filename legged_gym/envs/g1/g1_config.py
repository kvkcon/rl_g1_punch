from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.65] # x,y,z [m] # 降低初始高度，从0.8降到0.65
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint' : 0.15,    # 增加髋关节外展，加宽站姿
            'left_hip_roll_joint' : 0.1,             
            'left_hip_pitch_joint' : -0.3,   # 增加髋关节弯曲，降低重心      
            'left_knee_joint' : 0.6,         # 增加膝关节弯曲
            'left_ankle_pitch_joint' : -0.3, # 调整踝关节保持平衡
            'left_ankle_roll_joint' : 0,     
            'right_hip_yaw_joint' : -0.15,   # 右侧对称
            'right_hip_roll_joint' : -0.1, 
            'right_hip_pitch_joint' : -0.3,                                     
            'right_knee_joint' : 0.6,                                           
            'right_ankle_pitch_joint': -0.3,                            
            'right_ankle_roll_joint' : 0,       
            'torso_joint' : 0.
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
      
    class commands:
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
            'hip_roll': 150,
            'hip_pitch': 150,
            'knee': 200,       # 增加膝关节刚度
            'ankle': 60,
        } # [N*m/rad]
        damping = {
            'hip_yaw': 3,      # 增加阻尼
            'hip_roll': 3,
            'hip_pitch': 3,
            'knee': 6,
            'ankle': 3,
        } # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.6  # 降低目标高度，从0.78降到0.6
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

            # 添加新的奖励项
            stance_width = 2.0      # 奖励保持宽步态
            low_posture = 1.0       # 奖励低重心姿态
            knee_bend = 0.5         # 奖励膝关节弯曲

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

  
