import math

def reward_shaping(
    frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info
):
    reward = 0

    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _env_info.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z
    
    # 获取拿到的宝箱数目
    num_collected_treasures = _env_info.game_info.treasure_count

    # Get the grid-based distance of the current agent's position relative to the end point, buff, and treasure chest
    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    end_dist = _obs.feature.end_pos.grid_distance
    buff_dist = _obs.feature.buff_pos.grid_distance
    treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]

    # 环境宝箱数目
    max_treasures = len(treasure_dists)
    # 宝箱衰减因子, +1 保证大于 > 0, / 2 缩放到[0, 1], 余弦函数控制趋势
    decay_factor = (1 + math.cos(math.pi * num_collected_treasures / max_treasures)) / 2

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = env_info.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the grid-based distance of the agent's position from the previous
    # frame relative to the end point, buff, and treasure chest
    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = obs.feature.end_pos.grid_distance
    prev_buff_dist = obs.feature.buff_pos.grid_distance
    prev_treasure_dists = [pos.grid_distance for pos in obs.feature.treasure_pos]

    # Get the status of the buff
    # 获取buff的状态
    buff_availability = 0
    for organ in env_info.frame_state.organs:
        if organ.sub_type == 2:
            buff_availability = organ.status

    # Get the acceleration status of the agent
    # 获取智能体的加速状态
    prev_speed_up = env_info.frame_state.heroes[0].speed_up
    speed_up = _env_info.frame_state.heroes[0].speed_up

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励

    # Boundary handling: At the first frame, prev_end_dist is initialized to 1,
    # and no reward is calculated at this time
    # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
    if prev_end_dist != 1:
        reward_end_dist += 0.2 if end_dist < prev_end_dist else 0
        # 靠近终点的奖励前期少一点，先拿宝箱
        if num_collected_treasures == len(treasure_dists):
            reward_end_dist = 0.9

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        reward_win += 15
        # 获胜但是宝箱没拿全，加惩罚，用 ( 1 - 宝箱衰减因子 )
        if num_collected_treasures < len(treasure_dists):
            rate = 1 - decay_factor
            reward_win *= rate

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)

    if len(treasure_dists) > 0:
        min_treasure_dist = min(treasure_dists)
        min_prev_treasure_dist = min(prev_treasure_dists)
        # 狠狠地加大奖励
        reward_treasure_dist += 1.8 if min_treasure_dist < min_prev_treasure_dist else 0

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    if prev_treasure_dists.count(1.0) < treasure_dists.count(1.0):
        # 步数两千的情况下, 宝箱的初始价值实际上是远高于步数的
        reward_treasure += 20
    
    # 靠近宝箱和获得宝箱都要做衰减哦
    reward_treasure_dist *= decay_factor
    reward_treasure  *= decay_factor

    # 加个值保个底( 拿最后一个宝箱, 虽然这一版在没拿满hh )
    reward_treasure_dist += 0.25
    reward_treasure += 2
    # 拿满了就去掉奖励, 虽然好像本来就会去掉, 保险一点
    if num_collected_treasures == len(treasure_dists):
        reward_treasure_dist = 0

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = 0

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励
    reward_buff = 0

    # 感觉 buff 不用第二次, 懒得写了, 早晚路过
    # 要写就要分段, 拿满宝箱再拿 buff 收益肯定没有之前拿高 
    # 但是早晚路过, 感觉没必要

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # Reward 4.1 Penalty for flickering into the wall (TODO)
    # 奖励4.1 撞墙闪现的惩罚 (TODO)

    # Reward 4.2 Reward for normal flickering (TODO)
    # 奖励4.2 正常闪现的奖励 (TODO)

    # Reward 4.3 Reward for super flickering (TODO)
    # 奖励4.3 超级闪现的奖励 (TODO)

    # 只用一次, 懒得写, 不过改改或许有惊喜呢hh

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    # 步数惩罚(最后合并的时候有负因子哦), 递增
    reward_step = 1.2 * (1 - decay_factor) + 0.1


    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # (TODO: Give penalty after collecting all the treasure chests, encourage full collection)
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0


    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0

    # Determine whether it bumps into the wall
    # 判断是否撞墙
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    # 这个大一点, 主要防止闪现撞墙
    if is_bump:
        reward_bump = 10

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    REWARD_CONFIG = {
        "reward_end_dist": "1",
        "reward_win": "1",
        "reward_buff_dist": "1",
        "reward_buff": "1",
        "reward_treasure_dists": "1",
        "reward_treasure": "1",
        "reward_flicker": "1",
        "reward_step": "-1",
        "reward_bump": "-1",
        "reward_memory": "-1",
    }

    reward = [
        reward_end_dist * float(REWARD_CONFIG["reward_end_dist"]),
        reward_win * float(REWARD_CONFIG["reward_win"]),
        reward_buff_dist * float(REWARD_CONFIG["reward_buff_dist"]),
        reward_buff * float(REWARD_CONFIG["reward_buff"]),
        reward_treasure_dist * float(REWARD_CONFIG["reward_treasure_dists"]),
        reward_treasure * float(REWARD_CONFIG["reward_treasure"]),
        reward_flicker * float(REWARD_CONFIG["reward_flicker"]),
        reward_step * float(REWARD_CONFIG["reward_step"]),
        reward_bump * float(REWARD_CONFIG["reward_bump"]),
        reward_memory * float(REWARD_CONFIG["reward_memory"]),
    ]

    return sum(reward), is_bump