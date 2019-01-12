import numpy as np
#####################  hyper parameters  ####################
TARGET_INITPOS = 25
TARGET_SPEED = 10
EGO_SPEED = 0
TASK_LENGTH = 6
TASK_NUM = 5
UPDATE_TIME = 0.1
ROAD_LENGTH = 50
ROAD_WIDTH = 7.2
VEHICLE_WIDTH = 2
VEHICLE_LENGTH = 4
ZEBRALINE_WIDTH = 4
MAX_TIME = 50

class Target_Ego(object):
    def __init__(self, task_num=TASK_NUM,):
        self.target_speed = TARGET_SPEED  # 目标车辆的动作/速度
        self.t = UPDATE_TIME  # 车辆的更新时间
        self.ego_speed = EGO_SPEED  # 本车速度
        self.task_length = TASK_LENGTH  # 一个任务的长度
        self.task_num = task_num  # 任务数量
        self.target_pos = TARGET_INITPOS + self.task_num * self.task_length  # 目标车辆的初始位置
        self.road_width = ROAD_WIDTH  # 两车道的宽度
        self.road_length = ROAD_LENGTH  # 下边界到隔离带的车道长度
        self.vehicle_width = VEHICLE_WIDTH  # 车辆的宽度
        self.vehicle_length = VEHICLE_LENGTH  # 车辆的长度
        self.zebraline_width = ZEBRALINE_WIDTH  # 斑马线宽度
        self.edlb = self.road_length - self.zebraline_width  # 本车到下边界的距离
        self.edhb = self.road_length*2 + self.road_width - self.edlb  # 本车到上边界的距离
        self.edsl = 0
        self.CountTime = 0
        self.IsCrash = False
        self.done = False

    # Target车辆状态/位置更新
    def T_choose_state(self,):
        self.s = self.target_pos + self.target_speed * self.t
        self.target_pos = self.s
        return self.s

    # 本车状态（Ev,Tv,ttc,Edsl,Edlb,Edhb)
    def E_choose_state(self, a,):
        self.ego_speed = self.ego_speed + a*self.t
        if self.ego_speed < 0:
            self.ego_speed = 0
            a = 0
        self.target_speed = self.target_speed
        self.target_s = Target_Ego().T_choose_state()
        self.ttc = (self.road_length + 0.75*self.road_width - self.target_s)/self.target_speed
        self.edsl = self.edsl + self.ego_speed*self.t + 0.5*a*self.t**2

        self.edlb = self.edlb + (self.ego_speed*self.t + 0.5*a*self.t**2)
        self.edhb = self.edhb - (self.ego_speed*self.t + 0.5*a*self.t**2)
        self.s_ = [self.ego_speed, self.target_speed, self.ttc, self.edsl, self.edlb, self.edhb]
        self.CountTime += 0.1
        return self.s_

    def reset(self, task_num):
        self.task_num = task_num
        self.ego_speed = EGO_SPEED
        self.target_speed = TARGET_SPEED
        self.target_pos = TARGET_INITPOS + self.task_num * self.task_length  # 目标车辆的初始位置
        self.ttc = (self.road_length + 0.75*self.road_width - self.target_pos)/self.target_speed
        self.edsl = 0
        self.edlb = self.road_length - self.zebraline_width  # 本车到下边界的距离
        self.edhb = self.road_length * 2 + self.road_width - self.edlb  # 本车到上边界的距离
        self.CountTime = 0
        self.s = [self.ego_speed, self.target_speed, self.ttc, self.edsl, self.edlb, self.edhb]
        self.IsCrash = False
        self.done = False
        return self.s

    def get_env_feedback(self, a, c1=1, c2=1, c3=0.1, c4=-2, c5=-2, c6=2, T_react= 2, ):
        self.c1 = c1 # constant
        self.c2 = c2 # constant
        self.c3 = c3 # constant
        self.c4 = c4 # constant reward
        self.T_react = T_react
        self.c5 = c5
        self.c6 = c6

        # reward1
        D_des_ego = self.road_width + self.zebraline_width + self.vehicle_length - self.edsl
        D_des_init = self.road_width + self.zebraline_width + self.vehicle_length
        R = self.c1*(D_des_ego**2/D_des_init**2) - self.c2
        '''
        # reward2
        self.Dsafe = self.TA * self.T_react
        D_TE = (self.Ev*self.Ti + 0.5*a*self.Ti**2)**2
        if self.Dsafe >= D_TE:
            R2 = self.c3*(self.Dsafe - D_TE)
        else:
            R2 = -(R1 + 0)
        # reward3 /crash
        R = R1 +R2
        '''
        if self.road_length + 0.75*self.road_width - 0.5*self.vehicle_width <= self.T_choose_state()\
                <= self.road_length + 0.75 * self.road_width + 0.5 * self.vehicle_width + self.vehicle_length and \
                self.road_length + 0.25*self.road_width - 0.5*self.vehicle_width <= self.edlb <= self.road_length + \
                0.25*self.road_width + 0.5*self.vehicle_width + self.vehicle_length:
            R3 = self.c4
            R = R + R3
            self.IsCrash = True
        if D_des_ego > 0 and self.CountTime >= MAX_TIME:
            R4 = self.c5
            R = R +R4
            self.IsCrash = True
        if D_des_ego <= 0:
            R5 = self.c6
            R = R + R5
            self.done = True
        return R

    def sample_a(self, size, lb, ub):
        return np.random.random(size) * (ub - lb) + lb

