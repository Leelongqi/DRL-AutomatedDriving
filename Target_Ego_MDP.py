import numpy as np
#####################  hyper parameters  ####################
Target_InitPos = 0
Target_Speed = 10
Ego_Speed = 8
Task_Length = 6
TASK_NUM = 5
Update_Time = 0.1
E_Dlb = 20
Road_Way = 50
Road_Width = 7.2
Road_Length = 100
Vehicle_Width = 2
Vehicle_Length = 4
MAX_TIME = 50

class Target_Ego(object):
    def __init__(self, T_pos = Target_InitPos, T_v =Target_Speed, t=Update_Time, E_v = Ego_Speed, Task_l = Task_Length,
                 E_dlb = E_Dlb, Road_way = Road_Way, Road_w=Road_Width, Road_l = Road_Length,
                 V_w = Vehicle_Width, V_l = Vehicle_Length,TaskNum = TASK_NUM,):
        self.TA = T_v # Tagert车辆的动作/速度
        self.Ti = t # 车辆的更新时间
        self.Ev = E_v # 本车速度
        self.Taskl = Task_l
        self.TS = T_pos + TaskNum * self.Taskl  # Target 车辆的位置
        self.Edlb = E_dlb # 本车到低边界的距离
        self.Roadway = Road_way # 停止线到低边界的距离
        self.Roadwidth = Road_w # 两车道的宽度
        self.Roadlength = Road_l # 一条道路总长度
        self.Vehiclewidth = V_w # 车辆的宽度
        self.Vehiclelength = V_l # 车辆的长度
        self.TaskNum = TaskNum
        self.Edhb = self.Roadlength - self.Edlb
        self.Edsl = self.Roadway-self.Edlb
        self.CountTime = 0
        self.IsCrash = False
        self.done = False

    # Target车辆状态/位置更新
    def T_choose_state(self,):
        self.s_ = self.TS + self.Ti* self.TA
        self.TS = self.s_
        return self.s_

    # 本车状态（Ev,Tv,ttc,Edsl, Edlb, Edhb)
    def E_choose_state(self, a,):
        self.Ev = self.Ev + a*self.Ti
        self.Tv = self.TA
        self.TS = Target_Ego().T_choose_state()
        self.TTC = (self.Roadway + 0.75*self.Roadwidth-0.5*self.Vehiclewidth - self.TS)/self.Tv
        self.Edsl = (self.Roadway-self.Edlb)-(self.Ev*self.Ti + 0.5*a*self.Ti**2)
        self.Edlb = self.Edlb + (self.Ev*self.Ti + 0.5*a*self.Ti**2)
        self.Edhb = self.Roadlength - self.Edlb - (self.Ev*self.Ti + 0.5*a*self.Ti**2)
        self.s_ = [self.Ev, self.Tv, self.TTC, self.Edsl, self.Edlb, self.Edhb]
        self.CountTime += 0.1
        if self.Edhb - self.Roadway + self.Vehiclelength/2 < 0:
            self.done = True
        return self.s_

    def reset(self,TaskNum):
        self.TaskNum = TaskNum
        self.Ev = Ego_Speed
        self.Tv = Target_Speed
        self.TTC = (self.Roadway + 0.75*self.Roadwidth-0.5*self.Vehiclewidth - self.TS)/self.Tv
        self.Edlb = E_Dlb
        self.Edsl = self.Roadway - self.Edlb
        self.Edhb = self.Roadlength - self.Edlb
        self.CountTime = 0
        self.IsCrash = False

        self.s = [self.Ev, self.Tv, self.TTC, self.Edsl, self.Edlb, self.Edhb]
        return self.s

    def get_env_feedback(self, a, c1=0.1, c2=0.1, c3=0.1, c4=-0.1, c5 = -0.2,  T_react= 2, ):
        self.c1 = c1 # constant
        self.c2 = c2 # constant
        self.c3 = c3 # constant
        self.c4 = c4 # constant reward
        self.T_react = T_react
        self.c5 = c5

        # reward1
        d1 = self.Edhb **2
        d2 = (self.Roadlength - self.Edlb)**2
        R1 = self.c1*(d1/d2) - self.c2
        # reward2
        self.Dsafe = self.TA * self.T_react
        D_TE = (self.Ev*self.Ti + 0.5*a*self.Ti**2)**2
        if self.Dsafe >= D_TE:
            R2 = self.c3*(self.Dsafe - D_TE)
        else:
            R2 = -(R1 + 0)
        # reward3 /crash
        R = R1 +R2
        if self.Edlb - 0.5*(self.Vehiclewidth+ self.Vehiclelength) <= self.T_choose_state() <= \
                self.Edlb + 0.5*(self.Vehiclewidth+ self.Vehiclelength)\
                and self.Roadway+0.75*self.Roadwidth- 0.5*(self.Vehiclelength+ self.Vehiclewidth) <= self.Edlb <=\
                self.Roadway+0.75*self.Roadwidth + 0.5*(self.Vehiclelength+ self.Vehiclewidth):
            R3 = self.c4
            R = R + R3
            self.IsCrash = True
        if self.Edhb - self.Roadway >0 and self.CountTime >= MAX_TIME:
            R4 = self.c5
            R = R +R4
            self.IsCrash = True
        return R

    def sample_a(self, size, lb, ub):
        return np.random.random(size) * (ub - lb) + lb

