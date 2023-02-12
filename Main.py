# Aidan Morgan
# 2-9-2023
# Created from Jonathan Vogel's Lap Sim in matlab
# FSAE Lap Simulation

import pandas as pd
import random
import numpy as np
import scipy.io
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import math
from csaps import csaps

################################################
# CONSTANTS #
# TYRE CONSTANTS
TYRE_SCALING_FACTOR_X = .6
TYRE_SCALING_FACTOR_Y = .47
TYRE_RADIUS = 9.05 / 12 / 3.28
TIRE_DATA_FILE_PATH = "Config\Hoosier_R25B_18.0x7.5-10_FX_12psi.mat"
TIRE_DATA = scipy.io.loadmat(TIRE_DATA_FILE_PATH)['full_send_x']['coefs']


# ENGINE CONSTANTS
ENGINE_RPM_LOWER_LIMIT = 6200
ENGINE_RPM_STEP_SIZE = 100
ENGINE_RPM_UPPER_LIMIT = 14100
ENGINE_TORQUE_CSV_FILE_NAME = "Config\Engine_Torque.csv"    # (Nm)
OPTIMAL_SHIFT_RPM = 14000
SHIFT_TIME = .25    # (seconds)
engine_RPM = []
engine_torque = []


# DRIVETRAIN CONSTANTS
PRIMARY_REDUCTION = 76/36
GEAR_RATIOS_CSV_FILE_NAME = "Config\Gear_Ratios.csv"
FINAL_DRIVE = 40/12     # large sproket / small sprocket
DRIVETRAIN_LOSS_CONSTANT = .85  # percentage of torque that makes it to the rear wheels
DIF_LOCK = .90       # percentage: 0 = open, 1 = locked


# VEHICLE CHARACTERISTICS
LAT_LOAD_TRANSFER_DIST = .515   # percentage: lateral load transfer distribution 0 - 1
WEIGHT = 660    # vehicle + driver weight (lbs)
MASS = WEIGHT / 32.2    # 32.2 to convert weight to lbm
FRONT_WEIGHT_DIST = .50     # percentage: front weight distribution 
CENTER_OF_GRAVITY = 13.2/12     # center of gravity height (ft)
WHEELBASE = 60.5/12     # (ft)
FRONT_TRACK_WIDTH = 46/12   # (ft)
REAR_TRACK_WIDTH = 44/12    # (ft)
FRONT_WEIGHT = WEIGHT * FRONT_WEIGHT_DIST
REAR_WEIGHT = WEIGHT * (1 - FRONT_WEIGHT_DIST)
FRONT_AXLE_TO_CG = 1 * (1 - FRONT_WEIGHT_DIST)
REAR_AXLE_TO_CG = 1 * FRONT_WEIGHT_DIST


# SUSPENSION CONSTANTS   Note: can leave 0
FRONT_ROLL_GRAD = 0     # front roll gradient (deg/g)
REAR_ROLL_GRAD = 0      # rear roll gradient (deg/g)
PITCH_GRAD = 0          # pitch gradient (deg/g)
FRONT_RIDE_RATE = 180   # (lbs/in)
REAR_RIDE_RATE = 180   # (lbs/in)
FRONT_STATIC_CAMBER_ANGLE = math.radians(0)
REAR_STATIC_CAMBER_ANGLE = math.radians(0)
FRONT_CAMBER_COMPENSATION = .1      # percentage
REAR_CAMBER_COMPENSATION = .2       # percentage
FRONT_CASTER_ANGLE = math.radians(0)
FRONT_KINGPIN_INCL_ANGLE = math.radians(0)
REAR_CASTER_ANGLE = math.radians(4.1568)
REAR_KINGPIN_INCL_ANGLE = math.radians(0)
FRONT_INDUCED_ROLL = math.asin(2 / FRONT_TRACK_WIDTH / 12)  # change name
REAR_INDUCED_ROLL = math.asin(2 / REAR_TRACK_WIDTH / 12)   # change name
FRONT_CAMBER_GAIN = FRONT_INDUCED_ROLL * FRONT_CAMBER_COMPENSATION
REAR_CAMBER_GAIN = REAR_INDUCED_ROLL * REAR_CAMBER_COMPENSATION


# AERO CONSTANTS
LIFT_COEFFICIENT = .0418
DRAG_COEFFICIENT = .0184
FRONT_DOWNFORCE_DIST = .48  # percentage
REAR_DOWNFORCE_DIST = (1 - FRONT_DOWNFORCE_DIST) # percentage

#################################################

# PowerTrainSimulation
def calcPowerTrain(initial_velocity):
    


# replace this with fitment for a specific tyre curve
def fitTyreData(a, arr):
    return random.random() * arr[0] + random.random()

# The approach to cacluating GGV is as follows
# 1. calculate lateral Acceleration
# 2. calculate braking performance


def generateGGV(velocity_range, radii):
    current_gear = 1
    accel_array = []
    for velocity in velocity_range :
        gear = current_gear
        downforce = LIFT_COEFFICIENT * (velocity ** 2) # (lbs)
        
        # calculate suspension drop from downforce (inches)
        sus_drop_front = downforce * FRONT_DOWNFORCE_DIST / 2 / FRONT_RIDE_RATE
        sus_drop_rear = downforce * REAR_DOWNFORCE_DIST / 2 / REAR_RIDE_RATE
        
        # calculate camber gain from ride hight drop (inches)
        initial_camber_gain_front = FRONT_STATIC_CAMBER_ANGLE - sus_drop_front * FRONT_CAMBER_GAIN
        initial_camber_gain_rear = REAR_STATIC_CAMBER_ANGLE - sus_drop_rear * REAR_CAMBER_GAIN
        
        # find load on each tyre (lbs)
        default_front_tyre_load = (FRONT_WEIGHT + downforce * FRONT_DOWNFORCE_DIST) / 2
        default_rear_tyre_load = (REAR_WEIGHT + downforce * REAR_DOWNFORCE_DIST) / 2
        
        
        # Start guessing acceleration, starting with 0
        accel = 0
        pitch_angle = -accel * PITCH_GRAD * math.pi / 180   # (rad)
        
        # recalculate wheel loads due to load transfer (lbs)
        front_tyre_load = default_front_tyre_load - accel * CENTER_OF_GRAVITY * MASS / 2 / WHEELBASE
        rear_tyre_load = default_rear_tyre_load - accel * CENTER_OF_GRAVITY * MASS / 2 / WHEELBASE
        
        # recalculate camber angles due to pitch
        camber_gain_front = -WHEELBASE * 12 * math.sin(pitch_angle) / 2 * FRONT_CAMBER_GAIN + initial_camber_gain_front
        camber_gain_rear = -WHEELBASE * 12 * math.sin(pitch_angle) / 2 * REAR_CAMBER_GAIN + initial_camber_gain_rear
        
        # range of slip ratios
        slip_ratio_range = np.arange(0, .11, .01)
        
        # evaluate tractive force capactiy from each tire for each slip ratio
        front_tractive_force_cap = []
        rear_tractive_force_cap = []
        for slip_ratio in slip_ratio_range:
            # print([slip_ratio, - front_tyre_load, math.radians(- camber_gain_front)])
            front_tractive_force_cap.append(fitTyreData('HoosierX', [slip_ratio, - front_tyre_load, math.radians(- camber_gain_front)]))    # fix
            rear_tractive_force_cap.append(fitTyreData('HoosierX', [slip_ratio, - rear_tyre_load, math.radians(- camber_gain_rear)]))       #fix
        
        # get the max force capacity from each tire
        # it seems like he ommits values larger than 1000
        front_tractive_force_cap = [cap for cap in front_tractive_force_cap if abs(cap) <= 1000]
        rear_tractive_force_cap = [cap for cap in rear_tractive_force_cap if abs(cap) <= 1000]
        
        front_max_force_cap = max(front_tractive_force_cap)     # NOTE: might need to make the whole array positive
        rear_max_force_cap = max(rear_tractive_force_cap)       # another note: we probably done even need front_max_force_cap
        
        total_tyre_tractive_force = abs(2 * rear_max_force_cap)
        
        total_lateral_accel = total_tyre_tractive_force / WEIGHT
        accel_delta = total_lateral_accel - accel
        
        while accel_delta > 0 :
            accel = accel + .01     # increase acceleration by a step value of .01
            pitch_angle = -accel * PITCH_GRAD * math.pi / 180   # (rad)
            
            # recalculate wheel loads due to load transfer (lbs) * we also divide by 24 here for some reason
            front_tyre_load = default_front_tyre_load - accel * CENTER_OF_GRAVITY * MASS / 2 / WHEELBASE / 24
            rear_tyre_load = default_rear_tyre_load - accel * CENTER_OF_GRAVITY * MASS / 2 / WHEELBASE / 24
            
            camber_gain_front = - WHEELBASE * 12 * math.sin(pitch_angle) / 2 * FRONT_CAMBER_GAIN + initial_camber_gain_front
            camber_gain_rear = WHEELBASE * 12 * math.sin(pitch_angle) / 2 * REAR_CAMBER_GAIN + initial_camber_gain_rear
            
            # FZ_vals = np.arange(-250, -50, 1)
            slip_ratio_range = np.arange(0, .11, .01)
        
            # evaluate tractive force capactiy from each tire for each slip ratio
            front_tractive_force_cap = []
            rear_tractive_force_cap = []
            
            for slip_ratio in slip_ratio_range:
                # print([slip_ratio, - front_tyre_load, math.radians(- camber_gain_front)])
                front_tractive_force_cap.append(fitTyreData('HoosierX', [slip_ratio, - front_tyre_load, math.radians(- camber_gain_front)]))    # fix
                rear_tractive_force_cap.append(fitTyreData('HoosierX', [slip_ratio, - rear_tyre_load, math.radians(- camber_gain_rear)]))       #fix
            
            # get the max force capacity from each tire
            # it seems like he ommits values larger than 1000
            front_tractive_force_cap = [cap for cap in front_tractive_force_cap if abs(cap) <= 1000]
            rear_tractive_force_cap = [cap for cap in rear_tractive_force_cap if abs(cap) <= 1000]
            front_max_force_cap = max(front_tractive_force_cap)     # NOTE: might need to make the whole array positive
            rear_max_force_cap = max(rear_tractive_force_cap)       # another note: we probably done even need front_max_force_cap
            total_tyre_tractive_force = abs(2 * rear_max_force_cap)
        
            total_lateral_accel = total_tyre_tractive_force / WEIGHT
            accel_delta = total_lateral_accel - accel
            
        accel_array.append(total_lateral_accel)
        
        
            
            
        
def main():

    # engine setup 
    engine_RPM = np.arange(ENGINE_RPM_LOWER_LIMIT, ENGINE_RPM_UPPER_LIMIT, ENGINE_RPM_STEP_SIZE, dtype=int)
    engine_torque = pd.read_csv(ENGINE_TORQUE_CSV_FILE_NAME, sep=',', header=None).to_numpy()[0]
    
    # drivetrain setup
    gear_ratios = pd.read_csv(GEAR_RATIOS_CSV_FILE_NAME, sep=',', header=None).to_numpy()[0]
    gear_total = gear_ratios[-1] * FINAL_DRIVE * PRIMARY_REDUCTION
    Vmax = math.floor(3.28 * OPTIMAL_SHIFT_RPM / (gear_total / TYRE_RADIUS * 60 / (2 * math.pi)))   # ft/s I think
    
    # generating GGV Diagram
    velocity_range = np.arange(15, Vmax, 5)
    radii = np.arange(15, 155, 10)  # might be wrong
    
    # calculate the acceleration envelope
    generateGGV(velocity_range, radii)
    
    
    
    
    print(FRONT_CAMBER_GAIN)


if __name__ == '__main__' :
    main()