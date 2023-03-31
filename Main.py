# Aidan Morgan
# atmorga3@asu.edu
# 2-9-2023
# Reworked and expanded on from Jonathan Vogel's Lap Sim in matlab
# FSAE Lap Simulation

# IMPORTS
import pandas as pd
import random
import numpy as np
import scipy.io
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import math
from csaps import csaps
import statistics as stat

################################################
# CONSTANTS #
# TYRE CONSTANTS
TYRE_SCALING_FACTOR_X = .6
TYRE_SCALING_FACTOR_Y = .47
TYRE_RADIUS = 9.05 / 12 / 3.28
TIRE_DATA_FILE_PATH = "Config\Hoosier_R25B_18.0x7.5-10_FX_12psi.mat"
TIRE_DATA = scipy.io.loadmat(TIRE_DATA_FILE_PATH)['full_send_x']['coefs']
MAGIC_FORMULA_A_CONSTANTS_FILE_PATH = "Config\A1654run21_MF52_Fy_12.mat"
MAGIC_FORMULA_A_CONSTANTS = scipy.io.loadmat(MAGIC_FORMULA_A_CONSTANTS_FILE_PATH)['A'][0]


# ENGINE CONSTANTS
ENGINE_RPM_LOWER_LIMIT = 6200
ENGINE_RPM_STEP_SIZE = 100
ENGINE_RPM_UPPER_LIMIT = 14100
ENGINE_RPM = np.arange(ENGINE_RPM_LOWER_LIMIT, ENGINE_RPM_UPPER_LIMIT, ENGINE_RPM_STEP_SIZE, dtype=int)
ENGINE_TORQUE_CSV_FILE_NAME = "Config\Engine_Torque.csv"    # (Nm)
ENGINE_TORQUE = pd.read_csv(ENGINE_TORQUE_CSV_FILE_NAME, sep=',', header=None).to_numpy()[0]
OPTIMAL_SHIFT_RPM = 14000
SHIFT_TIME = .25    # (seconds)


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
FRONT_ACKERMANN = 0     # not sure if this is correct
REAR_ACKERMANN = 0


# AERO CONSTANTS
LIFT_COEFFICIENT = .0418
DRAG_COEFFICIENT = .0184
FRONT_DOWNFORCE_DIST = .48  # percentage
REAR_DOWNFORCE_DIST = (1 - FRONT_DOWNFORCE_DIST) # percentage
# TODO: Add drs based on turn radius? Swap to a different aero coefficients is a DRS constant is 1?
#################################################

# updates the accel given 
def calcAccel(current_speed, vehicle_side_slip_angle, radius, front_axle_load, rear_axle_load, steer_angle, initial_camber_gain_front, initial_camber_gain_rear, grip_scale): 
    accel_y = current_speed**2 / radius
        
    # calc lateral load transfer
    new_weight = accel_y * CENTER_OF_GRAVITY * WEIGHT / stat.mean([FRONT_TRACK_WIDTH, REAR_TRACK_WIDTH])/ 32.2 / 12
    
    # split weight to front and rear using LAT_LOAD_TRANSFER_DIST
    new_weight_front = new_weight * LAT_LOAD_TRANSFER_DIST
    new_weight_rear = new_weight * (1 - LAT_LOAD_TRANSFER_DIST)
    
    # calculate front / rear roll (rad)
    phi_front = accel_y * FRONT_ROLL_GRAD * math.pi / 180 / 32.2
    phi_rear = accel_y * REAR_ROLL_GRAD * math.pi / 180 / 32.2
    
    # update individual wheel loads
    wheel_inside_front = front_axle_load - new_weight_front
    wheel_outside_front = front_axle_load + new_weight_front
    wheel_inside_rear = rear_axle_load - new_weight_rear
    wheel_outside_rear = rear_axle_load + new_weight_rear
    
    # update individual wheel camber from roll and then from steering effects
    
    camber_inside_front, camber_outside_front, camber_inside_rear, camber_outside_rear = updateCamber(phi_front, phi_rear, initial_camber_gain_front, initial_camber_gain_rear, steer_angle, REAR_ACKERMANN)
    
    yaw_rate = accel_y / current_speed
    
    # calulate slip angles from side slip and steer
    
    slip_angle_front = vehicle_side_slip_angle + FRONT_AXLE_TO_CG * yaw_rate / current_speed - steer_angle
    slip_angle_rear = vehicle_side_slip_angle - REAR_AXLE_TO_CG * yaw_rate / current_speed - REAR_ACKERMANN
    
    # calculate lateral force at front accounting for slip angle, load, and camber by plugging it in to da magic formula
    front_force_inside = -getLateralForce(MAGIC_FORMULA_A_CONSTANTS, -math.degrees(slip_angle_front), wheel_inside_front, -math.degrees(camber_inside_front)) * TYRE_SCALING_FACTOR_Y * math.cos(steer_angle)
    front_force_outside = getLateralForce(MAGIC_FORMULA_A_CONSTANTS, math.degrees(slip_angle_front), wheel_outside_front, -math.degrees(camber_outside_front)) * TYRE_SCALING_FACTOR_Y * math.cos(steer_angle)
    
    # for the rear tire forces we must see what forces the the dif has to overcome
    # drag on the front tires from aero and front tires
    force_x = DRAG_COEFFICIENT  * current_speed ** 2 + (front_force_inside + front_force_outside) * math.sin(steer_angle) / math.cos(steer_angle)
    
    # calculate the grip penalty becuase the rear tyres must over come that
    grip_scale = 1 - (force_x / WEIGHT / grip_scale(current_speed))**2
    
    # now calculate rear tire forces with the penalty
    rear_force_inside = -getLateralForce(MAGIC_FORMULA_A_CONSTANTS, -math.degrees(slip_angle_rear), wheel_inside_rear, - math.degrees(camber_inside_rear)) * TYRE_SCALING_FACTOR_Y * grip_scale
    rear_force_outside = getLateralForce(MAGIC_FORMULA_A_CONSTANTS, math.degrees(slip_angle_rear), wheel_outside_rear, - math.degrees(camber_outside_rear)) * TYRE_SCALING_FACTOR_Y * grip_scale
    
    # now we calculate the sum of the forces and moments
    sum_f_y = front_force_inside + front_force_outside + rear_force_inside + rear_force_outside
    moment_z_diff = force_x * DIF_LOCK * REAR_TRACK_WIDTH / 2   #differential contribution
    moment_z = (front_force_inside + front_force_outside) * FRONT_AXLE_TO_CG - (rear_force_inside + rear_force_outside) * REAR_AXLE_TO_CG - moment_z_diff
    
    #calculate resultant lateral acceleration
    actual_lateral_accel = sum_f_y / WEIGHT / 32.2
    accel_difference = accel_y - actual_lateral_accel
    return accel_difference

# uses the magic formula to find lateral force on the tyres
def getLateralForce(magic_constants, slip_angle, wheel_load, camber_angle):
    A = magic_constants
    Alpha = math.radians(slip_angle)    #converts slip angle to radians
    Fz = abs(wheel_load)
    Gamma = math.radians(camber_angle)  #converts camber angle to radians

    Gammay = Gamma
    Fz0PR = -150
    DFz = (Fz - Fz0PR) / Fz0PR
    
    SHy = (A[11] + A[12] * DFz) * 1 + A[13] * Gammay
    Alphay = Alpha + SHy
    Cy = A[0]
    MUy = (A[1] + A[2] * DFz) * (1 - A[3] * Gammay**2) * 1
    Dy = MUy * Fz
    Ky = A[8] * -150 * math.sin(2 * math.atan(Fz / (A[9] * -150 ))) * (1 - A[10] * abs(Gammay)) 
    By = Ky / (Cy * Dy)
    
    Ey = (A[4] + A[5] * DFz) * (1-(A[6] + A[7] * Gammay) * np.sign(Alphay))
    
    SVy = Fz * ((A[14] + A[15] * DFz) * 1+(A[16] + A[17] * DFz) * Gammay)
    Fy0 = Dy * math.sin(Cy * math.atan(By * Alphay - Ey * (By * Alphay - math.atan(By * Alphay)))) + SVy
    return Fy0

# updates camber from roll and then from steering
def updateCamber(phi_front, phi_rear, initial_camber_gain_front, initial_camber_gain_rear, ackermannf, ackermannr):
    camber_fin = -FRONT_TRACK_WIDTH * math.sin(phi_front) * 12 / 2 * FRONT_CAMBER_GAIN - initial_camber_gain_front - FRONT_KINGPIN_INCL_ANGLE * (1 - math.cos(ackermannf)) - FRONT_CASTER_ANGLE * math.sin(ackermannf) + phi_front
    camber_fout = -FRONT_TRACK_WIDTH * math.sin(phi_front) * 12 / 2 * FRONT_CAMBER_GAIN + initial_camber_gain_front + FRONT_KINGPIN_INCL_ANGLE * (1 - math.cos(ackermannf)) - FRONT_CASTER_ANGLE * math.sin(ackermannf) + phi_front
    camber_rin = REAR_TRACK_WIDTH * math.sin(phi_rear) * 12 / 2 * FRONT_CAMBER_GAIN - initial_camber_gain_rear - REAR_KINGPIN_INCL_ANGLE * (1 - math.cos(ackermannr)) - FRONT_CASTER_ANGLE * math.sin(ackermannr) + phi_rear
    camber_rout = REAR_TRACK_WIDTH * math.sin(phi_rear) * 12 / 2 * FRONT_CAMBER_GAIN + initial_camber_gain_rear + REAR_KINGPIN_INCL_ANGLE * (1 - math.cos(ackermannr)) - FRONT_CASTER_ANGLE * math.sin(ackermannr) + phi_rear
    return camber_fin, camber_fout, camber_rin, camber_rout
    
    
# PowerTrainSimulation
# returns 
def calcPowerTrain(initial_velocity, gear_ratios):
    guessRPM = ENGINE_RPM_UPPER_LIMIT
    current_gear = 0
    gearTot = 0

    while guessRPM > OPTIMAL_SHIFT_RPM :
        current_gear += 1
        gearTot = gear_ratios[current_gear] * FINAL_DRIVE * PRIMARY_REDUCTION
        guessRPM = initial_velocity * gearTot / TYRE_RADIUS * 60 / (2 * math.pi)

    # NOTE: There has to be a better way to do this
    index = 2
    while guessRPM > ENGINE_RPM[index] :
        index += 1
    torque = ENGINE_TORQUE[index-1]+(ENGINE_TORQUE[index]-ENGINE_TORQUE[index-1])/(ENGINE_RPM[index]-ENGINE_RPM[index-1])*(guessRPM-ENGINE_RPM[index-1])

    torque = torque * gearTot * DRIVETRAIN_LOSS_CONSTANT #torque output at the wheels

    contact_force = torque / TYRE_RADIUS    #force on contact patch from drivtrain (N)

    return(contact_force, current_gear)

# replace this with fitment for a specific tyre curve
def fitTyreData(a, arr):
    return random.random() * arr[0] + random.random()

# The approach to cacluating GGV is as follows
# 1. calculate lateral Acceleration
# 2. calculate cornering envelope
# 3. calculate braking performance
def generateGGV(velocity_range, radii, gear_ratios):
    # calculating the acceleration capacity
    # calculates the max acceleration at every velocity
    current_gear = 1
    accel_array = []
    another_accel_array = []
    front_forces = []
    front_accel = []
    gears = []
    shift_count = 1
    shift_velocities = [0]
    for velocity in velocity_range :
        gear_prev = current_gear
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
        contact_force, current_gear = calcPowerTrain(max(7.5, velocity/3.28), gear_ratios)

        front_tractive_force_cap = contact_force * .2248    # why .2248
        front_tractive_force_cap -= DRAG_COEFFICIENT * velocity **2
        front_forces.append(front_tractive_force_cap / WEIGHT)
        front_accel.append(min(front_tractive_force_cap/WEIGHT, total_lateral_accel))
        contact_force, current_gear = calcPowerTrain(velocity/3.28, gear_ratios)
        gears.append(current_gear)
        if current_gear > gear_prev :
            shift_count += 1
            shift_velocities.append(velocity)
        another_accel_array.append(front_accel[-1])
    
    # if the acceleration is less than 0 make it 0
    another_accel_array = [a if a > 0 else  0 for a in another_accel_array]

    accel_spline = csaps(velocity_range, another_accel_array)
    grip_spline = csaps(velocity_range, accel_array)
    print(grip_spline, accel_spline)


    # calculate cornering envelope by guessing ay and then
    # increasing it incrementally
    # we evaluate cornering performance based on the radius of a turn instead of speed

    lat_accel_cap = 1
    
    for radius in radii :
        current_speed = math.sqrt(radius * 32.2 * lat_accel_cap)
        downforce = LIFT_COEFFICIENT * (current_speed ** 2)
        
        # account for downforce in suspension travel aka heave (in)
        sus_drop_front = downforce * FRONT_DOWNFORCE_DIST / 2 / FRONT_RIDE_RATE
        sus_drop_rear = downforce * REAR_DOWNFORCE_DIST / 2 / REAR_RIDE_RATE
        
        # from suspension heave, update static camber (rad)
        initial_camber_gain_front = FRONT_STATIC_CAMBER_ANGLE - sus_drop_front * FRONT_CAMBER_GAIN
        initial_camber_gain_rear = REAR_STATIC_CAMBER_ANGLE - sus_drop_rear * REAR_CAMBER_GAIN
        
        # update load on each axle (lbs)
        front_axle_load = (FRONT_WEIGHT + downforce * FRONT_DOWNFORCE_DIST) / 2
        rear_axle_load = (REAR_WEIGHT + downforce * REAR_DOWNFORCE_DIST) / 2

        # guess ACKERMANN (for now we guess the ackermann)
        steer_angle = WHEELBASE / radius
        delta_ackermann = steer_angle * .01   # .01 is a change constanst it seems
        
        # assume vehicle sideslip starts at 0 (rad)
        vehicle_side_slip_angle = 0
        
        accel_difference = calcAccel(current_speed, vehicle_side_slip_angle, radius, front_axle_load, rear_axle_load, steer_angle, initial_camber_gain_front, initial_camber_gain_rear, grip_spline)
        
        # change sideslip angle until the initial guess and resultant are the same
        # NOTE: This can probably be made faster by changing how much we vary beta based on the magnitude of accel_difference
        while abs(accel_difference) > 0.5:
            slip_angle_increment = .0025
            if accel_difference > 0 :
                vehicle_side_slip_angle += slip_angle_increment     # increase in slip angle
            else :
                vehicle_side_slip_angle -= slip_angle_increment     # decrease the slip angle
            
            accel_difference = calcAccel(current_speed, vehicle_side_slip_angle, radius, front_axle_load, rear_axle_load, steer_angle, initial_camber_gain_front, initial_camber_gain_rear, grip_spline)
            print(accel_difference)
            
            
         
        
    
        
            
            
        
def main():
    # drivetrain setup
    gear_ratios = pd.read_csv(GEAR_RATIOS_CSV_FILE_NAME, sep=',', header=None).to_numpy()[0]
    gear_total = gear_ratios[-1] * FINAL_DRIVE * PRIMARY_REDUCTION
    Vmax = math.floor(3.28 * OPTIMAL_SHIFT_RPM / (gear_total / TYRE_RADIUS * 60 / (2 * math.pi)))   # ft/s I think
    
    # generating GGV Diagram
    velocity_range = np.arange(15, Vmax, 5)
    radii = np.arange(15, 155, 10)  # might be wrong
    
    # FOR TESTING
    # print(getLateralForce(MAGIC_FORMULA_A_CONSTANTS, 9.6289, 161.5368, -0.0067))
    # print(getLateralForce(MAGIC_FORMULA_A_CONSTANTS, 5.7773, 164.7671, -0.0112))
    # print(calcPowerTrain(20, gear_ratios))
    # print(MAGIC_FORMULA_A_CONSTANTS)

    # calculate the acceleration envelope
    generateGGV(velocity_range, radii, gear_ratios)
    print("done")
    


if __name__ == '__main__' :
    main()