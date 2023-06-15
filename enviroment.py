
import torch as tr
import numpy as np


device=tr.device('cuda')
def force(x,y,U0,type='mexican'):
    if type == 'mexican':
        r = tr.sqrt(x**2+y**2)
        bool = r>0.4
        fr = -64*U0*(r**2-0.25)
        fr[bool] = 0
        # f = tr.stack([fr*x,fr*y],dim=1).to(device)
        F_x = fr*x
        F_y = fr*y
        return F_x,F_y

def goal_check(state,target_area):
    x, y = state[:,0], state[:,1]
    target_bool_x_left = target_area[0][0] <= x
    target_bool_x_right = x <= target_area[0][1]
    target_bool_x = target_bool_x_left*target_bool_x_right
    target_bool_y_left = target_area[1][0] <= y
    target_bool_y_right = y <= target_area[1][1]
    target_bool_y = target_bool_y_left*target_bool_y_right
    target_bool = target_bool_x*target_bool_y
    return target_bool

# def vorticity(batch_size,device='cuda'):
#     omega_d = tr.zeros(batch_size,1).to(device)
#     return  omega_d

#v = e + F, r_new = r+v delta T
def update_state(state, action, dt, U0, batch_size, char_length, device = 'cuda'):

    x, y, F_x, F_y, theta = state[:,0], state[:,1], state[:,2], state[:,3], state[:,4]
    
    F_x, F_y = force(x,y,U0)
    # raise Exception(action.shape)
    theta = action #+ vorticity[:,0]*dt
    noise = tr.normal(tr.zeros(batch_size),tr.ones(batch_size)).to(device)
    theta = theta + np.sqrt(dt)*char_length*noise
    # theta = theta % (2*np.pi)
    wall = 0.75*tr.ones(x.shape,dtype=tr.float).to(device)
    e_x = tr.cos(theta)
    v_x = e_x + F_x
    x_new = x + v_x*dt
        
    wall_bool_x_left = -wall <= x_new
    wall_bool_x_right =  x_new <= wall
    wall_bool_x = wall_bool_x_left*wall_bool_x_right

    e_y = tr.sin(theta)
    v_y = e_y + F_y
    y_new = y + v_y*dt

    wall_bool_y_left = -wall <= y_new
    wall_bool_y_right =  y_new <= wall
    wall_bool_y = wall_bool_y_left*wall_bool_y_right

    wall_bool = wall_bool_x * wall_bool_y #Bool that has True entries if the move stays within the constraints
    
    # movement within constraints
    x_temp = x.clone()
    y_temp = y.clone()
    
    x_temp[wall_bool] = x_new[wall_bool]
    y_temp[wall_bool] = y_new[wall_bool]

    new_state = tr.stack([x_temp, y_temp, F_x, F_y, theta], dim = 1)
    return new_state, wall_bool

def reward(state, dt, target_bool, wall_bool, device = 'cuda'):
    x_shape= state[:,0].shape
    reward_t = -tr.ones(x_shape,dtype = tr.float).to(device)*dt/100 

    reward_t[target_bool] = 100 #batch_size #Large reward that scales with batch_size
    reward_t[wall_bool] = -10
    # reward_t = tr.where(target_bool,-10,reward_t)
    return reward_t