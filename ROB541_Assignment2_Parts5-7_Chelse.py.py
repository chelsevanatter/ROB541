import numdifftools as nd 
import math
import numpy as np
import matplotlib.pyplot as plt

class Group():
    
    def __init__(self, identity, inverse, representation, derepresentation): 
       self.identity = identity
       self.representation = representation
       self.derepresentation = derepresentation
       self.inverse_func = inverse

    def element(self,value):
        return GroupElement(value,self)
    
    def identity_element(self):
        return self.element(self.identity)

class GroupElement():
    def __init__(self, value, group):
        self.group = group
        self.value = value

    def left_action(self, element):
        self_rep = self.group.representation(self.value)
        element_rep = element.group.representation(element.value)
        value = self.group.derepresentation(self_rep @ element_rep)
        
        return GroupElement(value, self.group) 

    def left_lifted_action(self, h):
        matrix = self.group.representation(self.value) @ h.group.representation(h.value)
        value = self.group.derepresentation(matrix)
        return GroupElement(value, self.group)
    
    def action_helper(self, matrix):
        return np.array([[matrix[0]], [0], [matrix[1]],[1]])


    def right_lifted_action(self, h):
        matrix =  h.group.representation(h.value) @ self.group.representation(self.value)
        value = self.group.derepresentation(matrix)
        return GroupElement(value, self.group)

    def right_action(self, element):
        self_rep = self.group.representation(self.value)
        element_rep = element.group.representation(element.value)
        value =  self.group.derepresentation(element_rep @ self_rep)
        
        return GroupElement(value, self.group) 

    def inverseElement(self):
        matrix = self.group.representation(self.value)
        inverse_value = self.group.derepresentation(self.group.inverse_func(matrix))
        return GroupElement(inverse_value, self.group)
    
class Vector_Tangent():
    def  __init__(self, config, velocity, velocity_derep): 
        self.config  = config # tail end of vector
        self.velocity = velocity # vector velocity
        self.velocity_derep = velocity_derep
     
    def ThLg (self, h, h_dot):
        JL = self.config.right_lifted_action(h) 
        matrix = self.velocity_derep(h_dot.group.representation(h_dot.value))
        velocity = np.array(matrix) @ np.array(JL.group.representation(JL.value))
        return Vector_Tangent(self.config,velocity, self.velocity_derep )

    def ThRg (self, h, h_dot):
        JL = self.config.left_lifted_action(h)
        matrix = self.velocity_derep(h_dot.group.representation(h_dot.value))
        velocity = np.array(JL.group.representation(JL.value)) @ np.array(matrix)
        return Vector_Tangent(self.config,velocity, self.velocity_derep)
    
    def Ad_g_gcirc (self,g, g_dot):
        g_inverse= g.inverseElement()
        Ad = g.group.representation(g.value) @ g_dot.group.representation(g_dot.value) @ g_inverse.group.representation(g_inverse.value)
        velocity_derep = self.velocity_derep(self.config.group.representation(self.config.value) @ Ad)
        
        return Vector_Tangent(self.config,velocity_derep, self.velocity_derep)

    def Ad_g_gcirc_inv (self, g, g_dot):
        g_inverse= g.inverseElement()
        Ad_inv = g_inverse.group.representation(g_inverse.value) @ g_dot.group.representation(g_dot.value) @ g.group.representation(g.value)
        velocity_derep = self.velocity_derep(self.config.group.representation(self.config.value) @ Ad_inv)
       
        return Vector_Tangent(self.config,velocity_derep, self.velocity_derep)

def lifted_actions_plot(configs_x, configs_y, title):
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        plt.quiver(config_x.config.value[0], config_x.config.value[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'k')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config.value[0], config_y.config.value[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'r')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()

def adjoint_plot(configs_x, configs_y, title):
    
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        plt.quiver(config_x.config.value[0], config_x.config.value[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'k')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config.value[0], config_y.config.value[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'r')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()



def main():
    entry_points = []
    for x in range (0,5):
        for y in range (-2,3):
            if x != 0: 
                entry_points.append ([x/2,y/2])

    #Part 6
    G = Group(1,inverse_func, representation, derepresentation) 
    h1 = GroupElement([1,0], G)
    h2 = GroupElement([1,0], G)
    h_dot1 =GroupElement([1,0], G)
    h_dot2 = GroupElement([0,1], G)
    configs_x_lifted_left = []
    configs_y_lifted_left = []

    for point in entry_points: 
        entry_point = GroupElement(point, G)
        config = Vector_Tangent(entry_point, 0, velocity_derep)
        config_x = config.ThLg(h1, h_dot1)
        configs_x_lifted_left.append(config_x)
        config_y = config.ThLg(h2, h_dot2)
        configs_y_lifted_left.append(config_y)
   
    lifted_actions_plot(configs_x_lifted_left, configs_y_lifted_left, "Left Lifted Action")
    

    configs_x_lifted_right = []
    configs_y_lifted_right = []

    for point in entry_points: 
        entry_point = GroupElement(point, G)
        config = Vector_Tangent(entry_point, 0, velocity_derep)
        config_x = config.ThRg(h1, h_dot1)
        configs_x_lifted_right.append(config_x)
        config_y = config.ThRg(h2, h_dot2)
        configs_y_lifted_right.append(config_y)

    lifted_actions_plot(configs_x_lifted_right, configs_y_lifted_right, "Right Lifted Action")
    
    # Part 6 Adjoint
    g = GroupElement([0.5,-1], G)
    g_circ = GroupElement([1,0.5], G)
    
    configs_adjoint = []
    configs_adjoint_inv = []
    
    for point in entry_points: 
        entry_point = GroupElement(point, G)
        config = Vector_Tangent(entry_point, 0, velocity_derep)
        configs = config.Ad_g_gcirc(g, g_circ)
        configs_adjoint.append(configs)
        configs_inv = config.Ad_g_gcirc_inv(g, g_circ)
        configs_adjoint_inv.append(configs_inv)
       
    adjoint_plot(configs_adjoint,configs_adjoint_inv, "Adjoint and Adjoint Inverse with G_circ with matrix representation")
  

def representation(v):
    return np.array([[v[0],v[1]], [0, 1]])

def derepresentation(matrix):
    x = matrix[0][0]
    y = matrix[0][1]
    return ([x,y])

def inverse_func(matrix):
    return np.linalg.inv(matrix)

def velocity_derep(velocity):
    row, col =  np.array(velocity).shape
    vectorized = []
    for j in range (0, col):
        for i in range (0, row):
            vectorized.append(velocity[i][j]) 
    
    d_dp = np.array([[1,0], [0, 0], [0,1], [0, 0]])
    d_dp = np.linalg.pinv(np.array(d_dp)) 
    vectorized = np.array(vectorized)

    vel_derep = d_dp @ vectorized

    return vel_derep

main()