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

    def left_lifted_action(self, element):
        element_val = [element]
        return nd.Jacobian (lambda element_val: self.left_action_helper(self, element_val))(self.value)


    def left_action_helper(self, value, hval):
            hval_element = GroupElement(hval, self.group )
            return (value.group.representation(value.value) @ hval_element.group.representation(hval_element.value)) 

    def right_lifted_action(self, element):
        element_val = [element]
        return nd.Jacobian (lambda element_val: self.right_action_helper(self, element_val))(self.value) 


    def right_action_helper(self, value, hval):
            hval_element = GroupElement(hval, self.group )
            return (hval_element.group.representation(hval_element.value) @ value.group.representation(value.value) ) 

    def right_action(self, element):
        self_rep = self.group.representation(self.value)
        element_rep = element.group.representation(element.value)
        value =  self.group.derepresentation(element_rep @ self_rep)
        
        return GroupElement(value, self.group) 


class Vector_Tangent():
    def  __init__(self, config, velocity): 
        self.config  = config # tail end of vector
        self.velocity = velocity # vector velocity
    
    def derivative_in_the_direction(self, fcn, entry_point):
        config = lambda delta:fcn(entry_point, delta)
        velocity = nd.Derivative(config)(0) 
      
        return Vector_Tangent(entry_point,velocity)
    
    def derivative_in_the_direction_jacobian(self, fcn, entry_point):
        velocity = nd.Jacobian(lambda delta:fcn(entry_point, delta))(0)
      
        return Vector_Tangent(entry_point, velocity )
    
    def direction_of_derivative_group_action(self, entry_point, group):
        dx = nd.Jacobian(lambda delta:self.helper_with_group_action(entry_point, delta, 0, group))(0)
        dy = nd.Jacobian(lambda delta:self.helper_with_group_action(entry_point, delta, 1, group))(0)
        config_x = Vector_Tangent(entry_point.value, dx ) 
        config_y = Vector_Tangent(entry_point.value, dy)
        return  (config_x, config_y)
        
    def helper_with_group_action (self,entry_point, delta, index, group):
        value = [0,0]
        value[index] = delta[0]
        gE = GroupElement(value, group)
        return np.array(gE.left_action(entry_point).value)
    
    def direction_of_derivative_group_action_right(self, entry_point, group):
        dx = nd.Jacobian(lambda delta:self.helper_with_group_action_right(entry_point, delta, 0, group))(0)
        dy = nd.Jacobian(lambda delta:self.helper_with_group_action_right(entry_point, delta, 1, group))(0)
        config_x = Vector_Tangent(entry_point.value, dx ) 
        config_y = Vector_Tangent(entry_point.value, dy)
        return  (config_x, config_y)
        
    def helper_with_group_action_right (self, entry_point, delta, index, group):
        value = [0,0]
        value[index] = delta[0]
        gE = GroupElement(value, group)
        return np.array(gE.right_action(entry_point).value)
    
    def ThLg (self, h, entry_point, h_dot):
        JL = np.array(entry_point.right_lifted_action(h)[0])
        velocity = h_dot @ JL

        return Vector_Tangent(entry_point.value, velocity)

    def ThRg (self, h, entry_point, h_dot):
        JL = np.array(entry_point.left_lifted_action(h)[0])
        velocity = JL @ h_dot 

        return Vector_Tangent(entry_point.value, velocity)

    def helper_with_group_action_lifted (self, entry_point, delta, index, group):
        value = [0,0]
        value[index] = delta[0]
        gE = GroupElement(value, group)
        return np.array(gE.left_action_lifted(entry_point).value)

    def Ad_g_gcirc (self, entry_point, g, g_dot):
        Ad = g.group.inverse_func(g.right_lifted_action(g.left_lifted_action(g_dot.value))[0])
        velocity = (entry_point.value) @ Ad
        
        return Vector_Tangent(self.config, velocity)

    def Ad_g_gcirc_inv (self,entry_point, g, g_dot):
        Ad_inv = g.group.inverse_func(g.left_lifted_action(g.right_lifted_action(g_dot.value))[0])
        velocity = (entry_point.value) @ Ad_inv
        
        return Vector_Tangent(self.config, velocity)

def f1 (config, delta):
    p = 1 + delta/math.sqrt(config[0]**2+config[1]**2)
    theta =[[math.cos(delta), -1*math.sin(delta)],[math.sin(delta),math.cos(delta)]]
    config_matrix = np.array([config[0], config[1]])
    return (p* np.matmul(theta, config_matrix))

def direction_plot(configs, title ):
    fig,ax = plt.subplots()
    for config in configs: 
        plt.quiver(config.config[0], config.config[1],config.velocity[0], config.velocity[1], angles = 'xy', scale_units = 'xy', scale = 3)
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.show()

def lifted_actions_plot(configs_x, configs_y, title):
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        plt.quiver(config_x.config[0], config_x.config[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'r')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config[0], config_y.config[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'k')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()

def adjoint_plot(configs_x, configs_y, title):
    
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        plt.quiver(config_x.config[0], config_x.config[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 8, color = 'r')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config[0], config_y.config[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 8, color = 'k')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()

def main():

    #part 2 
    configs = []
    entry_points = []

    for x in range (-2,3):
        for y in range (-2,3):
            entry_points.append ([x,y])
    
    for point in entry_points:
        config = Vector_Tangent(point, 0) 
        new_config = config.derivative_in_the_direction(f1, point)
        configs.append(new_config)

    direction_plot(configs, "Direction of Derivative with Derivative")
    
    for point in entry_points: 
        config_jacobian = Vector_Tangent(point, 0)
        new_config = config_jacobian.derivative_in_the_direction_jacobian(f1, point)
        configs.append(new_config)

    direction_plot(configs, "Direction of Derivative with Jacobian")
    
    entry_points = []
    for x in range (0,5):
        for y in range (-2,3):
            if x != 0: 
                entry_points.append ([x/2,y/2])

    #part 3
    G = Group(1,inverse_func, representation, derepresentation) 
    configs_x = []
    configs_y = []
    for point in entry_points: 
        entry_point = GroupElement(point, G)

        config = Vector_Tangent(point, 0)
        configs = config.direction_of_derivative_group_action(entry_point,G)
        configs_x.append (configs[0])
        configs_y.append (configs[1])

    lifted_actions_plot(configs_x, configs_y, "Over Partial Lg")

    configs_x_right = []
    configs_y_right = []
    for point in entry_points: 
        entry_point = GroupElement(point, G)

        config = Vector_Tangent(point, 0)
        configs = config.direction_of_derivative_group_action_right(entry_point,G)
        configs_x_right.append (configs[0])
        configs_y_right.append (configs[1])

    lifted_actions_plot(configs_x_right, configs_y_right, "Over Partial Rh")
    
    #Part 4 first deliverable
    h1 = GroupElement([1,0], G)
    h2 = GroupElement([1,0], G)
    h_dot1 =np.array([1,0])
    h_dot2 = np.array([0,1])
    configs_x_lifted = []
    configs_y_lifted = []

    for point in entry_points: 
        config = Vector_Tangent(point, 0)
        entry_point = GroupElement(point, G)
        config_x = config.ThLg(h1, entry_point, h_dot1)
        configs_x_lifted.append(config_x)
        config_y = config.ThLg(h2, entry_point, h_dot2)
        configs_y_lifted.append(config_y)

    lifted_actions_plot(configs_x_lifted, configs_y_lifted, "Left Lifted Action")
    
    configs_x_lifted_right = []
    configs_y_lifted_right = []

    for point in entry_points: 
        config = Vector_Tangent(point, 0)
        entry_point = GroupElement(point, G)
        config_x = config.ThRg(h1, entry_point, h_dot1)
        configs_x_lifted_right.append(config_x)
        config_y = config.ThRg(h2, entry_point, h_dot2)
        configs_y_lifted_right.append(config_y)

    lifted_actions_plot(configs_x_lifted_right, configs_y_lifted_right, "Right Lifted Action")

    # Part 4 Adjoint
    g = GroupElement([0.5,-1], G)
    g_circ = GroupElement([1,0.5], G)
    
    configs_adjoint = []
    configs_adjoint_inv = []

    for point in entry_points: 
        config = Vector_Tangent(point, 0)
        entry_point = GroupElement(point, G)
        configs = config.Ad_g_gcirc(entry_point, g, g_circ)
        configs_adjoint.append(configs)
        configs_inv = config.Ad_g_gcirc_inv(entry_point, g, g_circ)
        configs_adjoint_inv.append(configs_inv)
       

    adjoint_plot(configs_adjoint,configs_adjoint_inv, "Adjoint and Adjoint Inverse")


def representation(value):
    return np.array([[value[0],value[1]], [0, 1]])

def derepresentation(matrix):
    x = matrix[0][0]
    y = matrix[0][1]
    return ([x,y])

def inverse_func(matrix):
    return np.linalg.inv(matrix)

main()
