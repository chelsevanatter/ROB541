import numpy as np
import matplotlib.pyplot as plt

class Group():
    def __init__(self, identity, operation, inverse_function, representation, derepresentation):
        self.identity = identity
        self.operation = operation
        self.inverse_function = inverse_function
        self.representation = representation
        self.derepresentation = derepresentation
        
    def operation_function(input_1,input_2,operation_type):
        return operation_type
    
    def element(self,value):
        return GroupElement(value,self)
    
    def identity_element(self):
        return self.element(self.identity)

class GroupElement():
    def __init__(self,value,group):
        self.group = group
        self.value = value

    def left_action_part1(self, element):
        value = self.group.operation(self.value, element.value)
        new_left_action_element = GroupElement(value, self.group ) 
        return new_left_action_element 

    def right_action_part1(self, element):
        value = self.group.operation(element.value,self.value)
        new_right_action_element = GroupElement(value, self.group ) 
        return new_right_action_element 
    
    def inverse_element_part1(self):
        inverse_value = self.group.inverse_function(self.value)
        element_inverse = GroupElement(inverse_value, self.group)
        return element_inverse
    
    def left_action_part2(self,element):
        g1_representation = self.group.representation(self.value)
        element_representation = element.group.representation(element.value)
        representation_matrix = g1_representation @ element_representation
        derep_value = self.group.derepresentation(representation_matrix)
        new_left_action_element = GroupElement(derep_value, self.group ) 
        return new_left_action_element

    def right_action_part2(self,element):
        g1_representation = self.group.representation(self.value)
        element_representation = element.group.representation(element.value)
        representation_matrix = element_representation @ g1_representation
        derep_value = self.group.derepresentation(representation_matrix)
        new_right_action_element = GroupElement(derep_value, self.group ) 
        return new_right_action_element
    
    def inverse_element_part2(self):
        matrix = self.group.representation(self.value)
        inverse_value = self.group.inverse_function(matrix)
        inverse_value = self.group.derepresentation(inverse_value)
        element_inverse = GroupElement(inverse_value, self.group)
        return element_inverse
    
    def AD(self,element):
        representation = self.group.representation(self.value) @ element.group.representation(element.value) @ np.linalg.inv(self.group.representation(self.value))
        derepresentation = self.group.derepresentation(representation)
        return GroupElement(derepresentation, self.group)
    
    def AD_inverse(self,element):
       representation = np.linalg.inv(self.group.representation(self.value)) @ element.group.representation(element.value) @ self.group.representation(self.value)
       derepresentation = self.group.derepresentation(representation)
       return GroupElement(derepresentation, self.group)
    
    def multiply_two_pos (self, element):
        self_representation =  self.group.representation(self.value)
        element_representation = self.group.representation(element.value)
        product = self_representation @ element_representation
        derepresentation = self.group.derepresentation(product)
        return GroupElement(derepresentation, self.group)

def affine_add(input1,input2):
    return input1 + input2

def scalar_mult(input1,input2):
    return input1 * input2

def modular_add(input1,input2,phi):
    return (input1+input2) % phi

def composition(input1,input2):
    element_1 = input1[0] + input2[0] * np.cos(input1[2]) - input2[1] * np.sin(input1[2])
    element_2 = input1[1] + input2[0] * np.sin(input1[2]) + input2[1] * np.cos(input1[2])
    element_3 = input1[2] + input2[2]
    return np.array([element_1,element_2,element_3])

def inverse_function_part1(value):
    matrix = np.array([[np.cos(value[2]), -1*np.sin(value[2]), value[0]], [np.sin(value[2]), np.cos(value[2]), value[1]], [0, 0, 1]])
    matrix = np.linalg.inv(matrix)
    theta = np.arctan2(matrix[1][0], matrix[1][1])
    x = matrix[0][2]
    y = matrix[1][2]
    return (np.array([x, y, theta]))

def inverse_function_part2(matrix):
    return np.linalg.inv(matrix)

def representation(i):
    return np.array([[np.cos(i[2]), -1*np.sin(i[2]), i[0]], [np.sin(i[2]), np.cos(i[2]), i[1]], [0, 0, 1]])

def derepresentation(matrix):
    theta = np.arctan2(matrix[1][0], matrix[1][1])
    x = matrix[0][2]
    y = matrix[1][2]
    return (np.array([x, y, theta]))

def make_plot(g,h,gh,hg,h_position_relative_to_g,g_position_relative_to_h):
    plt.grid()
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('x')
    plt.ylabel('y')
    
    c0 = 'red'
    c1= 'blue'
    c2 = 'purple'
    c3 = 'pink'
    c4 = 'green'
    c5 = 'orange'

    plt.quiver(g.value[0],g.value[1],np.cos(g.value[2]),np.sin(g.value[2]), color = c0)
    plt.quiver(h.value[0],h.value[1],np.cos(h.value[2]),np.sin(h.value[2]), color = c1)
    plt.quiver(gh.value[0],gh.value[1],np.cos(gh.value[2]),np.sin(gh.value[2]), color = c2)
    plt.quiver(hg.value[0],hg.value[1],np.cos(hg.value[2]),np.sin(hg.value[2]), color = c3)
    plt.quiver(h_position_relative_to_g.value[0],h_position_relative_to_g.value[1],np.cos(h_position_relative_to_g.value[2]),np.sin(h_position_relative_to_g.value[2]), color = c4)
    plt.quiver(g_position_relative_to_h.value[0],g_position_relative_to_h.value[1],np.cos(g_position_relative_to_h.value[2]),np.sin(g_position_relative_to_h.value[2]), color = c5)

    plt.plot(g.value[0],g.value[1],label = 'g', color = c0, marker = 'o')
    plt.plot(h.value[0],h.value[1],label = 'h', color = c1, marker = 'o')
    plt.plot(gh.value[0],gh.value[1],label = 'gh', color = c2, marker = 'o')
    plt.plot(hg.value[0],hg.value[1],label = 'hg', color = c3, marker = 'o')
    plt.plot(h_position_relative_to_g.value[0],h_position_relative_to_g.value[1],label = 'h pos rel to g', color = c4, marker = 'o')
    plt.plot(g_position_relative_to_h.value[0],g_position_relative_to_h.value[1],label = 'g pos rel to h', color = c5, marker = 'o')
  
    text_offset = 0.1
    plt.text(g.value[0]-text_offset,g.value[1]+0.1,f'g', fontsize=16, color=c0)
    plt.text(h.value[0]-text_offset,h.value[1]+0.1,f'h', fontsize=16, color=c1)
    plt.text(gh.value[0]-text_offset,gh.value[1]+0.1,f'gh', fontsize=16, color=c2)
    plt.text(hg.value[0]-text_offset,hg.value[1]+0.1,f'hg', fontsize=16, color=c3)
    plt.text(h_position_relative_to_g.value[0]-text_offset,h_position_relative_to_g.value[1]+0.1,f'h rel g', fontsize=16, color=c4)
    plt.text(g_position_relative_to_h.value[0]-text_offset,g_position_relative_to_h.value[1]-0.3,f'g rel h', fontsize=16, color=c5)
    
    plt.legend()

def part1():
    G = Group(1,composition,inverse_function_part1,representation, derepresentation)

    g = G.element(np.array([0,1,-np.pi/4]))
    h = G.element(np.array([1,2,-np.pi/2]))
    
    gh = g.left_action_part1(h)
    hg = h.left_action_part1(g)

    h_inverse = h.inverse_element_part1()
    print('h inverse ', h_inverse.value)
    g_inverse = g.inverse_element_part1()
    print('g inverse ', g_inverse.value)

    h_position_relative_to_g = g_inverse.left_action_part1(h)
    print('position of h relative to g ',h_position_relative_to_g.value)
    g_position_relative_to_h = h_inverse.left_action_part1(g)
    print('position of g relative to h ',g_position_relative_to_h.value)
    
    part1_plot = plt.figure(1)
    plt.title ('Part 1: Plot with g, h, gh, hg, g rel h, h rel g not using matrix')
    make_plot(g,h,gh,hg,h_position_relative_to_g,g_position_relative_to_h)
    part1_plot.show()

def part2():
    G = Group(1,composition,inverse_function_part2,representation, derepresentation)

    g = G.element(np.array([0,1,-np.pi/4]))
    h = G.element(np.array([1,2,-np.pi/2]))
    
    gh = g.left_action_part2(h)
    hg = h.left_action_part2(g)

    h_inverse = h.inverse_element_part2()
    g_inverse = g.inverse_element_part2()

    h_position_relative_to_g = g_inverse.left_action_part2(h)
    print('position of h relative to g ',h_position_relative_to_g.value)
    g_position_relative_to_h = h_inverse.left_action_part2(g)
    
    part2_plot = plt.figure(2)
    plt.title ('Part 2: Plot with g, h, gh, hg, g rel h, h rel g using matrix')
    make_plot(g,h,gh,hg,h_position_relative_to_g,g_position_relative_to_h)
    part2_plot.show()

def part3():
    # First deliverable
    G = Group(1,composition,inverse_function_part2,representation, derepresentation)

    g = G.element(np.array([0,1,-np.pi/4]))
    h = G.element(np.array([1,2,-np.pi/2]))
    
    h_inverse = h.inverse_element_part2()
    g_inverse = g.inverse_element_part2()

    h_position_relative_to_g = g_inverse.left_action_part2(h)
    
    h21 = g.AD(h_position_relative_to_g)
    h21_of_g = h21.left_action_part2(g)
    print('g2: ',h.value)
    print('AD of g1: ',h21_of_g.value)

    part3_deliverable1_plot = plt.figure(3)
    plt.title ('Part 3: Demonstration of how left action of the adjointed relative position on g1 brings it to g2')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel("x")
    plt.ylabel("y")

    plt.scatter(g.value[0], g.value[1], marker='o', label = 'g',color ='red')
    plt.scatter(h21.value[0], h21.value[1], marker='o', label = 'h21',color = 'blue')
    plt.scatter(h21_of_g.value[0], h21_of_g.value[1], marker='o', label = 'h21 composed with g',color = 'purple')
  
    plt.quiver(g.value[0],g.value[1],np.cos(g.value[2]), np.sin(g.value[2]),color ='red')
    plt.quiver(h21.value[0],h21.value[1],np.cos(h21.value[2]), np.sin(h21.value[2]),color = 'blue')
    plt.quiver(h21_of_g.value[0],h21_of_g.value[1],np.cos(h21_of_g.value[2]), np.sin(h21_of_g.value[2]),color = 'purple')

    plt.text(g.value[0]+0.1, g.value[1]+0.1,'g',color ='red')
    plt.text(h21.value[0]+0.1, h21.value[1]+0.1,'h21',color = 'blue')
    plt.text(h21_of_g.value[0]+0.1, h21_of_g.value[1]+0.1,'h21 of g',color = 'purple')
    
    plt.legend()
    part3_deliverable1_plot.show()

    # Second deliverable
    h1 = G.element(np.array([-1,0,np.pi/2 ]))
    g2 = G.element(np.array([1,2,-np.pi/2 ]))
    gh1 = g.left_action_part2(h1)
    g1_inv = g.inverse_element_part2()
    h21 = g1_inv.left_action_part2(g2)
    AD_inv = g.AD_inverse(h1)
    h2 = AD_inv.multiply_two_pos(h1) 
    print("H2: ", h2.value)
    gh2 = g2.left_action_part2(h2)
    AD_inv_of_h21 = h21.AD_inverse(h1)
    print("AD inverse of h21", h2.value)
    
    part3_deliverable2_plot = plt.figure(4)
    plt.title ('Part 3: Demonstration of preservation of relative diplacement moving g1 by h1 and g2 by h2')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel("x")
    plt.ylabel("y")

    plt.scatter(g.value[0], g.value[1], marker='o', label = 'g1',color ='red')
    plt.scatter(g2.value[0], g2.value[1], marker='o', label = 'g2',color ='pink')
    plt.scatter(h1.value[0], h1.value[1], marker='o', label = 'h1',color ='orange')
    plt.scatter(h2.value[0], h2.value[1], marker='o', label = 'h2',color ='green')
    plt.scatter(h21.value[0], h21.value[1], marker='o', label = 'h21',color = 'blue')
    plt.scatter(gh1.value[0], gh1.value[1], marker='o', label = 'gh1',color = 'cyan')
    plt.scatter(gh2.value[0], gh2.value[1], marker='o', label = 'gh2',color = 'magenta')
    plt.scatter(AD_inv_of_h21.value[0], AD_inv_of_h21.value[1], marker='o', label = 'AD inverse of h21',color = 'purple')
  
    plt.quiver(g.value[0],g.value[1],np.cos(g.value[2]), np.sin(g.value[2]),color ='red')
    plt.quiver(g2.value[0],g2.value[1],np.cos(g2.value[2]), np.sin(g2.value[2]),color ='pink')
    plt.quiver(h1.value[0],h1.value[1],np.cos(h1.value[2]), np.sin(h1.value[2]),color ='orange')
    plt.quiver(h2.value[0],h2.value[1],np.cos(h2.value[2]), np.sin(h2.value[2]),color ='green')
    plt.quiver(h21.value[0],h21.value[1],np.cos(h21.value[2]), np.sin(h21.value[2]),color = 'blue')
    plt.quiver(gh1.value[0],gh1.value[1],np.cos(gh1.value[2]), np.sin(gh1.value[2]),color = 'cyan')
    plt.quiver(gh2.value[0],gh2.value[1],np.cos(gh2.value[2]), np.sin(gh2.value[2]),color = 'magenta')
    plt.quiver(AD_inv_of_h21.value[0],AD_inv_of_h21.value[1],np.cos(AD_inv_of_h21.value[2]), np.sin(AD_inv_of_h21.value[2]),color = 'purple')

    plt.text(g.value[0]-0.1, g.value[1]+0.1,'g',color ='red')
    plt.text(g2.value[0]+0.1, g2.value[1]+0.1,'g2',color ='pink')
    plt.text(h1.value[0]+0.1, h1.value[1]+0.1,'h1',color ='orange')
    plt.text(h2.value[0]+0.1, h2.value[1]+0.1, 'h2',color ='green')
    plt.text(h21.value[0]+0.1, h21.value[1]+0.1,'h21',color = 'blue')
    plt.text(gh1.value[0]+0.1, gh1.value[1]-0.1, 'gh1',color = 'cyan')
    plt.text(gh2.value[0]+0.1, gh2.value[1]+0.1,'gh2',color = 'magenta')
    plt.text(AD_inv_of_h21.value[0]+0.1, AD_inv_of_h21.value[1]+0.1,'AD inv of h21',color = 'purple')
    
    plt.legend()
    part3_deliverable2_plot.show()

if __name__ == "__main__":
   part1()
   part2()
   part3()
   input()
   
