#! /usr/bin/python3
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

from geomotion import (
    utilityfunctions as ut,
    rigidbody as rb)
from ROB541_Assignment3_Chelse import simplekinematicchain as kc
import numpy as np
from matplotlib import pyplot as plt


# Set the group as SE2 from rigidbody
G = rb.SE2


class DiffKinematicChain(kc.KinematicChain):

    def __init__(self,
                 links,
                 joint_axes):

        """Initialize a kinematic chain augmented with Jacobian methods"""

        # Call the constructor for the base KinematicChain class
        super().__init__(links, joint_axes)

        # Create placeholders for the last-calculated Jacobian value and the index it was requested for
        self.last_Jacobian = None 
        self.index = 0 
        
    
    def Jacobian_Ad_inv(self,
                        link_index,  # Link number (with 1 as the first link)
                        output_frame='body'):  # options are world, body, spatial

        """Calculate the Jacobian by using the Adjoint_inverse to transfer velocities from the joints to the links"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        J = np.zeros((G.n_dim,len(self.joint_axes)))

        ########
        # Populate the Jacobian matrix by finding the transform from each joint before the chosen link to the
        # end of the link, and using its Adjoint-inverse to transform the joint axis to the body frame of
        # the selected link, and then transform this velocity to the world, or spatial coordinates if requested

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry
        link_positions_with_base = [G.identity_element()]
        link_positions_with_base.extend(self.link_positions)

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):

            # create a transform g_rel that describes the position of the selected link relative the jth joint (which
            # is at the (j-1)th location in link_positions_with_base
            g_rel= link_positions_with_base[j-1].inverse.L(self.link_positions[link_index-1])

            # use the Adjoint-inverse of this relative transformation to map the jth joint axis ( (j-1)th entry)
            # out to the end of the selected link in the link's body frame
            J_joint = g_rel.Ad_inv(self.joint_axes[j-1])

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates
            if output_frame == 'world':
                J_joint = self.link_positions[link_index-1] * J_joint 

            # If the output_frame input is 'spatial', use the adjoint of the link position to map the axis back to
            # the identity
            if output_frame == 'spatial': 
                J_joint = J_joint.Ad(self.link_positions[link_index-1])

            # Insert the value of J_joint into the (j-1)th index of J
            J[:,j-1]= J_joint.value

            # Store J and the last requested index
            self.last_Jacobian = J 
            self.index = link_index 
        return J
    
    def Jacobian_Ad(self,
                    link_index,  # Link number (with 1 as the first link)
                    output_frame='body'):  # options are world, body, spatial

        """Calculate the Jacobian by using the Adjoint to transfer velocities from the joints to the origin"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        J = np.zeros((G.n_dim,len(self.joint_axes)))

        ########
        # Populate the Jacobian matrix by finding the position of each joint in the world (which is the same as the
        # position of the previous link), and using its Adjoint to send the axis into spatial coordinates

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry
        link_positions_with_base = [G.identity_element()]
        link_positions_with_base.extend(self.link_positions)

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):

            # use the Adjoint of the position of this joint to map its joint axis ( (j-1)th entry)
            # back to the identity of the group
            J_joint = link_positions_with_base[j-1].Ad(self.joint_axes[j-1])

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates
            if output_frame == 'world':
                J_joint =  J_joint * self.link_positions[link_index-1]

            # If the output_frame input is 'body', use the adjoint-inverse of the link position to map the axis back to
            # the identity
            if output_frame == 'body': 
                J_joint = J_joint.Ad_inv(self.link_positions[link_index-1])

            # Insert the value of J_joint into the (j-1)th index of J
            J[:,j-1]= J_joint.value

            # Store J and the last requested index
            self.last_Jacobian = J 
            self.index = link_index 

        return J

    def draw_Jacobian(self,
                      ax,title):

        """ Draw the components of the last-requested Jacobian"""

        # Get the location of the last-requested link, and use ut.column to make into a numpy column array
        last_link_location = ut.column(self.link_positions[self.index-1])

        # Use np.tile to make a matrix in which each column is the coordinates of the link end
        matrix_tile = np.tile(last_link_location,len(self.last_Jacobian[0]))
        print("Mat tile: ", matrix_tile)

        # Use ax.quiver to plot a set of arrows at the selected link end, (remembering to use only the xy components
        # and not the theta component)
        x = matrix_tile[0,:].flatten()
        y = matrix_tile[1,:].flatten()
        j_x = self.last_Jacobian[0,:]
        j_y = self.last_Jacobian[1,:]
        ax.quiver(x, y, j_x, j_y,angles = 'xy', scale_units = 'xy', scale = 4 )
        ax.set_aspect ("equal")
        ax.set_xlim(0,6)
        ax.set_ylim(-1,3)
        ax.set_title(title)


if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc = DiffKinematicChain(links, joint_axes)

    # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    kc.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])
    
    # Create a plotting axis
    ax = plt.subplot(2, 3, 1)


    # Draw the chain
    kc.draw(ax)

    for i in range (1,4):
        J_Ad_inv = kc.Jacobian_Ad_inv(i, 'world')
        title = f'Jacobian_Ad_inv for {[round(.15 * np.pi,2), round(-.5 * np.pi,2), round(.9 * np.pi,2)]}'
        kc.draw_Jacobian(ax,title)

    ax = plt.subplot(2, 3, 4)

    # Draw the chain
    kc.draw(ax)

    for i in range (1,4):
        J_Ad_inv = kc.Jacobian_Ad(i, 'world')
        title = f'Jacobian_Ad for {[round(.15 * np.pi,2), round(-.5 * np.pi,2), round(.9 * np.pi,2)]}'
        kc.draw_Jacobian(ax,title)

    kc.set_configuration([.1 * np.pi, 0 * np.pi, .75 * np.pi])

    ax = plt.subplot(2, 3, 2)


    # Draw the chain
    kc.draw(ax)

    for i in range (1,4):
        J_Ad_inv = kc.Jacobian_Ad_inv(i, 'world')
        title = f'Jacobian_Ad_inv for {[round(.1 * np.pi,2), round(0 * np.pi,2), round(.75 * np.pi,2)]}'
        kc.draw_Jacobian(ax,title)

    ax = plt.subplot(2, 3, 5)

    # Draw the chain
    kc.draw(ax)

    for i in range (1,4):
        J_Ad_inv = kc.Jacobian_Ad(i, 'world')
        title = f'Jacobian_Ad for {[round(.1 * np.pi,2), round(0 * np.pi,2), round(.75 * np.pi,2)]}'
        kc.draw_Jacobian(ax,title)


    kc.set_configuration([.15 * np.pi, -.5 * np.pi, .9 * np.pi])

    ax = plt.subplot(2, 3, 3)


    # Draw the chain
    kc.draw(ax)

    for i in range (1,4):
        J_Ad_inv = kc.Jacobian_Ad_inv(i, 'world')
        title = f'Jacobian_Ad_inv for {[round(.15 * np.pi,2), round(-.5 * np.pi,2), round(.9 * np.pi,2)]}'
        kc.draw_Jacobian(ax,title)

    ax = plt.subplot(2, 3, 6)

    # Draw the chain
    kc.draw(ax)

    for i in range (1,4):
        J_Ad_inv = kc.Jacobian_Ad(i, 'world')
        title = f'Jacobian_Ad for {[round(.15 * np.pi,2), round(-.5 * np.pi,2), round(.9 * np.pi,2)]}'
        kc.draw_Jacobian(ax,title)

    # Tell pyplot to draw
    plt.show()
