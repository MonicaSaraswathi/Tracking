# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dt = params.dt
        self.q=params.q
        self.dim_state=params.dim_state

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        F=np.matrix([[1,0,0,self.dt,0,0],[0,1,0,0,self.dt,0],[0,0,1,0,0,self.dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dtsq=1/2*self.q*self.dt**2
        dtcube=1/3*self.q*self.dt**3
        dt=self.dt*self.q
        Q=np.matrix([[dtcube,0,0,dtsq,0,0],[0,dtcube,0,0,dtsq,0],[0,0,dtcube,0,0,dtsq],[dtsq,0,0,dt,0,0],[0,dtsq,0,0,dt,0],[0,0,dtsq,0,0,dt]])
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        x = self.F()*track.x
        F=self.F()
        P = F*track.P*F.transpose()+self.Q()
        track.set_x(x)
        track.set_P(P)
        #print(P)
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        x= track.x
        P=track.P
        H= meas.sensor.get_H(x)
        res=self.gamma(track,meas)
        S=self.S(track,meas,H)
        K = (P*H.transpose()*np.linalg.inv(S))
        x = x + K*res
        I=np.identity(self.dim_state)
        P = (I-K*H)*P
        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        res = (meas.z-meas.sensor.get_H(track.x)*track.x)
        return res
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S = H*track.P*H.transpose()+meas.R
        return S
        
        ############
        # END student code
        ############ 