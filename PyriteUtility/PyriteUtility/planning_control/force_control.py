import sys
import os
sys.path.append(os.path.join(sys.path[0], '..')) # PyriteUtility

import time
import copy
import numpy as np

import PyriteUtility.spatial_math.spatial_utilities as su

# define a class for force control
class AdmittanceController():
    def __init__(self,
                 dt,
                 stiffness_mat6,
                 inertia_mat6,
                 damping_mat6,
                 force_limit,
                 direct_force_control_P_trans,
                 direct_force_control_I_trans,
                 direct_force_control_D_trans,
                 direct_force_control_P_rot,
                 direct_force_control_I_rot,
                 direct_force_control_D_rot,
                 logging = False
    ):
        #  parameters
        self.param_dt = dt
        self.param_stiffness_mat6 = stiffness_mat6
        self.param_inertia_mat6 = inertia_mat6
        self.param_damping_mat6 = damping_mat6
        self.param_force_limit = np.array(force_limit)
        self.param_direct_force_control_P_trans = direct_force_control_P_trans
        self.param_direct_force_control_I_trans = direct_force_control_I_trans
        self.param_direct_force_control_D_trans = direct_force_control_D_trans
        self.param_direct_force_control_P_rot = direct_force_control_P_rot
        self.param_direct_force_control_I_rot = direct_force_control_I_rot
        self.param_direct_force_control_D_rot = direct_force_control_D_rot
        #  internal controller states
        self.Tr = np.zeros([6,6])
        self.Tr_inv = np.zeros([6,6])
        self.v_force_selection = np.zeros(6)
        self.v_velocity_selection = np.zeros(6)
        self.diag_force_selection = np.zeros([6,6])
        self.diag_velocity_selection = np.zeros([6,6])
        self.m_anni = np.zeros([6,6])
        self.SE3_WTref = np.eye(4)
        self.SE3_WT = np.eye(4)
        self.SE3_TrefTadj = np.eye(4)
        self.SE3_WTadj = np.eye(4)
        self.SE3_TTadj = np.eye(4)
        self.SE3_WT_cmd = np.eye(4)
        self.spt_TTadj = np.zeros(6)
        self.spt_TTadj_new = np.zeros(6)
        self.Adj_WT = np.zeros([6,6])
        self.Adj_TW = np.zeros([6,6])
        self.Jac_bodyV_twist = np.zeros([6,6])
        self.Jac_twist_bodyV = np.zeros([6,6])
        self.v_spatial_cmd = np.zeros(6)
        self.v_body_ref_last = np.zeros(6)
        self.v_body_ref_adj = np.zeros(6)
        self.v_Tr_f = np.zeros(6)
        self.v_Tr_v = np.zeros(6)
        self.v_Tr = np.zeros(6)
        self.vd_Tr = np.zeros(6)
        self.wrench_T_Err_prev = np.zeros(6)
        self.wrench_T_Err_I = np.zeros(6)
        self.wrench_T_fb = np.zeros(6)
        self.wrench_Tr_cmd = np.zeros(6)
        self.wrench_T_spring = np.zeros(6)
        self.wrench_Tr_spring = np.zeros(6)
        self.wrench_T_cmd = np.zeros(6)
        self.wrench_T_Err = np.zeros(6)
        self.wrench_T_PID = np.zeros(6)
        self.wrench_Tr_PID = np.zeros(6)
        self.wrench_Tr_Err = np.zeros(6)
        self.wrench_Tr_damping = np.zeros(6)
        self.wrench_Tr_All = np.zeros(6)

        self.flag_logging = logging
        self.logs = {
            "SE3_WT": [],
            "SE3_WTref": [],
            "SE3_WTadj": [],
            "SE3_WT_cmd": [],
            "v_spatial_cmd": [],
            "v_body_ref_last": [],
            "v_body_ref_adj": [],
            "v_Tr_f": [],
            "v_Tr_v": [],
            "v_Tr": [],
            "vd_Tr": [],
            "wrench_T_Err_prev": [],
            "wrench_T_Err_I": [],
            "wrench_T_fb": [],
            "wrench_Tr_cmd": [],
            "wrench_T_spring": [],
            "wrench_Tr_spring": [],
            "wrench_T_cmd": [],
            "wrench_T_Err": [],
            "wrench_T_PID": [],
            "wrench_Tr_PID": [],
            "wrench_Tr_Err": [],
            "wrench_Tr_damping": [],
            "wrench_Tr_All": []
        }
    
    def setRobotStatus(self, SE3_WT:np.ndarray, wrench_WT:np.ndarray):
        # make sure the input is numpy array
        if not isinstance(SE3_WT, np.ndarray):
            raise ValueError("SE3_WT should be a numpy array")
        if not isinstance(wrench_WT, np.ndarray):
            raise ValueError("wrench_WT should be a numpy array")
        self.SE3_WT = SE3_WT
        self.wrench_T_fb = wrench_WT

    def setRobotReference(self, SE3_WT, wrench_WTr):
        if not isinstance(SE3_WT, np.ndarray):
            raise ValueError("SE3_WT should be a numpy array")
        if not isinstance(wrench_WTr, np.ndarray):
            raise ValueError("wrench_WTr should be a numpy array")
        self.SE3_WTref = SE3_WT
        self.wrench_Tr_cmd = wrench_WTr

    def setForceControlledAxis(self, Tr_new, n_af):
        """
        Set the force controlled axis and update the admittance control parameters accordingly.
        After axis update, the goal pose with offset should be equal to current pose
        in the new velocity controlled axes. To satisfy this requirement, we need to
        change SE3_TrefTadj accordingly
        """
        self.v_force_selection = np.zeros(6)
        self.v_velocity_selection = np.ones(6)
        for i in range(n_af):
            self.v_force_selection[i] = 1
            self.v_velocity_selection[i] = 0
        self.diag_force_selection = np.diag(self.v_force_selection)
        self.diag_velocity_selection = np.diag(self.v_velocity_selection)

        self.m_anni = self.diag_velocity_selection @ Tr_new @ self.Jac_bodyV_twist
        # project the current Tadj to new Force axis, so the new Tadj is safe under new axes
        self.spt_TTadj_new = (np.eye(6) - np.linalg.pinv(self.m_anni) @ self.m_anni) @ self.spt_TTadj
        # Tref doesnt change, so we need to compute the new TrefTadj to get the new Tadj
        self.SE3_TrefTadj = su.SE3_inv(self.SE3_WTref) @ self.SE3_WT @ su.twc_to_SE3(self.spt_TTadj_new) 
        # self.SE3_TrefTadj = self.SE3_WT @ su.twc_to_SE3(self.spt_TTadj_new) @ su.SE3_inv(self.SE3_WTref)

        # self.SE3_TrefTadj = np.eye(4) # debug

        self.wrench_T_Err_I = self.Tr_inv @ self.diag_force_selection @ self.Tr @ self.wrench_T_Err_I
        self.wrench_T_Err_prev = self.Tr_inv @ self.diag_force_selection @ self.Tr @ self.wrench_T_Err_prev

        self.Tr = Tr_new
        self.Tr_inv = np.linalg.inv(self.Tr)    

    def step(self):
        # ----------------------------------------
        #  Compute Forces in Generalized space
        # ----------------------------------------
        # Position updates
        self.SE3_WTadj = self.SE3_WTref @ self.SE3_TrefTadj 
        self.SE3_TTadj = su.SE3_inv(self.SE3_WT) @ self.SE3_WTadj
        self.spt_TTadj = su.SE3_to_spt(self.SE3_TTadj)

        self.Jac_twist_bodyV = su.JacTwist2BodyV(self.SE3_WT[:3,:3])
        self.Jac_bodyV_twist = np.linalg.inv(self.Jac_twist_bodyV)

        self.Adj_WT = su.SE3_to_adj(self.SE3_WT)
        self.Adj_TW = su.SE3_to_adj(su.SE3_inv(self.SE3_WT))

        # Velocity updates
        self.v_body_ref_last = self.Adj_TW @ self.v_spatial_cmd
        self.v_Tr = self.Tr @ self.v_body_ref_last

        # Wrench updates
        self.wrench_T_spring = self.Jac_bodyV_twist @ self.param_stiffness_mat6 @ self.spt_TTadj
        self.wrench_Tr_spring = self.Tr @ self.wrench_T_spring

        # Force error, PID force control
        self.wrench_T_cmd = self.Tr_inv @ self.wrench_Tr_cmd
        self.wrench_T_Err = self.wrench_T_cmd - self.wrench_T_fb
        self.wrench_T_Err_I += self.wrench_T_Err
        np.clip(self.wrench_T_Err_I, -self.param_force_limit, self.param_force_limit, out=self.wrench_T_Err_I)

        self.wrench_T_PID[:3] = self.param_direct_force_control_P_trans * self.wrench_T_Err[:3] + \
                                self.param_direct_force_control_I_trans * self.wrench_T_Err_I[:3] + \
                                self.param_direct_force_control_D_trans * (self.wrench_T_Err[:3] - self.wrench_T_Err_prev[:3])
        self.wrench_T_PID[3:] = self.param_direct_force_control_P_rot * self.wrench_T_Err[3:] + \
                                self.param_direct_force_control_I_rot * self.wrench_T_Err_I[3:] + \
                                self.param_direct_force_control_D_rot * (self.wrench_T_Err[3:] - self.wrench_T_Err_prev[3:])
        self.wrench_Tr_PID = self.Tr @ self.wrench_T_PID
        self.wrench_T_Err_prev = self.wrench_T_Err
        self.wrench_Tr_Err = self.Tr @ self.wrench_T_Err

        # Force damping
        self.wrench_Tr_damping = -self.Tr @ self.param_damping_mat6 @ self.v_body_ref_last
        self.wrench_Tr_All = self.diag_force_selection @ (self.wrench_Tr_spring + self.wrench_Tr_Err + self.wrench_Tr_PID + self.wrench_Tr_damping)

        # ----------------------------------------
        #  force to velocityR
        # ----------------------------------------
        # Newton's Law
        #  Axes are no longer independent when we take
        #      rotation in to consideration.
        #  Newton's Law in body (Tool) frame:
        #      W=M*vd
        #          W: body wrench
        #          M: Inertia matrix in body frame
        #          vd: body velocity time derivative
        #  Newton's law in transformed space
        #      TW=TMTinv Tvd
        #      W_Tr = TMTinv vd_Tr
        self.vd_Tr = np.linalg.solve(self.Tr @ self.param_inertia_mat6 @ self.Tr_inv, self.wrench_Tr_All)

        # Velocity in the force-controlled direction: integrate acc computed from
        # Newton's law
        self.v_Tr += self.param_dt * self.vd_Tr
        self.v_Tr_f = self.diag_force_selection @ self.v_Tr # delete velocity in the velocity-controlled direction

        # print('Force v_W: ', self.Adj_WT @ self.Tr_inv @ self.v_Tr)
    
        # Velocity in the velocity-controlled direction: derive from reference pose
        self.v_body_ref_adj = self.Jac_twist_bodyV @ self.spt_TTadj / self.param_dt # reference velocity, derived from reference pose
        self.v_Tr_v = self.diag_velocity_selection @ self.Tr @ self.v_body_ref_adj
        self.v_Tr = self.v_Tr_f + self.v_Tr_v
        self.v_spatial_cmd = self.Adj_WT @ self.Tr_inv @ self.v_Tr

        # ----------------------------------------
        #  velocity to pose
        # ----------------------------------------
        self.SE3_WT_cmd = self.SE3_WT + su.wedge6(self.v_spatial_cmd) @ self.SE3_WT * self.param_dt

        # ----------------------------------------
        #  Logging
        # ----------------------------------------
        if self.flag_logging:
            self.logs["SE3_WT"].append(self.SE3_WT)
            self.logs["SE3_WTref"].append(self.SE3_WTref)
            self.logs["SE3_WTadj"].append(self.SE3_WTadj)
            self.logs["SE3_WT_cmd"].append(self.SE3_WT_cmd)
            self.logs["v_spatial_cmd"].append(self.v_spatial_cmd)
            self.logs["v_body_ref_last"].append(self.v_body_ref_last)
            self.logs["v_body_ref_adj"].append(self.v_body_ref_adj)
            self.logs["v_Tr_f"].append(self.v_Tr_f)
            self.logs["v_Tr_v"].append(self.v_Tr_v)
            self.logs["v_Tr"].append(self.v_Tr)
            self.logs["vd_Tr"].append(self.vd_Tr)
            self.logs["wrench_T_Err_prev"].append(self.wrench_T_Err_prev)
            self.logs["wrench_T_Err_I"].append(self.wrench_T_Err_I)
            self.logs["wrench_T_fb"].append(self.wrench_T_fb)
            self.logs["wrench_Tr_cmd"].append(self.wrench_Tr_cmd)
            self.logs["wrench_T_spring"].append(self.wrench_T_spring)
            self.logs["wrench_Tr_spring"].append(self.wrench_Tr_spring)
            self.logs["wrench_T_cmd"].append(self.wrench_T_cmd)
            self.logs["wrench_T_Err"].append(self.wrench_T_Err)
            self.logs["wrench_T_PID"].append(self.wrench_T_PID)
            self.logs["wrench_Tr_PID"].append(self.wrench_Tr_PID)
            self.logs["wrench_Tr_Err"].append(self.wrench_Tr_Err)
            self.logs["wrench_Tr_damping"].append(self.wrench_Tr_damping)
            self.logs["wrench_Tr_All"].append(self.wrench_Tr_All)
        return self.SE3_WT_cmd
    
    def plot_logs(self):
        assert(self.flag_logging)

        for key, item in self.logs.items():
            self.logs[key] = np.array(item)
        
        from plotly.offline import init_notebook_mode, iplot
        from plotly.subplots import make_subplots
        import plotly.graph_objs as go
        import plotly.io as pio
        import plotly.express as px

        pio.templates.default = "plotly_dark"
        pio.renderers.default = "browser"
        # pio.renderers.default = "vscode"
        
        fig = make_subplots(
            rows=6, cols=5,
            shared_xaxes='all',
            subplot_titles=('x (World)',  'v1', 'v1 (Tr)', 'wrench1 (Tool)', 'wrench1 (Tr)',
                            'y (World)',  'v2', 'v2 (Tr)', 'wrench2 (Tool)', 'wrench2 (Tr)',
                            'z (World)',  'v3', 'v3 (Tr)', 'wrench3 (Tool)', 'wrench3 (Tr)',
                            'qw (World)', 'v4', 'v4 (Tr)', 'wrench4 (Tool)', 'wrench4 (Tr)',
                            'qx (World)', 'v5', 'v5 (Tr)', 'wrench5 (Tool)', 'wrench5 (Tr)',
                            'qy (World)', 'v6', 'v6 (Tr)', 'wrench6 (Tool)', 'wrench6 (Tr)'
                            )
        )
        marker=dict(
            size=3,
            line=dict(
                width=1
            ),
            opacity=0.5,
        )
        
        time_s = np.arange(0, len(self.logs["SE3_WT"])) * self.param_dt
        
        pose_WT = su.SE3_to_pose7(self.logs["SE3_WT"])
        pose_WTref = su.SE3_to_pose7(self.logs["SE3_WTref"])
        pose_WTadj = su.SE3_to_pose7(self.logs["SE3_WTadj"])
        pose_WT_cmd = su.SE3_to_pose7(self.logs["SE3_WT_cmd"])
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT[:, 0], name='pose fb', legendgroup='group1'),  row=1, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT[:, 1], name='pose fb', legendgroup='group1', showlegend=False),  row=2, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT[:, 2], name='pose fb', legendgroup='group1', showlegend=False),  row=3, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT[:, 3], name='pose fb', legendgroup='group1', showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT[:, 4], name='pose fb', legendgroup='group1', showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT[:, 5], name='pose fb', legendgroup='group1', showlegend=False), row=6, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTref[:, 0], name='pose ref', legendgroup='group2'),  row=1, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTref[:, 1], name='pose ref', legendgroup='group2', showlegend=False),  row=2, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTref[:, 2], name='pose ref', legendgroup='group2', showlegend=False),  row=3, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTref[:, 3], name='pose ref', legendgroup='group2', showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTref[:, 4], name='pose ref', legendgroup='group2', showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTref[:, 5], name='pose ref', legendgroup='group2', showlegend=False), row=6, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTadj[:, 0], name='pose adjusted', legendgroup='group3'),  row=1, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTadj[:, 1], name='pose adjusted', legendgroup='group3', showlegend=False),  row=2, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTadj[:, 2], name='pose adjusted', legendgroup='group3', showlegend=False),  row=3, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTadj[:, 3], name='pose adjusted', legendgroup='group3', showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTadj[:, 4], name='pose adjusted', legendgroup='group3', showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WTadj[:, 5], name='pose adjusted', legendgroup='group3', showlegend=False), row=6, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT_cmd[:, 0], name='pose cmd', legendgroup='group4'),  row=1, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT_cmd[:, 1], name='pose cmd', legendgroup='group4', showlegend=False),  row=2, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT_cmd[:, 2], name='pose cmd', legendgroup='group4', showlegend=False),  row=3, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT_cmd[:, 3], name='pose cmd', legendgroup='group4', showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT_cmd[:, 4], name='pose cmd', legendgroup='group4', showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=pose_WT_cmd[:, 5], name='pose cmd', legendgroup='group4', showlegend=False), row=6, col=1)
        

        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_last"][:, 0], name='vel body prev', legendgroup='group5'), row=1, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_last"][:, 1], name='vel body prev', legendgroup='group5', showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_last"][:, 2], name='vel body prev', legendgroup='group5', showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_last"][:, 3], name='vel body prev', legendgroup='group5', showlegend=False), row=4, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_last"][:, 4], name='vel body prev', legendgroup='group5', showlegend=False), row=5, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_last"][:, 5], name='vel body prev', legendgroup='group5', showlegend=False), row=6, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_adj"][:, 0], name='vel body ref adjusted', legendgroup='group6'), row=1, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_adj"][:, 1], name='vel body ref adjusted', legendgroup='group6', showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_adj"][:, 2], name='vel body ref adjusted', legendgroup='group6', showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_adj"][:, 3], name='vel body ref adjusted', legendgroup='group6', showlegend=False), row=4, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_adj"][:, 4], name='vel body ref adjusted', legendgroup='group6', showlegend=False), row=5, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_body_ref_adj"][:, 5], name='vel body ref adjusted', legendgroup='group6', showlegend=False), row=6, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_spatial_cmd"][:, 0], name='vel spatial output', legendgroup='group7'), row=1, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_spatial_cmd"][:, 1], name='vel spatial output', legendgroup='group7', showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_spatial_cmd"][:, 2], name='vel spatial output', legendgroup='group7', showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_spatial_cmd"][:, 3], name='vel spatial output', legendgroup='group7', showlegend=False), row=4, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_spatial_cmd"][:, 4], name='vel spatial output', legendgroup='group7', showlegend=False), row=5, col=2)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_spatial_cmd"][:, 5], name='vel spatial output', legendgroup='group7', showlegend=False), row=6, col=2)

        fig.add_trace(go.Scatter(x=time_s, y=self.logs["vd_Tr"][:, 0], name='Tr acc output', legendgroup='group8'), row=1, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["vd_Tr"][:, 1], name='Tr acc output', legendgroup='group8', showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["vd_Tr"][:, 2], name='Tr acc output', legendgroup='group8', showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["vd_Tr"][:, 3], name='Tr acc output', legendgroup='group8', showlegend=False), row=4, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["vd_Tr"][:, 4], name='Tr acc output', legendgroup='group8', showlegend=False), row=5, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["vd_Tr"][:, 5], name='Tr acc output', legendgroup='group8', showlegend=False), row=6, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_f"][:, 0], name='Tr vel output f', legendgroup='group9'), row=1, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_f"][:, 1], name='Tr vel output f', legendgroup='group9', showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_f"][:, 2], name='Tr vel output f', legendgroup='group9', showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_f"][:, 3], name='Tr vel output f', legendgroup='group9', showlegend=False), row=4, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_f"][:, 4], name='Tr vel output f', legendgroup='group9', showlegend=False), row=5, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_f"][:, 5], name='Tr vel output f', legendgroup='group9', showlegend=False), row=6, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_v"][:, 0], name='Tr vel output v', legendgroup='group10'), row=1, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_v"][:, 1], name='Tr vel output v', legendgroup='group10', showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_v"][:, 2], name='Tr vel output v', legendgroup='group10', showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_v"][:, 3], name='Tr vel output v', legendgroup='group10', showlegend=False), row=4, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_v"][:, 4], name='Tr vel output v', legendgroup='group10', showlegend=False), row=5, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr_v"][:, 5], name='Tr vel output v', legendgroup='group10', showlegend=False), row=6, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr"][:, 0], name='Tr vel output', legendgroup='group11'), row=1, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr"][:, 1], name='Tr vel output', legendgroup='group11', showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr"][:, 2], name='Tr vel output', legendgroup='group11', showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr"][:, 3], name='Tr vel output', legendgroup='group11', showlegend=False), row=4, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr"][:, 4], name='Tr vel output', legendgroup='group11', showlegend=False), row=5, col=3)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["v_Tr"][:, 5], name='Tr vel output', legendgroup='group11', showlegend=False), row=6, col=3)
        
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_cmd"][:, 0], name='wrench T cmd', legendgroup='group12'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_cmd"][:, 1], name='wrench T cmd', legendgroup='group12', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_cmd"][:, 2], name='wrench T cmd', legendgroup='group12', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_cmd"][:, 3], name='wrench T cmd', legendgroup='group12', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_cmd"][:, 4], name='wrench T cmd', legendgroup='group12', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_cmd"][:, 5], name='wrench T cmd', legendgroup='group12', showlegend=False), row=6, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_fb"][:, 0], name='wrench T fb', legendgroup='group13'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_fb"][:, 1], name='wrench T fb', legendgroup='group13', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_fb"][:, 2], name='wrench T fb', legendgroup='group13', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_fb"][:, 3], name='wrench T fb', legendgroup='group13', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_fb"][:, 4], name='wrench T fb', legendgroup='group13', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_fb"][:, 5], name='wrench T fb', legendgroup='group13', showlegend=False), row=6, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err"][:, 0], name='wrench T Err', legendgroup='group14'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err"][:, 1], name='wrench T Err', legendgroup='group14', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err"][:, 2], name='wrench T Err', legendgroup='group14', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err"][:, 3], name='wrench T Err', legendgroup='group14', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err"][:, 4], name='wrench T Err', legendgroup='group14', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err"][:, 5], name='wrench T Err', legendgroup='group14', showlegend=False), row=6, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_prev"][:, 0], name='wrench T Err prev', legendgroup='group15'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_prev"][:, 1], name='wrench T Err prev', legendgroup='group15', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_prev"][:, 2], name='wrench T Err prev', legendgroup='group15', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_prev"][:, 3], name='wrench T Err prev', legendgroup='group15', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_prev"][:, 4], name='wrench T Err prev', legendgroup='group15', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_prev"][:, 5], name='wrench T Err prev', legendgroup='group15', showlegend=False), row=6, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_I"][:, 0], name='wrench T Err I', legendgroup='group16'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_I"][:, 1], name='wrench T Err I', legendgroup='group16', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_I"][:, 2], name='wrench T Err I', legendgroup='group16', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_I"][:, 3], name='wrench T Err I', legendgroup='group16', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_I"][:, 4], name='wrench T Err I', legendgroup='group16', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_Err_I"][:, 5], name='wrench T Err I', legendgroup='group16', showlegend=False), row=6, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_PID"][:, 0], name='wrench T PID', legendgroup='group17'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_PID"][:, 1], name='wrench T PID', legendgroup='group17', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_PID"][:, 2], name='wrench T PID', legendgroup='group17', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_PID"][:, 3], name='wrench T PID', legendgroup='group17', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_PID"][:, 4], name='wrench T PID', legendgroup='group17', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_PID"][:, 5], name='wrench T PID', legendgroup='group17', showlegend=False), row=6, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_spring"][:, 0], name='wrench T spring', legendgroup='group18'), row=1, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_spring"][:, 1], name='wrench T spring', legendgroup='group18', showlegend=False), row=2, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_spring"][:, 2], name='wrench T spring', legendgroup='group18', showlegend=False), row=3, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_spring"][:, 3], name='wrench T spring', legendgroup='group18', showlegend=False), row=4, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_spring"][:, 4], name='wrench T spring', legendgroup='group18', showlegend=False), row=5, col=4)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_T_spring"][:, 5], name='wrench T spring', legendgroup='group18', showlegend=False), row=6, col=4)
        
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_cmd"][:, 0], name='wrench Tr cmd', legendgroup='group19'), row=1, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_cmd"][:, 1], name='wrench Tr cmd', legendgroup='group19', showlegend=False), row=2, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_cmd"][:, 2], name='wrench Tr cmd', legendgroup='group19', showlegend=False), row=3, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_cmd"][:, 3], name='wrench Tr cmd', legendgroup='group19', showlegend=False), row=4, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_cmd"][:, 4], name='wrench Tr cmd', legendgroup='group19', showlegend=False), row=5, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_cmd"][:, 5], name='wrench Tr cmd', legendgroup='group19', showlegend=False), row=6, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_Err"][:, 0], name='wrench Tr Err', legendgroup='group21'), row=1, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_Err"][:, 1], name='wrench Tr Err', legendgroup='group21', showlegend=False), row=2, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_Err"][:, 2], name='wrench Tr Err', legendgroup='group21', showlegend=False), row=3, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_Err"][:, 3], name='wrench Tr Err', legendgroup='group21', showlegend=False), row=4, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_Err"][:, 4], name='wrench Tr Err', legendgroup='group21', showlegend=False), row=5, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_Err"][:, 5], name='wrench Tr Err', legendgroup='group21', showlegend=False), row=6, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_PID"][:, 0], name='wrench Tr PID', legendgroup='group22'), row=1, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_PID"][:, 1], name='wrench Tr PID', legendgroup='group22', showlegend=False), row=2, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_PID"][:, 2], name='wrench Tr PID', legendgroup='group22', showlegend=False), row=3, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_PID"][:, 3], name='wrench Tr PID', legendgroup='group22', showlegend=False), row=4, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_PID"][:, 4], name='wrench Tr PID', legendgroup='group22', showlegend=False), row=5, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_PID"][:, 5], name='wrench Tr PID', legendgroup='group22', showlegend=False), row=6, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_damping"][:, 0], name='wrench Tr damping', legendgroup='group23'), row=1, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_damping"][:, 1], name='wrench Tr damping', legendgroup='group23', showlegend=False), row=2, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_damping"][:, 2], name='wrench Tr damping', legendgroup='group23', showlegend=False), row=3, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_damping"][:, 3], name='wrench Tr damping', legendgroup='group23', showlegend=False), row=4, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_damping"][:, 4], name='wrench Tr damping', legendgroup='group23', showlegend=False), row=5, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_damping"][:, 5], name='wrench Tr damping', legendgroup='group23', showlegend=False), row=6, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_spring"][:, 0], name='wrench Tr spring', legendgroup='group24'), row=1, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_spring"][:, 1], name='wrench Tr spring', legendgroup='group24', showlegend=False), row=2, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_spring"][:, 2], name='wrench Tr spring', legendgroup='group24', showlegend=False), row=3, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_spring"][:, 3], name='wrench Tr spring', legendgroup='group24', showlegend=False), row=4, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_spring"][:, 4], name='wrench Tr spring', legendgroup='group24', showlegend=False), row=5, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_spring"][:, 5], name='wrench Tr spring', legendgroup='group24', showlegend=False), row=6, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_All"][:, 0], name='wrench Tr All', legendgroup='group25'), row=1, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_All"][:, 1], name='wrench Tr All', legendgroup='group25', showlegend=False), row=2, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_All"][:, 2], name='wrench Tr All', legendgroup='group25', showlegend=False), row=3, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_All"][:, 3], name='wrench Tr All', legendgroup='group25', showlegend=False), row=4, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_All"][:, 4], name='wrench Tr All', legendgroup='group25', showlegend=False), row=5, col=5)
        fig.add_trace(go.Scatter(x=time_s, y=self.logs["wrench_Tr_All"][:, 5], name='wrench Tr All', legendgroup='group25', showlegend=False), row=6, col=5)
        
        fig.update_layout(height=1200, width=2500, title_text="Admittance Control Logs")
        fig.update_layout(hovermode="x unified")

        fig.show()
        