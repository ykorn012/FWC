#======================================================================================================
# !/usr/bin/env python
# title          : GlobalW2W_FWC_Run.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2018-07-31
# version        : v0.8
# usage          : python FwcRun.py
# notes          : Reference Paper "An Approach for Factory-Wide Control Utilizing Virtual Metrology"
# python_version : v3.5.3
#======================================================================================================
from simulator.Global_FWC_P1_Simulator import Global_FWC_P1_Simulator
from simulator.Global_FWC_P2_Simulator import Global_FWC_P2_Simulator
import numpy as np

Tgt_p1 = np.array([0, 50])
A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

Tgt_p2 = np.array([0, 0])
A_p2 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p2 = np.array([[0, 0.05], [0, 0.05]])
C_p2 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p2 = np.array([[0.05, 0], [0, 0.05]])

def main():
    fwc_p1_r2r = Global_FWC_P1_Simulator(Tgt_p1, A_p1, d_p1, C_p1, 1000000)
    fwc_p1_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, isR2R=True)
    p1_r2r_VMOutput = fwc_p1_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, isR2R=True)

    fwc_p1_l2l = Global_FWC_P1_Simulator(Tgt_p1, A_p1, d_p1, C_p1, 100000)
    fwc_p1_l2l.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, isR2R=False)
    fwc_p1_l2l.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, isR2R=False)

    fwc_p2_r2r = Global_FWC_P2_Simulator(Tgt_p2, A_p2, d_p2, C_p2, None, 1000000)
    fwc_p2_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=None, isR2R=True)
    fwc_p2_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=None, isR2R=True)

    fwc_p2_pre_r2r = Global_FWC_P2_Simulator(Tgt_p2, A_p2, d_p2, C_p2, F_p2, 100000)
    fwc_p2_pre_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=p1_r2r_VMOutput, isR2R=True)
    fwc_p2_pre_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=p1_r2r_VMOutput, isR2R=True)

    fwc_p2_l2l = Global_FWC_P2_Simulator(Tgt_p2, A_p2, d_p2, C_p2, None, 5000000)
    fwc_p2_l2l.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=None, isR2R=False)
    fwc_p2_l2l.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=None, isR2R=False)

    fwc_p2_pre_r2r = Global_FWC_P2_Simulator(Tgt_p2, A_p2, d_p2, C_p2, F_p2, 100000)
    fwc_p2_pre_r2r.DoE_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=10, M=10, f=p1_r2r_VMOutput, isR2R=True)
    fwc_p2_pre_r2r.VM_Run(lamda_PLS=1, dEWMA_Wgt1=0.45, dEWMA_Wgt2=0.35, Z=20, M=10, f=p1_r2r_VMOutput, isR2R=True)

if __name__ == "__main__":
    main()
