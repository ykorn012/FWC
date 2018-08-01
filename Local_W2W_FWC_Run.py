#======================================================================================================
# !/usr/bin/env python
# title          : LocalW2W_FWC_Run.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2018-07-31
# version        : v0.8
# usage          : python LocalW2W_FWC_Run.py
# notes          : Reference Paper "Virtual metrology and feedback control for semiconductor manufacturing"
# python_version : v3.5.3
#======================================================================================================
from simulator.Local_FWC_Simulator import Local_FWC_Simulator
import numpy as np

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])
d_p1 = np.array([[0.1, 0], [0.05, 0]])
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]]))

def main():
    fwc_p1_vm = Local_FWC_Simulator(A_p1, d_p1, C_p1, 1000000)
    fwc_p1_vm.DoE_Run(Z=12, M=10)
    fwc_p1_vm.VM_Run(lamda_PLS=0.1, Z=40, M=10)


if __name__ == "__main__":
    main()
