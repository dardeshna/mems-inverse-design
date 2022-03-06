import os
import sys

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
os.chdir(base)

import FreeCAD
import ObjectsFem
from femtools import ccxtools

doc = FreeCAD.activeDocument()
analysis = ObjectsFem.makeAnalysis(doc, "Analysis")

solver = ObjectsFem.makeSolverCalculixCcxTools(doc, "CalculiX")
analysis.addObject(solver)

fea = ccxtools.FemToolsCcx(analysis=analysis,solver=solver)
fea.inp_file_name = 'accel_blank_quad'
fea.load_results()