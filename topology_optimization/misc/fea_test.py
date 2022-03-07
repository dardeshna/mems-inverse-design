import os
import sys
import subprocess

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)

import FreeCAD
import FreeCADGui
import ObjectsFem
from femtools import ccxtools

temp_dir = os.path.join(base, "temp")

# new document
doc = FreeCAD.newDocument("Scripted_CalculiX_Cantilever3D")

# part
box_obj = doc.addObject('Part::Box', 'Box')
box_obj.Height = box_obj.Width = 1000
box_obj.Length = 8000

# see how our part looks like
if FreeCAD.GuiUp:
    FreeCADGui.ActiveDocument.activeView().viewAxonometric()
    FreeCADGui.SendMsgToActiveView("ViewFit")

# analysis
analysis_object = ObjectsFem.makeAnalysis(doc, "Analysis")

# solver (we gone use the well tested CcxTools solver object)
solver_object = ObjectsFem.makeSolverCalculixCcxTools(doc, "CalculiX")
solver_object.GeometricalNonlinearity = 'linear'
solver_object.ThermoMechSteadyState = True
solver_object.MatrixSolverType = 'default'
solver_object.IterationsControlParameterTimeUse = False
analysis_object.addObject(solver_object)

# material
material_object = ObjectsFem.makeMaterialSolid(doc, "MaterialSolid")
mat = material_object.Material
mat['Name'] = "Steel-Generic"
mat['YoungsModulus'] = "210000 MPa"
mat['PoissonRatio'] = "0.30"
mat['Density'] = "7900 kg/m^3"
material_object.Material = mat
analysis_object.addObject(material_object)

# fixed_constraint
fixed_constraint = ObjectsFem.makeConstraintFixed(doc, "FemConstraintFixed")
fixed_constraint.References = [(doc.Box, "Face1")]
analysis_object.addObject(fixed_constraint)

# force_constraint
force_constraint = ObjectsFem.makeConstraintForce(doc, "FemConstraintForce")
force_constraint.References = [(doc.Box, "Face2")]
force_constraint.Force = 9000000.0
force_constraint.Direction = (doc.Box, ["Edge5"])
force_constraint.Reversed = True
analysis_object.addObject(force_constraint)

mesh = doc.addObject('Fem::FemMeshShapeNetgenObject', 'FEMMeshNetgen')
mesh.Shape = doc.Box
mesh.MaxSize = 1000
mesh.Fineness = "Moderate"
mesh.Optimize = True
mesh.SecondOrder = True

doc.recompute()

analysis_object.addObject(mesh)

fea = ccxtools.FemToolsCcx()
fea.update_objects()
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
fea.setup_working_dir(temp_dir)
fea.setup_ccx()
message = fea.check_prerequisites()
if not message:
    fea.purge_results()
    fea.write_inp_file()
    subprocess.call(
        [fea.ccx_binary, "-i ", os.path.splitext(os.path.basename(fea.inp_file_name))[0]],
        cwd=temp_dir
    )
    fea.load_results()

if FreeCAD.GuiUp:
    for m in analysis_object.Group:
        if m.isDerivedFrom('Fem::FemResultObject'):
            result_object = m
            break

    result_object.Mesh.ViewObject.setNodeDisplacementByVectors(result_object.NodeNumbers, result_object.DisplacementVectors)
    result_object.Mesh.ViewObject.applyDisplacement(20)

if not FreeCAD.GuiUp:
    if os.path.exists(os.path.join(base, doc.Name)):
        os.remove(os.path.join(base, doc.Name))
    doc.saveAs(os.path.join(base, doc.Name))