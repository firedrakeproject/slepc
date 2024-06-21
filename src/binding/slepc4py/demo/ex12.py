# ------------------------------------------------------------------------
#   Tests use of setArbitrarySelection()
#   - selection criterion is the projection onto a precomputed eigenvector
# ------------------------------------------------------------------------

import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy

opts = PETSc.Options()
n = opts.getInt('n', 30)

# Create matrix tridiag([-1 0 -1])
A = PETSc.Mat(); A.create()
A.setSizes([n, n])
A.setFromOptions()
rstart, rend = A.getOwnershipRange()
for i in range(rstart, rend):
    if i>0: A[i, i-1] = -1
    if i<n-1: A[i, i+1] = -1
A.assemble()

E = SLEPc.EPS(); E.create()
E.setOperators(A)
E.setProblemType(SLEPc.EPS.ProblemType.HEP)
E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
E.setFromOptions()

# Solve eigenproblem and store some solution
E.solve()
nconv = E.getConverged()
Print = PETSc.Sys.Print
vw = PETSc.Viewer.STDOUT()
if nconv>0:
    sx, _ = A.createVecs()
    E.getEigenpair(0, sx)
    vw.pushFormat(PETSc.Viewer.Format.ASCII_INFO_DETAIL)
    E.errorView(viewer=vw)
    def myArbitrarySel(evalue, xr, xi, sx):
        return abs(xr.dot(sx))
    E.setArbitrarySelection(myArbitrarySel,sx)
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    E.solve()
    E.errorView(viewer=vw)
    vw.popFormat()
else:
    Print( "No eigenpairs converged" )

