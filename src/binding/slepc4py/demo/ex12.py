# Tests use of setArbitrarySelection()

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
A.setUp()
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
    sxr, sxi = A.createVecs()
    E.getEigenpair(0, sxr, sxi)
    vw.pushFormat(PETSc.Viewer.Format.ASCII_INFO_DETAIL)
    E.errorView(viewer=vw)
    def myArbitrarySel(evalue, xr, xi, sxr):
        return abs(xr.dot(sxr))
    E.setArbitrarySelection(myArbitrarySel,sxr)
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    E.solve()
    E.errorView(viewer=vw)
    vw.popFormat()
else:
    Print( "No eigenpairs converged" )

