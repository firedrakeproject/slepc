# ------------------------------------------------------------------------
#   Solve 1-D PDE
#            -u'' = lambda*u
#   on [0,1] subject to
#            u(0)=0, u'(1)=u(1)*lambda*kappa/(kappa-lambda)
# ------------------------------------------------------------------------

import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
from numpy import sqrt, sin

Print = PETSc.Sys.Print

class MyPDE(object):

    def __init__(self, kappa, h):
        self.kappa = kappa
        self.h     = h

    def formFunction(self, nep, mu, F, B):
        n, m = F.getSize()
        Istart, Iend = F.getOwnershipRange()
        i1 = Istart
        if Istart==0: i1 = i1 + 1
        i2 = Iend
        if Iend==n: i2 = i2 - 1
        h = self.h
        c = self.kappa/(mu-self.kappa)
        d = n

        # Interior grid points
        for i in range(i1,i2):
            val = -d-mu*h/6.0
            F[i,i-1] = val
            F[i,i]   = 2.0*(d-mu*h/3.0)
            F[i,i+1] = val

        # Boundary points
        if Istart==0:
            F[0,0] = 2.0*(d-mu*h/3.0)
            F[0,1] = -d-mu*h/6.0
        if Iend==n:
            F[n-1,n-2] = -d-mu*h/6.0
            F[n-1,n-1] = d-mu*h/3.0+mu*c

        F.assemble()
        if B != F: B.assemble()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def formJacobian(self, nep, mu, F):
        n, m = J.getSize()
        Istart, Iend = J.getOwnershipRange()
        i1 = Istart
        if Istart==0: i1 = i1 + 1
        i2 = Iend
        if Iend==n: i2 = i2 - 1
        h = self.h
        c = self.kappa/(mu-self.kappa)

        # Interior grid points
        for i in range(i1,i2):
            J[i,i-1] = -h/6.0
            J[i,i]   = -2.0*h/3.0
            J[i,i+1] = -h/6.0

        # Boundary points
        if Istart==0:
            J[0,0] = -2.0*h/3.0
            J[0,1] = -h/6.0
        if Iend==n:
            J[n-1,n-2] = -h/6.0
            J[n-1,n-1] = -h/3.0-c*c

        J.assemble()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def checkSolution(self, mu, y):
        nu = sqrt(mu)
        u = y.duplicate()
        n = u.getSize()
        Istart, Iend = J.getOwnershipRange()
        h = self.h
        for i in range(Istart,Iend):
            x = (i+1)*h
            u[i] = sin(nu*x);
        u.assemble()
        u.normalize()
        u.axpy(-1.0,y)
        return u.norm()

def FixSign(x):
    # Force the eigenfunction to be real and positive, since
    # some eigensolvers may return the eigenvector multiplied
    # by a complex number of modulus one.
    comm = x.getComm()
    rank = comm.getRank()
    n = 1 if rank == 0 else 0
    aux = PETSc.Vec().createMPI((n, PETSc.DECIDE), comm=comm)
    if rank == 0: aux[0] = x[0]
    aux.assemble()
    x0 = aux.sum()
    sign = x0/abs(x0)
    x.scale(1.0/sign)

opts = PETSc.Options()
n = opts.getInt('n', 128)
kappa = opts.getReal('kappa', 1.0)
pde = MyPDE(kappa, 1.0/n)

# Setup the solver
nep = SLEPc.NEP().create()
F = PETSc.Mat().create()
F.setSizes([n, n])
F.setType('aij')
F.setPreallocationNNZ(3)
nep.setFunction(pde.formFunction, F)

J = PETSc.Mat().create()
J.setSizes([n, n])
J.setType('aij')
J.setPreallocationNNZ(3)
nep.setJacobian(pde.formJacobian, J)

nep.setTolerances(tol=1e-9)
nep.setDimensions(1)
nep.setFromOptions()

# Solve the problem
x = F.createVecs('right')
x.set(1.0)
nep.setInitialSpace(x)
nep.solve()

its = nep.getIterationNumber()
Print("Number of iterations of the method: %i" % its)
sol_type = nep.getType()
Print("Solution method: %s" % sol_type)
nev, ncv, mpd = nep.getDimensions()
Print("")
Print("Subspace dimension: %i" % ncv)
tol, maxit = nep.getTolerances()
Print("Stopping condition: tol=%.4g" % tol)
Print("")

nconv = nep.getConverged()
Print( "Number of converged eigenpairs %d" % nconv )

if nconv > 0:
  Print()
  Print("        k              ||T(k)x||          error ")
  Print("----------------- ------------------ ------------------")
  for i in range(nconv):
    k = nep.getEigenpair(i, x)
    FixSign(x)
    res = nep.computeError(i)
    error = pde.checkSolution(k.real, x)
    if k.imag != 0.0:
      Print( " %9f%+9f j %12g     %12g" % (k.real, k.imag, res, error) )
    else:
      Print( " %12f       %12g     %12g" % (k.real, res, error) )
  Print()


