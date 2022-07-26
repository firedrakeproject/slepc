/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This example illustrates the use of Phi functions in exponential integrators.
   In particular, it implements the Norsett-Euler scheme of stiff order 1.

   The problem is the 1-D heat equation with source term

             y_t = y_xx + 1/(1+u^2) + psi

   where psi is chosen so that the exact solution is yex = x*(1-x)*exp(tend).
   The space domain is [0,1] and the time interval is [0,tend].

       [1] M. Hochbruck and A. Ostermann, "Explicit exponential Runge-Kutta
           methods for semilinear parabolic problems", SIAM J. Numer. Anal. 43(3),
           1069-1090, 2005.
*/

static char help[] = "Exponential integrator for the heat equation with source term.\n\n"
  "The command line options are:\n"
  "  -n <idim>, where <idim> = dimension of the spatial discretization.\n"
  "  -tend <rval>, where <rval> = real value that corresponding to the final time.\n"
  "  -deltat <rval>, where <rval> = real value for the time increment.\n"
  "  -combine <bool>, to represent the phi function with FNCOMBINE instead of FNPHI.\n\n";

#include <slepcmfn.h>

/*
   BuildFNPhi: builds an FNCOMBINE object representing the phi_1 function

        f(x) = (exp(x)-1)/x

   with the following tree:

            f(x)                  f(x)              (combined by division)
           /    \                 p(x) = x          (polynomial)
        a(x)    p(x)              a(x)              (combined by addition)
       /    \                     e(x) = exp(x)     (exponential)
     e(x)   c(x)                  c(x) = -1         (constant)
*/
PetscErrorCode BuildFNPhi(FN fphi)
{
  FN             fexp,faux,fconst,fpol;
  PetscScalar    coeffs[2];

  PetscFunctionBeginUser;
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fexp));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fconst));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&faux));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fpol));

  PetscCall(FNSetType(fexp,FNEXP));

  PetscCall(FNSetType(fconst,FNRATIONAL));
  coeffs[0] = -1.0;
  PetscCall(FNRationalSetNumerator(fconst,1,coeffs));

  PetscCall(FNSetType(faux,FNCOMBINE));
  PetscCall(FNCombineSetChildren(faux,FN_COMBINE_ADD,fexp,fconst));

  PetscCall(FNSetType(fpol,FNRATIONAL));
  coeffs[0] = 1.0; coeffs[1] = 0.0;
  PetscCall(FNRationalSetNumerator(fpol,2,coeffs));

  PetscCall(FNSetType(fphi,FNCOMBINE));
  PetscCall(FNCombineSetChildren(fphi,FN_COMBINE_DIVIDE,faux,fpol));

  PetscCall(FNDestroy(&faux));
  PetscCall(FNDestroy(&fpol));
  PetscCall(FNDestroy(&fconst));
  PetscCall(FNDestroy(&fexp));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat               L;
  Vec               u,w,z,yex;
  MFN               mfnexp,mfnphi;
  FN                fexp,fphi;
  PetscBool         combine=PETSC_FALSE;
  PetscInt          i,k,Istart,Iend,n=199,steps;
  PetscReal         t,tend=1.0,deltat=0.01,nrmd,nrmu,x,h;
  const PetscReal   half=0.5;
  PetscScalar       value,c,uval,*warray;
  const PetscScalar *uarray;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tend",&tend,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-deltat",&deltat,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-combine",&combine,NULL));
  h = 1.0/(n+1.0);
  c = (n+1)*(n+1);

  steps = (PetscInt)(tend/deltat);
  PetscCheck(PetscAbsReal(tend-steps*deltat)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example requires tend being a multiple of deltat");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nHeat equation via phi functions, n=%" PetscInt_FMT ", tend=%g, deltat=%g%s\n\n",n,(double)tend,(double)deltat,combine?" (combine)":""));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Build the 1-D Laplacian and various vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&L));
  PetscCall(MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(L));
  PetscCall(MatSetUp(L));
  PetscCall(MatGetOwnershipRange(L,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(L,i,i-1,c,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(L,i,i+1,c,INSERT_VALUES));
    PetscCall(MatSetValue(L,i,i,-2.0*c,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(L,NULL,&u));
  PetscCall(VecDuplicate(u,&yex));
  PetscCall(VecDuplicate(u,&w));
  PetscCall(VecDuplicate(u,&z));

  /*
     Compute various vectors:
     - the exact solution yex = x*(1-x)*exp(tend)
     - the initial condition u = abs(x-0.5)-0.5
  */
  for (i=Istart;i<Iend;i++) {
    x = (i+1)*h;
    value = x*(1.0-x)*PetscExpReal(tend);
    PetscCall(VecSetValue(yex,i,value,INSERT_VALUES));
    value = PetscAbsReal(x-half)-half;
    PetscCall(VecSetValue(u,i,value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(yex));
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(yex));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecViewFromOptions(yex,NULL,"-exact_sol"));
  PetscCall(VecViewFromOptions(u,NULL,"-initial_cond"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create two MFN solvers, for exp() and phi_1()
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MFNCreate(PETSC_COMM_WORLD,&mfnexp));
  PetscCall(MFNSetOperator(mfnexp,L));
  PetscCall(MFNGetFN(mfnexp,&fexp));
  PetscCall(FNSetType(fexp,FNEXP));
  PetscCall(FNSetScale(fexp,deltat,1.0));
  PetscCall(MFNSetErrorIfNotConverged(mfnexp,PETSC_TRUE));
  PetscCall(MFNSetFromOptions(mfnexp));

  PetscCall(MFNCreate(PETSC_COMM_WORLD,&mfnphi));
  PetscCall(MFNSetOperator(mfnphi,L));
  PetscCall(MFNGetFN(mfnphi,&fphi));
  if (combine) PetscCall(BuildFNPhi(fphi));
  else {
    PetscCall(FNSetType(fphi,FNPHI));
    PetscCall(FNPhiSetIndex(fphi,1));
  }
  PetscCall(FNSetScale(fphi,deltat,1.0));
  PetscCall(MFNSetErrorIfNotConverged(mfnphi,PETSC_TRUE));
  PetscCall(MFNSetFromOptions(mfnphi));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Solve the problem with the Norsett-Euler scheme
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  t = 0.0;
  for (k=0;k<steps;k++) {

    /* evaluate nonlinear part */
    PetscCall(VecGetArrayRead(u,&uarray));
    PetscCall(VecGetArray(w,&warray));
    for (i=Istart;i<Iend;i++) {
      x = (i+1)*h;
      uval = uarray[i-Istart];
      value = x*(1.0-x)*PetscExpReal(t);
      value = value + 2.0*PetscExpReal(t) - 1.0/(1.0+value*value);
      value = value + 1.0/(1.0+uval*uval);
      warray[i-Istart] = deltat*value;
    }
    PetscCall(VecRestoreArrayRead(u,&uarray));
    PetscCall(VecRestoreArray(w,&warray));
    PetscCall(MFNSolve(mfnphi,w,z));

    /* evaluate linear part */
    PetscCall(MFNSolve(mfnexp,u,u));
    PetscCall(VecAXPY(u,1.0,z));
    t = t + deltat;

  }
  PetscCall(VecViewFromOptions(u,NULL,"-computed_sol"));

  /*
     Compare with exact solution and show error norm
  */
  PetscCall(VecCopy(u,z));
  PetscCall(VecAXPY(z,-1.0,yex));
  PetscCall(VecNorm(z,NORM_2,&nrmd));
  PetscCall(VecNorm(u,NORM_2,&nrmu));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," The relative error at t=%g is %.4f\n\n",(double)t,(double)(nrmd/nrmu)));

  /*
     Free work space
  */
  PetscCall(MFNDestroy(&mfnexp));
  PetscCall(MFNDestroy(&mfnphi));
  PetscCall(MatDestroy(&L));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&yex));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&z));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -n 127 -tend 0.125 -mfn_tol 1e-3 -deltat 0.025
      timeoutfactor: 2

   test:
      suffix: 2
      args: -n 127 -tend 0.125 -mfn_tol 1e-3 -deltat 0.025 -combine
      filter: sed -e "s/ (combine)//"
      requires: !single
      output_file: output/ex39_1.out
      timeoutfactor: 2

TEST*/
