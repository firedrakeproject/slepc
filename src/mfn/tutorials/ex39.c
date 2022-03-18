/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fexp));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fconst));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&faux));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fpol));

  CHKERRQ(FNSetType(fexp,FNEXP));

  CHKERRQ(FNSetType(fconst,FNRATIONAL));
  coeffs[0] = -1.0;
  CHKERRQ(FNRationalSetNumerator(fconst,1,coeffs));

  CHKERRQ(FNSetType(faux,FNCOMBINE));
  CHKERRQ(FNCombineSetChildren(faux,FN_COMBINE_ADD,fexp,fconst));

  CHKERRQ(FNSetType(fpol,FNRATIONAL));
  coeffs[0] = 1.0; coeffs[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(fpol,2,coeffs));

  CHKERRQ(FNSetType(fphi,FNCOMBINE));
  CHKERRQ(FNCombineSetChildren(fphi,FN_COMBINE_DIVIDE,faux,fpol));

  CHKERRQ(FNDestroy(&faux));
  CHKERRQ(FNDestroy(&fpol));
  CHKERRQ(FNDestroy(&fconst));
  CHKERRQ(FNDestroy(&fexp));
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
  PetscErrorCode    ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tend",&tend,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-deltat",&deltat,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-combine",&combine,NULL));
  h = 1.0/(n+1.0);
  c = (n+1)*(n+1);

  steps = (PetscInt)(tend/deltat);
  PetscCheck(PetscAbsReal(tend-steps*deltat)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example requires tend being a multiple of deltat");
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nHeat equation via phi functions, n=%" PetscInt_FMT ", tend=%g, deltat=%g%s\n\n",n,(double)tend,(double)deltat,combine?" (combine)":""));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Build the 1-D Laplacian and various vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&L));
  CHKERRQ(MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(L));
  CHKERRQ(MatSetUp(L));
  CHKERRQ(MatGetOwnershipRange(L,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(L,i,i-1,c,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(L,i,i+1,c,INSERT_VALUES));
    CHKERRQ(MatSetValue(L,i,i,-2.0*c,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(L,NULL,&u));
  CHKERRQ(VecDuplicate(u,&yex));
  CHKERRQ(VecDuplicate(u,&w));
  CHKERRQ(VecDuplicate(u,&z));

  /*
     Compute various vectors:
     - the exact solution yex = x*(1-x)*exp(tend)
     - the initial condition u = abs(x-0.5)-0.5
  */
  for (i=Istart;i<Iend;i++) {
    x = (i+1)*h;
    value = x*(1.0-x)*PetscExpReal(tend);
    CHKERRQ(VecSetValue(yex,i,value,INSERT_VALUES));
    value = PetscAbsReal(x-half)-half;
    CHKERRQ(VecSetValue(u,i,value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(yex));
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(yex));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecViewFromOptions(yex,NULL,"-exact_sol"));
  CHKERRQ(VecViewFromOptions(u,NULL,"-initial_cond"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create two MFN solvers, for exp() and phi_1()
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfnexp));
  CHKERRQ(MFNSetOperator(mfnexp,L));
  CHKERRQ(MFNGetFN(mfnexp,&fexp));
  CHKERRQ(FNSetType(fexp,FNEXP));
  CHKERRQ(FNSetScale(fexp,deltat,1.0));
  CHKERRQ(MFNSetErrorIfNotConverged(mfnexp,PETSC_TRUE));
  CHKERRQ(MFNSetFromOptions(mfnexp));

  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfnphi));
  CHKERRQ(MFNSetOperator(mfnphi,L));
  CHKERRQ(MFNGetFN(mfnphi,&fphi));
  if (combine) {
    CHKERRQ(BuildFNPhi(fphi));
  } else {
    CHKERRQ(FNSetType(fphi,FNPHI));
    CHKERRQ(FNPhiSetIndex(fphi,1));
  }
  CHKERRQ(FNSetScale(fphi,deltat,1.0));
  CHKERRQ(MFNSetErrorIfNotConverged(mfnphi,PETSC_TRUE));
  CHKERRQ(MFNSetFromOptions(mfnphi));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Solve the problem with the Norsett-Euler scheme
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  t = 0.0;
  for (k=0;k<steps;k++) {

    /* evaluate nonlinear part */
    CHKERRQ(VecGetArrayRead(u,&uarray));
    CHKERRQ(VecGetArray(w,&warray));
    for (i=Istart;i<Iend;i++) {
      x = (i+1)*h;
      uval = uarray[i-Istart];
      value = x*(1.0-x)*PetscExpReal(t);
      value = value + 2.0*PetscExpReal(t) - 1.0/(1.0+value*value);
      value = value + 1.0/(1.0+uval*uval);
      warray[i-Istart] = deltat*value;
    }
    CHKERRQ(VecRestoreArrayRead(u,&uarray));
    CHKERRQ(VecRestoreArray(w,&warray));
    CHKERRQ(MFNSolve(mfnphi,w,z));

    /* evaluate linear part */
    CHKERRQ(MFNSolve(mfnexp,u,u));
    CHKERRQ(VecAXPY(u,1.0,z));
    t = t + deltat;

  }
  CHKERRQ(VecViewFromOptions(u,NULL,"-computed_sol"));

  /*
     Compare with exact solution and show error norm
  */
  CHKERRQ(VecCopy(u,z));
  CHKERRQ(VecAXPY(z,-1.0,yex));
  CHKERRQ(VecNorm(z,NORM_2,&nrmd));
  CHKERRQ(VecNorm(u,NORM_2,&nrmu));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," The relative error at t=%g is %.4f\n\n",(double)t,(double)(nrmd/nrmu)));

  /*
     Free work space
  */
  CHKERRQ(MFNDestroy(&mfnexp));
  CHKERRQ(MFNDestroy(&mfnphi));
  CHKERRQ(MatDestroy(&L));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&yex));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&z));
  ierr = SlepcFinalize();
  return ierr;
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
