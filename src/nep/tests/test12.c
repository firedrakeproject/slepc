/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test some NLEIGS interface functions.\n\n"
  "Based on ex27.c. The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n";

/*
   Solve T(lambda)x=0 using NLEIGS solver
      with T(lambda) = -D+sqrt(lambda)*I
      where D is the Laplacian operator in 1 dimension
      and with the interpolation interval [.01,16]
*/

#include <slepcnep.h>

/*
   User-defined routines
*/
PetscErrorCode ComputeSingularities(NEP,PetscInt*,PetscScalar*,void*);

int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            A[2];
  PetscInt       n=100,Istart,Iend,i,ns,nsin;
  PetscErrorCode ierr;
  PetscBool      terse,fb;
  RG             rg;
  FN             f[2];
  PetscScalar    coeffs,shifts[]={1.06,1.1,1.12,1.15},*rkshifts,val;
  PetscErrorCode (*fsing)(NEP,PetscInt*,PetscScalar*,void*);

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSquare root eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetType(nep,NEPNLEIGS));
  CHKERRQ(NEPNLEIGSSetSingularitiesFunction(nep,ComputeSingularities,NULL));
  CHKERRQ(NEPGetRG(nep,&rg));
  CHKERRQ(RGSetType(rg,RGINTERVAL));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(RGIntervalSetEndpoints(rg,0.01,16.0,-0.001,0.001));
#else
  CHKERRQ(RGIntervalSetEndpoints(rg,0.01,16.0,0,0));
#endif
  CHKERRQ(NEPSetTarget(nep,1.1));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define the nonlinear problem in split form
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Create matrices */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[0]));
  CHKERRQ(MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[0]));
  CHKERRQ(MatSetUp(A[0]));
  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A[0],i,i-1,1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A[0],i,i+1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A[0],i,i,-2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,&A[1]));

  /* Define functions */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
  CHKERRQ(FNSetType(f[0],FNRATIONAL));
  coeffs = 1.0;
  CHKERRQ(FNRationalSetNumerator(f[0],1,&coeffs));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
  CHKERRQ(FNSetType(f[1],FNSQRT));
  CHKERRQ(NEPSetSplitOperator(nep,2,A,f,SUBSET_NONZERO_PATTERN));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPNLEIGSSetFullBasis(nep,PETSC_FALSE));
  CHKERRQ(NEPNLEIGSSetRKShifts(nep,4,shifts));
  CHKERRQ(NEPSetFromOptions(nep));

  CHKERRQ(NEPNLEIGSGetFullBasis(nep,&fb));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using full basis = %s\n",fb?"true":"false"));
  CHKERRQ(NEPNLEIGSGetRKShifts(nep,&ns,&rkshifts));
  if (ns) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using %" PetscInt_FMT " RK shifts =",ns));
    for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %g",(double)PetscRealPart(rkshifts[i])));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(PetscFree(rkshifts));
  }
  CHKERRQ(NEPNLEIGSGetSingularitiesFunction(nep,&fsing,NULL));
  nsin = 1;
  CHKERRQ((*fsing)(nep,&nsin,&val,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," First returned singularity = %g\n",(double)PetscRealPart(val)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(NEPSolve(nep));

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(NEPErrorView(nep,NEP_ERROR_BACKWARD,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(NEPDestroy(&nep));
  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(FNDestroy(&f[0]));
  CHKERRQ(FNDestroy(&f[1]));
  ierr = SlepcFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */
/*
   ComputeSingularities - Computes maxnp points (at most) in the complex plane where
   the function T(.) is not analytic.

   In this case, we discretize the singularity region (-inf,0)~(-10e+6,-10e-6)
*/
PetscErrorCode ComputeSingularities(NEP nep,PetscInt *maxnp,PetscScalar *xi,void *pt)
{
  PetscReal h;
  PetscInt  i;

  PetscFunctionBeginUser;
  h = 11.0/(*maxnp-1);
  xi[0] = -1e-5; xi[*maxnp-1] = -1e+6;
  for (i=1;i<*maxnp-1;i++) xi[i] = -PetscPowReal(10,-5+h*i);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -nep_nev 3 -nep_nleigs_interpolation_degree 20 -terse -nep_view
      requires: double
      filter: grep -v tolerance | sed -e "s/[+-]0\.0*i//g"

TEST*/
