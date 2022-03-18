/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This example implements one of the problems found at
       NLEVP: A Collection of Nonlinear Eigenvalue Problems,
       The University of Manchester.
   The details of the collection can be found at:
       [1] T. Betcke et al., "NLEVP: A Collection of Nonlinear Eigenvalue
           Problems", ACM Trans. Math. Software 39(2), Article 7, 2013.

   The loaded_string problem is a rational eigenvalue problem for the
   finite element model of a loaded vibrating string.
*/

static char help[] = "Illustrates computation of left eigenvectors and resolvent.\n\n"
  "This is based on loaded_string from the NLEVP collection.\n"
  "The command line options are:\n"
  "  -n <n>, dimension of the matrices.\n"
  "  -kappa <kappa>, stiffness of elastic spring.\n"
  "  -mass <m>, mass of the attached load.\n\n";

#include <slepcnep.h>

#define NMAT 3

int main(int argc,char **argv)
{
  Mat            A[NMAT];         /* problem matrices */
  FN             f[NMAT];         /* functions to define the nonlinear operator */
  NEP            nep;             /* nonlinear eigensolver context */
  RG             rg;
  Vec            v,r,z,w;
  PetscInt       n=100,Istart,Iend,i,nconv;
  PetscReal      kappa=1.0,m=1.0,nrm,tol;
  PetscScalar    lambda,sigma,numer[2],denom[2],omega1,omega2;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mass",&m,NULL));
  sigma = kappa/m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Loaded vibrating string, n=%" PetscInt_FMT " kappa=%g m=%g\n\n",n,(double)kappa,(double)m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Build the problem matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* initialize matrices */
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[i]));
    CHKERRQ(MatSetSizes(A[i],PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(A[i]));
    CHKERRQ(MatSetUp(A[i]));
  }
  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));

  /* A0 */
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[0],i,i,(i==n-1)?1.0*n:2.0*n,INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[0],i,i-1,-1.0*n,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A[0],i,i+1,-1.0*n,INSERT_VALUES));
  }

  /* A1 */
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[1],i,i,(i==n-1)?2.0/(6.0*n):4.0/(6.0*n),INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[1],i,i-1,1.0/(6.0*n),INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A[1],i,i+1,1.0/(6.0*n),INSERT_VALUES));
  }

  /* A2 */
  if (Istart<=n-1 && n-1<Iend) {
    CHKERRQ(MatSetValue(A[2],n-1,n-1,kappa,INSERT_VALUES));
  }

  /* assemble matrices */
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY));
  }
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Create the problem functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* f1=1 */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
  CHKERRQ(FNSetType(f[0],FNRATIONAL));
  numer[0] = 1.0;
  CHKERRQ(FNRationalSetNumerator(f[0],1,numer));

  /* f2=-lambda */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
  CHKERRQ(FNSetType(f[1],FNRATIONAL));
  numer[0] = -1.0; numer[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f[1],2,numer));

  /* f3=lambda/(lambda-sigma) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[2]));
  CHKERRQ(FNSetType(f[2],FNRATIONAL));
  numer[0] = 1.0; numer[1] = 0.0;
  denom[0] = 1.0; denom[1] = -sigma;
  CHKERRQ(FNRationalSetNumerator(f[2],2,numer));
  CHKERRQ(FNRationalSetDenominator(f[2],2,denom));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetSplitOperator(nep,3,A,f,SUBSET_NONZERO_PATTERN));
  CHKERRQ(NEPSetProblemType(nep,NEP_RATIONAL));
  CHKERRQ(NEPSetDimensions(nep,8,PETSC_DEFAULT,PETSC_DEFAULT));

  /* set two-sided NLEIGS solver */
  CHKERRQ(NEPSetType(nep,NEPNLEIGS));
  CHKERRQ(NEPNLEIGSSetFullBasis(nep,PETSC_TRUE));
  CHKERRQ(NEPSetTwoSided(nep,PETSC_TRUE));
  CHKERRQ(NEPGetRG(nep,&rg));
  CHKERRQ(RGSetType(rg,RGINTERVAL));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(RGIntervalSetEndpoints(rg,4.0,700.0,-0.001,0.001));
#else
  CHKERRQ(RGIntervalSetEndpoints(rg,4.0,700.0,0,0));
#endif
  CHKERRQ(NEPSetTarget(nep,5.0));

  CHKERRQ(NEPSetFromOptions(nep));
  CHKERRQ(NEPSolve(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Check left residual
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreateVecs(A[0],&v,&r));
  CHKERRQ(VecDuplicate(v,&w));
  CHKERRQ(VecDuplicate(v,&z));
  CHKERRQ(NEPGetConverged(nep,&nconv));
  CHKERRQ(NEPGetTolerances(nep,&tol,NULL));
  for (i=0;i<nconv;i++) {
    CHKERRQ(NEPGetEigenpair(nep,i,&lambda,NULL,NULL,NULL));
    CHKERRQ(NEPGetLeftEigenvector(nep,i,v,NULL));
    CHKERRQ(NEPApplyAdjoint(nep,lambda,v,w,r,NULL,NULL));
    CHKERRQ(VecNorm(r,NORM_2,&nrm));
    if (nrm>tol*PetscAbsScalar(lambda)) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Left residual i=%" PetscInt_FMT " is above tolerance --> %g\n",i,(double)(nrm/PetscAbsScalar(lambda))));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Operate with resolvent
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  omega1 = 20.0;
  omega2 = 150.0;
  CHKERRQ(VecSet(v,0.0));
  CHKERRQ(VecSetValue(v,0,-1.0,INSERT_VALUES));
  CHKERRQ(VecSetValue(v,1,3.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(NEPApplyResolvent(nep,NULL,omega1,v,r));
  CHKERRQ(VecNorm(r,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"resolvent, omega=%g: norm of computed vector=%g\n",(double)PetscRealPart(omega1),(double)nrm));
  CHKERRQ(NEPApplyResolvent(nep,NULL,omega2,v,r));
  CHKERRQ(VecNorm(r,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"resolvent, omega=%g: norm of computed vector=%g\n",(double)PetscRealPart(omega2),(double)nrm));
  CHKERRQ(VecSet(v,1.0));
  CHKERRQ(NEPApplyResolvent(nep,NULL,omega1,v,r));
  CHKERRQ(VecNorm(r,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"resolvent, omega=%g: norm of computed vector=%g\n",(double)PetscRealPart(omega1),(double)nrm));
  CHKERRQ(NEPApplyResolvent(nep,NULL,omega2,v,r));
  CHKERRQ(VecNorm(r,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"resolvent, omega=%g: norm of computed vector=%g\n",(double)PetscRealPart(omega2),(double)nrm));

  /* clean up */
  CHKERRQ(NEPDestroy(&nep));
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatDestroy(&A[i]));
    CHKERRQ(FNDestroy(&f[i]));
  }
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&z));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      requires: !single

TEST*/
