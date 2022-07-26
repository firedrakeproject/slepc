/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
   This example solves the loaded_string problem by first transforming
   it to a quadratic eigenvalue problem.
*/

static char help[] = "Finite element model of a loaded vibrating string.\n\n"
  "The command line options are:\n"
  "  -n <n>, dimension of the matrices.\n"
  "  -kappa <kappa>, stiffness of elastic spring.\n"
  "  -mass <m>, mass of the attached load.\n\n";

#include <slepcpep.h>

#define NMAT 3

int main(int argc,char **argv)
{
  Mat            A[3],M;      /* problem matrices */
  PEP            pep;         /* polynomial eigenproblem solver context */
  PetscInt       n=100,Istart,Iend,i;
  PetscBool      terse;
  PetscReal      kappa=1.0,m=1.0;
  PetscScalar    sigma;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mass",&m,NULL));
  sigma = kappa/m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Loaded vibrating string (QEP), n=%" PetscInt_FMT " kappa=%g m=%g\n\n",n,(double)kappa,(double)m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* initialize matrices */
  for (i=0;i<NMAT;i++) {
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A[i]));
    PetscCall(MatSetSizes(A[i],PETSC_DECIDE,PETSC_DECIDE,n,n));
    PetscCall(MatSetFromOptions(A[i]));
    PetscCall(MatSetUp(A[i]));
  }
  PetscCall(MatGetOwnershipRange(A[0],&Istart,&Iend));

  /* A0 */
  for (i=Istart;i<Iend;i++) {
    PetscCall(MatSetValue(A[0],i,i,(i==n-1)?1.0*n:2.0*n,INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[0],i,i-1,-1.0*n,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A[0],i,i+1,-1.0*n,INSERT_VALUES));
  }

  /* A1 */
  for (i=Istart;i<Iend;i++) {
    PetscCall(MatSetValue(A[1],i,i,(i==n-1)?2.0/(6.0*n):4.0/(6.0*n),INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[1],i,i-1,1.0/(6.0*n),INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A[1],i,i+1,1.0/(6.0*n),INSERT_VALUES));
  }

  /* A2 */
  if (Istart<=n-1 && n-1<Iend) PetscCall(MatSetValue(A[2],n-1,n-1,kappa,INSERT_VALUES));

  /* assemble matrices */
  for (i=0;i<NMAT;i++) PetscCall(MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY));
  for (i=0;i<NMAT;i++) PetscCall(MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY));

  /* build matrices for the QEP */
  PetscCall(MatAXPY(A[2],1.0,A[0],DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAXPY(A[2],sigma,A[1],SAME_NONZERO_PATTERN));
  PetscCall(MatScale(A[2],-1.0));
  PetscCall(MatScale(A[0],sigma));
  M = A[1];
  A[1] = A[2];
  A[2] = M;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetFromOptions(pep));
  PetscCall(PEPSolve(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PEPDestroy(&pep));
  for (i=0;i<NMAT;i++) PetscCall(MatDestroy(&A[i]));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -pep_hyperbolic -pep_interval 4,900 -pep_type stoar -st_type sinvert -st_pc_type cholesky -terse
      requires: !single

TEST*/
