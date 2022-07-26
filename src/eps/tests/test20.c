/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to EPSSolve changing ncv.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;
  EPS            eps;
  PetscReal      tol=PetscMax(1000*PETSC_MACHINE_EPSILON,1e-9);
  PetscInt       n=30,i,Istart,Iend,nev,ncv;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the solver, call EPSSolve() twice
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetProblemType(eps,EPS_HEP));
  PetscCall(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  PetscCall(EPSSetFromOptions(eps));

  /* First solve */
  PetscCall(EPSSolve(eps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," - - - First solve, default subspace dimension - - -\n"));
  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  /* Second solve */
  PetscCall(EPSGetDimensions(eps,&nev,&ncv,NULL));
  PetscCall(EPSSetDimensions(eps,nev,ncv+2,PETSC_DEFAULT));
  PetscCall(EPSSolve(eps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," - - - Second solve, subspace of increased size - - -\n"));
  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -n 18 -eps_type {{krylovschur arnoldi gd jd rqcg lobpcg lapack}} -eps_max_it 1500
      output_file: output/test20_1.out

   test:
      suffix: 1_lanczos
      args: -n 18 -eps_type lanczos -eps_lanczos_reorthog full -eps_max_it 1500
      output_file: output/test20_1.out

TEST*/
