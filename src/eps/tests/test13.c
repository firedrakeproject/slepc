/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test EPSSetArbitrarySelection.\n\n";

#include <slepceps.h>

PetscErrorCode MyArbitrarySelection(PetscScalar eigr,PetscScalar eigi,Vec xr,Vec xi,PetscScalar *rr,PetscScalar *ri,void *ctx)
{
  Vec             xref = *(Vec*)ctx;

  PetscFunctionBeginUser;
  CHKERRQ(VecDot(xr,xref,rr));
  *rr = PetscAbsScalar(*rr);
  if (ri) *ri = 0.0;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrices */
  EPS            eps;         /* eigenproblem solver context */
  PetscScalar    seigr,seigi;
  PetscReal      tol=1000*PETSC_MACHINE_EPSILON;
  Vec            sxr,sxi;
  PetscInt       n=30,i,Istart,Iend,nconv;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTridiagonal with zero diagonal, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create matrix tridiag([-1 0 -1])
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Create the eigensolver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve eigenproblem and store some solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(MatCreateVecs(A,&sxr,NULL));
  CHKERRQ(MatCreateVecs(A,&sxi,NULL));
  CHKERRQ(EPSGetConverged(eps,&nconv));
  if (nconv>0) {
    CHKERRQ(EPSGetEigenpair(eps,0,&seigr,&seigi,sxr,sxi));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Solve eigenproblem using an arbitrary selection
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    CHKERRQ(EPSSetArbitrarySelection(eps,MyArbitrarySelection,&sxr));
    CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE));
    CHKERRQ(EPSSolve(eps));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Problem: no eigenpairs converged.\n"));

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(VecDestroy(&sxr));
  CHKERRQ(VecDestroy(&sxi));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -eps_max_it 5000 -st_pc_type jacobi
      output_file: output/test13_1.out
      filter: sed -e "s/-1.98975/-1.98974/"
      test:
         suffix: 1
         args: -eps_type {{krylovschur gd jd}}
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion
      test:
         suffix: 2
         args: -eps_non_hermitian -eps_type {{krylovschur gd jd}}
      test:
         suffix: 2_gd2
         args: -eps_non_hermitian -eps_type gd -eps_gd_double_expansion
         timeoutfactor: 2

TEST*/
