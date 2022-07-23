/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests the usage of object prefix.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  EPS            eps;         /* eigenproblem solver context */
  PetscInt       n=30,i,Istart,Iend;
  const char     *prefix;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(A,i,i,i+1,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create the eigensolver and mess up with the prefix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOptionsPrefix(eps,"check_"));
  PetscCall(EPSAppendOptionsPrefix(eps,"myprefix_"));
  PetscCall(EPSGetOptionsPrefix(eps,&prefix));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"EPS prefix is currently: %s\n\n",prefix));

  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetProblemType(eps,EPS_HEP));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -check_myprefix_eps_nev 2 -check_myprefix_st_type sinvert

TEST*/
