/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Diagonal eigenproblem. Illustrates use of shell preconditioner.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n"
  "  -seed <s>, where <s> = seed for random number generation.\n\n";

#include <slepceps.h>

PetscErrorCode PCApply_User(PC pc,Vec x,Vec y)
{
  PetscFunctionBeginUser;
  CHKERRQ(VecCopy(x,y));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  EPS            eps;         /* eigenproblem solver context */
  Vec            v0;          /* initial vector */
  PetscRandom    rand;
  PetscReal      tol=PETSC_SMALL;
  PetscInt       n=30,i,Istart,Iend,seed=0x12345678;
  PetscErrorCode ierr;
  ST             st;
  KSP            ksp;
  PC             pc;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A,i,i,i+1,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCSHELL));
  CHKERRQ(PCShellSetApply(pc,PCApply_User));

  /* set random initial vector */
  CHKERRQ(MatCreateVecs(A,&v0,NULL));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL));
  CHKERRQ(PetscRandomSetSeed(rand,seed));
  CHKERRQ(PetscRandomSeed(rand));
  CHKERRQ(VecSetRandom(v0,rand));
  CHKERRQ(EPSSetInitialSpace(eps,1,&v0));
  /* call the solver */
  CHKERRQ(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v0));
  CHKERRQ(PetscRandomDestroy(&rand));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      requires: !single
      output_file: output/test12_1.out
      test:
         suffix: 1
         args: -eps_type {{krylovschur subspace arnoldi gd jd}} -eps_nev 4
      test:
         suffix: 1_power
         args: -eps_type power -eps_max_it 10000 -eps_nev 4
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion -eps_nev 4

TEST*/
