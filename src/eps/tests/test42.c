/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Block-diagonal orthogonal eigenproblem.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -seed <s>, where <s> = seed for random number generation.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  EPS            eps;
  Mat            A;
  PetscRandom    rand;
  PetscScalar    val,c,s;
  PetscInt       n=30,i,seed=0x12345678;
  PetscMPIInt    rank;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nOrthogonal eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL));
  CHKERRQ(PetscRandomSetSeed(rand,seed));
  CHKERRQ(PetscRandomSeed(rand));
  CHKERRQ(PetscRandomSetInterval(rand,0,2*PETSC_PI));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  if (!rank) {
    for (i=0;i<n/2;i++) {
      CHKERRQ(PetscRandomGetValue(rand,&val));
      c = PetscCosReal(PetscRealPart(val));
      s = PetscSinReal(PetscRealPart(val));
      CHKERRQ(MatSetValue(A,2*i,2*i,c,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,2*i+1,2*i+1,c,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,2*i,2*i+1,s,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,2*i+1,2*i,-s,INSERT_VALUES));
    }
    if (n%2) CHKERRQ(MatSetValue(A,n-1,n-1,-1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL));
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      requires: complex !single
      args: -eps_type ciss -eps_all -rg_type ring -rg_ring_center 0 -rg_ring_radius 1 -rg_ring_width 0.05 -rg_ring_startangle .93 -rg_ring_endangle .07
      filter: sed -e "s/[+-]\([0-9]\.[0-9]*i\)/+-\\1/g"
      test:
         suffix: 1_ring
         args: -eps_ciss_extraction {{ritz hankel}}

TEST*/
