/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface to external package PRIMME.\n\n"
  "This is based on ex3.c. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat             A;           /* matrix */
  EPS             eps;         /* eigenproblem solver context */
  ST              st;          /* spectral transformation context */
  KSP             ksp;
  PC              pc;
  PetscInt        N,n=35,m,Istart,Iend,II,i,j,bs;
  PetscBool       flag;
  EPSPRIMMEMethod meth;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nStandard eigenproblem with PRIMME, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetType(eps,EPSPRIMME));

  /*
     Set several options
  */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STPRECOND));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(PCSetType(pc,PCICC));

  CHKERRQ(EPSPRIMMESetBlockSize(eps,4));
  CHKERRQ(EPSPRIMMESetMethod(eps,EPS_PRIMME_GD_OLSEN_PLUSK));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Compute eigenvalues and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSPRIMMEGetBlockSize(eps,&bs));
  CHKERRQ(EPSPRIMMEGetMethod(eps,&meth));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," PRIMME: using block size %" PetscInt_FMT ", method %s\n",bs,EPSPRIMMEMethods[meth]));

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_ABSOLUTE,NULL));

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   build:
      requires: primme

   testset:
      args: -eps_nev 4
      requires: primme
      output_file: output/test34_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         args: -st_pc_type bjacobi -eps_target 0.01 -eps_target_real -eps_refined
         nsize: 2
      test:
         suffix: 3
         args: -eps_smallest_magnitude -eps_harmonic

TEST*/
