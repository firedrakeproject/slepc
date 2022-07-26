/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nStandard eigenproblem with PRIMME, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetProblemType(eps,EPS_HEP));
  PetscCall(EPSSetType(eps,EPSPRIMME));

  /*
     Set several options
  */
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  PetscCall(EPSGetST(eps,&st));
  PetscCall(STSetType(st,STPRECOND));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(PCSetType(pc,PCICC));

  PetscCall(EPSPRIMMESetBlockSize(eps,4));
  PetscCall(EPSPRIMMESetMethod(eps,EPS_PRIMME_GD_OLSEN_PLUSK));
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Compute eigenvalues and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSolve(eps));
  PetscCall(EPSPRIMMEGetBlockSize(eps,&bs));
  PetscCall(EPSPRIMMEGetMethod(eps,&meth));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," PRIMME: using block size %" PetscInt_FMT ", method %s\n",bs,EPSPRIMMEMethods[meth]));

  PetscCall(EPSErrorView(eps,EPS_ERROR_ABSOLUTE,NULL));

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
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
