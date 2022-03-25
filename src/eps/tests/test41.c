/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface to external package EVSL.\n\n"
  "This is based on ex3.c. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;           /* matrix */
  EPS            eps;         /* eigenproblem solver context */
  PetscInt       N,n=20,m,Istart,Iend,II,i,j,nslice;
  PetscReal      a,b,ra,rb;
  PetscBool      flag;
  EPSEVSLDamping damping;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nStandard eigenproblem with EVSL, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

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
  CHKERRQ(EPSSetType(eps,EPSEVSL));

  /*
     Set several options
  */
  CHKERRQ(EPSSetInterval(eps,1.3,1.44));
  CHKERRQ(EPSEVSLSetRange(eps,0,8));
  CHKERRQ(EPSEVSLSetSlices(eps,3));
  CHKERRQ(EPSEVSLSetDamping(eps,EPS_EVSL_DAMPING_SIGMA));
  CHKERRQ(EPSEVSLSetDOSParameters(eps,EPS_EVSL_DOS_KPM,50,450,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(EPSEVSLSetPolParameters(eps,4000,0.85));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSEVSLGetSlices(eps,&nslice));
  CHKERRQ(EPSGetInterval(eps,&a,&b));
  CHKERRQ(EPSEVSLGetRange(eps,&ra,&rb));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," EVSL: solving interval [%g,%g] with %" PetscInt_FMT " slices (spectral range [%g,%g])\n",a,b,nslice,ra,rb));
  CHKERRQ(EPSEVSLGetDamping(eps,&damping));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," EVSL: damping type is %s\n",EPSEVSLDampings[damping]));

  CHKERRQ(EPSView(eps,NULL));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_ABSOLUTE,NULL));

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   build:
      requires: evsl

   test:
      requires: evsl
      args: -eps_evsl_damping none
      filter: grep -v ncv

TEST*/
