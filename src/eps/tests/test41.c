/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nStandard eigenproblem with EVSL, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

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
  PetscCall(EPSSetType(eps,EPSEVSL));

  /*
     Set several options
  */
  PetscCall(EPSSetInterval(eps,1.3,1.44));
  PetscCall(EPSEVSLSetRange(eps,0,8));
  PetscCall(EPSEVSLSetSlices(eps,3));
  PetscCall(EPSEVSLSetDamping(eps,EPS_EVSL_DAMPING_SIGMA));
  PetscCall(EPSEVSLSetDOSParameters(eps,EPS_EVSL_DOS_KPM,50,450,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(EPSEVSLSetPolParameters(eps,4000,0.85));
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSolve(eps));
  PetscCall(EPSEVSLGetSlices(eps,&nslice));
  PetscCall(EPSGetInterval(eps,&a,&b));
  PetscCall(EPSEVSLGetRange(eps,&ra,&rb));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," EVSL: solving interval [%g,%g] with %" PetscInt_FMT " slices (spectral range [%g,%g])\n",a,b,nslice,ra,rb));
  PetscCall(EPSEVSLGetDamping(eps,&damping));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," EVSL: damping type is %s\n",EPSEVSLDampings[damping]));

  PetscCall(EPSView(eps,NULL));
  PetscCall(EPSErrorView(eps,EPS_ERROR_ABSOLUTE,NULL));

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
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
