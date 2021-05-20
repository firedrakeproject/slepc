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
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nStandard eigenproblem with EVSL, N=%D (%Dx%D grid)\n\n",N,n,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) { ierr = MatSetValue(A,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetType(eps,EPSEVSL);CHKERRQ(ierr);

  /*
     Set several options
  */
  ierr = EPSSetInterval(eps,1.3,1.44);CHKERRQ(ierr);
  ierr = EPSEVSLSetRange(eps,0,8);CHKERRQ(ierr);
  ierr = EPSEVSLSetSlices(eps,3);CHKERRQ(ierr);
  ierr = EPSEVSLSetDamping(eps,EPS_EVSL_DAMPING_SIGMA);CHKERRQ(ierr);
  ierr = EPSEVSLSetDOSParameters(eps,EPS_EVSL_DOS_KPM,50,450,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSEVSLSetPolParameters(eps,4000,0.85);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSEVSLGetSlices(eps,&nslice);CHKERRQ(ierr);
  ierr = EPSGetInterval(eps,&a,&b);CHKERRQ(ierr);
  ierr = EPSEVSLGetRange(eps,&ra,&rb);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," EVSL: solving interval [%g,%g] with %D slices (spectral range [%g,%g])\n",a,b,nslice,ra,rb);CHKERRQ(ierr);
  ierr = EPSEVSLGetDamping(eps,&damping);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," EVSL: damping type is %s\n",EPSEVSLDampings[damping]);CHKERRQ(ierr);

  ierr = EPSView(eps,NULL);CHKERRQ(ierr);
  ierr = EPSErrorView(eps,EPS_ERROR_ABSOLUTE,NULL);CHKERRQ(ierr);

  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   build:
      requires: evsl

   test:
      requires: evsl
      args: -eps_evsl_damping none
      filter: grep -v ncv

TEST*/
