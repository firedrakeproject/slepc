/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Compute all eigenvalues in an interval of a symmetric-definite problem.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B;         /* matrices */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  KSP            ksp;
  PC             pc;
  PetscInt       N,n=35,m,Istart,Iend,II,nev,i,j,k,*inertias;
  PetscBool      flag,showinertia=PETSC_TRUE;
  PetscReal      int0,int1,*shifts;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-showinertia",&showinertia,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSymmetric-definite problem with two intervals, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,4.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,II,II,2.0,INSERT_VALUES));
  }
  if (Istart==0) {
    PetscCall(MatSetValue(B,0,0,6.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,0,1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,1,0,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,1,1,1.0,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,B));
  PetscCall(EPSSetProblemType(eps,EPS_GHEP));

  /*
     Set first interval and other settings for spectrum slicing
  */
  PetscCall(EPSSetType(eps,EPSKRYLOVSCHUR));
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_ALL));
  PetscCall(EPSSetInterval(eps,1.1,1.3));
  PetscCall(EPSGetST(eps,&st));
  PetscCall(STSetType(st,STSINVERT));
  PetscCall(EPSKrylovSchurGetKSP(eps,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(PCSetType(pc,PCCHOLESKY));
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Solve for first interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
  PetscCall(EPSGetInterval(eps,&int0,&int1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));
  if (showinertia) {
    PetscCall(EPSKrylovSchurGetInertias(eps,&k,&shifts,&inertias));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Used %" PetscInt_FMT " shifts (inertia):\n",k));
    for (i=0;i<k;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
    PetscCall(PetscFree(shifts));
    PetscCall(PetscFree(inertias));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Solve for second interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSSetInterval(eps,1.499,1.6));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
  PetscCall(EPSGetInterval(eps,&int0,&int1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));
  if (showinertia) {
    PetscCall(EPSKrylovSchurGetInertias(eps,&k,&shifts,&inertias));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Used %" PetscInt_FMT " shifts (inertia):\n",k));
    for (i=0;i<k;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
    PetscCall(PetscFree(shifts));
    PetscCall(PetscFree(inertias));
  }

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -showinertia 0 -eps_error_relative
      requires: !single

TEST*/
