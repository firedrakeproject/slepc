/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface functions of spectrum-slicing STOAR.\n\n"
  "This is based on ex38.c. The command line options are:\n"
  "  -n <n> ... dimension of the matrices.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3]; /* problem matrices */
  PEP            pep;        /* polynomial eigenproblem solver context */
  ST             st;         /* spectral transformation context */
  KSP            ksp;
  PC             pc;
  PetscBool      showinertia=PETSC_TRUE,lock,detect,checket;
  PetscInt       n=100,Istart,Iend,i,*inertias,ns,nev,ncv,mpd;
  PetscReal      mu=1.0,tau=10.0,kappa=5.0,int0,int1,*shifts;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-showinertia",&showinertia,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum slicing on PEP, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is a tridiagonal */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&K));
  CHKERRQ(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(K));
  CHKERRQ(MatSetUp(K));

  CHKERRQ(MatGetOwnershipRange(K,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(K,i,i-1,-kappa,INSERT_VALUES));
    CHKERRQ(MatSetValue(K,i,i,kappa*3.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(K,i,i+1,-kappa,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is a tridiagonal */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(C,i,i-1,-tau,INSERT_VALUES));
    CHKERRQ(MatSetValue(C,i,i,tau*3.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(C,i,i+1,-tau,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&M));
  CHKERRQ(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(M));
  CHKERRQ(MatSetUp(M));
  CHKERRQ(MatGetOwnershipRange(M,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(M,i,i,mu,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPSetProblemType(pep,PEP_HYPERBOLIC));
  CHKERRQ(PEPSetType(pep,PEPSTOAR));

  /*
     Set interval and other settings for spectrum slicing
  */
  int0 = -11.3;
  int1 = -9.5;
  CHKERRQ(PEPSetInterval(pep,int0,int1));
  CHKERRQ(PEPSetWhichEigenpairs(pep,PEP_ALL));
  CHKERRQ(PEPGetST(pep,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));

  /*
     Test interface functions of STOAR solver
  */
  CHKERRQ(PEPSTOARGetDetectZeros(pep,&detect));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Detect zeros before changing = %d",(int)detect));
  CHKERRQ(PEPSTOARSetDetectZeros(pep,PETSC_TRUE));
  CHKERRQ(PEPSTOARGetDetectZeros(pep,&detect));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)detect));

  CHKERRQ(PEPSTOARGetLocking(pep,&lock));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Locking flag before changing = %d",(int)lock));
  CHKERRQ(PEPSTOARSetLocking(pep,PETSC_TRUE));
  CHKERRQ(PEPSTOARGetLocking(pep,&lock));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)lock));

  CHKERRQ(PEPSTOARGetCheckEigenvalueType(pep,&checket));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Check eigenvalue type flag before changing = %d",(int)checket));
  CHKERRQ(PEPSTOARSetCheckEigenvalueType(pep,PETSC_FALSE));
  CHKERRQ(PEPSTOARGetCheckEigenvalueType(pep,&checket));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)checket));

  CHKERRQ(PEPSTOARGetDimensions(pep,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Sub-solve dimensions before changing = [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]",nev,ncv,mpd));
  CHKERRQ(PEPSTOARSetDimensions(pep,30,60,60));
  CHKERRQ(PEPSTOARGetDimensions(pep,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]\n",nev,ncv,mpd));

  CHKERRQ(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPSetUp(pep));
  CHKERRQ(PEPSTOARGetInertias(pep,&ns,&shifts,&inertias));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Inertias (after setup):\n"));
  for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
  CHKERRQ(PetscFree(shifts));
  CHKERRQ(PetscFree(inertias));

  CHKERRQ(PEPSolve(pep));
  CHKERRQ(PEPGetDimensions(pep,&nev,NULL,NULL));
  CHKERRQ(PEPGetInterval(pep,&int0,&int1));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));

  if (showinertia) {
    CHKERRQ(PEPSTOARGetInertias(pep,&ns,&shifts,&inertias));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Used %" PetscInt_FMT " shifts (inertia):\n",ns));
    for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
    CHKERRQ(PetscFree(shifts));
    CHKERRQ(PetscFree(inertias));
  }

  CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&K));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      requires: !single
      args: -showinertia 0

TEST*/
