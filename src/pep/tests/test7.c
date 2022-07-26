/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-showinertia",&showinertia,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum slicing on PEP, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is a tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&K));
  PetscCall(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(K));
  PetscCall(MatSetUp(K));

  PetscCall(MatGetOwnershipRange(K,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(K,i,i-1,-kappa,INSERT_VALUES));
    PetscCall(MatSetValue(K,i,i,kappa*3.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(K,i,i+1,-kappa,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is a tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(C,i,i-1,-tau,INSERT_VALUES));
    PetscCall(MatSetValue(C,i,i,tau*3.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(C,i,i+1,-tau,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&M));
  PetscCall(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(M));
  PetscCall(MatSetUp(M));
  PetscCall(MatGetOwnershipRange(M,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(M,i,i,mu,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetProblemType(pep,PEP_HYPERBOLIC));
  PetscCall(PEPSetType(pep,PEPSTOAR));

  /*
     Set interval and other settings for spectrum slicing
  */
  int0 = -11.3;
  int1 = -9.5;
  PetscCall(PEPSetInterval(pep,int0,int1));
  PetscCall(PEPSetWhichEigenpairs(pep,PEP_ALL));
  PetscCall(PEPGetST(pep,&st));
  PetscCall(STSetType(st,STSINVERT));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCCHOLESKY));

  /*
     Test interface functions of STOAR solver
  */
  PetscCall(PEPSTOARGetDetectZeros(pep,&detect));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Detect zeros before changing = %d",(int)detect));
  PetscCall(PEPSTOARSetDetectZeros(pep,PETSC_TRUE));
  PetscCall(PEPSTOARGetDetectZeros(pep,&detect));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)detect));

  PetscCall(PEPSTOARGetLocking(pep,&lock));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Locking flag before changing = %d",(int)lock));
  PetscCall(PEPSTOARSetLocking(pep,PETSC_TRUE));
  PetscCall(PEPSTOARGetLocking(pep,&lock));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)lock));

  PetscCall(PEPSTOARGetCheckEigenvalueType(pep,&checket));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Check eigenvalue type flag before changing = %d",(int)checket));
  PetscCall(PEPSTOARSetCheckEigenvalueType(pep,PETSC_FALSE));
  PetscCall(PEPSTOARGetCheckEigenvalueType(pep,&checket));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)checket));

  PetscCall(PEPSTOARGetDimensions(pep,&nev,&ncv,&mpd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Sub-solve dimensions before changing = [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]",nev,ncv,mpd));
  PetscCall(PEPSTOARSetDimensions(pep,30,60,60));
  PetscCall(PEPSTOARGetDimensions(pep,&nev,&ncv,&mpd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]\n",nev,ncv,mpd));

  PetscCall(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPSetUp(pep));
  PetscCall(PEPSTOARGetInertias(pep,&ns,&shifts,&inertias));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Inertias (after setup):\n"));
  for (i=0;i<ns;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
  PetscCall(PetscFree(shifts));
  PetscCall(PetscFree(inertias));

  PetscCall(PEPSolve(pep));
  PetscCall(PEPGetDimensions(pep,&nev,NULL,NULL));
  PetscCall(PEPGetInterval(pep,&int0,&int1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));

  if (showinertia) {
    PetscCall(PEPSTOARGetInertias(pep,&ns,&shifts,&inertias));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Used %" PetscInt_FMT " shifts (inertia):\n",ns));
    for (i=0;i<ns;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
    PetscCall(PetscFree(shifts));
    PetscCall(PetscFree(inertias));
  }

  PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PEPDestroy(&pep));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&K));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      requires: !single
      args: -showinertia 0

TEST*/
