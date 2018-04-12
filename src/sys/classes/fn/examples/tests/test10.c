/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test Phi functions.\n\n";

#include <slepcfn.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FN             phi0,phi1,phik,phicopy;
  PetscInt       k;
  PetscScalar    x,y,yp,tau,eta;
  char           strx[50],str[50];

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* phi_0(x) = exp(x) */
  ierr = FNCreate(PETSC_COMM_WORLD,&phi0);CHKERRQ(ierr);
  ierr = FNSetType(phi0,FNPHI);CHKERRQ(ierr);
  ierr = FNPhiSetIndex(phi0,0);CHKERRQ(ierr);
  ierr = FNView(phi0,NULL);CHKERRQ(ierr);
  x = 2.2;
  ierr = SlepcSNPrintfScalar(strx,50,x,PETSC_FALSE);CHKERRQ(ierr);
  ierr = FNEvaluateFunction(phi0,x,&y);CHKERRQ(ierr);
  ierr = FNEvaluateDerivative(phi0,x,&yp);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,y,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,yp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str);CHKERRQ(ierr);

  /* phi_1(x) = (exp(x)-1)/x with scaling factors eta*phi_1(tau*x) */
  ierr = FNCreate(PETSC_COMM_WORLD,&phi1);CHKERRQ(ierr);
  ierr = FNSetType(phi1,FNPHI);CHKERRQ(ierr);  /* default index should be 1 */
  tau = 0.2;
  eta = 1.3;
  ierr = FNSetScale(phi1,tau,eta);CHKERRQ(ierr);
  ierr = FNView(phi1,NULL);CHKERRQ(ierr);
  x = 2.2;
  ierr = SlepcSNPrintfScalar(strx,50,x,PETSC_FALSE);CHKERRQ(ierr);
  ierr = FNEvaluateFunction(phi1,x,&y);CHKERRQ(ierr);
  ierr = FNEvaluateDerivative(phi1,x,&yp);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,y,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,yp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str);CHKERRQ(ierr);

  /* phi_k(x) with index set from command-line arguments */
  ierr = FNCreate(PETSC_COMM_WORLD,&phik);CHKERRQ(ierr);
  ierr = FNSetType(phik,FNPHI);CHKERRQ(ierr);
  ierr = FNSetFromOptions(phik);CHKERRQ(ierr);

  ierr = FNDuplicate(phik,PETSC_COMM_WORLD,&phicopy);CHKERRQ(ierr);
  ierr = FNPhiGetIndex(phicopy,&k);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Index of phi function is %D\n",k);CHKERRQ(ierr);
  ierr = FNView(phicopy,NULL);CHKERRQ(ierr);
  x = 2.2;
  ierr = SlepcSNPrintfScalar(strx,50,x,PETSC_FALSE);CHKERRQ(ierr);
  ierr = FNEvaluateFunction(phicopy,x,&y);CHKERRQ(ierr);
  ierr = FNEvaluateDerivative(phicopy,x,&yp);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,y,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,yp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str);CHKERRQ(ierr);

  ierr = FNDestroy(&phi0);CHKERRQ(ierr);
  ierr = FNDestroy(&phi1);CHKERRQ(ierr);
  ierr = FNDestroy(&phik);CHKERRQ(ierr);
  ierr = FNDestroy(&phicopy);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
