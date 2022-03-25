/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test exponential function.\n\n";

#include <slepcfn.h>

int main(int argc,char **argv)
{
  FN             fn,fncopy;
  PetscScalar    x,y,yp,tau,eta,alpha,beta;
  char           strx[50],str[50];

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetFromOptions(fn));

  /* plain exponential exp(x) */
  CHKERRQ(FNSetType(fn,FNEXP));
  CHKERRQ(FNView(fn,NULL));
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* exponential with scaling factors eta*exp(tau*x) */
  CHKERRQ(FNSetType(fn,FNEXP));
  tau = -0.2;
  eta = 1.3;
  CHKERRQ(FNSetScale(fn,tau,eta));
  CHKERRQ(FNView(fn,NULL));
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* test FNDuplicate */
  CHKERRQ(FNDuplicate(fn,PetscObjectComm((PetscObject)fn),&fncopy));

  /* test FNGetScale */
  CHKERRQ(FNGetScale(fncopy,&alpha,&beta));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Parameters:\n - alpha: "));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),alpha,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n - beta: "));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),beta,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));

  CHKERRQ(FNDestroy(&fn));
  CHKERRQ(FNDestroy(&fncopy));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      filter: grep -v "computing matrix functions"

TEST*/
