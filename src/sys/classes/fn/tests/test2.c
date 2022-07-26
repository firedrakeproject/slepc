/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fn));
  PetscCall(FNSetFromOptions(fn));

  /* plain exponential exp(x) */
  PetscCall(FNSetType(fn,FNEXP));
  PetscCall(FNView(fn,NULL));
  x = 2.2;
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* exponential with scaling factors eta*exp(tau*x) */
  PetscCall(FNSetType(fn,FNEXP));
  tau = -0.2;
  eta = 1.3;
  PetscCall(FNSetScale(fn,tau,eta));
  PetscCall(FNView(fn,NULL));
  x = 2.2;
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* test FNDuplicate */
  PetscCall(FNDuplicate(fn,PetscObjectComm((PetscObject)fn),&fncopy));

  /* test FNGetScale */
  PetscCall(FNGetScale(fncopy,&alpha,&beta));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Parameters:\n - alpha: "));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),alpha,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n - beta: "));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),beta,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));

  PetscCall(FNDestroy(&fn));
  PetscCall(FNDestroy(&fncopy));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      filter: grep -v "computing matrix functions"

TEST*/
