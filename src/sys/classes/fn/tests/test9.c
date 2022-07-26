/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test logarithm function.\n\n";

#include <slepcfn.h>

int main(int argc,char **argv)
{
  FN             fn;
  PetscScalar    x,y,yp,tau,eta;
  char           strx[50],str[50];

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fn));

  /* plain logarithm log(x) */
  PetscCall(FNSetType(fn,FNLOG));
  PetscCall(FNView(fn,NULL));
  x = 2.2;
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* logarithm with scaling factors eta*log(tau*x) */
  PetscCall(FNSetType(fn,FNLOG));
  tau = 0.2;
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

  PetscCall(FNDestroy(&fn));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      filter: grep -v "computing matrix functions"

TEST*/
