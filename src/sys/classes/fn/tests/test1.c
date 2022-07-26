/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test rational function.\n\n";

#include <slepcfn.h>

int main(int argc,char **argv)
{
  FN             fn;
  PetscInt       i,na,nb;
  PetscScalar    x,y,yp,p[10],q[10],five=5.0,*pp,*qq;
  char           strx[50],str[50];

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fn));

  /* polynomial p(x) */
  na = 5;
  p[0] = -3.1; p[1] = 1.1; p[2] = 1.0; p[3] = -2.0; p[4] = 3.5;
  PetscCall(FNSetType(fn,FNRATIONAL));
  PetscCall(FNRationalSetNumerator(fn,na,p));
  PetscCall(FNView(fn,NULL));
  x = 2.2;
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* inverse of polynomial 1/q(x) */
  nb = 3;
  q[0] = -3.1; q[1] = 1.1; q[2] = 1.0;
  PetscCall(FNSetType(fn,FNRATIONAL));
  PetscCall(FNRationalSetNumerator(fn,0,NULL));  /* reset previous values */
  PetscCall(FNRationalSetDenominator(fn,nb,q));
  PetscCall(FNView(fn,NULL));
  x = 2.2;
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* rational p(x)/q(x) */
  na = 2; nb = 3;
  p[0] = 1.1; p[1] = 1.1;
  q[0] = 1.0; q[1] = -2.0; q[2] = 3.5;
  PetscCall(FNSetType(fn,FNRATIONAL));
  PetscCall(FNRationalSetNumerator(fn,na,p));
  PetscCall(FNRationalSetDenominator(fn,nb,q));
  PetscCall(FNSetScale(fn,1.2,0.5));
  PetscCall(FNView(fn,NULL));
  x = 2.2;
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  PetscCall(FNRationalGetNumerator(fn,&na,&pp));
  PetscCall(FNRationalGetDenominator(fn,&nb,&qq));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Coefficients:\n  Numerator: "));
  for (i=0;i<na;i++) {
    PetscCall(SlepcSNPrintfScalar(str,sizeof(str),pp[i],PETSC_FALSE));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n  Denominator: "));
  for (i=0;i<nb;i++) {
    PetscCall(SlepcSNPrintfScalar(str,sizeof(str),qq[i],PETSC_FALSE));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  PetscCall(PetscFree(pp));
  PetscCall(PetscFree(qq));

  /* constant */
  PetscCall(FNSetType(fn,FNRATIONAL));
  PetscCall(FNRationalSetNumerator(fn,1,&five));
  PetscCall(FNRationalSetDenominator(fn,0,NULL));  /* reset previous values */
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

TEST*/
