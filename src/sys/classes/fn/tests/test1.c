/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));

  /* polynomial p(x) */
  na = 5;
  p[0] = -3.1; p[1] = 1.1; p[2] = 1.0; p[3] = -2.0; p[4] = 3.5;
  CHKERRQ(FNSetType(fn,FNRATIONAL));
  CHKERRQ(FNRationalSetNumerator(fn,na,p));
  CHKERRQ(FNView(fn,NULL));
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* inverse of polynomial 1/q(x) */
  nb = 3;
  q[0] = -3.1; q[1] = 1.1; q[2] = 1.0;
  CHKERRQ(FNSetType(fn,FNRATIONAL));
  CHKERRQ(FNRationalSetNumerator(fn,0,NULL));  /* reset previous values */
  CHKERRQ(FNRationalSetDenominator(fn,nb,q));
  CHKERRQ(FNView(fn,NULL));
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* rational p(x)/q(x) */
  na = 2; nb = 3;
  p[0] = 1.1; p[1] = 1.1;
  q[0] = 1.0; q[1] = -2.0; q[2] = 3.5;
  CHKERRQ(FNSetType(fn,FNRATIONAL));
  CHKERRQ(FNRationalSetNumerator(fn,na,p));
  CHKERRQ(FNRationalSetDenominator(fn,nb,q));
  CHKERRQ(FNSetScale(fn,1.2,0.5));
  CHKERRQ(FNView(fn,NULL));
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  CHKERRQ(FNRationalGetNumerator(fn,&na,&pp));
  CHKERRQ(FNRationalGetDenominator(fn,&nb,&qq));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Coefficients:\n  Numerator: "));
  for (i=0;i<na;i++) {
    CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),pp[i],PETSC_FALSE));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n  Denominator: "));
  for (i=0;i<nb;i++) {
    CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),qq[i],PETSC_FALSE));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s ",str));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  CHKERRQ(PetscFree(pp));
  CHKERRQ(PetscFree(qq));

  /* constant */
  CHKERRQ(FNSetType(fn,FNRATIONAL));
  CHKERRQ(FNRationalSetNumerator(fn,1,&five));
  CHKERRQ(FNRationalSetDenominator(fn,0,NULL));  /* reset previous values */
  CHKERRQ(FNView(fn,NULL));
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  CHKERRQ(FNDestroy(&fn));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

TEST*/
