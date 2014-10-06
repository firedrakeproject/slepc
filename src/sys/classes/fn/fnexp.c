/*
   Exponential function  f(x) = beta*exp(alpha*x).

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc-private/fnimpl.h>
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction_Exp"
PetscErrorCode FNEvaluateFunction_Exp(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscScalar arg;

  PetscFunctionBegin;
  if (!fn->na) arg = x;
  else arg = fn->alpha[0]*x;
  if (!fn->nb) *y = PetscExpScalar(arg);
  else *y = fn->beta[0]*PetscExpScalar(arg);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Exp"
PetscErrorCode FNEvaluateDerivative_Exp(FN fn,PetscScalar x,PetscScalar *yp)
{
  PetscScalar arg,scal;

  PetscFunctionBegin;
  if (!fn->na) {
    arg = x;
    scal = 1.0;
  } else {
    arg = fn->alpha[0]*x;
    scal = fn->alpha[0];
  }
  if (fn->nb) scal *= fn->beta[0];
  *yp = scal*PetscExpScalar(arg);
  PetscFunctionReturn(0);
}

#define MAX_PADE 6
#define SWAP(a,b,t) {t=a;a=b;b=t;}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMat_Exp"
PetscErrorCode FNEvaluateFunctionMat_Exp(FN fn,Mat A,Mat B)
{
#if defined(PETSC_MISSING_LAPACK_GESV) || defined(SLEPC_MISSING_LAPACK_LANGE)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESV/LANGE - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,ld2,*ipiv,info,inc=1;
  PetscInt       m,j,k;
  PetscBool      odd;
  const PetscInt p=MAX_PADE;
  PetscReal      c[MAX_PADE+1],s,*rwork;
  PetscScalar    scale,mone=-1.0,one=1.0,two=2.0,zero=0.0;
  PetscScalar    *Aa,*Ba,*As,*A2,*Q,*P,*W,*aux;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld  = n;
  ld2 = ld*ld;
  P   = Ba;
  ierr = PetscMalloc6(m*m,&Q,m*m,&W,m*m,&As,m*m,&A2,ld,&rwork,ld,&ipiv);CHKERRQ(ierr);
  ierr = PetscMemcpy(As,Aa,ld2*sizeof(PetscScalar));CHKERRQ(ierr);

  /* Pade' coefficients */
  c[0] = 1.0;
  for (k=1;k<=p;k++) c[k] = c[k-1]*(p+1-k)/(k*(2*p+1-k));

  /* Scaling */
  if (fn->na) PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,fn->alpha,As,&inc));
  s = LAPACKlange_("I",&n,&n,As,&ld,rwork);
  if (s>0.5) {
    s = PetscMax(0,(int)(PetscLogReal(s)/PetscLogReal(2.0))+2);
    scale = PetscPowScalar(2,-s);
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&scale,As,&inc));
  }

  /* Horner evaluation */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,As,&ld,As,&ld,&zero,A2,&ld));
  ierr = PetscMemzero(Q,ld2*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(P,ld2*sizeof(PetscScalar));CHKERRQ(ierr);
  for (j=0;j<n;j++) {
    Q[j+j*ld] = c[p];
    P[j+j*ld] = c[p-1];
  }

  odd = PETSC_TRUE;
  for (k=p-1;k>0;k--) {
    if (odd) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,A2,&ld,&zero,W,&ld));
      SWAP(Q,W,aux);
      for (j=0;j<n;j++) Q[j+j*ld] += c[k-1];
      odd = PETSC_FALSE;
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,A2,&ld,&zero,W,&ld));
      SWAP(P,W,aux);
      for (j=0;j<n;j++) P[j+j*ld] += c[k-1];
      odd = PETSC_TRUE;
    }
  }
  if (odd) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,As,&ld,&zero,W,&ld));
    SWAP(Q,W,aux);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&ld2,&mone,P,&inc,Q,&inc));
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Q,&ld,ipiv,P,&ld,&info));
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&two,P,&inc));
    for (j=0;j<n;j++) P[j+j*ld] += 1.0;
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&mone,P,&inc));
  } else {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,As,&ld,&zero,W,&ld));
    SWAP(P,W,aux);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&ld2,&mone,P,&inc,Q,&inc));
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Q,&ld,ipiv,P,&ld,&info));
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&two,P,&inc));
    for (j=0;j<n;j++) P[j+j*ld] += 1.0;
  }

  for (k=1;k<=s;k++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,P,&ld,&zero,W,&ld));
    ierr = PetscMemcpy(P,W,ld2*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (P!=Ba) { ierr = PetscMemcpy(Ba,P,ld2*sizeof(PetscScalar));CHKERRQ(ierr); }

  if (fn->nb) PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,fn->beta,Ba,&inc));
  ierr = PetscFree6(Q,W,As,A2,rwork,ipiv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "FNView_Exp"
PetscErrorCode FNView_Exp(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!fn->nb) {
      if (!fn->na) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: exp(x)\n");CHKERRQ(ierr);
      } else {
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha[0],PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: exp(%s*x)\n",str);CHKERRQ(ierr);
      }
    } else {
      ierr = SlepcSNPrintfScalar(str,50,fn->beta[0],PETSC_TRUE);CHKERRQ(ierr);
      if (!fn->na) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: %s*exp(x)\n",str);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: %s",str);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha[0],PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*exp(%s*x)\n",str);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate_Exp"
PETSC_EXTERN PetscErrorCode FNCreate_Exp(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction    = FNEvaluateFunction_Exp;
  fn->ops->evaluatederivative  = FNEvaluateDerivative_Exp;
  fn->ops->evaluatefunctionmat = FNEvaluateFunctionMat_Exp;
  fn->ops->view                = FNView_Exp;
  PetscFunctionReturn(0);
}

