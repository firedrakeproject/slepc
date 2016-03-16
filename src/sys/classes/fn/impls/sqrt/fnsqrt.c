/*
   Square root function  sqrt(x)

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction_Sqrt"
PetscErrorCode FNEvaluateFunction_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  *y = PetscSqrtScalar(x);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Sqrt"
PetscErrorCode FNEvaluateDerivative_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  if (x==0.0) SETERRQ(PETSC_COMM_SELF,1,"Derivative not defined in the requested value");
  *y = 1.0/(2.0*PetscSqrtScalar(x));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMat_Sqrt"
PetscErrorCode FNEvaluateFunctionMat_Sqrt(FN fn,Mat A,Mat B)
{
#if defined(SLEPC_MISSING_LAPACK_GEES) || defined(SLEPC_MISSING_LAPACK_TRSYL)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEES/TRSYL - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,sdim,lwork,info;
  PetscScalar    *wr,*Aa,*T,*W,*Q,*work,one=1.0,zero=0.0;
  PetscInt       m,i,j;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       k;
  PetscReal      *rwork;
#else
  PetscBLASInt   si,sj,r,ione=1;
  PetscScalar    mone=-1.0;
  PetscReal      *wi,alpha,theta,mu,mu2;
#endif

  PetscFunctionBegin;
  ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld = n;
  lwork = 5*n;

  /* compute Schur decomposition A*Q = Q*T */
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc5(m,&wr,m,&wi,m*m,&W,m*m,&Q,lwork,&work);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgees",LAPACKgees_("V","N",NULL,&n,T,&ld,&sdim,wr,wi,Q,&ld,work,&lwork,NULL,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEES %d",info);
#else
  ierr = PetscMalloc5(m,&wr,m,&rwork,m*m,&W,m*m,&Q,lwork,&work);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgees",LAPACKgees_("V","N",NULL,&n,T,&ld,&sdim,wr,Q,&ld,work,&lwork,rwork,NULL,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEES %d",info);
#endif

  /* evaluate sqrt(T) */
#if defined(PETSC_USE_COMPLEX)
  for (j=0;j<m;j++) {
    T[j+j*ld] = PetscSqrtScalar(T[j+j*ld]);
    for (i=j-1;i>=0;i--) {
      T[i+j*ld] /= (T[i+i*ld]+T[j+j*ld]);
      for (k=0;k<i;k++) T[k+j*ld] -= T[k+i*ld]*T[i+j*ld];
    }
  }
#else
  /* real Schur form, use algorithm of Higham (LAA 88, 1987) */
  for (j=0;j<m;j++) {
    sj = (j==m-1 || T[j+1+j*ld] == 0.0)? 1: 2;
    if (sj==1) {
      if (T[j+j*ld]<0.0) SETERRQ(PETSC_COMM_SELF,1,"Matrix has a real negative eigenvalue, no real primary square root exists");
      T[j+j*ld] = PetscSqrtReal(T[j+j*ld]);
    } else {
      /* square root of 2x2 block */
      theta = (T[j+j*ld]+T[j+1+(j+1)*ld])/2.0;
      mu = (T[j+j*ld]-T[j+1+(j+1)*ld])/2.0;
      mu2 = -mu*mu-T[j+1+j*ld]*T[j+(j+1)*ld];
      mu = PetscSqrtReal(mu2);
      if (theta>0.0) alpha = PetscSqrtReal((theta+PetscSqrtReal(theta*theta+mu2))/2.0);
      else alpha = mu/PetscSqrtReal(2.0*(-theta+PetscSqrtReal(theta*theta+mu2)));
      T[j+j*ld]       /= 2.0*alpha;
      T[j+1+(j+1)*ld] /= 2.0*alpha;
      T[j+(j+1)*ld]   /= 2.0*alpha;
      T[j+1+j*ld]     /= 2.0*alpha;
      T[j+j*ld]       += alpha-theta/(2.0*alpha);
      T[j+1+(j+1)*ld] += alpha-theta/(2.0*alpha);
    }
    for (i=j-1;i>=0;i--) {
      si = (i==0 || T[i+(i-1)*ld] == 0.0)? 1: 2;
      if (si==2) i--;
      /* solve Sylvester equation of order si x sj */
      r = j-i-si;
      if (r) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&si,&sj,&r,&mone,T+i+(i+si)*ld,&ld,T+i+si+j*ld,&ld,&one,T+i+j*ld,&ld));
      PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","N",&ione,&si,&sj,T+i+i*ld,&ld,T+j+j*ld,&ld,T+i+j*ld,&ld,&one,&info));
    }
    if (sj==2) j++;
  }
#endif

  /* backtransform B = Q*T*Q' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,T,&ld,&zero,W,&ld));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&one,W,&ld,Q,&ld,&zero,T,&ld));

  ierr = MatDenseRestoreArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree5(wr,wi,W,Q,work);CHKERRQ(ierr);
#else
  ierr = PetscFree5(wr,rwork,W,Q,work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "FNView_Sqrt"
PetscErrorCode FNView_Sqrt(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(x)\n");CHKERRQ(ierr);
      } else {
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(%s*x)\n",str);CHKERRQ(ierr);
      }
    } else {
      ierr = SlepcSNPrintfScalar(str,50,fn->beta,PETSC_TRUE);CHKERRQ(ierr);
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: %s*sqrt(x)\n",str);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: %s",str);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*sqrt(%s*x)\n",str);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate_Sqrt"
PETSC_EXTERN PetscErrorCode FNCreate_Sqrt(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction    = FNEvaluateFunction_Sqrt;
  fn->ops->evaluatederivative  = FNEvaluateDerivative_Sqrt;
  fn->ops->evaluatefunctionmat = FNEvaluateFunctionMat_Sqrt;
  fn->ops->view                = FNView_Sqrt;
  PetscFunctionReturn(0);
}

