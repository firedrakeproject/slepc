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
#define __FUNCT__ "DenseMatSqrt_Unblocked"
/*
   Compute the square root of an upper quasi-triangular matrix T,
   using Higham's algorithm (LAA 88, 1987). T is overwritten with sqrtm(T).
 */
static PetscErrorCode DenseMatSqrt_Unblocked(PetscBLASInt n,PetscScalar *T,PetscBLASInt ld)
{
#if defined(SLEPC_MISSING_LAPACK_TRSYL)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TRSYL - Lapack routine unavailable");
#else
  PetscScalar  one=1.0,mone=-1.0;
  PetscReal    done=1.0;
  PetscBLASInt i,j,si,sj,r,ione=1,info;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal    alpha,theta,mu,mu2;
#endif

  PetscFunctionBegin;
  for (j=0;j<n;j++) {
#if defined(PETSC_USE_COMPLEX)
    sj = 1;
    T[j+j*ld] = PetscSqrtScalar(T[j+j*ld]);
#else
    sj = (j==n-1 || T[j+1+j*ld] == 0.0)? 1: 2;
    if (sj==1) {
      if (T[j+j*ld]<0.0) SETERRQ(PETSC_COMM_SELF,1,"Matrix has a real negative eigenvalue, no real primary square root exists");
      T[j+j*ld] = PetscSqrtReal(T[j+j*ld]);
    } else {
      /* square root of 2x2 block */
      theta = (T[j+j*ld]+T[j+1+(j+1)*ld])/2.0;
      mu    = (T[j+j*ld]-T[j+1+(j+1)*ld])/2.0;
      mu2   = -mu*mu-T[j+1+j*ld]*T[j+(j+1)*ld];
      mu    = PetscSqrtReal(mu2);
      if (theta>0.0) alpha = PetscSqrtReal((theta+PetscSqrtReal(theta*theta+mu2))/2.0);
      else alpha = mu/PetscSqrtReal(2.0*(-theta+PetscSqrtReal(theta*theta+mu2)));
      T[j+j*ld]       /= 2.0*alpha;
      T[j+1+(j+1)*ld] /= 2.0*alpha;
      T[j+(j+1)*ld]   /= 2.0*alpha;
      T[j+1+j*ld]     /= 2.0*alpha;
      T[j+j*ld]       += alpha-theta/(2.0*alpha);
      T[j+1+(j+1)*ld] += alpha-theta/(2.0*alpha);
    }
#endif
    for (i=j-1;i>=0;i--) {
#if defined(PETSC_USE_COMPLEX)
      si = 1;
#else
      si = (i==0 || T[i+(i-1)*ld] == 0.0)? 1: 2;
      if (si==2) i--;
#endif
      /* solve Sylvester equation of order si x sj */
      r = j-i-si;
      if (r) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&si,&sj,&r,&mone,T+i+(i+si)*ld,&ld,T+i+si+j*ld,&ld,&one,T+i+j*ld,&ld));
      PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","N",&ione,&si,&sj,T+i+i*ld,&ld,T+j+j*ld,&ld,T+i+j*ld,&ld,&done,&info));
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTRSYL %d",info);
    }
    if (sj==2) j++;
  }
  PetscFunctionReturn(0);
#endif
}

#define BLOCKSIZE 64

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMat_Sqrt"
PetscErrorCode FNEvaluateFunctionMat_Sqrt(FN fn,Mat A,Mat B)
{
#if defined(SLEPC_MISSING_LAPACK_GEES) || defined(SLEPC_MISSING_LAPACK_TRSYL)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEES/TRSYL - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   i,j,r,ione=1,n,ld,sdim,lwork,*s,*p,info,bs=BLOCKSIZE;
  PetscScalar    *wr,*Aa,*T,*W,*Q,*work,one=1.0,zero=0.0,mone=-1.0;
  PetscInt       m,nblk;
  PetscReal      done=1.0;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else
  PetscReal      *wi;
#endif

  PetscFunctionBegin;
  ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld = n;
  nblk = (m+bs-1)/bs;
  lwork = 5*n;

  /* compute Schur decomposition A*Q = Q*T */
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc7(m,&wr,m,&wi,m*m,&W,m*m,&Q,lwork,&work,nblk,&s,nblk,&p);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgees",LAPACKgees_("V","N",NULL,&n,T,&ld,&sdim,wr,wi,Q,&ld,work,&lwork,NULL,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEES %d",info);
#else
  ierr = PetscMalloc7(m,&wr,m,&rwork,m*m,&W,m*m,&Q,lwork,&work,nblk,&s,nblk,&p);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgees",LAPACKgees_("V","N",NULL,&n,T,&ld,&sdim,wr,Q,&ld,work,&lwork,rwork,NULL,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEES %d",info);
#endif

  /* determine block sizes and positions, to avoid cutting 2x2 blocks */
  j = 0;
  p[j] = 0;
  do {
    s[j] = PetscMin(bs,n-p[j]);
#if !defined(PETSC_USE_COMPLEX)
    if (p[j]+s[j]!=n && T[p[j]+s[j]+(p[j]+s[j]-1)*ld]!=0.0) s[j]++;
#endif
    if (p[j]+s[j]==n) break;
    j++;
    p[j] = p[j-1]+s[j-1];
  } while (1);
  nblk = j+1;

  /* evaluate sqrt(T) */
  for (j=0;j<nblk;j++) {
    ierr = DenseMatSqrt_Unblocked(s[j],T+p[j]+p[j]*ld,ld);CHKERRQ(ierr);
    for (i=j-1;i>=0;i--) {
      /* solve Sylvester equation for block (i,j) */
      r = p[j]-p[i]-s[i];
      if (r) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",s+i,s+j,&r,&mone,T+p[i]+(p[i]+s[i])*ld,&ld,T+p[i]+s[i]+p[j]*ld,&ld,&one,T+p[i]+p[j]*ld,&ld));
      PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","N",&ione,s+i,s+j,T+p[i]+p[i]*ld,&ld,T+p[j]+p[j]*ld,&ld,T+p[i]+p[j]*ld,&ld,&done,&info));
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTRSYL %d",info);
    }
  }

  /* backtransform B = Q*T*Q' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,T,&ld,&zero,W,&ld));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&n,&n,&one,W,&ld,Q,&ld,&zero,T,&ld));

  ierr = MatDenseRestoreArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree7(wr,wi,W,Q,work,s,p);CHKERRQ(ierr);
#else
  ierr = PetscFree7(wr,rwork,W,Q,work,s,p);CHKERRQ(ierr);
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

