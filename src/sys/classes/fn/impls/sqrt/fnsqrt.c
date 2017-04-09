/*
   Square root function  sqrt(x)

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

PetscErrorCode FNEvaluateFunction_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  if (x<0.0) SETERRQ(PETSC_COMM_SELF,1,"Function not defined in the requested value");
#endif
  *y = PetscSqrtScalar(x);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  if (x==0.0) SETERRQ(PETSC_COMM_SELF,1,"Derivative not defined in the requested value");
#if !defined(PETSC_USE_COMPLEX)
  if (x<0.0) SETERRQ(PETSC_COMM_SELF,1,"Derivative not defined in the requested value");
#endif
  *y = 1.0/(2.0*PetscSqrtScalar(x));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Schur(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = SlepcSqrtmSchur(n,T,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMatVec_Sqrt_Schur(FN fn,Mat A,Vec v)
{
  PetscErrorCode ierr;
  PetscBLASInt   n;
  PetscScalar    *T;
  PetscInt       m;
  Mat            B;

  PetscFunctionBegin;
  ierr = FN_AllocateWorkMat(fn,A,&B);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = SlepcSqrtmSchur(n,T,n,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetColumnVector(B,v,0);CHKERRQ(ierr);
  ierr = FN_FreeWorkMat(fn,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = SlepcSqrtmDenmanBeavers(n,T,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define NSMAXIT 50

/*
   Computes the principal square root of the matrix A using the
   Newton-Schulz iteration. A is overwritten with sqrtm(A).
 */
static PetscErrorCode SlepcSqrtmNewtonSchulz(PetscBLASInt n,PetscScalar *A,PetscBLASInt ld)
{
  PetscScalar        *Y=A,*Yold,*Z,*Zold,*M,alpha,sqrtnrm;
  PetscScalar        szero=0.0,sone=1.0,smone=-1.0,spfive=0.5,sthree=3.0;
  PetscReal          tol,Yres,nrm;
  PetscBLASInt       i,it,N;
  const PetscBLASInt one=1;
  PetscBool          converged=PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;

  ierr = PetscMalloc4(N,&Yold,N,&Z,N,&Zold,N,&M);CHKERRQ(ierr);

  /* scale A so that ||I-A||_2 < 1 */
  ierr = PetscMemcpy(Z,A,N*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) Z[i+i*ld] -= 1.0;
  nrm = LAPACKlange_("fro",&n,&n,(PetscScalar*)Z,&n,PETSC_NULL);
  sqrtnrm = PetscSqrtReal(nrm);
  alpha = 1.0/nrm;
  PetscStackCallBLAS("BLASscal",BLASscal_(&N,&alpha,A,&one));
  tol *= nrm;
  ierr = PetscInfo2(NULL,"||I-A||_2 = %g, new tol: %g\n",(double)nrm,(double)tol);
  ierr = PetscLogFlops(2.0*n*n);CHKERRQ(ierr);

  /* Z = I */
  ierr = PetscMemzero(Z,N*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) Z[i+i*ld] = 1.0;

  for (it=0;it<NSMAXIT && !converged;it++) {
    /* Yold = Y, Zold = Z */
    ierr = PetscMemcpy(Yold,Y,N*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemcpy(Zold,Z,N*sizeof(PetscScalar));CHKERRQ(ierr);

    /* M = (3*I-Zold*Yold) */
    ierr = PetscMemzero(M,N*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) M[i+i*ld] = sthree;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&smone,Zold,&ld,Yold,&ld,&sone,M,&ld));

    /* Y = (1/2)*Yold*M, Z = (1/2)*M*Zold */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&spfive,Yold,&ld,M,&ld,&szero,Y,&ld));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&spfive,M,&ld,Zold,&ld,&szero,Z,&ld));

    /* reldiff = norm(Y-Yold,'fro')/norm(Y,'fro') */
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&smone,Y,&one,Yold,&one));
    Yres = LAPACKlange_("fro",&n,&n,(PetscScalar*)Yold,&n,PETSC_NULL);
    ierr = PetscIsNanReal(Yres);CHKERRQ(ierr);
    if (Yres<=tol) converged = PETSC_TRUE;
    ierr = PetscInfo2(NULL,"it: %D res: %g\n",it,(double)Yres);

    ierr = PetscLogFlops(6.0*n*n*n+2.0*n*n);CHKERRQ(ierr);
  }

  if (Yres>tol) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations",NSMAXIT);

  /* undo scaling */
  PetscStackCallBLAS("BLASscal",BLASscal_(&N,&sqrtnrm,A,&one));

  ierr = PetscFree4(Yold,Z,Zold,M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = SlepcSqrtmNewtonSchulz(n,Ba,n);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNView_Sqrt(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  char           str[50];
  const char     *methodname[] = {
                  "Schur method for the square root",
                  "Denman-Beavers (product form)",
                  "Newton-Schulz iteration"
  };
  const int      nmeth=sizeof(methodname)/sizeof(methodname[0]);

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
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*sqrt(%s*x)\n",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    if (fn->method<nmeth) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing matrix functions with: %s\n",methodname[fn->method]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode FNCreate_Sqrt(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction          = FNEvaluateFunction_Sqrt;
  fn->ops->evaluatederivative        = FNEvaluateDerivative_Sqrt;
  fn->ops->evaluatefunctionmat[0]    = FNEvaluateFunctionMat_Sqrt_Schur;
  fn->ops->evaluatefunctionmat[1]    = FNEvaluateFunctionMat_Sqrt_DBP;
  fn->ops->evaluatefunctionmat[2]    = FNEvaluateFunctionMat_Sqrt_NS;
  fn->ops->evaluatefunctionmatvec[0] = FNEvaluateFunctionMatVec_Sqrt_Schur;
  fn->ops->view                      = FNView_Sqrt;
  PetscFunctionReturn(0);
}

