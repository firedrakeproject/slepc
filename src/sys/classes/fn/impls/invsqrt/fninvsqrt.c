/*
   Inverse square root function  x^(-1/2)

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

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction_Invsqrt"
PetscErrorCode FNEvaluateFunction_Invsqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  if (x==0.0) SETERRQ(PETSC_COMM_SELF,1,"Function not defined in the requested value");
  *y = 1.0/PetscSqrtScalar(x);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Invsqrt"
PetscErrorCode FNEvaluateDerivative_Invsqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  if (x==0.0) SETERRQ(PETSC_COMM_SELF,1,"Derivative not defined in the requested value");
  *y = -1.0/(2.0*PetscPowScalarReal(x,1.5));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMat_Invsqrt"
PetscErrorCode FNEvaluateFunctionMat_Invsqrt(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,*ipiv,info;
  PetscScalar    *Ba,*Wa;
  PetscInt       m;
  Mat            W;

  PetscFunctionBegin;
  ierr = FN_AllocateWorkMat(fn,A,&W);CHKERRQ(ierr);
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatDenseGetArray(W,&Wa);CHKERRQ(ierr);
  /* compute B = sqrtm(A) */
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld = n;
  ierr = SlepcSchurParlettSqrt(n,Ba,n,PETSC_FALSE);CHKERRQ(ierr);
  /* compute B = A\B */
  ierr = PetscMalloc1(ld,&ipiv);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Wa,&ld,ipiv,Ba,&ld,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESV %d",info);
  ierr = PetscFree(ipiv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(W,&Wa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  ierr = FN_FreeWorkMat(fn,&W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMatVec_Invsqrt"
PetscErrorCode FNEvaluateFunctionMatVec_Invsqrt(FN fn,Mat A,Vec v)
{
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,*ipiv,info,one=1;
  PetscScalar    *Ba,*Wa;
  PetscInt       m;
  Mat            B,W;

  PetscFunctionBegin;
  ierr = FN_AllocateWorkMat(fn,A,&B);CHKERRQ(ierr);
  ierr = FN_AllocateWorkMat(fn,A,&W);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatDenseGetArray(W,&Wa);CHKERRQ(ierr);
  /* compute B_1 = sqrtm(A)*e_1 */
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld = n;
  ierr = SlepcSchurParlettSqrt(n,Ba,n,PETSC_TRUE);CHKERRQ(ierr);
  /* compute B_1 = A\B_1 */
  ierr = PetscMalloc1(ld,&ipiv);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&one,Wa,&ld,ipiv,Ba,&ld,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESV %d",info);
  ierr = PetscFree(ipiv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(W,&Wa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetColumnVector(B,v,0);CHKERRQ(ierr);
  ierr = FN_FreeWorkMat(fn,&W);CHKERRQ(ierr);
  ierr = FN_FreeWorkMat(fn,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNView_Invsqrt"
PetscErrorCode FNView_Invsqrt(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Inverse square root: x^(-1/2)\n");CHKERRQ(ierr);
      } else {
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Inverse square root: (%s*x)^(-1/2)\n",str);CHKERRQ(ierr);
      }
    } else {
      ierr = SlepcSNPrintfScalar(str,50,fn->beta,PETSC_TRUE);CHKERRQ(ierr);
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Inverse square root: %s*x^(-1/2)\n",str);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Inverse square root: %s",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*(%s*x)^(-1/2)\n",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate_Invsqrt"
PETSC_EXTERN PetscErrorCode FNCreate_Invsqrt(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction       = FNEvaluateFunction_Invsqrt;
  fn->ops->evaluatederivative     = FNEvaluateDerivative_Invsqrt;
  fn->ops->evaluatefunctionmat    = FNEvaluateFunctionMat_Invsqrt;
  fn->ops->evaluatefunctionmatvec = FNEvaluateFunctionMatVec_Invsqrt;
  fn->ops->view                   = FNView_Invsqrt;
  PetscFunctionReturn(0);
}

