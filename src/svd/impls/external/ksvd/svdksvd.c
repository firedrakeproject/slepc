/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to the KSVD SVD solver
*/

#include <slepc/private/svdimpl.h>    /*I "slepcsvd.h" I*/
#include <slepc/private/slepcscalapack.h>
#include <ksvd.h>

typedef struct {
  Mat As;        /* converted matrix */
} SVD_KSVD;

PetscErrorCode SVDSetUp_KSVD(SVD svd)
{
  SVD_KSVD  *ctx = (SVD_KSVD*)svd->data;
  PetscInt  M,N;

  PetscFunctionBegin;
  SVDCheckStandard(svd);
  SVDCheckDefinite(svd);
  PetscCall(MatGetSize(svd->A,&M,&N));
  PetscCheck(M==N,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"The interface to KSVD does not support rectangular matrices");
  svd->ncv = N;
  if (svd->mpd!=PETSC_DEFAULT) PetscCall(PetscInfo(svd,"Warning: parameter mpd ignored\n"));
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = 1;
  svd->leftbasis = PETSC_TRUE;
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  PetscCall(SVDAllocateSolution(svd,0));

  /* convert matrix */
  PetscCall(MatDestroy(&ctx->As));
  PetscCall(MatConvert(svd->OP,MATSCALAPACK,MAT_INITIAL_MATRIX,&ctx->As));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVDSolve_KSVD(SVD svd)
{
  SVD_KSVD       *ctx = (SVD_KSVD*)svd->data;
  Mat            A = ctx->As,Z,Q,U,V;
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data,*q,*z;
  PetscScalar    *work,minlwork;
  PetscBLASInt   info,lwork=-1,*iwork,liwork=-1,minliwork,one=1;
  PetscInt       M,N,m,n,mn;
#if defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lrwork;
  PetscReal      *rwork,dummy;
#endif

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  mn = (M>=N)? n: m;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&Z));
  PetscCall(MatSetSizes(Z,m,mn,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(Z,MATSCALAPACK));
  PetscCall(MatSetUp(Z));
  PetscCall(MatAssemblyBegin(Z,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Z,MAT_FINAL_ASSEMBLY));
  z = (Mat_ScaLAPACK*)Z->data;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&Q));
  PetscCall(MatSetSizes(Q,mn,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(Q,MATSCALAPACK));
  PetscCall(MatSetUp(Q));
  PetscCall(MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY));
  q = (Mat_ScaLAPACK*)Q->data;

  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  /* allocate workspace */
  PetscStackCallExternalVoid("pdgeqsvd",pdgeqsvd("V","V","r",a->M,a->N,a->loc,one,one,a->desc,svd->sigma,z->loc,one,one,z->desc,q->loc,one,one,q->desc,&minlwork,lwork,&minliwork,liwork,&info));
  PetscCheck(!info,PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"Error in KSVD subroutine pdgeqsvd: info=%d",(int)info);
  PetscCall(PetscBLASIntCast((PetscInt)minlwork,&lwork));
  PetscCall(PetscBLASIntCast(minliwork,&liwork));
  PetscCall(PetscMalloc2(lwork,&work,liwork,&iwork));
  /* call computational routine */
  PetscStackCallExternalVoid("pdgeqsvd",pdgeqsvd("V","V","r",a->M,a->N,a->loc,one,one,a->desc,svd->sigma,z->loc,one,one,z->desc,q->loc,one,one,q->desc,work,lwork,iwork,liwork,&info));
  PetscCheck(!info,PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"Error in KSVD subroutine pdgeqsvd: info=%d",(int)info);
  PetscCall(PetscFree2(work,iwork));
  PetscCall(PetscFPTrapPop());

  PetscCall(BVGetMat(svd->U,&U));
  PetscCall(BVGetMat(svd->V,&V));
  if (M>=N) {
    PetscCall(MatConvert(Z,MATDENSE,MAT_REUSE_MATRIX,&U));
    PetscCall(MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&V));
  } else {
    PetscCall(MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&U));
    PetscCall(MatConvert(Z,MATDENSE,MAT_REUSE_MATRIX,&V));
  }
  PetscCall(BVRestoreMat(svd->U,&U));
  PetscCall(BVRestoreMat(svd->V,&V));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&Q));

  svd->nconv  = svd->ncv;
  svd->its    = 1;
  svd->reason = SVD_CONVERGED_TOL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVDDestroy_KSVD(SVD svd)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(svd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SVDReset_KSVD(SVD svd)
{
  SVD_KSVD  *ctx = (SVD_KSVD*)svd->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ctx->As));
  PetscFunctionReturn(PETSC_SUCCESS);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_KSVD(SVD svd)
{
  SVD_KSVD  *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  svd->data = (void*)ctx;

  svd->ops->solve          = SVDSolve_KSVD;
  svd->ops->setup          = SVDSetUp_KSVD;
  svd->ops->destroy        = SVDDestroy_KSVD;
  svd->ops->reset          = SVDReset_KSVD;
  PetscFunctionReturn(PETSC_SUCCESS);
}
