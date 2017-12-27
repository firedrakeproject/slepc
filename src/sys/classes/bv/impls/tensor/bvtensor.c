/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Tensor BV that is represented in compact form as V = (I otimes U) S
*/

#include <slepc/private/bvimpl.h>

typedef struct {
  BV          U;        /* first factor */
  Mat         S;        /* second factor */
  PetscScalar *qB;      /* auxiliary matrix used in non-standard inner products */
  PetscInt    d;        /* degree of the tensor BV */
  PetscInt    ld;       /* leading dimension of S */
} BV_TENSOR;

PetscErrorCode BVMult_Tensor(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultVec_Tensor(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlace_Tensor(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVMultInPlaceTranspose_Tensor(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVDot_Tensor(BV X,BV Y,Mat M)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Tensor(BV X,Vec y,PetscScalar *q)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVDotVec_Local_Tensor(BV X,Vec y,PetscScalar *m)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVScale_Tensor(BV bv,PetscInt j,PetscScalar alpha)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Tensor(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVNorm_Local_Tensor(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVMatMult_Tensor(BV V,Mat A,BV W)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVCopy_Tensor(BV V,BV W)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVResize_Tensor(BV bv,PetscInt m,PetscBool copy)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetColumn_Tensor(BV bv,PetscInt j,Vec *v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreColumn_Tensor(BV bv,PetscInt j,Vec *v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArray_Tensor(BV bv,PetscScalar **a)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArray_Tensor(BV bv,PetscScalar **a)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVGetArrayRead_Tensor(BV bv,const PetscScalar **a)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVRestoreArrayRead_Tensor(BV bv,const PetscScalar **a)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation not implemented in BVTENSOR");
  PetscFunctionReturn(0);
}

PetscErrorCode BVView_Tensor(BV bv,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  BV_TENSOR         *ctx = (BV_TENSOR*)bv->data;
  PetscViewerFormat format;
  PetscBool         isascii;
  const char        *bvname,*uname,*sname;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPrintf(viewer,"number of tensor blocks (degree): %D\n",ctx->d);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"number of columns of U factor: %D\n",ctx->ld);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = BVView(ctx->U,viewer);CHKERRQ(ierr);
    ierr = MatView(ctx->S,viewer);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)bv,&bvname);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject)ctx->U,&uname);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject)ctx->S,&sname);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s=kron(eye(%D),%s)*%s;\n",bvname,ctx->d,uname,sname);CHKERRQ(ierr);
    }
  } else {
    ierr = BVView(ctx->U,viewer);CHKERRQ(ierr);
    ierr = MatView(ctx->S,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode BVTensorGetDegree_Tensor(BV bv,PetscInt *d)
{
  BV_TENSOR *ctx = (BV_TENSOR*)bv->data;

  PetscFunctionBegin;
  *d = ctx->d;
  PetscFunctionReturn(0);
}

/*@
   BVTensorGetDegree - Returns the number of blocks (degree) of the tensor BV.

   Not collective

   Input Parameter:
.  bv - the basis vectors context

   Output Parameter:
.  d - the degree

   Level: advanced

.seealso: BVCreateTensor()
@*/
PetscErrorCode BVTensorGetDegree(BV bv,PetscInt *d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidPointer(d,2);
  ierr = PetscUseMethod(bv,"BVTensorGetDegree_C",(BV,PetscInt*),(bv,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode BVTensorGetFactors_Tensor(BV V,BV *U,Mat *S)
{
  BV_TENSOR *ctx = (BV_TENSOR*)V->data;

  PetscFunctionBegin;
  if (U) *U = ctx->U;
  if (S) *S = ctx->S;
  PetscFunctionReturn(0);
}

/*@C
   BVTensorGetFactors - Returns the two factors involved in the definition of the
   tensor basis vectors object, V = (I otimes U) S.

   Logically Collective on BV

   Input Parameter:
.  V - the basis vectors context

   Output Parameters:
+  U - the BV factor
-  S - the Mat factor

   Notes:
   The returned factors are references (not copies) of the internal factors,
   so modifying them will change the tensor BV as well. Some operations of the
   tensor BV assume that U has orthonormal columns, so if the user modifies U
   this restriction must be taken into account.

   The returned factors must not be destroyed. BVTensorRestoreFactors() must
   be called when they are no longer needed.

   Pass a NULL vector for any of the arguments that is not needed.

   Level: advanced

.seealso: BVTensorRestoreFactors()
@*/
PetscErrorCode BVTensorGetFactors(BV V,BV *U,Mat *S)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  ierr = PetscUseMethod(V,"BVTensorGetFactors_C",(BV,BV*,Mat*),(V,U,S));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode BVTensorRestoreFactors_Tensor(BV V,BV *U,Mat *S)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  if (U) *U = NULL;
  if (S) *S = NULL;
  PetscFunctionReturn(0);
}

/*@C
   BVTensorRestoreFactors - Restore the two factors that were obtained with
   BVTensorGetFactors().

   Logically Collective on BV

   Input Parameters:
+  V - the basis vectors context
.  U - the BV factor (or NULL)
-  S - the Mat factor (or NULL)

   Nots:
   The arguments must match the corresponding call to BVTensorGetFactors().

   Level: advanced

.seealso: BVTensorGetFactors()
@*/
PetscErrorCode BVTensorRestoreFactors(BV V,BV *U,Mat *S)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  if (U) PetscValidHeaderSpecific(*U,BV_CLASSID,2);
  if (S) PetscValidHeaderSpecific(*S,MAT_CLASSID,3);
  ierr = PetscUseMethod(V,"BVTensorRestoreFactors_C",(BV,BV*,Mat*),(V,U,S));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVDestroy_Tensor(BV bv)
{
  PetscErrorCode ierr;
  BV_TENSOR      *ctx = (BV_TENSOR*)bv->data;

  PetscFunctionBegin;
  ierr = BVDestroy(&ctx->U);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = PetscFree(ctx->qB);CHKERRQ(ierr);
  ierr = VecDestroy(&bv->cv[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&bv->cv[1]);CHKERRQ(ierr);
  ierr = PetscFree(bv->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)bv,"BVTensorGetDegree_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)bv,"BVTensorGetFactors_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)bv,"BVTensorRestoreFactors_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode BVCreate_Tensor(BV bv)
{
  PetscErrorCode ierr;
  BV_TENSOR      *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  if (bv->issplit) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"BVGetSplit() not implemented for tensor BV");

  ierr = VecDuplicateEmpty(bv->t,&bv->cv[0]);CHKERRQ(ierr);
  ierr = VecDuplicateEmpty(bv->t,&bv->cv[1]);CHKERRQ(ierr);

  bv->ops->mult             = BVMult_Tensor;
  bv->ops->multvec          = BVMultVec_Tensor;
  bv->ops->multinplace      = BVMultInPlace_Tensor;
  bv->ops->multinplacetrans = BVMultInPlaceTranspose_Tensor;
  bv->ops->dot              = BVDot_Tensor;
  bv->ops->dotvec           = BVDotVec_Tensor;
  bv->ops->dotvec_local     = BVDotVec_Local_Tensor;
  bv->ops->scale            = BVScale_Tensor;
  bv->ops->norm             = BVNorm_Tensor;
  bv->ops->norm_local       = BVNorm_Local_Tensor;
  bv->ops->matmult          = BVMatMult_Tensor;
  bv->ops->copy             = BVCopy_Tensor;
  bv->ops->resize           = BVResize_Tensor;
  bv->ops->getcolumn        = BVGetColumn_Tensor;
  bv->ops->restorecolumn    = BVRestoreColumn_Tensor;
  bv->ops->getarray         = BVGetArray_Tensor;
  bv->ops->restorearray     = BVRestoreArray_Tensor;
  bv->ops->getarrayread     = BVGetArrayRead_Tensor;
  bv->ops->restorearrayread = BVRestoreArrayRead_Tensor;
  bv->ops->destroy          = BVDestroy_Tensor;
  bv->ops->view             = BVView_Tensor;

  ierr = PetscObjectComposeFunction((PetscObject)bv,"BVTensorGetDegree_C",BVTensorGetDegree_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)bv,"BVTensorGetFactors_C",BVTensorGetFactors_Tensor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)bv,"BVTensorRestoreFactors_C",BVTensorRestoreFactors_Tensor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVCreateTensor - Creates a tensor BV that is represented in compact form
   as V = (I otimes U) S, where U has orthonormal columns.

   Collective on BV

   Input Parameters:
+  U - a basis vectors object
-  d - the number of blocks (degree) of the tensor BV

   Output Parameter:
.  V - the new basis vectors context

   Notes:
   The new basis vectors object is V = (I otimes U) S, where otimes denotes
   the Kronecker product, I is the identity matrix of order d, and S is a
   sequential matrix allocated internally. This compact representation is
   used e.g. to represent the Krylov basis generated with the linearization
   of a matrix polynomial of degree d.

   The size of V (number of rows) is equal to d times n, where n is the size
   of U. The dimensions of S are d times m rows and m-d columns, where m is
   the number of columns of U, so m should be at least d+1.

   The communicator of V will be the same as U.

   On input, U must have orthonormal columns, as this property is assumed
   by some operations.

   Level: advanced

.seealso: BVTensorGetDegree(), BVTensorGetFactors()
@*/
PetscErrorCode BVCreateTensor(BV U,PetscInt d,BV *V)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       n,N,m;
  BV_TENSOR      *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(U,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(U,d,2);
  ierr = PetscObjectTypeCompare((PetscObject)U,BVTENSOR,&match);CHKERRQ(ierr);
  if (match) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"U cannot be of type tensor");

  ierr = BVCreate(PetscObjectComm((PetscObject)U),V);CHKERRQ(ierr);
  ierr = BVGetSizes(U,&n,&N,&m);CHKERRQ(ierr);
  if (m<d+1) SETERRQ2(PetscObjectComm((PetscObject)U),PETSC_ERR_ARG_SIZ,"U has %D columns, it should have at least d+1=%D",m,d+1);
  ierr = BVSetSizes(*V,d*n,d*N,m-d);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*V,BVTENSOR);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(BV_Create,*V,0,0,0);CHKERRQ(ierr);
  ierr = BVCreate_Tensor(*V);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Create,*V,0,0,0);CHKERRQ(ierr);

  ctx = (BV_TENSOR*)(*V)->data;
  ctx->U  = U;
  ctx->d  = d;
  ctx->ld = m;
  ierr = PetscObjectReference((PetscObject)U);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,d*m,m,NULL,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)*V,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)ctx->S,"S");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

