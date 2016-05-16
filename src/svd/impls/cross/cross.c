/*

   SLEPc singular value solver: "cross"

   Method: Uses a Hermitian eigensolver for A^T*A

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

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/

typedef struct {
  EPS       eps;
  Mat       mat;
  Vec       w,diag;
} SVD_CROSS;

#undef __FUNCT__
#define __FUNCT__ "MatMult_Cross"
static PetscErrorCode MatMult_Cross(Mat B,Vec x,Vec y)
{
  PetscErrorCode ierr;
  SVD            svd;
  SVD_CROSS      *cross;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  cross = (SVD_CROSS*)svd->data;
  ierr = SVDMatMult(svd,PETSC_FALSE,x,cross->w);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_TRUE,cross->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateVecs_Cross"
static PetscErrorCode MatCreateVecs_Cross(Mat B,Vec *right,Vec *left)
{
  PetscErrorCode ierr;
  SVD            svd;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  if (right) {
    ierr = SVDMatCreateVecs(svd,right,NULL);CHKERRQ(ierr);
    if (left) { ierr = VecDuplicate(*right,left);CHKERRQ(ierr); }
  } else {
    ierr = SVDMatCreateVecs(svd,left,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Cross"
static PetscErrorCode MatGetDiagonal_Cross(Mat B,Vec d)
{
  PetscErrorCode    ierr;
  SVD               svd;
  SVD_CROSS         *cross;
  PetscMPIInt       len;
  PetscInt          N,n,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  cross = (SVD_CROSS*)svd->data;
  if (!cross->diag) {
    /* compute diagonal from rows and store in cross->diag */
    ierr = VecDuplicate(d,&cross->diag);CHKERRQ(ierr);
    ierr = SVDMatGetSize(svd,NULL,&N);CHKERRQ(ierr);
    ierr = SVDMatGetLocalSize(svd,NULL,&n);CHKERRQ(ierr);
    ierr = PetscMalloc2(N,&work1,N,&work2);CHKERRQ(ierr);
    for (i=0;i<n;i++) work1[i] = work2[i] = 0.0;
    if (svd->AT) {
      ierr = MatGetOwnershipRange(svd->AT,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(svd->AT,i,&ncols,NULL,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++)
          work1[i] += vals[j]*vals[j];
        ierr = MatRestoreRow(svd->AT,i,&ncols,NULL,&vals);CHKERRQ(ierr);
      }
    } else {
      ierr = MatGetOwnershipRange(svd->A,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++)
          work1[cols[j]] += vals[j]*vals[j];
        ierr = MatRestoreRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      }
    }
    ierr = PetscMPIIntCast(N,&len);CHKERRQ(ierr);
    ierr = MPI_Allreduce(work1,work2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)svd));CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(cross->diag,&start,&end);CHKERRQ(ierr);
    ierr = VecGetArray(cross->diag,&diag);CHKERRQ(ierr);
    for (i=start;i<end;i++) diag[i-start] = work2[i];
    ierr = VecRestoreArray(cross->diag,&diag);CHKERRQ(ierr);
    ierr = PetscFree2(work1,work2);CHKERRQ(ierr);
  }
  ierr = VecCopy(cross->diag,d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetUp_Cross"
PetscErrorCode SVDSetUp_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscInt       n;
  PetscBool      trackall;

  PetscFunctionBegin;
  if (!cross->mat) {
    ierr = SVDMatGetLocalSize(svd,NULL,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)svd),n,n,PETSC_DETERMINE,PETSC_DETERMINE,svd,&cross->mat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(cross->mat,MATOP_MULT,(void(*)(void))MatMult_Cross);CHKERRQ(ierr);
    ierr = MatShellSetOperation(cross->mat,MATOP_GET_VECS,(void(*)(void))MatCreateVecs_Cross);CHKERRQ(ierr);
    ierr = MatShellSetOperation(cross->mat,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cross);CHKERRQ(ierr);
    ierr = SVDMatCreateVecs(svd,NULL,&cross->w);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->mat);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->w);CHKERRQ(ierr);
  }

  if (!cross->eps) { ierr = SVDCrossGetEPS(svd,&cross->eps);CHKERRQ(ierr); }
  ierr = EPSSetOperators(cross->eps,cross->mat,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(cross->eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(cross->eps,svd->which == SVD_LARGEST ? EPS_LARGEST_REAL : EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetDimensions(cross->eps,svd->nsv,svd->ncv?svd->ncv:PETSC_DEFAULT,svd->mpd?svd->mpd:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(cross->eps,svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:svd->tol,svd->max_it?svd->max_it:PETSC_DEFAULT);CHKERRQ(ierr);
  switch (svd->conv) {
  case SVD_CONV_ABS:
    ierr = EPSSetConvergenceTest(cross->eps,EPS_CONV_ABS);CHKERRQ(ierr);break;
  case SVD_CONV_REL:
    ierr = EPSSetConvergenceTest(cross->eps,EPS_CONV_REL);CHKERRQ(ierr);break;
  case SVD_CONV_USER:
    SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
  }
  if (svd->stop!=SVD_STOP_BASIC) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined stopping test not supported in this solver");
  /* Transfer the trackall option from svd to eps */
  ierr = SVDGetTrackAll(svd,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(cross->eps,trackall);CHKERRQ(ierr);
  ierr = EPSSetUp(cross->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cross->eps,NULL,&svd->ncv,&svd->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(cross->eps,NULL,&svd->max_it);CHKERRQ(ierr);
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;
  /* Transfer the initial space from svd to eps */
  if (svd->nini < 0) {
    ierr = EPSSetInitialSpace(cross->eps,-svd->nini,svd->IS);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->nini,&svd->IS);CHKERRQ(ierr);
  }
  svd->leftbasis = PETSC_FALSE;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSolve_Cross"
PetscErrorCode SVDSolve_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscInt       i;
  PetscScalar    sigma;
  Vec            v;

  PetscFunctionBegin;
  ierr = EPSSolve(cross->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cross->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cross->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cross->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = EPSGetEigenpair(cross->eps,i,&sigma,NULL,v,NULL);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    if (PetscRealPart(sigma)<0.0) SETERRQ(PetscObjectComm((PetscObject)svd),1,"Negative eigenvalue computed by EPS");
    svd->sigma[i] = PetscSqrtReal(PetscRealPart(sigma));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSMonitor_Cross"
static PetscErrorCode EPSMonitor_Cross(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<PetscMin(nest,svd->ncv);i++) {
    er = eigr[i]; ei = eigi[i];
    ierr = STBackTransform(eps->st,1,&er,&ei);CHKERRQ(ierr);
    svd->sigma[i] = PetscSqrtReal(PetscRealPart(er));
    svd->errest[i] = errest[i];
  }
  ierr = SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetFromOptions_Cross"
PetscErrorCode SVDSetFromOptions_Cross(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD Cross Options");CHKERRQ(ierr);
  if (!cross->eps) { ierr = SVDCrossGetEPS(svd,&cross->eps);CHKERRQ(ierr); }
  ierr = EPSSetFromOptions(cross->eps);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCrossSetEPS_Cross"
static PetscErrorCode SVDCrossSetEPS_Cross(SVD svd,EPS eps)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&cross->eps);CHKERRQ(ierr);
  cross->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->eps);CHKERRQ(ierr);
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCrossSetEPS"
/*@
   SVDCrossSetEPS - Associate an eigensolver object (EPS) to the
   singular value solver.

   Collective on SVD

   Input Parameters:
+  svd - singular value solver
-  eps - the eigensolver object

   Level: advanced

.seealso: SVDCrossGetEPS()
@*/
PetscErrorCode SVDCrossSetEPS(SVD svd,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscTryMethod(svd,"SVDCrossSetEPS_C",(SVD,EPS),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCrossGetEPS_Cross"
static PetscErrorCode SVDCrossGetEPS_Cross(SVD svd,EPS *eps)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  ST             st;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!cross->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)svd),&cross->eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(cross->eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(cross->eps,"svd_cross_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)cross->eps,(PetscObject)svd,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->eps);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(cross->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
    ierr = EPSMonitorSet(cross->eps,EPSMonitor_Cross,svd,NULL);CHKERRQ(ierr);
    ierr = EPSGetST(cross->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  *eps = cross->eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCrossGetEPS"
/*@
   SVDCrossGetEPS - Retrieve the eigensolver object (EPS) associated
   to the singular value solver.

   Not Collective

   Input Parameter:
.  svd - singular value solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: SVDCrossSetEPS()
@*/
PetscErrorCode SVDCrossGetEPS(SVD svd,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscUseMethod(svd,"SVDCrossGetEPS_C",(SVD,EPS*),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDView_Cross"
PetscErrorCode SVDView_Cross(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!cross->eps) { ierr = SVDCrossGetEPS(svd,&cross->eps);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = EPSView(cross->eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDReset_Cross"
PetscErrorCode SVDReset_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  if (cross->eps) { ierr = EPSReset(cross->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&cross->mat);CHKERRQ(ierr);
  ierr = VecDestroy(&cross->w);CHKERRQ(ierr);
  ierr = VecDestroy(&cross->diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDDestroy_Cross"
PetscErrorCode SVDDestroy_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&cross->eps);CHKERRQ(ierr);
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetEPS_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCreate_Cross"
PETSC_EXTERN PetscErrorCode SVDCreate_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&cross);CHKERRQ(ierr);
  svd->data = (void*)cross;

  svd->ops->solve          = SVDSolve_Cross;
  svd->ops->setup          = SVDSetUp_Cross;
  svd->ops->setfromoptions = SVDSetFromOptions_Cross;
  svd->ops->destroy        = SVDDestroy_Cross;
  svd->ops->reset          = SVDReset_Cross;
  svd->ops->view           = SVDView_Cross;
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetEPS_C",SVDCrossSetEPS_Cross);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetEPS_C",SVDCrossGetEPS_Cross);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

