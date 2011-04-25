/*                       

   SLEPc singular value solver: "cross"

   Method: Uses a Hermitian eigensolver for A^T*A

   Last update: Jun 2007

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <private/epsimpl.h>                /*I "slepceps.h" I*/

typedef struct {
  EPS       eps;
  PetscBool setfromoptionscalled;
  Mat       mat;
  Vec       w,diag;
} SVD_CROSS;

#undef __FUNCT__  
#define __FUNCT__ "ShellMatMult_CROSS"
PetscErrorCode ShellMatMult_CROSS(Mat B,Vec x, Vec y)
{
  PetscErrorCode ierr;
  SVD            svd;
  SVD_CROSS      *cross;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  cross = (SVD_CROSS *)svd->data;
  ierr = SVDMatMult(svd,PETSC_FALSE,x,cross->w);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_TRUE,cross->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ShellMatGetDiagonal_CROSS"
PetscErrorCode ShellMatGetDiagonal_CROSS(Mat B,Vec d)
{
  PetscErrorCode    ierr;
  SVD               svd;
  SVD_CROSS         *cross;
  PetscInt          N,n,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  cross = (SVD_CROSS *)svd->data;
  if (!cross->diag) {
    /* compute diagonal from rows and store in cross->diag */
    ierr = VecDuplicate(d,&cross->diag);CHKERRQ(ierr);
    ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
    ierr = SVDMatGetLocalSize(svd,PETSC_NULL,&n);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*N,&work1);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*N,&work2);CHKERRQ(ierr);
    for (i=0;i<n;i++) work1[i] = work2[i] = 0.0;
    if (svd->AT) {
      ierr = MatGetOwnershipRange(svd->AT,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(svd->AT,i,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++)
          work1[i] += vals[j]*vals[j];
        ierr = MatRestoreRow(svd->AT,i,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
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
    ierr = MPI_Allreduce(work1,work2,N,MPIU_SCALAR,MPI_SUM,((PetscObject)svd)->comm);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(cross->diag,&start,&end);CHKERRQ(ierr);
    ierr = VecGetArray(cross->diag,&diag);CHKERRQ(ierr);
    for (i=start;i<end;i++)
      diag[i-start] = work2[i];
    ierr = VecRestoreArray(cross->diag,&diag);CHKERRQ(ierr);
    ierr = PetscFree(work1);CHKERRQ(ierr);
    ierr = PetscFree(work2);CHKERRQ(ierr);
  }
  ierr = VecCopy(cross->diag,d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_CROSS"
PetscErrorCode SVDSetUp_CROSS(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CROSS         *cross = (SVD_CROSS *)svd->data;
  PetscInt          n,i;
  PetscBool         trackall;

  PetscFunctionBegin;
  ierr = MatDestroy(&cross->mat);CHKERRQ(ierr);
  ierr = VecDestroy(&cross->w);CHKERRQ(ierr);
  ierr = VecDestroy(&cross->diag);CHKERRQ(ierr);
  
  ierr = SVDMatGetLocalSize(svd,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = MatCreateShell(((PetscObject)svd)->comm,n,n,PETSC_DETERMINE,PETSC_DETERMINE,svd,&cross->mat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(cross->mat,MATOP_MULT,(void(*)(void))ShellMatMult_CROSS);CHKERRQ(ierr);  
  ierr = MatShellSetOperation(cross->mat,MATOP_GET_DIAGONAL,(void(*)(void))ShellMatGetDiagonal_CROSS);CHKERRQ(ierr);  
  ierr = SVDMatGetVecs(svd,PETSC_NULL,&cross->w);CHKERRQ(ierr);

  ierr = EPSSetOperators(cross->eps,cross->mat,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(cross->eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(cross->eps,svd->which == SVD_LARGEST ? EPS_LARGEST_REAL : EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetDimensions(cross->eps,svd->nsv,svd->ncv,svd->mpd);CHKERRQ(ierr);
  ierr = EPSSetTolerances(cross->eps,svd->tol,svd->max_it);CHKERRQ(ierr);
  /* Transfer the trackall option from svd to eps */
  ierr = SVDGetTrackAll(svd,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(cross->eps,trackall);CHKERRQ(ierr);
  if (cross->setfromoptionscalled) {
    ierr = EPSSetFromOptions(cross->eps);CHKERRQ(ierr);
    cross->setfromoptionscalled = PETSC_FALSE;
  }
  ierr = EPSSetUp(cross->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cross->eps,PETSC_NULL,&svd->ncv,&svd->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(cross->eps,&svd->tol,&svd->max_it);CHKERRQ(ierr);
  /* Transfer the initial space from svd to eps */
  if (svd->nini < 0) {
    ierr = EPSSetInitialSpace(cross->eps,-svd->nini,svd->IS);CHKERRQ(ierr);
    for(i=0; i<-svd->nini; i++) {
      ierr = VecDestroy(&svd->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(svd->IS);CHKERRQ(ierr);
    svd->nini = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_CROSS"
PetscErrorCode SVDSolve_CROSS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS *)svd->data;
  PetscInt       i;
  PetscScalar    sigma;
  
  PetscFunctionBegin;
  ierr = EPSSolve(cross->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cross->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cross->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cross->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = EPSGetEigenpair(cross->eps,i,&sigma,PETSC_NULL,svd->V[i],PETSC_NULL);CHKERRQ(ierr);
    svd->sigma[i] = sqrt(PetscRealPart(sigma));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitor_CROSS"
PetscErrorCode SVDMonitor_CROSS(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<nest;i++) {
    er = eigr[i]; ei = eigi[i];
    ierr = STBackTransform(eps->OP, 1, &er, &ei); CHKERRQ(ierr);
    svd->sigma[i] = sqrt(PetscRealPart(er));
    svd->errest[i] = errest[i];
  }
  SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_CROSS"
PetscErrorCode SVDSetFromOptions_CROSS(SVD svd)
{
  SVD_CROSS      *cross = (SVD_CROSS *)svd->data;

  PetscFunctionBegin;
  cross->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCrossSetEPS_CROSS"
PetscErrorCode SVDCrossSetEPS_CROSS(SVD svd,EPS eps)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS *)svd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(cross->eps);CHKERRQ(ierr);  
  cross->eps = eps;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  ierr = PetscTryMethod(svd,"SVDCrossSetEPS_C",(SVD,EPS),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCrossGetEPS_CROSS"
PetscErrorCode SVDCrossGetEPS_CROSS(SVD svd,EPS *eps)
{
  SVD_CROSS *cross = (SVD_CROSS *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(eps,2);
  *eps = cross->eps;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  ierr = PetscTryMethod(svd,"SVDCrossGetEPS_C",(SVD,EPS*),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_CROSS"
PetscErrorCode SVDView_CROSS(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS *)svd->data;

  PetscFunctionBegin;
  ierr = EPSView(cross->eps,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_CROSS"
PetscErrorCode SVDDestroy_CROSS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS *)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(cross->eps);CHKERRQ(ierr);
  ierr = MatDestroy(&cross->mat);CHKERRQ(ierr);
  ierr = VecDestroy(&cross->w);CHKERRQ(ierr);
  ierr = VecDestroy(&cross->diag);CHKERRQ(ierr);
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCrossSetEPS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCrossGetEPS_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_CROSS"
PetscErrorCode SVDCreate_CROSS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross;
  ST             st;
  
  PetscFunctionBegin;
  ierr = PetscNew(SVD_CROSS,&cross);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_CROSS));
  svd->data                = (void *)cross;
  svd->ops->solve          = SVDSolve_CROSS;
  svd->ops->setup          = SVDSetUp_CROSS;
  svd->ops->setfromoptions = SVDSetFromOptions_CROSS;
  svd->ops->destroy        = SVDDestroy_CROSS;
  svd->ops->view           = SVDView_CROSS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCrossSetEPS_C","SVDCrossSetEPS_CROSS",SVDCrossSetEPS_CROSS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCrossGetEPS_C","SVDCrossGetEPS_CROSS",SVDCrossGetEPS_CROSS);CHKERRQ(ierr);

  ierr = EPSCreate(((PetscObject)svd)->comm,&cross->eps);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(cross->eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
  ierr = EPSAppendOptionsPrefix(cross->eps,"svd_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)cross->eps,(PetscObject)svd,1);CHKERRQ(ierr);  
  PetscLogObjectParent(svd,cross->eps);
  ierr = EPSSetIP(cross->eps,svd->ip);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(cross->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSMonitorSet(cross->eps,SVDMonitor_CROSS,svd,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSGetST(cross->eps,&st);CHKERRQ(ierr);
  ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  cross->mat = PETSC_NULL;
  cross->w = PETSC_NULL;
  cross->diag = PETSC_NULL;
  cross->setfromoptionscalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
