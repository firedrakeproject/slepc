/*

   SLEPc singular value solver: "cyclic"

   Method: Uses a Hermitian eigensolver for H(A) = [ 0  A ; A^T 0 ]

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
  PetscBool explicitmatrix;
  EPS       eps;
  Mat       mat;
  Vec       x1,x2,y1,y2;
} SVD_CYCLIC;

#undef __FUNCT__
#define __FUNCT__ "MatMult_Cyclic"
static PetscErrorCode MatMult_Cyclic(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  SVD               svd;
  SVD_CYCLIC        *cyclic;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  cyclic = (SVD_CYCLIC*)svd->data;
  ierr = SVDMatGetLocalSize(svd,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->y2,py+m);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_FALSE,cyclic->x2,cyclic->y1);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_TRUE,cyclic->x1,cyclic->y2);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->x1);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->x2);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->y1);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Cyclic"
static PetscErrorCode MatGetDiagonal_Cyclic(Mat B,Vec diag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(diag,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetUp_Cyclic"
PetscErrorCode SVDSetUp_Cyclic(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          M,N,m,n,i,isl,Istart,Iend;
  const PetscScalar *isa;
  PetscScalar       *va;
  PetscBool         trackall,gpu,issinv;
  Vec               v;
  Mat               Zm,Zn;
  ST                st;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)svd->A,&gpu,MATSEQAIJCUSP,MATMPIAIJCUSP,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,"");CHKERRQ(ierr);
  if (gpu) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Solver not implemented for GPU matrices");
  ierr = SVDMatGetSize(svd,&M,&N);CHKERRQ(ierr);
  ierr = SVDMatGetLocalSize(svd,&m,&n);CHKERRQ(ierr);
  if (!cyclic->mat) {
    if (cyclic->explicitmatrix) {
      if (!svd->AT) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
      ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Zm);CHKERRQ(ierr);
      ierr = MatSetSizes(Zm,m,m,M,M);CHKERRQ(ierr);
      ierr = MatSetFromOptions(Zm);CHKERRQ(ierr);
      ierr = MatSetUp(Zm);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(Zm,&Istart,&Iend);CHKERRQ(ierr);
      for (i=Istart;i<Iend;i++) {
        ierr = MatSetValue(Zm,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(Zm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Zm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatCreate(PetscObjectComm((PetscObject)svd),&Zn);CHKERRQ(ierr);
      ierr = MatSetSizes(Zn,n,n,N,N);CHKERRQ(ierr);
      ierr = MatSetFromOptions(Zn);CHKERRQ(ierr);
      ierr = MatSetUp(Zn);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(Zn,&Istart,&Iend);CHKERRQ(ierr);
      for (i=Istart;i<Iend;i++) {
        ierr = MatSetValue(Zn,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(Zn,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = SlepcMatTile(1.0,Zm,1.0,svd->A,1.0,svd->AT,1.0,Zn,&cyclic->mat);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->mat);CHKERRQ(ierr);
      ierr = MatDestroy(&Zm);CHKERRQ(ierr);
      ierr = MatDestroy(&Zn);CHKERRQ(ierr);
    } else {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,m,M,NULL,&cyclic->x1);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,n,N,NULL,&cyclic->x2);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,m,M,NULL,&cyclic->y1);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,n,N,NULL,&cyclic->y2);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->x1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->x2);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->y1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->y2);CHKERRQ(ierr);
      ierr = MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,svd,&cyclic->mat);CHKERRQ(ierr);
      ierr = MatShellSetOperation(cyclic->mat,MATOP_MULT,(void(*)(void))MatMult_Cyclic);CHKERRQ(ierr);
      ierr = MatShellSetOperation(cyclic->mat,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cyclic);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->mat);CHKERRQ(ierr);
  }

  if (!cyclic->eps) { ierr = SVDCyclicGetEPS(svd,&cyclic->eps);CHKERRQ(ierr); }
  ierr = EPSSetOperators(cyclic->eps,cyclic->mat,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(cyclic->eps,EPS_HEP);CHKERRQ(ierr);
  if (svd->which == SVD_LARGEST) {
    ierr = EPSGetST(cyclic->eps,&st);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)st,STSINVERT,&issinv);CHKERRQ(ierr);
    if (issinv) {
      ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
    } else {
      ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
    }
  } else {
    ierr = EPSSetEigenvalueComparison(cyclic->eps,SlepcCompareSmallestPosReal,NULL);CHKERRQ(ierr);
    ierr = EPSSetTarget(cyclic->eps,0.0);CHKERRQ(ierr);
  }
  ierr = EPSSetDimensions(cyclic->eps,svd->nsv,svd->ncv?svd->ncv:PETSC_DEFAULT,svd->mpd?svd->mpd:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(cyclic->eps,svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:svd->tol,svd->max_it?svd->max_it:PETSC_DEFAULT);CHKERRQ(ierr);
  switch (svd->conv) {
  case SVD_CONV_ABS:
    ierr = EPSSetConvergenceTest(cyclic->eps,EPS_CONV_ABS);CHKERRQ(ierr);break;
  case SVD_CONV_REL:
    ierr = EPSSetConvergenceTest(cyclic->eps,EPS_CONV_REL);CHKERRQ(ierr);break;
  case SVD_CONV_USER:
    SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
  }
  if (svd->stop!=SVD_STOP_BASIC) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined stopping test not supported in this solver");
  /* Transfer the trackall option from svd to eps */
  ierr = SVDGetTrackAll(svd,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(cyclic->eps,trackall);CHKERRQ(ierr);
  /* Transfer the initial subspace from svd to eps */
  if (svd->nini<0 || svd->ninil<0) {
    for (i=0;i<-PetscMin(svd->nini,svd->ninil);i++) {
      ierr = MatCreateVecs(cyclic->mat,&v,NULL);CHKERRQ(ierr);
      ierr = VecGetArray(v,&va);CHKERRQ(ierr);
      if (i<-svd->ninil) {
        ierr = VecGetSize(svd->ISL[i],&isl);CHKERRQ(ierr);
        if (isl!=m) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for left initial vector");
        ierr = VecGetArrayRead(svd->ISL[i],&isa);CHKERRQ(ierr);
        ierr = PetscMemcpy(va,isa,sizeof(PetscScalar)*m);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(svd->IS[i],&isa);CHKERRQ(ierr);
      } else {
        ierr = PetscMemzero(&va,sizeof(PetscScalar)*m);CHKERRQ(ierr);
      }
      if (i<-svd->nini) {
        ierr = VecGetSize(svd->IS[i],&isl);CHKERRQ(ierr);
        if (isl!=n) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for right initial vector");
        ierr = VecGetArrayRead(svd->IS[i],&isa);CHKERRQ(ierr);
        ierr = PetscMemcpy(va+m,isa,sizeof(PetscScalar)*n);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(svd->IS[i],&isa);CHKERRQ(ierr);
      } else {
        ierr = PetscMemzero(va+m,sizeof(PetscScalar)*n);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
      ierr = VecDestroy(&svd->IS[i]);CHKERRQ(ierr);
      svd->IS[i] = v;
    }
    svd->nini = PetscMin(svd->nini,svd->ninil);
    ierr = EPSSetInitialSpace(cyclic->eps,-svd->nini,svd->IS);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->nini,&svd->IS);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->ninil,&svd->ISL);CHKERRQ(ierr);
  }
  ierr = EPSSetUp(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cyclic->eps,NULL,&svd->ncv,&svd->mpd);CHKERRQ(ierr);
  svd->ncv = PetscMin(svd->ncv,PetscMin(M,N));
  ierr = EPSGetTolerances(cyclic->eps,NULL,&svd->max_it);CHKERRQ(ierr);
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  svd->leftbasis = PETSC_TRUE;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSolve_Cyclic"
PetscErrorCode SVDSolve_Cyclic(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          i,j,M,N,m,n;
  PetscScalar       sigma;
  const PetscScalar *px;
  Vec               x,x1,x2;

  PetscFunctionBegin;
  ierr = EPSSolve(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cyclic->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cyclic->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cyclic->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);

  ierr = MatCreateVecs(cyclic->mat,&x,NULL);CHKERRQ(ierr);
  ierr = SVDMatGetSize(svd,&M,&N);CHKERRQ(ierr);
  ierr = SVDMatGetLocalSize(svd,&m,&n);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,m,M,NULL,&x1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,n,N,NULL,&x2);CHKERRQ(ierr);
  for (i=0,j=0;i<svd->nconv;i++) {
    ierr = EPSGetEigenpair(cyclic->eps,i,&sigma,NULL,x,NULL);CHKERRQ(ierr);
    if (PetscRealPart(sigma) > 0.0) {
      svd->sigma[j] = PetscRealPart(sigma);
      ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
      ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
      ierr = BVInsertVec(svd->U,j,x1);CHKERRQ(ierr);
      ierr = BVScaleColumn(svd->U,j,1.0/PetscSqrtReal(2.0));CHKERRQ(ierr);
      ierr = BVInsertVec(svd->V,j,x2);CHKERRQ(ierr);
      ierr = BVScaleColumn(svd->V,j,1.0/PetscSqrtReal(2.0));CHKERRQ(ierr);
      ierr = VecResetArray(x1);CHKERRQ(ierr);
      ierr = VecResetArray(x2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
      j++;
    }
  }
  svd->nconv = j;

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSMonitor_Cyclic"
static PetscErrorCode EPSMonitor_Cyclic(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i,j;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0,j=0;i<PetscMin(nest,svd->ncv);i++) {
    er = eigr[i]; ei = eigi[i];
    ierr = STBackTransform(eps->st,1,&er,&ei);CHKERRQ(ierr);
    if (PetscRealPart(er) > 0.0) {
      svd->sigma[j] = PetscRealPart(er);
      svd->errest[j] = errest[i];
      if (errest[i] && errest[i] < svd->tol) nconv++;
      j++;
    }
  }
  nest = j;
  ierr = SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetFromOptions_Cyclic"
PetscErrorCode SVDSetFromOptions_Cyclic(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  ST             st;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD Cyclic Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-svd_cyclic_explicitmatrix","Use cyclic explicit matrix","SVDCyclicSetExplicitMatrix",cyclic->explicitmatrix,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = SVDCyclicSetExplicitMatrix(svd,val);CHKERRQ(ierr);
  }
  if (!cyclic->eps) { ierr = SVDCyclicGetEPS(svd,&cyclic->eps);CHKERRQ(ierr); }
  ierr = EPSSetFromOptions(cyclic->eps);CHKERRQ(ierr);
  if (!cyclic->explicitmatrix) {
    /* use as default an ST with shell matrix and Jacobi */
    ierr = EPSGetST(cyclic->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicSetExplicitMatrix_Cyclic"
static PetscErrorCode SVDCyclicSetExplicitMatrix_Cyclic(SVD svd,PetscBool explicitmatrix)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  cyclic->explicitmatrix = explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicSetExplicitMatrix"
/*@
   SVDCyclicSetExplicitMatrix - Indicate if the eigensolver operator
   H(A) = [ 0  A ; A^T 0 ] must be computed explicitly.

   Logically Collective on SVD

   Input Parameters:
+  svd      - singular value solver
-  explicit - boolean flag indicating if H(A) is built explicitly

   Options Database Key:
.  -svd_cyclic_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: SVDCyclicGetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicSetExplicitMatrix(SVD svd,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmatrix,2);
  ierr = PetscTryMethod(svd,"SVDCyclicSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicGetExplicitMatrix_Cyclic"
static PetscErrorCode SVDCyclicGetExplicitMatrix_Cyclic(SVD svd,PetscBool *explicitmatrix)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  *explicitmatrix = cyclic->explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicGetExplicitMatrix"
/*@
   SVDCyclicGetExplicitMatrix - Returns the flag indicating if H(A) is built explicitly

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicit - the mode flag

   Level: advanced

.seealso: SVDCyclicSetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicGetExplicitMatrix(SVD svd,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(explicitmatrix,2);
  ierr = PetscUseMethod(svd,"SVDCyclicGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicSetEPS_Cyclic"
static PetscErrorCode SVDCyclicSetEPS_Cyclic(SVD svd,EPS eps)
{
  PetscErrorCode  ierr;
  SVD_CYCLIC      *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&cyclic->eps);CHKERRQ(ierr);
  cyclic->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->eps);CHKERRQ(ierr);
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicSetEPS"
/*@
   SVDCyclicSetEPS - Associate an eigensolver object (EPS) to the
   singular value solver.

   Collective on SVD

   Input Parameters:
+  svd - singular value solver
-  eps - the eigensolver object

   Level: advanced

.seealso: SVDCyclicGetEPS()
@*/
PetscErrorCode SVDCyclicSetEPS(SVD svd,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscTryMethod(svd,"SVDCyclicSetEPS_C",(SVD,EPS),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicGetEPS_Cyclic"
static PetscErrorCode SVDCyclicGetEPS_Cyclic(SVD svd,EPS *eps)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (!cyclic->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)svd),&cyclic->eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(cyclic->eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(cyclic->eps,"svd_cyclic_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)cyclic->eps,(PetscObject)svd,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cyclic->eps);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
    ierr = EPSMonitorSet(cyclic->eps,EPSMonitor_Cyclic,svd,NULL);CHKERRQ(ierr);
  }
  *eps = cyclic->eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicGetEPS"
/*@
   SVDCyclicGetEPS - Retrieve the eigensolver object (EPS) associated
   to the singular value solver.

   Not Collective

   Input Parameter:
.  svd - singular value solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: SVDCyclicSetEPS()
@*/
PetscErrorCode SVDCyclicGetEPS(SVD svd,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscUseMethod(svd,"SVDCyclicGetEPS_C",(SVD,EPS*),(svd,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDView_Cyclic"
PetscErrorCode SVDView_Cyclic(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!cyclic->eps) { ierr = SVDCyclicGetEPS(svd,&cyclic->eps);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  Cyclic: %s matrix\n",cyclic->explicitmatrix?"explicit":"implicit");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = EPSView(cyclic->eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDReset_Cyclic"
PetscErrorCode SVDReset_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (!cyclic->eps) { ierr = EPSReset(cyclic->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&cyclic->mat);CHKERRQ(ierr);
  ierr = VecDestroy(&cyclic->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&cyclic->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&cyclic->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&cyclic->y2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDDestroy_Cyclic"
PetscErrorCode SVDDestroy_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&cyclic->eps);CHKERRQ(ierr);
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCreate_Cyclic"
PETSC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&cyclic);CHKERRQ(ierr);
  svd->data                      = (void*)cyclic;
  svd->ops->solve                = SVDSolve_Cyclic;
  svd->ops->setup                = SVDSetUp_Cyclic;
  svd->ops->setfromoptions       = SVDSetFromOptions_Cyclic;
  svd->ops->destroy              = SVDDestroy_Cyclic;
  svd->ops->reset                = SVDReset_Cyclic;
  svd->ops->view                 = SVDView_Cyclic;
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",SVDCyclicSetEPS_Cyclic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",SVDCyclicGetEPS_Cyclic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",SVDCyclicSetExplicitMatrix_Cyclic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",SVDCyclicGetExplicitMatrix_Cyclic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

