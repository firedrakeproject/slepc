/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "cross"

   Method: Uses a Hermitian eigensolver for A^T*A
*/

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/

typedef struct {
  PetscBool explicitmatrix;
  EPS       eps;
  PetscBool usereps;
  Mat       C,D;
} SVD_CROSS;

typedef struct {
  Mat       A,AT;
  Vec       w,diag;
  PetscBool swapped;
} SVD_CROSS_SHELL;

static PetscErrorCode MatMult_Cross(Mat B,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  SVD_CROSS_SHELL *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,x,ctx->w);CHKERRQ(ierr);
  ierr = MatMult(ctx->AT,ctx->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Cross(Mat B,Vec d)
{
  PetscErrorCode    ierr;
  SVD_CROSS_SHELL   *ctx;
  PetscMPIInt       len;
  PetscInt          N,n,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  if (!ctx->diag) {
    /* compute diagonal from rows and store in ctx->diag */
    ierr = VecDuplicate(d,&ctx->diag);CHKERRQ(ierr);
    ierr = MatGetSize(ctx->A,NULL,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(ctx->A,NULL,&n);CHKERRQ(ierr);
    ierr = PetscCalloc2(N,&work1,N,&work2);CHKERRQ(ierr);
    if (ctx->swapped) {
      ierr = MatGetOwnershipRange(ctx->AT,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(ctx->AT,i,&ncols,NULL,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) work1[i] += vals[j]*vals[j];
        ierr = MatRestoreRow(ctx->AT,i,&ncols,NULL,&vals);CHKERRQ(ierr);
      }
    } else {
      ierr = MatGetOwnershipRange(ctx->A,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(ctx->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) work1[cols[j]] += vals[j]*vals[j];
        ierr = MatRestoreRow(ctx->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      }
    }
    ierr = PetscMPIIntCast(N,&len);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(work1,work2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)B));CHKERRMPI(ierr);
    ierr = VecGetOwnershipRange(ctx->diag,&start,&end);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(ctx->diag,&diag);CHKERRQ(ierr);
    for (i=start;i<end;i++) diag[i-start] = work2[i];
    ierr = VecRestoreArrayWrite(ctx->diag,&diag);CHKERRQ(ierr);
    ierr = PetscFree2(work1,work2);CHKERRQ(ierr);
  }
  ierr = VecCopy(ctx->diag,d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Cross(Mat B)
{
  PetscErrorCode  ierr;
  SVD_CROSS_SHELL *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->diag);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossGetProductMat(SVD svd,Mat A,Mat AT,Mat *C)
{
  PetscErrorCode  ierr;
  SVD_CROSS       *cross = (SVD_CROSS*)svd->data;
  SVD_CROSS_SHELL *ctx;
  PetscInt        n;
  VecType         vtype;

  PetscFunctionBegin;
  if (cross->explicitmatrix) {
    if (svd->expltrans) {  /* explicit transpose */
      ierr = MatProductCreate(AT,A,NULL,C);CHKERRQ(ierr);
      ierr = MatProductSetType(*C,MATPRODUCT_AB);CHKERRQ(ierr);
    } else {  /* implicit transpose */
#if defined(PETSC_USE_COMPLEX)
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Must use explicit transpose with complex scalars");
#else
      if (!svd->swapped) {
        ierr = MatProductCreate(A,A,NULL,C);CHKERRQ(ierr);
        ierr = MatProductSetType(*C,MATPRODUCT_AtB);CHKERRQ(ierr);
      } else {
        ierr = MatProductCreate(AT,AT,NULL,C);CHKERRQ(ierr);
        ierr = MatProductSetType(*C,MATPRODUCT_ABt);CHKERRQ(ierr);
      }
#endif
    }
    ierr = MatProductSetFromOptions(*C);CHKERRQ(ierr);
    ierr = MatProductSymbolic(*C);CHKERRQ(ierr);
    ierr = MatProductNumeric(*C);CHKERRQ(ierr);
  } else {
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    ierr = MatCreateVecs(A,NULL,&ctx->w);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->w);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,NULL,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)svd),n,n,PETSC_DETERMINE,PETSC_DETERMINE,(void*)ctx,C);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cross);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cross);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_Cross);CHKERRQ(ierr);
    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = MatSetVecType(*C,vtype);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetUp_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  ST             st;
  PetscBool      trackall,issinv;

  PetscFunctionBegin;
  if (!cross->eps) { ierr = SVDCrossGetEPS(svd,&cross->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&cross->C);CHKERRQ(ierr);
  ierr = MatDestroy(&cross->D);CHKERRQ(ierr);
  ierr = SVDCrossGetProductMat(svd,svd->A,svd->AT,&cross->C);CHKERRQ(ierr);
  if (svd->isgeneralized) {
    ierr = SVDCrossGetProductMat(svd,svd->B,svd->BT,&cross->D);CHKERRQ(ierr);
    ierr = EPSSetOperators(cross->eps,cross->C,cross->D);CHKERRQ(ierr);
    ierr = EPSSetProblemType(cross->eps,EPS_GHEP);CHKERRQ(ierr);
  } else {
    ierr = EPSSetOperators(cross->eps,cross->C,NULL);CHKERRQ(ierr);
    ierr = EPSSetProblemType(cross->eps,EPS_HEP);CHKERRQ(ierr);
  }
  if (!cross->usereps) {
    ierr = EPSGetST(cross->eps,&st);CHKERRQ(ierr);
    if (svd->isgeneralized && svd->which==SVD_SMALLEST) {
      ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
      ierr = EPSSetTarget(cross->eps,0.0);CHKERRQ(ierr);
      ierr = EPSSetWhichEigenpairs(cross->eps,EPS_TARGET_REAL);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectTypeCompare((PetscObject)st,STSINVERT,&issinv);CHKERRQ(ierr);
      if (issinv) {
        ierr = EPSSetWhichEigenpairs(cross->eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
      } else {
        ierr = EPSSetWhichEigenpairs(cross->eps,svd->which==SVD_LARGEST?EPS_LARGEST_REAL:EPS_SMALLEST_REAL);CHKERRQ(ierr);
      }
    }
    ierr = EPSSetDimensions(cross->eps,svd->nsv,svd->ncv,svd->mpd);CHKERRQ(ierr);
    ierr = EPSSetTolerances(cross->eps,svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:svd->tol,svd->max_it);CHKERRQ(ierr);
    switch (svd->conv) {
    case SVD_CONV_ABS:
      ierr = EPSSetConvergenceTest(cross->eps,EPS_CONV_ABS);CHKERRQ(ierr);break;
    case SVD_CONV_REL:
      ierr = EPSSetConvergenceTest(cross->eps,EPS_CONV_REL);CHKERRQ(ierr);break;
    case SVD_CONV_NORM:
      ierr = EPSSetConvergenceTest(cross->eps,EPS_CONV_NORM);CHKERRQ(ierr);break;
    case SVD_CONV_MAXIT:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Maxit convergence test not supported in this solver");
    case SVD_CONV_USER:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
    }
  }
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  /* Transfer the trackall option from svd to eps */
  ierr = SVDGetTrackAll(svd,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(cross->eps,trackall);CHKERRQ(ierr);
  /* Transfer the initial space from svd to eps */
  if (svd->nini<0) {
    ierr = EPSSetInitialSpace(cross->eps,-svd->nini,svd->IS);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->nini,&svd->IS);CHKERRQ(ierr);
  }
  ierr = EPSSetUp(cross->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cross->eps,NULL,&svd->ncv,&svd->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(cross->eps,NULL,&svd->max_it);CHKERRQ(ierr);
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  svd->leftbasis = PETSC_FALSE;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscInt       i;
  PetscScalar    lambda;
  PetscReal      sigma;

  PetscFunctionBegin;
  ierr = EPSSolve(cross->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cross->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cross->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cross->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = EPSGetEigenvalue(cross->eps,i,&lambda,NULL);CHKERRQ(ierr);
    sigma = PetscRealPart(lambda);
    if (sigma<-10*PETSC_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject)svd),1,"Negative eigenvalue computed by EPS: %g",sigma);
    if (sigma<0.0) {
      ierr = PetscInfo1(svd,"Negative eigenvalue computed by EPS: %g, resetting to 0\n",sigma);CHKERRQ(ierr);
      sigma = 0.0;
    }
    svd->sigma[i] = PetscSqrtReal(sigma);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDComputeVectors_Cross(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CROSS         *cross = (SVD_CROSS*)svd->data;
  PetscInt          i,mloc,ploc;
  Vec               u,v,x,uv;
  PetscScalar       *dst,alpha,lambda;
  const PetscScalar *src;
  PetscReal         nrm;

  PetscFunctionBegin;
  if (svd->isgeneralized) {
    ierr = MatCreateVecs(svd->A,NULL,&u);CHKERRQ(ierr);
    ierr = VecGetLocalSize(u,&mloc);CHKERRQ(ierr);
    ierr = MatCreateVecs(svd->B,NULL,&v);CHKERRQ(ierr);
    ierr = VecGetLocalSize(v,&ploc);CHKERRQ(ierr);
    for (i=0;i<svd->nconv;i++) {
      ierr = BVGetColumn(svd->V,i,&x);CHKERRQ(ierr);
      ierr = EPSGetEigenpair(cross->eps,i,&lambda,NULL,x,NULL);CHKERRQ(ierr);
      ierr = MatMult(svd->A,x,u);CHKERRQ(ierr);     /* u_i*c_i/alpha = A*x_i */
      ierr = VecNormalize(u,NULL);CHKERRQ(ierr);
      ierr = MatMult(svd->B,x,v);CHKERRQ(ierr);     /* v_i*s_i/alpha = B*x_i */
      ierr = VecNormalize(v,&nrm);CHKERRQ(ierr);    /* ||v||_2 = s_i/alpha   */
      alpha = 1.0/(PetscSqrtReal(1.0+PetscRealPart(lambda))*nrm);    /* alpha=s_i/||v||_2 */
      ierr = VecScale(x,alpha);CHKERRQ(ierr);
      ierr = BVRestoreColumn(svd->V,i,&x);CHKERRQ(ierr);
      /* copy [u;v] to U[i] */
      ierr = BVGetColumn(svd->U,i,&uv);CHKERRQ(ierr);
      ierr = VecGetArrayWrite(uv,&dst);CHKERRQ(ierr);
      ierr = VecGetArrayRead(u,&src);CHKERRQ(ierr);
      ierr = PetscArraycpy(dst,src,mloc);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(u,&src);CHKERRQ(ierr);
      ierr = VecGetArrayRead(v,&src);CHKERRQ(ierr);
      ierr = PetscArraycpy(dst+mloc,src,ploc);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(v,&src);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(uv,&dst);CHKERRQ(ierr);
      ierr = BVRestoreColumn(svd->U,i,&uv);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = VecDestroy(&u);CHKERRQ(ierr);
  } else {
    for (i=0;i<svd->nconv;i++) {
      ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
      ierr = EPSGetEigenvector(cross->eps,i,v,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    }
    ierr = SVDComputeVectors_Left(svd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
    svd->sigma[i] = PetscSqrtReal(PetscAbsReal(PetscRealPart(er)));
    svd->errest[i] = errest[i];
  }
  ierr = SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_Cross(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  ST             st;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD Cross Options");CHKERRQ(ierr);

    ierr = PetscOptionsBool("-svd_cross_explicitmatrix","Use cross explicit matrix","SVDCrossSetExplicitMatrix",cross->explicitmatrix,&val,&set);CHKERRQ(ierr);
    if (set) { ierr = SVDCrossSetExplicitMatrix(svd,val);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!cross->eps) { ierr = SVDCrossGetEPS(svd,&cross->eps);CHKERRQ(ierr); }
  if (!cross->explicitmatrix && !cross->usereps) {
    /* use as default an ST with shell matrix and Jacobi */
    ierr = EPSGetST(cross->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = EPSSetFromOptions(cross->eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossSetExplicitMatrix_Cross(SVD svd,PetscBool explicitmatrix)
{
  SVD_CROSS *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  if (cross->explicitmatrix != explicitmatrix) {
    cross->explicitmatrix = explicitmatrix;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDCrossSetExplicitMatrix - Indicate if the eigensolver operator A^T*A must
   be computed explicitly.

   Logically Collective on svd

   Input Parameters:
+  svd      - singular value solver
-  explicit - boolean flag indicating if A^T*A is built explicitly

   Options Database Key:
.  -svd_cross_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: SVDCrossGetExplicitMatrix()
@*/
PetscErrorCode SVDCrossSetExplicitMatrix(SVD svd,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmatrix,2);
  ierr = PetscTryMethod(svd,"SVDCrossSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossGetExplicitMatrix_Cross(SVD svd,PetscBool *explicitmatrix)
{
  SVD_CROSS *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  *explicitmatrix = cross->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   SVDCrossGetExplicitMatrix - Returns the flag indicating if A^T*A is built explicitly.

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicit - the mode flag

   Level: advanced

.seealso: SVDCrossSetExplicitMatrix()
@*/
PetscErrorCode SVDCrossGetExplicitMatrix(SVD svd,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(explicitmatrix,2);
  ierr = PetscUseMethod(svd,"SVDCrossGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossSetEPS_Cross(SVD svd,EPS eps)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&cross->eps);CHKERRQ(ierr);
  cross->eps = eps;
  cross->usereps = PETSC_TRUE;
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->eps);CHKERRQ(ierr);
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   SVDCrossSetEPS - Associate an eigensolver object (EPS) to the
   singular value solver.

   Collective on svd

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

static PetscErrorCode SVDCrossGetEPS_Cross(SVD svd,EPS *eps)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!cross->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)svd),&cross->eps);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)cross->eps,(PetscObject)svd,1);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(cross->eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(cross->eps,"svd_cross_");CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->eps);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)cross->eps,((PetscObject)svd)->options);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(cross->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
    ierr = EPSMonitorSet(cross->eps,EPSMonitor_Cross,svd,NULL);CHKERRQ(ierr);
  }
  *eps = cross->eps;
  PetscFunctionReturn(0);
}

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

PetscErrorCode SVDView_Cross(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!cross->eps) { ierr = SVDCrossGetEPS(svd,&cross->eps);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  %s matrix\n",cross->explicitmatrix?"explicit":"implicit");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = EPSView(cross->eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  ierr = EPSReset(cross->eps);CHKERRQ(ierr);
  ierr = MatDestroy(&cross->C);CHKERRQ(ierr);
  ierr = MatDestroy(&cross->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&cross->eps);CHKERRQ(ierr);
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Cross(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CROSS      *cross;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&cross);CHKERRQ(ierr);
  svd->data = (void*)cross;

  svd->ops->solve          = SVDSolve_Cross;
  svd->ops->solveg         = SVDSolve_Cross;
  svd->ops->setup          = SVDSetUp_Cross;
  svd->ops->setfromoptions = SVDSetFromOptions_Cross;
  svd->ops->destroy        = SVDDestroy_Cross;
  svd->ops->reset          = SVDReset_Cross;
  svd->ops->view           = SVDView_Cross;
  svd->ops->computevectors = SVDComputeVectors_Cross;
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetEPS_C",SVDCrossSetEPS_Cross);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetEPS_C",SVDCrossGetEPS_Cross);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetExplicitMatrix_C",SVDCrossSetExplicitMatrix_Cross);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetExplicitMatrix_C",SVDCrossGetExplicitMatrix_Cross);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

