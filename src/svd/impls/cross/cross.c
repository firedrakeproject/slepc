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
  SVD_CROSS_SHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(MatMult(ctx->A,x,ctx->w));
  CHKERRQ(MatMult(ctx->AT,ctx->w,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Cross(Mat B,Vec d)
{
  SVD_CROSS_SHELL   *ctx;
  PetscMPIInt       len;
  PetscInt          N,n,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  if (!ctx->diag) {
    /* compute diagonal from rows and store in ctx->diag */
    CHKERRQ(VecDuplicate(d,&ctx->diag));
    CHKERRQ(MatGetSize(ctx->A,NULL,&N));
    CHKERRQ(MatGetLocalSize(ctx->A,NULL,&n));
    CHKERRQ(PetscCalloc2(N,&work1,N,&work2));
    if (ctx->swapped) {
      CHKERRQ(MatGetOwnershipRange(ctx->AT,&start,&end));
      for (i=start;i<end;i++) {
        CHKERRQ(MatGetRow(ctx->AT,i,&ncols,NULL,&vals));
        for (j=0;j<ncols;j++) work1[i] += vals[j]*vals[j];
        CHKERRQ(MatRestoreRow(ctx->AT,i,&ncols,NULL,&vals));
      }
    } else {
      CHKERRQ(MatGetOwnershipRange(ctx->A,&start,&end));
      for (i=start;i<end;i++) {
        CHKERRQ(MatGetRow(ctx->A,i,&ncols,&cols,&vals));
        for (j=0;j<ncols;j++) work1[cols[j]] += vals[j]*vals[j];
        CHKERRQ(MatRestoreRow(ctx->A,i,&ncols,&cols,&vals));
      }
    }
    CHKERRQ(PetscMPIIntCast(N,&len));
    CHKERRMPI(MPIU_Allreduce(work1,work2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)B)));
    CHKERRQ(VecGetOwnershipRange(ctx->diag,&start,&end));
    CHKERRQ(VecGetArrayWrite(ctx->diag,&diag));
    for (i=start;i<end;i++) diag[i-start] = work2[i];
    CHKERRQ(VecRestoreArrayWrite(ctx->diag,&diag));
    CHKERRQ(PetscFree2(work1,work2));
  }
  CHKERRQ(VecCopy(ctx->diag,d));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Cross(Mat B)
{
  SVD_CROSS_SHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&ctx));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(VecDestroy(&ctx->diag));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossGetProductMat(SVD svd,Mat A,Mat AT,Mat *C)
{
  SVD_CROSS       *cross = (SVD_CROSS*)svd->data;
  SVD_CROSS_SHELL *ctx;
  PetscInt        n;
  VecType         vtype;

  PetscFunctionBegin;
  if (cross->explicitmatrix) {
    if (svd->expltrans) {  /* explicit transpose */
      CHKERRQ(MatProductCreate(AT,A,NULL,C));
      CHKERRQ(MatProductSetType(*C,MATPRODUCT_AB));
    } else {  /* implicit transpose */
#if defined(PETSC_USE_COMPLEX)
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Must use explicit transpose with complex scalars");
#else
      if (!svd->swapped) {
        CHKERRQ(MatProductCreate(A,A,NULL,C));
        CHKERRQ(MatProductSetType(*C,MATPRODUCT_AtB));
      } else {
        CHKERRQ(MatProductCreate(AT,AT,NULL,C));
        CHKERRQ(MatProductSetType(*C,MATPRODUCT_ABt));
      }
#endif
    }
    CHKERRQ(MatProductSetFromOptions(*C));
    CHKERRQ(MatProductSymbolic(*C));
    CHKERRQ(MatProductNumeric(*C));
  } else {
    CHKERRQ(PetscNew(&ctx));
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    CHKERRQ(MatCreateVecs(A,NULL,&ctx->w));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->w));
    CHKERRQ(MatGetLocalSize(A,NULL,&n));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)svd),n,n,PETSC_DETERMINE,PETSC_DETERMINE,(void*)ctx,C));
    CHKERRQ(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cross));
    CHKERRQ(MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cross));
    CHKERRQ(MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_Cross));
    CHKERRQ(MatGetVecType(A,&vtype));
    CHKERRQ(MatSetVecType(*C,vtype));
  }
  CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)*C));
  PetscFunctionReturn(0);
}

/* Convergence test relative to the norm of R (used in GSVD only) */
static PetscErrorCode EPSConv_Cross(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  SVD svd = (SVD)ctx;

  PetscFunctionBegin;
  *errest = res/PetscMax(svd->nrma,svd->nrmb);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetUp_Cross(SVD svd)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  ST             st;
  PetscBool      trackall,issinv;

  PetscFunctionBegin;
  if (!cross->eps) CHKERRQ(SVDCrossGetEPS(svd,&cross->eps));
  CHKERRQ(MatDestroy(&cross->C));
  CHKERRQ(MatDestroy(&cross->D));
  CHKERRQ(SVDCrossGetProductMat(svd,svd->A,svd->AT,&cross->C));
  if (svd->isgeneralized) {
    CHKERRQ(SVDCrossGetProductMat(svd,svd->B,svd->BT,&cross->D));
    CHKERRQ(EPSSetOperators(cross->eps,cross->C,cross->D));
    CHKERRQ(EPSSetProblemType(cross->eps,EPS_GHEP));
  } else {
    CHKERRQ(EPSSetOperators(cross->eps,cross->C,NULL));
    CHKERRQ(EPSSetProblemType(cross->eps,EPS_HEP));
  }
  if (!cross->usereps) {
    CHKERRQ(EPSGetST(cross->eps,&st));
    if (svd->isgeneralized && svd->which==SVD_SMALLEST) {
      CHKERRQ(STSetType(st,STSINVERT));
      CHKERRQ(EPSSetTarget(cross->eps,0.0));
      CHKERRQ(EPSSetWhichEigenpairs(cross->eps,EPS_TARGET_REAL));
    } else {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)st,STSINVERT,&issinv));
      if (issinv) {
        CHKERRQ(EPSSetWhichEigenpairs(cross->eps,EPS_TARGET_MAGNITUDE));
      } else {
        CHKERRQ(EPSSetWhichEigenpairs(cross->eps,svd->which==SVD_LARGEST?EPS_LARGEST_REAL:EPS_SMALLEST_REAL));
      }
    }
    CHKERRQ(EPSSetDimensions(cross->eps,svd->nsv,svd->ncv,svd->mpd));
    CHKERRQ(EPSSetTolerances(cross->eps,svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:svd->tol,svd->max_it));
    switch (svd->conv) {
    case SVD_CONV_ABS:
      CHKERRQ(EPSSetConvergenceTest(cross->eps,EPS_CONV_ABS));break;
    case SVD_CONV_REL:
      CHKERRQ(EPSSetConvergenceTest(cross->eps,EPS_CONV_REL));break;
    case SVD_CONV_NORM:
      if (svd->isgeneralized) {
        if (!svd->nrma) CHKERRQ(MatNorm(svd->OP,NORM_INFINITY,&svd->nrma));
        if (!svd->nrmb) CHKERRQ(MatNorm(svd->OPb,NORM_INFINITY,&svd->nrmb));
        CHKERRQ(EPSSetConvergenceTestFunction(cross->eps,EPSConv_Cross,svd,NULL));
      } else {
        CHKERRQ(EPSSetConvergenceTest(cross->eps,EPS_CONV_NORM));break;
      }
      break;
    case SVD_CONV_MAXIT:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Maxit convergence test not supported in this solver");
    case SVD_CONV_USER:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
    }
  }
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  /* Transfer the trackall option from svd to eps */
  CHKERRQ(SVDGetTrackAll(svd,&trackall));
  CHKERRQ(EPSSetTrackAll(cross->eps,trackall));
  /* Transfer the initial space from svd to eps */
  if (svd->nini<0) {
    CHKERRQ(EPSSetInitialSpace(cross->eps,-svd->nini,svd->IS));
    CHKERRQ(SlepcBasisDestroy_Private(&svd->nini,&svd->IS));
  }
  CHKERRQ(EPSSetUp(cross->eps));
  CHKERRQ(EPSGetDimensions(cross->eps,NULL,&svd->ncv,&svd->mpd));
  CHKERRQ(EPSGetTolerances(cross->eps,NULL,&svd->max_it));
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  svd->leftbasis = PETSC_FALSE;
  CHKERRQ(SVDAllocateSolution(svd,0));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Cross(SVD svd)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscInt       i;
  PetscScalar    lambda;
  PetscReal      sigma;

  PetscFunctionBegin;
  CHKERRQ(EPSSolve(cross->eps));
  CHKERRQ(EPSGetConverged(cross->eps,&svd->nconv));
  CHKERRQ(EPSGetIterationNumber(cross->eps,&svd->its));
  CHKERRQ(EPSGetConvergedReason(cross->eps,(EPSConvergedReason*)&svd->reason));
  for (i=0;i<svd->nconv;i++) {
    CHKERRQ(EPSGetEigenvalue(cross->eps,i,&lambda,NULL));
    sigma = PetscRealPart(lambda);
    PetscCheck(sigma>-10*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)svd),PETSC_ERR_FP,"Negative eigenvalue computed by EPS: %g",(double)sigma);
    if (sigma<0.0) {
      CHKERRQ(PetscInfo(svd,"Negative eigenvalue computed by EPS: %g, resetting to 0\n",(double)sigma));
      sigma = 0.0;
    }
    svd->sigma[i] = PetscSqrtReal(sigma);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDComputeVectors_Cross(SVD svd)
{
  SVD_CROSS         *cross = (SVD_CROSS*)svd->data;
  PetscInt          i,mloc,ploc;
  Vec               u,v,x,uv;
  PetscScalar       *dst,alpha,lambda;
  const PetscScalar *src;
  PetscReal         nrm;

  PetscFunctionBegin;
  if (svd->isgeneralized) {
    CHKERRQ(MatCreateVecs(svd->A,NULL,&u));
    CHKERRQ(VecGetLocalSize(u,&mloc));
    CHKERRQ(MatCreateVecs(svd->B,NULL,&v));
    CHKERRQ(VecGetLocalSize(v,&ploc));
    for (i=0;i<svd->nconv;i++) {
      CHKERRQ(BVGetColumn(svd->V,i,&x));
      CHKERRQ(EPSGetEigenpair(cross->eps,i,&lambda,NULL,x,NULL));
      CHKERRQ(MatMult(svd->A,x,u));     /* u_i*c_i/alpha = A*x_i */
      CHKERRQ(VecNormalize(u,NULL));
      CHKERRQ(MatMult(svd->B,x,v));     /* v_i*s_i/alpha = B*x_i */
      CHKERRQ(VecNormalize(v,&nrm));    /* ||v||_2 = s_i/alpha   */
      alpha = 1.0/(PetscSqrtReal(1.0+PetscRealPart(lambda))*nrm);    /* alpha=s_i/||v||_2 */
      CHKERRQ(VecScale(x,alpha));
      CHKERRQ(BVRestoreColumn(svd->V,i,&x));
      /* copy [u;v] to U[i] */
      CHKERRQ(BVGetColumn(svd->U,i,&uv));
      CHKERRQ(VecGetArrayWrite(uv,&dst));
      CHKERRQ(VecGetArrayRead(u,&src));
      CHKERRQ(PetscArraycpy(dst,src,mloc));
      CHKERRQ(VecRestoreArrayRead(u,&src));
      CHKERRQ(VecGetArrayRead(v,&src));
      CHKERRQ(PetscArraycpy(dst+mloc,src,ploc));
      CHKERRQ(VecRestoreArrayRead(v,&src));
      CHKERRQ(VecRestoreArrayWrite(uv,&dst));
      CHKERRQ(BVRestoreColumn(svd->U,i,&uv));
    }
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(VecDestroy(&u));
  } else {
    for (i=0;i<svd->nconv;i++) {
      CHKERRQ(BVGetColumn(svd->V,i,&v));
      CHKERRQ(EPSGetEigenvector(cross->eps,i,v,NULL));
      CHKERRQ(BVRestoreColumn(svd->V,i,&v));
    }
    CHKERRQ(SVDComputeVectors_Left(svd));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSMonitor_Cross(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;

  PetscFunctionBegin;
  for (i=0;i<PetscMin(nest,svd->ncv);i++) {
    er = eigr[i]; ei = eigi[i];
    CHKERRQ(STBackTransform(eps->st,1,&er,&ei));
    svd->sigma[i] = PetscSqrtReal(PetscAbsReal(PetscRealPart(er)));
    svd->errest[i] = errest[i];
  }
  CHKERRQ(SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_Cross(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscBool      set,val;
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  ST             st;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SVD Cross Options"));

    CHKERRQ(PetscOptionsBool("-svd_cross_explicitmatrix","Use cross explicit matrix","SVDCrossSetExplicitMatrix",cross->explicitmatrix,&val,&set));
    if (set) CHKERRQ(SVDCrossSetExplicitMatrix(svd,val));

  CHKERRQ(PetscOptionsTail());

  if (!cross->eps) CHKERRQ(SVDCrossGetEPS(svd,&cross->eps));
  if (!cross->explicitmatrix && !cross->usereps) {
    /* use as default an ST with shell matrix and Jacobi */
    CHKERRQ(EPSGetST(cross->eps,&st));
    CHKERRQ(STSetMatMode(st,ST_MATMODE_SHELL));
  }
  CHKERRQ(EPSSetFromOptions(cross->eps));
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
+  svd         - singular value solver
-  explicitmat - boolean flag indicating if A^T*A is built explicitly

   Options Database Key:
.  -svd_cross_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: SVDCrossGetExplicitMatrix()
@*/
PetscErrorCode SVDCrossSetExplicitMatrix(SVD svd,PetscBool explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmat,2);
  CHKERRQ(PetscTryMethod(svd,"SVDCrossSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossGetExplicitMatrix_Cross(SVD svd,PetscBool *explicitmat)
{
  SVD_CROSS *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  *explicitmat = cross->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   SVDCrossGetExplicitMatrix - Returns the flag indicating if A^T*A is built explicitly.

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicitmat - the mode flag

   Level: advanced

.seealso: SVDCrossSetExplicitMatrix()
@*/
PetscErrorCode SVDCrossGetExplicitMatrix(SVD svd,PetscBool *explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(explicitmat,2);
  CHKERRQ(PetscUseMethod(svd,"SVDCrossGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossSetEPS_Cross(SVD svd,EPS eps)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)eps));
  CHKERRQ(EPSDestroy(&cross->eps));
  cross->eps = eps;
  cross->usereps = PETSC_TRUE;
  CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->eps));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  CHKERRQ(PetscTryMethod(svd,"SVDCrossSetEPS_C",(SVD,EPS),(svd,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDCrossGetEPS_Cross(SVD svd,EPS *eps)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  if (!cross->eps) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)svd),&cross->eps));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)cross->eps,(PetscObject)svd,1));
    CHKERRQ(EPSSetOptionsPrefix(cross->eps,((PetscObject)svd)->prefix));
    CHKERRQ(EPSAppendOptionsPrefix(cross->eps,"svd_cross_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)svd,(PetscObject)cross->eps));
    CHKERRQ(PetscObjectSetOptions((PetscObject)cross->eps,((PetscObject)svd)->options));
    CHKERRQ(EPSSetWhichEigenpairs(cross->eps,EPS_LARGEST_REAL));
    CHKERRQ(EPSMonitorSet(cross->eps,EPSMonitor_Cross,svd,NULL));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(eps,2);
  CHKERRQ(PetscUseMethod(svd,"SVDCrossGetEPS_C",(SVD,EPS*),(svd,eps)));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_Cross(SVD svd,PetscViewer viewer)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (!cross->eps) CHKERRQ(SVDCrossGetEPS(svd,&cross->eps));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %s matrix\n",cross->explicitmatrix?"explicit":"implicit"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(EPSView(cross->eps,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_Cross(SVD svd)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(EPSReset(cross->eps));
  CHKERRQ(MatDestroy(&cross->C));
  CHKERRQ(MatDestroy(&cross->D));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Cross(SVD svd)
{
  SVD_CROSS      *cross = (SVD_CROSS*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(EPSDestroy(&cross->eps));
  CHKERRQ(PetscFree(svd->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetExplicitMatrix_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetExplicitMatrix_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Cross(SVD svd)
{
  SVD_CROSS      *cross;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(svd,&cross));
  svd->data = (void*)cross;

  svd->ops->solve          = SVDSolve_Cross;
  svd->ops->solveg         = SVDSolve_Cross;
  svd->ops->setup          = SVDSetUp_Cross;
  svd->ops->setfromoptions = SVDSetFromOptions_Cross;
  svd->ops->destroy        = SVDDestroy_Cross;
  svd->ops->reset          = SVDReset_Cross;
  svd->ops->view           = SVDView_Cross;
  svd->ops->computevectors = SVDComputeVectors_Cross;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetEPS_C",SVDCrossSetEPS_Cross));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetEPS_C",SVDCrossGetEPS_Cross));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossSetExplicitMatrix_C",SVDCrossSetExplicitMatrix_Cross));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDCrossGetExplicitMatrix_C",SVDCrossGetExplicitMatrix_Cross));
  PetscFunctionReturn(0);
}
