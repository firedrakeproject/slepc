/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc quadratic eigensolver: "qarnoldi"

   Method: Q-Arnoldi

   Algorithm:

       Quadratic Arnoldi with Krylov-Schur type restart.

   References:

       [1] K. Meerbergen, "The Quadratic Arnoldi method for the solution
           of the quadratic eigenvalue problem", SIAM J. Matrix Anal.
           Appl. 30(4):1462-1482, 2008.
*/

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include <petscblaslapack.h>

typedef struct {
  PetscReal keep;         /* restart parameter */
  PetscBool lock;         /* locking/non-locking variant */
} PEP_QARNOLDI;

PetscErrorCode PEPSetUp_QArnoldi(PEP pep)
{
  PEP_QARNOLDI   *ctx = (PEP_QARNOLDI*)pep->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PEPCheckQuadratic(pep);
  PEPCheckShiftSinvert(pep);
  CHKERRQ(PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd));
  PetscCheck(ctx->lock || pep->mpd>=pep->ncv,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (pep->max_it==PETSC_DEFAULT) pep->max_it = PetscMax(100,4*pep->n/pep->ncv);
  if (!pep->which) CHKERRQ(PEPSetWhichEigenpairs_Default(pep));
  PetscCheck(pep->which!=PEP_ALL,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues");

  CHKERRQ(STGetTransform(pep->st,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag set, see STSetTransform()");

  /* set default extraction */
  if (!pep->extract) pep->extract = PEP_EXTRACT_NONE;
  PEPCheckUnsupported(pep,PEP_FEATURE_NONMONOMIAL | PEP_FEATURE_EXTRACT);

  if (!ctx->keep) ctx->keep = 0.5;

  CHKERRQ(PEPAllocateSolution(pep,0));
  CHKERRQ(PEPSetWorkVecs(pep,4));

  CHKERRQ(DSSetType(pep->ds,DSNHEP));
  CHKERRQ(DSSetExtraRow(pep->ds,PETSC_TRUE));
  CHKERRQ(DSAllocate(pep->ds,pep->ncv+1));

  PetscFunctionReturn(0);
}

PetscErrorCode PEPExtractVectors_QArnoldi(PEP pep)
{
  PetscInt       i,k=pep->nconv,ldds;
  PetscScalar    *X,*pX0;
  Mat            X0;

  PetscFunctionBegin;
  if (pep->nconv==0) PetscFunctionReturn(0);
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ldds));
  CHKERRQ(DSVectors(pep->ds,DS_MAT_X,NULL,NULL));
  CHKERRQ(DSGetArray(pep->ds,DS_MAT_X,&X));

  /* update vectors V = V*X */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&X0));
  CHKERRQ(MatDenseGetArrayWrite(X0,&pX0));
  for (i=0;i<k;i++) CHKERRQ(PetscArraycpy(pX0+i*k,X+i*ldds,k));
  CHKERRQ(MatDenseRestoreArrayWrite(X0,&pX0));
  CHKERRQ(BVMultInPlace(pep->V,X0,0,k));
  CHKERRQ(MatDestroy(&X0));
  CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_X,&X));
  PetscFunctionReturn(0);
}

/*
  Compute a step of Classical Gram-Schmidt orthogonalization
*/
static PetscErrorCode PEPQArnoldiCGS(PEP pep,PetscScalar *H,PetscBLASInt ldh,PetscScalar *h,PetscBLASInt j,BV V,Vec t,Vec v,Vec w,PetscReal *onorm,PetscReal *norm,PetscScalar *work)
{
  PetscBLASInt   ione = 1,j_1 = j+1;
  PetscReal      x,y;
  PetscScalar    dot,one = 1.0,zero = 0.0;

  PetscFunctionBegin;
  /* compute norm of v and w */
  if (onorm) {
    CHKERRQ(VecNorm(v,NORM_2,&x));
    CHKERRQ(VecNorm(w,NORM_2,&y));
    *onorm = PetscSqrtReal(x*x+y*y);
  }

  /* orthogonalize: compute h */
  CHKERRQ(BVDotVec(V,v,h));
  CHKERRQ(BVDotVec(V,w,work));
  if (j>0)
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&j_1,&j,&one,H,&ldh,work,&ione,&one,h,&ione));
  CHKERRQ(VecDot(w,t,&dot));
  h[j] += dot;

  /* orthogonalize: update v and w */
  CHKERRQ(BVMultVec(V,-1.0,1.0,v,h));
  if (j>0) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&j_1,&j,&one,H,&ldh,h,&ione,&zero,work,&ione));
    CHKERRQ(BVMultVec(V,-1.0,1.0,w,work));
  }
  CHKERRQ(VecAXPY(w,-h[j],t));

  /* compute norm of v and w */
  if (norm) {
    CHKERRQ(VecNorm(v,NORM_2,&x));
    CHKERRQ(VecNorm(w,NORM_2,&y));
    *norm = PetscSqrtReal(x*x+y*y);
  }
  PetscFunctionReturn(0);
}

/*
  Compute a run of Q-Arnoldi iterations
*/
static PetscErrorCode PEPQArnoldi(PEP pep,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,Vec v,Vec w,PetscReal *beta,PetscBool *breakdown,PetscScalar *work)
{
  PetscInt           i,j,l,m = *M;
  Vec                t = pep->work[2],u = pep->work[3];
  BVOrthogRefineType refinement;
  PetscReal          norm=0.0,onorm,eta;
  PetscScalar        *c = work + m;

  PetscFunctionBegin;
  CHKERRQ(BVGetOrthogonalization(pep->V,NULL,&refinement,&eta,NULL));
  CHKERRQ(BVInsertVec(pep->V,k,v));
  for (j=k;j<m;j++) {
    /* apply operator */
    CHKERRQ(VecCopy(w,t));
    if (pep->Dr) CHKERRQ(VecPointwiseMult(v,v,pep->Dr));
    CHKERRQ(STMatMult(pep->st,0,v,u));
    CHKERRQ(VecCopy(t,v));
    if (pep->Dr) CHKERRQ(VecPointwiseMult(t,t,pep->Dr));
    CHKERRQ(STMatMult(pep->st,1,t,w));
    CHKERRQ(VecAXPY(u,pep->sfactor,w));
    CHKERRQ(STMatSolve(pep->st,u,w));
    CHKERRQ(VecScale(w,-1.0/(pep->sfactor*pep->sfactor)));
    if (pep->Dr) CHKERRQ(VecPointwiseDivide(w,w,pep->Dr));
    CHKERRQ(VecCopy(v,t));
    CHKERRQ(BVSetActiveColumns(pep->V,0,j+1));

    /* orthogonalize */
    switch (refinement) {
      case BV_ORTHOG_REFINE_NEVER:
        CHKERRQ(PEPQArnoldiCGS(pep,H,ldh,H+ldh*j,j,pep->V,t,v,w,NULL,&norm,work));
        *breakdown = PETSC_FALSE;
        break;
      case BV_ORTHOG_REFINE_ALWAYS:
        CHKERRQ(PEPQArnoldiCGS(pep,H,ldh,H+ldh*j,j,pep->V,t,v,w,NULL,NULL,work));
        CHKERRQ(PEPQArnoldiCGS(pep,H,ldh,c,j,pep->V,t,v,w,&onorm,&norm,work));
        for (i=0;i<=j;i++) H[ldh*j+i] += c[i];
        if (norm < eta * onorm) *breakdown = PETSC_TRUE;
        else *breakdown = PETSC_FALSE;
        break;
      case BV_ORTHOG_REFINE_IFNEEDED:
        CHKERRQ(PEPQArnoldiCGS(pep,H,ldh,H+ldh*j,j,pep->V,t,v,w,&onorm,&norm,work));
        /* ||q|| < eta ||h|| */
        l = 1;
        while (l<3 && norm < eta * onorm) {
          l++;
          onorm = norm;
          CHKERRQ(PEPQArnoldiCGS(pep,H,ldh,c,j,pep->V,t,v,w,NULL,&norm,work));
          for (i=0;i<=j;i++) H[ldh*j+i] += c[i];
        }
        if (norm < eta * onorm) *breakdown = PETSC_TRUE;
        else *breakdown = PETSC_FALSE;
        break;
    }
    CHKERRQ(VecScale(v,1.0/norm));
    CHKERRQ(VecScale(w,1.0/norm));

    H[j+1+ldh*j] = norm;
    if (j<m-1) CHKERRQ(BVInsertVec(pep->V,j+1,v));
  }
  *beta = norm;
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSolve_QArnoldi(PEP pep)
{
  PEP_QARNOLDI   *ctx = (PEP_QARNOLDI*)pep->data;
  PetscInt       j,k,l,lwork,nv,ld,nconv;
  Vec            v=pep->work[0],w=pep->work[1];
  Mat            Q;
  PetscScalar    *S,*work;
  PetscReal      beta=0.0,norm,x,y;
  PetscBool      breakdown=PETSC_FALSE,sinv;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ld));
  lwork = 7*pep->ncv;
  CHKERRQ(PetscMalloc1(lwork,&work));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv));
  CHKERRQ(RGPushScale(pep->rg,sinv?pep->sfactor:1.0/pep->sfactor));
  CHKERRQ(STScaleShift(pep->st,sinv?pep->sfactor:1.0/pep->sfactor));

  /* Get the starting Arnoldi vector */
  for (j=0;j<2;j++) {
    if (j>=pep->nini) CHKERRQ(BVSetRandomColumn(pep->V,j));
  }
  CHKERRQ(BVCopyVec(pep->V,0,v));
  CHKERRQ(BVCopyVec(pep->V,1,w));
  CHKERRQ(VecNorm(v,NORM_2,&x));
  CHKERRQ(VecNorm(w,NORM_2,&y));
  norm = PetscSqrtReal(x*x+y*y);
  CHKERRQ(VecScale(v,1.0/norm));
  CHKERRQ(VecScale(w,1.0/norm));

  /* clean projected matrix (including the extra-arrow) */
  CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&S));
  CHKERRQ(PetscArrayzero(S,ld*ld));
  CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&S));

   /* Restart loop */
  l = 0;
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(pep->nconv+pep->mpd,pep->ncv);
    CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&S));
    CHKERRQ(PEPQArnoldi(pep,S,ld,pep->nconv+l,&nv,v,w,&beta,&breakdown,work));
    CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&S));
    CHKERRQ(DSSetDimensions(pep->ds,nv,pep->nconv,pep->nconv+l));
    CHKERRQ(DSSetState(pep->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    CHKERRQ(BVSetActiveColumns(pep->V,pep->nconv,nv));

    /* Solve projected problem */
    CHKERRQ(DSSolve(pep->ds,pep->eigr,pep->eigi));
    CHKERRQ(DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL));
    CHKERRQ(DSUpdateExtraRow(pep->ds));
    CHKERRQ(DSSynchronize(pep->ds,pep->eigr,pep->eigi));

    /* Check convergence */
    CHKERRQ(PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,nv-pep->nconv,beta,&k));
    CHKERRQ((*pep->stopping)(pep,pep->its,pep->max_it,k,pep->nev,&pep->reason,pep->stoppingctx));
    nconv = k;

    /* Update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      CHKERRQ(DSGetTruncateSize(pep->ds,k,nv,&l));
    }
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) CHKERRQ(PetscInfo(pep,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (pep->reason == PEP_CONVERGED_ITERATING) {
      if (PetscUnlikely(breakdown)) {
        /* Stop if breakdown */
        CHKERRQ(PetscInfo(pep,"Breakdown Quadratic Arnoldi method (it=%" PetscInt_FMT " norm=%g)\n",pep->its,(double)beta));
        pep->reason = PEP_DIVERGED_BREAKDOWN;
      } else {
        /* Prepare the Rayleigh quotient for restart */
        CHKERRQ(DSTruncate(pep->ds,k+l,PETSC_FALSE));
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    CHKERRQ(DSGetMat(pep->ds,DS_MAT_Q,&Q));
    CHKERRQ(BVMultInPlace(pep->V,Q,pep->nconv,k+l));
    CHKERRQ(MatDestroy(&Q));

    pep->nconv = k;
    CHKERRQ(PEPMonitor(pep,pep->its,nconv,pep->eigr,pep->eigi,pep->errest,nv));
  }
  CHKERRQ(BVSetActiveColumns(pep->V,0,pep->nconv));
  for (j=0;j<pep->nconv;j++) {
    pep->eigr[j] *= pep->sfactor;
    pep->eigi[j] *= pep->sfactor;
  }

  CHKERRQ(STScaleShift(pep->st,sinv?1.0/pep->sfactor:pep->sfactor));
  CHKERRQ(RGPopScale(pep->rg));

  CHKERRQ(DSTruncate(pep->ds,pep->nconv,PETSC_TRUE));
  CHKERRQ(PetscFree(work));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPQArnoldiSetRestart_QArnoldi(PEP pep,PetscReal keep)
{
  PEP_QARNOLDI *ctx = (PEP_QARNOLDI*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    PetscCheck(keep>=0.1 && keep<=0.9,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPQArnoldiSetRestart - Sets the restart parameter for the Q-Arnoldi
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on pep

   Input Parameters:
+  pep  - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -pep_qarnoldi_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: PEPQArnoldiGetRestart()
@*/
PetscErrorCode PEPQArnoldiSetRestart(PEP pep,PetscReal keep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  CHKERRQ(PetscTryMethod(pep,"PEPQArnoldiSetRestart_C",(PEP,PetscReal),(pep,keep)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPQArnoldiGetRestart_QArnoldi(PEP pep,PetscReal *keep)
{
  PEP_QARNOLDI *ctx = (PEP_QARNOLDI*)pep->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

/*@
   PEPQArnoldiGetRestart - Gets the restart parameter used in the Q-Arnoldi method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: PEPQArnoldiSetRestart()
@*/
PetscErrorCode PEPQArnoldiGetRestart(PEP pep,PetscReal *keep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidRealPointer(keep,2);
  CHKERRQ(PetscUseMethod(pep,"PEPQArnoldiGetRestart_C",(PEP,PetscReal*),(pep,keep)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPQArnoldiSetLocking_QArnoldi(PEP pep,PetscBool lock)
{
  PEP_QARNOLDI *ctx = (PEP_QARNOLDI*)pep->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

/*@
   PEPQArnoldiSetLocking - Choose between locking and non-locking variants of
   the Q-Arnoldi method.

   Logically Collective on pep

   Input Parameters:
+  pep  - the eigenproblem solver context
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -pep_qarnoldi_locking - Sets the locking flag

   Notes:
   The default is to lock converged eigenpairs when the method restarts.
   This behaviour can be changed so that all directions are kept in the
   working subspace even if already converged to working accuracy (the
   non-locking variant).

   Level: advanced

.seealso: PEPQArnoldiGetLocking()
@*/
PetscErrorCode PEPQArnoldiSetLocking(PEP pep,PetscBool lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,lock,2);
  CHKERRQ(PetscTryMethod(pep,"PEPQArnoldiSetLocking_C",(PEP,PetscBool),(pep,lock)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPQArnoldiGetLocking_QArnoldi(PEP pep,PetscBool *lock)
{
  PEP_QARNOLDI *ctx = (PEP_QARNOLDI*)pep->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

/*@
   PEPQArnoldiGetLocking - Gets the locking flag used in the Q-Arnoldi method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: PEPQArnoldiSetLocking()
@*/
PetscErrorCode PEPQArnoldiGetLocking(PEP pep,PetscBool *lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidBoolPointer(lock,2);
  CHKERRQ(PetscUseMethod(pep,"PEPQArnoldiGetLocking_C",(PEP,PetscBool*),(pep,lock)));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetFromOptions_QArnoldi(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscBool      flg,lock;
  PetscReal      keep;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"PEP Q-Arnoldi Options"));

    CHKERRQ(PetscOptionsReal("-pep_qarnoldi_restart","Proportion of vectors kept after restart","PEPQArnoldiSetRestart",0.5,&keep,&flg));
    if (flg) CHKERRQ(PEPQArnoldiSetRestart(pep,keep));

    CHKERRQ(PetscOptionsBool("-pep_qarnoldi_locking","Choose between locking and non-locking variants","PEPQArnoldiSetLocking",PETSC_FALSE,&lock,&flg));
    if (flg) CHKERRQ(PEPQArnoldiSetLocking(pep,lock));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode PEPView_QArnoldi(PEP pep,PetscViewer viewer)
{
  PEP_QARNOLDI   *ctx = (PEP_QARNOLDI*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep)));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using the %slocking variant\n",ctx->lock?"":"non-"));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPDestroy_QArnoldi(PEP pep)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(pep->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiSetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiGetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiSetLocking_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiGetLocking_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode PEPCreate_QArnoldi(PEP pep)
{
  PEP_QARNOLDI   *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pep,&ctx));
  pep->data = (void*)ctx;

  pep->lineariz = PETSC_TRUE;
  ctx->lock     = PETSC_TRUE;

  pep->ops->solve          = PEPSolve_QArnoldi;
  pep->ops->setup          = PEPSetUp_QArnoldi;
  pep->ops->setfromoptions = PEPSetFromOptions_QArnoldi;
  pep->ops->destroy        = PEPDestroy_QArnoldi;
  pep->ops->view           = PEPView_QArnoldi;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_QArnoldi;
  pep->ops->setdefaultst   = PEPSetDefaultST_Transform;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiSetRestart_C",PEPQArnoldiSetRestart_QArnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiGetRestart_C",PEPQArnoldiGetRestart_QArnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiSetLocking_C",PEPQArnoldiSetLocking_QArnoldi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPQArnoldiGetLocking_C",PEPQArnoldiGetLocking_QArnoldi));
  PetscFunctionReturn(0);
}
