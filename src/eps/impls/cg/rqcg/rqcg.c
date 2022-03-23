/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "rqcg"

   Method: Rayleigh Quotient Conjugate Gradient

   Algorithm:

       Conjugate Gradient minimization of the Rayleigh quotient with
       periodic Rayleigh-Ritz acceleration.

   References:

       [1] L. Bergamaschi et al., "Parallel preconditioned conjugate gradient
           optimization of the Rayleigh quotient for the solution of sparse
           eigenproblems", Appl. Math. Comput. 175(2):1694-1715, 2006.
*/

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/

PetscErrorCode EPSSolve_RQCG(EPS);

typedef struct {
  PetscInt nrest;         /* user-provided reset parameter */
  PetscInt allocsize;     /* number of columns of work BV's allocated at setup */
  BV       AV,W,P,G;
} EPS_RQCG;

PetscErrorCode EPSSetUp_RQCG(EPS eps)
{
  PetscInt       nmat;
  EPS_RQCG       *ctx = (EPS_RQCG*)eps->data;

  PetscFunctionBegin;
  EPSCheckHermitianDefinite(eps);
  CHKERRQ(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  PetscCheck(eps->which==EPS_SMALLEST_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only smallest real eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION);
  EPSCheckIgnored(eps,EPS_FEATURE_BALANCE);

  if (!ctx->nrest) ctx->nrest = 20;

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(EPS_SetInnerProduct(eps));

  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  if (!ctx->allocsize) {
    ctx->allocsize = eps->mpd;
    CHKERRQ(BVDuplicateResize(eps->V,eps->mpd,&ctx->AV));
    CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->AV));
    if (nmat>1) {
      CHKERRQ(BVDuplicate(ctx->AV,&ctx->W));
      CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->W));
    }
    CHKERRQ(BVDuplicate(ctx->AV,&ctx->P));
    CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->P));
    CHKERRQ(BVDuplicate(ctx->AV,&ctx->G));
    CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->G));
  } else if (ctx->allocsize!=eps->mpd) {
    ctx->allocsize = eps->mpd;
    CHKERRQ(BVResize(ctx->AV,eps->mpd,PETSC_FALSE));
    if (nmat>1) CHKERRQ(BVResize(ctx->W,eps->mpd,PETSC_FALSE));
    CHKERRQ(BVResize(ctx->P,eps->mpd,PETSC_FALSE));
    CHKERRQ(BVResize(ctx->G,eps->mpd,PETSC_FALSE));
  }
  CHKERRQ(DSSetType(eps->ds,DSHEP));
  CHKERRQ(DSAllocate(eps->ds,eps->ncv));
  CHKERRQ(EPSSetWorkVecs(eps,1));
  PetscFunctionReturn(0);
}

/*
   ExtractSubmatrix - Returns B = A(k+1:end,k+1:end).
*/
static PetscErrorCode ExtractSubmatrix(Mat A,PetscInt k,Mat *B)
{
  PetscInt          j,m,n;
  PetscScalar       *pB;
  const PetscScalar *pA;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&m,&n));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,m-k,n-k,NULL,B));
  CHKERRQ(MatDenseGetArrayRead(A,&pA));
  CHKERRQ(MatDenseGetArrayWrite(*B,&pB));
  for (j=k;j<n;j++) CHKERRQ(PetscArraycpy(pB+(j-k)*(m-k),pA+j*m+k,m-k));
  CHKERRQ(MatDenseRestoreArrayRead(A,&pA));
  CHKERRQ(MatDenseRestoreArrayWrite(*B,&pB));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_RQCG(EPS eps)
{
  EPS_RQCG       *ctx = (EPS_RQCG*)eps->data;
  PetscInt       i,j,k,ld,nv,ncv = eps->ncv,kini,nmat;
  PetscScalar    *C,*gamma,g,pap,pbp,pbx,pax,nu,mu,alpha,beta;
  PetscReal      resnorm,a,b,c,d,disc,t;
  PetscBool      reset;
  Mat            A,B,Q,Q1;
  Vec            v,av,bv,p,w=eps->work[0];

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  if (nmat>1) CHKERRQ(STGetMatrix(eps->st,1,&B));
  else B = NULL;
  CHKERRQ(PetscMalloc1(eps->mpd,&gamma));

  kini = eps->nini;
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,0));
    for (;kini<nv;kini++) { /* Generate more initial vectors if necessary */
      CHKERRQ(BVSetRandomColumn(eps->V,kini));
      CHKERRQ(BVOrthonormalizeColumn(eps->V,kini,PETSC_TRUE,NULL,NULL));
    }
    reset = (eps->its>1 && (eps->its-1)%ctx->nrest==0)? PETSC_TRUE: PETSC_FALSE;

    if (reset) {
      /* Prevent BVDotVec below to use B-product, restored at the end */
      CHKERRQ(BVSetMatrix(eps->V,NULL,PETSC_FALSE));

      /* Compute Rayleigh quotient */
      CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));
      CHKERRQ(BVSetActiveColumns(ctx->AV,0,nv-eps->nconv));
      CHKERRQ(BVMatMult(eps->V,A,ctx->AV));
      CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&C));
      for (i=eps->nconv;i<nv;i++) {
        CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,i+1));
        CHKERRQ(BVGetColumn(ctx->AV,i-eps->nconv,&av));
        CHKERRQ(BVDotVec(eps->V,av,C+eps->nconv+i*ld));
        CHKERRQ(BVRestoreColumn(ctx->AV,i-eps->nconv,&av));
        for (j=eps->nconv;j<i-1;j++) C[i+j*ld] = PetscConj(C[j+i*ld]);
      }
      CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&C));
      CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));

      /* Solve projected problem */
      CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
      CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
      CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

      /* Update vectors V(:,idx) = V * Y(:,idx) */
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&Q));
      CHKERRQ(BVMultInPlace(eps->V,Q,eps->nconv,nv));
      CHKERRQ(ExtractSubmatrix(Q,eps->nconv,&Q1));
      CHKERRQ(BVMultInPlace(ctx->AV,Q1,0,nv-eps->nconv));
      CHKERRQ(MatDestroy(&Q));
      CHKERRQ(MatDestroy(&Q1));
      if (B) CHKERRQ(BVSetMatrix(eps->V,B,PETSC_FALSE));
    } else {
      /* No need to do Rayleigh-Ritz, just take diag(V'*A*V) */
      for (i=eps->nconv;i<nv;i++) {
        CHKERRQ(BVGetColumn(eps->V,i,&v));
        CHKERRQ(BVGetColumn(ctx->AV,i-eps->nconv,&av));
        CHKERRQ(MatMult(A,v,av));
        CHKERRQ(VecDot(av,v,eps->eigr+i));
        CHKERRQ(BVRestoreColumn(eps->V,i,&v));
        CHKERRQ(BVRestoreColumn(ctx->AV,i-eps->nconv,&av));
      }
    }

    /* Compute gradient and check convergence */
    k = -1;
    for (i=eps->nconv;i<nv;i++) {
      CHKERRQ(BVGetColumn(eps->V,i,&v));
      CHKERRQ(BVGetColumn(ctx->AV,i-eps->nconv,&av));
      CHKERRQ(BVGetColumn(ctx->G,i-eps->nconv,&p));
      if (B) {
        CHKERRQ(BVGetColumn(ctx->W,i-eps->nconv,&bv));
        CHKERRQ(MatMult(B,v,bv));
        CHKERRQ(VecWAXPY(p,-eps->eigr[i],bv,av));
        CHKERRQ(BVRestoreColumn(ctx->W,i-eps->nconv,&bv));
      } else CHKERRQ(VecWAXPY(p,-eps->eigr[i],v,av));
      CHKERRQ(BVRestoreColumn(eps->V,i,&v));
      CHKERRQ(BVRestoreColumn(ctx->AV,i-eps->nconv,&av));
      CHKERRQ(VecNorm(p,NORM_2,&resnorm));
      CHKERRQ(BVRestoreColumn(ctx->G,i-eps->nconv,&p));
      CHKERRQ((*eps->converged)(eps,eps->eigr[i],0.0,resnorm,&eps->errest[i],eps->convergedctx));
      if (k==-1 && eps->errest[i] >= eps->tol) k = i;
    }
    if (k==-1) k = nv;
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx));

    /* The next lines are necessary to avoid DS zeroing eigr */
    CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&C));
    for (i=eps->nconv;i<k;i++) C[i+i*ld] = eps->eigr[i];
    CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&C));

    if (eps->reason == EPS_CONVERGED_ITERATING) {

      /* Search direction */
      for (i=0;i<nv-eps->nconv;i++) {
        CHKERRQ(BVGetColumn(ctx->G,i,&v));
        CHKERRQ(STApply(eps->st,v,w));
        CHKERRQ(VecDot(w,v,&g));
        CHKERRQ(BVRestoreColumn(ctx->G,i,&v));
        beta = (!reset && eps->its>1)? g/gamma[i]: 0.0;
        gamma[i] = g;
        CHKERRQ(BVGetColumn(ctx->P,i,&v));
        CHKERRQ(VecAXPBY(v,1.0,beta,w));
        if (i+eps->nconv>0) {
          CHKERRQ(BVSetActiveColumns(eps->V,0,i+eps->nconv));
          CHKERRQ(BVOrthogonalizeVec(eps->V,v,NULL,NULL,NULL));
        }
        CHKERRQ(BVRestoreColumn(ctx->P,i,&v));
      }

      /* Minimization problem */
      for (i=eps->nconv;i<nv;i++) {
        CHKERRQ(BVGetColumn(eps->V,i,&v));
        CHKERRQ(BVGetColumn(ctx->AV,i-eps->nconv,&av));
        CHKERRQ(BVGetColumn(ctx->P,i-eps->nconv,&p));
        CHKERRQ(VecDot(av,v,&nu));
        CHKERRQ(VecDot(av,p,&pax));
        CHKERRQ(MatMult(A,p,w));
        CHKERRQ(VecDot(w,p,&pap));
        if (B) {
          CHKERRQ(BVGetColumn(ctx->W,i-eps->nconv,&bv));
          CHKERRQ(VecDot(bv,v,&mu));
          CHKERRQ(VecDot(bv,p,&pbx));
          CHKERRQ(BVRestoreColumn(ctx->W,i-eps->nconv,&bv));
          CHKERRQ(MatMult(B,p,w));
          CHKERRQ(VecDot(w,p,&pbp));
        } else {
          CHKERRQ(VecDot(v,v,&mu));
          CHKERRQ(VecDot(v,p,&pbx));
          CHKERRQ(VecDot(p,p,&pbp));
        }
        CHKERRQ(BVRestoreColumn(ctx->AV,i-eps->nconv,&av));
        a = PetscRealPart(pap*pbx-pax*pbp);
        b = PetscRealPart(nu*pbp-mu*pap);
        c = PetscRealPart(mu*pax-nu*pbx);
        t = PetscMax(PetscMax(PetscAbsReal(a),PetscAbsReal(b)),PetscAbsReal(c));
        if (t!=0.0) { a /= t; b /= t; c /= t; }
        disc = b*b-4.0*a*c;
        d = PetscSqrtReal(PetscAbsReal(disc));
        if (b>=0.0 && a!=0.0) alpha = (b+d)/(2.0*a);
        else if (b!=d) alpha = 2.0*c/(b-d);
        else alpha = 0;
        /* Next iterate */
        if (alpha!=0.0) CHKERRQ(VecAXPY(v,alpha,p));
        CHKERRQ(BVRestoreColumn(eps->V,i,&v));
        CHKERRQ(BVRestoreColumn(ctx->P,i-eps->nconv,&p));
        CHKERRQ(BVOrthonormalizeColumn(eps->V,i,PETSC_TRUE,NULL,NULL));
      }
    }

    CHKERRQ(EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv));
    eps->nconv = k;
  }

  CHKERRQ(PetscFree(gamma));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSRQCGSetReset_RQCG(EPS eps,PetscInt nrest)
{
  EPS_RQCG *ctx = (EPS_RQCG*)eps->data;

  PetscFunctionBegin;
  if (nrest==PETSC_DEFAULT) {
    ctx->nrest = 0;
    eps->state = EPS_STATE_INITIAL;
  } else {
    PetscCheck(nrest>0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reset parameter must be >0");
    ctx->nrest = nrest;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSRQCGSetReset - Sets the reset parameter of the RQCG iteration. Every
   nrest iterations, the solver performs a Rayleigh-Ritz projection step.

   Logically Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
-  nrest - the number of iterations between resets

   Options Database Key:
.  -eps_rqcg_reset - Sets the reset parameter

   Level: advanced

.seealso: EPSRQCGGetReset()
@*/
PetscErrorCode EPSRQCGSetReset(EPS eps,PetscInt nrest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,nrest,2);
  CHKERRQ(PetscTryMethod(eps,"EPSRQCGSetReset_C",(EPS,PetscInt),(eps,nrest)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSRQCGGetReset_RQCG(EPS eps,PetscInt *nrest)
{
  EPS_RQCG *ctx = (EPS_RQCG*)eps->data;

  PetscFunctionBegin;
  *nrest = ctx->nrest;
  PetscFunctionReturn(0);
}

/*@
   EPSRQCGGetReset - Gets the reset parameter used in the RQCG method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  nrest - the reset parameter

   Level: advanced

.seealso: EPSRQCGSetReset()
@*/
PetscErrorCode EPSRQCGGetReset(EPS eps,PetscInt *nrest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(nrest,2);
  CHKERRQ(PetscUseMethod(eps,"EPSRQCGGetReset_C",(EPS,PetscInt*),(eps,nrest)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_RQCG(EPS eps)
{
  EPS_RQCG       *ctx = (EPS_RQCG*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(BVDestroy(&ctx->AV));
  CHKERRQ(BVDestroy(&ctx->W));
  CHKERRQ(BVDestroy(&ctx->P));
  CHKERRQ(BVDestroy(&ctx->G));
  ctx->allocsize = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_RQCG(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscBool      flg;
  PetscInt       nrest;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS RQCG Options"));

    CHKERRQ(PetscOptionsInt("-eps_rqcg_reset","Reset parameter","EPSRQCGSetReset",20,&nrest,&flg));
    if (flg) CHKERRQ(EPSRQCGSetReset(eps,nrest));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_RQCG(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSRQCGSetReset_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSRQCGGetReset_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_RQCG(EPS eps,PetscViewer viewer)
{
  EPS_RQCG       *ctx = (EPS_RQCG*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  reset every %" PetscInt_FMT " iterations\n",ctx->nrest));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_RQCG(EPS eps)
{
  EPS_RQCG       *rqcg;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&rqcg));
  eps->data = (void*)rqcg;

  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_PRECOND;

  eps->ops->solve          = EPSSolve_RQCG;
  eps->ops->setup          = EPSSetUp_RQCG;
  eps->ops->setupsort      = EPSSetUpSort_Default;
  eps->ops->setfromoptions = EPSSetFromOptions_RQCG;
  eps->ops->destroy        = EPSDestroy_RQCG;
  eps->ops->reset          = EPSReset_RQCG;
  eps->ops->view           = EPSView_RQCG;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_GMRES;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSRQCGSetReset_C",EPSRQCGSetReset_RQCG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSRQCGGetReset_C",EPSRQCGGetReset_RQCG));
  PetscFunctionReturn(0);
}
