/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "lobpcg"

   Method: Locally Optimal Block Preconditioned Conjugate Gradient

   Algorithm:

       LOBPCG with soft and hard locking. Follows the implementation
       in BLOPEX [2].

   References:

       [1] A. V. Knyazev, "Toward the optimal preconditioned eigensolver:
           locally optimal block preconditioned conjugate gradient method",
           SIAM J. Sci. Comput. 23(2):517-541, 2001.

       [2] A. V. Knyazev et al., "Block Locally Optimal Preconditioned
           Eigenvalue Xolvers (BLOPEX) in Hypre and PETSc", SIAM J. Sci.
           Comput. 29(5):2224-2239, 2007.
*/

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/

typedef struct {
  PetscInt  bs;        /* block size */
  PetscBool lock;      /* soft locking active/inactive */
  PetscReal restart;   /* restart parameter */
  PetscInt  guard;     /* number of guard vectors */
} EPS_LOBPCG;

PetscErrorCode EPSSetDimensions_LOBPCG(EPS eps,PetscInt nev,PetscInt *ncv,PetscInt *mpd)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;
  PetscInt   k;

  PetscFunctionBegin;
  k = PetscMax(3*ctx->bs,((eps->nev-1)/ctx->bs+3)*ctx->bs);
  if (*ncv!=PETSC_DEFAULT) { /* ncv set */
    PetscCheck(*ncv>=k,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv is not sufficiently large");
  } else *ncv = k;
  if (*mpd==PETSC_DEFAULT) *mpd = 3*ctx->bs;
  else PetscCheck(*mpd==3*ctx->bs,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"This solver does not allow a value of mpd different from 3*blocksize");
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUp_LOBPCG(EPS eps)
{
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  EPSCheckHermitianDefinite(eps);
  if (!ctx->bs) ctx->bs = PetscMin(16,eps->nev);
  PetscCheck(eps->n-eps->nds>=5*ctx->bs,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The problem size is too small relative to the block size");
  CHKERRQ(EPSSetDimensions_LOBPCG(eps,eps->nev,&eps->ncv,&eps->mpd));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  PetscCheck(eps->which==EPS_SMALLEST_REAL || eps->which==EPS_LARGEST_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only smallest real or largest real eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION);
  EPSCheckIgnored(eps,EPS_FEATURE_BALANCE);

  if (!ctx->restart) ctx->restart = 0.9;

  /* number of guard vectors */
  if (ctx->bs==1) ctx->guard = 0;
  else ctx->guard = PetscMin((PetscInt)((1.0-ctx->restart)*ctx->bs+0.45),ctx->bs-1);

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(EPS_SetInnerProduct(eps));
  CHKERRQ(DSSetType(eps->ds,DSGHEP));
  CHKERRQ(DSAllocate(eps->ds,eps->mpd));
  CHKERRQ(EPSSetWorkVecs(eps,1));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;
  PetscInt       i,j,k,ld,nv,ini,nmat,nc,nconv,locked,its,prev=0;
  PetscReal      norm;
  PetscScalar    *eigr,dot;
  PetscBool      breakdown,countc,flip=PETSC_FALSE,checkprecond=PETSC_FALSE;
  Mat            A,B,M,V=NULL,W=NULL;
  Vec            v,z,w=eps->work[0];
  BV             X,Y=NULL,Z,R,P,AX,BX;
  SlepcSC        sc;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  if (nmat>1) CHKERRQ(STGetMatrix(eps->st,1,&B));
  else B = NULL;

  if (eps->which==EPS_LARGEST_REAL) {  /* flip spectrum */
    flip = PETSC_TRUE;
    CHKERRQ(DSGetSlepcSC(eps->ds,&sc));
    sc->comparison = SlepcCompareSmallestReal;
  }

  /* undocumented option to check for a positive-definite preconditioner (turn-off by default) */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-eps_lobpcg_checkprecond",&checkprecond,NULL));

  /* 1. Allocate memory */
  CHKERRQ(PetscCalloc1(3*ctx->bs,&eigr));
  CHKERRQ(BVDuplicateResize(eps->V,3*ctx->bs,&Z));
  CHKERRQ(BVDuplicateResize(eps->V,ctx->bs,&X));
  CHKERRQ(BVDuplicateResize(eps->V,ctx->bs,&R));
  CHKERRQ(BVDuplicateResize(eps->V,ctx->bs,&P));
  CHKERRQ(BVDuplicateResize(eps->V,ctx->bs,&AX));
  if (B) CHKERRQ(BVDuplicateResize(eps->V,ctx->bs,&BX));
  nc = eps->nds;
  if (nc>0 || eps->nev>ctx->bs-ctx->guard) CHKERRQ(BVDuplicateResize(eps->V,nc+eps->nev,&Y));
  if (nc>0) {
    for (j=0;j<nc;j++) {
      CHKERRQ(BVGetColumn(eps->V,-nc+j,&v));
      CHKERRQ(BVInsertVec(Y,j,v));
      CHKERRQ(BVRestoreColumn(eps->V,-nc+j,&v));
    }
    CHKERRQ(BVSetActiveColumns(Y,0,nc));
  }

  /* 2. Apply the constraints to the initial vectors */
  /* 3. B-orthogonalize initial vectors */
  for (k=eps->nini;k<eps->ncv-ctx->bs;k++) { /* Generate more initial vectors if necessary */
    CHKERRQ(BVSetRandomColumn(eps->V,k));
    CHKERRQ(BVOrthonormalizeColumn(eps->V,k,PETSC_TRUE,NULL,NULL));
  }
  nv = ctx->bs;
  CHKERRQ(BVSetActiveColumns(eps->V,0,nv));
  CHKERRQ(BVSetActiveColumns(Z,0,nv));
  CHKERRQ(BVCopy(eps->V,Z));
  CHKERRQ(BVCopy(Z,X));

  /* 4. Compute initial Ritz vectors */
  CHKERRQ(BVMatMult(X,A,AX));
  CHKERRQ(DSSetDimensions(eps->ds,nv,0,0));
  CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&M));
  CHKERRQ(BVMatProject(AX,NULL,X,M));
  if (flip) CHKERRQ(MatScale(M,-1.0));
  CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&M));
  CHKERRQ(DSSetIdentity(eps->ds,DS_MAT_B));
  CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
  CHKERRQ(DSSolve(eps->ds,eigr,NULL));
  CHKERRQ(DSSort(eps->ds,eigr,NULL,NULL,NULL,NULL));
  CHKERRQ(DSSynchronize(eps->ds,eigr,NULL));
  for (j=0;j<nv;j++) eps->eigr[j] = flip? -eigr[j]: eigr[j];
  CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));
  CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&M));
  CHKERRQ(BVMultInPlace(X,M,0,nv));
  CHKERRQ(BVMultInPlace(AX,M,0,nv));
  CHKERRQ(MatDestroy(&M));

  /* 5. Initialize range of active iterates */
  locked = 0;  /* hard-locked vectors, the leading locked columns of V are eigenvectors */
  nconv  = 0;  /* number of converged eigenvalues in the current block */
  its    = 0;  /* iterations for the current block */

  /* 6. Main loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {

    if (ctx->lock) {
      CHKERRQ(BVSetActiveColumns(R,nconv,ctx->bs));
      CHKERRQ(BVSetActiveColumns(AX,nconv,ctx->bs));
      if (B) CHKERRQ(BVSetActiveColumns(BX,nconv,ctx->bs));
    }

    /* 7. Compute residuals */
    ini = (ctx->lock)? nconv: 0;
    CHKERRQ(BVCopy(AX,R));
    if (B) CHKERRQ(BVMatMult(X,B,BX));
    for (j=ini;j<ctx->bs;j++) {
      CHKERRQ(BVGetColumn(R,j,&v));
      CHKERRQ(BVGetColumn(B?BX:X,j,&z));
      CHKERRQ(VecAXPY(v,-eps->eigr[locked+j],z));
      CHKERRQ(BVRestoreColumn(R,j,&v));
      CHKERRQ(BVRestoreColumn(B?BX:X,j,&z));
    }

    /* 8. Compute residual norms and update index set of active iterates */
    k = ini;
    countc = PETSC_TRUE;
    for (j=ini;j<ctx->bs;j++) {
      i = locked+j;
      CHKERRQ(BVGetColumn(R,j,&v));
      CHKERRQ(VecNorm(v,NORM_2,&norm));
      CHKERRQ(BVRestoreColumn(R,j,&v));
      CHKERRQ((*eps->converged)(eps,eps->eigr[i],eps->eigi[i],norm,&eps->errest[i],eps->convergedctx));
      if (countc) {
        if (eps->errest[i] < eps->tol) k++;
        else countc = PETSC_FALSE;
      }
      if (!countc && !eps->trackall) break;
    }
    nconv = k;
    eps->nconv = locked + nconv;
    if (its) CHKERRQ(EPSMonitor(eps,eps->its+its,eps->nconv,eps->eigr,eps->eigi,eps->errest,locked+ctx->bs));
    CHKERRQ((*eps->stopping)(eps,eps->its+its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx));
    if (eps->reason != EPS_CONVERGED_ITERATING || nconv >= ctx->bs-ctx->guard) {
      CHKERRQ(BVSetActiveColumns(eps->V,locked,eps->nconv));
      CHKERRQ(BVSetActiveColumns(X,0,nconv));
      CHKERRQ(BVCopy(X,eps->V));
    }
    if (eps->reason != EPS_CONVERGED_ITERATING) {
      break;
    } else if (nconv >= ctx->bs-ctx->guard) {
      eps->its += its-1;
      its = 0;
    } else its++;

    if (nconv >= ctx->bs-ctx->guard) {  /* force hard locking of vectors and compute new R */

      /* extend constraints */
      CHKERRQ(BVSetActiveColumns(Y,nc+locked,nc+locked+nconv));
      CHKERRQ(BVCopy(X,Y));
      CHKERRQ(BVSetActiveColumns(Y,0,nc+locked+nconv));

      /* shift work BV's */
      for (j=nconv;j<ctx->bs;j++) {
        CHKERRQ(BVCopyColumn(X,j,j-nconv));
        CHKERRQ(BVCopyColumn(R,j,j-nconv));
        CHKERRQ(BVCopyColumn(P,j,j-nconv));
        CHKERRQ(BVCopyColumn(AX,j,j-nconv));
        if (B) CHKERRQ(BVCopyColumn(BX,j,j-nconv));
      }

      /* set new initial vectors */
      CHKERRQ(BVSetActiveColumns(eps->V,locked+ctx->bs,locked+ctx->bs+nconv));
      CHKERRQ(BVSetActiveColumns(X,ctx->bs-nconv,ctx->bs));
      CHKERRQ(BVCopy(eps->V,X));
      for (j=ctx->bs-nconv;j<ctx->bs;j++) {
        CHKERRQ(BVGetColumn(X,j,&v));
        CHKERRQ(BVOrthogonalizeVec(Y,v,NULL,&norm,&breakdown));
        if (norm>0.0 && !breakdown) CHKERRQ(VecScale(v,1.0/norm));
        else {
          CHKERRQ(PetscInfo(eps,"Orthogonalization of initial vector failed\n"));
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          goto diverged;
        }
        CHKERRQ(BVRestoreColumn(X,j,&v));
      }
      locked += nconv;
      nconv = 0;
      CHKERRQ(BVSetActiveColumns(X,nconv,ctx->bs));

      /* B-orthogonalize initial vectors */
      CHKERRQ(BVOrthogonalize(X,NULL));
      CHKERRQ(BVSetActiveColumns(Z,nconv,ctx->bs));
      CHKERRQ(BVSetActiveColumns(AX,nconv,ctx->bs));
      CHKERRQ(BVCopy(X,Z));

      /* compute initial Ritz vectors */
      nv = ctx->bs;
      CHKERRQ(BVMatMult(X,A,AX));
      CHKERRQ(DSSetDimensions(eps->ds,nv,0,0));
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&M));
      CHKERRQ(BVMatProject(AX,NULL,X,M));
      if (flip) CHKERRQ(MatScale(M,-1.0));
      CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&M));
      CHKERRQ(DSSetIdentity(eps->ds,DS_MAT_B));
      CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
      CHKERRQ(DSSolve(eps->ds,eigr,NULL));
      CHKERRQ(DSSort(eps->ds,eigr,NULL,NULL,NULL,NULL));
      CHKERRQ(DSSynchronize(eps->ds,eigr,NULL));
      for (j=0;j<nv;j++) if (locked+j<eps->ncv) eps->eigr[locked+j] = flip? -eigr[j]: eigr[j];
      CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&M));
      CHKERRQ(BVMultInPlace(X,M,0,nv));
      CHKERRQ(BVMultInPlace(AX,M,0,nv));
      CHKERRQ(MatDestroy(&M));

      continue;   /* skip the rest of the iteration */
    }

    ini = (ctx->lock)? nconv: 0;
    if (ctx->lock) {
      CHKERRQ(BVSetActiveColumns(R,nconv,ctx->bs));
      CHKERRQ(BVSetActiveColumns(P,nconv,ctx->bs));
      CHKERRQ(BVSetActiveColumns(AX,nconv,ctx->bs));
      if (B) CHKERRQ(BVSetActiveColumns(BX,nconv,ctx->bs));
    }

    /* 9. Apply preconditioner to the residuals */
    CHKERRQ(BVGetMat(R,&V));
    if (prev != ctx->bs-ini) {
      prev = ctx->bs-ini;
      CHKERRQ(MatDestroy(&W));
      CHKERRQ(MatDuplicate(V,MAT_SHARE_NONZERO_PATTERN,&W));
    }
    CHKERRQ(STApplyMat(eps->st,V,W));
    if (checkprecond) {
      for (j=ini;j<ctx->bs;j++) {
        CHKERRQ(MatDenseGetColumnVecRead(V,j-ini,&v));
        CHKERRQ(MatDenseGetColumnVecRead(W,j-ini,&w));
        CHKERRQ(VecDot(v,w,&dot));
        CHKERRQ(MatDenseRestoreColumnVecRead(W,j-ini,&w));
        CHKERRQ(MatDenseRestoreColumnVecRead(V,j-ini,&v));
        if (PetscRealPart(dot)<0.0) {
          CHKERRQ(PetscInfo(eps,"The preconditioner is not positive-definite\n"));
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          goto diverged;
        }
      }
    }
    if (nc+locked>0) {
      for (j=ini;j<ctx->bs;j++) {
        CHKERRQ(MatDenseGetColumnVecWrite(W,j-ini,&w));
        CHKERRQ(BVOrthogonalizeVec(Y,w,NULL,&norm,&breakdown));
        if (norm>0.0 && !breakdown) CHKERRQ(VecScale(w,1.0/norm));
        CHKERRQ(MatDenseRestoreColumnVecWrite(W,j-ini,&w));
        if (norm<=0.0 || breakdown) {
          CHKERRQ(PetscInfo(eps,"Orthogonalization of preconditioned residual failed\n"));
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          goto diverged;
        }
      }
    }
    CHKERRQ(MatCopy(W,V,SAME_NONZERO_PATTERN));
    CHKERRQ(BVRestoreMat(R,&V));

    /* 11. B-orthonormalize preconditioned residuals */
    CHKERRQ(BVOrthogonalize(R,NULL));

    /* 13-16. B-orthonormalize conjugate directions */
    if (its>1) CHKERRQ(BVOrthogonalize(P,NULL));

    /* 17-23. Compute symmetric Gram matrices */
    CHKERRQ(BVSetActiveColumns(Z,0,ctx->bs));
    CHKERRQ(BVSetActiveColumns(X,0,ctx->bs));
    CHKERRQ(BVCopy(X,Z));
    CHKERRQ(BVSetActiveColumns(Z,ctx->bs,2*ctx->bs-ini));
    CHKERRQ(BVCopy(R,Z));
    if (its>1) {
      CHKERRQ(BVSetActiveColumns(Z,2*ctx->bs-ini,3*ctx->bs-2*ini));
      CHKERRQ(BVCopy(P,Z));
    }

    if (its>1) nv = 3*ctx->bs-2*ini;
    else nv = 2*ctx->bs-ini;

    CHKERRQ(BVSetActiveColumns(Z,0,nv));
    CHKERRQ(DSSetDimensions(eps->ds,nv,0,0));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&M));
    CHKERRQ(BVMatProject(Z,A,Z,M));
    if (flip) CHKERRQ(MatScale(M,-1.0));
    CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&M));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_B,&M));
    CHKERRQ(BVMatProject(Z,B,Z,M)); /* covers also the case B=NULL */
    CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_B,&M));

    /* 24. Solve the generalized eigenvalue problem */
    CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
    CHKERRQ(DSSolve(eps->ds,eigr,NULL));
    CHKERRQ(DSSort(eps->ds,eigr,NULL,NULL,NULL,NULL));
    CHKERRQ(DSSynchronize(eps->ds,eigr,NULL));
    for (j=0;j<nv;j++) if (locked+j<eps->ncv) eps->eigr[locked+j] = flip? -eigr[j]: eigr[j];
    CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));

    /* 25-33. Compute Ritz vectors */
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&M));
    CHKERRQ(BVSetActiveColumns(Z,ctx->bs,nv));
    if (ctx->lock) CHKERRQ(BVSetActiveColumns(P,0,ctx->bs));
    CHKERRQ(BVMult(P,1.0,0.0,Z,M));
    CHKERRQ(BVCopy(P,X));
    if (ctx->lock) CHKERRQ(BVSetActiveColumns(P,nconv,ctx->bs));
    CHKERRQ(BVSetActiveColumns(Z,0,ctx->bs));
    CHKERRQ(BVMult(X,1.0,1.0,Z,M));
    if (ctx->lock) CHKERRQ(BVSetActiveColumns(X,nconv,ctx->bs));
    CHKERRQ(BVMatMult(X,A,AX));
    CHKERRQ(MatDestroy(&M));
  }

diverged:
  eps->its += its;

  if (flip) sc->comparison = SlepcCompareLargestReal;
  CHKERRQ(PetscFree(eigr));
  CHKERRQ(MatDestroy(&W));
  if (V) CHKERRQ(BVRestoreMat(R,&V)); /* only needed when goto diverged is reached */
  CHKERRQ(BVDestroy(&Z));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&R));
  CHKERRQ(BVDestroy(&P));
  CHKERRQ(BVDestroy(&AX));
  if (B) CHKERRQ(BVDestroy(&BX));
  if (nc>0 || eps->nev>ctx->bs-ctx->guard) CHKERRQ(BVDestroy(&Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLOBPCGSetBlockSize_LOBPCG(EPS eps,PetscInt bs)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  if (bs == PETSC_DEFAULT || bs == PETSC_DECIDE) bs = 1;
  PetscCheck(bs>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size %" PetscInt_FMT,bs);
  if (ctx->bs != bs) {
    ctx->bs = bs;
    eps->state = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSLOBPCGSetBlockSize - Sets the block size of the LOBPCG method.

   Logically Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
-  bs  - the block size

   Options Database Key:
.  -eps_lobpcg_blocksize - Sets the block size

   Level: advanced

.seealso: EPSLOBPCGGetBlockSize()
@*/
PetscErrorCode EPSLOBPCGSetBlockSize(EPS eps,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,bs,2);
  CHKERRQ(PetscTryMethod(eps,"EPSLOBPCGSetBlockSize_C",(EPS,PetscInt),(eps,bs)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLOBPCGGetBlockSize_LOBPCG(EPS eps,PetscInt *bs)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  *bs = ctx->bs;
  PetscFunctionReturn(0);
}

/*@
   EPSLOBPCGGetBlockSize - Gets the block size used in the LOBPCG method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  bs - the block size

   Level: advanced

.seealso: EPSLOBPCGSetBlockSize()
@*/
PetscErrorCode EPSLOBPCGGetBlockSize(EPS eps,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(bs,2);
  CHKERRQ(PetscUseMethod(eps,"EPSLOBPCGGetBlockSize_C",(EPS,PetscInt*),(eps,bs)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLOBPCGSetRestart_LOBPCG(EPS eps,PetscReal restart)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  if (restart==PETSC_DEFAULT) restart = 0.9;
  PetscCheck(restart>=0.1 && restart<=1.0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The restart argument %g must be in the range [0.1,1.0]",(double)restart);
  if (restart != ctx->restart) {
    ctx->restart = restart;
    eps->state = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSLOBPCGSetRestart - Sets the restart parameter for the LOBPCG method.

   Logically Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
-  restart - the percentage of the block of vectors to force a restart

   Options Database Key:
.  -eps_lobpcg_restart - Sets the restart parameter

   Notes:
   The meaning of this parameter is the proportion of vectors within the
   current block iterate that must have converged in order to force a
   restart with hard locking.
   Allowed values are in the range [0.1,1.0]. The default is 0.9.

   Level: advanced

.seealso: EPSLOBPCGGetRestart()
@*/
PetscErrorCode EPSLOBPCGSetRestart(EPS eps,PetscReal restart)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,restart,2);
  CHKERRQ(PetscTryMethod(eps,"EPSLOBPCGSetRestart_C",(EPS,PetscReal),(eps,restart)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLOBPCGGetRestart_LOBPCG(EPS eps,PetscReal *restart)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  *restart = ctx->restart;
  PetscFunctionReturn(0);
}

/*@
   EPSLOBPCGGetRestart - Gets the restart parameter used in the LOBPCG method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  restart - the restart parameter

   Level: advanced

.seealso: EPSLOBPCGSetRestart()
@*/
PetscErrorCode EPSLOBPCGGetRestart(EPS eps,PetscReal *restart)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidRealPointer(restart,2);
  CHKERRQ(PetscUseMethod(eps,"EPSLOBPCGGetRestart_C",(EPS,PetscReal*),(eps,restart)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLOBPCGSetLocking_LOBPCG(EPS eps,PetscBool lock)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

/*@
   EPSLOBPCGSetLocking - Choose between locking and non-locking variants of
   the LOBPCG method.

   Logically Collective on eps

   Input Parameters:
+  eps  - the eigenproblem solver context
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -eps_lobpcg_locking - Sets the locking flag

   Notes:
   This flag refers to soft locking (converged vectors within the current
   block iterate), since hard locking is always used (when nev is larger
   than the block size).

   Level: advanced

.seealso: EPSLOBPCGGetLocking()
@*/
PetscErrorCode EPSLOBPCGSetLocking(EPS eps,PetscBool lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,lock,2);
  CHKERRQ(PetscTryMethod(eps,"EPSLOBPCGSetLocking_C",(EPS,PetscBool),(eps,lock)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLOBPCGGetLocking_LOBPCG(EPS eps,PetscBool *lock)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

/*@
   EPSLOBPCGGetLocking - Gets the locking flag used in the LOBPCG method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: EPSLOBPCGSetLocking()
@*/
PetscErrorCode EPSLOBPCGGetLocking(EPS eps,PetscBool *lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidBoolPointer(lock,2);
  CHKERRQ(PetscUseMethod(eps,"EPSLOBPCGGetLocking_C",(EPS,PetscBool*),(eps,lock)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_LOBPCG(EPS eps,PetscViewer viewer)
{
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  block size %" PetscInt_FMT "\n",ctx->bs));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  restart parameter=%g (using %" PetscInt_FMT " guard vectors)\n",(double)ctx->restart,ctx->guard));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  soft locking %sactivated\n",ctx->lock?"":"de"));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_LOBPCG(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscBool      lock,flg;
  PetscInt       bs;
  PetscReal      restart;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS LOBPCG Options"));

    CHKERRQ(PetscOptionsInt("-eps_lobpcg_blocksize","Block size","EPSLOBPCGSetBlockSize",20,&bs,&flg));
    if (flg) CHKERRQ(EPSLOBPCGSetBlockSize(eps,bs));

    CHKERRQ(PetscOptionsReal("-eps_lobpcg_restart","Percentage of the block of vectors to force a restart","EPSLOBPCGSetRestart",0.5,&restart,&flg));
    if (flg) CHKERRQ(EPSLOBPCGSetRestart(eps,restart));

    CHKERRQ(PetscOptionsBool("-eps_lobpcg_locking","Choose between locking and non-locking variants","EPSLOBPCGSetLocking",PETSC_TRUE,&lock,&flg));
    if (flg) CHKERRQ(EPSLOBPCGSetLocking(eps,lock));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_LOBPCG(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetBlockSize_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetBlockSize_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetLocking_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetLocking_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_LOBPCG(EPS eps)
{
  EPS_LOBPCG     *lobpcg;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&lobpcg));
  eps->data = (void*)lobpcg;
  lobpcg->lock = PETSC_TRUE;

  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_PRECOND;

  eps->ops->solve          = EPSSolve_LOBPCG;
  eps->ops->setup          = EPSSetUp_LOBPCG;
  eps->ops->setupsort      = EPSSetUpSort_Default;
  eps->ops->setfromoptions = EPSSetFromOptions_LOBPCG;
  eps->ops->destroy        = EPSDestroy_LOBPCG;
  eps->ops->view           = EPSView_LOBPCG;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_GMRES;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetBlockSize_C",EPSLOBPCGSetBlockSize_LOBPCG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetBlockSize_C",EPSLOBPCGGetBlockSize_LOBPCG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetRestart_C",EPSLOBPCGSetRestart_LOBPCG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetRestart_C",EPSLOBPCGGetRestart_LOBPCG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetLocking_C",EPSLOBPCGSetLocking_LOBPCG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetLocking_C",EPSLOBPCGGetLocking_LOBPCG));
  PetscFunctionReturn(0);
}
