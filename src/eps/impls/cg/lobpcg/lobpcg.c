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

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepc/private/dsimpl.h>                 /*I "slepcds.h" I*/

typedef struct {
  PetscInt  bs;     /* block size */
  PetscBool lock;   /* soft locking active/inactive */
} EPS_LOBPCG;

#undef __FUNCT__
#define __FUNCT__ "EPSSetDimensions_LOBPCG"
PetscErrorCode EPSSetDimensions_LOBPCG(EPS eps,PetscInt nev,PetscInt *ncv,PetscInt *mpd)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;
  PetscInt   k;

  PetscFunctionBegin;
  k = PetscMax(3*ctx->bs,((eps->nev-1)/ctx->bs+3)*ctx->bs);
  if (*ncv) { /* ncv set */
    if (*ncv<k) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv is not sufficiently large");
  } else *ncv = k;
  if (!*mpd) *mpd = 3*ctx->bs;
  else if (*mpd!=3*ctx->bs) SETERRQ(PetscObjectComm((PetscObject)eps),1,"This solver does not allow a value of mpd different from 3*blocksize");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_LOBPCG"
PetscErrorCode EPSSetUp_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;
  PetscBool      precond;

  PetscFunctionBegin;
  if (!eps->ishermitian || (eps->isgeneralized && !eps->ispositive)) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"LOBPCG only works for Hermitian problems");
  if (eps->nev>ctx->bs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Block size smaller than nev not supported yet");
  if (eps->n-eps->nds<5*ctx->bs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The problem size is too small relative to the block size");
  ierr = EPSSetDimensions_LOBPCG(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  if (eps->which!=EPS_SMALLEST_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STPRECOND,&precond);CHKERRQ(ierr);
  if (!precond) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"LOBPCG only works with precond ST");

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSGHEP);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->mpd);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_LOBPCG"
PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;
  PetscInt       j,k,ld,nv,ini,kini,nmat,nc;
  PetscReal      norm;
  PetscBool      breakdown;
  Mat            A,B,M;
  Vec            v,w=eps->work[0];
  BV             X,Y,R,P,AX,AR,AP,BX,BR,BP;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;

  /* 1. Allocate memory */
  ierr = BVDuplicateResize(eps->V,ctx->bs,&X);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->bs,&R);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->bs,&P);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->bs,&AX);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->bs,&AR);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->bs,&AP);CHKERRQ(ierr);
  if (B) {
    ierr = BVDuplicateResize(eps->V,ctx->bs,&BX);CHKERRQ(ierr);
    ierr = BVDuplicateResize(eps->V,ctx->bs,&BR);CHKERRQ(ierr);
    ierr = BVDuplicateResize(eps->V,ctx->bs,&BP);CHKERRQ(ierr);
  }
  nc = eps->nds;
  if (nc>0) {
    ierr = BVDuplicateResize(eps->V,nc,&Y);CHKERRQ(ierr);
    for (j=0;j<nc;j++) {
      ierr = BVGetColumn(eps->V,-nc+j,&v);CHKERRQ(ierr);
      ierr = BVInsertVec(Y,j,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,-nc+j,&v);CHKERRQ(ierr);
    }
  }

  /* 2. Apply the constraints to the initial vectors */
  kini = eps->nini;
  while (kini<ctx->bs) { /* Generate more initial vectors if necessary */
    ierr = BVSetRandomColumn(eps->V,kini,eps->rand);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(eps->V,kini,NULL,&norm,&breakdown);CHKERRQ(ierr);
    if (norm>0.0 && !breakdown) {
      ierr = BVScaleColumn(eps->V,kini,1.0/norm);CHKERRQ(ierr);
      kini++;
    }
  }
  nv = ctx->bs;
  ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
  ierr = BVCopy(eps->V,X);CHKERRQ(ierr);

  /* 3. B-orthogonalize initial vectors */
  /* TODO: X already B-orthogonal but need to store B*X in BX */
  if (B) {
    ierr = BVMatMult(X,B,BX);CHKERRQ(ierr);
  }

  /* 4. Compute initial Ritz vectors */
  ierr = BVMatMult(eps->V,A,AX);CHKERRQ(ierr);
  ierr = DSSetDimensions(eps->ds,nv,0,0,0);CHKERRQ(ierr);
  ierr = DSGetMat(eps->ds,DS_MAT_A,&M);CHKERRQ(ierr);
  ierr = BVMatProject(AX,NULL,X,M);CHKERRQ(ierr);
  ierr = DSRestoreMat(eps->ds,DS_MAT_A,&M);CHKERRQ(ierr);
  ierr = DSSetIdentity(eps->ds,DS_MAT_B);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
  ierr = DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetMat(eps->ds,DS_MAT_X,&M);CHKERRQ(ierr);
  ierr = BVMultInPlace(X,M,0,nv);CHKERRQ(ierr);
  ierr = BVMultInPlace(AX,M,0,nv);CHKERRQ(ierr);
  ierr = DSRestoreMat(eps->ds,DS_MAT_X,&M);CHKERRQ(ierr);

  /* 5. Initialize index set of active iterates TODO */

  /* 6. Main loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {

    if (ctx->lock) {
      ierr = BVSetActiveColumns(R,eps->nconv,ctx->bs);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(AX,eps->nconv,ctx->bs);CHKERRQ(ierr);
      if (B) {
        ierr = BVSetActiveColumns(BX,eps->nconv,ctx->bs);CHKERRQ(ierr);
      }
    }

    /* 7. Compute residuals */
    ierr = DSGetMat(eps->ds,DS_MAT_A,&M);CHKERRQ(ierr);
    ierr = BVCopy(AX,R);CHKERRQ(ierr);
    if (B) {
      ierr = BVMult(R,-1.0,1.0,BX,M);CHKERRQ(ierr);
    } else {
      ierr = BVMult(R,-1.0,1.0,X,M);CHKERRQ(ierr);
    }
    ierr = DSRestoreMat(eps->ds,DS_MAT_A,&M);CHKERRQ(ierr);

    /* 8. Compute residual norms and (TODO) update index set of active iterates */
    ini = (ctx->lock)? eps->nconv: 0;
    k = ini;
    for (j=ini;j<ctx->bs;j++) {   /* TODO: optimize computation of norms */
      ierr = BVGetColumn(R,j,&v);CHKERRQ(ierr);
      ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
      ierr = BVRestoreColumn(R,j,&v);CHKERRQ(ierr);
      ierr = (*eps->converged)(eps,eps->eigr[j],eps->eigi[j],norm,&eps->errest[j],eps->convergedctx);CHKERRQ(ierr);
      if (eps->errest[k]<eps->tol) k++;
    }
    eps->nconv = k;
    if (eps->its) {
      ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ctx->bs);CHKERRQ(ierr);
    }
    if (eps->nconv >= eps->nev) {
      ierr = BVSetActiveColumns(eps->V,0,ctx->bs);CHKERRQ(ierr);  /* TODO: avoid copies */
      ierr = BVSetActiveColumns(X,0,ctx->bs);CHKERRQ(ierr);
      ierr = BVCopy(X,eps->V);CHKERRQ(ierr);
      eps->reason = EPS_CONVERGED_TOL;
    }
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (eps->reason != EPS_CONVERGED_ITERATING) break;
    eps->its++;

    ini = (ctx->lock)? eps->nconv: 0;
    if (ctx->lock) {
      ierr = BVSetActiveColumns(R,eps->nconv,ctx->bs);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(P,eps->nconv,ctx->bs);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(AX,eps->nconv,ctx->bs);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(AR,eps->nconv,ctx->bs);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(AP,eps->nconv,ctx->bs);CHKERRQ(ierr);
      if (B) {
        ierr = BVSetActiveColumns(BX,eps->nconv,ctx->bs);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(BR,eps->nconv,ctx->bs);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(BP,eps->nconv,ctx->bs);CHKERRQ(ierr);
      }
    }

    /* 9. Apply preconditioner to the residuals */
    for (j=ini;j<ctx->bs;j++) {
      ierr = BVGetColumn(R,j,&v);CHKERRQ(ierr);
      ierr = STMatSolve(eps->st,v,w);CHKERRQ(ierr);
      if (nc>0) {
        ierr = BVOrthogonalizeVec(Y,w,NULL,NULL,NULL);CHKERRQ(ierr);
        ierr = VecNormalize(w,NULL);CHKERRQ(ierr);
      }
      ierr = VecCopy(w,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(R,j,&v);CHKERRQ(ierr);
    }

    /* 10. Enforce the constraints on the preconditioned residuals */
    /* TODO */

    /* 11. B-orthonormalize preconditioned residuals */
    if (B) {
      ierr = BVMatMult(R,B,BR);CHKERRQ(ierr);  /* TODO: reuse BR in orthogonalization */
    }
    ierr = BVOrthogonalize(R,NULL);CHKERRQ(ierr);

    /* 12. Compute AR */
    ierr = BVMatMult(R,A,AR);CHKERRQ(ierr);

    /* 13-16. B-orthonormalize conjugate directions */
    if (eps->its>1) {
      ierr = BVOrthogonalize(P,NULL);CHKERRQ(ierr);
      ierr = BVMatMult(P,A,AP);CHKERRQ(ierr);  /* TODO: avoid this, instead AP=AP\cholR */
      if (B) {
        ierr = BVMatMult(P,B,BP);CHKERRQ(ierr);  /* TODO: avoid this, instead BP=BP\cholR */
      }
    }

    /* 17-23. Compute symmetric Gram matrices */
    ierr = BVSetActiveColumns(eps->V,0,ctx->bs);CHKERRQ(ierr);  /* TODO: avoid copies */
    ierr = BVSetActiveColumns(X,0,ctx->bs);CHKERRQ(ierr);
    ierr = BVCopy(X,eps->V);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,ctx->bs,2*ctx->bs-ini);CHKERRQ(ierr);
    ierr = BVCopy(R,eps->V);CHKERRQ(ierr);
    if (eps->its>1) {
      ierr = BVSetActiveColumns(eps->V,2*ctx->bs-ini,3*ctx->bs-2*ini);CHKERRQ(ierr);
      ierr = BVCopy(P,eps->V);CHKERRQ(ierr);
    }

    if (eps->its>1) nv = 3*ctx->bs-2*ini;
    else nv = 2*ctx->bs-ini;

    ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_A,&M);CHKERRQ(ierr);  /* TODO: optimize following lines */
    ierr = BVMatProject(eps->V,A,eps->V,M);CHKERRQ(ierr);
    ierr = DSRestoreMat(eps->ds,DS_MAT_A,&M);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_B,&M);CHKERRQ(ierr);
    if (B) {
      ierr = BVMatProject(eps->V,B,eps->V,M);CHKERRQ(ierr);
    } else {
      ierr = BVDot(eps->V,eps->V,M);CHKERRQ(ierr);
    }
    ierr = DSRestoreMat(eps->ds,DS_MAT_B,&M);CHKERRQ(ierr);
    
    /* 24. Solve the generalized eigenvalue problem */
    ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    
    /* 25-33. Compute Ritz vectors */
    ierr = DSGetMat(eps->ds,DS_MAT_X,&M);CHKERRQ(ierr);
    if (eps->its>1) {
      ierr = BVSetActiveColumns(eps->V,ctx->bs,nv);CHKERRQ(ierr);
      if (ctx->lock) {
        ierr = BVSetActiveColumns(P,0,ctx->bs);CHKERRQ(ierr);
      }
      ierr = BVMult(P,1.0,0.0,eps->V,M);CHKERRQ(ierr);
      ierr = BVCopy(P,X);CHKERRQ(ierr);
      if (ctx->lock) {
        ierr = BVSetActiveColumns(P,eps->nconv,ctx->bs);CHKERRQ(ierr);
      }
      ierr = BVMatMult(P,A,AP);CHKERRQ(ierr);  /* TODO: avoid this, instead use AR,AP */
      if (B) {
        ierr = BVMatMult(P,B,BP);CHKERRQ(ierr);  /* TODO: avoid this, instead use BR,BP */
      }
      ierr = BVSetActiveColumns(eps->V,0,ctx->bs);CHKERRQ(ierr);
      ierr = BVMult(X,1.0,1.0,eps->V,M);CHKERRQ(ierr);
      if (ctx->lock) {
        ierr = BVSetActiveColumns(X,eps->nconv,ctx->bs);CHKERRQ(ierr);
      }
      ierr = BVMatMult(X,A,AX);CHKERRQ(ierr);  /* TODO: avoid this, instead use AX,AP */
      if (B) {
        ierr = BVMatMult(X,B,BX);CHKERRQ(ierr);  /* TODO: avoid this, instead use BX,BP */
      }
    } else {  /* TODO: move redundant code out of if */
      ierr = BVSetActiveColumns(eps->V,ctx->bs,nv);CHKERRQ(ierr);
      if (ctx->lock) {
        ierr = BVSetActiveColumns(P,0,ctx->bs);CHKERRQ(ierr);
      }
      ierr = BVMult(P,1.0,0.0,eps->V,M);CHKERRQ(ierr);
      ierr = BVCopy(P,X);CHKERRQ(ierr);
      if (ctx->lock) {
        ierr = BVSetActiveColumns(P,eps->nconv,ctx->bs);CHKERRQ(ierr);
      }
      ierr = BVMatMult(P,A,AP);CHKERRQ(ierr);  /* TODO: avoid this, instead use AR */
      if (B) {
        ierr = BVMatMult(P,B,BP);CHKERRQ(ierr);  /* TODO: avoid this, instead use BR */
      }
      ierr = BVSetActiveColumns(eps->V,0,ctx->bs);CHKERRQ(ierr);
      ierr = BVMult(X,1.0,1.0,eps->V,M);CHKERRQ(ierr);
      if (ctx->lock) {
        ierr = BVSetActiveColumns(X,eps->nconv,ctx->bs);CHKERRQ(ierr);
      }
      ierr = BVMatMult(X,A,AX);CHKERRQ(ierr);  /* TODO: avoid this, instead use AX */
      if (B) {
        ierr = BVMatMult(X,B,BX);CHKERRQ(ierr);  /* TODO: avoid this, instead use BX */
      }
    }
    ierr = DSRestoreMat(eps->ds,DS_MAT_X,&M);CHKERRQ(ierr);
  }

  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = BVDestroy(&R);CHKERRQ(ierr);
  ierr = BVDestroy(&P);CHKERRQ(ierr);
  ierr = BVDestroy(&AX);CHKERRQ(ierr);
  ierr = BVDestroy(&AR);CHKERRQ(ierr);
  ierr = BVDestroy(&AP);CHKERRQ(ierr);
  if (B) {
    ierr = BVDestroy(&BX);CHKERRQ(ierr);
    ierr = BVDestroy(&BR);CHKERRQ(ierr);
    ierr = BVDestroy(&BP);CHKERRQ(ierr);
  }
  if (nc>0) {
    ierr = BVDestroy(&Y);CHKERRQ(ierr);
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGSetBlockSize_LOBPCG"
static PetscErrorCode EPSLOBPCGSetBlockSize_LOBPCG(EPS eps,PetscInt bs)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  ctx->bs = bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGSetBlockSize"
/*@
   EPSLOBPCGSetBlockSize - Sets the block size of the LOBPCG method.

   Logically Collective on EPS

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,bs,2);
  ierr = PetscTryMethod(eps,"EPSLOBPCGSetBlockSize_C",(EPS,PetscInt),(eps,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGGetBlockSize_LOBPCG"
static PetscErrorCode EPSLOBPCGGetBlockSize_LOBPCG(EPS eps,PetscInt *bs)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  *bs = ctx->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGGetBlockSize"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(bs,2);
  ierr = PetscTryMethod(eps,"EPSLOBPCGGetBlockSize_C",(EPS,PetscInt*),(eps,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGSetLocking_LOBPCG"
static PetscErrorCode EPSLOBPCGSetLocking_LOBPCG(EPS eps,PetscBool lock)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGSetLocking"
/*@
   EPSLOBPCGSetLocking - Choose between locking and non-locking variants of
   the LOBPCG method.

   Logically Collective on EPS

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,lock,2);
  ierr = PetscTryMethod(eps,"EPSLOBPCGSetLocking_C",(EPS,PetscBool),(eps,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGGetLocking_LOBPCG"
static PetscErrorCode EPSLOBPCGGetLocking_LOBPCG(EPS eps,PetscBool *lock)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGGetLocking"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(lock,2);
  ierr = PetscTryMethod(eps,"EPSLOBPCGGetLocking_C",(EPS,PetscBool*),(eps,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_LOBPCG"
PetscErrorCode EPSView_LOBPCG(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  LOBPCG: block size %D\n",ctx->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  LOBPCG: soft locking %sactivated\n",ctx->lock?"":"de");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_LOBPCG"
PetscErrorCode EPSSetFromOptions_LOBPCG(PetscOptions *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      lock,flg;
  PetscInt       bs;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS LOBPCG Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_lobpcg_blocksize","LOBPCG block size","EPSLOBPCGSetBlockSize",20,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSLOBPCGSetBlockSize(eps,bs);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-eps_lobpcg_locking","Choose between locking and non-locking variants","EPSLOBPCGSetLocking",PETSC_TRUE,&lock,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSLOBPCGSetLocking(eps,lock);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_LOBPCG"
PetscErrorCode EPSDestroy_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetLocking_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_LOBPCG"
PETSC_EXTERN PetscErrorCode EPSCreate_LOBPCG(EPS eps)
{
  EPS_LOBPCG     *lobpcg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&lobpcg);CHKERRQ(ierr);
  eps->data = (void*)lobpcg;
  lobpcg->bs   = 1;
  lobpcg->lock = PETSC_TRUE;

  eps->ops->setup          = EPSSetUp_LOBPCG;
  eps->ops->solve          = EPSSolve_LOBPCG;
  eps->ops->setfromoptions = EPSSetFromOptions_LOBPCG;
  eps->ops->destroy        = EPSDestroy_LOBPCG;
  eps->ops->view           = EPSView_LOBPCG;
  eps->ops->backtransform  = EPSBackTransform_Default;
  ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetBlockSize_C",EPSLOBPCGSetBlockSize_LOBPCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetBlockSize_C",EPSLOBPCGGetBlockSize_LOBPCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetLocking_C",EPSLOBPCGSetLocking_LOBPCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetLocking_C",EPSLOBPCGGetLocking_LOBPCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

