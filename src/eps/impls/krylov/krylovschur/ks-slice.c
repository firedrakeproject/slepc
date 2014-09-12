/*

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with spectrum slicing for symmetric eigenproblems

   References:

       [1] R.G. Grimes et al., "A shifted block Lanczos algorithm for
           solving sparse symmetric generalized eigenproblems", SIAM J.
           Matrix Anal. Appl. 15(1):228-272, 1994.

       [2] C. Campos and J.E. Roman, "Spectrum slicing strategies based
           on restarted Lanczos methods", Numer. Algor. 60(2):279-295,
           2012.

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

#include <slepc-private/epsimpl.h>
#include "krylovschur.h"

#undef __FUNCT__
#define __FUNCT__ "EPSAllocateSolutionSlice"
/*
  EPSAllocateSolutionSlice - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors. The argument extra is used for methods
  that require a working basis slightly larger than ncv.
*/
PetscErrorCode EPSAllocateSolutionSlice(EPS eps,PetscInt extra)
{
  PetscErrorCode ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        requested;
  PetscReal       eta;
  PetscLogDouble  cnt;
  BVType          type;
  BVOrthogType    orthog_type;
  BVOrthogRefineType orthog_ref;
  Mat             matrix;
  Vec             t;
  EPS_SR          sr = ctx->sr;

  PetscFunctionBegin;
  requested = ctx->ncv + extra;

  /* allocate space for eigenvalues and friends */
  ierr = PetscMalloc4(requested,&sr->eigr,requested,&sr->eigi,requested,&sr->errest,requested,&sr->perm);CHKERRQ(ierr);
  cnt = 2*requested*sizeof(PetscScalar) + 2*requested*sizeof(PetscReal) + requested*sizeof(PetscInt);
  ierr = PetscLogObjectMemory((PetscObject)eps,cnt);CHKERRQ(ierr);

  /* allocate sr->V and transfer options from eps->V */
  ierr = BVCreate(PetscObjectComm((PetscObject)eps),&sr->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)sr->V);CHKERRQ(ierr);
  if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
  if (!((PetscObject)(eps->V))->type_name) {
    ierr = BVSetType(sr->V,BVSVEC);CHKERRQ(ierr);
  } else {
    ierr = BVGetType(eps->V,&type);CHKERRQ(ierr);
    ierr = BVSetType(sr->V,type);CHKERRQ(ierr);
  }
  ierr = STMatGetVecs(eps->st,&t,NULL);CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(sr->V,t,requested);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  ierr = BVGetMatrix(eps->V,&matrix,NULL);CHKERRQ(ierr);
  ierr = BVSetMatrix(sr->V,matrix,PETSC_FALSE);CHKERRQ(ierr);
  ierr = BVGetOrthogonalization(eps->V,&orthog_type,&orthog_ref,&eta);CHKERRQ(ierr);
  ierr = BVSetOrthogonalization(sr->V,orthog_type,orthog_ref,eta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_KrylovSchur_Slice"
PetscErrorCode EPSSetUp_KrylovSchur_Slice(EPS eps)
{
  PetscErrorCode  ierr;
  PetscBool       issinv;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR          sr;
  KSP             ksp;
  PC              pc;
  Mat             F;

  PetscFunctionBegin;
  if (eps->intb >= PETSC_MAX_REAL && eps->inta <= PETSC_MIN_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
  if (eps->inta==0.0 && eps->intb==0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Must define a computational interval when using EPS_ALL");
  if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing only available for symmetric/Hermitian eigenproblems");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs cannot be used with spectrum slicing");
  if (!((PetscObject)(eps->st))->type_name) { /* default to shift-and-invert */
    ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)eps->st,&issinv,STSINVERT,STCAYLEY,"");CHKERRQ(ierr);
  if (!issinv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Shift-and-invert or Cayley ST is needed for spectrum slicing");
  if (eps->tol==PETSC_DEFAULT) eps->tol = SLEPC_DEFAULT_TOL*1e-2;  /* use tighter tolerance */
  if (!eps->max_it) eps->max_it = 100;
  if (ctx->nev==1) ctx->nev = 40;  /* nev not set, use default value */
  if (ctx->nev<10) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"nev cannot be less than 10 in spectrum slicing runs");
  eps->ops->backtransform = NULL;

  /* create spectrum slicing context and initialize it */
  ierr = EPSReset_KrylovSchur(eps);CHKERRQ(ierr);
  ierr = PetscNewLog(eps,&sr);CHKERRQ(ierr);
  ctx->sr = sr;
  sr->itsKs = 0;
  sr->nleap = 0;
  sr->nMAXCompl = ctx->nev/4;
  sr->iterCompl = eps->max_it/4;
  sr->sPres = NULL;
  sr->nS = 0;

  /* check presence of ends and finding direction */
  if ((eps->inta > PETSC_MIN_REAL && eps->inta != 0.0) || eps->intb >= PETSC_MAX_REAL) {
    sr->int0 = eps->inta;
    sr->int1 = eps->intb;
    sr->dir = 1;
    if (eps->intb >= PETSC_MAX_REAL) { /* Right-open interval */
      sr->hasEnd = PETSC_FALSE;
      sr->inertia1 = eps->n;
    } else sr->hasEnd = PETSC_TRUE;
  } else {
    sr->int0 = eps->intb;
    sr->int1 = eps->inta;
    sr->dir = -1;
    if (eps->inta <= PETSC_MIN_REAL) { /* Left-open interval */
      sr->hasEnd = PETSC_FALSE;
      sr->inertia1 = 0;
    } else sr->hasEnd = PETSC_TRUE;
  }

  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  /* compute inertia1 if necessary */
  if (sr->hasEnd) {
    ierr = STSetShift(eps->st,sr->int1);CHKERRQ(ierr);
    ierr = STSetUp(eps->st);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    ierr = MatGetInertia(F,&sr->inertia1,NULL,NULL);CHKERRQ(ierr);
  }

  /* compute inertia0 */
  ierr = STSetShift(eps->st,sr->int0);CHKERRQ(ierr);
  ierr = STSetUp(eps->st);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
  ierr = MatGetInertia(F,&sr->inertia0,NULL,NULL);CHKERRQ(ierr);

  /* number of eigenvalues in interval */
  sr->numEigs = (sr->dir)*(sr->inertia1 - sr->inertia0);
  eps->nev = sr->numEigs;
  eps->ncv = sr->numEigs;
  eps->mpd = sr->numEigs;
  ierr = EPSSetDimensions_Default(eps,ctx->nev,&ctx->ncv,&ctx->mpd);CHKERRQ(ierr);

  /* allocate solution for subsolves */
  if (sr->numEigs) {
    ierr = EPSAllocateSolutionSlice(eps,1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Fills the fields of a shift structure
*/
#undef __FUNCT__
#define __FUNCT__ "EPSCreateShift"
static PetscErrorCode EPSCreateShift(EPS eps,PetscReal val,EPS_shift neighb0,EPS_shift neighb1)
{
  PetscErrorCode  ierr;
  EPS_shift       s,*pending2;
  PetscInt        i;
  EPS_SR          sr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  ierr = PetscNewLog(eps,&s);CHKERRQ(ierr);
  s->value = val;
  s->neighb[0] = neighb0;
  if (neighb0) neighb0->neighb[1] = s;
  s->neighb[1] = neighb1;
  if (neighb1) neighb1->neighb[0] = s;
  s->comp[0] = PETSC_FALSE;
  s->comp[1] = PETSC_FALSE;
  s->index = -1;
  s->neigs = 0;
  s->nconv[0] = s->nconv[1] = 0;
  s->nsch[0] = s->nsch[1]=0;
  /* Inserts in the stack of pending shifts */
  /* If needed, the array is resized */
  if (sr->nPend >= sr->maxPend) {
    sr->maxPend *= 2;
    ierr = PetscMalloc1(sr->maxPend,&pending2);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,sizeof(EPS_shift));CHKERRQ(ierr);
    for (i=0;i<sr->nPend;i++) pending2[i] = sr->pending[i];
    ierr = PetscFree(sr->pending);CHKERRQ(ierr);
    sr->pending = pending2;
  }
  sr->pending[sr->nPend++]=s;
  PetscFunctionReturn(0);
}

/* Prepare for Rational Krylov update */
#undef __FUNCT__
#define __FUNCT__ "EPSPrepareRational"
static PetscErrorCode EPSPrepareRational(EPS eps)
{
  EPS_KRYLOVSCHUR  *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscErrorCode   ierr;
  PetscInt         dir,i,k,ld,nv;
  PetscScalar      *A;
  EPS_SR           sr = ctx->sr;
  Vec              v;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  dir = (sr->sPres->neighb[0] == sr->sPrev)?1:-1;
  dir*=sr->dir;
  k = 0;
  for (i=0;i<sr->nS;i++) {
    if (dir*PetscRealPart(sr->S[i])>0.0) {
      sr->S[k] = sr->S[i];
      sr->S[sr->nS+k] = sr->S[sr->nS+i];
      ierr = BVGetColumn(sr->Vnext,k,&v);CHKERRQ(ierr);
      ierr = BVCopyVec(sr->V,eps->nconv+i,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(sr->Vnext,k,&v);CHKERRQ(ierr);
      k++;
      if (k>=sr->nS/2)break;
    }
  }
  /* Copy to DS */
  ierr = DSGetArray(eps->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    A[i*(1+ld)] = sr->S[i];
    A[k+i*ld] = sr->S[sr->nS+i];
  }
  sr->nS = k;
  ierr = DSRestoreArray(eps->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = DSGetDimensions(eps->ds,&nv,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSSetDimensions(eps->ds,nv,0,0,k);CHKERRQ(ierr);
  /* Append u to V */
  ierr = BVGetColumn(sr->Vnext,sr->nS,&v);CHKERRQ(ierr);
  ierr = BVCopyVec(sr->V,sr->nv,v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(sr->Vnext,sr->nS,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Provides next shift to be computed */
#undef __FUNCT__
#define __FUNCT__ "EPSExtractShift"
static PetscErrorCode EPSExtractShift(EPS eps)
{
  PetscErrorCode   ierr;
  PetscInt         iner;
  Mat              F;
  PC               pc;
  KSP              ksp;
  EPS_KRYLOVSCHUR  *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR           sr;

  PetscFunctionBegin;
  sr = ctx->sr;
  if (sr->nPend > 0) {
    sr->sPrev = sr->sPres;
    sr->sPres = sr->pending[--sr->nPend];
    ierr = STSetShift(eps->st,sr->sPres->value);CHKERRQ(ierr);
    ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    ierr = MatGetInertia(F,&iner,NULL,NULL);CHKERRQ(ierr);
    sr->sPres->inertia = iner;
    eps->target = sr->sPres->value;
    eps->reason = EPS_CONVERGED_ITERATING;
    eps->its = 0;
  } else sr->sPres = NULL;
  PetscFunctionReturn(0);
}

/*
   Symmetric KrylovSchur adapted to spectrum slicing:
   Allows searching an specific amount of eigenvalues in the subintervals left and right.
   Returns whether the search has succeeded
*/
#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchur_Slice"
static PetscErrorCode EPSKrylovSchur_Slice(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,conv,k,l,ld,nv,*iwork,j,p;
  Mat             U;
  PetscScalar     *Q,*A,rtmp,*eigrsave,*eigisave;
  PetscReal       *a,*b,beta,*errestsave;
  PetscBool       breakdown;
  PetscInt        count0,count1;
  PetscReal       lambda;
  EPS_shift       sPres;
  PetscBool       complIterating;
  PetscBool       sch0,sch1;
  PetscInt        iterCompl=0,n0,n1;
  EPS_SR          sr = ctx->sr;
  BV              bvsave;

  PetscFunctionBegin;
  bvsave = eps->V;  /* temporarily swap basis vectors */
  eps->V = sr->V;
  eigrsave = eps->eigr;
  eps->eigr = sr->eigr;
  eigisave = eps->eigi;
  eps->eigi = sr->eigi;
  errestsave = eps->errest;
  eps->errest = sr->errest;
  /* Spectrum slicing data */
  sPres = sr->sPres;
  complIterating =PETSC_FALSE;
  sch1 = sch0 = PETSC_TRUE;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*ld,&iwork);CHKERRQ(ierr);
  count0=0;count1=0; /* Found on both sides */
  if (sr->nS > 0 && (sPres->neighb[0] == sr->sPrev || sPres->neighb[1] == sr->sPrev)) {
    /* Rational Krylov */
    ierr = DSTranslateRKS(eps->ds,sr->sPrev->value-sPres->value);CHKERRQ(ierr);
    ierr = DSGetDimensions(eps->ds,NULL,NULL,NULL,&l,NULL);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,l+1,0,0,0);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(sr->V,0,l+1);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(sr->V,U,0,l+1);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
  } else {
    /* Get the starting Lanczos vector */
    ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
    l = 0;
  }
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++; sr->itsKs++;
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+ctx->mpd,ctx->ncv);
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    ierr = EPSFullLanczos(eps,a,b,eps->nconv+l,&nv,&breakdown);CHKERRQ(ierr);
    sr->nv = nv;
    beta = b[nv-1];
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(sr->V,eps->nconv,nv);CHKERRQ(ierr);

    /* Solve projected problem and compute residual norm estimates */
    if (eps->its == 1 && l > 0) {/* After rational update */
      ierr = DSGetArray(eps->ds,DS_MAT_A,&A);CHKERRQ(ierr);
      ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      b = a + ld;
      k = eps->nconv+l;
      A[k*ld+k-1] = A[(k-1)*ld+k];
      A[k*ld+k] = a[k];
      for (j=k+1; j< nv; j++) {
        A[j*ld+j] = a[j];
        A[j*ld+j-1] = b[j-1] ;
        A[(j-1)*ld+j] = b[j-1];
      }
      ierr = DSRestoreArray(eps->ds,DS_MAT_A,&A);CHKERRQ(ierr);
      ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      ierr = DSSolve(eps->ds,sr->eigr,NULL);CHKERRQ(ierr);
      ierr = DSSort(eps->ds,sr->eigr,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = DSSetCompact(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
    } else { /* Restart */
      ierr = DSSolve(eps->ds,sr->eigr,NULL);CHKERRQ(ierr);
      ierr = DSSort(eps->ds,sr->eigr,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    /* Residual */
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,eps->nconv,nv-eps->nconv,beta,1.0,&k);CHKERRQ(ierr);

    /* Check convergence */
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    conv = 0;
    j = k = eps->nconv;
    for (i=eps->nconv;i<nv;i++) if (sr->errest[i] < eps->tol) conv++;
    for (i=eps->nconv;i<nv;i++) {
      if (sr->errest[i] < eps->tol) {
        iwork[j++]=i;
      } else iwork[conv+k++]=i;
    }
    for (i=eps->nconv;i<nv;i++) {
      a[i]=PetscRealPart(sr->eigr[i]);
      b[i]=sr->errest[i];
    }
    for (i=eps->nconv;i<nv;i++) {
      sr->eigr[i] = a[iwork[i]];
      sr->errest[i] = b[iwork[i]];
    }
    for (i=eps->nconv;i<nv;i++) {
      a[i]=PetscRealPart(sr->eigr[i]);
      b[i]=sr->errest[i];
    }
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    for (i=eps->nconv;i<nv;i++) {
      p=iwork[i];
      if (p!=i) {
        j=i+1;
        while (iwork[j]!=i) j++;
        iwork[j]=p;iwork[i]=i;
        for (k=0;k<nv;k++) {
          rtmp=Q[k+p*ld];Q[k+p*ld]=Q[k+i*ld];Q[k+i*ld]=rtmp;
        }
      }
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    k=eps->nconv+conv;

    /* Checking values obtained for completing */
    for (i=0;i<k;i++) {
      sr->back[i]=sr->eigr[i];
    }
    ierr = STBackTransform(eps->st,k,sr->back,sr->eigi);CHKERRQ(ierr);
    count0=count1=0;
    for (i=0;i<k;i++) {
      lambda = PetscRealPart(sr->back[i]);
      if (((sr->dir)*(sPres->value - lambda) > 0) && ((sr->dir)*(lambda - sPres->ext[0]) > 0)) count0++;
      if (((sr->dir)*(lambda - sPres->value) > 0) && ((sr->dir)*(sPres->ext[1] - lambda) > 0)) count1++;
    }
    if (k>ctx->nev && ctx->ncv-k<5) eps->reason = EPS_CONVERGED_TOL;
    else {
      /* Checks completion */
      if ((!sch0||count0 >= sPres->nsch[0]) && (!sch1 ||count1 >= sPres->nsch[1])) {
        eps->reason = EPS_CONVERGED_TOL;
      } else {
        if (!complIterating && eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
        if (complIterating) {
          if (--iterCompl <= 0) eps->reason = EPS_DIVERGED_ITS;
        } else if (k >= ctx->nev) {
          n0 = sPres->nsch[0]-count0;
          n1 = sPres->nsch[1]-count1;
          if (sr->iterCompl>0 && ((n0>0 && n0<= sr->nMAXCompl)||(n1>0&&n1<=sr->nMAXCompl))) {
            /* Iterating for completion*/
            complIterating = PETSC_TRUE;
            if (n0 >sr->nMAXCompl)sch0 = PETSC_FALSE;
            if (n1 >sr->nMAXCompl)sch1 = PETSC_FALSE;
            iterCompl = sr->iterCompl;
          } else eps->reason = EPS_CONVERGED_TOL;
        }
      }
    }
    /* Update l */
    if (eps->reason == EPS_CONVERGED_ITERATING) l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
    else l = nv-k;
    if (breakdown) l=0;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%g)\n",eps->its,(double)beta);CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,k,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        b = a + ld;
        for (i=k;i<k+l;i++) {
          a[i] = PetscRealPart(sr->eigr[i]);
          b[i] = PetscRealPart(Q[nv-1+i*ld]*beta);
        }
        ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,U,eps->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);

    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(sr->V,nv,k+l);CHKERRQ(ierr);
    }
    /* Monitor */
    eps->nconv = k;
    ierr = EPSMonitor(eps,ctx->sr->itsKs,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    if (eps->reason != EPS_CONVERGED_ITERATING) {
      /* Store approximated values for next shift */
      ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
      sr->nS = l;
      for (i=0;i<l;i++) {
        sr->S[i] = sr->eigr[i+k];/* Diagonal elements */
        sr->S[i+l] = Q[nv-1+(i+k)*ld]*beta; /* Out of diagonal elements */
      }
      ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    }
  }
  /* Check for completion */
  for (i=0;i< eps->nconv; i++) {
    if ((sr->dir)*PetscRealPart(sr->eigr[i])>0) sPres->nconv[1]++;
    else sPres->nconv[0]++;
  }
  sPres->comp[0] = (count0 >= sPres->nsch[0])?PETSC_TRUE:PETSC_FALSE;
  sPres->comp[1] = (count1 >= sPres->nsch[1])?PETSC_TRUE:PETSC_FALSE;
  if (count0 > sPres->nsch[0] || count1 > sPres->nsch[1])SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unexpected error in Spectrum Slicing!\nMismatch between number of values found and information from inertia");
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  eps->V = bvsave;   /* restore basis */
  eps->eigr = eigrsave;
  eps->eigi = eigisave;
  eps->errest = errestsave;
  PetscFunctionReturn(0);
}

/*
  Obtains value of subsequent shift
*/
#undef __FUNCT__
#define __FUNCT__ "EPSGetNewShiftValue"
static PetscErrorCode EPSGetNewShiftValue(EPS eps,PetscInt side,PetscReal *newS)
{
  PetscReal       lambda,d_prev;
  PetscInt        i,idxP;
  EPS_SR          sr;
  EPS_shift       sPres,s;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  sPres = sr->sPres;
  if (sPres->neighb[side]) {
  /* Completing a previous interval */
    if (!sPres->neighb[side]->neighb[side] && sPres->neighb[side]->nconv[side]==0) { /* One of the ends might be too far from eigenvalues */
      if (side) *newS = (sPres->value + PetscRealPart(eps->eigr[eps->perm[sr->indexEig-1]]))/2;
      else *newS = (sPres->value + PetscRealPart(eps->eigr[eps->perm[0]]))/2;
    } else *newS=(sPres->value + sPres->neighb[side]->value)/2;
  } else { /* (Only for side=1). Creating a new interval. */
    if (sPres->neigs==0) {/* No value has been accepted*/
      if (sPres->neighb[0]) {
        /* Multiplying by 10 the previous distance */
        *newS = sPres->value + 10*(sr->dir)*PetscAbsReal(sPres->value - sPres->neighb[0]->value);
        sr->nleap++;
        /* Stops when the interval is open and no values are found in the last 5 shifts (there might be infinite eigenvalues) */
        if (!sr->hasEnd && sr->nleap > 5) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unable to compute the wanted eigenvalues with open interval");
      } else { /* First shift */
        if (eps->nconv != 0) {
          /* Unaccepted values give information for next shift */
          idxP=0;/* Number of values left from shift */
          for (i=0;i<eps->nconv;i++) {
            lambda = PetscRealPart(sr->eigr[i]);
            if ((sr->dir)*(lambda - sPres->value) <0) idxP++;
            else break;
          }
          /* Avoiding subtraction of eigenvalues (might be the same).*/
          if (idxP>0) {
            d_prev = PetscAbsReal(sPres->value - PetscRealPart(sr->eigr[0]))/(idxP+0.3);
          } else {
            d_prev = PetscAbsReal(sPres->value - PetscRealPart(sr->eigr[eps->nconv-1]))/(eps->nconv+0.3);
          }
          *newS = sPres->value + ((sr->dir)*d_prev*ctx->nev)/2;
        } else { /* No values found, no information for next shift */
          SETERRQ(PetscObjectComm((PetscObject)eps),1,"First shift renders no information");
        }
      }
    } else { /* Accepted values found */
      sr->nleap = 0;
      /* Average distance of values in previous subinterval */
      s = sPres->neighb[0];
      while (s && PetscAbs(s->inertia - sPres->inertia)==0) {
        s = s->neighb[0];/* Looking for previous shifts with eigenvalues within */
      }
      if (s) {
        d_prev = PetscAbsReal((sPres->value - s->value)/(sPres->inertia - s->inertia));
      } else { /* First shift. Average distance obtained with values in this shift */
        /* first shift might be too far from first wanted eigenvalue (no values found outside the interval)*/
        if ((sr->dir)*(PetscRealPart(eps->eigr[0])-sPres->value)>0 && PetscAbsReal((PetscRealPart(eps->eigr[sr->indexEig-1]) - PetscRealPart(eps->eigr[0]))/PetscRealPart(eps->eigr[0])) > PetscSqrtReal(eps->tol)) {
          d_prev =  PetscAbsReal((PetscRealPart(eps->eigr[sr->indexEig-1]) - PetscRealPart(eps->eigr[0])))/(sPres->neigs+0.3);
        } else {
          d_prev = PetscAbsReal(PetscRealPart(eps->eigr[sr->indexEig-1]) - sPres->value)/(sPres->neigs+0.3);
        }
      }
      /* Average distance is used for next shift by adding it to value on the right or to shift */
      if ((sr->dir)*(PetscRealPart(eps->eigr[sPres->index + sPres->neigs -1]) - sPres->value)>0) {
        *newS = PetscRealPart(eps->eigr[sPres->index + sPres->neigs -1])+ ((sr->dir)*d_prev*(ctx->nev))/2;
      } else { /* Last accepted value is on the left of shift. Adding to shift */
        *newS = sPres->value + ((sr->dir)*d_prev*(ctx->nev))/2;
      }
    }
    /* End of interval can not be surpassed */
    if ((sr->dir)*(sr->int1 - *newS) < 0) *newS = sr->int1;
  }/* of neighb[side]==null */
  PetscFunctionReturn(0);
}

/*
  Function for sorting an array of real values
*/
#undef __FUNCT__
#define __FUNCT__ "sortRealEigenvalues"
static PetscErrorCode sortRealEigenvalues(PetscScalar *r,PetscInt *perm,PetscInt nr,PetscBool prev,PetscInt dir)
{
  PetscReal      re;
  PetscInt       i,j,tmp;

  PetscFunctionBegin;
  if (!prev) for (i=0;i<nr;i++) perm[i] = i;
  /* Insertion sort */
  for (i=1;i<nr;i++) {
    re = PetscRealPart(r[perm[i]]);
    j = i-1;
    while (j>=0 && dir*(re - PetscRealPart(r[perm[j]])) <= 0) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
    }
  }
  PetscFunctionReturn(0);
}

/* Stores the pairs obtained since the last shift in the global arrays */
#undef __FUNCT__
#define __FUNCT__ "EPSStoreEigenpairs"
static PetscErrorCode EPSStoreEigenpairs(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscReal       lambda,err,norm;
  PetscInt        i,count;
  PetscBool       iscayley;
  EPS_SR          sr = ctx->sr;
  EPS_shift       sPres;
  Vec             v,w;
 
  PetscFunctionBegin;
  sPres = sr->sPres;
  sPres->index = sr->indexEig;
  count = sr->indexEig;
  /* Back-transform */
  ierr = STBackTransform(eps->st,eps->nconv,sr->eigr,sr->eigi);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STCAYLEY,&iscayley);CHKERRQ(ierr);
  /* Sort eigenvalues */
  ierr = sortRealEigenvalues(sr->eigr,sr->perm,eps->nconv,PETSC_FALSE,sr->dir);CHKERRQ(ierr);
  /* Values stored in global array */
  for (i=0;i<eps->nconv;i++) {
    lambda = PetscRealPart(sr->eigr[sr->perm[i]]);
    err = sr->errest[sr->perm[i]];

    if ((sr->dir)*(lambda - sPres->ext[0]) > 0 && (sr->dir)*(sPres->ext[1] - lambda) > 0) {/* Valid value */
      if (count>=sr->numEigs) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unexpected error in Spectrum Slicing");
      eps->eigr[count] = lambda;
      eps->errest[count] = err;
      /* Explicit purification */
      ierr = BVGetColumn(eps->V,count,&v);CHKERRQ(ierr);
      ierr = BVGetColumn(sr->V,sr->perm[i],&w);CHKERRQ(ierr);
      ierr = STApply(eps->st,w,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,count,&v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(sr->V,sr->perm[i],&w);CHKERRQ(ierr);
      ierr = BVNormColumn(eps->V,count,NORM_2,&norm);CHKERRQ(ierr);
      ierr = BVScaleColumn(eps->V,count,1.0/norm);CHKERRQ(ierr);
      count++;
    }
  }
  sPres->neigs = count - sr->indexEig;
  sr->indexEig = count;
  /* Global ordering array updating */
  ierr = sortRealEigenvalues(eps->eigr,eps->perm,count,PETSC_TRUE,sr->dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLookForDeflation"
static PetscErrorCode EPSLookForDeflation(EPS eps)
{
  PetscErrorCode  ierr;
  PetscReal       val;
  PetscInt        i,count0=0,count1=0;
  EPS_shift       sPres;
  PetscInt        ini,fin,k,idx0,idx1;
  EPS_SR          sr;
  Vec             v;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  sPres = sr->sPres;

  if (sPres->neighb[0]) ini = (sr->dir)*(sPres->neighb[0]->inertia - sr->inertia0);
  else ini = 0;
  fin = sr->indexEig;
  /* Selection of ends for searching new values */
  if (!sPres->neighb[0]) sPres->ext[0] = sr->int0;/* First shift */
  else sPres->ext[0] = sPres->neighb[0]->value;
  if (!sPres->neighb[1]) {
    if (sr->hasEnd) sPres->ext[1] = sr->int1;
    else sPres->ext[1] = (sr->dir > 0)?PETSC_MAX_REAL:PETSC_MIN_REAL;
  } else sPres->ext[1] = sPres->neighb[1]->value;
  /* Selection of values between right and left ends */
  for (i=ini;i<fin;i++) {
    val=PetscRealPart(eps->eigr[eps->perm[i]]);
    /* Values to the right of left shift */
    if ((sr->dir)*(val - sPres->ext[1]) < 0) {
      if ((sr->dir)*(val - sPres->value) < 0) count0++;
      else count1++;
    } else break;
  }
  /* The number of values on each side are found */
  if (sPres->neighb[0]) {
    sPres->nsch[0] = (sr->dir)*(sPres->inertia - sPres->neighb[0]->inertia)-count0;
    if (sPres->nsch[0]<0)SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unexpected error in Spectrum Slicing!\nMismatch between number of values found and information from inertia");
  } else sPres->nsch[0] = 0;

  if (sPres->neighb[1]) {
    sPres->nsch[1] = (sr->dir)*(sPres->neighb[1]->inertia - sPres->inertia) - count1;
    if (sPres->nsch[1]<0)SETERRQ(PetscObjectComm((PetscObject)eps),1,"Unexpected error in Spectrum Slicing!\nMismatch between number of values found and information from inertia");
  } else sPres->nsch[1] = (sr->dir)*(sr->inertia1 - sPres->inertia);

  /* Completing vector of indexes for deflation */
  idx0 = ini;
  idx1 = ini+count0+count1;
  k=0;
  for (i=idx0;i<idx1;i++) sr->idxDef[k++]=eps->perm[i];
  ierr = BVDuplicateResize(sr->V,k+ctx->ncv+1,&sr->Vnext);CHKERRQ(ierr);
  ierr = BVSetNumConstraints(sr->Vnext,k);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = BVGetColumn(sr->Vnext,-i-1,&v);CHKERRQ(ierr);
    ierr = BVCopyVec(eps->V,sr->idxDef[i],v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(sr->Vnext,-i-1,&v);CHKERRQ(ierr);
  }

  /* For rational Krylov */
  if (sr->nS>0 && (sr->sPrev == sr->sPres->neighb[0] || sr->sPrev == sr->sPres->neighb[1])) {
    ierr = EPSPrepareRational(eps);CHKERRQ(ierr);
  }
  eps->nconv = 0;
  /* Get rid of temporary Vnext */
  ierr = BVDestroy(&sr->V);CHKERRQ(ierr);
  sr->V = sr->Vnext;
  sr->Vnext = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Slice"
PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS eps)
{
  PetscErrorCode  ierr;
  PetscInt        i,lds;
  PetscReal       newS;
  EPS_SR          sr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  /* Only with eigenvalues present in the interval ...*/
  if (sr->numEigs==0) {
    eps->reason = EPS_CONVERGED_TOL;
    PetscFunctionReturn(0);
  }
  /* Array of pending shifts */
  sr->maxPend = 100; /* Initial size */
  sr->nPend = 0;
  ierr = PetscMalloc1(sr->maxPend,&sr->pending);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,(sr->maxPend)*sizeof(EPS_shift));CHKERRQ(ierr);
  ierr = EPSCreateShift(eps,sr->int0,NULL,NULL);CHKERRQ(ierr);
  /* extract first shift */
  sr->sPrev = NULL;
  sr->sPres = sr->pending[--sr->nPend];
  sr->sPres->inertia = sr->inertia0;
  eps->target = sr->sPres->value;
  sr->s0 = sr->sPres;
  sr->indexEig = 0;
  /* Memory reservation for auxiliary variables */
  lds = PetscMin(ctx->mpd,ctx->ncv);
  ierr = PetscCalloc1(lds*lds,&sr->S);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->ncv,&sr->back);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,(sr->numEigs+2*ctx->ncv)*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<ctx->ncv;i++) {
    sr->eigr[i]    = 0.0;
    sr->eigi[i]   = 0.0;
    sr->errest[i] = 0.0;
  }
  for (i=0;i<sr->numEigs;i++) eps->perm[i] = i;
  /* Vectors for deflation */
  ierr = PetscMalloc1(sr->numEigs,&sr->idxDef);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,sr->numEigs*sizeof(PetscInt));CHKERRQ(ierr);
  sr->indexEig = 0;
  /* Main loop */
  while (sr->sPres) {
    /* Search for deflation */
    ierr = EPSLookForDeflation(eps);CHKERRQ(ierr);
    /* KrylovSchur */
    ierr = EPSKrylovSchur_Slice(eps);CHKERRQ(ierr);

    ierr = EPSStoreEigenpairs(eps);CHKERRQ(ierr);
    /* Select new shift */
    if (!sr->sPres->comp[1]) {
      ierr = EPSGetNewShiftValue(eps,1,&newS);CHKERRQ(ierr);
      ierr = EPSCreateShift(eps,newS,sr->sPres,sr->sPres->neighb[1]);CHKERRQ(ierr);
    }
    if (!sr->sPres->comp[0]) {
      /* Completing earlier interval */
      ierr = EPSGetNewShiftValue(eps,0,&newS);CHKERRQ(ierr);
      ierr = EPSCreateShift(eps,newS,sr->sPres->neighb[0],sr->sPres);CHKERRQ(ierr);
    }
    /* Preparing for a new search of values */
    ierr = EPSExtractShift(eps);CHKERRQ(ierr);
  }

  /* Updating eps values prior to exit */
  ierr = BVDestroy(&sr->V);CHKERRQ(ierr);
  ierr = PetscFree4(sr->eigr,sr->eigi,sr->errest,sr->perm);CHKERRQ(ierr);
  ierr = PetscFree(sr->S);CHKERRQ(ierr);
  ierr = PetscFree(sr->idxDef);CHKERRQ(ierr);
  ierr = PetscFree(sr->pending);CHKERRQ(ierr);
  ierr = PetscFree(sr->back);CHKERRQ(ierr);
  eps->nconv  = sr->indexEig;
  eps->reason = EPS_CONVERGED_TOL;
  eps->its    = sr->itsKs;
  eps->nds    = 0;
  PetscFunctionReturn(0);
}

