/*                       

   SLEPc eigensolver: "subspace"

   Method: Subspace Iteration

   Algorithm:

       Subspace iteration with Rayleigh-Ritz projection and locking,
       based on the SRRIT implementation.

   References:

       [1] "Subspace Iteration in SLEPc", SLEPc Technical Report STR-3, 
           available at http://www.grycap.upv.es/slepc.

   Last update: Feb 2009

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode EPSSolve_Subspace(EPS);

typedef struct {
  Vec *AV;
} EPS_SUBSPACE;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_Subspace"
PetscErrorCode EPSSetUp_Subspace(EPS eps)
{
  PetscErrorCode ierr;
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE*)eps->data;

  PetscFunctionBegin;
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must be at least nev"); 
  } else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
    else {
      eps->mpd = 500;
      eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
    }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) { ierr = EPSDefaultSetWhich(eps);CHKERRQ(ierr); }
  if (eps->which!=EPS_LARGEST_MAGNITUDE && eps->which!=EPS_TARGET_MAGNITUDE) SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbit_func) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(eps->t,eps->ncv,&ctx->AV);CHKERRQ(ierr);
  if (eps->ishermitian) {
    ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
  }
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_Subspace;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSubspaceFindGroup"
/*
   EPSSubspaceFindGroup - Find a group of nearly equimodular eigenvalues, provided 
   in arrays wr and wi, according to the tolerance grptol. Also the 2-norms
   of the residuals must be passed in (rsd). Arrays are processed from index 
   l to index m only. The output information is:

   ngrp - number of entries of the group
   ctr  - (w(l)+w(l+ngrp-1))/2
   ae   - average of wr(l),...,wr(l+ngrp-1)
   arsd - average of rsd(l),...,rsd(l+ngrp-1)
*/
static PetscErrorCode EPSSubspaceFindGroup(PetscInt l,PetscInt m,PetscScalar *wr,PetscScalar *wi,PetscReal *rsd,PetscReal grptol,PetscInt *ngrp,PetscReal *ctr,PetscReal *ae,PetscReal *arsd)
{
  PetscInt  i;
  PetscReal rmod,rmod1;

  PetscFunctionBegin;
  *ngrp = 0;
  *ctr = 0;
  rmod = SlepcAbsEigenvalue(wr[l],wi[l]);

  for (i=l;i<m;) {
    rmod1 = SlepcAbsEigenvalue(wr[i],wi[i]);
    if (PetscAbsReal(rmod-rmod1) > grptol*(rmod+rmod1)) break;
    *ctr = (rmod+rmod1)/2.0;
    if (wi[i] != 0.0) {
      (*ngrp)+=2;
      i+=2;
    } else {
      (*ngrp)++;
      i++;
    }
  }

  *ae = 0;
  *arsd = 0;
  if (*ngrp) {
    for (i=l;i<l+*ngrp;i++) {
      (*ae) += PetscRealPart(wr[i]);
      (*arsd) += rsd[i]*rsd[i];
    }
    *ae = *ae / *ngrp;
    *arsd = PetscSqrtScalar(*arsd / *ngrp);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSubspaceResidualNorms"
/*
   EPSSubspaceResidualNorms - Computes the column norms of residual vectors
   OP*V(1:n,l:m) - V*T(1:m,l:m), where, on entry, OP*V has been computed and 
   stored in AV. ldt is the leading dimension of T. On exit, rsd(l) to
   rsd(m) contain the computed norms.
*/
static PetscErrorCode EPSSubspaceResidualNorms(Vec *V,Vec *AV,PetscScalar *T,PetscInt l,PetscInt m,PetscInt ldt,Vec w,PetscReal *rsd)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  PetscScalar    t;

  PetscFunctionBegin;
  for (i=l;i<m;i++) {
    if (i==m-1 || T[i+1+ldt*i]==0.0) k=i+1;
    else k=i+2;
    ierr = VecCopy(AV[i],w);CHKERRQ(ierr);
    ierr = SlepcVecMAXPBY(w,1.0,-1.0,k,T+ldt*i,V);CHKERRQ(ierr);
    ierr = VecDot(w,w,&t);CHKERRQ(ierr);
    rsd[i] = PetscRealPart(t);
  }
  for (i=l;i<m;i++) {
    if (i == m-1) {
      rsd[i] = PetscSqrtReal(rsd[i]);  
    } else if (T[i+1+(ldt*i)]==0.0) {
      rsd[i] = PetscSqrtReal(rsd[i]);
    } else {
      rsd[i] = PetscSqrtReal((rsd[i]+rsd[i+1])/2.0);
      rsd[i+1] = rsd[i];
      i++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_Subspace"
PetscErrorCode EPSSolve_Subspace(EPS eps)
{
  PetscErrorCode ierr;
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE*)eps->data;
  PetscInt       i,k,ld,ngrp,nogrp,*itrsd,*itrsdold;
  PetscInt       nxtsrr,idsrr,idort,nxtort,nv,ncv = eps->ncv,its;
  PetscScalar    *T,*U;
  PetscReal      arsd,oarsd,ctr,octr,ae,oae,*rsd,norm,tcond=1.0;
  PetscBool      breakdown;
  /* Parameters */
  PetscInt       init = 5;        /* Number of initial iterations */
  PetscReal      stpfac = 1.5;    /* Max num of iter before next SRR step */
  PetscReal      alpha = 1.0;     /* Used to predict convergence of next residual */
  PetscReal      beta = 1.1;      /* Used to predict convergence of next residual */
  PetscReal      grptol = 1e-8;   /* Tolerance for EPSSubspaceFindGroup */
  PetscReal      cnvtol = 1e-6;   /* Convergence criterion for cnv */
  PetscInt       orttol = 2;      /* Number of decimal digits whose loss
                                     can be tolerated in orthogonalization */

  PetscFunctionBegin;
  its = 0;
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncv,&itrsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncv,&itrsdold);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);

  for (i=0;i<ncv;i++) {
    rsd[i] = 0.0;
    itrsd[i] = -1;
  }
  
  /* Complete the initial basis with random vectors and orthonormalize them */
  k = eps->nini;
  while (k<ncv) {
    ierr = SlepcVecSetRandom(eps->V[k],eps->rand);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,k,PETSC_NULL,eps->V,eps->V[k],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr); 
    if (norm>0.0 && !breakdown) {
      ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
      k++;
    }
  }

  while (eps->its<eps->max_it) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    ierr = DSSetDimensions(eps->ds,nv,PETSC_IGNORE,eps->nconv,0);CHKERRQ(ierr);
    
    /* Find group in previously computed eigenvalues */
    ierr = EPSSubspaceFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,rsd,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

    /* AV(:,idx) = OP * V(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = STApply(eps->st,eps->V[i],ctx->AV[i]);CHKERRQ(ierr);
    }

    /* T(:,idx) = V' * AV(:,idx) */
    ierr = DSGetArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);
    for (i=eps->nconv;i<nv;i++) {
      ierr = VecMDot(ctx->AV[i],nv,eps->V,T+i*ld);CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    /* Update vectors V(:,idx) = V * U(:,idx) */
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,ctx->AV,eps->nconv,nv,U,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,nv,U,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    
    /* Convergence check */
    ierr = DSGetArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);
    ierr = EPSSubspaceResidualNorms(eps->V,ctx->AV,T,eps->nconv,nv,ld,eps->work[0],rsd);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);

    for (i=eps->nconv;i<nv;i++) { 
      itrsdold[i] = itrsd[i];
      itrsd[i] = its;
      eps->errest[i] = rsd[i];
    }
  
    for (;;) {
      /* Find group in currently computed eigenvalues */
      ierr = EPSSubspaceFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsScalar(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=nv) break;
    }
    
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    if (eps->nconv>=eps->nev) break;
    
    /* Compute nxtsrr (iteration of next projection step) */
    nxtsrr = PetscMin(eps->max_it,PetscMax((PetscInt)floor(stpfac*its),init));
    
    if (ngrp!=nogrp || ngrp==0 || arsd>=oarsd) {
      idsrr = nxtsrr - its;
    } else {
      idsrr = (PetscInt)floor(alpha+beta*(itrsdold[eps->nconv]-itrsd[eps->nconv])*log(arsd/eps->tol)/log(arsd/oarsd));
      idsrr = PetscMax(1,idsrr);
    }
    nxtsrr = PetscMin(nxtsrr,its+idsrr);

    /* Compute nxtort (iteration of next orthogonalization step) */
    ierr = DSCond(eps->ds,&tcond);CHKERRQ(ierr);
    idort = PetscMax(1,(PetscInt)floor(orttol/PetscMax(1,log10(tcond))));    
    nxtort = PetscMin(its+idort,nxtsrr);

    /* V(:,idx) = AV(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = VecCopy(ctx->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    its++;

    /* Orthogonalization loop */
    do {
      while (its<nxtort) {
      
        /* AV(:,idx) = OP * V(:,idx) */
        for (i=eps->nconv;i<nv;i++) {
          ierr = STApply(eps->st,eps->V[i],ctx->AV[i]);CHKERRQ(ierr);
        }
        
        /* V(:,idx) = AV(:,idx) with normalization */
        for (i=eps->nconv;i<nv;i++) {
          ierr = VecCopy(ctx->AV[i],eps->V[i]);CHKERRQ(ierr);
          ierr = VecNorm(eps->V[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
        }
        its++;
      }
      /* Orthonormalize vectors */
      for (i=eps->nconv;i<nv;i++) {
        ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,i,PETSC_NULL,eps->V,eps->V[i],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          ierr = SlepcVecSetRandom(eps->V[i],eps->rand);CHKERRQ(ierr);
          ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,i,PETSC_NULL,eps->V,eps->V[i],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        }
        ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
      }
      nxtort = PetscMin(its+idort,nxtsrr);
    } while (its<nxtsrr);
  }

  ierr = PetscFree(rsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsdold);CHKERRQ(ierr);

  if (eps->nconv == eps->nev) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
  /* truncate Schur decomposition and change the state to raw so that
     PSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(eps->ds,eps->nconv,PETSC_IGNORE,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_Subspace"
PetscErrorCode EPSReset_Subspace(EPS eps)
{
  PetscErrorCode ierr;
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE*)eps->data;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(eps->ncv,&ctx->AV);CHKERRQ(ierr);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_Subspace"
PetscErrorCode EPSDestroy_Subspace(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_Subspace"
PetscErrorCode EPSCreate_Subspace(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_SUBSPACE,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_Subspace;
  eps->ops->destroy              = EPSDestroy_Subspace;
  eps->ops->reset                = EPSReset_Subspace;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

