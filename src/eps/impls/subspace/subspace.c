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

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
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
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE *)eps->data;

  PetscFunctionBegin;
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must be at least nev"); 
  }
  else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
  }
  else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
    else { eps->mpd = 500; eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd); }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type\n");
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(eps->t,eps->ncv,&ctx->AV);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_Subspace;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSHessCond"
/*
   EPSHessCond - Compute the inf-norm condition number of the upper 
   Hessenberg matrix H: cond(H) = norm(H)*norm(inv(H)).
   This routine uses Gaussian elimination with partial pivoting to 
   compute the inverse explicitly. 
*/
static PetscErrorCode EPSHessCond(PetscInt n_,PetscScalar* H,PetscInt ldh_,PetscReal* cond)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(SLEPC_MISSING_LAPACK_GETRI) || defined(SLEPC_MISSING_LAPACK_LANGE) || defined(SLEPC_MISSING_LAPACK_LANHS)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF,GETRI - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   *ipiv,lwork,info,n=n_,ldh=ldh_;
  PetscScalar    *work;
  PetscReal      hn,hin,*rwork;
  
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscBLASInt)*n,&ipiv);CHKERRQ(ierr);
  lwork = n*n;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*n,&rwork);CHKERRQ(ierr);
  hn = LAPACKlanhs_("I",&n,H,&ldh,rwork);
  LAPACKgetrf_(&n,&n,H,&ldh,ipiv,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRF %d",info);
  LAPACKgetri_(&n,H,&ldh,ipiv,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRI %d",info);
  hin = LAPACKlange_("I",&n,&n,H,&ldh,rwork);
  *cond = hn * hin;
  ierr = PetscFree(ipiv);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFindGroup"
/*
   EPSFindGroup - Find a group of nearly equimodular eigenvalues, provided 
   in arrays wr and wi, according to the tolerance grptol. Also the 2-norms
   of the residuals must be passed-in (rsd). Arrays are processed from index 
   l to index m only. The output information is:

   ngrp - number of entries of the group
   ctr  - (w(l)+w(l+ngrp-1))/2
   ae   - average of wr(l),...,wr(l+ngrp-1)
   arsd - average of rsd(l),...,rsd(l+ngrp-1)
*/
static PetscErrorCode EPSFindGroup(PetscInt l,PetscInt m,PetscScalar *wr,PetscScalar *wi,PetscReal *rsd,
  PetscReal grptol,PetscInt *ngrp,PetscReal *ctr,PetscReal *ae,PetscReal *arsd)
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
#define __FUNCT__ "EPSSchurResidualNorms"
/*
   EPSSchurResidualNorms - Computes the column norms of residual vectors
   OP*V(1:n,l:m) - V*T(1:m,l:m), where, on entry, OP*V has been computed and 
   stored in AV. ldt is the leading dimension of T. On exit, rsd(l) to
   rsd(m) contain the computed norms.
*/
static PetscErrorCode EPSSchurResidualNorms(EPS eps,Vec *V,Vec *AV,PetscScalar *T,PetscInt l,PetscInt m,PetscInt ldt,PetscReal *rsd)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    t;
#endif

  PetscFunctionBegin;
  for (i=l;i<m;i++) {
    if (i==m-1 || T[i+1+ldt*i]==0.0) k=i+1; else k=i+2;
    ierr = VecCopy(AV[i],eps->work[0]);CHKERRQ(ierr);
    ierr = SlepcVecMAXPBY(eps->work[0],1.0,-1.0,k,T+ldt*i,V);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = VecDot(eps->work[0],eps->work[0],rsd+i);CHKERRQ(ierr);
#else
    ierr = VecDot(eps->work[0],eps->work[0],&t);CHKERRQ(ierr);
    rsd[i] = PetscRealPart(t);
#endif    
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
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE *)eps->data;
  PetscInt       i,k,ngrp,nogrp,*itrsd,*itrsdold,
                 nxtsrr,idsrr,idort,nxtort,nv,ncv = eps->ncv,its;
  PetscScalar    *T=eps->T,*U;
  PetscReal      arsd,oarsd,ctr,octr,ae,oae,*rsd,norm,tcond=1.0;
  PetscBool      breakdown;
  /* Parameters */
  PetscInt       init = 5;        /* Number of initial iterations */
  PetscReal      stpfac = 1.5,    /* Max num of iter before next SRR step */
                 alpha = 1.0,     /* Used to predict convergence of next residual */
                 beta = 1.1,      /* Used to predict convergence of next residual */
                 grptol = 1e-8,   /* Tolerance for EPSFindGroup */
                 cnvtol = 1e-6;   /* Convergence criterion for cnv */
  PetscInt       orttol = 2;      /* Number of decimal digits whose loss
                                     can be tolerated in orthogonalization */

  PetscFunctionBegin;
  its = 0;
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv*ncv,&U);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncv,&itrsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncv,&itrsdold);CHKERRQ(ierr);

  for (i=0;i<ncv;i++) {
    rsd[i] = 0.0;
    itrsd[i] = -1;
  }
  
  /* Complete the initial basis with random vectors and orthonormalize them */
  k = eps->nini;
  while (k<ncv) {
    ierr = SlepcVecSetRandom(eps->V[k],eps->rand);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,k,PETSC_NULL,eps->V,eps->V[k],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr); 
    if (norm>0.0 && !breakdown) {
      ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
      k++;
    }
  }

  while (eps->its<eps->max_it) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    
    /* Find group in previously computed eigenvalues */
    ierr = EPSFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,rsd,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

    /* Compute a Rayleigh-Ritz projection step 
       on the active columns (idx) */

    /* 1. AV(:,idx) = OP * V(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = STApply(eps->OP,eps->V[i],ctx->AV[i]);CHKERRQ(ierr);
    }

    /* 2. T(:,idx) = V' * AV(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = VecMDot(ctx->AV[i],nv,eps->V,T+i*ncv);CHKERRQ(ierr);
    }

    /* 3. Reduce projected matrix to Hessenberg form: [U,T] = hess(T) */
    ierr = EPSDenseHessenberg(nv,eps->nconv,T,ncv,U);CHKERRQ(ierr);
    
    /* 4. Reduce T to quasi-triangular (Schur) form */
    ierr = EPSDenseSchur(nv,eps->nconv,T,ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* 5. Sort diagonal elements in T and accumulate rotations on U */
    ierr = EPSSortDenseSchur(eps,nv,eps->nconv,T,ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);
    
    /* 6. AV(:,idx) = AV * U(:,idx) */
    ierr = SlepcUpdateVectors(nv,ctx->AV,eps->nconv,nv,U,nv,PETSC_FALSE);CHKERRQ(ierr);
    
    /* 7. V(:,idx) = V * U(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,nv,U,nv,PETSC_FALSE);CHKERRQ(ierr);
    
    /* Convergence check */
    ierr = EPSSchurResidualNorms(eps,eps->V,ctx->AV,T,eps->nconv,nv,ncv,rsd);CHKERRQ(ierr);

    for (i=eps->nconv;i<nv;i++) { 
      itrsdold[i] = itrsd[i];
      itrsd[i] = its;
      eps->errest[i] = rsd[i];
    }
  
    for (;;) {
      /* Find group in currently computed eigenvalues */
      ierr = EPSFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
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
    ierr = PetscMemcpy(U,T,sizeof(PetscScalar)*ncv*ncv);CHKERRQ(ierr);
    ierr = EPSHessCond(nv,U,ncv,&tcond);CHKERRQ(ierr);
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
          ierr = STApply(eps->OP,eps->V[i],ctx->AV[i]);CHKERRQ(ierr);
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
        ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,i,PETSC_NULL,eps->V,eps->V[i],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          ierr = SlepcVecSetRandom(eps->V[i],eps->rand);CHKERRQ(ierr);
          ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,i,PETSC_NULL,eps->V,eps->V[i],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        }
        ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
      }
      nxtort = PetscMin(its+idort,nxtsrr);
    } while (its<nxtsrr);
  }

  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(rsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsdold);CHKERRQ(ierr);

  if (eps->nconv == eps->nev) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_Subspace"
PetscErrorCode EPSReset_Subspace(EPS eps)
{
  PetscErrorCode ierr;
  EPS_SUBSPACE   *ctx = (EPS_SUBSPACE *)eps->data;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(eps->ncv,&ctx->AV);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
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

