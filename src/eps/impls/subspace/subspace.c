/*                       

   SLEPc eigensolver: "subspace"

   Method: Subspace Iteration

   Algorithm:

       Subspace iteration with Rayleigh-Ritz projection and locking,
       based on the SRRIT implementation.

   References:

       [1] "Subspace Iteration in SLEPc", SLEPc Technical Report STR-3, 
           available at http://www.grycap.upv.es/slepc.

   Last update: June 2004

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_SUBSPACE"
PetscErrorCode EPSSetUp_SUBSPACE(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,N,nloc;
  PetscScalar    *pAV;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(N,eps->nev+eps->mpd);
  }
  else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
    else { eps->mpd = 500; eps->ncv = PetscMin(N,eps->nev+eps->mpd); }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (!eps->max_it) eps->max_it = PetscMax(100,2*N/eps->ncv);
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) {
    SETERRQ(PETSC_ERR_SUP,"Unsupported extraction type\n");
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = VecGetLocalSize(eps->vec_initial,&nloc);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->AV);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*nloc*sizeof(PetscScalar),&pAV);CHKERRQ(ierr);
  for (i=0;i<eps->ncv;i++) {
    ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,nloc,PETSC_DECIDE,pAV+i*nloc,&eps->AV[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
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
  SETERRQ(PETSC_ERR_SUP,"GETRF,GETRI - Lapack routines are unavailable.");
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
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGETRF %d",info);
  LAPACKgetri_(&n,H,&ldh,ipiv,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGETRI %d",info);
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
    ierr = VecSet(eps->work[0],0.0);CHKERRQ(ierr);
    if (i==m-1 || T[i+1+ldt*i]==0.0) k=i+1; else k=i+2;
    ierr = VecMAXPY(eps->work[0],k,T+ldt*i,V);CHKERRQ(ierr);
    ierr = VecWAXPY(eps->work[1],-1.0,eps->work[0],AV[i]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = VecDot(eps->work[1],eps->work[1],rsd+i);CHKERRQ(ierr);
#else
    ierr = VecDot(eps->work[1],eps->work[1],&t);CHKERRQ(ierr);
    rsd[i] = PetscRealPart(t);
#endif    
  }

  for (i=l;i<m;i++) {
    if (i == m-1) {
      rsd[i] = sqrt(rsd[i]);  
    } else if (T[i+1+(ldt*i)]==0.0) {
      rsd[i] = sqrt(rsd[i]);
    } else {
      rsd[i] = sqrt((rsd[i]+rsd[i+1])/2.0);
      rsd[i+1] = rsd[i];
      i++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_SUBSPACE"
PetscErrorCode EPSSolve_SUBSPACE(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,ngrp,nogrp,*itrsd,*itrsdold,
                 nxtsrr,idsrr,idort,nxtort,nv,ncv = eps->ncv,its;
  PetscScalar    *T=eps->T,*U;
  PetscReal      arsd,oarsd,ctr,octr,ae,oae,*rsd,*rsdold,norm,tcond=1.0;
  PetscTruth     breakdown;
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
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncv,&itrsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*ncv,&itrsdold);CHKERRQ(ierr);

  /* Generate a set of random initial vectors and orthonormalize them */
  for (i=0;i<ncv;i++) {
    ierr = SlepcVecSetRandom(eps->V[i]);CHKERRQ(ierr);
    rsd[i] = 0.0;
    itrsd[i] = -1;
  }
  ierr = IPQRDecomposition(eps->ip,eps->V,0,ncv,PETSC_NULL,0,eps->work[0]);CHKERRQ(ierr);
  
  while (eps->its<eps->max_it) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    
    /* Find group in previously computed eigenvalues */
    ierr = EPSFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,rsd,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

    /* Compute a Rayleigh-Ritz projection step 
       on the active columns (idx) */

    /* 1. AV(:,idx) = OP * V(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
    }

    /* 2. T(:,idx) = V' * AV(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = VecMDot(eps->AV[i],nv,eps->V,T+i*ncv);CHKERRQ(ierr);
    }

    /* 3. Reduce projected matrix to Hessenberg form: [U,T] = hess(T) */
    ierr = EPSDenseHessenberg(nv,eps->nconv,T,ncv,U);CHKERRQ(ierr);
    
    /* 4. Reduce T to quasi-triangular (Schur) form */
    ierr = EPSDenseSchur(nv,eps->nconv,T,ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* 5. Sort diagonal elements in T and accumulate rotations on U */
    ierr = EPSSortDenseSchur(nv,eps->nconv,T,ncv,U,eps->eigr,eps->eigi,eps->which);CHKERRQ(ierr);
    
    /* 6. AV(:,idx) = AV * U(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->AV,eps->nconv,nv,U,nv,PETSC_FALSE);CHKERRQ(ierr);
    
    /* 7. V(:,idx) = V * U(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,nv,U,nv,PETSC_FALSE);CHKERRQ(ierr);
    
    /* Compute residuals */
    for (i=0;i<nv;i++) { rsdold[i] = rsd[i]; }

    ierr = EPSSchurResidualNorms(eps,eps->V,eps->AV,T,eps->nconv,nv,ncv,rsd);CHKERRQ(ierr);

    for (i=0;i<nv;i++) { 
      eps->errest[i] = rsd[i] / SlepcAbsEigenvalue(eps->eigr[i],eps->eigi[i]); 
    }
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv); 
  
    /* Convergence check */
    for (i=0;i<nv;i++) { itrsdold[i] = itrsd[i]; }
    for (i=eps->nconv;i<nv;i++) { itrsd[i] = its; }
    
    for (;;) {
      /* Find group in currently computed eigenvalues */
      ierr = EPSFindGroup(eps->nconv,nv,eps->eigr,eps->eigi,rsd,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsScalar(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=nv) break;
    }
    
    if (eps->nconv>=eps->nev) break;
    
    /* Compute nxtsrr (iteration of next projection step) */
    nxtsrr = PetscMin(eps->max_it,PetscMax((PetscInt)floor(stpfac*its), init));
    
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
    nxtort = PetscMin(its+idort, nxtsrr);

    /* V(:,idx) = AV(:,idx) */
    for (i=eps->nconv;i<nv;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    its++;

    /* Orthogonalization loop */
    do {
      while (its<nxtort) {
      
        /* AV(:,idx) = OP * V(:,idx) */
        for (i=eps->nconv;i<nv;i++) {
          ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
        }
        
        /* V(:,idx) = AV(:,idx) with normalization */
        for (i=eps->nconv;i<nv;i++) {
          ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
          ierr = VecNorm(eps->V[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
        }
      
        its++;
      }
      /* Orthonormalize vectors */
      for (i=eps->nconv;i<nv;i++) {
        ierr = IPOrthogonalize(eps->ip,i+eps->nds,PETSC_NULL,eps->DSV,eps->V[i],PETSC_NULL,&norm,&breakdown,eps->work[0],PETSC_NULL);CHKERRQ(ierr);
        if (breakdown) {
          ierr = SlepcVecSetRandom(eps->V[i]);CHKERRQ(ierr);
          ierr = IPOrthogonalize(eps->ip,i+eps->nds,PETSC_NULL,eps->DSV,eps->V[i],PETSC_NULL,&norm,&breakdown,eps->work[0],PETSC_NULL);CHKERRQ(ierr);
        }
        ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
      }
      nxtort = PetscMin(its+idort,nxtsrr);
    } while (its<nxtsrr);
  }

  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(rsd);CHKERRQ(ierr);
  ierr = PetscFree(rsdold);CHKERRQ(ierr);
  ierr = PetscFree(itrsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsdold);CHKERRQ(ierr);

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_SUBSPACE"
PetscErrorCode EPSDestroy_SUBSPACE(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *pAV;

  PetscFunctionBegin;
  ierr = VecGetArray(eps->AV[0],&pAV);CHKERRQ(ierr);
  for (i=0;i<eps->ncv;i++) {
    ierr = VecDestroy(eps->AV[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pAV);CHKERRQ(ierr);
  ierr = PetscFree(eps->AV);CHKERRQ(ierr);
  ierr = EPSDestroy_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_SUBSPACE"
PetscErrorCode EPSCreate_SUBSPACE(EPS eps)
{
  PetscFunctionBegin;
  eps->ops->solve                = EPSSolve_SUBSPACE;
  eps->ops->setup                = EPSSetUp_SUBSPACE;
  eps->ops->destroy              = EPSDestroy_SUBSPACE;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

