/*                       

   SLEPc eigensolver: "subspace"

   Method: Subspace Iteration

   Description:

       This solver implements a version of the subspace iteration (or
       simultaneous iteration) for computing an orthogonal basis of an
       invariant subspace associated to the dominant eigenpairs. 

   Algorithm:

       The implemented algorithm is based on the SRRIT implementation (see
       reference below).

       The basic subspace iteration is a generalization of the power
       method to m vectors, enforcing orthogonality between them to avoid
       linear dependence. In addition, this implementation performs a
       Rayleigh-Ritz projection procedure in order to improve convergence.
       Deflation is handled by locking converged eigenvectors. For better
       performance, orthogonalization and projection are performed only
       when necessary.

   References:

       [1] G.W. Stewart and Z. Bai, "Algorithm 776. SRRIT - A Fortran 
       Subroutine to Calculate the Dominant Invariant Subspace of a 
       Nonsymmetric Matrix", ACM Transactions on Mathematical Software, 
       23(4), pp. 494-513 (1997).

   Last update: June 2004

*/
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_SUBSPACE"
PetscErrorCode EPSSetUp_SUBSPACE(EPS eps)
{
  PetscErrorCode ierr;
  int            N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->T) { ierr = PetscFree(eps->T);CHKERRQ(ierr); }  
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv);CHKERRQ(ierr);
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
static PetscErrorCode EPSHessCond(PetscScalar* H,int n, PetscReal* cond)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(SLEPC_MISSING_LAPACK_GETRI) || defined(SLEPC_MISSING_LAPACK_LANGE) || defined(SLEPC_MISSING_LAPACK_LANHS)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GETRF,GETRI - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  int            *ipiv,lwork,info;
  PetscScalar    *work;
  PetscReal      hn,hin,*rwork;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(int)*n,&ipiv);CHKERRQ(ierr);
  lwork = n*n;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*n,&rwork);CHKERRQ(ierr);
  hn = LAPACKlanhs_("I",&n,H,&n,rwork,1);
  LAPACKgetrf_(&n,&n,H,&n,ipiv,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGETRF %d",info);
  LAPACKgetri_(&n,H,&n,ipiv,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGETRI %d",info);
  hin = LAPACKlange_("I",&n,&n,H,&n,rwork,1);
  *cond = hn * hin;
  ierr = PetscFree(ipiv);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
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
static PetscErrorCode EPSFindGroup(int l,int m,PetscScalar *wr,PetscScalar *wi,PetscReal *rsd,
  PetscReal grptol,int *ngrp,PetscReal *ctr,PetscReal *ae,PetscReal *arsd)
{
  int       i;
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
   OP*V(1:n,l:m) - V*T(1:m,l:m) were on entry, OP*V has been computed and 
   stored in AV. ldt is the leading dimension of T. On exit, rsd(l) to
   rsd(m) contain the computed norms.
*/
static PetscErrorCode EPSSchurResidualNorms(EPS eps,Vec *V,Vec *AV,PetscScalar *T,int l,int m,int ldt,PetscReal *rsd)
{
  PetscErrorCode ierr;
  int            i;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    t;
#endif

  PetscFunctionBegin;
  for (i=l;i<m;i++) {
    ierr = VecSet(eps->work[0],0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(eps->work[0],m,T+ldt*i,V);CHKERRQ(ierr);
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
      rsd[i] = sqrt(rsd[i]+rsd[i+1])/2.0;
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
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(SLEPC_MISSING_LAPACK_UNGHR)
  SETERRQ(PETSC_ERR_SUP,"GEHRD,ORGHR/UNGHR - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  int            i,j,ilo,lwork,info,ngrp,nogrp,*itrsd,*itrsdold,
                 nxtsrr,idsrr,*iwork,idort,nxtort,ncv = eps->ncv;
  PetscScalar    *T=eps->T,*U,*tau,*work;
  PetscReal      arsd,oarsd,ctr,octr,ae,oae,*rsd,*rsdold,norm,tcond;
  PetscTruth     breakdown;
  /* Parameters */
  int            init = 5;        /* Number of initial iterations */
  PetscReal      stpfac = 1.5,    /* Max num of iter before next SRR step */
                 alpha = 1.0,     /* Used to predict convergence of next residual */
                 beta = 1.1,      /* Used to predict convergence of next residual */
                 grptol = 1e-8,   /* Tolerance for EPSFindGroup */
                 cnvtol = 1e-6;   /* Convergence criterion for cnv */
  int            orttol = 2;      /* Number of decimal digits whose loss
                                     can be tolerated in orthogonalization */

  PetscFunctionBegin;
  eps->its = 0;
  eps->nconv = 0;
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv*ncv,&U);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv,&tau);CHKERRQ(ierr);
  lwork = ncv*ncv;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&itrsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&itrsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&iwork);CHKERRQ(ierr);

  /* Generate a set of random initial vectors and orthonormalize them */
  for (i=0;i<ncv;i++) {
    ierr = SlepcVecSetRandom(eps->V[i]);CHKERRQ(ierr);
    eps->eigr[i] = 0.0;
    eps->eigi[i] = 0.0;
    rsd[i] = 0.0;
    itrsd[i] = -1;
  }
  ierr = EPSQRDecomposition(eps,eps->V,0,ncv,PETSC_NULL,0);CHKERRQ(ierr);
  
  while (eps->its<eps->max_it) {

    /* Find group in previously computed eigenvalues */
    ierr = EPSFindGroup(eps->nconv,ncv,eps->eigr,eps->eigi,rsd,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

    /* Compute a Rayleigh-Ritz projection step 
       on the active columns (idx) */

    /* 1. AV(:,idx) = OP * V(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
    }

    /* 2. T(:,idx) = V' * AV(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecMDot(ncv,eps->AV[i],eps->V,T+i*ncv);CHKERRQ(ierr);
    }

    /* 3. Reduce projected matrix to Hessenberg form: [U,T] = hess(T) */
    ilo = eps->nconv + 1;
    LAPACKgehrd_(&ncv,&ilo,&ncv,T,&ncv,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
    for (j=0;j<ncv-1;j++) {
      for (i=j+2;i<ncv;i++) {
        U[i+j*ncv] = T[i+j*ncv];
        T[i+j*ncv] = 0.0;
      }      
    }
    LAPACKorghr_(&ncv,&ilo,&ncv,U,&ncv,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
    
    /* 4. Reduce T to quasi-triangular (Schur) form */
    ierr = EPSDenseSchur(ncv,eps->nconv,T,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* 5. Sort diagonal elements in T and accumulate rotations on U */
    ierr = EPSSortDenseSchur(ncv,eps->nconv,T,U,eps->eigr,eps->eigi);CHKERRQ(ierr);
    
    /* 6. AV(:,idx) = AV * U(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecSet(eps->work[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->work[i],ncv,U+ncv*i,eps->AV);CHKERRQ(ierr);
    }    
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->work[i],eps->AV[i]);CHKERRQ(ierr);
    }    
    
    /* 7. V(:,idx) = V * U(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecSet(eps->work[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->work[i],ncv,U+ncv*i,eps->V);CHKERRQ(ierr);
    }    
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->work[i],eps->V[i]);CHKERRQ(ierr);
    }    
    
    /* Compute residuals */
    for (i=0;i<ncv;i++) { rsdold[i] = rsd[i]; }

    ierr = EPSSchurResidualNorms(eps,eps->V,eps->AV,T,eps->nconv,ncv,ncv,rsd);CHKERRQ(ierr);

    for (i=0;i<ncv;i++) { 
      eps->errest[i] = rsd[i] / SlepcAbsEigenvalue(eps->eigr[i],eps->eigi[i]); 
    }
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv); 
  
    /* Convergence check */
    for (i=0;i<ncv;i++) { itrsdold[i] = itrsd[i]; }
    for (i=eps->nconv;i<ncv;i++) { itrsd[i] = eps->its; }
    
    for (;;) {
      /* Find group in currently computed eigenvalues */
      ierr = EPSFindGroup(eps->nconv,ncv,eps->eigr,eps->eigi,rsd,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsScalar(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=ncv) break;
    }
    
    if (eps->nconv>=eps->nev) break;
    
    /* Compute nxtsrr (iteration of next projection step) */
    nxtsrr = PetscMin(eps->max_it,PetscMax((int)floor(stpfac*eps->its), init));
    
    if (ngrp!=nogrp || ngrp==0 || arsd>=oarsd) {
      idsrr = nxtsrr - eps->its;
    } else {
      idsrr = (int)floor(alpha+beta*(itrsdold[eps->nconv]-itrsd[eps->nconv])*log(arsd/eps->tol)/log(arsd/oarsd));
      idsrr = PetscMax(1,idsrr);
    }
    nxtsrr = PetscMin(nxtsrr,eps->its+idsrr);

    /* Compute nxtort (iteration of next orthogonalization step) */
    ierr = PetscMemcpy(U,T,sizeof(PetscScalar)*ncv);CHKERRQ(ierr);
    ierr = EPSHessCond(U,ncv,&tcond);CHKERRQ(ierr);
    idort = PetscMax(1,(int)floor(orttol/PetscMax(1,log10(tcond))));    
    nxtort = PetscMin(eps->its+idort, nxtsrr);

    /* V(:,idx) = AV(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    eps->its++;

    /* Orthogonalization loop */
    do {
      while (eps->its<nxtort) {
      
        /* AV(:,idx) = OP * V(:,idx) */
        for (i=eps->nconv;i<ncv;i++) {
          ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
        }
        
        /* V(:,idx) = AV(:,idx) with normalization */
        for (i=eps->nconv;i<ncv;i++) {
          ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
          ierr = VecNorm(eps->V[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
        }
      
        eps->its++;
      }
      /* Orthonormalize vectors */
      for (i=eps->nconv;i<ncv;i++) {
        ierr = EPSOrthogonalize(eps,i+eps->nds,eps->DSV,eps->V[i],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          ierr = SlepcVecSetRandom(eps->V[i]);CHKERRQ(ierr);
          ierr = EPSOrthogonalize(eps,i+eps->nds,eps->DSV,eps->V[i],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        }
        ierr = VecScale(eps->V[i],1/norm);CHKERRQ(ierr);
      }
      nxtort = PetscMin(eps->its+idort,nxtsrr);
    } while (eps->its<nxtsrr);
  }

  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(rsd);CHKERRQ(ierr);
  ierr = PetscFree(rsdold);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(itrsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsdold);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
#endif 
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_SUBSPACE"
PetscErrorCode EPSCreate_SUBSPACE(EPS eps)
{
  PetscFunctionBegin;
  eps->ops->solve                = EPSSolve_SUBSPACE;
  eps->ops->setup                = EPSSetUp_SUBSPACE;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

