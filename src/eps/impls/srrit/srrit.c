
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_SRRIT"
static int EPSSetUp_SRRIT(EPS eps)
{
  int       ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+15);
  eps->ncv = PetscMin(eps->ncv,N);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSdcond"
static int EPSdcond(PetscScalar* H,int n, PetscReal* cond)
{
  int         ierr,*ipiv,lwork;
  PetscScalar *work;
  PetscReal   hn,hin,*rwork;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(int)*n,&ipiv);CHKERRQ(ierr);
  lwork = n*n;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*n,&rwork);CHKERRQ(ierr);
  hn = LAlanhs_("I",&n,H,&n,rwork,1);
  LAgetrf_(&n,&n,H,&n,ipiv,&ierr);
  if (ierr) SETERRQ(PETSC_ERR_LIB,"Error in Lapack xGETRF");
  LAgetri_(&n,H,&n,ipiv,work,&lwork,&ierr);
  if (ierr) SETERRQ(PETSC_ERR_LIB,"Error in Lapack xGETRI");
  hin = LAlange_("I",&n,&n,H,&n,rwork,1);
  *cond = hn * hin;
  ierr = PetscFree(ipiv);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSdgroup"
static int EPSdgroup(int l,int m,PetscScalar *wr,PetscScalar *wi,PetscReal *rsd,
  PetscReal grptol,int *ngrp,PetscReal *ctr,PetscReal *ae,PetscReal *arsd)
{
  int       i;
  PetscReal rmod,rmod1;

  PetscFunctionBegin;
  *ngrp = 0;
  *ctr = 0;
      
#if !defined(PETSC_USE_COMPLEX)
  rmod = LAlapy2_(wr+l,wi+l);
#else 
  rmod = PetscAbsScalar(wr[l]);
#endif

  for (i=l;i<m;) {
#if !defined(PETSC_USE_COMPLEX)
    rmod1 = LAlapy2_(wr+i,wi+i);
#else 
    rmod1 = PetscAbsScalar(wr[i]);
#endif
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
static int EPSSchurResidualNorms(EPS eps,Vec *V,Vec *AV,PetscScalar *T,int l,int m,int ldt,PetscReal *rsd)
{
  int         ierr,i;
  PetscScalar zero = 0.0,minus = -1.0;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar t;
#endif

  PetscFunctionBegin;
  for (i=l;i<m;i++) {
    ierr = VecSet(&zero,eps->work[0]);CHKERRQ(ierr);
    ierr = VecMAXPY(m,T+ldt*i,eps->work[0],V);CHKERRQ(ierr);
    ierr = VecWAXPY(&minus,eps->work[0],AV[i],eps->work[1]);CHKERRQ(ierr);
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
#define __FUNCT__ "EPSSolve_SRRIT"
static int EPSSolve_SRRIT(EPS eps)
{
  int         ierr,i,j,ilo,lwork,ngrp,nogrp,*itrsd,*itrsdold,
              nxtsrr,idsrr,*iwork,idort,nxtort,ncv = eps->ncv;
  PetscScalar *T,*U,*tau,*work,t;
  PetscReal   arsd,oarsd,ctr,octr,ae,oae,*rsdold,norm,tcond;
  /* Parameters */
  int         init = 5;
  PetscReal   stpfac = 1.5,
              alpha = 1.0,
              beta = 1.1,
              grptol = 1e-8,
              cnvtol = 1e-6;
  int         orttol = 2;

  PetscFunctionBegin;
  eps->its = 0;
  eps->nconv = 0;
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv*ncv,&T);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv*ncv,&U);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv,&tau);CHKERRQ(ierr);
  lwork = ncv*ncv;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&itrsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&itrsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&iwork);CHKERRQ(ierr);
  for (i=0;i<ncv;i++) {
    ierr = SlepcVecSetRandom(eps->V[i]);CHKERRQ(ierr);
    eps->eigr[i] = 0;
    eps->eigi[i] = 0;
    eps->errest[i] = 0;
    itrsd[i] = -1;
  }
  ierr = EPSQRDecomposition(eps,eps->V,0,ncv,PETSC_NULL,0);CHKERRQ(ierr);
  
  while (eps->its<eps->max_it) {

    /* [ nogrp, octr, oae, oarsd ] = dgroup( nconv+1, ncv, wr, wi, rsd, grptol ) */
    ierr = EPSdgroup(eps->nconv,ncv,eps->eigr,eps->eigi,eps->errest,grptol,&nogrp,&octr,&oae,&oarsd);CHKERRQ(ierr);

    /* AV(:,idx) = stapply(st,V(:,idx)) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
    }

    /* T(:,idx) = V'*AV(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecMDot(ncv,eps->AV[i],eps->V,T+i*ncv);CHKERRQ(ierr);
    }

    /* [U,H] = hess(T) */
    ilo = eps->nconv + 1;
    LAgehrd_(&ncv,&ilo,&ncv,T,&ncv,tau,work,&lwork,&ierr);
    if (ierr) SETERRQ(PETSC_ERR_LIB,"Error in Lapack xGEHRD");
    for (j=0;j<ncv-1;j++) {
      for (i=j+2;i<ncv;i++) {
        U[i+j*ncv] = T[i+j*ncv];
        T[i+j*ncv] = 0.0;
      }      
    }
    LAorghr_(&ncv,&ilo,&ncv,U,&ncv,tau,work,&lwork,&ierr);
    if (ierr) SETERRQ(PETSC_ERR_LIB,"Error in Lapack xORGHR");
    
    /* [T,wr,wi,U] = laqr3(H,U) */
    ierr = EPSDenseSchur(T,U,eps->eigr,eps->eigi,eps->nconv,ncv);CHKERRQ(ierr);
    
    /* AV(:,idx) = AV*U(:,idx) */
    ierr = EPSReverseProjection(eps,eps->AV,U,eps->nconv,ncv,eps->work);CHKERRQ(ierr);
    
    /* V(:,idx) = V*U(:,idx) */
    ierr = EPSReverseProjection(eps,eps->V,U,eps->nconv,ncv,eps->work);CHKERRQ(ierr);
    
    /* rsdold = rsd */
    for (i=0;i<ncv;i++) { rsdold[i] = eps->errest[i]; }

    /* rsd(idx) = SchurResidualNorms(V,AV,T,nconv,ncv) */
    ierr = EPSSchurResidualNorms(eps,eps->V,eps->AV,T,eps->nconv,ncv,ncv,eps->errest);CHKERRQ(ierr);

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv); 
  
    /* itrsdold = itrsd;
       for j=idx, itrsd(j)=its; end */
    for (i=0;i<ncv;i++) { itrsdold[i] = itrsd[i]; }
    for (i=eps->nconv;i<ncv;i++) { itrsd[i] = eps->its; }
    
    for (;;) {
      /* [ ngrp, ctr, ae, arsd ] = dgroup( nconv+1, ncv, wr, wi, rsd, grptol ) */
      ierr = EPSdgroup(eps->nconv,ncv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd);CHKERRQ(ierr);
      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsScalar(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=ncv) break;
    }
    
    if (eps->nconv>=eps->nev) break;
    
    /* nxtsrr = min([maxit max([fix(stpfac*its) init])]) */
    nxtsrr = PetscMin(eps->max_it,PetscMax(floor(stpfac*eps->its), init));
    
    if (ngrp!=nogrp || ngrp==0 || arsd>=oarsd) {
      idsrr = nxtsrr - eps->its;
    } else {
      /* idsrr = max([1 alpha+beta*(itrsdold(nconv+1)-itrsd(nconv+1))*log(arsd/tol)/log(arsd/oarsd)]) */
      idsrr = floor(alpha+beta*(itrsdold[eps->nconv]-itrsd[eps->nconv])*log(arsd/eps->tol)/log(arsd/oarsd));
      idsrr = PetscMax(1,idsrr);
    }
    nxtsrr = PetscMin(nxtsrr,eps->its+idsrr);

    /* tcond = cond(T,inf) */
    ierr = PetscMemcpy(U,T,sizeof(PetscScalar)*ncv);CHKERRQ(ierr);
    /* tcond = dcond_(&ncv,U,&ncv,iwork,work,&ierr); */
    ierr = EPSdcond(U,ncv,&tcond);CHKERRQ(ierr);
    
    /* idort = max([1 fix(orttol/max([1 log10(tcond)]))]) */
    idort = PetscMax(1,floor(orttol/PetscMax(1,log10(tcond))));    
    nxtort = PetscMin(eps->its+idort, nxtsrr);

    /* V(:,idx) = AV(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    eps->its++;

    do {
      while (eps->its<nxtort) {
      
        /* AV(:,idx) = stapply(st,V(:,idx)) */
        for (i=eps->nconv;i<ncv;i++) {
          ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
        }
        
        /* V(:,idx) = AV(:,idx)/norm(AV(:,idx),inf) */
        for (i=eps->nconv;i<ncv;i++) {
          ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
          ierr = VecNorm(eps->V[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          t = 1 / norm;
          ierr = VecScale(&t,eps->V[i]);CHKERRQ(ierr);
        }
      
        eps->its++;
      }
      for (i=eps->nconv;i<ncv;i++) {
        /* v = repgs(V(:,1:j-1),V(:,j))
           V(:,j) = v/norm(v) */
        ierr = (*eps->orthog)(eps,i+eps->nds,eps->DSV,eps->V[i],PETSC_NULL,&norm);CHKERRQ(ierr);
        if (norm < 1e-8) { SETERRQ(1,"Norm is zero"); }
        t = 1 / norm;
        ierr = VecScale(&t,eps->V[i]);CHKERRQ(ierr);
      }
      nxtort = PetscMin(eps->its+idort,nxtsrr);
    } while (eps->its<nxtsrr);
  }

  ierr = PetscFree(T);CHKERRQ(ierr);
  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(rsdold);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(itrsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsdold);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_SRRIT"
int EPSComputeVectors_SRRIT(EPS eps)
{
  int ierr;
  PetscFunctionBegin;
  ierr = EPSComputeVectors_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_SRRIT"
int EPSCreate_SRRIT(EPS eps)
{
  PetscFunctionBegin;
  eps->ops->setup                = EPSSetUp_SRRIT;
  eps->ops->solve                = EPSSolve_SRRIT;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->computevectors            = EPSComputeVectors_SRRIT;
  PetscFunctionReturn(0);
}
EXTERN_C_END

