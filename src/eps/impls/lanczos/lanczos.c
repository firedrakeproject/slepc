/*                       

   SLEPc eigensolver: "lanczos"

   Method: 

   Description:

   Algorithm:

   References:

   Last update: 

*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LANCZOS"
PetscErrorCode EPSSetUp_LANCZOS(EPS eps)
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
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->T) { ierr = PetscFree(eps->T);CHKERRQ(ierr); }  
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicLanczos"
/*
   EPSBasicLanczos - 
                    OP * V - V * H = f * e_m^T
*/
static PetscErrorCode EPSBasicLanczos(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            j;
  PetscReal      norm;
  PetscScalar    t;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,V,V[j+1],H+m*j,&norm,&breakdown);CHKERRQ(ierr);
    H[(m+1)*j+1] = norm;
    if (breakdown) {
      PetscLogInfo(eps,"Breakdown in Arnoldi method (norm=%g)\n",norm);
      ierr = SlepcVecSetRandom(V[j+1]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[j+1],&norm);CHKERRQ(ierr);
    }
    t = 1 / norm;
    ierr = VecScale(&t,V[j+1]);CHKERRQ(ierr);
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  ierr = EPSOrthogonalize(eps,m,V,f,H+m*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LANCZOS"
PetscErrorCode EPSSolve_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  int            i,j,n,m,*iwork,*ifail,info,ncv=eps->ncv,nconv;
  Vec            f=eps->work[ncv];
  PetscScalar    *T=eps->T,*D,*E,*Y,*W,*work,ts;
  PetscReal      norm,beta,abstol;

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"xSTEVX - Lapack routine is unavailable.");
#endif 
  ierr = PetscMemzero(T,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&D);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&W);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(5*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(5*ncv*sizeof(PetscScalar),&iwork);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&ifail);CHKERRQ(ierr);

  /* The first vector is the normalized initial vector */
  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = STNorm(eps->OP,eps->V[0],&norm);CHKERRQ(ierr);
  ts = 1 / norm;
  ierr = VecScale(&ts,eps->V[0]);CHKERRQ(ierr);
  
  abstol = 2*LAlamch_("S",1);
  
  eps->nconv = nconv = 0;
  eps->its = 0;
  while (eps->its<eps->max_it) {
    ierr = EPSBasicLanczos(eps,T,eps->V,nconv,ncv,f,&beta);CHKERRQ(ierr);

    n = ncv - nconv;
    for (i=0;i<n;i++) {
      D[i] = T[(i+nconv)*(ncv+1)];
      E[i] = T[(i+nconv)*(ncv+1)+1];
    }
    LAstevx_("V","A",&n,D,E,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&abstol,&m,W,Y,&n,work,iwork,ifail,&info,1,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEVX %i",info);
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) */
    for (i=0;i<n;i++) {
      eps->eigr[ncv-i-1] = W[i];
      eps->errest[ncv-i-1] = beta*PetscAbsScalar(Y[i*n+n-1]);
    }
    
    /* Update V(:,idx) = V*U(:,idx) */
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++) 
          T[i+nconv+(j+nconv)*ncv] = Y[i+(n-j-1)*n];
    ierr = EPSReverseProjection(eps,eps->V,T,nconv,ncv,eps->work);CHKERRQ(ierr);

    /* Look for converged eigenpairs */
    while (nconv<ncv && eps->errest[nconv]<eps->tol) nconv++;

    EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    eps->its = eps->its + ncv - eps->nconv;
    eps->nconv = nconv;
    if (eps->nconv >= eps->nev) break;
  }
  
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(D);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(W);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(ifail);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LANCZOS"
PetscErrorCode EPSCreate_LANCZOS(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->solve                = EPSSolve_LANCZOS;
  eps->ops->setup                = EPSSetUp_LANCZOS;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

