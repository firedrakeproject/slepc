
/*                       
       This implements the Arnoldi method with explicit restart and
       deflation.
*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARNOLDI"
PetscErrorCode EPSSetUp_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  int            N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+8);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->T) { ierr = PetscFree(eps->T);CHKERRQ(ierr); }  
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi"
static PetscErrorCode EPSBasicArnoldi(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            j;
  PetscReal      norm;
  PetscScalar    t;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,V,f,H+m*j,&norm,&breakdown);CHKERRQ(ierr);
    if (breakdown) SETERRQ(1,"Breakdown in Arnoldi method");
    H[(m+1)*j+1] = norm;
    t = 1 / norm;
    ierr = VecScale(&t,f);CHKERRQ(ierr);
    ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
  }
  ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
  ierr = EPSOrthogonalize(eps,j+1,V,f,H+m*j,beta,&breakdown);CHKERRQ(ierr);
  if (breakdown) SETERRQ(1,"Breakdown in Arnoldi method");
  t = 1 / *beta;
  ierr = VecScale(&t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
PetscErrorCode EPSSolve_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  int            i,mout,info,ncv=eps->ncv;
  Vec            f=eps->work[ncv];
  PetscScalar    *H=eps->T,*U,*work,t;
  PetscReal      norm,beta;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#endif 
  ierr = PetscMemzero(H,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = VecNorm(eps->V[0],NORM_2,&norm);CHKERRQ(ierr);
  t = 1 / norm;
  ierr = VecScale(&t,eps->V[0]);CHKERRQ(ierr);
  
  eps->nconv = 0;
  eps->its = 0;
  while (eps->its<eps->max_it) {
    eps->its = eps->its + 1;
  /* [H,V,f,beta] = karnoldi(es,H,V,nconv+1,m) % Arnoldi factorization */
    ierr = EPSBasicArnoldi(eps,H,eps->V,eps->nconv,ncv,f,&beta);CHKERRQ(ierr);
  /* U = eye(m,m) */
    ierr = PetscMemzero(U,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<ncv;i++) { U[i*(ncv+1)] = 1.0; }
  /* [T,wr0,wi0,U] = laqr3(H,U,nconv+1,ncv) */
    ierr = EPSDenseSchur(H,U,eps->eigr,eps->eigi,eps->nconv,ncv);CHKERRQ(ierr);
  /* V(:,idx) = V*U(:,idx) */
    ierr = EPSReverseProjection(eps,eps->V,U,eps->nconv,ncv,eps->work);CHKERRQ(ierr);
  /* [Y,dummy] = eig(H) */
#if !defined(PETSC_USE_COMPLEX)
    LAtrevc_("R","B",PETSC_NULL,&ncv,H,&ncv,PETSC_NULL,&ncv,U,&ncv,&ncv,&mout,work,&info,1,1);
#else
    LAtrevc_("R","B",PETSC_NULL,&ncv,H,&ncv,PETSC_NULL,&ncv,U,&ncv,&ncv,&mout,work,rwork,&info,1,1);
#endif
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  /* rsd = beta*abs(Y(m,:)) */
    for (i=eps->nconv;i<ncv;i++) { 
      eps->errest[i] = beta*PetscAbsScalar(U[i*ncv+ncv-1]); 
      if (eps->errest[i] < eps->tol) eps->nconv = i + 1;
    }
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    if (eps->nconv >= eps->nev) break;
  }
  
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
PetscErrorCode EPSCreate_ARNOLDI(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

