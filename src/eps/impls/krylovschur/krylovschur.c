#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_KRYLOVSCHUR"
PetscErrorCode EPSSetUp_KRYLOVSCHUR(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv) {
    if (eps->ncv > N) eps->ncv = N;
    if (eps->ncv<eps->nev+1) SETERRQ(1,"The value of ncv must be at least nev+1"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    ierr = PetscFree(eps->Tl);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->Tl);CHKERRQ(ierr);
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  }
  else { ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KRYLOVSCHUR"
PetscErrorCode EPSSolve_KRYLOVSCHUR(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k,lwork,info,ilo,l=0;
  Vec            u=eps->work[0];
  PetscScalar    *S=eps->T,*Q,*tau,*work,*b;
  PetscReal      beta;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&tau);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&b);CHKERRQ(ierr);
  lwork = (eps->ncv+4)*eps->ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  
  eps->nconv = 0;
  eps->its = 0;
  EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {

    /* Compute an nv-step Arnoldi factorization */
    eps->nv = eps->ncv;
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,eps->V,eps->nconv+l,&eps->nv,u,&beta,&breakdown);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/beta);CHKERRQ(ierr);

    /* Reduce S to (quasi-)triangular form, S <- Q S Q' */
    if (l==0) {
      ierr = PetscMemzero(Q,eps->nv*eps->nv*sizeof(PetscScalar));CHKERRQ(ierr);
      for (i=0;i<eps->nv;i++) 
        Q[i*(eps->nv+1)] = 1.0;
    } else {
      ilo = eps->nconv+1;
      LAPACKgehrd_(&eps->nv,&ilo,&eps->nv,S,&eps->ncv,tau,work,&lwork,&info);
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
      ierr = PetscMemcpy(Q,S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
      LAPACKorghr_(&eps->nv,&ilo,&eps->nv,Q,&eps->ncv,tau,work,&lwork,&info);
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
    }
    ierr = EPSDenseSchur(eps->nv,eps->nconv,S,eps->ncv,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);
    
    /* Sort the remaining columns of the Schur form */
    ierr = EPSSortDenseSchur(eps->nv,eps->nconv,S,eps->ncv,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* Compute residual norm estimates */
    ierr = ArnoldiResiduals(S,eps->ncv,Q,beta,eps->nconv,eps->nv,eps->eigr,eps->eigi,eps->errest,work);CHKERRQ(ierr);

    /* Check convergence */
    k = eps->nconv;
    while (k<eps->nv && eps->errest[k]<eps->tol) k++;
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (eps->nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      if (eps->eigi[k+l-1] > 0) {
        if (k+l<eps->nv-1) l = l+1;
	else l = l-1;
      }
#endif
    }
    
    /* Lock converged eigenpairs and update the corresponding vectors,
       V(:,idx) = V*Q(:,idx) */
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecSet(eps->AV[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->AV[i],eps->nv,Q+eps->nv*i,eps->V);CHKERRQ(ierr);
    }
    for (i=eps->nconv;i<k+l;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);
    
    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
	/* start a new Arnoldi factorization */
	eps->count_breakdown++;
	PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%i norm=%g)\n",eps->its,beta);
	ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
	if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
	  PetscInfo(eps,"Unable to generate more start vectors\n");
	}
      } else {
        /* update the Arnoldi-Schur decomposition */
	for (i=k;i<k+l;i++) {
          S[i*eps->ncv+k+l] = Q[i*eps->nv+eps->nv-1]*beta;
	}
	ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
      }
    }    
  } 
  
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_KRYLOVSCHUR"
PetscErrorCode EPSCreate_KRYLOVSCHUR(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = PETSC_NULL;
  eps->ops->solve                = EPSSolve_KRYLOVSCHUR;
  eps->ops->solvets              = PETSC_NULL;
  eps->ops->setup                = EPSSetUp_KRYLOVSCHUR;
  eps->ops->setfromoptions       = PETSC_NULL;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = PETSC_NULL;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

