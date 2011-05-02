/*                       

   SLEPc eigensolver: "dsitrlanczos"

   Method: Thick restart Lanczos with full reorthogonalization and dynamic shift and invert

   Last update: Jan 2010

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

PetscErrorCode EPSSolve_DSITRLanczos(EPS);

extern PetscErrorCode EPSProjectedKSSym(EPS eps,PetscInt n,PetscInt l,PetscReal *a,PetscReal *b,PetscScalar *eig,PetscScalar *Q,PetscReal *work,PetscInt *perm);

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_DSITRLanczos"
PetscErrorCode EPSSetUp_DSITRLanczos(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      isSinv;

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
  if (eps->ncv>eps->nev+eps->mpd) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);

  if (!eps->ishermitian)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;
  if (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY)
    SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } if (eps->extraction != EPS_RITZ) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
  }

  ierr = PetscTypeCompare((PetscObject)eps->OP,STSINVERT,&isSinv);CHKERRQ(ierr);
  if (!isSinv) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Shift-and-invert ST is needed");
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_DSITRLanczos;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_DSITRLanczos"
PetscErrorCode EPSSolve_DSITRLanczos(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lds,lt,nv,m,*iwork;
  Vec            u=eps->work[0];
  PetscScalar    *Q, sigma, lambda, zero = 0.0;
  PetscReal      *a,*b,*work,beta,distance = 1e-3;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-eps_distance",&distance,PETSC_NULL);CHKERRQ(ierr);
  lds = PetscMin(eps->mpd,eps->ncv);
  ierr = PetscMalloc(lds*lds*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(2*lds*sizeof(PetscInt),&iwork);CHKERRQ(ierr);
  lt = PetscMin(eps->nev+eps->mpd,eps->ncv);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&a);CHKERRQ(ierr);  
  ierr = PetscMalloc(lt*sizeof(PetscReal),&b);CHKERRQ(ierr);  

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    m = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = EPSFullLanczos(eps,a+l,b+l,eps->V,eps->nconv+l,&m,u,&breakdown);CHKERRQ(ierr);
    nv = m - eps->nconv;
    beta = b[nv-1];

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = EPSProjectedKSSym(eps,nv,l,a,b,eps->eigr+eps->nconv,Q,work,iwork);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,eps->nconv,nv,PETSC_NULL,nv,Q,eps->V+eps->nconv,nv,beta,1.0,&k,PETSC_NULL);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Transform converged eigenvalues to the original problem */
    ierr = STBackTransform(eps->OP,k-eps->nconv,eps->eigr+eps->nconv,eps->eigi+eps->nconv);CHKERRQ(ierr);

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (eps->nconv+nv-k)/2;
      /* Update shift */
      ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);
      lambda = eps->eigr[k+1];
      ierr = STBackTransform(eps->OP,1,&lambda,&zero);CHKERRQ(ierr);
      if (PetscAbsScalar(lambda - sigma)/PetscAbsScalar(sigma) > distance) {
        ierr = PetscInfo2(eps,"Shift update its=%i sigma=%g\n",eps->its,lambda);
        PetscPushErrorHandler(PetscReturnErrorHandler,PETSC_NULL);
        ierr = STSetShift(eps->OP,lambda);
        PetscPopErrorHandler();
        switch (ierr) {
        case PETSC_ERR_MAT_LU_ZRPVT:
        case PETSC_ERR_MAT_CH_ZRPVT:
          ierr = PetscInfo2(eps,"Factorization error in shift update its=%i sigma=%g\n",eps->its,lambda);
          ierr = STSetShift(eps->OP,sigma);CHKERRQ(ierr);
          break;
        default:
          CHKERRQ(ierr);
          l = 0; /* do not use restart vectors */
        }
      }
    } 

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        PetscInfo2(eps,"Breakdown in TR Lanczos method (it=%i norm=%g)\n",eps->its,beta);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          PetscInfo(eps,"Unable to generate more start vectors\n");
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=0;i<l;i++) {
          a[i] = PetscRealPart(eps->eigr[i+k]);
          b[i] = PetscRealPart(Q[nv-1+(i+k-eps->nconv)*nv]*beta);
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv+eps->nconv);CHKERRQ(ierr);
    eps->nconv = k;
  } 
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_DSITRLanczos"
PetscErrorCode EPSCreate_DSITRLanczos(EPS eps)
{
  PetscFunctionBegin;
  eps->data                = PETSC_NULL;
  eps->ops->setup          = EPSSetUp_DSITRLanczos;
  eps->ops->setfromoptions = PETSC_NULL;
  eps->ops->destroy        = EPSDestroy_Default;
  eps->ops->view           = PETSC_NULL;
  eps->ops->computevectors = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

