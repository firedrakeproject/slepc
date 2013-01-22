/*                       

   Q-Lanczos method for quadratic eigenproblems.

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

#include <slepc-private/qepimpl.h>         /*I "slepcqep.h" I*/
#include <petscblaslapack.h>

typedef struct {
  KSP ksp;
} QEP_QLANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "QEPSetUp_QLanczos"
PetscErrorCode QEPSetUp_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QLANCZOS   *ctx = (QEP_QLANCZOS*)qep->data;
  
  PetscFunctionBegin;
  if (qep->ncv) { /* ncv set */
    if (qep->ncv<qep->nev) SETERRQ(((PetscObject)qep)->comm,1,"The value of ncv must be at least nev"); 
  } else if (qep->mpd) { /* mpd set */
    qep->ncv = PetscMin(qep->n,qep->nev+qep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (qep->nev<500) qep->ncv = PetscMin(qep->n,PetscMax(2*qep->nev,qep->nev+15));
    else {
      qep->mpd = 500;
      qep->ncv = PetscMin(qep->n,qep->nev+qep->mpd);
    }
  }
  if (!qep->mpd) qep->mpd = qep->ncv;
  if (qep->ncv>qep->nev+qep->mpd) SETERRQ(((PetscObject)qep)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!qep->max_it) qep->max_it = PetscMax(100,2*qep->n/qep->ncv);
  if (!qep->which) qep->which = QEP_LARGEST_MAGNITUDE;
  if (qep->problem_type!=QEP_HERMITIAN) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  ierr = QEPAllocateSolution(qep);CHKERRQ(ierr);
  ierr = QEPDefaultGetWork(qep,4);CHKERRQ(ierr);

  ierr = DSSetType(qep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSSetExtraRow(qep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(qep->ds,qep->ncv+1);CHKERRQ(ierr);

  ierr = KSPSetOperators(ctx->ksp,qep->M,qep->M,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPQLanczosCGS"
/*
  Compute a step of Classical Gram-Schmidt orthogonalization 
*/
PetscErrorCode QEPQLanczosCGS(QEP qep,PetscScalar *H,PetscBLASInt ldh,PetscScalar *h,PetscBLASInt j,Vec *V,Vec t,Vec v,Vec w,PetscReal *onorm,PetscReal *norm,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscBLASInt   ione = 1,j_1 = j+1;
  PetscReal      x,y;
  PetscScalar    dot,one = 1.0,zero = 0.0;

  PetscFunctionBegin;
  /* compute norm of v and w */
  if (onorm) {
    ierr = VecNorm(v,NORM_2,&x);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&y);CHKERRQ(ierr);
    *onorm = PetscSqrtReal(x*x+y*y);
  }

  /* orthogonalize: compute h */
  ierr = VecMDot(v,j_1,V,h);CHKERRQ(ierr);
  ierr = VecMDot(w,j_1,V,work);CHKERRQ(ierr);
  if (j>0)
    BLASgemv_("C",&j_1,&j,&one,H,&ldh,work,&ione,&one,h,&ione);
  ierr = VecDot(w,t,&dot);CHKERRQ(ierr);
  h[j] += dot;

  /* orthogonalize: update v and w */
  ierr = SlepcVecMAXPBY(v,1.0,-1.0,j_1,h,V);CHKERRQ(ierr);
  if (j>0) {
    BLASgemv_("N",&j_1,&j,&one,H,&ldh,h,&ione,&zero,work,&ione);
    ierr = SlepcVecMAXPBY(w,1.0,-1.0,j_1,work,V);CHKERRQ(ierr);
  }
  ierr = VecAXPY(w,-h[j],t);CHKERRQ(ierr);
    
  /* compute norm of v and w */
  if (norm) {
    ierr = VecNorm(v,NORM_2,&x);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&y);CHKERRQ(ierr);
    *norm = PetscSqrtReal(x*x+y*y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPQLanczos"
/*
  Compute a run of Q-Arnoldi iterations
*/
PetscErrorCode QEPQLanczos(QEP qep,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec v,Vec w,PetscReal *beta,PetscBool *breakdown,PetscScalar *work)
{
  PetscErrorCode     ierr;
  PetscInt           i,j,l,m = *M;
  QEP_QLANCZOS       *ctx = (QEP_QLANCZOS*)qep->data;
  Vec                t = qep->work[2],u = qep->work[3];
  IPOrthogRefineType refinement;
  PetscReal          norm,onorm,eta;
  PetscScalar        *c = work + m;

  PetscFunctionBegin;
  ierr = IPGetOrthogonalization(qep->ip,PETSC_NULL,&refinement,&eta);CHKERRQ(ierr);
  ierr = VecCopy(v,qep->V[k]);CHKERRQ(ierr);
  
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = VecCopy(w,t);CHKERRQ(ierr);
    ierr = MatMult(qep->K,v,u);CHKERRQ(ierr);
    ierr = MatMult(qep->C,t,w);CHKERRQ(ierr);
    ierr = VecAXPY(u,qep->sfactor,w);CHKERRQ(ierr);
    ierr = KSPSolve(ctx->ksp,u,w);CHKERRQ(ierr);
    ierr = VecScale(w,-1.0/(qep->sfactor*qep->sfactor));CHKERRQ(ierr);
    ierr = VecCopy(t,v);CHKERRQ(ierr);

    /* orthogonalize */
    switch (refinement) {
      case IP_ORTHOG_REFINE_NEVER:
        ierr = QEPQLanczosCGS(qep,H,ldh,H+ldh*j,j,V,t,v,w,PETSC_NULL,&norm,work);CHKERRQ(ierr);
        *breakdown = PETSC_FALSE;
        break;
      case IP_ORTHOG_REFINE_ALWAYS:
        ierr = QEPQLanczosCGS(qep,H,ldh,H+ldh*j,j,V,t,v,w,PETSC_NULL,PETSC_NULL,work);CHKERRQ(ierr);
        ierr = QEPQLanczosCGS(qep,H,ldh,c,j,V,t,v,w,&onorm,&norm,work);CHKERRQ(ierr);
        for (i=0;i<=j;i++) H[ldh*j+i] += c[i];
        if (norm < eta * onorm) *breakdown = PETSC_TRUE;
        else *breakdown = PETSC_FALSE;
        break;
      case IP_ORTHOG_REFINE_IFNEEDED:
        ierr = QEPQLanczosCGS(qep,H,ldh,H+ldh*j,j,V,t,v,w,&onorm,&norm,work);CHKERRQ(ierr);
        /* ||q|| < eta ||h|| */
        l = 1;
        while (l<3 && norm < eta * onorm) {
          l++;
          onorm = norm;
          ierr = QEPQLanczosCGS(qep,H,ldh,c,j,V,t,v,w,PETSC_NULL,&norm,work);CHKERRQ(ierr);
          for (i=0;i<=j;i++) H[ldh*j+i] += c[i];
        }
        if (norm < eta * onorm) *breakdown = PETSC_TRUE;
        else *breakdown = PETSC_FALSE;
        break;
      default: SETERRQ(((PetscObject)qep)->comm,1,"Wrong value of ip->orth_ref");
    }
    ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
    ierr = VecScale(w,1.0/norm);CHKERRQ(ierr);
    
    H[j+1+ldh*j] = norm;
    if (j<m-1) {
      ierr = VecCopy(v,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSolve_QLanczos"
PetscErrorCode QEPSolve_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       j,k,l,lwork,nv,ld,newn;
  Vec            v=qep->work[0],w=qep->work[1];
  PetscScalar    *S,*Q,*work;
  PetscReal      beta,norm,x,y;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(qep->ds,&ld);CHKERRQ(ierr);
  lwork = 7*qep->ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  if (qep->nini>0) {
    ierr = VecCopy(qep->V[0],v);CHKERRQ(ierr);
  } else {
    ierr = SlepcVecSetRandom(v,qep->rand);CHKERRQ(ierr);
  }
  /* w is always a random vector */
  ierr = SlepcVecSetRandom(w,qep->rand);CHKERRQ(ierr);
  ierr = VecNorm(v,NORM_2,&x);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&y);CHKERRQ(ierr);
  norm = PetscSqrtReal(x*x+y*y);CHKERRQ(ierr);
  ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
  ierr = VecScale(w,1.0/norm);CHKERRQ(ierr);
  
  /* Restart loop */
  l = 0;
  while (qep->reason == QEP_CONVERGED_ITERATING) {
    qep->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(qep->nconv+qep->mpd,qep->ncv);
    ierr = DSGetArray(qep->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = QEPQLanczos(qep,S,ld,qep->V,qep->nconv+l,&nv,v,w,&beta,&breakdown,work);CHKERRQ(ierr);
    ierr = DSRestoreArray(qep->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = DSSetDimensions(qep->ds,nv,PETSC_IGNORE,qep->nconv,qep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(qep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */ 
    ierr = DSSolve(qep->ds,qep->eigr,qep->eigi);CHKERRQ(ierr);
    ierr = DSSort(qep->ds,qep->eigr,qep->eigi,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(qep->ds);CHKERRQ(ierr);

    /* Check convergence */ 
    ierr = QEPKrylovConvergence(qep,PETSC_FALSE,qep->nconv,nv-qep->nconv,nv,beta,&k);CHKERRQ(ierr);
    if (qep->its >= qep->max_it) qep->reason = QEP_DIVERGED_ITS;
    if (k >= qep->nev) qep->reason = QEP_CONVERGED_TOL;
    
    /* Update l */
    if (qep->reason != QEP_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (nv-k)/2;

    if (qep->reason == QEP_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Stop if breakdown */
        ierr = PetscInfo2(qep,"Breakdown Quadratic Arnoldi method (it=%D norm=%G)\n",qep->its,beta);CHKERRQ(ierr);
        qep->reason = QEP_DIVERGED_BREAKDOWN;
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSTruncate(qep->ds,k+l);CHKERRQ(ierr);
        ierr = DSGetDimensions(qep->ds,&newn,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        l = newn-k;
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,qep->V,qep->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    qep->nconv = k;
    ierr = QEPMonitor(qep,qep->its,qep->nconv,qep->eigr,qep->eigi,qep->errest,nv);CHKERRQ(ierr);
  } 

  for (j=0;j<qep->nconv;j++) {
    qep->eigr[j] *= qep->sfactor;
    qep->eigi[j] *= qep->sfactor;
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(qep->ds,qep->nconv,PETSC_IGNORE,0,0);CHKERRQ(ierr);
  ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  /* Compute eigenvectors */
  if (qep->nconv > 0) {
    ierr = QEPComputeVectors_Schur(qep);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetFromOptions_QLanczos"
PetscErrorCode QEPSetFromOptions_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QLANCZOS   *ctx = (QEP_QLANCZOS*)qep->data;
 
  PetscFunctionBegin;
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPView_QLanczos"
PetscErrorCode QEPView_QLanczos(QEP qep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  QEP_QLANCZOS   *ctx = (QEP_QLANCZOS*)qep->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = KSPView(ctx->ksp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPReset_QLanczos"
PetscErrorCode QEPReset_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QLANCZOS   *ctx = (QEP_QLANCZOS*)qep->data;

  PetscFunctionBegin;
  ierr = KSPReset(ctx->ksp);CHKERRQ(ierr);
  ierr = QEPDefaultFreeWork(qep);CHKERRQ(ierr);
  ierr = QEPFreeSolution(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDestroy_QLanczos"
PetscErrorCode QEPDestroy_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QLANCZOS   *ctx = (QEP_QLANCZOS*)qep->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = PetscFree(qep->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPCreate_QLanczos"
PetscErrorCode QEPCreate_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QLANCZOS   *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(qep,QEP_QLANCZOS,&ctx);CHKERRQ(ierr);
  qep->data                      = ctx;
  qep->ops->solve                = QEPSolve_QLanczos;
  qep->ops->setup                = QEPSetUp_QLanczos;
  qep->ops->setfromoptions       = QEPSetFromOptions_QLanczos;
  qep->ops->destroy              = QEPDestroy_QLanczos;
  qep->ops->reset                = QEPReset_QLanczos;
  qep->ops->view                 = QEPView_QLanczos;
  ierr = KSPCreate(((PetscObject)qep)->comm,&ctx->ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ctx->ksp,((PetscObject)qep)->prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(ctx->ksp,"qep_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)qep,1);CHKERRQ(ierr);  
  ierr = PetscLogObjectParent(qep,ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

