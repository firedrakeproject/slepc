/*                       

   Q-Arnoldi method for quadratic eigenproblems.

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

#include <private/qepimpl.h>         /*I "slepcqep.h" I*/
#include <petscblaslapack.h>

typedef struct {
  KSP ksp;
} QEP_QARNOLDI;

#undef __FUNCT__  
#define __FUNCT__ "QEPSetUp_QARNOLDI"
PetscErrorCode QEPSetUp_QARNOLDI(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QARNOLDI   *ctx = (QEP_QARNOLDI *)qep->data;
  
  PetscFunctionBegin;

  if (qep->ncv) { /* ncv set */
    if (qep->ncv<qep->nev) SETERRQ(((PetscObject)qep)->comm,1,"The value of ncv must be at least nev"); 
  }
  else if (qep->mpd) { /* mpd set */
    qep->ncv = PetscMin(qep->n,qep->nev+qep->mpd);
  }
  else { /* neither set: defaults depend on nev being small or large */
    if (qep->nev<500) qep->ncv = PetscMin(qep->n,PetscMax(2*qep->nev,qep->nev+15));
    else { qep->mpd = 500; qep->ncv = PetscMin(qep->n,qep->nev+qep->mpd); }
  }
  if (!qep->mpd) qep->mpd = qep->ncv;
  if (qep->ncv>qep->nev+qep->mpd) SETERRQ(((PetscObject)qep)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!qep->max_it) qep->max_it = PetscMax(100,2*qep->n/qep->ncv);
  if (!qep->which) qep->which = QEP_LARGEST_MAGNITUDE;
  if (qep->problem_type != QEP_GENERAL)
    SETERRQ(((PetscObject)qep)->comm,1,"Wrong value of qep->problem_type");

  ierr = PetscFree(qep->T);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*qep->ncv*sizeof(PetscScalar),&qep->T);CHKERRQ(ierr);
  ierr = QEPDefaultGetWork(qep,4);CHKERRQ(ierr);

  ierr = KSPSetOperators(ctx->ksp,qep->M,qep->M,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(ctx->ksp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPQArnoldiCGS"
/*
  Compute a step of Classical Gram-Schmidt orthogonalization 
*/
PetscErrorCode QEPQArnoldiCGS(QEP qep,PetscScalar *H,PetscBLASInt ldh,PetscScalar *h,PetscBLASInt j,Vec *V,Vec t,Vec v,Vec w,PetscReal *onorm,PetscReal *norm,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscBLASInt   ione = 1, j_1 = j+1;
  PetscReal      x, y;
  PetscScalar    dot, one = 1.0, zero = 0.0;

  PetscFunctionBegin;
  /* compute norm of v and w */
  if (onorm) {
    ierr = VecNorm(v,NORM_2,&x);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&y);CHKERRQ(ierr);
    *onorm = sqrt(x*x+y*y);
  }

  /* orthogonalize: compute h */
  ierr = VecMDot(v,j_1,V,h);CHKERRQ(ierr);
  ierr = VecMDot(w,j_1,V,work);CHKERRQ(ierr);
  if (j>0)
    BLASgemv_("C",&j_1,&j,&one,H,&ldh,work,&ione,&one,h,&ione);
  ierr = VecDot(t,w,&dot);CHKERRQ(ierr);
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
    *norm = sqrt(x*x+y*y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPQArnoldi"
/*
  Compute a run of Q-Arnoldi iterations
*/
PetscErrorCode QEPQArnoldi(QEP qep,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec v,Vec w,PetscReal *beta,PetscBool *breakdown,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscInt       i,j,l,m = *M;
  QEP_QARNOLDI   *ctx = (QEP_QARNOLDI *)qep->data;
  Vec            t = qep->work[2], u = qep->work[3];
  IPOrthogonalizationRefinementType refinement;
  PetscReal      norm,onorm,eta;
  PetscScalar    *c = work + m;

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
      case IP_ORTH_REFINE_NEVER:
        ierr = QEPQArnoldiCGS(qep,H,ldh,H+ldh*j,j,V,t,v,w,PETSC_NULL,&norm,work);CHKERRQ(ierr);
        *breakdown = PETSC_FALSE;
        break;
      case IP_ORTH_REFINE_ALWAYS:
        ierr = QEPQArnoldiCGS(qep,H,ldh,H+ldh*j,j,V,t,v,w,PETSC_NULL,PETSC_NULL,work);CHKERRQ(ierr);
        ierr = QEPQArnoldiCGS(qep,H,ldh,c,j,V,t,v,w,&onorm,&norm,work);CHKERRQ(ierr);
        for (i=0;i<j;i++) H[ldh*j+i] += c[i];
        if (norm < eta * onorm) *breakdown = PETSC_TRUE;
        else *breakdown = PETSC_FALSE;
        break;
      case IP_ORTH_REFINE_IFNEEDED:
        ierr = QEPQArnoldiCGS(qep,H,ldh,H+ldh*j,j,V,t,v,w,&onorm,&norm,work);CHKERRQ(ierr);
        /* ||q|| < eta ||h|| */
        l = 1;
        while (l<3 && norm < eta * onorm) {
          l++;
          onorm = norm;
          ierr = QEPQArnoldiCGS(qep,H,ldh,c,j,V,t,v,w,PETSC_NULL,&norm,work);CHKERRQ(ierr);
          for (i=0;i<j;i++) H[ldh*j+i] += c[i];
        }
        if (norm < eta * onorm) *breakdown = PETSC_TRUE;
        else *breakdown = PETSC_FALSE;
        break;
      default: SETERRQ(((PetscObject)qep)->comm,1,"Wrong value of ip->orth_ref");
    }
    ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
    ierr = VecScale(w,1.0/norm);CHKERRQ(ierr);
    
    if (j<m-1) {
      H[j+1+ldh*j] = norm;
      ierr = VecCopy(v,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "QEPProjectedKSNonsym"
/*
   QEPProjectedKSNonsym - Solves the projected eigenproblem in the Krylov-Schur
   method (non-symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (leading dimension is lds)

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the corresponding Schur vectors (order n, leading dimension n)
*/
PetscErrorCode QEPProjectedKSNonsym(QEP qep,PetscInt l,PetscScalar *S,PetscInt lds,PetscScalar *Q,PetscInt n)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (l==0) {
    ierr = PetscMemzero(Q,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) 
      Q[i*(n+1)] = 1.0;
  } else {
    /* Reduce S to Hessenberg form, S <- Q S Q' */
    ierr = EPSDenseHessenberg(n,qep->nconv,S,lds,Q);CHKERRQ(ierr);
  }
  /* Reduce S to (quasi-)triangular form, S <- Q S Q' */
  ierr = EPSDenseSchur(n,qep->nconv,S,lds,Q,qep->eigr,qep->eigi);CHKERRQ(ierr);
  /* Sort the remaining columns of the Schur form */
  ierr = QEPSortDenseSchur(qep,n,qep->nconv,S,lds,Q,qep->eigr,qep->eigi);CHKERRQ(ierr);    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSolve_QARNOLDI"
PetscErrorCode QEPSolve_QARNOLDI(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,l,lwork,nv;
  Vec            v=qep->work[0],w=qep->work[1];
  PetscScalar    *S=qep->T,*Q,*work;
  PetscReal      beta,norm,x,y;
  PetscBool      breakdown;

  PetscFunctionBegin;

  ierr = PetscMemzero(S,qep->ncv*qep->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*qep->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
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
  norm = sqrt(x*x+y*y);CHKERRQ(ierr);
  ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
  ierr = VecScale(w,1.0/norm);CHKERRQ(ierr);
  
  /* Restart loop */
  l = 0;
  while (qep->reason == QEP_CONVERGED_ITERATING) {
    qep->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(qep->nconv+qep->mpd,qep->ncv);
    ierr = QEPQArnoldi(qep,S,qep->ncv,qep->V,qep->nconv+l,&nv,v,w,&beta,&breakdown,work);CHKERRQ(ierr);

    /* Solve projected problem */ 
    ierr = QEPProjectedKSNonsym(qep,l,S,qep->ncv,Q,nv);CHKERRQ(ierr);

    /* Check convergence */ 
    ierr = QEPKrylovConvergence(qep,qep->nconv,nv-qep->nconv,S,qep->ncv,Q,nv,beta,&k,work);CHKERRQ(ierr);
    if (qep->its >= qep->max_it) qep->reason = QEP_DIVERGED_ITS;
    if (k >= qep->nev) qep->reason = QEP_CONVERGED_TOL;
    
    /* Update l */
    if (qep->reason != QEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      if (S[(k+l-1)*(qep->ncv+1)+1] != 0.0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
#endif
    }

    if (qep->reason == QEP_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Stop if breakdown */
        PetscInfo2(qep,"Breakdown Quadratic Arnoldi method (it=%i norm=%g)\n",qep->its,beta);
        qep->reason = QEP_DIVERGED_BREAKDOWN;
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=k;i<k+l;i++) {
          S[i*qep->ncv+k+l] = Q[(i+1)*nv-1]*beta;
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,qep->V,qep->nconv,k+l,Q,nv,PETSC_FALSE);CHKERRQ(ierr);

    qep->nconv = k;

    ierr = QEPMonitor(qep,qep->its,qep->nconv,qep->eigr,qep->eigi,qep->errest,nv);CHKERRQ(ierr);
    
  } 

  for (j=0;j<qep->nconv;j++) {
    qep->eigr[j] *= qep->sfactor;
    qep->eigi[j] *= qep->sfactor;
  }

  /* Compute eigenvectors */
  if (qep->nconv > 0) {
    ierr = QEPComputeVectors_Schur(qep);
  }

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetFromOptions_QARNOLDI"
PetscErrorCode QEPSetFromOptions_QARNOLDI(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QARNOLDI   *ctx = (QEP_QARNOLDI *)qep->data;
 
  PetscFunctionBegin;
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPView_QARNOLDI"
PetscErrorCode QEPView_QARNOLDI(QEP qep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  QEP_QARNOLDI   *ctx = (QEP_QARNOLDI *)qep->data;

  PetscFunctionBegin;
  ierr = KSPView(ctx->ksp,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDestroy_QARNOLDI"
PetscErrorCode QEPDestroy_QARNOLDI(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QARNOLDI   *ctx = (QEP_QARNOLDI *)qep->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = QEPDestroy_Default(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPCreate_QARNOLDI"
PetscErrorCode QEPCreate_QARNOLDI(QEP qep)
{
  PetscErrorCode ierr;
  QEP_QARNOLDI   *ctx;

  PetscFunctionBegin;
  ierr = PetscNew(QEP_QARNOLDI,&ctx);CHKERRQ(ierr);
  PetscLogObjectMemory(qep,sizeof(QEP_QARNOLDI));
  qep->data                      = ctx;
  qep->ops->solve                = QEPSolve_QARNOLDI;
  qep->ops->setup                = QEPSetUp_QARNOLDI;
  qep->ops->setfromoptions       = QEPSetFromOptions_QARNOLDI;
  qep->ops->destroy              = QEPDestroy_QARNOLDI;
  qep->ops->view                 = QEPView_QARNOLDI;

  ierr = KSPCreate(((PetscObject)qep)->comm,&ctx->ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ctx->ksp,((PetscObject)qep)->prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(ctx->ksp,"qep_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)qep,1);CHKERRQ(ierr);  
  PetscLogObjectParent(qep,ctx->ksp);
  PetscFunctionReturn(0);
}
EXTERN_C_END

