/*

   SLEPc quadratic eigensolver: "qlanczos"

   Method: Q-Lanczos

   Algorithm:

       Quadratic Lanczos with indefinite B-orthogonalization and
       thick restart.

   References:

       [1] C. Campos and J.E. Roman, "A thick-restart Q-Lanczos method
           for quadratic eigenvalue problems", in preparation, 2013.

   Last update: Mar 2013

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#undef __FUNCT__
#define __FUNCT__ "QEPSetUp_QLanczos"
PetscErrorCode QEPSetUp_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  PetscBool      sinv;

  PetscFunctionBegin;
  if (qep->ncv) { /* ncv set */
    if (qep->ncv<qep->nev) SETERRQ(PetscObjectComm((PetscObject)qep),1,"The value of ncv must be at least nev");
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
  if (qep->ncv>qep->nev+qep->mpd) SETERRQ(PetscObjectComm((PetscObject)qep),1,"The value of ncv must not be larger than nev+mpd");
  if (!qep->max_it) qep->max_it = PetscMax(100,2*qep->n/qep->ncv); /* -- */
  if (!qep->which) {
    ierr = PetscObjectTypeCompare((PetscObject)qep->st,STSINVERT,&sinv);CHKERRQ(ierr);
    if (sinv) qep->which = QEP_TARGET_MAGNITUDE;
    else qep->which = QEP_LARGEST_MAGNITUDE;
  }
  if (qep->problem_type!=QEP_HERMITIAN) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  ierr = QEPAllocateSolution(qep,0);CHKERRQ(ierr);
  ierr = QEPSetWorkVecs(qep,4);CHKERRQ(ierr);

  ierr = DSSetType(qep->ds,DSGHIEP);CHKERRQ(ierr);
  ierr = DSSetCompact(qep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(qep->ds,qep->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QEPQLanczosNorm_private"
/*
  Compute B-norm of v=[v1;v2] whith  B=diag(-qep->T[0],qep->T[2]) 
*/
PetscErrorCode QEPQLanczosNorm_private(QEP qep,Vec v1,Vec v2,PetscReal *norm,Vec vw)
{
  PetscErrorCode ierr;
  PetscScalar    p1,p2;
  
  PetscFunctionBegin;
  ierr = STMatMult(qep->st,0,v1,vw);CHKERRQ(ierr);
  ierr = VecDot(vw,v1,&p1);CHKERRQ(ierr);
  ierr = STMatMult(qep->st,2,v2,vw);CHKERRQ(ierr);
  ierr = VecDot(vw,v2,&p2);CHKERRQ(ierr);
  *norm = PetscRealPart(-p1+p2);
  *norm = (*norm>0.0)?PetscSqrtReal(*norm):-PetscSqrtReal(-*norm);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QEPQLanczosCGS"
/*
  Compute a step of Classical Gram-Schmidt orthogonalization
*/
static PetscErrorCode QEPQLanczosCGS(QEP qep,PetscScalar *H,PetscBLASInt ldh,PetscReal *omega,PetscScalar *h,PetscBLASInt j,Vec *V,Vec t,Vec v,Vec w,PetscReal *onorm,PetscReal *norm,PetscScalar *work,Vec vw)
{
  PetscErrorCode ierr;
  PetscBLASInt   ione = 1,j_1 = j+1,i;
  PetscScalar    dot,one = 1.0,zero = 0.0;

  PetscFunctionBegin;
  /* compute norm of [v;w] */
  if (onorm) {
    ierr = QEPQLanczosNorm_private(qep,v,w,onorm,vw);CHKERRQ(ierr);
  }

  /* orthogonalize: compute h */
  ierr = STMatMult(qep->st,0,v,vw);CHKERRQ(ierr);
  ierr = VecMDot(vw,j_1,V,h);CHKERRQ(ierr);
  for (i=0;i<=j;i++) h[i] = -h[i];
  ierr = STMatMult(qep->st,2,w,vw);CHKERRQ(ierr);
  if (j>0) {
    ierr = VecMDot(vw,j_1,V,work);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&j_1,&j,&one,H,&ldh,work,&ione,&one,h,&ione));
  }
  ierr = VecDot(vw,t,&dot);CHKERRQ(ierr);
  h[j] += dot;
  /* apply inverse of signature */
  for (i=0;i<=j;i++) h[i] /= omega[i];
  /* orthogonalize: update v and w */
  ierr = SlepcVecMAXPBY(v,1.0,-1.0,j_1,h,V);CHKERRQ(ierr);
  if (j>0) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&j_1,&j,&one,H,&ldh,h,&ione,&zero,work,&ione));
    ierr = SlepcVecMAXPBY(w,1.0,-1.0,j_1,work,V);CHKERRQ(ierr);
  }
  ierr = VecAXPY(w,-h[j],t);CHKERRQ(ierr);
  /* H pseudosymmetric, signature is not revert*/

  /* compute norm of v and w */
  if (norm) {
    ierr = QEPQLanczosNorm_private(qep,v,w,norm,vw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPQLanczos"
/*
  Compute a run of Q-Lanczos iterations
*/
static PetscErrorCode QEPQLanczos(QEP qep,PetscScalar *H,PetscInt ldh,PetscReal *omega,Vec *V,PetscInt k,PetscInt *M,Vec v,Vec w,PetscBool *breakdown,PetscScalar *work,Vec *t_)
{
  PetscErrorCode     ierr;
  PetscInt           i,j,m = *M,nwu = 0,l;
  Vec                t = t_[0],u = t_[1];
  IPOrthogRefineType refinement;
  PetscReal          onorm,norm,eta;
  PetscScalar        *c;

  PetscFunctionBegin;
  ierr = IPGetOrthogonalization(qep->ip,NULL,&refinement,&eta);CHKERRQ(ierr);
  nwu += m+1;
  c = work+nwu;

  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = VecCopy(w,t);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,0,v,u);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,1,t,w);CHKERRQ(ierr);
    ierr = VecAXPY(u,1.0,w);CHKERRQ(ierr);
    ierr = STMatSolve(qep->st,2,u,w);CHKERRQ(ierr);
    ierr = VecScale(w,-1.0);CHKERRQ(ierr);
    ierr = VecCopy(t,v);CHKERRQ(ierr);

    /* orthogonalize */
    switch (refinement) {
      case IP_ORTHOG_REFINE_NEVER:
        ierr = QEPQLanczosCGS(qep,H,ldh,omega,H+ldh*j,j,V,t,v,w,NULL,&norm,work,u);CHKERRQ(ierr);
        break;
      case IP_ORTHOG_REFINE_ALWAYS:
        ierr = QEPQLanczosCGS(qep,H,ldh,omega,H+ldh*j,j,V,t,v,w,NULL,NULL,work,u);CHKERRQ(ierr);
        ierr = QEPQLanczosCGS(qep,H,ldh,omega,c,j,V,t,v,w,&onorm,&norm,work,u);CHKERRQ(ierr);
        for (i=0;i<=j;i++) H[ldh*j+i] += c[i];
        break;
      case IP_ORTHOG_REFINE_IFNEEDED: /* -- */
        ierr = QEPQLanczosCGS(qep,H,ldh,omega,H+ldh*j,j,V,t,v,w,&onorm,&norm,work,u);CHKERRQ(ierr);
        /* ||q|| < eta ||h|| */
        l = 1;
        while (l<3 && PetscAbsReal(norm) < eta * PetscAbsReal(onorm)) {
          l++;
          onorm = norm;
          ierr = QEPQLanczosCGS(qep,H,ldh,omega,c,j,V,t,v,w,NULL,&norm,work,u);CHKERRQ(ierr);
          for (i=0;i<=j;i++) H[ldh*j+i] += c[i];
        }
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)qep),1,"Wrong value of ip->orth_ref");
    }
    if (breakdown) *breakdown = PETSC_FALSE; /* -- */
    ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
    ierr = VecScale(w,1.0/norm);CHKERRQ(ierr);

    H[j+1+ldh*j] = norm;
    omega[j+1] = (norm<0.0)?-1.0:1.0;
    if (j<m-1) {
      ierr = VecCopy(v,V[j+1]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSolve_QLanczos"
PetscErrorCode QEPSolve_QLanczos(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       j,k,l,lwork,nv,ld,ti,t;
  Vec            v=qep->work[0],w=qep->work[1],w_=qep->work[2];
  PetscScalar    *S,*Q,*work;
  PetscReal      beta,norm,*omega,*a,*b,*r,t1,t2;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(qep->ds,&ld);CHKERRQ(ierr);
  lwork = 7*qep->ncv;  /* -- */
  ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  if (qep->nini==0) {
    ierr = SlepcVecSetRandom(qep->V[0],qep->rand);CHKERRQ(ierr);
  }
  /* w is always a random vector */
  ierr = SlepcVecSetRandom(w,qep->rand);CHKERRQ(ierr);
  ierr = QEPQLanczosNorm_private(qep,qep->V[0],w,&norm,w_);CHKERRQ(ierr);
  if (norm==0.0) SETERRQ(((PetscObject)qep)->comm,1,"Initial vector is zero or belongs to the deflation space"); 
  ierr = VecScale(qep->V[0],1.0/norm);CHKERRQ(ierr);
  ierr = VecScale(w,1.0/norm);CHKERRQ(ierr);
  ierr = VecCopy(qep->V[0],v);CHKERRQ(ierr);
  ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = (norm > 0)?1.0:-1.0;
  ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);

  /* Restart loop */
  l = 0;
  while (qep->reason == QEP_CONVERGED_ITERATING) {
    qep->its++;

    /* Form projected problem matrix on S */
    ierr = DSGetArray(qep->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = DSGetArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a+ld;
    r = b+ld;
    ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = PetscMemzero(S,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    ti = qep->nconv+l;
    for (j=0;j<ti;j++) {
      S[j+j*ld] = a[j]*omega[j];
      S[j+1+j*ld] = b[j]*omega[j+1];
      S[j+(j+1)*ld] = b[j]*omega[j];
    }
    for (j=qep->nconv;j<ti;j++) {
      S[ti+j*ld] = r[j]*omega[ti];
    }

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(qep->nconv+qep->mpd,qep->ncv);
    ierr = QEPQLanczos(qep,S,ld,omega,qep->V,qep->nconv+l,&nv,v,w,&breakdown,work,&qep->work[2]);CHKERRQ(ierr);
    for (j=ti;j<nv-1;j++) {
      a[j] = PetscRealPart(S[j+j*ld]*omega[j]);
      b[j] = PetscAbsReal(PetscRealPart(S[j+1+j*ld]));
    }
    a[nv-1] = PetscRealPart(S[nv-1+(nv-1)*ld]*omega[nv-1]);
    beta = PetscAbsReal(PetscRealPart(S[nv+(nv-1)*ld]));
    ierr = DSRestoreArray(qep->ds,DS_MAT_A,&S);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = DSSetDimensions(qep->ds,nv,0,qep->nconv,qep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(qep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(qep->ds,qep->eigr,qep->eigi);CHKERRQ(ierr);
    ierr = DSSort(qep->ds,qep->eigr,qep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

    /* Check convergence */
    ierr = VecNorm(v,NORM_2,&t1);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
    t1 = SlepcAbs(t1,norm);
    ierr = STMatMult(qep->st,0,v,w_);CHKERRQ(ierr);
    ierr = VecNorm(w_,NORM_2,&t2);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,2,w,w_);CHKERRQ(ierr);
    ierr = VecNorm(w_,NORM_2,&norm);CHKERRQ(ierr);
    t2 = SlepcAbs(t2,norm);
    norm = PetscMax(t1,t2);
    ierr = DSGetDimensions(qep->ds,NULL,NULL,NULL,NULL,&t);CHKERRQ(ierr);    
    ierr = QEPKrylovConvergence(qep,PETSC_FALSE,qep->nconv,t-qep->nconv,nv,beta*norm,&k);CHKERRQ(ierr);
    if (qep->its >= qep->max_it) qep->reason = QEP_DIVERGED_ITS;
    if (k >= qep->nev) qep->reason = QEP_CONVERGED_TOL;

    /* Update l */
    if (qep->reason != QEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      l = PetscMin(l,t);
      ierr = DSGetArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      if (*(a+ld+k+l-1)!=0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
      ierr = DSRestoreArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    }

    if (qep->reason == QEP_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Stop if breakdown */
        ierr = PetscInfo2(qep,"Breakdown Quadratic Lanczos method (it=%D norm=%G)\n",qep->its,beta);CHKERRQ(ierr);
        qep->reason = QEP_DIVERGED_BREAKDOWN;
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSGetArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        r = a + 2*ld;
        for (j=k;j<k+l;j++) {
          r[j] = PetscRealPart(Q[nv-1+j*ld]*beta);
        }
        b = a+ld;
        b[k+l-1] = r[k+l-1];
        omega[k+l] = omega[nv];
        ierr = DSRestoreArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,qep->V,qep->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    
    /* Append v to V*/
    if (qep->reason == QEP_CONVERGED_ITERATING) { 
      ierr = VecCopy(v,qep->V[k+l]);CHKERRQ(ierr);
    }
    qep->nconv = k;
    ierr = QEPMonitor(qep,qep->its,qep->nconv,qep->eigr,qep->eigi,qep->errest,nv);CHKERRQ(ierr);
  }

  for (j=0;j<qep->nconv;j++) {
    qep->eigr[j] *= qep->sfactor;
    qep->eigi[j] *= qep->sfactor;
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(qep->ds,qep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  /* Compute eigenvectors */
  if (qep->nconv > 0) {
    ierr = QEPComputeVectors_Indefinite(qep);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPCreate_QLanczos"
PETSC_EXTERN PetscErrorCode QEPCreate_QLanczos(QEP qep)
{
  PetscFunctionBegin;
  qep->ops->solve                = QEPSolve_QLanczos;
  qep->ops->setup                = QEPSetUp_QLanczos;
  qep->ops->reset                = QEPReset_Default;
  PetscFunctionReturn(0);
}

