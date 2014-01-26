/*

   SLEPc quadratic eigensolver: "toar"

   Method: TOAR

   Algorithm:

       Two-Level Orthogonal Arnoldi.

   References:

       [1] D. Lu and Y. Su, "Two-level orthogonal Arnoldi process
           for the solution of quadratic eigenvalue problems".

   Last update: Oct 2013

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
#include <slepcblaslapack.h>


#undef __FUNCT__
#define __FUNCT__ "QEPSetUp_TOAR"
PetscErrorCode QEPSetUp_TOAR(QEP qep)
{
  PetscErrorCode ierr;
  PetscBool      sinv;
  ST             st;

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
  if (!qep->max_it) qep->max_it = PetscMax(100,2*qep->n/qep->ncv); 
  if (!qep->which) {
    ierr = PetscObjectTypeCompare((PetscObject)qep->st,STSINVERT,&sinv);CHKERRQ(ierr);
    if (sinv) qep->which = QEP_TARGET_MAGNITUDE;
    else qep->which = QEP_LARGEST_MAGNITUDE;
  }
  ierr = QEPAllocateSolution(qep,2);CHKERRQ(ierr);
  ierr = QEPSetWorkVecs(qep,4);CHKERRQ(ierr);
  ierr = DSSetType(qep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSSetExtraRow(qep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(qep->ds,qep->ncv+1);CHKERRQ(ierr);
  ierr = QEPGetST(qep,&st);CHKERRQ(ierr);
  ierr = STSetUp(st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPTOARSNorm2"
/*
  Norm of [sp;sq] 
*/
static PetscErrorCode QEPTOARSNorm2(PetscInt n,PetscScalar *sp,PetscScalar *sq,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,one=1;
  PetscReal      x,y;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  x = BLASnrm2_(&n_,sp,&one);
  y = BLASnrm2_(&n_,sq,&one);
  *norm = SlepcAbs(x,y);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QEPTOAROrth2"
/*
 Computes GS orthogonalization   [z;x] - [Sp;Sq]*y,
 where y = ([Sp;Sq]'*[z;x]).
   k: Column from S to be orthogonalized against previous columns.
   Sq = Sp+ld
*/
static PetscErrorCode QEPTOAROrth2(PetscScalar *S,PetscInt ld,PetscInt k,PetscScalar *y,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,lds_,k_,one=1;
  PetscScalar    sonem=-1.0,sone=1.0,szero=0.0,*xp,*xq,*c;
  PetscInt       lwa,nwu=0,i,lds=2*ld,n;
  
  PetscFunctionBegin;
  n = k+1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr); /* Size of qK and qM */
  ierr = PetscBLASIntCast(2*ld,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  lwa = k;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
  c = work+nwu;
  nwu += k;
  xp = S+k*lds;
  xq = S+ld+k*lds;
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,xp,&one,&szero,y,&one));
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ld,&lds_,xq,&one,&sone,y,&one));
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,y,&one,&sone,xp,&one));
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ld,&lds_,y,&one,&sone,xq,&one));
  /* twice */
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,xp,&one,&szero,c,&one));
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ld,&lds_,xq,&one,&sone,c,&one));
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,c,&one,&sone,xp,&one));
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ld,&lds_,c,&one,&sone,xq,&one));
  for (i=0;i<k;i++) y[i] += c[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPTOARrun"
/*
  Compute a run of Arnoldi iterations
*/
static PetscErrorCode QEPTOARrun(QEP qep,PetscScalar *S,PetscInt ld,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscScalar *work,PetscInt nw,Vec *t_,PetscInt nwv)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M,nwu=0,lwa;
  PetscInt       lds=ld*2;
  Vec            v=t_[0],t=t_[1],q=t_[2];
  PetscReal      norm;

  PetscFunctionBegin;
  if (!t_||nwv<3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",12);
  lwa = ld*4;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = VecZeroEntries(v);CHKERRQ(ierr);
    ierr = VecMAXPY(v,j+2,S+j*lds,V);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,0,v,t);CHKERRQ(ierr);
    ierr = VecZeroEntries(v);CHKERRQ(ierr);
    ierr = VecMAXPY(v,j+2,S+ld+j*lds,V);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,1,v,q);CHKERRQ(ierr);
    ierr = VecAXPY(t,1.0,q);CHKERRQ(ierr);
    ierr = STMatSolve(qep->st,2,t,q);CHKERRQ(ierr);
    ierr = VecScale(q,-1.0);CHKERRQ(ierr);
    
    /* orthogonalize */
    ierr = IPOrthogonalize(qep->ip,0,NULL,j+2,NULL,qep->V,q,S+ld+(j+1)*lds,&norm,breakdown);CHKERRQ(ierr);
    *(S+ld+(j+1)*lds+j+2) = norm;
    ierr = VecScale(q,1.0/norm);CHKERRQ(ierr);
    ierr = VecCopy(q,V[j+2]);CHKERRQ(ierr);
    for (i=0;i<=j+1;i++) *(S+(j+1)*lds+i) = *(S+ld+j*lds+i);

    /* Level-2 orthogonalization */
    ierr = QEPTOAROrth2(S,ld,j+1,H+j*ldh,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = QEPTOARSNorm2(j+3,S+(j+1)*lds,S+ld+(j+1)*lds,&norm);CHKERRQ(ierr);
    for (i=0;i<=j+2;i++) {
      S[i+(j+1)*lds] /= norm;
      S[i+ld+(j+1)*lds] /= norm;
    }
    H[j+1+ldh*j] = norm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPTOARTrunc"
PetscErrorCode QEPTOARTrunc(QEP qep,PetscScalar *S,PetscInt ld,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscInt nw,PetscReal *rwork,PetscInt nrw)
{
  PetscErrorCode ierr;
  PetscInt       lwa,nwu=0,lrwa,nrwu=0;
  PetscInt       j,i,n,lds=2*ld;
  PetscScalar    *M,*V,*U,t;
  PetscReal      *sg;
  PetscBLASInt   cs1_,rs1_,cs1t2,n_,info,lw_;

  PetscFunctionBegin;
  n = (rs1>2*cs1)?2*cs1:rs1;
  lwa = cs1*rs1*4+n*(rs1+2*cs1);
  lrwa = 6*n;
  if (!work||nw<lwa) {
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",5);
  }
  if (!rwork||nrw<lrwa) {
    if (nrw<lrwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",8);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",7);
  }
  M = work+nwu;
  nwu += rs1*cs1*2;
  sg = rwork+nrwu;
  nrwu += n;
  U = work+nwu;
  nwu += rs1*n;
  V = work+nwu;
  nwu += 2*cs1*n;
  for (i=0;i<cs1;i++) {
    ierr = PetscMemcpy(M+i*rs1,S+i*lds,rs1*sizeof(PetscScalar));CHKERRQ(ierr);  
    ierr = PetscMemcpy(M+(i+cs1)*rs1,S+i*lds+ld,rs1*sizeof(PetscScalar));CHKERRQ(ierr);  
  }
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*2,&cs1t2);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1t2,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1t2,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));  
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  
  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = SlepcUpdateVectors(rs1,qep->V,0,cs1+1,U,rs1,PETSC_FALSE);CHKERRQ(ierr);
  
  /* Update S */
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<cs1+1;i++) {
    t = sg[i];
    PetscStackCall("BLASscal",BLASscal_(&cs1t2,&t,V+i,&n_));
  }
  for (j=0;j<cs1;j++) { 
    ierr = PetscMemcpy(S+j*lds,V+j*n,(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemcpy(S+ld+j*lds,V+(cs1+j)*n,(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPTOARSupdate"
/*
  S <- S*Q 
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
*/
PetscErrorCode QEPTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       lwa,j,lds=2*ld;

  PetscFunctionBegin;
  lwa = sr*ncu;
  if (!work||nw<lwa) {
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",9);
  }
  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(qr,&qr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ncu,&ncu_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldq,&ldq_);CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S,&lds_,Q,&ldq_,&b,work,&sr_));
  for (j=0;j<ncu;j++) {
    ierr = PetscMemcpy(S+lds*(s+j),work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S+ld,&lds_,Q,&ldq_,&b,work,&sr_));
  for (j=0;j<ncu;j++) {
    ierr = PetscMemcpy(S+lds*(s+j)+ld,work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSolve_TOAR"
PetscErrorCode QEPSolve_TOAR(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       j,k,l,nv=0,ld,lds,off,ldds,newn;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0;
  PetscScalar    *S,*Q,*work,*H;
  PetscReal      beta,norm,*rwork;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ld = qep->ncv+2;
  lds = 2*ld;
  lwa = 9*ld*ld+5*ld;
  lrwa = 8*ld;
  ierr = PetscCalloc3(lwa,&work,lrwa,&rwork,2*ld*ld,&S);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(qep->ds,&ldds);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  if (qep->nini==0) {  
    ierr = SlepcVecSetRandom(qep->V[0],qep->rand);CHKERRQ(ierr);
  }
  ierr = SlepcVecSetRandom(qep->V[1],qep->rand);CHKERRQ(ierr);
  ierr = IPNorm(qep->ip,qep->V[0],&norm);CHKERRQ(ierr);
  ierr = VecScale(qep->V[0],1/norm);CHKERRQ(ierr);
  S[0] = norm;
  ierr = IPOrthogonalize(qep->ip,0,NULL,1,NULL,qep->V,qep->V[1],S+ld,&norm,NULL);CHKERRQ(ierr);
  ierr = VecScale(qep->V[1],1/norm);CHKERRQ(ierr);
  S[1+ld] = norm;
  if (norm<PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)qep),1,"Problem with initial vector");
  ierr = QEPTOARSNorm2(2,S,S+ld,&norm);CHKERRQ(ierr);
  for (j=0;j<2;j++) {
    S[j] /= norm;
    S[j+ld] /= norm;
  }
  /* Restart loop */
  l = 0;
  while (qep->reason == QEP_CONVERGED_ITERATING) {
    qep->its++;
    
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(qep->nconv+qep->mpd,qep->ncv);
    ierr = DSGetArray(qep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = QEPTOARrun(qep,S,ld,H,ldds,qep->V,qep->nconv+l,&nv,&breakdown,work+nwu,lwa-nwu,qep->work,3);CHKERRQ(ierr);
    beta = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    ierr = DSRestoreArray(qep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetDimensions(qep->ds,nv,0,qep->nconv,qep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(qep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(qep->ds,qep->eigr,qep->eigi);CHKERRQ(ierr);
    ierr = DSSort(qep->ds,qep->eigr,qep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);;
    ierr = DSUpdateExtraRow(qep->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = QEPKrylovConvergence(qep,PETSC_FALSE,qep->nconv,nv-qep->nconv,nv,beta,&k);CHKERRQ(ierr);
    if (qep->its >= qep->max_it) qep->reason = QEP_DIVERGED_ITS;
    if (k >= qep->nev) qep->reason = QEP_CONVERGED_TOL;

    /* Update l */
    if (qep->reason != QEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      if (!breakdown) {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSTruncate(qep->ds,k+l);CHKERRQ(ierr);
        ierr = DSGetDimensions(qep->ds,&newn,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        l = newn-k;
      }
    }

    /* Update S */
    off = qep->nconv*ldds;
    ierr = DSGetArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = QEPTOARSupdate(S,ld,nv+2,qep->nconv,k+l-qep->nconv,nv,Q+off,ldds,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (qep->reason == QEP_CONVERGED_ITERATING) {
      if (breakdown) {

        /* Stop if breakdown */
        ierr = PetscInfo2(qep,"Breakdown TOAR method (it=%D norm=%G)\n",qep->its,beta);CHKERRQ(ierr);
        qep->reason = QEP_DIVERGED_BREAKDOWN;
      } else {
        /* Truncate S */
        ierr = QEPTOARTrunc(qep,S,ld,nv+2,k+l+1,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);        
      }
    }
    qep->nconv = k;
    ierr = QEPMonitor(qep,qep->its,qep->nconv,qep->eigr,qep->eigi,qep->errest,nv);CHKERRQ(ierr);
  }

  /* Update vectors V = V*S */  
  if (qep->nconv>0) {
    ierr = SlepcUpdateVectors(nv+2,qep->V,0,qep->nconv,S,lds,PETSC_FALSE);CHKERRQ(ierr);
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
    ierr = QEPComputeVectors_Schur(qep);CHKERRQ(ierr);
  }
  ierr = PetscFree3(work,rwork,S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPCreate_TOAR"
PETSC_EXTERN PetscErrorCode QEPCreate_TOAR(QEP qep)
{
  PetscFunctionBegin;
  qep->ops->solve                = QEPSolve_TOAR;
  qep->ops->setup                = QEPSetUp_TOAR;
  qep->ops->reset                = QEPReset_Default;
  PetscFunctionReturn(0);
}
