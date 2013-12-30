/*

   SLEPc polynomial eigensolver: "toar"

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

#include <slepc-private/pepimpl.h>         /*I "slepcpep.h" I*/
#include <slepcblaslapack.h>


#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_TOAR"
PetscErrorCode PEPSetUp_TOAR(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      sinv;

  PetscFunctionBegin;
  if (pep->ncv) { /* ncv set */
    if (pep->ncv<pep->nev) SETERRQ(PetscObjectComm((PetscObject)pep),1,"The value of ncv must be at least nev");
  } else if (pep->mpd) { /* mpd set */
    pep->ncv = PetscMin(pep->n,pep->nev+pep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (pep->nev<500) pep->ncv = PetscMin(pep->n,PetscMax(2*pep->nev,pep->nev+15));
    else {
      pep->mpd = 500;
      pep->ncv = PetscMin(pep->n,pep->nev+pep->mpd);
    }
  }
  if (!pep->mpd) pep->mpd = pep->ncv;
  if (pep->ncv>pep->nev+pep->mpd) SETERRQ(PetscObjectComm((PetscObject)pep),1,"The value of ncv must not be larger than nev+mpd");
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv); 
  if (!pep->which) {
    ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
    if (sinv) pep->which = PEP_TARGET_MAGNITUDE;
    else pep->which = PEP_LARGEST_MAGNITUDE;
  }
  ierr = PEPAllocateSolution(pep,pep->nmat-1);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,4);CHKERRQ(ierr);
  ierr = DSSetType(pep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSSetExtraRow(pep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(pep->ds,pep->ncv+1);CHKERRQ(ierr);
  ierr = STSetUp(pep->st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSNorm2"
/*
  Norm of [sp;sq] 
*/
static PetscErrorCode PEPTOARSNorm2(PetscInt n,PetscScalar *S,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,one=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  *norm = BLASnrm2_(&n_,S,&one);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOAROrth2"
/*
 Computes GS orthogonalization   [z;x] - [Sp;Sq]*y,
 where y = ([Sp;Sq]'*[z;x]).
   k: Column from S to be orthogonalized against previous columns.
   Sq = Sp+ld
*/
static PetscErrorCode PEPTOAROrth2(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt k,PetscScalar *y,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,lds_,k_,one=1;
  PetscScalar    sonem=-1.0,sone=1.0,szero=0.0,*x0,*x,*c;
  PetscInt       lwa,nwu=0,i,lds=deg*ld,n;
  
  PetscFunctionBegin;
  n = k+deg-1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(deg*ld,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  lwa = k;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
  c = work+nwu;
  nwu += k;
  x0 = S+k*lds;
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,x0,&one,&szero,y,&one));
  for (i=1;i<deg;i++) {
    x = S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+i*ld,&lds_,x,&one,&sone,y,&one));
  }
  for (i=0;i<deg;i++) {
    x= S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+i*ld,&lds_,y,&one,&sone,x,&one));
  }
  /* twice */
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,x0,&one,&szero,c,&one));
  for (i=1;i<deg;i++) {
    x = S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+i*ld,&lds_,x,&one,&sone,c,&one));
  }
  for (i=0;i<deg;i++) {
    x= S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+i*ld,&lds_,c,&one,&sone,x,&one));
  }
  for (i=0;i<k;i++) y[i] += c[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARApplyOp"
/*
  Apply the matrix operator over a vector which is decomposed 
   on the TOAR way 
*/
static PetscErrorCode PEPTOARApplyOp(PEP pep,PetscScalar *S,PetscInt ld,PetscInt nv,Vec *V,Vec t,Vec *t_,PetscInt nwv)
{
  PetscErrorCode ierr;
  PetscInt       deg=pep->nmat-1,i;
  Vec            v=t_[0],q=t_[1];
  PetscScalar    alpha=1.0;

  PetscFunctionBegin;
  if (!t_||nwv<2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",12);
  ierr = VecZeroEntries(v);CHKERRQ(ierr);
  ierr = VecMAXPY(v,nv,S,V);CHKERRQ(ierr);
  if (pep->Dr) { /* Balancing */
    ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
  }
  ierr = STMatMult(pep->st,0,v,q);CHKERRQ(ierr);
  for (i=1;i<deg;i++) {
    alpha *= pep->sfactor;
    ierr = VecZeroEntries(v);CHKERRQ(ierr);
    ierr = VecMAXPY(v,nv,S+i*ld,V);CHKERRQ(ierr);
    if (pep->Dr) { /* Balancing */
      ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
    }
    ierr = STMatMult(pep->st,i,v,t);CHKERRQ(ierr);
    ierr = VecAXPY(q,alpha,t);CHKERRQ(ierr);
  }
  alpha *= pep->sfactor;
  ierr = STMatSolve(pep->st,deg,q,t);CHKERRQ(ierr);
  ierr = VecScale(t,-1.0/alpha);CHKERRQ(ierr);
  if (pep->Dr) { /* Balancing */
    ierr = VecPointwiseDivide(t,t,pep->Dr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARrun"
/*
  Compute a run of Arnoldi iterations
*/
static PetscErrorCode PEPTOARrun(PEP pep,PetscScalar *S,PetscInt ld,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscScalar *work,PetscInt nw,Vec *t_,PetscInt nwv)
{
  PetscErrorCode ierr;
  PetscInt       i,j,p,m=*M,nwu=0,lwa,deg=pep->nmat-1;
  PetscInt       lds=ld*deg;
  Vec            t=t_[0];
  PetscReal      norm;

  PetscFunctionBegin;
  if (!t_||nwv<3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",12);
  lwa = ld;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = PEPTOARApplyOp(pep,S+j*lds,ld,j+deg,V,t,t_+1,2);CHKERRQ(ierr);
    
    /* orthogonalize */
    ierr = IPOrthogonalize(pep->ip,0,NULL,j+deg,NULL,pep->V,t,S+(deg-1)*ld+(j+1)*lds,&norm,breakdown);CHKERRQ(ierr);
    *(S+(deg-1)*ld+(j+1)*lds+j+deg) = norm;
    ierr = VecScale(t,1.0/norm);CHKERRQ(ierr);
    ierr = VecCopy(t,V[j+deg]);CHKERRQ(ierr);
    for (p=0;p<deg-1;p++) {
      for (i=0;i<=j+deg;i++) *(S+p*ld+(j+1)*lds+i) = *(S+(p+1)*ld+j*lds+i);
    }

    /* Level-2 orthogonalization */
    ierr = PEPTOAROrth2(S,ld,deg,j+1,H+j*ldh,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = PEPTOARSNorm2(lds,S+(j+1)*lds,&norm);CHKERRQ(ierr);
    for (p=0;p<deg;p++) {
      for (i=0;i<=j+deg;i++) {
        S[i+p*ld+(j+1)*lds] /= norm;
      }
    }
    H[j+1+ldh*j] = norm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARTrunc"
PetscErrorCode PEPTOARTrunc(PEP pep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscInt nw,PetscReal *rwork,PetscInt nrw)
{
  PetscErrorCode ierr;
  PetscInt       lwa,nwu=0,lrwa,nrwu=0;
  PetscInt       j,i,n,lds=deg*ld;
  PetscScalar    *M,*V,*U,t;
  PetscReal      *sg;
  PetscBLASInt   cs1_,rs1_,cs1tdeg,n_,info,lw_;

  PetscFunctionBegin;
  n = (rs1>deg*cs1)?deg*cs1:rs1;
  lwa = 5*ld*lds;
  lrwa = 6*n;
  if (!work||nw<lwa) {
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",5);
  }
  if (!rwork||nrw<lrwa) {
    if (nrw<lrwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",8);
    if (!rwork) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",7);
  }
  M = work+nwu;
  nwu += rs1*cs1*deg;
  sg = rwork+nrwu;
  nrwu += n;
  U = work+nwu;
  nwu += rs1*n;
  V = work+nwu;
  nwu += deg*cs1*n;
  for (i=0;i<cs1;i++) {
    for (j=0;j<deg;j++) {
      ierr = PetscMemcpy(M+(i+j*cs1)*rs1,S+i*lds+j*ld,rs1*sizeof(PetscScalar));CHKERRQ(ierr);
    } 
  }
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*deg,&cs1tdeg);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1tdeg,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1tdeg,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));  
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  
  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = SlepcUpdateVectors(rs1,pep->V,0,cs1+deg-1,U,rs1,PETSC_FALSE);CHKERRQ(ierr);
  
  /* Update S */
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<cs1+deg-1;i++) {
    t = sg[i];
    PetscStackCall("BLASscal",BLASscal_(&cs1tdeg,&t,V+i,&n_));
  }
  for (j=0;j<cs1;j++) {
    for (i=0;i<deg;i++) {
      ierr = PetscMemcpy(S+j*lds+i*ld,V+(cs1*i+j)*n,(cs1+deg-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSupdate"
/*
  S <- S*Q 
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
*/
PetscErrorCode PEPTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       lwa,j,lds=deg*ld,i;

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
  for (i=0;i<deg;i++) {
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S+i*ld,&lds_,Q,&ldq_,&b,work,&sr_));
    for (j=0;j<ncu;j++) {
      ierr = PetscMemcpy(S+lds*(s+j)+i*ld,work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPExtractInvariantPair"
static PetscErrorCode PEPExtractInvariantPair(PetscInt k,PetscScalar *S,PetscInt ld,PetscInt deg,PetscScalar *H,PetscInt ldh,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscInt       sr,i,j,nwu=0,lwa,lds;
  PetscScalar    *At,*Bt,*W,*Hj,sone=1.0,zero=0.0;
  PetscBLASInt   k_,sr_,lds_,ldh_,info,*p;

  PetscFunctionBegin;
  if (k==0) PetscFunctionReturn(0);
  sr = k+deg-1;
  lwa = 4*sr*k;
  if (!work||nw<lwa) {
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",9);
  }
  lds = deg*ld;
  At = work+nwu;
  nwu += sr*k;
  Bt = work+nwu;
  nwu += k*k;
  ierr = PetscMemzero(Bt,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
  W = work+nwu;
  nwu += k*k;
  Hj = work+nwu;
  nwu += k*k;
  ierr = PetscMalloc1(k,&p);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  for (j=0;j<sr;j++) {
    for (i=0;i<k;i++) {
      At[j*k+i] = PetscConj(S[i*lds+j]);
    }
    Bt[j+j*k] = 1.0;
  }
  for (i=1;i<deg;i++) {
    if (i>1) {
      ierr = PetscMemcpy(W,Hj,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
      PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,W,&k_,H,&ldh_,&zero,Hj,&k_));   
    } else {
      for (j=0;j<k;j++) {
        ierr = PetscMemcpy(Hj+j*k,H+j*ldh,k*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
    PetscStackCall("BLASgemm",BLASgemm_("N","C",&k_,&sr_,&k_,&sone,Hj,&k_,S+i*ld,&lds_,&sone,At,&k_));
    PetscStackCall("BLASgemm",BLASgemm_("N","C",&k_,&k_,&k_,&sone,Hj,&k_,Hj,&k_,&sone,Bt,&k_));    
  }
  PetscStackCall("LAPACKgesv",LAPACKgesv_(&k_,&sr_,Bt,&k_,p,At,&k_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESV %d",info);
  ierr = PetscFree(p);CHKERRQ(ierr);
  for (j=0;j<sr;j++) {
    for (i=0;i<k;i++) {
      S[i*lds+j] = PetscConj(At[j*k+i]);
    }
  }    
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_TOAR"
PetscErrorCode PEPSolve_TOAR(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,l,nv,ld,lds,off,ldds,newn;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0,deg=pep->nmat-1;
  PetscScalar    *S,*Q,*work,*H;
  PetscReal      beta,norm,*rwork;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ld = pep->ncv+deg;
  lds = deg*ld;
  lwa = (deg+5)*ld*lds;
  lrwa = 7*lds;
  ierr = PetscMalloc3(lwa,&work,lrwa,&rwork,lds*ld,&S);CHKERRQ(ierr);
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  if (pep->nini==0) {  
    ierr = SlepcVecSetRandom(pep->V[0],pep->rand);CHKERRQ(ierr);
  }
  ierr = IPNorm(pep->ip,pep->V[0],&norm);CHKERRQ(ierr);
  ierr = VecScale(pep->V[0],1/norm);CHKERRQ(ierr);
  S[0] = norm;
  for (i=1;i<deg;i++) {
    ierr = SlepcVecSetRandom(pep->V[i],pep->rand);CHKERRQ(ierr);
    ierr = IPOrthogonalize(pep->ip,0,NULL,i,NULL,pep->V,pep->V[i],S+i*ld,&norm,NULL);CHKERRQ(ierr);
    ierr = VecScale(pep->V[i],1/norm);CHKERRQ(ierr);
    S[i+i*ld] = norm;
    if (norm<PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)pep),1,"Problem with initial vector");
  }
  ierr = PEPTOARSNorm2(lds,S,&norm);CHKERRQ(ierr);
  for (j=0;j<deg;j++) {
    for (i=0;i<=j;i++) S[i+j*ld] /= norm;
  }
  /* Restart loop */
  l = 0;
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;
    
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(pep->nconv+pep->mpd,pep->ncv);
    ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = PEPTOARrun(pep,S,ld,H,ldds,pep->V,pep->nconv+l,&nv,&breakdown,work+nwu,lwa-nwu,pep->work,3);CHKERRQ(ierr);
    beta = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetDimensions(pep->ds,nv,0,pep->nconv,pep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(pep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);;
    ierr = DSUpdateExtraRow(pep->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,nv-pep->nconv,nv,beta,&k);CHKERRQ(ierr);
    if (pep->its >= pep->max_it) pep->reason = PEP_DIVERGED_ITS;
    if (k >= pep->nev) pep->reason = PEP_CONVERGED_TOL;

    /* Update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      if (!breakdown) {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSTruncate(pep->ds,k+l);CHKERRQ(ierr);
        ierr = DSGetDimensions(pep->ds,&newn,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        l = newn-k;
      }
    }

    /* Update S */
    off = pep->nconv*ldds;
    ierr = DSGetArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = PEPTOARSupdate(S,ld,deg,nv+deg,pep->nconv,k+l-pep->nconv,nv,Q+off,ldds,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (pep->reason == PEP_CONVERGED_ITERATING) {
      if (breakdown) {

        /* Stop if breakdown */
        ierr = PetscInfo2(pep,"Breakdown TOAR method (it=%D norm=%G)\n",pep->its,beta);CHKERRQ(ierr);
        pep->reason = PEP_DIVERGED_BREAKDOWN;
      } else {
        /* Truncate S */
        ierr = PEPTOARTrunc(pep,S,ld,deg,nv+deg,k+l+1,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);
      }
    }
    pep->nconv = k;
    ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }
  if (pep->nconv>0) {
  /* Extract invariant pair */
  /* ////////////////////////// */
    {PetscBool ext=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,"-extraction",&ext,NULL);CHKERRQ(ierr);
    ierr = PEPTOARTrunc(pep,S,ld,deg,nv+deg,pep->nconv,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);
    if (ext) {
  /* /////////////////////////// */
      ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      ierr = PEPExtractInvariantPair(pep->nconv,S,ld,deg,H,ldds,work+nwu,lwa-nwu);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    }
    }/* ///////////////// */
  /* Update vectors V = V*S */    
    ierr = SlepcUpdateVectors(pep->nconv+deg-1,pep->V,0,pep->nconv,S,lds,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (j=0;j<pep->nconv;j++) {
    pep->eigr[j] *= pep->sfactor;
    pep->eigi[j] *= pep->sfactor;
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  /* Compute eigenvectors */
  if (pep->nconv > 0) {
    ierr = PEPComputeVectors_Schur(pep);CHKERRQ(ierr);
  }
  ierr = PetscFree3(work,rwork,S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_TOAR"
PETSC_EXTERN PetscErrorCode PEPCreate_TOAR(PEP pep)
{
  PetscFunctionBegin;
  pep->ops->solve                = PEPSolve_TOAR;
  pep->ops->setup                = PEPSetUp_TOAR;
  pep->ops->reset                = PEPReset_Default;
  PetscFunctionReturn(0);
}
