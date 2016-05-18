/*

   SLEPc polynomial eigensolver: "toar"

   Method: TOAR

   Algorithm:

       Two-Level Orthogonal Arnoldi.

   References:

       [1] Y. Su, J. Zhang and Z. Bai, "A compact Arnoldi algorithm for
           polynomial eigenvalue problems", talk presented at RANMEP 2008.

       [2] C. Campos and J.E. Roman, "Parallel Krylov solvers for the
           polynomial eigenvalue problem in SLEPc", SIAM J. Sci. Comput.
           to appear, 2016.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include "../src/pep/impls/krylov/pepkrylov.h"
#include <slepcblaslapack.h>

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@Article{slepc-pep,\n"
  "   author = \"C. Campos and J. E. Roman\",\n"
  "   title = \"Parallel {Krylov} solvers for the polynomial eigenvalue problem in {SLEPc}\",\n"
  "   journal = \"{SIAM} J. Sci. Comput.\",\n"
  "   volume = \"to appear\",\n"
  "   number = \"\",\n"
  "   pages = \"\",\n"
  "   year = \"2016,\"\n"
  "   doi = \"http://dx.doi.org/10.xxxx/yyyy\"\n"
  "}\n";

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
#define __FUNCT__ "PEPSetUp_TOAR"
PetscErrorCode PEPSetUp_TOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBool      shift,sinv,flg,lindep;
  PetscInt       i,lds,deg=pep->nmat-1,j;
  PetscReal      norm;

  PetscFunctionBegin;
  pep->lineariz = PETSC_TRUE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!ctx->lock && pep->mpd<pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (!pep->max_it) pep->max_it = PetscMax(100,2*(pep->nmat-1)*pep->n/pep->ncv);
  /* Set STSHIFT as the default ST */
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STSHIFT);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSHIFT,&shift);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
  if (!shift && !sinv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Only STSHIFT and STSINVERT spectral transformations can be used");
  if (!pep->which) {
    if (sinv) pep->which = PEP_TARGET_MAGNITUDE;
    else pep->which = PEP_LARGEST_MAGNITUDE;
  }
  if (pep->problem_type!=PEP_GENERAL) {
    ierr = PetscInfo(pep,"Problem type ignored, performing a non-symmetric linearization\n");CHKERRQ(ierr);
  }

  if (!ctx->keep) ctx->keep = 0.5;

  ierr = PEPAllocateSolution(pep,pep->nmat-1);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,3);CHKERRQ(ierr);
  ierr = DSSetType(pep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSSetExtraRow(pep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(pep->ds,pep->ncv+1);CHKERRQ(ierr);

  ierr = PEPBasisCoefficients(pep,pep->pbc);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscMalloc1(pep->nmat,&pep->solvematcoeffs);CHKERRQ(ierr);
    if (sinv) {
      ierr = PEPEvaluateBasis(pep,pep->target,0,pep->solvematcoeffs,NULL);CHKERRQ(ierr);
    } else {
      for (i=0;i<pep->nmat-1;i++) pep->solvematcoeffs[i] = 0.0;
      pep->solvematcoeffs[pep->nmat-1] = 1.0;
    }
  }
  ctx->ld = pep->ncv+(pep->nmat-1);   /* number of rows of each fragment of S */
  lds = (pep->nmat-1)*ctx->ld;
  ierr = PetscCalloc1(lds*ctx->ld,&ctx->S);CHKERRQ(ierr);

  /* process starting vector */
  ctx->nq = 0;
  for (i=0;i<deg;i++) {
    if (pep->nini>-deg) {
      ierr = BVSetRandomColumn(pep->V,ctx->nq);CHKERRQ(ierr);
    } else {
      ierr = BVInsertVec(pep->V,ctx->nq,pep->IS[i]);CHKERRQ(ierr);
    }
    ierr = BVOrthogonalizeColumn(pep->V,ctx->nq,ctx->S+i*ctx->ld,&norm,&lindep);CHKERRQ(ierr);
    if (!lindep) {
      ierr = BVScaleColumn(pep->V,ctx->nq,1.0/norm);CHKERRQ(ierr);
      ctx->S[ctx->nq+i*ctx->ld] = norm;
      ctx->nq++;
    }
  }
  if (ctx->nq==0) SETERRQ(PetscObjectComm((PetscObject)pep),1,"PEP: Problem with initial vector");
  ierr = PEPTOARSNorm2(lds,ctx->S,&norm);CHKERRQ(ierr);
  for (j=0;j<deg;j++) {
    for (i=0;i<=j;i++) ctx->S[i+j*ctx->ld] /= norm;
  }
  if (pep->nini<0) {
    ierr = SlepcBasisDestroy_Private(&pep->nini,&pep->IS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOAROrth2"
/*
 Computes GS orthogonalization   [z;x] - [Sp;Sq]*y,
 where y = ([Sp;Sq]'*[z;x]).
   k: Column from S to be orthogonalized against previous columns.
   Sq = Sp+ld
   dim(work)>=k
*/
static PetscErrorCode PEPTOAROrth2(PEP pep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt k,PetscScalar *y,PetscReal *norm,PetscBool *lindep,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,lds_,k_,one=1;
  PetscScalar    sonem=-1.0,sone=1.0,szero=0.0,*x0,*x,*c;
  PetscInt       i,lds=deg*ld,n;
  PetscReal      eta,onorm;

  PetscFunctionBegin;
  ierr = BVGetOrthogonalization(pep->V,NULL,NULL,&eta,NULL);CHKERRQ(ierr);
  n = k+deg-1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(deg*ld,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* number of vectors to orthogonalize against them */
  c = work;
  x0 = S+k*lds;
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,x0,&one,&szero,y,&one));
  for (i=1;i<deg;i++) {
    x = S+i*ld+k*lds;
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+i*ld,&lds_,x,&one,&sone,y,&one));
  }
  for (i=0;i<deg;i++) {
    x= S+i*ld+k*lds;
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+i*ld,&lds_,y,&one,&sone,x,&one));
  }
  ierr = PEPTOARSNorm2(lds,S+k*lds,&onorm);CHKERRQ(ierr);
  /* twice */
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,x0,&one,&szero,c,&one));
  for (i=1;i<deg;i++) {
    x = S+i*ld+k*lds;
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+i*ld,&lds_,x,&one,&sone,c,&one));
  }
  for (i=0;i<deg;i++) {
    x= S+i*ld+k*lds;
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+i*ld,&lds_,c,&one,&sone,x,&one));
  }
  for (i=0;i<k;i++) y[i] += c[i];
  if (norm) {
    ierr = PEPTOARSNorm2(lds,S+k*lds,norm);CHKERRQ(ierr);
    if (lindep) *lindep = (*norm < eta * onorm)?PETSC_TRUE:PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARExtendBasis"
/*
  Extend the TOAR basis by applying the the matrix operator
  over a vector which is decomposed in the TOAR way
  Input:
    - pbc: array containing the polynomial basis coefficients
    - S,V: define the latest Arnoldi vector (nv vectors in V)
  Output:
    - t: new vector extending the TOAR basis
    - r: temporary coefficients to compute the TOAR coefficients
         for the new Arnoldi vector
  Workspace: t_ (two vectors)
*/
static PetscErrorCode PEPTOARExtendBasis(PEP pep,PetscBool sinvert,PetscScalar sigma,PetscScalar *S,PetscInt ls,PetscInt nv,BV V,Vec t,PetscScalar *r,PetscInt lr,Vec *t_)
{
  PetscErrorCode ierr;
  PetscInt       nmat=pep->nmat,deg=nmat-1,k,j,off=0,lss;
  Vec            v=t_[0],ve=t_[1],q=t_[2];
  PetscScalar    alpha=1.0,*ss,a;
  PetscReal      *ca=pep->pbc,*cb=pep->pbc+nmat,*cg=pep->pbc+2*nmat;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(pep->V,0,nv);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (sinvert) {
    for (j=0;j<nv;j++) {
      if (deg>1) r[lr+j] = S[j]/ca[0];
      if (deg>2) r[2*lr+j] = (S[ls+j]+(sigma-cb[1])*r[lr+j])/ca[1];
    }
    for (k=2;k<deg-1;k++) {
      for (j=0;j<nv;j++) r[(k+1)*lr+j] = (S[k*ls+j]+(sigma-cb[k])*r[k*lr+j]-cg[k]*r[(k-1)*lr+j])/ca[k];
    }
    k = deg-1;
    for (j=0;j<nv;j++) r[j] = (S[k*ls+j]+(sigma-cb[k])*r[k*lr+j]-cg[k]*r[(k-1)*lr+j])/ca[k];
    ss = r; lss = lr; off = 1; alpha = -1.0; a = pep->sfactor;
  } else {
    ss = S; lss = ls; off = 0; alpha = -ca[deg-1]; a = 1.0;
  }
  ierr = BVMultVec(V,1.0,0.0,v,ss+off*lss);CHKERRQ(ierr);
  if (pep->Dr) { /* balancing */
    ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
  }
  ierr = STMatMult(pep->st,off,v,q);CHKERRQ(ierr);
  ierr = VecScale(q,a);CHKERRQ(ierr);
  for (j=1+off;j<deg+off-1;j++) {
    ierr = BVMultVec(V,1.0,0.0,v,ss+j*lss);CHKERRQ(ierr);
    if (pep->Dr) {
      ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
    }
    ierr = STMatMult(pep->st,j,v,t);CHKERRQ(ierr);
    a *= pep->sfactor;
    ierr = VecAXPY(q,a,t);CHKERRQ(ierr);
  }
  if (sinvert) {
    ierr = BVMultVec(V,1.0,0.0,v,ss);CHKERRQ(ierr);
    if (pep->Dr) {
      ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
    }
    ierr = STMatMult(pep->st,deg,v,t);CHKERRQ(ierr);
    a *= pep->sfactor;
    ierr = VecAXPY(q,a,t);CHKERRQ(ierr);
  } else {
    ierr = BVMultVec(V,1.0,0.0,ve,ss+(deg-1)*lss);CHKERRQ(ierr);
    if (pep->Dr) {
      ierr = VecPointwiseMult(ve,ve,pep->Dr);CHKERRQ(ierr);
    }
    a *= pep->sfactor;
    ierr = STMatMult(pep->st,deg-1,ve,t);CHKERRQ(ierr);
    ierr = VecAXPY(q,a,t);CHKERRQ(ierr);
    a *= pep->sfactor;
  }
  if (flg || !sinvert) alpha /= a;
  ierr = STMatSolve(pep->st,q,t);CHKERRQ(ierr);
  ierr = VecScale(t,alpha);CHKERRQ(ierr);
  if (!sinvert) {
    if (cg[deg-1]!=0) { ierr = VecAXPY(t,cg[deg-1],v);CHKERRQ(ierr); }
    if (cb[deg-1]!=0) { ierr = VecAXPY(t,cb[deg-1],ve);CHKERRQ(ierr); }
  }
  if (pep->Dr) {
    ierr = VecPointwiseDivide(t,t,pep->Dr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARCoefficients"
/*
  Compute TOAR coefficients of the blocks of the new Arnoldi vector computed
*/
static PetscErrorCode PEPTOARCoefficients(PEP pep,PetscBool sinvert,PetscScalar sigma,PetscInt nv,PetscScalar *S,PetscInt ls,PetscScalar *r,PetscInt lr,PetscScalar *x)
{
  PetscInt    k,j,nmat=pep->nmat,d=nmat-1;
  PetscReal   *ca=pep->pbc,*cb=pep->pbc+nmat,*cg=pep->pbc+2*nmat;
  PetscScalar t=1.0,tp=0.0,tt;

  PetscFunctionBegin;
  if (sinvert) {
    for (k=1;k<d;k++) {
      tt = t;
      t = ((sigma-cb[k-1])*t-cg[k-1]*tp)/ca[k-1]; /* k-th basis polynomial */
      tp = tt;
      for (j=0;j<=nv;j++) r[k*lr+j] += t*x[j];
    }
  } else {
    for (j=0;j<=nv;j++) r[j] = (cb[0]-sigma)*S[j]+ca[0]*S[ls+j];
    for (k=1;k<d-1;k++) {
      for (j=0;j<=nv;j++) r[k*lr+j] = (cb[k]-sigma)*S[k*ls+j]+ca[k]*S[(k+1)*ls+j]+cg[k]*S[(k-1)*ls+j];
    }
    if (sigma!=0.0) for (j=0;j<=nv;j++) r[(d-1)*lr+j] -= sigma*S[(d-1)*ls+j];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARrun"
/*
  Compute a run of Arnoldi iterations dim(work)=ld
*/
static PetscErrorCode PEPTOARrun(PEP pep,PetscScalar sigma,PetscInt *nq,PetscScalar *S,PetscInt ld,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscScalar *work,Vec *t_)
{
  PetscErrorCode ierr;
  PetscInt       i,j,p,m=*M,nwu=0,deg=pep->nmat-1;
  PetscInt       lds=ld*deg,nqt=*nq;
  Vec            t;
  PetscReal      norm;
  PetscBool      flg,sinvert=PETSC_FALSE,lindep;
  PetscScalar    *x;

  PetscFunctionBegin;
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg) {
    /* spectral transformation handled by the solver */
    ierr = PetscObjectTypeCompareAny((PetscObject)pep->st,&flg,STSINVERT,STSHIFT,"");CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"STtype not supported fr TOAR without transforming matrices");
    ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinvert);CHKERRQ(ierr);
  }
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = BVGetColumn(pep->V,nqt,&t);CHKERRQ(ierr);
    ierr = PEPTOARExtendBasis(pep,sinvert,sigma,S+j*lds,ld,nqt,pep->V,t,S+(j+1)*lds,ld,t_);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,nqt,&t);CHKERRQ(ierr);

    /* orthogonalize */
    if (sinvert) x = S+(j+1)*lds;
    else x = S+(deg-1)*ld+(j+1)*lds;
    ierr = BVOrthogonalizeColumn(pep->V,nqt,x,&norm,&lindep);CHKERRQ(ierr);
    if (!lindep) {
      x[nqt] = norm;
      ierr = BVScaleColumn(pep->V,nqt,1.0/norm);CHKERRQ(ierr);
      nqt++;
    }

    ierr = PEPTOARCoefficients(pep,sinvert,sigma,nqt-1,S+j*lds,ld,S+(j+1)*lds,ld,x);CHKERRQ(ierr);
    /* level-2 orthogonalization */
    ierr = PEPTOAROrth2(pep,S,ld,deg,j+1,H+j*ldh,&norm,breakdown,work+nwu);CHKERRQ(ierr);
    H[j+1+ldh*j] = norm;
    *nq = nqt;
    if (*breakdown) {
      *M = j+1;
      break;
    }
    for (p=0;p<deg;p++) {
      for (i=0;i<=j+deg;i++) {
        S[i+p*ld+(j+1)*lds] /= norm;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARTrunc"
/*
  dim(rwork)=6*n; dim(work)=6*ld*lds+2*cs1
*/
static PetscErrorCode PEPTOARTrunc(PEP pep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt *rs1a,PetscInt cs1,PetscInt lock,PetscInt newc,PetscBool final,PetscScalar *work,PetscReal *rwork)
{
#if defined(PETSC_MISSING_LAPACK_GESVD) || defined(PETSC_MISSING_LAPACK_GEQRF) || defined(PETSC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD/GEQRF/ORGQR - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       nwu=0,nrwu=0,nnc,nrow,lwa;
  PetscInt       j,i,k,n,lds=deg*ld,rs1=*rs1a,rk=0,offu;
  PetscScalar    *M,*V,*pU,*SS,*SS2,t,sone=1.0,zero=0.0,mone=-1.0,*p,*tau;
  PetscReal      *sg,tol;
  PetscBLASInt   cs1_,rs1_,cs1tdeg,n_,info,lw_,newc_,newctdeg,nnc_,nrow_,nnctdeg,lds_,rk_;
  Mat            U;

  PetscFunctionBegin;
  if (cs1==0) PetscFunctionReturn(0);
  lwa = 6*ld*lds+2*cs1;
  n = (rs1>deg*cs1)?deg*cs1:rs1;
  nnc = cs1-lock-newc;
  nrow = rs1-lock;
  ierr = PetscMalloc4(deg*newc*nnc,&SS,newc*nnc,&SS2,(rs1+lock+newc)*n,&pU,deg*rs1,&tau);CHKERRQ(ierr);
  offu = lock*(rs1+1);
  M = work+nwu;
  nwu += rs1*cs1*deg;
  sg = rwork+nrwu;
  nrwu += n;
  ierr = PetscMemzero(pU,rs1*n*sizeof(PetscScalar));CHKERRQ(ierr);
  V = work+nwu;
  nwu += deg*cs1*n;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nnc,&nnc_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(newc,&newc_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(newc*deg,&newctdeg);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nnc*deg,&nnctdeg);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*deg,&cs1tdeg);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nrow,&nrow_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  if (newc>0) {
  /* truncate columns associated with new converged eigenpairs */
    for (j=0;j<deg;j++) {
      for (i=lock;i<lock+newc;i++) {
        ierr = PetscMemcpy(M+(i-lock+j*newc)*nrow,S+i*lds+j*ld+lock,nrow*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
#if !defined (PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&nrow_,&newctdeg,M,&nrow_,sg,pU+offu,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&nrow_,&newctdeg,M,&nrow_,sg,pU+offu,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
    /* SVD has rank min(newc,nrow) */
    rk = PetscMin(newc,nrow);
    for (i=0;i<rk;i++) {
      t = sg[i];
      PetscStackCallBLAS("BLASscal",BLASscal_(&newctdeg,&t,V+i,&n_));
    }
    for (i=0;i<deg;i++) {
      for (j=lock;j<lock+newc;j++) {
        ierr = PetscMemcpy(S+j*lds+i*ld+lock,V+(newc*i+j-lock)*n,rk*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemzero(S+j*lds+i*ld+lock+rk,(ld-lock-rk)*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
    /*
      update columns associated with non-converged vectors, orthogonalize
       against pU so that next M has rank nnc+d-1 insted of nrow+d-1
    */
    for (i=0;i<deg;i++) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&newc_,&nnc_,&nrow_,&sone,pU+offu,&rs1_,S+(lock+newc)*lds+i*ld+lock,&lds_,&zero,SS+i*newc*nnc,&newc_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&nrow_,&nnc_,&newc_,&mone,pU+offu,&rs1_,SS+i*newc*nnc,&newc_,&sone,S+(lock+newc)*lds+i*ld+lock,&lds_));
      /* repeat orthogonalization step */
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&newc_,&nnc_,&nrow_,&sone,pU+offu,&rs1_,S+(lock+newc)*lds+i*ld+lock,&lds_,&zero,SS2,&newc_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&nrow_,&nnc_,&newc_,&mone,pU+offu,&rs1_,SS2,&newc_,&sone,S+(lock+newc)*lds+i*ld+lock,&lds_));
      for (j=0;j<newc*nnc;j++) *(SS+i*newc*nnc+j) += SS2[j];
    }
  }
  /* truncate columns associated with non-converged eigenpairs */
  for (j=0;j<deg;j++) {
    for (i=lock+newc;i<cs1;i++) {
      ierr = PetscMemcpy(M+(i-lock-newc+j*nnc)*nrow,S+i*lds+j*ld+lock,nrow*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&nrow_,&nnctdeg,M,&nrow_,sg,pU+offu+newc*rs1,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&nrow_,&nnctdeg,M,&nrow_,sg,pU+offu+newc*rs1,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  tol = PetscMax(rs1,deg*cs1)*PETSC_MACHINE_EPSILON*sg[0];
  for (i=0;i<PetscMin(n_,nnctdeg);i++) if (sg[i]>tol) rk++;
  rk = PetscMin(nnc+deg-1,rk);
  /* the SVD has rank (atmost) nnc+deg-1 */
  for (i=0;i<rk;i++) {
    t = sg[i];
    PetscStackCallBLAS("BLASscal",BLASscal_(&nnctdeg,&t,V+i,&n_));
  }
  /* update S */
  ierr = PetscMemzero(S+cs1*lds,(ld-cs1)*lds*sizeof(PetscScalar));CHKERRQ(ierr);
  k = ld-lock-newc-rk;
  for (i=0;i<deg;i++) {
    for (j=lock+newc;j<cs1;j++) {
      ierr = PetscMemcpy(S+j*lds+i*ld+lock+newc,V+(nnc*i+j-lock-newc)*n,rk*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscMemzero(S+j*lds+i*ld+lock+newc+rk,k*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  if (newc>0) {
    for (i=0;i<deg;i++) {
      p = SS+nnc*newc*i;
      for (j=lock+newc;j<cs1;j++) {
        for (k=0;k<newc;k++) S[j*lds+i*ld+lock+k] = *(p++);
      }
    }
  }

  /* orthogonalize pU */
  rk = rk+newc;
  ierr = PetscBLASIntCast(rk,&rk_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1-lock,&nnc_);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&nrow_,&rk_,pU+offu,&rs1_,tau,work+nwu,&lw_,&info));
  for (i=0;i<deg;i++) {
    PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&rk_,&nnc_,&sone,pU+offu,&rs1_,S+lock*lds+lock+i*ld,&lds_));
  }
  PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&nrow_,&rk_,&rk_,pU+offu,&rs1_,tau,work+nwu,&lw_,&info));

  /* update vectors V(:,idx) = V*Q(:,idx) */
  rk = rk+lock;
  for (i=0;i<lock;i++) pU[(i+1)*rs1] = 1.0;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs1,rk,pU,&U);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,lock,rs1);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,U,lock,rk);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,rk);CHKERRQ(ierr);
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  *rs1a = rk;

  /* free work space */
  ierr = PetscFree4(SS,SS2,pU,tau);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSupdate"
/*
  S <- S*Q
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
  dim(work)=sr*ncu
*/
static PetscErrorCode PEPTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       j,lds=deg*ld,i;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(qr,&qr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ncu,&ncu_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldq,&ldq_);CHKERRQ(ierr);
  for (i=0;i<deg;i++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S+i*ld,&lds_,Q,&ldq_,&b,work,&sr_));
    for (j=0;j<ncu;j++) {
      ierr = PetscMemcpy(S+lds*(s+j)+i*ld,work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPEvaluateBasisM"
/*
  Computes T_j = phi_idx(T). In T_j and T_p are phi_{idx-1}(T)
   and phi_{idx-2}(T) respectively or null if idx=0,1.
   Tp and Tj are input/output arguments
*/
static PetscErrorCode PEPEvaluateBasisM(PEP pep,PetscInt k,PetscScalar *T,PetscInt ldt,PetscInt idx,PetscScalar **Tp,PetscScalar **Tj)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      *ca,*cb,*cg;
  PetscScalar    *pt,g,a;
  PetscBLASInt   k_,ldt_;

  PetscFunctionBegin;
  if (idx==0) {
    ierr = PetscMemzero(*Tj,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(*Tp,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<k;i++) (*Tj)[i+i*k] = 1.0;
  } else {
    ierr = PetscBLASIntCast(ldt,&ldt_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
    ca = pep->pbc; cb = pep->pbc+pep->nmat; cg = pep->pbc+2*pep->nmat;
    for (i=0;i<k;i++) T[i*ldt+i] -= cb[idx-1];
    a = 1/ca[idx-1];
    g = (idx==1)?0.0:-cg[idx-1]/ca[idx-1];
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&a,T,&ldt_,*Tj,&k_,&g,*Tp,&k_));
    pt = *Tj; *Tj = *Tp; *Tp = pt;
    for (i=0;i<k;i++) T[i*ldt+i] += cb[idx-1];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPExtractInvariantPair"
/* dim(work)=6*sr*k;*/
static PetscErrorCode PEPExtractInvariantPair(PEP pep,PetscScalar sigma,PetscInt sr,PetscInt k,PetscScalar *S,PetscInt ld,PetscInt deg,PetscScalar *H,PetscInt ldh,PetscScalar *work)
{
#if defined(PETSC_MISSING_LAPACK_GESV) || defined(PETSC_MISSING_LAPACK_GETRI) || defined(PETSC_MISSING_LAPACK_GETRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESV/GETRI/GETRF - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       nw,i,j,jj,nwu=0,lds,ldt,d=pep->nmat-1,idxcpy=0;
  PetscScalar    *At,*Bt,*Hj,*Hp,*T,sone=1.0,g,a,*pM;
  PetscBLASInt   k_,sr_,lds_,ldh_,info,*p,lwork,ldt_;
  PetscBool      transf=PETSC_FALSE,flg;
  PetscReal      nrm,norm,maxnrm,*rwork;
  BV             *R,Y;
  Mat            M,*A;
  Vec            v;

  PetscFunctionBegin;
  if (k==0) PetscFunctionReturn(0);
  nw = 6*sr*k;
  lds = deg*ld;
  At = work+nwu;
  nwu += sr*k;
  Bt = work+nwu;
  nwu += k*k;
  ierr = PetscMemzero(Bt,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
  Hj = work+nwu;
  nwu += k*k;
  Hp = work+nwu;
  nwu += k*k;
  ierr = PetscMemzero(Hp,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc1(k,&p);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr =  PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&flg);CHKERRQ(ierr);
    if (flg || sigma!=0.0) transf=PETSC_TRUE;
  }
  if (transf) {
    ldt = k;
    T = work+nwu;
    nwu += k*k;
    for (i=0;i<k;i++) {
      ierr = PetscMemcpy(T+k*i,H+i*ldh,k*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flg) {
      PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&k_,&k_,T,&k_,p,&info));
      ierr = PetscBLASIntCast(nw-nwu,&lwork);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&k_,T,&k_,p,work+nwu,&lwork,&info));
    }
    if (sigma!=0.0) for (i=0;i<k;i++) T[i+k*i] += sigma;
  } else {
    T = H; ldt = ldh;
  }
  ierr = PetscBLASIntCast(ldt,&ldt_);CHKERRQ(ierr);
  switch (pep->extract) {
  case PEP_EXTRACT_NONE:
    break;
  case PEP_EXTRACT_NORM:
    if (pep->basis == PEP_BASIS_MONOMIAL) {
      ierr = PetscBLASIntCast(ldt,&ldt_);CHKERRQ(ierr);
      ierr = PetscMalloc1(k,&rwork);CHKERRQ(ierr);
      norm = LAPACKlange_("F",&k_,&k_,T,&ldt_,rwork);
      PetscFree(rwork);CHKERRQ(ierr);
      if (norm>1.0) idxcpy = d-1;
    } else {
      ierr = PetscBLASIntCast(ldt,&ldt_);CHKERRQ(ierr);
      ierr = PetscMalloc1(k,&rwork);CHKERRQ(ierr);
      maxnrm = 0.0;
      for (i=0;i<pep->nmat-1;i++) {
        ierr = PEPEvaluateBasisM(pep,k,T,ldt,i,&Hp,&Hj);CHKERRQ(ierr);
        norm = LAPACKlange_("F",&k_,&k_,Hj,&k_,rwork);
        if (norm > maxnrm) {
          idxcpy = i;
          maxnrm = norm;
        }
      }
      PetscFree(rwork);CHKERRQ(ierr);
    }
    if (idxcpy>0) {
      /* copy block idxcpy of S to the first one */
      for (j=0;j<k;j++) {
        ierr = PetscMemcpy(S+j*lds,S+idxcpy*ld+j*lds,sr*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
    break;
  case PEP_EXTRACT_RESIDUAL:
    ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscMalloc1(pep->nmat,&A);CHKERRQ(ierr);
      for (i=0;i<pep->nmat;i++) {
        ierr = STGetTOperators(pep->st,i,A+i);CHKERRQ(ierr);
      }
    } else A = pep->A;
    ierr = PetscMalloc1(pep->nmat-1,&R);CHKERRQ(ierr);
    for (i=0;i<pep->nmat-1;i++) {
      ierr = BVDuplicateResize(pep->V,k,R+i);CHKERRQ(ierr);
    }
    ierr = BVDuplicateResize(pep->V,sr,&Y);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,sr,k,NULL,&M);CHKERRQ(ierr);
    g = 0.0; a = 1.0;
    ierr = BVSetActiveColumns(pep->V,0,sr);CHKERRQ(ierr);
    for (j=0;j<pep->nmat;j++) {
      ierr = BVMatMult(pep->V,A[j],Y);CHKERRQ(ierr);
      ierr = PEPEvaluateBasisM(pep,k,T,ldt,i,&Hp,&Hj);CHKERRQ(ierr);
      for (i=0;i<pep->nmat-1;i++) {
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&sr_,&k_,&k_,&a,S+i*ld,&lds_,Hj,&k_,&g,At,&sr_));
        ierr = MatDenseGetArray(M,&pM);CHKERRQ(ierr);
        for (jj=0;jj<k;jj++) {
          ierr = PetscMemcpy(pM+jj*sr,At+jj*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(M,&pM);CHKERRQ(ierr);
        ierr = BVMult(R[i],1.0,(i==0)?0.0:1.0,Y,M);CHKERRQ(ierr);
      }
    }

    /* frobenius norm */
    maxnrm = 0.0;
    for (i=0;i<pep->nmat-1;i++) {
      norm = 0.0;
      for (j=0;j<k;j++) {
        ierr = BVGetColumn(R[i],j,&v);CHKERRQ(ierr);
        ierr = VecNorm(v,NORM_2,&nrm);CHKERRQ(ierr);
        ierr = BVRestoreColumn(R[i],j,&v);CHKERRQ(ierr);
        norm += nrm*nrm;
      }
      norm = PetscSqrtReal(norm);
      if (maxnrm > norm) {
        maxnrm = norm;
        idxcpy = i;
      }
    }
    if (idxcpy>0) {
      /* copy block idxcpy of S to the first one */
      for (j=0;j<k;j++) {
        ierr = PetscMemcpy(S+j*lds,S+idxcpy*ld+j*lds,sr*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
    if (flg) PetscFree(A);CHKERRQ(ierr);
    for (i=0;i<pep->nmat-1;i++) {
      ierr = BVDestroy(&R[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(R);CHKERRQ(ierr);
    ierr = BVDestroy(&Y);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    break;
  case PEP_EXTRACT_STRUCTURED:
    for (j=0;j<k;j++) Bt[j+j*k] = 1.0;
    for (j=0;j<sr;j++) {
      for (i=0;i<k;i++) At[j*k+i] = PetscConj(S[i*lds+j]);
    }
    ierr = PEPEvaluateBasisM(pep,k,T,ldt,0,&Hp,&Hj);CHKERRQ(ierr);
    for (i=1;i<deg;i++) {
      ierr = PEPEvaluateBasisM(pep,k,T,ldt,i,&Hp,&Hj);CHKERRQ(ierr);
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&k_,&sr_,&k_,&sone,Hj,&k_,S+i*ld,&lds_,&sone,At,&k_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&k_,&k_,&k_,&sone,Hj,&k_,Hj,&k_,&sone,Bt,&k_));
    }
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&k_,&sr_,Bt,&k_,p,At,&k_,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESV %d",info);
    for (j=0;j<sr;j++) {
      for (i=0;i<k;i++) S[i*lds+j] = PetscConj(At[j*k+i]);
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Extraction not implemented in this solver");
  }
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_TOAR"
PetscErrorCode PEPSolve_TOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       i,j,k,l,nv=0,ld,lds,off,ldds,newn,nq=ctx->nq,nconv=0,locked=0,newc;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0,nmat=pep->nmat,deg=nmat-1;
  PetscScalar    *S,*Q,*work,*H,sigma;
  PetscReal      beta,*rwork;
  PetscBool      breakdown=PETSC_FALSE,flg,falselock=PETSC_FALSE,sinv=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  if (ctx->lock) {
    ierr = PetscOptionsGetBool(NULL,NULL,"-pep_toar_falselocking",&falselock,NULL);CHKERRQ(ierr);
  }
  ld = ctx->ld;
  S = ctx->S;
  lds = deg*ld;        /* leading dimension of S */
  lwa = (deg+6)*ld*lds;
  lrwa = 7*lds;
  ierr = PetscMalloc2(lwa,&work,lrwa,&rwork);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);
  ierr = STGetShift(pep->st,&sigma);CHKERRQ(ierr);

  /* update polynomial basis coefficients */
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (pep->sfactor!=1.0) {
    for (i=0;i<nmat;i++) {
      pep->pbc[nmat+i] /= pep->sfactor;
      pep->pbc[2*nmat+i] /= pep->sfactor*pep->sfactor;
    }
    if (!flg) {
      pep->target /= pep->sfactor;
      ierr = RGPushScale(pep->rg,1.0/pep->sfactor);CHKERRQ(ierr);
      ierr = STScaleShift(pep->st,1.0/pep->sfactor);CHKERRQ(ierr);
      sigma /= pep->sfactor;
    } else {
      ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
      ierr = RGPushScale(pep->rg,sinv?pep->sfactor:1.0/pep->sfactor);CHKERRQ(ierr);
      ierr = STScaleShift(pep->st,sinv?pep->sfactor:1.0/pep->sfactor);CHKERRQ(ierr);
    }
  }

  if (flg) sigma = 0.0;

  /* restart loop */
  l = 0;
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;

    /* compute an nv-step Lanczos factorization */
    nv = PetscMax(PetscMin(nconv+pep->mpd,pep->ncv),nv);
    ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = PEPTOARrun(pep,sigma,&nq,S,ld,H,ldds,pep->nconv+l,&nv,&breakdown,work+nwu,pep->work);CHKERRQ(ierr);
    beta = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetDimensions(pep->ds,nv,0,pep->nconv,pep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(pep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* solve projected problem */
    ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(pep->ds);CHKERRQ(ierr);

    /* check convergence */
    ierr = PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,nv-pep->nconv,beta,&k);CHKERRQ(ierr);
    ierr = (*pep->stopping)(pep,pep->its,pep->max_it,k,pep->nev,&pep->reason,pep->stoppingctx);CHKERRQ(ierr);

    /* update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv==k)?0:PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      if (!breakdown) {
        /* prepare the Rayleigh quotient for restart */
        ierr = DSTruncate(pep->ds,k+l);CHKERRQ(ierr);
        ierr = DSGetDimensions(pep->ds,&newn,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        l = newn-k;
      }
    }
    nconv = k;
    if (!ctx->lock && pep->reason == PEP_CONVERGED_ITERATING && !breakdown) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */

    /* update S */
    off = pep->nconv*ldds;
    ierr = DSGetArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = PEPTOARSupdate(S,ld,deg,nq,pep->nconv,k+l-pep->nconv,nv,Q+off,ldds,work+nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (breakdown) {
      /* stop if breakdown */
      ierr = PetscInfo2(pep,"Breakdown TOAR method (it=%D norm=%g)\n",pep->its,(double)beta);CHKERRQ(ierr);
      pep->reason = PEP_DIVERGED_BREAKDOWN;
    }
    if (pep->reason != PEP_CONVERGED_ITERATING) {l--; flg = PETSC_TRUE;}
    else flg = PETSC_FALSE;
    /* truncate S */
    if (k+l+deg<nq) {
      if (!falselock && ctx->lock) {
        newc = k-pep->nconv;
        ierr = PEPTOARTrunc(pep,S,ld,deg,&nq,k+l+1,locked,newc,flg,work+nwu,rwork+nrwu);CHKERRQ(ierr);
        locked += newc;
      } else {
        ierr = PEPTOARTrunc(pep,S,ld,deg,&nq,k+l+1,0,0,flg,work+nwu,rwork+nrwu);CHKERRQ(ierr);
      }
    }
    pep->nconv = k;
    ierr = PEPMonitor(pep,pep->its,nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }
  if (pep->nconv>0) {
    /* {V*S_nconv^i}_{i=0}^{d-1} has rank nconv instead of nconv+d-1. Force zeros in each S_nconv^i block */
    nq = pep->nconv;

    /* perform Newton refinement if required */
    if (pep->refine==PEP_REFINE_MULTIPLE && pep->rits>0) {
      /* extract invariant pair */
      ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      ierr = PEPExtractInvariantPair(pep,sigma,nq,pep->nconv,S,ld,deg,H,ldds,work+nwu);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
      ierr = PEPNewtonRefinement_TOAR(pep,sigma,&pep->rits,NULL,pep->nconv,S,lds,&nq);CHKERRQ(ierr);
      ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
      ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = DSGetArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
      ierr = PEPTOARSupdate(S,ld,deg,nq,0,pep->nconv,pep->nconv,Q,ldds,work+nwu);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    } else {
      ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
  }
  if (pep->refine!=PEP_REFINE_MULTIPLE || pep->rits==0) {
    ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
    if (!flg) {
      if (pep->ops->backtransform) {
        ierr = (*pep->ops->backtransform)(pep);CHKERRQ(ierr);
      }
      /* restore original values */
      pep->target *= pep->sfactor;
      ierr = STScaleShift(pep->st,pep->sfactor);CHKERRQ(ierr);
    } else {
      ierr = STScaleShift(pep->st,sinv?1.0/pep->sfactor:pep->sfactor);CHKERRQ(ierr);
    }
    if (pep->sfactor!=1.0) {
      for (j=0;j<pep->nconv;j++) {
        pep->eigr[j] *= pep->sfactor;
        pep->eigi[j] *= pep->sfactor;
      }
      /* restore original values */
      for (i=0;i<pep->nmat;i++){
        pep->pbc[pep->nmat+i] *= pep->sfactor;
        pep->pbc[2*pep->nmat+i] *= pep->sfactor*pep->sfactor;
      }
    }
  }
  if (pep->sfactor!=1.0) { ierr = RGPopScale(pep->rg);CHKERRQ(ierr); }

  /* change the state to raw so that DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSetRestart_TOAR"
static PetscErrorCode PEPTOARSetRestart_TOAR(PEP pep,PetscReal keep)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSetRestart"
/*@
   PEPTOARSetRestart - Sets the restart parameter for the TOAR
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -pep_toar_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: PEPTOARGetRestart()
@*/
PetscErrorCode PEPTOARSetRestart(PEP pep,PetscReal keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  ierr = PetscTryMethod(pep,"PEPTOARSetRestart_C",(PEP,PetscReal),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARGetRestart_TOAR"
static PetscErrorCode PEPTOARGetRestart_TOAR(PEP pep,PetscReal *keep)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARGetRestart"
/*@
   PEPTOARGetRestart - Gets the restart parameter used in the TOAR method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: PEPTOARSetRestart()
@*/
PetscErrorCode PEPTOARGetRestart(PEP pep,PetscReal *keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscUseMethod(pep,"PEPTOARGetRestart_C",(PEP,PetscReal*),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSetLocking_TOAR"
static PetscErrorCode PEPTOARSetLocking_TOAR(PEP pep,PetscBool lock)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARSetLocking"
/*@
   PEPTOARSetLocking - Choose between locking and non-locking variants of
   the TOAR method.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -pep_toar_locking - Sets the locking flag

   Notes:
   The default is to lock converged eigenpairs when the method restarts.
   This behaviour can be changed so that all directions are kept in the
   working subspace even if already converged to working accuracy (the
   non-locking variant).

   Level: advanced

.seealso: PEPTOARGetLocking()
@*/
PetscErrorCode PEPTOARSetLocking(PEP pep,PetscBool lock)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,lock,2);
  ierr = PetscTryMethod(pep,"PEPTOARSetLocking_C",(PEP,PetscBool),(pep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARGetLocking_TOAR"
static PetscErrorCode PEPTOARGetLocking_TOAR(PEP pep,PetscBool *lock)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPTOARGetLocking"
/*@
   PEPTOARGetLocking - Gets the locking flag used in the TOAR method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: PEPTOARSetLocking()
@*/
PetscErrorCode PEPTOARGetLocking(PEP pep,PetscBool *lock)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(lock,2);
  ierr = PetscUseMethod(pep,"PEPTOARGetLocking_C",(PEP,PetscBool*),(pep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_TOAR"
PetscErrorCode PEPSetFromOptions_TOAR(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg,lock;
  PetscReal      keep;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP TOAR Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pep_toar_restart","Proportion of vectors kept after restart","PEPTOARSetRestart",0.5,&keep,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPTOARSetRestart(pep,keep);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-pep_toar_locking","Choose between locking and non-locking variants","PEPTOARSetLocking",PETSC_FALSE,&lock,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPTOARSetLocking(pep,lock);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_TOAR"
PetscErrorCode PEPView_TOAR(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  TOAR: %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  TOAR: using the %slocking variant\n",ctx->lock?"":"non-");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_TOAR"
PetscErrorCode PEPDestroy_TOAR(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetLocking_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_TOAR"
PETSC_EXTERN PetscErrorCode PEPCreate_TOAR(PEP pep)
{
  PEP_TOAR       *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  pep->data = (void*)ctx;
  ctx->lock = PETSC_TRUE;

  pep->ops->solve          = PEPSolve_TOAR;
  pep->ops->setup          = PEPSetUp_TOAR;
  pep->ops->setfromoptions = PEPSetFromOptions_TOAR;
  pep->ops->destroy        = PEPDestroy_TOAR;
  pep->ops->view           = PEPView_TOAR;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_TOAR;
  pep->ops->reset          = PEPReset_TOAR;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetRestart_C",PEPTOARSetRestart_TOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetRestart_C",PEPTOARGetRestart_TOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetLocking_C",PEPTOARSetLocking_TOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetLocking_C",PEPTOARGetLocking_TOAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

