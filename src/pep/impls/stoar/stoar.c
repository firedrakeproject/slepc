/*

   SLEPc polynomial eigensolver: "stoar"

   Method: S-TOAR

   Algorithm:

       Symmetric Two-Level Orthogonal Arnoldi.

   References:

       [1] C. Campos and J.E. Roman, "A thick-restart Q-Lanczos method
           for quadratic eigenvalue problems", submitted, 2013.

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

typedef struct {
  PetscBool   lock;         /* locking/non-locking variant */
  PetscBool   monic;
  PetscInt    d,ld;
  PetscScalar *S,*qK;
  PetscReal   *qM;
} PEP_STOAR;

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_STOAR"
PetscErrorCode PEPSetUp_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      sinv,flg;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscInt       ld;

  PetscFunctionBegin;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!ctx->lock && pep->mpd<pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
  if (!pep->which) {
    ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
    if (sinv) pep->which = PEP_TARGET_MAGNITUDE;
    else pep->which = PEP_LARGEST_MAGNITUDE;
  }
  if (pep->problem_type!=PEP_HERMITIAN) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  if (pep->nmat!=3) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver only available for quadratic problems");
  if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver not implemented for non-monomial bases");
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag set, see STSetTransform()");

  ierr = PEPAllocateSolution(pep,2);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,4);CHKERRQ(ierr);
  ld = pep->ncv+2;
  ierr = DSSetType(pep->ds,DSGHIEP);CHKERRQ(ierr);
  ierr = DSSetCompact(pep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(pep->ds,ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(pep->st,&ctx->d);CHKERRQ(ierr);
  ctx->d--;
  ctx->ld = ld;
  ierr = PetscCalloc3(ctx->d*ld*ld,&ctx->S,ld,&ctx->qM,ld*ld,&ctx->qK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARNorm"
/*
  Compute B-norm of v=[v1;v2] whith  B=diag(-pep->T[0],pep->T[2]) 
*/
static PetscErrorCode PEPSTOARNorm(PEP pep,PetscInt j,PetscReal *norm,PetscScalar *w,PetscInt lw)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscBLASInt   n_,one=1,ld_;
  PetscScalar    sone=1.0,szero=0.0,*sp,*sq;
  PetscInt       lwa,n,i,lds=ctx->d*ctx->ld;

  PetscFunctionBegin;
  n = j+2;
  lwa = n;
  if (lw<lwa) SETERRQ1(PETSC_COMM_SELF,1,"Wrong value of lw",lw);
  sp = ctx->S+lds*j;
  sq = sp+ctx->ld;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,ctx->qK,&ld_,sp,&one,&szero,w,&one));
  *norm = 0.0;
  for (i=0;i<n;i++) *norm += PetscRealPart(w[i]*PetscConj(sp[i])+PetscConj(sq[i])*sq[i]*(*(ctx->qM+i)));
  *norm = (*norm>0.0)?PetscSqrtReal(*norm):-PetscSqrtReal(-*norm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOAROrth2"
/*
  Computes GS orthogonalization  x = [z;x] - [Sp;Sq]*y,
  where y = Omega\([Sp;Sq]'*[qK zeros(size(qK,1)) ;zeros(size(qK,1)) qM]*[z;x]).
  n: Column from S to be orthogonalized against previous columns.
*/
static PetscErrorCode PEPSTOAROrth2(PEP pep,PetscInt k,PetscReal *Omega,PetscScalar *y,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscBLASInt   n_,lds_,k_,one=1,ld_;
  PetscScalar    *S=ctx->S,sonem=-1.0,sone=1.0,szero=0.0,*tp,*tq,*xp,*xq,*c;
  PetscInt       lwa,nwu=0,i,lds=ctx->d*ctx->ld,n;
  
  PetscFunctionBegin;
  n = k+1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr); /* Size of qK and qM */
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  lwa = 3*n;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
  tp = work+nwu;
  nwu += n;
  tq = work+nwu;
  nwu += n;
  c = work+nwu;
  nwu += k;
  xp = S+k*lds;
  xq = S+ctx->ld+k*lds;
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,ctx->qK,&ld_,xp,&one,&szero,tp,&one));
  for (i=0;i<n;i++) tq[i] = *(ctx->qM+i)*xq[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,ctx->S,&lds_,tp,&one,&szero,y,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ctx->ld,&lds_,tq,&one,&sone,y,&one));
  for (i=0;i<n-1;i++) y[i] /= Omega[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,y,&one,&sone,xp,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ctx->ld,&lds_,y,&one,&sone,xq,&one));
  /* twice */
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,ctx->qK,&ld_,xp,&one,&szero,tp,&one));
  for (i=0;i<n;i++) tq[i] = *(ctx->qM+i)*xq[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,ctx->S,&lds_,tp,&one,&szero,c,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ctx->ld,&lds_,tq,&one,&sone,c,&one));
  for (i=0;i<k;i++) c[i] /= Omega[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,c,&one,&sone,xp,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ctx->ld,&lds_,c,&one,&sone,xq,&one));
  for (i=0;i<k;i++) y[i] += c[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARqKupdate"
static PetscErrorCode PEPSTOARqKupdate(PEP pep,PetscInt j,Vec *wv,PetscInt nwv)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscInt       i,ld=ctx->ld;
  PetscScalar    *qK=ctx->qK;
  Vec            vj,v1,v2;

  PetscFunctionBegin;
  if (!wv||nwv<2) {
    if (!wv) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",3);
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",4);
  }
  v1 = wv[0];
  v2 = wv[1];
  ierr = BVGetColumn(pep->V,j,&vj);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,0,vj,v1);CHKERRQ(ierr);
  ierr = BVRestoreColumn(pep->V,j,&vj);CHKERRQ(ierr);
  if (ctx->monic) {
    ierr = STMatSolve(pep->st,v1,v2);CHKERRQ(ierr);
    v1 = v2;
  }
  for (i=0;i<=j;i++) {
    ierr = BVGetColumn(pep->V,i,&vj);CHKERRQ(ierr);
    ierr = VecDot(v1,vj,qK+j*ld+i);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,i,&vj);CHKERRQ(ierr);
  }
  for (i=0;i<=j;i++) {
    qK[i+j*ld] = -PetscConj(qK[i+ld*j]);
    qK[j+i*ld] = qK[i+j*ld];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARrun"
/*
  Compute a run of Lanczos iterations
*/
static PetscErrorCode PEPSTOARrun(PEP pep,PetscReal *a,PetscReal *b,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscScalar *work,PetscInt nw,Vec *t_,PetscInt nwv)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscInt       i,j,m=*M,nwu=0,lwa;
  PetscInt       lds=ctx->d*ctx->ld,offq=ctx->ld;
  Vec            v=t_[0],t=t_[1],q=t_[2];
  PetscReal      norm;
  PetscScalar    *y,*S=ctx->S;

  PetscFunctionBegin;
  *breakdown = PETSC_FALSE; /* ----- */
  if (!t_||nwv<3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",12);
  lwa = (ctx->ld)*4;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
  y = work;
  nwu += ctx->ld;
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = BVSetActiveColumns(pep->V,0,j+2);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,v,S+j*lds);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,0,v,t);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,v,S+offq+j*lds);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,1,v,q);CHKERRQ(ierr);
    ierr = VecAXPY(t,1.0,q);CHKERRQ(ierr);
    ierr = STMatSolve(pep->st,t,q);CHKERRQ(ierr);
    ierr = VecScale(q,-1.0);CHKERRQ(ierr);

    /* orthogonalize */
    ierr = BVOrthogonalizeVec(pep->V,q,S+offq+(j+1)*lds,&norm,NULL);CHKERRQ(ierr);
    for (i=0;i<j+2;i++) *(S+offq+(j+1)*lds+i) *= *(ctx->qM+i);
    *(S+offq+(j+1)*lds+j+2) = norm;
    ierr = VecScale(q,1.0/norm);CHKERRQ(ierr);
    ierr = BVInsertVec(pep->V,j+2,q);CHKERRQ(ierr);
    for (i=0;i<=j+1;i++) *(S+(j+1)*lds+i) = *(S+offq+j*lds+i);
   
    /* Update qK and qM */
    *(ctx->qM+j+2) = (norm > 0)?1.0:-1.0;
    ierr = PEPSTOARqKupdate(pep,j+2,t_,2);CHKERRQ(ierr);

    /* Level-2 orthogonalization */
    ierr = PEPSTOAROrth2(pep,j+1,omega,y,work+nwu,lwa-nwu);CHKERRQ(ierr);
    a[j] = PetscRealPart(y[j])/omega[j];
    ierr = PEPSTOARNorm(pep,j+1,&norm,work+nwu,lwa-nwu);CHKERRQ(ierr);
    omega[j+1] = (norm > 0)?1.0:-1.0;
    for (i=0;i<=j+2;i++) {
      S[i+(j+1)*lds] /= norm;
      S[i+offq+(j+1)*lds] /= norm;
    }
    b[j] = PetscAbsReal(norm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IndefOrthog_CGS"
/*
  compute x = x - y*ss^{-1}*y^T*s*x where ss=y^T*s*y
  s diagonal (signature matrix)
*/
static PetscErrorCode IndefOrthog_CGS(PetscInt n,PetscReal *s,PetscInt nv,PetscScalar *Y,PetscInt ldy,PetscReal *ss,PetscScalar *x,PetscScalar *h,PetscScalar *work,PetscInt lw)
{
  PetscErrorCode ierr;
  PetscInt       i,nwall,nwu=0;
  PetscScalar    *h2,*h1,*t1,*t2,one=1.0,zero=0.0,onen=-1.0;
  PetscBLASInt   n_,nv_,ldy_,inc=1;

  PetscFunctionBegin;
  nwall = 3*n;
  if (!work || lw<nwall) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",11);
  t1 = work+nwu;
  nwu += n;
  t2 = work+nwu;
  nwu += n;
  h2 = work+nwu;
  nwu +=n;
  if (h) h1 = h;
  else h1 = h2;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nv,&nv_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldy,&ldy_);CHKERRQ(ierr);
  for (i=0;i<n;i++) t1[i] = s[i]*x[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&nv_,&one,Y,&ldy_,t1,&inc,&zero,t2,&inc));
  for (i=0;i<nv;i++) h1[i] = t2[i]/ss[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&nv_,&onen,Y,&ldy_,h1,&inc,&one,x,&inc));
  /* Repeat */
  for (i=0;i<n;i++) t1[i] = s[i]*x[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&nv_,&one,Y,&ldy_,t1,&inc,&zero,t2,&inc));
  for (i=0;i<nv;i++) h2[i] = t2[i]/ss[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&nv_,&onen,Y,&ldy_,h2,&inc,&one,x,&inc));
  if (h) {
    for (i=0;i<nv;i++) h[i] += h2[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IndefNorm"
/*
   normalization with a indefinite norm
*/
static PetscErrorCode IndefNorm(PetscInt n,PetscReal *s,PetscScalar *x,PetscReal *norm)
{
  PetscInt    i;
  PetscReal   r=0.0,t,max=0.0;
  PetscScalar c;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    t = PetscAbsScalar(x[i]);
    if (t > max) max = t;
  }
  for (i=0;i<n;i++) {
    c = x[i]/max;
    r += PetscRealPart(PetscConj(c)*c*s[i]);
  }
  if (r<0) r = -max*PetscSqrtReal(-r);
  else r = max*PetscSqrtReal(r);
  for (i=0;i<n;i++) x[i] /= r;
  if (norm) *norm = r;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARTrunc"
static PetscErrorCode PEPSTOARTrunc(PEP pep,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscInt nw,PetscReal *rwork,PetscInt nrw)
{
  PetscErrorCode  ierr;
  PEP_STOAR       *ctx = (PEP_STOAR*)pep->data;
  Mat             G;
  PetscInt        lwa,nwu=0,lrwa,nrwu=0;
  PetscInt        j,i,n,lds=2*ctx->ld;
  PetscScalar     *M,*V,*U,*S=ctx->S,*R=NULL,sone=1.0,zero=0.0,t;
  PetscReal       *sg,*qM=ctx->qM,*ss=NULL,norm;
  PetscBLASInt    cs1_,rs1_,cs1t2,cs1p1,n_,info,lw_,one=1,lds_,ld_;
  const PetscBool ismonic=ctx->monic;

  PetscFunctionBegin;
  n = (rs1>2*cs1)?2*cs1:rs1;
  lwa = cs1*rs1*4+n*(rs1+2*cs1)+(cs1+1)*(cs1+2);
  lrwa = n+cs1+1+5*n;
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
  U = work+nwu;
  nwu += rs1*n;
  V = work+nwu;
  nwu += 2*cs1*n;
  sg = rwork+nrwu;
  nrwu += n;
  for (i=0;i<cs1;i++) {
    ierr = PetscMemcpy(M+i*rs1,S+i*lds,rs1*sizeof(PetscScalar));CHKERRQ(ierr);  
    ierr = PetscMemcpy(M+(i+cs1)*rs1,S+i*lds+ctx->ld,rs1*sizeof(PetscScalar));CHKERRQ(ierr);  
  }
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*2,&cs1t2);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1+1,&cs1p1);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1t2,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1t2,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));  
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  if (!ismonic) {
    R = work+nwu;
    ierr = PetscMemzero(R,(cs1+1)*(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
    nwu += (cs1+1)*(cs1+1);
    nwu += cs1+1;
    ss = rwork+nrwu;
    nrwu += cs1+1;
    for (j=0;j<cs1+1;j++) {
      ierr = IndefOrthog_CGS(rs1,qM,j,U,rs1,ss,U+j*rs1,R+j*(cs1+1),work+nwu,lwa-nwu);CHKERRQ(ierr);
      ierr = IndefNorm(rs1,qM,U+j*rs1,&norm);CHKERRQ(ierr);
      ss[j] = (norm>0)?1.0:-1.0;
      R[j+j*(cs1+1)] = norm;
    }
  }

  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs1,2*cs1,U,&G);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,rs1);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,G,0,cs1+1);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  
  /* Update S */
  ierr = PetscMemzero(S,lds*ctx->ld*sizeof(PetscScalar));CHKERRQ(ierr);
  if (ismonic) {
    for (i=0;i<cs1+1;i++) {
      t = sg[i];
      PetscStackCallBLAS("BLASscal",BLASscal_(&cs1t2,&t,V+i,&n_));
    }
    for (i=0;i<cs1;i++) {
      ierr = PetscMemcpy(S+i*lds,V+i*n,(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscMemcpy(S+ctx->ld+i*lds,V+(cs1+i)*n,(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    for (i=0;i<cs1+1;i++) {
      t = sg[i];
      PetscStackCallBLAS("BLASscal",BLASscal_(&cs1p1,&t,R+i*(cs1+1),&one));
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&cs1p1,&cs1_,&cs1p1,&sone,R,&cs1p1,V,&n_,&zero,S,&lds_));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&cs1p1,&cs1_,&cs1p1,&sone,R,&cs1p1,V+cs1*n,&n_,&zero,S+ctx->ld,&lds_));
  }
  /* Update qM and qK */
  for (j=0;j<cs1+1;j++) qM[j] = ismonic? 1.0: ss[j];
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&rs1_,&sone,ctx->qK,&ld_,U,&rs1_,&zero,work+nwu,&rs1_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&cs1p1,&cs1p1,&rs1_,&sone,U,&rs1_,work+nwu,&rs1_,&zero,ctx->qK,&ld_));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARSupdate"
/*
  S <- S*Q 
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
*/
static PetscErrorCode PEPSTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work,PetscInt nw)
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
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S,&lds_,Q,&ldq_,&b,work,&sr_));
  for (j=0;j<ncu;j++) {
    ierr = PetscMemcpy(S+lds*(s+j),work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S+ld,&lds_,Q,&ldq_,&b,work,&sr_));
  for (j=0;j<ncu;j++) {
    ierr = PetscMemcpy(S+lds*(s+j)+ld,work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_STOAR"
PetscErrorCode PEPSolve_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscInt       i,j,k,l,nv=0,ld=ctx->ld,lds=ctx->d*ctx->ld,off,ldds,t;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0;
  Vec            vomega,w=pep->work[0],w2=pep->work[1];
  PetscScalar    *S=ctx->S,*Q,*work,*aux;
  PetscReal      beta,norm,t1,t2,*omega,*a,*b,*r,*qM=ctx->qM,*rwork;
  PetscBool      breakdown;
  Mat            M,G;

  PetscFunctionBegin;
  ierr = STGetTOperators(pep->st,1,&M);CHKERRQ(ierr);
  ierr = MatScale(M,1.0/pep->sfactor);CHKERRQ(ierr);
  ierr = STGetTOperators(pep->st,2,&M);CHKERRQ(ierr);
  ierr = MatScale(M,1.0/(pep->sfactor*pep->sfactor));CHKERRQ(ierr);
  if (ctx->monic) {
    ierr = BVSetMatrix(pep->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVSetMatrix(pep->V,M,PETSC_TRUE);CHKERRQ(ierr);
  }
  lwa = 9*ld*ld+5*ld;
  lrwa = 8*ld;
  ierr = PetscMalloc2(lwa,&work,lrwa,&rwork);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  if (pep->nini==0) {  
    ierr = BVSetRandomColumn(pep->V,0,pep->rand);CHKERRQ(ierr);
  }
  ierr = BVSetRandomColumn(pep->V,1,pep->rand);CHKERRQ(ierr);
  ierr = BVOrthogonalizeColumn(pep->V,0,NULL,&norm,NULL);CHKERRQ(ierr);
  ierr = BVScaleColumn(pep->V,0,1.0/norm);CHKERRQ(ierr);
  qM[0] = (norm>0)?1.0:-1.0;
  ierr = PEPSTOARqKupdate(pep,0,pep->work,2);CHKERRQ(ierr);
  S[0] = norm;
  ierr = BVOrthogonalizeColumn(pep->V,1,S+ld,&norm,NULL);CHKERRQ(ierr);
  *(S+ld) *= *(ctx->qM);
  ierr = BVScaleColumn(pep->V,1,1.0/norm);CHKERRQ(ierr);
  qM[1] = (norm>0)?1.0:-1.0;
  ierr = PEPSTOARqKupdate(pep,1,pep->work,2);CHKERRQ(ierr);
  S[1+ld] = norm;
  if (PetscAbsReal(norm)<PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)pep),1,"Problem with initial vector");
  ierr = PEPSTOARNorm(pep,0,&norm,work+nwu,lwa-nwu);CHKERRQ(ierr);
  ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = (norm > 0)?1.0:-1.0;
  ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  for (j=0;j<2;j++) {
    S[j] /= norm;
    S[j+ld] /= norm;
  }

  /* Restart loop */
  l = 0;
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;
    ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a+ldds;
    r = b+ldds;
    ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(pep->nconv+pep->mpd,pep->ncv);
    ierr = PEPSTOARrun(pep,a,b,omega,pep->nconv+l,&nv,&breakdown,work+nwu,lwa-nwu,pep->work,3);CHKERRQ(ierr);
    beta = b[nv-1];
    ierr = DSRestoreArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = DSSetDimensions(pep->ds,nv,0,pep->nconv,pep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(pep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

    /* Check convergence */
    ierr = BVSetActiveColumns(pep->V,0,nv+2);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,w,S+nv*lds);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&t1);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,0,w,w2);CHKERRQ(ierr);
    if (ctx->monic) {
      ierr = STMatSolve(pep->st,w2,w);CHKERRQ(ierr);
      ierr = VecNorm(w,NORM_2,&t2);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(w2,NORM_2,&t2);CHKERRQ(ierr);
    }
    ierr = BVMultVec(pep->V,1.0,0.0,w,S+ld+nv*lds);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
    t1 = SlepcAbs(norm,t1);
    if (!ctx->monic) {
      ierr = STMatMult(pep->st,2,w,w2);CHKERRQ(ierr);
      ierr = VecNorm(w2,NORM_2,&norm);CHKERRQ(ierr);
    }
    t2 = SlepcAbs(norm,t2);
    norm = PetscMax(t1,t2);
    ierr = DSGetDimensions(pep->ds,NULL,NULL,NULL,NULL,&t);CHKERRQ(ierr);    
    ierr = PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,t-pep->nconv,beta*norm,&k);CHKERRQ(ierr);
    if (pep->its >= pep->max_it) pep->reason = PEP_DIVERGED_ITS;
    if (k >= pep->nev) pep->reason = PEP_CONVERGED_TOL;

    /* Update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else { 
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      l = PetscMin(l,t);
      ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      if (*(a+ldds+k+l-1)!=0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
      ierr = DSRestoreArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    }
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */

    /* Update S */
    off = pep->nconv*ldds;
    ierr = DSGetArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = PEPSTOARSupdate(S,ld,nv+2,pep->nconv,k+l-pep->nconv,nv,Q+off,ldds,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (pep->reason == PEP_CONVERGED_ITERATING) {
      if (breakdown) {

        /* Stop if breakdown */
        ierr = PetscInfo2(pep,"Breakdown STOAR method (it=%D norm=%g)\n",pep->its,(double)beta);CHKERRQ(ierr);
        pep->reason = PEP_DIVERGED_BREAKDOWN;
      } else {

        /* Truncate S */
        ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        ierr = PEPSTOARTrunc(pep,nv+2,k+l+1,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);

        ierr = VecCreateSeq(PETSC_COMM_SELF,k+l+2,&vomega);CHKERRQ(ierr);
        ierr = VecGetArray(vomega,&aux);CHKERRQ(ierr);
        for (i=0;i<=k+l+1;i++) aux[i] = ctx->qM[i];
        ierr = VecRestoreArray(vomega,&aux);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(pep->V,0,k+l+2);CHKERRQ(ierr);
        ierr = BVSetSignature(pep->V,vomega);CHKERRQ(ierr);
        ierr = VecDestroy(&vomega);CHKERRQ(ierr);

        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        r = a + 2*ldds;
        for (j=k;j<k+l;j++) {
          r[j] = PetscRealPart(Q[nv-1+j*ldds]*beta);
        }
        b = a+ldds;
        b[k+l-1] = r[k+l-1];
        omega[k+l] = omega[nv];
        ierr = DSRestoreArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      }
    }
    pep->nconv = k;
    ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }

  /* Update vectors V = V*S */    
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nv+2,pep->nconv,NULL,&G);CHKERRQ(ierr);
  ierr = MatDenseGetArray(G,&aux);CHKERRQ(ierr);
  for (j=0;j<pep->nconv;j++) {
    ierr = PetscMemcpy(aux+j*(nv+2),S+j*lds,(nv+2)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(G,&aux);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,nv+2);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,G,0,pep->nconv);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  for (j=0;j<pep->nconv;j++) {
    pep->eigr[j] *= pep->sfactor;
    pep->eigi[j] *= pep->sfactor;
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);

  /* scale back matrices */
  ierr = STGetTOperators(pep->st,1,&M);CHKERRQ(ierr);
  ierr = MatScale(M,pep->sfactor);CHKERRQ(ierr);
  ierr = STGetTOperators(pep->st,2,&M);CHKERRQ(ierr);
  ierr = MatScale(M,pep->sfactor*pep->sfactor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_STOAR"
PetscErrorCode PEPSetFromOptions_STOAR(PetscOptions *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg,val,lock;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP STOAR Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pep_stoar_monic","Use monic variant of STOAR","PEPSTOARSetMonic",ctx->monic,&val,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPSTOARSetMonic(pep,val);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-pep_stoar_locking","Choose between locking and non-locking variants","PEPSTOARSetLocking",PETSC_FALSE,&lock,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPSTOARSetLocking(pep,lock);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARSetLocking_STOAR"
static PetscErrorCode PEPSTOARSetLocking_STOAR(PEP pep,PetscBool lock)
{
  PEP_STOAR *ctx = (PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARSetLocking"
/*@
   PEPSTOARSetLocking - Choose between locking and non-locking variants of
   the STOAR method.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -pep_stoar_locking - Sets the locking flag

   Notes:
   The default is to keep all directions in the working subspace even if
   already converged to working accuracy (the non-locking variant).
   This behaviour can be changed so that converged eigenpairs are locked
   when the method restarts.

   Note that the default behaviour is the opposite to Krylov solvers in EPS.

   Level: advanced

.seealso: PEPSTOARGetLocking()
@*/
PetscErrorCode PEPSTOARSetLocking(PEP pep,PetscBool lock)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,lock,2);
  ierr = PetscTryMethod(pep,"PEPSTOARSetLocking_C",(PEP,PetscBool),(pep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARGetLocking_STOAR"
static PetscErrorCode PEPSTOARGetLocking_STOAR(PEP pep,PetscBool *lock)
{
  PEP_STOAR *ctx = (PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARGetLocking"
/*@
   PEPSTOARGetLocking - Gets the locking flag used in the STOAR method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: PEPSTOARSetLocking()
@*/
PetscErrorCode PEPSTOARGetLocking(PEP pep,PetscBool *lock)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(lock,2);
  ierr = PetscTryMethod(pep,"PEPSTOARGetLocking_C",(PEP,PetscBool*),(pep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARSetMonic_STOAR"
static PetscErrorCode PEPSTOARSetMonic_STOAR(PEP pep,PetscBool monic)
{
  PEP_STOAR *ctx = (PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  ctx->monic = monic;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARSetMonic"
/*@
   PEPSTOARSetMonic - Set the monic variant of the STOAR solver.

   Logically Collective on PEP

   Input Parameters:
+  pep   - polynomial eigenvalue solver
-  monic - boolean flag to set the monic variant

   Options Database Key:
.  -pep_stoar_monic <boolean> - Indicates the boolean flag

   Note:
   The monic variant can be used only if the coefficient matrices
   after the spectral transformation, M_sigma, C_sigma and K_sigma,
   satisfy that M_sigma commutes with the other two. In this case,
   the solver implicitly transforms the problem to use a monic
   polynomial by multiplying with inv(M_sigma).

   Level: advanced

.seealso: PEPSTOARGetMonic()
@*/
PetscErrorCode PEPSTOARSetMonic(PEP pep,PetscBool monic)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,monic,2);
  ierr = PetscTryMethod(pep,"PEPSTOARSetMonic_C",(PEP,PetscBool),(pep,monic));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARGetMonic_STOAR"
static PetscErrorCode PEPSTOARGetMonic_STOAR(PEP pep,PetscBool *monic)
{
  PEP_STOAR *ctx = (PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  *monic = ctx->monic;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARGetMonic"
/*@
   PEPSTOARGetMonic - Returns the flag indicating that the monic variant
   is being used.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameter:
.  monic - the flag

   Level: advanced

.seealso: PEPSTOARSetMonic()
@*/
PetscErrorCode PEPSTOARGetMonic(PEP pep,PetscBool *monic)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(monic,2);
  ierr = PetscTryMethod(pep,"PEPSTOARGetMonic_C",(PEP,PetscBool*),(pep,monic));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_STOAR"
PetscErrorCode PEPView_STOAR(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (ctx->monic) {
      ierr = PetscViewerASCIIPrintf(viewer,"  STOAR: using the monic variant\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  STOAR: using the %slocking variant\n",ctx->lock?"":"non-");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_STOAR"
PetscErrorCode PEPDestroy_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  ierr = PetscFree3(ctx->S,ctx->qM,ctx->qK);CHKERRQ(ierr);
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetMonic_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetMonic_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetLocking_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_STOAR"
PETSC_EXTERN PetscErrorCode PEPCreate_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_STOAR      *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  pep->data = (void*)ctx;
  ctx->lock = PETSC_FALSE;

  pep->ops->solve          = PEPSolve_STOAR;
  pep->ops->setup          = PEPSetUp_STOAR;
  pep->ops->setfromoptions = PEPSetFromOptions_STOAR;
  pep->ops->view           = PEPView_STOAR;
  pep->ops->destroy        = PEPDestroy_STOAR;
  pep->ops->computevectors = PEPComputeVectors_Indefinite;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetMonic_C",PEPSTOARSetMonic_STOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetMonic_C",PEPSTOARGetMonic_STOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetLocking_C",PEPSTOARSetLocking_STOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetLocking_C",PEPSTOARGetLocking_STOAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

