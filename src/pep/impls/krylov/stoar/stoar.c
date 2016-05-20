/*

   SLEPc polynomial eigensolver: "stoar"

   Method: S-TOAR

   Algorithm:

       Symmetric Two-Level Orthogonal Arnoldi.

   References:

       [1] C. Campos and J.E. Roman, "Restarted Q-Arnoldi-type methods
           exploiting symmetry in quadratic eigenvalue problems", BIT
           Numer. Math. (in press), 2016.

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

#include <slepc/private/pepimpl.h>         /*I "slepcpep.h" I*/
#include "../src/pep/impls/krylov/pepkrylov.h"
#include <slepcblaslapack.h>

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@Article{slepc-stoar,\n"
  "   author = \"C. Campos and J. E. Roman\",\n"
  "   title = \"Restarted {Q-Arnoldi-type} methods exploiting symmetry in quadratic eigenvalue problems\",\n"
  "   journal = \"{BIT} Numer. Math.\",\n"
  "   volume = \"to appear\",\n"
  "   number = \"\",\n"
  "   pages = \"\",\n"
  "   year = \"2016,\"\n"
  "   doi = \"http://dx.doi.org/10.1007/s10543-016-0601-5\"\n"
  "}\n";

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARNorm"
/*
  Compute B-norm of v=[v1;v2] whith  B=diag(-pep->T[0],pep->T[2])
*/
static PetscErrorCode PEPSTOARNorm(PEP pep,PetscInt j,PetscReal *norm)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBLASInt   n_,one=1,ld_;
  PetscScalar    sone=1.0,szero=0.0,*sp,*sq,*w1,*w2,*qK,*qM;
  PetscInt       n,i,lds=ctx->d*ctx->ld;

  PetscFunctionBegin;
  qK = ctx->qB;
  qM = ctx->qB+ctx->ld*ctx->ld;
  n = j+2;
  ierr = PetscMalloc2(n,&w1,n,&w2);CHKERRQ(ierr);
  sp = ctx->S+lds*j;
  sq = sp+ctx->ld;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qK,&ld_,sp,&one,&szero,w1,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qM,&ld_,sq,&one,&szero,w2,&one));
  *norm = 0.0;
  for (i=0;i<n;i++) *norm += PetscRealPart(w1[i]*PetscConj(sp[i])+w2[i]*PetscConj(sq[i]));
  *norm = (*norm>0.0)?PetscSqrtReal(*norm):-PetscSqrtReal(-*norm);
  ierr = PetscFree2(w1,w2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARqKqMupdates"
static PetscErrorCode PEPSTOARqKqMupdates(PEP pep,PetscInt j,Vec *wv)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       i,ld=ctx->ld;
  PetscScalar    *qK,*qM;
  Vec            vj,v1,v2;

  PetscFunctionBegin;
  qK = ctx->qB;
  qM = ctx->qB+ctx->ld*ctx->ld;
  v1 = wv[0];
  v2 = wv[1];
  ierr = BVGetColumn(pep->V,j,&vj);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,0,vj,v1);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,2,vj,v2);CHKERRQ(ierr);
  ierr = BVRestoreColumn(pep->V,j,&vj);CHKERRQ(ierr);
  for (i=0;i<=j;i++) {
    ierr = BVGetColumn(pep->V,i,&vj);CHKERRQ(ierr);
    ierr = VecDot(v1,vj,qK+j*ld+i);CHKERRQ(ierr);
    ierr = VecDot(v2,vj,qM+j*ld+i);CHKERRQ(ierr);
    *(qM+j*ld+i) *= pep->sfactor*pep->sfactor;
    ierr = BVRestoreColumn(pep->V,i,&vj);CHKERRQ(ierr);
  }
  for (i=0;i<j;i++) {
    qK[i+j*ld] = -qK[i+ld*j];
    qK[j+i*ld] = PetscConj(qK[i+j*ld]);
    qM[j+i*ld] = PetscConj(qM[i+j*ld]);
  }
  qK[j+j*ld] = -PetscRealPart(qK[j+ld*j]);
  qM[j+j*ld] = PetscRealPart(qM[j+ld*j]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_STOAR"
PetscErrorCode PEPSetUp_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      shift,sinv,flg,lindep;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       ld,i;
  PetscReal      norm,*omega;

  PetscFunctionBegin;
  pep->lineariz = PETSC_TRUE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!ctx->lock && pep->mpd<pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
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
  ierr = PetscCalloc1(ctx->d*ld*ld,&ctx->S);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*ld*ld,&ctx->qB);CHKERRQ(ierr);

  /* process starting vector */
  if (pep->nini>-2) {
    ierr = BVSetRandomColumn(pep->V,0);CHKERRQ(ierr);
    ierr = BVSetRandomColumn(pep->V,1);CHKERRQ(ierr);
  } else {
    ierr = BVInsertVec(pep->V,0,pep->IS[0]);CHKERRQ(ierr);
    ierr = BVInsertVec(pep->V,1,pep->IS[1]);CHKERRQ(ierr);
  }
  ierr = BVOrthogonalizeColumn(pep->V,0,NULL,&norm,&lindep);CHKERRQ(ierr);
  if (!lindep) {
    ierr = BVScaleColumn(pep->V,0,1.0/norm);CHKERRQ(ierr);
    ctx->S[0] = norm;
    ierr = PEPSTOARqKqMupdates(pep,0,pep->work);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)pep),1,"Problem with initial vector");
  ierr = BVOrthogonalizeColumn(pep->V,1,ctx->S+ld,&norm,&lindep);CHKERRQ(ierr);
  if (!lindep) {
    ierr = BVScaleColumn(pep->V,1,1.0/norm);CHKERRQ(ierr);
    ctx->S[1] = norm;
    ierr = PEPSTOARqKqMupdates(pep,1,pep->work);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)pep),1,"Problem with initial vector");

  ierr = PEPSTOARNorm(pep,0,&norm);CHKERRQ(ierr);
  for (i=0;i<2;i++) { ctx->S[i+ld] /= norm; ctx->S[i] /= norm; }
  ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = (norm>0)?1.0:-1.0;
  ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  if (pep->nini<0) {
    ierr = SlepcBasisDestroy_Private(&pep->nini,&pep->IS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOAROrth2"
/*
  Computes GS orthogonalization  x = [z;x] - [Sp;Sq]*y,
  where y = Omega\([Sp;Sq]'*[qK zeros(size(qK,1)) ;zeros(size(qK,1)) qM]*[z;x]).
  n: Column from S to be orthogonalized against previous columns.
*/
static PetscErrorCode PEPSTOAROrth2(PEP pep,PetscInt k,PetscReal *Omega,PetscScalar *y)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBLASInt   n_,lds_,k_,one=1,ld_;
  PetscScalar    *S=ctx->S,sonem=-1.0,sone=1.0,szero=0.0,*tp,*tq,*xp,*xq,*c,*qK,*qM;
  PetscInt       i,lds=ctx->d*ctx->ld,n,j;

  PetscFunctionBegin;
  qK = ctx->qB;
  qM = ctx->qB+ctx->ld*ctx->ld;
  n = k+2;
  ierr = PetscMalloc3(n,&tp,n,&tq,k,&c);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr); /* Size of qK and qM */
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against */
  xp = S+k*lds;
  xq = S+ctx->ld+k*lds;
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qK,&ld_,xp,&one,&szero,tp,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qM,&ld_,xq,&one,&szero,tq,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,ctx->S,&lds_,tp,&one,&szero,y,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ctx->ld,&lds_,tq,&one,&sone,y,&one));
  for (i=0;i<k;i++) y[i] /= Omega[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,y,&one,&sone,xp,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ctx->ld,&lds_,y,&one,&sone,xq,&one));
  /* three times */
  for (j=0;j<2;j++) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qK,&ld_,xp,&one,&szero,tp,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qM,&ld_,xq,&one,&szero,tq,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,ctx->S,&lds_,tp,&one,&szero,c,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ctx->ld,&lds_,tq,&one,&sone,c,&one));
    for (i=0;i<k;i++) c[i] /= Omega[i];
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,c,&one,&sone,xp,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ctx->ld,&lds_,c,&one,&sone,xq,&one));
    for (i=0;i<k;i++) y[i] += c[i];
  }
  ierr = PetscFree3(tp,tq,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARrun"
/*
  Compute a run of Lanczos iterations. dim(work)=(ctx->ld)*4
*/
static PetscErrorCode PEPSTOARrun(PEP pep,PetscReal *a,PetscReal *b,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscBool *symmlost,PetscScalar *work,Vec *t_)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       i,j,m=*M,l;
  PetscInt       lds=ctx->d*ctx->ld,offq=ctx->ld;
  Vec            v=t_[0],t=t_[1],q=t_[2];
  PetscReal      norm,sym=0.0,fro=0.0,*f;
  PetscScalar    *y,*S=ctx->S;
  PetscBLASInt   j_,one=1;
  PetscBool      lindep;

  PetscFunctionBegin;
  *breakdown = PETSC_FALSE; /* ----- */
  ierr = DSGetDimensions(pep->ds,NULL,NULL,&l,NULL,NULL);CHKERRQ(ierr);
  y = work;
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = BVSetActiveColumns(pep->V,0,j+2);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,v,S+j*lds);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,0,v,t);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,v,S+offq+j*lds);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,1,v,q);CHKERRQ(ierr);
    ierr = VecAXPY(t,pep->sfactor,q);CHKERRQ(ierr);
    ierr = STMatSolve(pep->st,t,q);CHKERRQ(ierr);
    ierr = VecScale(q,-1.0/(pep->sfactor*pep->sfactor));CHKERRQ(ierr);

    /* orthogonalize */
    ierr = BVOrthogonalizeVec(pep->V,q,S+offq+(j+1)*lds,&norm,&lindep);CHKERRQ(ierr);
    if (lindep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"STOAR does not support detection of linearly dependent TOAR vectors");
    *(S+offq+(j+1)*lds+j+2) = norm;
    ierr = VecScale(q,1.0/norm);CHKERRQ(ierr);
    ierr = BVInsertVec(pep->V,j+2,q);CHKERRQ(ierr);
    for (i=0;i<=j+1;i++) *(S+(j+1)*lds+i) = *(S+offq+j*lds+i);

    /* update qK and qM */
    ierr = PEPSTOARqKqMupdates(pep,j+2,t_);CHKERRQ(ierr);

    /* level-2 orthogonalization */
    ierr = PEPSTOAROrth2(pep,j+1,omega,y);CHKERRQ(ierr);
    a[j] = PetscRealPart(y[j])/omega[j];
    ierr = PEPSTOARNorm(pep,j+1,&norm);CHKERRQ(ierr);
    omega[j+1] = (norm > 0)?1.0:-1.0;
    for (i=0;i<=j+2;i++) {
      S[i+(j+1)*lds] /= norm;
      S[i+offq+(j+1)*lds] /= norm;
    }
    b[j] = PetscAbsReal(norm);

    /* check symmetry */
    ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&f);CHKERRQ(ierr);
    if (j==k) {
      for (i=l;i<j-1;i++) y[i] = PetscAbsScalar(y[i])-PetscAbsReal(f[2*ctx->ld+i]);
      for (i=0;i<l;i++) y[i] = 0.0;
    }
    ierr = DSRestoreArrayReal(pep->ds,DS_MAT_T,&f);CHKERRQ(ierr);
    if (j>0) y[j-1] = PetscAbsScalar(y[j-1])-PetscAbsReal(b[j-1]);
    ierr = PetscBLASIntCast(j,&j_);CHKERRQ(ierr);
    sym = SlepcAbs(BLASnrm2_(&j_,y,&one),sym);
    fro = SlepcAbs(fro,SlepcAbs(a[j],b[j]));
    if (j>0) fro = SlepcAbs(fro,b[j-1]);
    if (sym/fro>PetscMax(PETSC_SQRT_MACHINE_EPSILON,10*pep->tol)) {
      *symmlost = PETSC_TRUE;
      *M=j+1;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARTrunc"
static PetscErrorCode PEPSTOARTrunc(PEP pep,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscReal *rwork)
{
#if defined(PETSC_MISSING_LAPACK_GESVD)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  Mat            G;
  PetscInt       lwa,nwu=0,nrwu=0;
  PetscInt       i,n,lds=2*ctx->ld;
  PetscScalar    *M,*V,*U,*S=ctx->S,sone=1.0,zero=0.0,t,*qK,*qM;
  PetscReal      *sg;
  PetscBLASInt   cs1_,rs1_,cs1t2,cs1p1,n_,info,lw_,lds_,ld_;

  PetscFunctionBegin;
  qK = ctx->qB;
  qM = ctx->qB+ctx->ld*ctx->ld;
  n = (rs1>2*cs1)?2*cs1:rs1;
  lwa = cs1*rs1*4+n*(rs1+2*cs1)+(cs1+1)*(cs1+2);
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

  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs1,2*cs1,U,&G);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,rs1);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,G,0,cs1+1);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);

  /* Update S */
  ierr = PetscMemzero(S,lds*ctx->ld*sizeof(PetscScalar));CHKERRQ(ierr);

  for (i=0;i<cs1+1;i++) {
    t = sg[i];
    PetscStackCallBLAS("BLASscal",BLASscal_(&cs1t2,&t,V+i,&n_));
  }
  for (i=0;i<cs1;i++) {
    ierr = PetscMemcpy(S+i*lds,V+i*n,(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemcpy(S+ctx->ld+i*lds,V+(cs1+i)*n,(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* Update qM and qK */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&rs1_,&sone,qK,&ld_,U,&rs1_,&zero,work+nwu,&rs1_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&cs1p1,&cs1p1,&rs1_,&sone,U,&rs1_,work+nwu,&rs1_,&zero,qK,&ld_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&rs1_,&sone,qM,&ld_,U,&rs1_,&zero,work+nwu,&rs1_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&cs1p1,&cs1p1,&rs1_,&sone,U,&rs1_,work+nwu,&rs1_,&zero,qM,&ld_));
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PEPSTOARSupdate"
/*
  S <- S*Q
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
  dim(work)=sr*ncu;
*/
static PetscErrorCode PEPSTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       j,lds=2*ld;

  PetscFunctionBegin;
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

#if 0
#undef __FUNCT__
#define __FUNCT__ "PEPSTOARpreKConvergence"
static PetscErrorCode PEPSTOARpreKConvergence(PEP pep,PetscInt nv,PetscReal *norm,Vec *w)
{
  PetscErrorCode ierr;
  PEP_TOAR      *ctx = (PEP_TOAR*)pep->data;
  PetscBLASInt   n_,one=1;
  PetscInt       lds=2*ctx->ld;
  PetscReal      t1,t2;
  PetscScalar    *S=ctx->S;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(nv+2,&n_);CHKERRQ(ierr);
  t1 = BLASnrm2_(&n_,S+nv*2*ctx->ld,&one);
  t2 = BLASnrm2_(&n_,S+(nv*2+1)*ctx->ld,&one);
  *norm = SlepcAbs(t1,t2);
  ierr = BVSetActiveColumns(pep->V,0,nv+2);CHKERRQ(ierr);
  ierr = BVMultVec(pep->V,1.0,0.0,w[1],S+nv*lds);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,0,w[1],w[2]);CHKERRQ(ierr);
  ierr = VecNorm(w[2],NORM_2,&t1);CHKERRQ(ierr);
  ierr = BVMultVec(pep->V,1.0,0.0,w[1],S+ctx->ld+nv*lds);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,2,w[1],w[2]);CHKERRQ(ierr);
  ierr = VecNorm(w[2],NORM_2,&t2);CHKERRQ(ierr);
  t2 *= pep->sfactor*pep->sfactor;
  *norm = PetscMax(*norm,SlepcAbs(t1,t2));
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_STOAR"
PetscErrorCode PEPSolve_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       j,k,l,nv=0,ld=ctx->ld,lds=ctx->d*ctx->ld,off,ldds,t;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0,nconv=0;
  PetscScalar    *S=ctx->S,*Q,*work;
  PetscReal      beta,norm=1.0,*omega,*a,*b,*r,*rwork;
  PetscBool      breakdown,symmlost=PETSC_FALSE,sinv;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  ierr = BVSetMatrix(pep->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
  lwa = 9*ld*ld+5*ld;
  lrwa = 8*ld;
  ierr = PetscMalloc2(lwa,&work,lrwa,&rwork);CHKERRQ(ierr); /* REVIEW */
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
  ierr = RGPushScale(pep->rg,sinv?pep->sfactor:1.0/pep->sfactor);CHKERRQ(ierr);
  ierr = STScaleShift(pep->st,sinv?pep->sfactor:1.0/pep->sfactor);CHKERRQ(ierr);

  /* Restart loop */
  l = 0;
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;
    ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a+ldds;
    ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(pep->nconv+pep->mpd,pep->ncv);
    ierr = PEPSTOARrun(pep,a,b,omega,pep->nconv+l,&nv,&breakdown,&symmlost,work+nwu,pep->work);CHKERRQ(ierr);
    beta = b[nv-1];
    if (symmlost) {
      pep->reason = PEP_DIVERGED_SYMMETRY_LOST;
      if (nv==pep->nconv+l+1) { pep->nconv = nconv; break; }
    }
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
    /* ierr = PEPSTOARpreKConvergence(pep,nv,&norm,pep->work);CHKERRQ(ierr);*/
    norm = 1.0;
    ierr = DSGetDimensions(pep->ds,NULL,NULL,NULL,NULL,&t);CHKERRQ(ierr);
    ierr = PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,t-pep->nconv,PetscAbsReal(beta)*norm,&k);CHKERRQ(ierr);
    nconv = k;
    ierr = (*pep->stopping)(pep,pep->its,pep->max_it,k,pep->nev,&pep->reason,pep->stoppingctx);CHKERRQ(ierr);

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
    ierr = PEPSTOARSupdate(S,ld,nv+2,pep->nconv,k+l-pep->nconv,nv,Q+off,ldds,work+nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (pep->reason == PEP_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Stop if breakdown */
        ierr = PetscInfo2(pep,"Breakdown STOAR method (it=%D norm=%g)\n",pep->its,(double)beta);CHKERRQ(ierr);
        pep->reason = PEP_DIVERGED_BREAKDOWN;
      } else {
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
        /* Truncate S */
        ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        ierr = PEPSTOARTrunc(pep,nv+2,k+l+1,work+nwu,rwork+nrwu);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      }
    }


    pep->nconv = k;
    ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }

  if (pep->nconv>0) {
    /* Truncate S */
    ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = PEPSTOARTrunc(pep,nv+2,pep->nconv,work+nwu,rwork+nrwu);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);

    /* Extraction */
    ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);

    for (j=0;j<pep->nconv;j++) {
      pep->eigr[j] *= pep->sfactor;
      pep->eigi[j] *= pep->sfactor;
    }
  }
  ierr = STScaleShift(pep->st,sinv?1.0/pep->sfactor:pep->sfactor);CHKERRQ(ierr);
  ierr = RGPopScale(pep->rg);CHKERRQ(ierr);

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_STOAR"
PetscErrorCode PEPSetFromOptions_STOAR(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg,lock;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP STOAR Options");CHKERRQ(ierr);
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
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

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
   The default is to lock converged eigenpairs when the method restarts.
   This behaviour can be changed so that all directions are kept in the
   working subspace even if already converged to working accuracy (the
   non-locking variant).

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
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

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
  ierr = PetscUseMethod(pep,"PEPSTOARGetLocking_C",(PEP,PetscBool*),(pep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_STOAR"
PetscErrorCode PEPView_STOAR(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_TOAR      *ctx = (PEP_TOAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  STOAR: using the %slocking variant\n",ctx->lock?"":"non-");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_STOAR"
PetscErrorCode PEPDestroy_STOAR(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetLocking_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_STOAR"
PETSC_EXTERN PetscErrorCode PEPCreate_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_TOAR      *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  pep->data = (void*)ctx;
  ctx->lock = PETSC_TRUE;

  pep->ops->solve          = PEPSolve_STOAR;
  pep->ops->setup          = PEPSetUp_STOAR;
  pep->ops->setfromoptions = PEPSetFromOptions_STOAR;
  pep->ops->view           = PEPView_STOAR;
  pep->ops->destroy        = PEPDestroy_STOAR;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_TOAR;
  pep->ops->reset          = PEPReset_TOAR;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetLocking_C",PEPSTOARSetLocking_STOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetLocking_C",PEPSTOARGetLocking_STOAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

