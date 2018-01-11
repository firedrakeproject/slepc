/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc polynomial eigensolver: "stoar"

   Method: S-TOAR

   Algorithm:

       Symmetric Two-Level Orthogonal Arnoldi.

   References:

       [1] C. Campos and J.E. Roman, "Restarted Q-Arnoldi-type methods
           exploiting symmetry in quadratic eigenvalue problems", BIT
           Numer. Math. (in press), 2016.
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
  "   volume = \"56\",\n"
  "   number = \"4\",\n"
  "   pages = \"1213--1236\",\n"
  "   year = \"2016,\"\n"
  "   doi = \"https://doi.org/10.1007/s10543-016-0601-5\"\n"
  "}\n";

/*
  Compute B-norm of v=[v1;v2] whith  B=diag(-pep->T[0],pep->T[2])
*/
static PetscErrorCode PEPSTOARNorm(PEP pep,PetscInt j,PetscReal *norm)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBLASInt   n_,one=1,ld_;
  PetscScalar    sone=1.0,szero=0.0,*sp,*sq,*w1,*w2,*qK,*qM,*S;
  PetscInt       n,i,lds,ld,d;
  Mat            MS;

  PetscFunctionBegin;
  ierr = BVGetSizes(pep->V,NULL,NULL,&ld);CHKERRQ(ierr);
  ierr = BVTensorGetDegree(ctx->V,&d);CHKERRQ(ierr);
  ierr = BVTensorGetFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  ierr = MatDenseGetArray(MS,&S);CHKERRQ(ierr);
  lds = d*ld;
  qK = ctx->qB;
  qM = ctx->qB+ld*ld;
  n = j+2;
  ierr = PetscMalloc2(n,&w1,n,&w2);CHKERRQ(ierr);
  sp = S+lds*j;
  sq = sp+ld;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qK,&ld_,sp,&one,&szero,w1,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qM,&ld_,sq,&one,&szero,w2,&one));
  *norm = 0.0;
  for (i=0;i<n;i++) *norm += PetscRealPart(w1[i]*PetscConj(sp[i])+w2[i]*PetscConj(sq[i]));
  *norm = (*norm>0.0)?PetscSqrtReal(*norm):-PetscSqrtReal(-*norm);
  ierr = PetscFree2(w1,w2);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(MS,&S);CHKERRQ(ierr);
  ierr = BVTensorRestoreFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPSTOARqKqMupdates(PEP pep,PetscInt j,Vec *wv)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       i,ld;
  PetscScalar    *qK,*qM;
  Vec            vj,v1,v2;

  PetscFunctionBegin;
  ierr = BVGetSizes(pep->V,NULL,NULL,&ld);CHKERRQ(ierr);
  qK = ctx->qB;
  qM = ctx->qB+ld*ld;
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

PetscErrorCode PEPSetUp_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      shift,sinv,flg;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       ld;

  PetscFunctionBegin;
  pep->lineariz = PETSC_TRUE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!ctx->lock && pep->mpd<pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (!pep->max_it) pep->max_it = PetscMax(100,2*(pep->nmat-1)*pep->n/pep->ncv);

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
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVCreateTensor(pep->V,pep->nmat-1,&ctx->V);CHKERRQ(ierr);
  ierr = PetscFree(ctx->qB);CHKERRQ(ierr); 
  ierr = BVGetSizes(pep->V,NULL,NULL,&ld);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*ld*ld,&ctx->qB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscScalar    *S,sonem=-1.0,sone=1.0,szero=0.0,*tp,*tq,*xp,*xq,*c,*qK,*qM;
  PetscInt       i,lds,n,j,ld,d;
  Mat            MS;

  PetscFunctionBegin;
  ierr = BVGetSizes(pep->V,NULL,NULL,&ld);CHKERRQ(ierr);
  ierr = BVTensorGetDegree(ctx->V,&d);CHKERRQ(ierr);
  ierr = BVTensorGetFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  ierr = MatDenseGetArray(MS,&S);CHKERRQ(ierr);
  lds = d*ld;
  qK = ctx->qB;
  qM = ctx->qB+ld*ld;
  n = k+2;
  ierr = PetscMalloc3(n,&tp,n,&tq,k,&c);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr); /* Size of qK and qM */
  ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against */
  xp = S+k*lds;
  xq = S+ld+k*lds;
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qK,&ld_,xp,&one,&szero,tp,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qM,&ld_,xq,&one,&szero,tq,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,tp,&one,&szero,y,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ld,&lds_,tq,&one,&sone,y,&one));
  for (i=0;i<k;i++) y[i] /= Omega[i];
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,y,&one,&sone,xp,&one));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ld,&lds_,y,&one,&sone,xq,&one));
  /* three times */
  for (j=0;j<2;j++) {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qK,&ld_,xp,&one,&szero,tp,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,qM,&ld_,xq,&one,&szero,tq,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,tp,&one,&szero,c,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ld,&lds_,tq,&one,&sone,c,&one));
    for (i=0;i<k;i++) c[i] /= Omega[i];
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,c,&one,&sone,xp,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ld,&lds_,c,&one,&sone,xq,&one));
    for (i=0;i<k;i++) y[i] += c[i];
  }
  ierr = PetscFree3(tp,tq,c);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(MS,&S);CHKERRQ(ierr);
  ierr = BVTensorRestoreFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Compute a run of Lanczos iterations. dim(work)=(ctx->ld)*4
*/
static PetscErrorCode PEPSTOARrun(PEP pep,PetscReal *a,PetscReal *b,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscBool *symmlost,PetscScalar *work,Vec *t_)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       i,j,m=*M,l,lock;
  PetscInt       lds,d,ld,offq,nqt;
  Vec            v=t_[0],t=t_[1],q=t_[2];
  PetscReal      norm,sym=0.0,fro=0.0,*f;
  PetscScalar    *y,*S;
  PetscBLASInt   j_,one=1;
  PetscBool      lindep;
  Mat            MS;

  PetscFunctionBegin;
  ierr = BVGetSizes(pep->V,NULL,NULL,&ld);CHKERRQ(ierr);
  ierr = BVTensorGetDegree(ctx->V,&d);CHKERRQ(ierr);
  ierr = BVTensorGetFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  ierr = MatDenseGetArray(MS,&S);CHKERRQ(ierr);
  ierr = BVGetActiveColumns(pep->V,&lock,&nqt);CHKERRQ(ierr);
  lds = d*ld;
  offq = ld;
  *breakdown = PETSC_FALSE; /* ----- */
  ierr = DSGetDimensions(pep->ds,NULL,NULL,&l,NULL,NULL);CHKERRQ(ierr);
  y = work;
  ierr = BVSetActiveColumns(ctx->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = BVGetColumn(pep->V,nqt,&t);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pep->V,0,nqt);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,v,S+j*lds);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,0,v,t);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,v,S+offq+j*lds);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,1,v,q);CHKERRQ(ierr);
    ierr = VecAXPY(q,pep->sfactor,t);CHKERRQ(ierr);
    ierr = STMatSolve(pep->st,q,t);CHKERRQ(ierr);
    ierr = VecScale(t,-1.0/(pep->sfactor*pep->sfactor));CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,nqt,&t);CHKERRQ(ierr);

    /* orthogonalize */
    ierr = BVOrthogonalizeColumn(pep->V,nqt,S+offq+(j+1)*lds,&norm,&lindep);CHKERRQ(ierr);
    if (!lindep) {
      *(S+offq+(j+1)*lds+nqt) = norm;
      ierr = BVScaleColumn(pep->V,nqt,1.0/norm);CHKERRQ(ierr);
      /* update qK and qM */
      ierr = PEPSTOARqKqMupdates(pep,nqt,t_);CHKERRQ(ierr);
      nqt++;
    }
    for (i=0;i<=nqt-1;i++) *(S+(j+1)*lds+i) = *(S+offq+j*lds+i);

    /* level-2 orthogonalization */
    ierr = PEPSTOAROrth2(pep,j+1,omega,y);CHKERRQ(ierr);
    a[j] = PetscRealPart(y[j])/omega[j];
    ierr = PEPSTOARNorm(pep,j+1,&norm);CHKERRQ(ierr);
    omega[j+1] = (norm > 0)?1.0:-1.0;
    ierr = BVScaleColumn(ctx->V,j+1,1.0/norm);CHKERRQ(ierr);
    b[j] = PetscAbsReal(norm);

    /* check symmetry */
    ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&f);CHKERRQ(ierr);
    if (j==k) {
      for (i=l;i<j-1;i++) y[i] = PetscAbsScalar(y[i])-PetscAbsReal(f[2*ld+i]);
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
      *M=j;
      break;
    }
  }
  ierr = BVSetActiveColumns(pep->V,lock,nqt);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(ctx->V,0,*M);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(MS,&S);CHKERRQ(ierr);
  ierr = BVTensorRestoreFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
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

PetscErrorCode PEPSolve_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       j,k,l,nv=0,ld,ldds,t,nq=0;
  PetscInt       lwa,lrwa,nwu=0,nconv=0,deg=pep->nmat-1;
  PetscScalar    *Q,*work,*S;
  PetscReal      beta,norm=1.0,*omega,*a,*b,*r,*rwork;
  PetscBool      breakdown,symmlost=PETSC_FALSE,sinv,falselock=PETSC_TRUE;
  Mat            MS,MQ;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  if (ctx->lock) {
    ierr = PetscOptionsGetBool(NULL,NULL,"-pep_stoar_falselocking",&falselock,NULL);CHKERRQ(ierr);
  }
  ierr = BVGetSizes(pep->V,NULL,NULL,&ld);CHKERRQ(ierr);
  lwa = 9*ld*ld+5*ld; 
  lrwa = 8*ld;
  ierr = PetscMalloc2(lwa,&work,lrwa,&rwork);CHKERRQ(ierr); /* REVIEW */
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
  ierr = RGPushScale(pep->rg,sinv?pep->sfactor:1.0/pep->sfactor);CHKERRQ(ierr);
  ierr = STScaleShift(pep->st,sinv?pep->sfactor:1.0/pep->sfactor);CHKERRQ(ierr);

  ierr = BVTensorGetFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  ierr = MatDenseGetArray(MS,&S);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = BVTensorBuildFirstColumn(ctx->V,pep->nini);CHKERRQ(ierr);
  ierr = BVGetActiveColumns(pep->V,NULL,&nq);CHKERRQ(ierr);
  for (j=0;j<nq;j++) {
    ierr = PEPSTOARqKqMupdates(pep,j,pep->work);CHKERRQ(ierr);
  }
  ierr = PEPSTOARNorm(pep,0,&norm);CHKERRQ(ierr);
  ierr = BVScaleColumn(ctx->V,0,1.0/norm);CHKERRQ(ierr);
  ierr = DSGetArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = (norm>0)?1.0:-1.0;
  ierr = DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);

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
    if (symmlost && nv==pep->nconv+l) {
      pep->reason = PEP_DIVERGED_SYMMETRY_LOST;
      pep->nconv = nconv;
      if (falselock || !ctx->lock) {
       ierr = BVSetActiveColumns(ctx->V,0,pep->nconv);CHKERRQ(ierr);
       ierr = BVTensorCompress(ctx->V,0);CHKERRQ(ierr);
      }
      break;
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
    ierr = DSSynchronize(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);

    /* Check convergence */
    /* ierr = PEPSTOARpreKConvergence(pep,nv,&norm,pep->work);CHKERRQ(ierr);*/
    norm = 1.0;
    ierr = DSGetDimensions(pep->ds,NULL,NULL,NULL,NULL,&t);CHKERRQ(ierr);
    ierr = PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,t-pep->nconv,PetscAbsReal(beta)*norm,&k);CHKERRQ(ierr);
    ierr = (*pep->stopping)(pep,pep->its,pep->max_it,k,pep->nev,&pep->reason,pep->stoppingctx);CHKERRQ(ierr);

    /* Update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      l = PetscMin(l,t);
      if (!breakdown) {
        ierr = DSGetArrayReal(pep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        if (*(a+ldds+k+l-1)!=0) {
          if (k+l<nv-1) l = l+1;
          else l = l-1;
        }
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(pep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
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
    nconv = k;
    if (!ctx->lock && pep->reason == PEP_CONVERGED_ITERATING && !breakdown) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */

    /* Update S */
    ierr = DSGetMat(pep->ds,DS_MAT_Q,&MQ);CHKERRQ(ierr);
    ierr = BVMultInPlace(ctx->V,MQ,pep->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&MQ);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = BVCopyColumn(ctx->V,nv,k+l);CHKERRQ(ierr);

     if (breakdown && pep->reason == PEP_CONVERGED_ITERATING) {
      /* stop if breakdown */
      ierr = PetscInfo2(pep,"Breakdown TOAR method (it=%D norm=%g)\n",pep->its,(double)beta);CHKERRQ(ierr);
      pep->reason = PEP_DIVERGED_BREAKDOWN;
    }
    if (pep->reason != PEP_CONVERGED_ITERATING) l--; 
    ierr = BVGetActiveColumns(pep->V,NULL,&nq);CHKERRQ(ierr);
    if (k+l+deg<=nq) {
      ierr = BVSetActiveColumns(ctx->V,pep->nconv,k+l+1);CHKERRQ(ierr);
      if (!falselock && ctx->lock) {
        ierr = BVTensorCompress(ctx->V,k-pep->nconv);CHKERRQ(ierr);
      } else {
        ierr = BVTensorCompress(ctx->V,0);CHKERRQ(ierr);
      }
      ierr = BVGetActiveColumns(pep->V,NULL,&nq);CHKERRQ(ierr);
      for (j=0;j<nq;j++) {
        ierr = PEPSTOARqKqMupdates(pep,j,pep->work);CHKERRQ(ierr);
      }
    }
    pep->nconv = k;
    ierr = PEPMonitor(pep,pep->its,nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }

  if (pep->nconv>0) {
    ierr = BVSetActiveColumns(ctx->V,0,pep->nconv);CHKERRQ(ierr);
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
  ierr = MatDenseRestoreArray(MS,&S);CHKERRQ(ierr);
  ierr = BVTensorRestoreFactors(ctx->V,NULL,&MS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetFromOptions_STOAR(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg,lock;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP STOAR Options");CHKERRQ(ierr);

    ierr = PetscOptionsBool("-pep_stoar_locking","Choose between locking and non-locking variants","PEPSTOARSetLocking",PETSC_FALSE,&lock,&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSTOARSetLocking(pep,lock);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPSTOARSetLocking_STOAR(PEP pep,PetscBool lock)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

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

static PetscErrorCode PEPSTOARGetLocking_STOAR(PEP pep,PetscBool *lock)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

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

PetscErrorCode PEPView_STOAR(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_TOAR      *ctx = (PEP_TOAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  using the %slocking variant\n",ctx->lock?"":"non-");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPDestroy_STOAR(PEP pep)
{
  PetscErrorCode ierr;
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = PetscFree(ctx->qB);CHKERRQ(ierr);
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetLocking_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  pep->ops->destroy        = PEPDestroy_STOAR;
  pep->ops->view           = PEPView_STOAR;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_TOAR;
  pep->ops->setdefaultst   = PEPSetDefaultST_Transform;

  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARSetLocking_C",PEPSTOARSetLocking_STOAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPSTOARGetLocking_C",PEPSTOARGetLocking_STOAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

