/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
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
           38(5):S385-S411, 2016.

       [3] D. Lu, Y. Su and Z. Bai, "Stability analysis of the two-level
           orthogonal Arnoldi procedure", SIAM J. Matrix Anal. App.
           37(1):195-214, 2016.
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
  "   volume = \"38\",\n"
  "   number = \"5\",\n"
  "   pages = \"S385--S411\",\n"
  "   year = \"2016,\"\n"
  "   doi = \"https://doi.org/10.1137/15M1022458\"\n"
  "}\n";

PetscErrorCode PEPSetUp_TOAR(PEP pep)
{
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBool      sinv,flg;
  PetscInt       i;

  PetscFunctionBegin;
  PEPCheckShiftSinvert(pep);
  CHKERRQ(PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd));
  PetscCheck(ctx->lock || pep->mpd>=pep->ncv,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (pep->max_it==PETSC_DEFAULT) pep->max_it = PetscMax(100,2*(pep->nmat-1)*pep->n/pep->ncv);
  if (!pep->which) CHKERRQ(PEPSetWhichEigenpairs_Default(pep));
  PetscCheck(pep->which!=PEP_ALL,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues");
  if (pep->problem_type!=PEP_GENERAL) CHKERRQ(PetscInfo(pep,"Problem type ignored, performing a non-symmetric linearization\n"));

  if (!ctx->keep) ctx->keep = 0.5;

  CHKERRQ(PEPAllocateSolution(pep,pep->nmat-1));
  CHKERRQ(PEPSetWorkVecs(pep,3));
  CHKERRQ(DSSetType(pep->ds,DSNHEP));
  CHKERRQ(DSSetExtraRow(pep->ds,PETSC_TRUE));
  CHKERRQ(DSAllocate(pep->ds,pep->ncv+1));

  CHKERRQ(PEPBasisCoefficients(pep,pep->pbc));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (!flg) {
    CHKERRQ(PetscFree(pep->solvematcoeffs));
    CHKERRQ(PetscMalloc1(pep->nmat,&pep->solvematcoeffs));
    CHKERRQ(PetscLogObjectMemory((PetscObject)pep,pep->nmat*sizeof(PetscScalar)));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv));
    if (sinv) CHKERRQ(PEPEvaluateBasis(pep,pep->target,0,pep->solvematcoeffs,NULL));
    else {
      for (i=0;i<pep->nmat-1;i++) pep->solvematcoeffs[i] = 0.0;
      pep->solvematcoeffs[pep->nmat-1] = 1.0;
    }
  }
  CHKERRQ(BVDestroy(&ctx->V));
  CHKERRQ(BVCreateTensor(pep->V,pep->nmat-1,&ctx->V));
  PetscFunctionReturn(0);
}

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
  PetscInt       nmat=pep->nmat,deg=nmat-1,k,j,off=0,lss;
  Vec            v=t_[0],ve=t_[1],q=t_[2];
  PetscScalar    alpha=1.0,*ss,a;
  PetscReal      *ca=pep->pbc,*cb=pep->pbc+nmat,*cg=pep->pbc+2*nmat;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(BVSetActiveColumns(pep->V,0,nv));
  CHKERRQ(STGetTransform(pep->st,&flg));
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
  CHKERRQ(BVMultVec(V,1.0,0.0,v,ss+off*lss));
  if (PetscUnlikely(pep->Dr)) { /* balancing */
    CHKERRQ(VecPointwiseMult(v,v,pep->Dr));
  }
  CHKERRQ(STMatMult(pep->st,off,v,q));
  CHKERRQ(VecScale(q,a));
  for (j=1+off;j<deg+off-1;j++) {
    CHKERRQ(BVMultVec(V,1.0,0.0,v,ss+j*lss));
    if (PetscUnlikely(pep->Dr)) CHKERRQ(VecPointwiseMult(v,v,pep->Dr));
    CHKERRQ(STMatMult(pep->st,j,v,t));
    a *= pep->sfactor;
    CHKERRQ(VecAXPY(q,a,t));
  }
  if (sinvert) {
    CHKERRQ(BVMultVec(V,1.0,0.0,v,ss));
    if (PetscUnlikely(pep->Dr)) CHKERRQ(VecPointwiseMult(v,v,pep->Dr));
    CHKERRQ(STMatMult(pep->st,deg,v,t));
    a *= pep->sfactor;
    CHKERRQ(VecAXPY(q,a,t));
  } else {
    CHKERRQ(BVMultVec(V,1.0,0.0,ve,ss+(deg-1)*lss));
    if (PetscUnlikely(pep->Dr)) CHKERRQ(VecPointwiseMult(ve,ve,pep->Dr));
    a *= pep->sfactor;
    CHKERRQ(STMatMult(pep->st,deg-1,ve,t));
    CHKERRQ(VecAXPY(q,a,t));
    a *= pep->sfactor;
  }
  if (flg || !sinvert) alpha /= a;
  CHKERRQ(STMatSolve(pep->st,q,t));
  CHKERRQ(VecScale(t,alpha));
  if (!sinvert) {
    CHKERRQ(VecAXPY(t,cg[deg-1],v));
    CHKERRQ(VecAXPY(t,cb[deg-1],ve));
  }
  if (PetscUnlikely(pep->Dr)) CHKERRQ(VecPointwiseDivide(t,t,pep->Dr));
  PetscFunctionReturn(0);
}

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

/*
  Compute a run of Arnoldi iterations dim(work)=ld
*/
static PetscErrorCode PEPTOARrun(PEP pep,PetscScalar sigma,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscBool *breakdown,Vec *t_)
{
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       j,m=*M,deg=pep->nmat-1,ld;
  PetscInt       lds,nqt,l;
  Vec            t;
  PetscReal      norm;
  PetscBool      flg,sinvert=PETSC_FALSE,lindep;
  PetscScalar    *x,*S;
  Mat            MS;

  PetscFunctionBegin;
  CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
  CHKERRQ(MatDenseGetArray(MS,&S));
  CHKERRQ(BVGetSizes(pep->V,NULL,NULL,&ld));
  lds = ld*deg;
  CHKERRQ(BVGetActiveColumns(pep->V,&l,&nqt));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (!flg) {
    /* spectral transformation handled by the solver */
    CHKERRQ(PetscObjectTypeCompareAny((PetscObject)pep->st,&flg,STSINVERT,STSHIFT,""));
    PetscCheck(flg,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"ST type not supported for TOAR without transforming matrices");
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinvert));
  }
  CHKERRQ(BVSetActiveColumns(ctx->V,0,m));
  for (j=k;j<m;j++) {
    /* apply operator */
    CHKERRQ(BVGetColumn(pep->V,nqt,&t));
    CHKERRQ(PEPTOARExtendBasis(pep,sinvert,sigma,S+j*lds,ld,nqt,pep->V,t,S+(j+1)*lds,ld,t_));
    CHKERRQ(BVRestoreColumn(pep->V,nqt,&t));

    /* orthogonalize */
    if (sinvert) x = S+(j+1)*lds;
    else x = S+(deg-1)*ld+(j+1)*lds;
    CHKERRQ(BVOrthogonalizeColumn(pep->V,nqt,x,&norm,&lindep));
    if (!lindep) {
      x[nqt] = norm;
      CHKERRQ(BVScaleColumn(pep->V,nqt,1.0/norm));
      nqt++;
    }

    CHKERRQ(PEPTOARCoefficients(pep,sinvert,sigma,nqt-1,S+j*lds,ld,S+(j+1)*lds,ld,x));

    /* level-2 orthogonalization */
    CHKERRQ(BVOrthogonalizeColumn(ctx->V,j+1,H+j*ldh,&norm,breakdown));
    H[j+1+ldh*j] = norm;
    if (PetscUnlikely(*breakdown)) {
      *M = j+1;
      break;
    }
    CHKERRQ(BVScaleColumn(ctx->V,j+1,1.0/norm));
    CHKERRQ(BVSetActiveColumns(pep->V,l,nqt));
  }
  CHKERRQ(BVSetActiveColumns(ctx->V,0,*M));
  CHKERRQ(MatDenseRestoreArray(MS,&S));
  CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
  PetscFunctionReturn(0);
}

/*
  Computes T_j = phi_idx(T). In T_j and T_p are phi_{idx-1}(T)
   and phi_{idx-2}(T) respectively or null if idx=0,1.
   Tp and Tj are input/output arguments
*/
static PetscErrorCode PEPEvaluateBasisM(PEP pep,PetscInt k,PetscScalar *T,PetscInt ldt,PetscInt idx,PetscScalar **Tp,PetscScalar **Tj)
{
  PetscInt       i;
  PetscReal      *ca,*cb,*cg;
  PetscScalar    *pt,g,a;
  PetscBLASInt   k_,ldt_;

  PetscFunctionBegin;
  if (idx==0) {
    CHKERRQ(PetscArrayzero(*Tj,k*k));
    CHKERRQ(PetscArrayzero(*Tp,k*k));
    for (i=0;i<k;i++) (*Tj)[i+i*k] = 1.0;
  } else {
    CHKERRQ(PetscBLASIntCast(ldt,&ldt_));
    CHKERRQ(PetscBLASIntCast(k,&k_));
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

static PetscErrorCode PEPExtractInvariantPair(PEP pep,PetscScalar sigma,PetscInt sr,PetscInt k,PetscScalar *S,PetscInt ld,PetscInt deg,PetscScalar *H,PetscInt ldh)
{
  PetscInt       i,j,jj,lds,ldt,d=pep->nmat-1,idxcpy=0;
  PetscScalar    *At,*Bt,*Hj,*Hp,*T,sone=1.0,g,a,*pM,*work;
  PetscBLASInt   k_,sr_,lds_,ldh_,info,*p,lwork,ldt_;
  PetscBool      transf=PETSC_FALSE,flg;
  PetscReal      norm,maxnrm,*rwork;
  BV             *R,Y;
  Mat            M,*A;

  PetscFunctionBegin;
  if (k==0) PetscFunctionReturn(0);
  lds = deg*ld;
  CHKERRQ(PetscCalloc6(k,&p,sr*k,&At,k*k,&Bt,k*k,&Hj,k*k,&Hp,sr*k,&work));
  CHKERRQ(PetscBLASIntCast(sr,&sr_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(ldh,&ldh_));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (!flg) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&flg));
    if (flg || sigma!=0.0) transf=PETSC_TRUE;
  }
  if (transf) {
    CHKERRQ(PetscMalloc1(k*k,&T));
    ldt = k;
    for (i=0;i<k;i++) CHKERRQ(PetscArraycpy(T+k*i,H+i*ldh,k));
    if (flg) {
      CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&k_,&k_,T,&k_,p,&info));
      SlepcCheckLapackInfo("getrf",info);
      CHKERRQ(PetscBLASIntCast(sr*k,&lwork));
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&k_,T,&k_,p,work,&lwork,&info));
      SlepcCheckLapackInfo("getri",info);
      CHKERRQ(PetscFPTrapPop());
    }
    if (sigma!=0.0) for (i=0;i<k;i++) T[i+k*i] += sigma;
  } else {
    T = H; ldt = ldh;
  }
  CHKERRQ(PetscBLASIntCast(ldt,&ldt_));
  switch (pep->extract) {
  case PEP_EXTRACT_NONE:
    break;
  case PEP_EXTRACT_NORM:
    if (pep->basis == PEP_BASIS_MONOMIAL) {
      CHKERRQ(PetscBLASIntCast(ldt,&ldt_));
      CHKERRQ(PetscMalloc1(k,&rwork));
      norm = LAPACKlange_("F",&k_,&k_,T,&ldt_,rwork);
      CHKERRQ(PetscFree(rwork));
      if (norm>1.0) idxcpy = d-1;
    } else {
      CHKERRQ(PetscBLASIntCast(ldt,&ldt_));
      CHKERRQ(PetscMalloc1(k,&rwork));
      maxnrm = 0.0;
      for (i=0;i<pep->nmat-1;i++) {
        CHKERRQ(PEPEvaluateBasisM(pep,k,T,ldt,i,&Hp,&Hj));
        norm = LAPACKlange_("F",&k_,&k_,Hj,&k_,rwork);
        if (norm > maxnrm) {
          idxcpy = i;
          maxnrm = norm;
        }
      }
      CHKERRQ(PetscFree(rwork));
    }
    if (idxcpy>0) {
      /* copy block idxcpy of S to the first one */
      for (j=0;j<k;j++) CHKERRQ(PetscArraycpy(S+j*lds,S+idxcpy*ld+j*lds,sr));
    }
    break;
  case PEP_EXTRACT_RESIDUAL:
    CHKERRQ(STGetTransform(pep->st,&flg));
    if (flg) {
      CHKERRQ(PetscMalloc1(pep->nmat,&A));
      for (i=0;i<pep->nmat;i++) CHKERRQ(STGetMatrixTransformed(pep->st,i,A+i));
    } else A = pep->A;
    CHKERRQ(PetscMalloc1(pep->nmat-1,&R));
    for (i=0;i<pep->nmat-1;i++) CHKERRQ(BVDuplicateResize(pep->V,k,R+i));
    CHKERRQ(BVDuplicateResize(pep->V,sr,&Y));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,sr,k,NULL,&M));
    g = 0.0; a = 1.0;
    CHKERRQ(BVSetActiveColumns(pep->V,0,sr));
    for (j=0;j<pep->nmat;j++) {
      CHKERRQ(BVMatMult(pep->V,A[j],Y));
      CHKERRQ(PEPEvaluateBasisM(pep,k,T,ldt,i,&Hp,&Hj));
      for (i=0;i<pep->nmat-1;i++) {
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&sr_,&k_,&k_,&a,S+i*ld,&lds_,Hj,&k_,&g,At,&sr_));
        CHKERRQ(MatDenseGetArray(M,&pM));
        for (jj=0;jj<k;jj++) CHKERRQ(PetscArraycpy(pM+jj*sr,At+jj*sr,sr));
        CHKERRQ(MatDenseRestoreArray(M,&pM));
        CHKERRQ(BVMult(R[i],1.0,(i==0)?0.0:1.0,Y,M));
      }
    }

    /* frobenius norm */
    maxnrm = 0.0;
    for (i=0;i<pep->nmat-1;i++) {
      CHKERRQ(BVNorm(R[i],NORM_FROBENIUS,&norm));
      if (maxnrm > norm) {
        maxnrm = norm;
        idxcpy = i;
      }
    }
    if (idxcpy>0) {
      /* copy block idxcpy of S to the first one */
      for (j=0;j<k;j++) CHKERRQ(PetscArraycpy(S+j*lds,S+idxcpy*ld+j*lds,sr));
    }
    if (flg) CHKERRQ(PetscFree(A));
    for (i=0;i<pep->nmat-1;i++) CHKERRQ(BVDestroy(&R[i]));
    CHKERRQ(PetscFree(R));
    CHKERRQ(BVDestroy(&Y));
    CHKERRQ(MatDestroy(&M));
    break;
  case PEP_EXTRACT_STRUCTURED:
    for (j=0;j<k;j++) Bt[j+j*k] = 1.0;
    for (j=0;j<sr;j++) {
      for (i=0;i<k;i++) At[j*k+i] = PetscConj(S[i*lds+j]);
    }
    CHKERRQ(PEPEvaluateBasisM(pep,k,T,ldt,0,&Hp,&Hj));
    for (i=1;i<deg;i++) {
      CHKERRQ(PEPEvaluateBasisM(pep,k,T,ldt,i,&Hp,&Hj));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&k_,&sr_,&k_,&sone,Hj,&k_,S+i*ld,&lds_,&sone,At,&k_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&k_,&k_,&k_,&sone,Hj,&k_,Hj,&k_,&sone,Bt,&k_));
    }
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&k_,&sr_,Bt,&k_,p,At,&k_,&info));
    CHKERRQ(PetscFPTrapPop());
    SlepcCheckLapackInfo("gesv",info);
    for (j=0;j<sr;j++) {
      for (i=0;i<k;i++) S[i*lds+j] = PetscConj(At[j*k+i]);
    }
    break;
  }
  if (transf) CHKERRQ(PetscFree(T));
  CHKERRQ(PetscFree6(p,At,Bt,Hj,Hp,work));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSolve_TOAR(PEP pep)
{
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscInt       i,j,k,l,nv=0,ld,lds,ldds,nq=0,nconv=0;
  PetscInt       nmat=pep->nmat,deg=nmat-1;
  PetscScalar    *S,*H,sigma;
  PetscReal      beta;
  PetscBool      breakdown=PETSC_FALSE,flg,falselock=PETSC_FALSE,sinv=PETSC_FALSE;
  Mat            MS,MQ;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));
  if (ctx->lock) {
    /* undocumented option to use a cheaper locking instead of the true locking */
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-pep_toar_falselocking",&falselock,NULL));
  }
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ldds));
  CHKERRQ(STGetShift(pep->st,&sigma));

  /* update polynomial basis coefficients */
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (pep->sfactor!=1.0) {
    for (i=0;i<nmat;i++) {
      pep->pbc[nmat+i] /= pep->sfactor;
      pep->pbc[2*nmat+i] /= pep->sfactor*pep->sfactor;
    }
    if (!flg) {
      pep->target /= pep->sfactor;
      CHKERRQ(RGPushScale(pep->rg,1.0/pep->sfactor));
      CHKERRQ(STScaleShift(pep->st,1.0/pep->sfactor));
      sigma /= pep->sfactor;
    } else {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv));
      pep->target = sinv?pep->target*pep->sfactor:pep->target/pep->sfactor;
      CHKERRQ(RGPushScale(pep->rg,sinv?pep->sfactor:1.0/pep->sfactor));
      CHKERRQ(STScaleShift(pep->st,sinv?pep->sfactor:1.0/pep->sfactor));
    }
  }

  if (flg) sigma = 0.0;

  /* clean projected matrix (including the extra-arrow) */
  CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
  CHKERRQ(PetscArrayzero(H,ldds*ldds));
  CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));

  /* Get the starting Arnoldi vector */
  CHKERRQ(BVTensorBuildFirstColumn(ctx->V,pep->nini));

  /* restart loop */
  l = 0;
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;

    /* compute an nv-step Lanczos factorization */
    nv = PetscMax(PetscMin(nconv+pep->mpd,pep->ncv),nv);
    CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
    CHKERRQ(PEPTOARrun(pep,sigma,H,ldds,pep->nconv+l,&nv,&breakdown,pep->work));
    beta = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
    CHKERRQ(DSSetDimensions(pep->ds,nv,pep->nconv,pep->nconv+l));
    CHKERRQ(DSSetState(pep->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    CHKERRQ(BVSetActiveColumns(ctx->V,pep->nconv,nv));

    /* solve projected problem */
    CHKERRQ(DSSolve(pep->ds,pep->eigr,pep->eigi));
    CHKERRQ(DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL));
    CHKERRQ(DSUpdateExtraRow(pep->ds));
    CHKERRQ(DSSynchronize(pep->ds,pep->eigr,pep->eigi));

    /* check convergence */
    CHKERRQ(PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,nv-pep->nconv,beta,&k));
    CHKERRQ((*pep->stopping)(pep,pep->its,pep->max_it,k,pep->nev,&pep->reason,pep->stoppingctx));

    /* update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv==k)?0:PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      CHKERRQ(DSGetTruncateSize(pep->ds,k,nv,&l));
      if (!breakdown) {
        /* prepare the Rayleigh quotient for restart */
        CHKERRQ(DSTruncate(pep->ds,k+l,PETSC_FALSE));
      }
    }
    nconv = k;
    if (!ctx->lock && pep->reason == PEP_CONVERGED_ITERATING && !breakdown) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) CHKERRQ(PetscInfo(pep,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    /* update S */
    CHKERRQ(DSGetMat(pep->ds,DS_MAT_Q,&MQ));
    CHKERRQ(BVMultInPlace(ctx->V,MQ,pep->nconv,k+l));
    CHKERRQ(MatDestroy(&MQ));

    /* copy last column of S */
    CHKERRQ(BVCopyColumn(ctx->V,nv,k+l));

    if (PetscUnlikely(breakdown && pep->reason == PEP_CONVERGED_ITERATING)) {
      /* stop if breakdown */
      CHKERRQ(PetscInfo(pep,"Breakdown TOAR method (it=%" PetscInt_FMT " norm=%g)\n",pep->its,(double)beta));
      pep->reason = PEP_DIVERGED_BREAKDOWN;
    }
    if (pep->reason != PEP_CONVERGED_ITERATING) l--;
    /* truncate S */
    CHKERRQ(BVGetActiveColumns(pep->V,NULL,&nq));
    if (k+l+deg<=nq) {
      CHKERRQ(BVSetActiveColumns(ctx->V,pep->nconv,k+l+1));
      if (!falselock && ctx->lock) CHKERRQ(BVTensorCompress(ctx->V,k-pep->nconv));
      else CHKERRQ(BVTensorCompress(ctx->V,0));
    }
    pep->nconv = k;
    CHKERRQ(PEPMonitor(pep,pep->its,nconv,pep->eigr,pep->eigi,pep->errest,nv));
  }
  if (pep->nconv>0) {
    /* {V*S_nconv^i}_{i=0}^{d-1} has rank nconv instead of nconv+d-1. Force zeros in each S_nconv^i block */
    CHKERRQ(BVSetActiveColumns(ctx->V,0,pep->nconv));
    CHKERRQ(BVGetActiveColumns(pep->V,NULL,&nq));
    CHKERRQ(BVSetActiveColumns(pep->V,0,nq));
    if (nq>pep->nconv) {
      CHKERRQ(BVTensorCompress(ctx->V,pep->nconv));
      CHKERRQ(BVSetActiveColumns(pep->V,0,pep->nconv));
      nq = pep->nconv;
    }

    /* perform Newton refinement if required */
    if (pep->refine==PEP_REFINE_MULTIPLE && pep->rits>0) {
      /* extract invariant pair */
      CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
      CHKERRQ(MatDenseGetArray(MS,&S));
      CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
      CHKERRQ(BVGetSizes(pep->V,NULL,NULL,&ld));
      lds = deg*ld;
      CHKERRQ(PEPExtractInvariantPair(pep,sigma,nq,pep->nconv,S,ld,deg,H,ldds));
      CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
      CHKERRQ(DSSetDimensions(pep->ds,pep->nconv,0,0));
      CHKERRQ(DSSetState(pep->ds,DS_STATE_RAW));
      CHKERRQ(PEPNewtonRefinement_TOAR(pep,sigma,&pep->rits,NULL,pep->nconv,S,lds));
      CHKERRQ(DSSolve(pep->ds,pep->eigr,pep->eigi));
      CHKERRQ(DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL));
      CHKERRQ(DSSynchronize(pep->ds,pep->eigr,pep->eigi));
      CHKERRQ(DSGetMat(pep->ds,DS_MAT_Q,&MQ));
      CHKERRQ(BVMultInPlace(ctx->V,MQ,0,pep->nconv));
      CHKERRQ(MatDestroy(&MQ));
      CHKERRQ(MatDenseRestoreArray(MS,&S));
      CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
    }
  }
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (pep->refine!=PEP_REFINE_MULTIPLE || pep->rits==0) {
    if (!flg && pep->ops->backtransform) CHKERRQ((*pep->ops->backtransform)(pep));
    if (pep->sfactor!=1.0) {
      for (j=0;j<pep->nconv;j++) {
        pep->eigr[j] *= pep->sfactor;
        pep->eigi[j] *= pep->sfactor;
      }
      /* restore original values */
      for (i=0;i<pep->nmat;i++) {
        pep->pbc[pep->nmat+i] *= pep->sfactor;
        pep->pbc[2*pep->nmat+i] *= pep->sfactor*pep->sfactor;
      }
    }
  }
  /* restore original values */
  if (!flg) {
    pep->target *= pep->sfactor;
    CHKERRQ(STScaleShift(pep->st,pep->sfactor));
  } else {
    CHKERRQ(STScaleShift(pep->st,sinv?1.0/pep->sfactor:pep->sfactor));
    pep->target = (sinv)?pep->target/pep->sfactor:pep->target*pep->sfactor;
  }
  if (pep->sfactor!=1.0) CHKERRQ(RGPopScale(pep->rg));

  /* change the state to raw so that DSVectors() computes eigenvectors from scratch */
  CHKERRQ(DSSetDimensions(pep->ds,pep->nconv,0,0));
  CHKERRQ(DSSetState(pep->ds,DS_STATE_RAW));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPTOARSetRestart_TOAR(PEP pep,PetscReal keep)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    PetscCheck(keep>=0.1 && keep<=0.9,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPTOARSetRestart - Sets the restart parameter for the TOAR
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on pep

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  CHKERRQ(PetscTryMethod(pep,"PEPTOARSetRestart_C",(PEP,PetscReal),(pep,keep)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPTOARGetRestart_TOAR(PEP pep,PetscReal *keep)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidRealPointer(keep,2);
  CHKERRQ(PetscUseMethod(pep,"PEPTOARGetRestart_C",(PEP,PetscReal*),(pep,keep)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPTOARSetLocking_TOAR(PEP pep,PetscBool lock)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

/*@
   PEPTOARSetLocking - Choose between locking and non-locking variants of
   the TOAR method.

   Logically Collective on pep

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,lock,2);
  CHKERRQ(PetscTryMethod(pep,"PEPTOARSetLocking_C",(PEP,PetscBool),(pep,lock)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPTOARGetLocking_TOAR(PEP pep,PetscBool *lock)
{
  PEP_TOAR *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidBoolPointer(lock,2);
  CHKERRQ(PetscUseMethod(pep,"PEPTOARGetLocking_C",(PEP,PetscBool*),(pep,lock)));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetFromOptions_TOAR(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscBool      flg,lock;
  PetscReal      keep;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"PEP TOAR Options"));

    CHKERRQ(PetscOptionsReal("-pep_toar_restart","Proportion of vectors kept after restart","PEPTOARSetRestart",0.5,&keep,&flg));
    if (flg) CHKERRQ(PEPTOARSetRestart(pep,keep));

    CHKERRQ(PetscOptionsBool("-pep_toar_locking","Choose between locking and non-locking variants","PEPTOARSetLocking",PETSC_FALSE,&lock,&flg));
    if (flg) CHKERRQ(PEPTOARSetLocking(pep,lock));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode PEPView_TOAR(PEP pep,PetscViewer viewer)
{
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep)));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using the %slocking variant\n",ctx->lock?"":"non-"));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPDestroy_TOAR(PEP pep)
{
  PEP_TOAR       *ctx = (PEP_TOAR*)pep->data;

  PetscFunctionBegin;
  CHKERRQ(BVDestroy(&ctx->V));
  CHKERRQ(PetscFree(pep->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetRestart_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetLocking_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetLocking_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode PEPCreate_TOAR(PEP pep)
{
  PEP_TOAR       *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pep,&ctx));
  pep->data = (void*)ctx;

  pep->lineariz = PETSC_TRUE;
  ctx->lock     = PETSC_TRUE;

  pep->ops->solve          = PEPSolve_TOAR;
  pep->ops->setup          = PEPSetUp_TOAR;
  pep->ops->setfromoptions = PEPSetFromOptions_TOAR;
  pep->ops->destroy        = PEPDestroy_TOAR;
  pep->ops->view           = PEPView_TOAR;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_TOAR;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetRestart_C",PEPTOARSetRestart_TOAR));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetRestart_C",PEPTOARGetRestart_TOAR));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARSetLocking_C",PEPTOARSetLocking_TOAR));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPTOARGetLocking_C",PEPTOARGetLocking_TOAR));
  PetscFunctionReturn(0);
}
