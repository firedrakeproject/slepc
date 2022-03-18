/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "davidson"

   Step: calculate the best eigenpairs in the subspace V

   For that, performs these steps:
     1) Update W <- A * V
     2) Update H <- V' * W
     3) Obtain eigenpairs of H
     4) Select some eigenpairs
     5) Compute the Ritz pairs of the selected ones
*/

#include "davidson.h"
#include <slepcblaslapack.h>

static PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d)
{
  PetscFunctionBegin;
  CHKERRQ(BVSetActiveColumns(d->eps->V,0,0));
  if (d->W) CHKERRQ(BVSetActiveColumns(d->W,0,0));
  CHKERRQ(BVSetActiveColumns(d->AX,0,0));
  if (d->BX) CHKERRQ(BVSetActiveColumns(d->BX,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode dvd_calcpairs_qz_d(dvdDashboard *d)
{
  PetscFunctionBegin;
  CHKERRQ(BVDestroy(&d->W));
  CHKERRQ(BVDestroy(&d->AX));
  CHKERRQ(BVDestroy(&d->BX));
  CHKERRQ(BVDestroy(&d->auxBV));
  CHKERRQ(MatDestroy(&d->H));
  CHKERRQ(MatDestroy(&d->G));
  CHKERRQ(MatDestroy(&d->auxM));
  CHKERRQ(SlepcVecPoolDestroy(&d->auxV));
  CHKERRQ(PetscFree(d->nBds));
  PetscFunctionReturn(0);
}

/* in complex, d->size_H real auxiliary values are needed */
static PetscErrorCode dvd_calcpairs_projeig_solve(dvdDashboard *d)
{
  Vec               v;
  PetscScalar       *pA;
  const PetscScalar *pv;
  PetscInt          i,lV,kV,n,ld;

  PetscFunctionBegin;
  CHKERRQ(BVGetActiveColumns(d->eps->V,&lV,&kV));
  n = kV-lV;
  CHKERRQ(DSSetDimensions(d->eps->ds,n,0,0));
  CHKERRQ(DSCopyMat(d->eps->ds,DS_MAT_A,0,0,d->H,lV,lV,n,n,PETSC_FALSE));
  if (d->G) {
    CHKERRQ(DSCopyMat(d->eps->ds,DS_MAT_B,0,0,d->G,lV,lV,n,n,PETSC_FALSE));
  }
  /* Set the signature on projected matrix B */
  if (DVD_IS(d->sEP,DVD_EP_INDEFINITE)) {
    CHKERRQ(DSGetLeadingDimension(d->eps->ds,&ld));
    CHKERRQ(DSGetArray(d->eps->ds,DS_MAT_B,&pA));
    CHKERRQ(PetscArrayzero(pA,n*ld));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,kV,&v));
    CHKERRQ(BVGetSignature(d->eps->V,v));
    CHKERRQ(VecGetArrayRead(v,&pv));
    for (i=0;i<n;i++) {
      pA[i+ld*i] = d->nBds[i] = PetscRealPart(pv[lV+i]);
    }
    CHKERRQ(VecRestoreArrayRead(v,&pv));
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(DSRestoreArray(d->eps->ds,DS_MAT_B,&pA));
  }
  CHKERRQ(DSSetState(d->eps->ds,DS_STATE_RAW));
  CHKERRQ(DSSolve(d->eps->ds,d->eigr,d->eigi));
  PetscFunctionReturn(0);
}

/*
   A(lA:kA-1,lA:kA-1) <- Z(l:k-1)'*A(l:k-1,l:k-1)*Q(l,k-1), where k=l+kA-lA
 */
static PetscErrorCode EPSXDUpdateProj(Mat Q,Mat Z,PetscInt l,Mat A,PetscInt lA,PetscInt kA,Mat aux)
{
  PetscScalar       one=1.0,zero=0.0;
  PetscInt          i,j,dA_=kA-lA,m0,n0,ldA_,ldQ_,ldZ_,nQ_;
  PetscBLASInt      dA,nQ,ldA,ldQ,ldZ;
  PetscScalar       *pA,*pW;
  const PetscScalar *pQ,*pZ;
  PetscBool         symm=PETSC_FALSE,set,flg;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&m0,&n0)); ldA_=m0;
  PetscAssert(m0==n0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"A should be square");
  PetscAssert(lA>=0 && lA<=m0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial row, column in A");
  PetscAssert(kA>=0 && kA>=lA && kA<=m0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid final row, column in A");
  CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
  symm = set? flg: PETSC_FALSE;
  CHKERRQ(MatGetSize(Q,&m0,&n0)); ldQ_=nQ_=m0;
  PetscAssert(l>=0 && l<=n0 && l+dA_<=n0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial column in Q");
  CHKERRQ(MatGetSize(Z,&m0,&n0)); ldZ_=m0;
  PetscAssert(l>=0 && l<=n0 && l+dA_<=n0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial column in Z");
  CHKERRQ(MatGetSize(aux,&m0,&n0));
  PetscAssert(m0*n0>=nQ_*dA_,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"aux should be larger");
  CHKERRQ(PetscBLASIntCast(dA_,&dA));
  CHKERRQ(PetscBLASIntCast(nQ_,&nQ));
  CHKERRQ(PetscBLASIntCast(ldA_,&ldA));
  CHKERRQ(PetscBLASIntCast(ldQ_,&ldQ));
  CHKERRQ(PetscBLASIntCast(ldZ_,&ldZ));
  CHKERRQ(MatDenseGetArray(A,&pA));
  CHKERRQ(MatDenseGetArrayRead(Q,&pQ));
  if (Q!=Z) CHKERRQ(MatDenseGetArrayRead(Z,&pZ));
  else pZ = pQ;
  CHKERRQ(MatDenseGetArrayWrite(aux,&pW));
  /* W = A*Q */
  if (symm) {
    /* symmetrize before multiplying */
    for (i=lA+1;i<lA+nQ;i++) {
      for (j=lA;j<i;j++) pA[i+j*ldA] = PetscConj(pA[j+i*ldA]);
    }
  }
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&nQ,&dA,&nQ,&one,&pA[ldA*lA+lA],&ldA,&pQ[ldQ*l+l],&ldQ,&zero,pW,&nQ));
  /* A = Q'*W */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&dA,&dA,&nQ,&one,&pZ[ldZ*l+l],&ldZ,pW,&nQ,&zero,&pA[ldA*lA+lA],&ldA));
  CHKERRQ(MatDenseRestoreArray(A,&pA));
  CHKERRQ(MatDenseRestoreArrayRead(Q,&pQ));
  if (Q!=Z) CHKERRQ(MatDenseRestoreArrayRead(Z,&pZ));
  CHKERRQ(MatDenseRestoreArrayWrite(aux,&pW));
  PetscFunctionReturn(0);
}

static PetscErrorCode dvd_calcpairs_updateproj(dvdDashboard *d)
{
  Mat            Q,Z;
  PetscInt       lV,kV;
  PetscBool      symm;

  PetscFunctionBegin;
  CHKERRQ(DSGetMat(d->eps->ds,DS_MAT_Q,&Q));
  if (d->W) CHKERRQ(DSGetMat(d->eps->ds,DS_MAT_Z,&Z));
  else Z = Q;
  CHKERRQ(BVGetActiveColumns(d->eps->V,&lV,&kV));
  CHKERRQ(EPSXDUpdateProj(Q,Z,0,d->H,lV,lV+d->V_tra_e,d->auxM));
  if (d->G) CHKERRQ(EPSXDUpdateProj(Q,Z,0,d->G,lV,lV+d->V_tra_e,d->auxM));
  CHKERRQ(MatDestroy(&Q));
  if (d->W) CHKERRQ(MatDestroy(&Z));

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)d->eps->ds,&symm,DSHEP,DSGHIEP,DSGHEP,""));
  if (d->V_tra_s==0 || symm) PetscFunctionReturn(0);
  /* Compute upper part of H (and G): H(0:l-1,l:k-1) <- W(0:l-1)' * AV(l:k-1), where
     k=l+d->V_tra_s */
  CHKERRQ(BVSetActiveColumns(d->W?d->W:d->eps->V,0,lV));
  CHKERRQ(BVSetActiveColumns(d->AX,lV,lV+d->V_tra_s));
  CHKERRQ(BVDot(d->AX,d->W?d->W:d->eps->V,d->H));
  if (d->G) {
    CHKERRQ(BVSetActiveColumns(d->BX?d->BX:d->eps->V,lV,lV+d->V_tra_s));
    CHKERRQ(BVDot(d->BX?d->BX:d->eps->V,d->W?d->W:d->eps->V,d->G));
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)d->eps->ds,DSGHEP,&symm));
  if (!symm) {
    CHKERRQ(BVSetActiveColumns(d->W?d->W:d->eps->V,lV,lV+d->V_tra_s));
    CHKERRQ(BVSetActiveColumns(d->AX,0,lV));
    CHKERRQ(BVDot(d->AX,d->W?d->W:d->eps->V,d->H));
    if (d->G) {
      CHKERRQ(BVSetActiveColumns(d->BX?d->BX:d->eps->V,0,lV));
      CHKERRQ(BVDot(d->BX?d->BX:d->eps->V,d->W?d->W:d->eps->V,d->G));
    }
  }
  CHKERRQ(BVSetActiveColumns(d->eps->V,lV,kV));
  CHKERRQ(BVSetActiveColumns(d->AX,lV,kV));
  if (d->BX) CHKERRQ(BVSetActiveColumns(d->BX,lV,kV));
  if (d->W) CHKERRQ(BVSetActiveColumns(d->W,lV,kV));
  if (d->W) CHKERRQ(dvd_harm_updateproj(d));
  PetscFunctionReturn(0);
}

/*
   BV <- BV*MT
 */
static inline PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,BV bv,DSMatType mat)
{
  PetscInt       l,k,n;
  Mat            auxM;

  PetscFunctionBegin;
  CHKERRQ(BVGetActiveColumns(d->eps->V,&l,&k));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&auxM));
  CHKERRQ(MatZeroEntries(auxM));
  CHKERRQ(DSGetDimensions(d->eps->ds,&n,NULL,NULL,NULL));
  PetscAssert(k-l==n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Consistency broken");
  CHKERRQ(DSCopyMat(d->eps->ds,mat,0,0,auxM,l,l,n,d->V_tra_e,PETSC_TRUE));
  CHKERRQ(BVMultInPlace(bv,auxM,l,l+d->V_tra_e));
  CHKERRQ(MatDestroy(&auxM));
  PetscFunctionReturn(0);
}

static PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d)
{
  PetscInt       i,l,k;
  Vec            v1,v2;
  PetscScalar    *pv;

  PetscFunctionBegin;
  CHKERRQ(BVGetActiveColumns(d->eps->V,&l,&k));
  /* Update AV, BV, W and the projected matrices */
  /* 1. S <- S*MT */
  if (d->V_tra_s != d->V_tra_e || d->V_tra_e > 0) {
    CHKERRQ(dvd_calcpairs_updateBV0_gen(d,d->eps->V,DS_MAT_Q));
    if (d->W) CHKERRQ(dvd_calcpairs_updateBV0_gen(d,d->W,DS_MAT_Z));
    CHKERRQ(dvd_calcpairs_updateBV0_gen(d,d->AX,DS_MAT_Q));
    if (d->BX) CHKERRQ(dvd_calcpairs_updateBV0_gen(d,d->BX,DS_MAT_Q));
    CHKERRQ(dvd_calcpairs_updateproj(d));
    /* Update signature */
    if (d->nBds) {
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,l+d->V_tra_e,&v1));
      CHKERRQ(BVSetActiveColumns(d->eps->V,0,l+d->V_tra_e));
      CHKERRQ(BVGetSignature(d->eps->V,v1));
      CHKERRQ(VecGetArray(v1,&pv));
      for (i=0;i<d->V_tra_e;i++) pv[l+i] = d->nBds[i];
      CHKERRQ(VecRestoreArray(v1,&pv));
      CHKERRQ(BVSetSignature(d->eps->V,v1));
      CHKERRQ(BVSetActiveColumns(d->eps->V,l,k));
      CHKERRQ(VecDestroy(&v1));
    }
    k = l+d->V_tra_e;
    l+= d->V_tra_s;
  } else {
    /* 2. V <- orth(V, V_new) */
    CHKERRQ(dvd_orthV(d->eps->V,l+d->V_new_s,l+d->V_new_e));
    /* 3. AV <- [AV A * V(V_new_s:V_new_e-1)] */
    /* Check consistency */
    PetscAssert(k-l==d->V_new_s,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Consistency broken");
    for (i=l+d->V_new_s;i<l+d->V_new_e;i++) {
      CHKERRQ(BVGetColumn(d->eps->V,i,&v1));
      CHKERRQ(BVGetColumn(d->AX,i,&v2));
      CHKERRQ(MatMult(d->A,v1,v2));
      CHKERRQ(BVRestoreColumn(d->eps->V,i,&v1));
      CHKERRQ(BVRestoreColumn(d->AX,i,&v2));
    }
    /* 4. BV <- [BV B * V(V_new_s:V_new_e-1)] */
    if (d->BX) {
      /* Check consistency */
      PetscAssert(k-l==d->V_new_s,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Consistency broken");
      for (i=l+d->V_new_s;i<l+d->V_new_e;i++) {
        CHKERRQ(BVGetColumn(d->eps->V,i,&v1));
        CHKERRQ(BVGetColumn(d->BX,i,&v2));
        CHKERRQ(MatMult(d->B,v1,v2));
        CHKERRQ(BVRestoreColumn(d->eps->V,i,&v1));
        CHKERRQ(BVRestoreColumn(d->BX,i,&v2));
      }
    }
    /* 5. W <- [W f(AV,BV)] */
    if (d->W) {
      CHKERRQ(d->calcpairs_W(d));
      CHKERRQ(dvd_orthV(d->W,l+d->V_new_s,l+d->V_new_e));
    }
    /* 6. H <- W' * AX; G <- W' * BX */
    CHKERRQ(BVSetActiveColumns(d->eps->V,l+d->V_new_s,l+d->V_new_e));
    CHKERRQ(BVSetActiveColumns(d->AX,l+d->V_new_s,l+d->V_new_e));
    if (d->BX) CHKERRQ(BVSetActiveColumns(d->BX,l+d->V_new_s,l+d->V_new_e));
    if (d->W) CHKERRQ(BVSetActiveColumns(d->W,l+d->V_new_s,l+d->V_new_e));
    CHKERRQ(BVMatProject(d->AX,NULL,d->W?d->W:d->eps->V,d->H));
    if (d->G) CHKERRQ(BVMatProject(d->BX?d->BX:d->eps->V,NULL,d->W?d->W:d->eps->V,d->G));
    CHKERRQ(BVSetActiveColumns(d->eps->V,l,k));
    CHKERRQ(BVSetActiveColumns(d->AX,l,k));
    if (d->BX) CHKERRQ(BVSetActiveColumns(d->BX,l,k));
    if (d->W) CHKERRQ(BVSetActiveColumns(d->W,l,k));

    /* Perform the transformation on the projected problem */
    if (d->W) {
      CHKERRQ(d->calcpairs_proj_trans(d));
    }
    k = l+d->V_new_e;
  }
  CHKERRQ(BVSetActiveColumns(d->eps->V,l,k));
  CHKERRQ(BVSetActiveColumns(d->AX,l,k));
  if (d->BX) CHKERRQ(BVSetActiveColumns(d->BX,l,k));
  if (d->W) CHKERRQ(BVSetActiveColumns(d->W,l,k));

  /* Solve the projected problem */
  CHKERRQ(dvd_calcpairs_projeig_solve(d));

  d->V_tra_s = d->V_tra_e = 0;
  d->V_new_s = d->V_new_e;
  PetscFunctionReturn(0);
}

static PetscErrorCode dvd_calcpairs_apply_arbitrary(dvdDashboard *d,PetscInt r_s,PetscInt r_e,PetscScalar *rr,PetscScalar *ri)
{
  PetscInt       i,k,ld;
  PetscScalar    *pX;
  Vec            *X,xr,xi;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       N=1;
#else
  PetscInt       N=2,j;
#endif

  PetscFunctionBegin;
  /* Quick exit without neither arbitrary selection nor harmonic extraction */
  if (!d->eps->arbitrary && !d->calcpairs_eig_backtrans) PetscFunctionReturn(0);

  /* Quick exit without arbitrary selection, but with harmonic extraction */
  if (d->calcpairs_eig_backtrans) {
    for (i=r_s; i<r_e; i++) {
      CHKERRQ(d->calcpairs_eig_backtrans(d,d->eigr[i],d->eigi[i],&rr[i-r_s],&ri[i-r_s]));
    }
  }
  if (!d->eps->arbitrary) PetscFunctionReturn(0);

  CHKERRQ(SlepcVecPoolGetVecs(d->auxV,N,&X));
  CHKERRQ(DSGetLeadingDimension(d->eps->ds,&ld));
  for (i=r_s;i<r_e;i++) {
    k = i;
    CHKERRQ(DSVectors(d->eps->ds,DS_MAT_X,&k,NULL));
    CHKERRQ(DSGetArray(d->eps->ds,DS_MAT_X,&pX));
    CHKERRQ(dvd_improvex_compute_X(d,i,k+1,X,pX,ld));
    CHKERRQ(DSRestoreArray(d->eps->ds,DS_MAT_X,&pX));
#if !defined(PETSC_USE_COMPLEX)
    if (d->nX[i] != 1.0) {
      for (j=i;j<k+1;j++) {
        CHKERRQ(VecScale(X[j-i],1.0/d->nX[i]));
      }
    }
    xr = X[0];
    xi = X[1];
    if (i == k) {
      CHKERRQ(VecSet(xi,0.0));
    }
#else
    xr = X[0];
    xi = NULL;
    if (d->nX[i] != 1.0) {
      CHKERRQ(VecScale(xr,1.0/d->nX[i]));
    }
#endif
    CHKERRQ((d->eps->arbitrary)(rr[i-r_s],ri[i-r_s],xr,xi,&rr[i-r_s],&ri[i-r_s],d->eps->arbitraryctx));
#if !defined(PETSC_USE_COMPLEX)
    if (i != k) {
      rr[i+1-r_s] = rr[i-r_s];
      ri[i+1-r_s] = ri[i-r_s];
      i++;
    }
#endif
  }
  CHKERRQ(SlepcVecPoolRestoreVecs(d->auxV,N,&X));
  PetscFunctionReturn(0);
}

static PetscErrorCode dvd_calcpairs_selectPairs(dvdDashboard *d,PetscInt n)
{
  PetscInt       k,lV,kV,nV;
  PetscScalar    *rr,*ri;

  PetscFunctionBegin;
  CHKERRQ(BVGetActiveColumns(d->eps->V,&lV,&kV));
  nV = kV - lV;
  n = PetscMin(n,nV);
  if (n <= 0) PetscFunctionReturn(0);
  /* Put the best n pairs at the beginning. Useful for restarting */
  if (d->eps->arbitrary || d->calcpairs_eig_backtrans) {
    CHKERRQ(PetscMalloc1(nV,&rr));
    CHKERRQ(PetscMalloc1(nV,&ri));
    CHKERRQ(dvd_calcpairs_apply_arbitrary(d,0,nV,rr,ri));
  } else {
    rr = d->eigr;
    ri = d->eigi;
  }
  k = n;
  CHKERRQ(DSSort(d->eps->ds,d->eigr,d->eigi,rr,ri,&k));
  /* Put the best pair at the beginning. Useful to check its residual */
#if !defined(PETSC_USE_COMPLEX)
  if (n != 1 && (n != 2 || d->eigi[0] == 0.0))
#else
  if (n != 1)
#endif
  {
    CHKERRQ(dvd_calcpairs_apply_arbitrary(d,0,nV,rr,ri));
    k = 1;
    CHKERRQ(DSSort(d->eps->ds,d->eigr,d->eigi,rr,ri,&k));
  }
  CHKERRQ(DSSynchronize(d->eps->ds,d->eigr,d->eigi));

  if (d->calcpairs_eigs_trans) {
    CHKERRQ(d->calcpairs_eigs_trans(d));
  }
  if (d->eps->arbitrary || d->calcpairs_eig_backtrans) {
    CHKERRQ(PetscFree(rr));
    CHKERRQ(PetscFree(ri));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSXDComputeDSConv(dvdDashboard *d)
{
  PetscInt          i,ld;
  Vec               v;
  PetscScalar       *pA;
  const PetscScalar *pv;
  PetscBool         symm;

  PetscFunctionBegin;
  CHKERRQ(BVSetActiveColumns(d->eps->V,0,d->eps->nconv));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)d->eps->ds,DSHEP,&symm));
  if (symm) PetscFunctionReturn(0);
  CHKERRQ(DSSetDimensions(d->eps->ds,d->eps->nconv,0,0));
  CHKERRQ(DSCopyMat(d->eps->ds,DS_MAT_A,0,0,d->H,0,0,d->eps->nconv,d->eps->nconv,PETSC_FALSE));
  if (d->G) {
    CHKERRQ(DSCopyMat(d->eps->ds,DS_MAT_B,0,0,d->G,0,0,d->eps->nconv,d->eps->nconv,PETSC_FALSE));
  }
  /* Set the signature on projected matrix B */
  if (DVD_IS(d->sEP,DVD_EP_INDEFINITE)) {
    CHKERRQ(DSGetLeadingDimension(d->eps->ds,&ld));
    CHKERRQ(DSGetArray(d->eps->ds,DS_MAT_B,&pA));
    CHKERRQ(PetscArrayzero(pA,d->eps->nconv*ld));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,d->eps->nconv,&v));
    CHKERRQ(BVGetSignature(d->eps->V,v));
    CHKERRQ(VecGetArrayRead(v,&pv));
    for (i=0;i<d->eps->nconv;i++) pA[i+ld*i] = pv[i];
    CHKERRQ(VecRestoreArrayRead(v,&pv));
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(DSRestoreArray(d->eps->ds,DS_MAT_B,&pA));
  }
  CHKERRQ(DSSetState(d->eps->ds,DS_STATE_RAW));
  CHKERRQ(DSSolve(d->eps->ds,d->eps->eigr,d->eps->eigi));
  CHKERRQ(DSSynchronize(d->eps->ds,d->eps->eigr,d->eps->eigi));
  if (d->W) {
    for (i=0; i<d->eps->nconv; i++) {
      CHKERRQ(d->calcpairs_eig_backtrans(d,d->eps->eigr[i],d->eps->eigi[i],&d->eps->eigr[i],&d->eps->eigi[i]));
    }
  }
  PetscFunctionReturn(0);
}

/*
   Compute the residual vectors R(i) <- (AV - BV*eigr(i))*pX(i), and also
   the norm associated to the Schur pair, where i = r_s..r_e
*/
static PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e)
{
  PetscInt       i,ldpX;
  PetscScalar    *pX;
  BV             BX = d->BX?d->BX:d->eps->V;
  Vec            *R;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(d->eps->ds,&ldpX));
  CHKERRQ(DSGetArray(d->eps->ds,DS_MAT_Q,&pX));
  /* nX(i) <- ||X(i)|| */
  CHKERRQ(dvd_improvex_compute_X(d,r_s,r_e,NULL,pX,ldpX));
  CHKERRQ(SlepcVecPoolGetVecs(d->auxV,r_e-r_s,&R));
  for (i=r_s;i<r_e;i++) {
    /* R(i-r_s) <- AV*pX(i) */
    CHKERRQ(BVMultVec(d->AX,1.0,0.0,R[i-r_s],&pX[ldpX*i]));
    /* R(i-r_s) <- R(i-r_s) - eigr(i)*BX*pX(i) */
    CHKERRQ(BVMultVec(BX,-d->eigr[i],1.0,R[i-r_s],&pX[ldpX*i]));
  }
  CHKERRQ(DSRestoreArray(d->eps->ds,DS_MAT_Q,&pX));
  CHKERRQ(d->calcpairs_proj_res(d,r_s,r_e,R));
  CHKERRQ(SlepcVecPoolRestoreVecs(d->auxV,r_e-r_s,&R));
  PetscFunctionReturn(0);
}

static PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *R)
{
  PetscInt       i,l,k;
  PetscBool      lindep=PETSC_FALSE;
  BV             cX;

  PetscFunctionBegin;
  if (d->W) cX = d->W; /* If left subspace exists, R <- orth(cY, R), nR[i] <- ||R[i]|| */
  else if (!(DVD_IS(d->sEP, DVD_EP_STD) && DVD_IS(d->sEP, DVD_EP_HERMITIAN))) cX = d->eps->V; /* If not HEP, R <- orth(cX, R), nR[i] <- ||R[i]|| */
  else cX = NULL; /* Otherwise, nR[i] <- ||R[i]|| */

  if (cX) {
    CHKERRQ(BVGetActiveColumns(cX,&l,&k));
    CHKERRQ(BVSetActiveColumns(cX,0,l));
    for (i=0;i<r_e-r_s;i++) {
      CHKERRQ(BVOrthogonalizeVec(cX,R[i],NULL,&d->nR[r_s+i],&lindep));
    }
    CHKERRQ(BVSetActiveColumns(cX,l,k));
    if (lindep || (PetscAbs(d->nR[r_s+i]) < PETSC_MACHINE_EPSILON)) {
      CHKERRQ(PetscInfo(d->eps,"The computed eigenvector residual %" PetscInt_FMT " is too low, %g!\n",r_s+i,(double)(d->nR[r_s+i])));
    }
  } else {
    for (i=0;i<r_e-r_s;i++) {
      CHKERRQ(VecNormBegin(R[i],NORM_2,&d->nR[r_s+i]));
    }
    for (i=0;i<r_e-r_s;i++) {
      CHKERRQ(VecNormEnd(R[i],NORM_2,&d->nR[r_s+i]));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d,dvdBlackboard *b,PetscBool borth,PetscBool harm)
{
  PetscBool      std_probl,her_probl,ind_probl;
  DSType         dstype;
  Vec            v1;

  PetscFunctionBegin;
  std_probl = DVD_IS(d->sEP,DVD_EP_STD)? PETSC_TRUE: PETSC_FALSE;
  her_probl = DVD_IS(d->sEP,DVD_EP_HERMITIAN)? PETSC_TRUE: PETSC_FALSE;
  ind_probl = DVD_IS(d->sEP,DVD_EP_INDEFINITE)? PETSC_TRUE: PETSC_FALSE;

  /* Setting configuration constrains */
  b->max_size_proj = PetscMax(b->max_size_proj,b->max_size_V);
  d->W_shift = d->B? PETSC_TRUE: PETSC_FALSE;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    d->max_size_P = b->max_size_P;
    d->max_size_proj = b->max_size_proj;
    /* Create a DS if the method works with Schur decompositions */
    d->calcPairs = dvd_calcpairs_proj;
    d->calcpairs_residual = dvd_calcpairs_res_0;
    d->calcpairs_proj_res = dvd_calcpairs_proj_res;
    d->calcpairs_selectPairs = dvd_calcpairs_selectPairs;
    /* Create and configure a DS for solving the projected problems */
    if (d->W) dstype = DSGNHEP;    /* If we use harmonics */
    else {
      if (ind_probl) dstype = DSGHIEP;
      else if (std_probl) dstype = her_probl? DSHEP : DSNHEP;
      else dstype = her_probl? DSGHEP : DSGNHEP;
    }
    CHKERRQ(DSSetType(d->eps->ds,dstype));
    CHKERRQ(DSAllocate(d->eps->ds,d->eps->ncv));
    /* Create various vector basis */
    if (harm) {
      CHKERRQ(BVDuplicateResize(d->eps->V,d->eps->ncv,&d->W));
      CHKERRQ(BVSetMatrix(d->W,NULL,PETSC_FALSE));
    } else d->W = NULL;
    CHKERRQ(BVDuplicateResize(d->eps->V,d->eps->ncv,&d->AX));
    CHKERRQ(BVSetMatrix(d->AX,NULL,PETSC_FALSE));
    CHKERRQ(BVDuplicateResize(d->eps->V,d->eps->ncv,&d->auxBV));
    CHKERRQ(BVSetMatrix(d->auxBV,NULL,PETSC_FALSE));
    if (d->B) {
      CHKERRQ(BVDuplicateResize(d->eps->V,d->eps->ncv,&d->BX));
      CHKERRQ(BVSetMatrix(d->BX,NULL,PETSC_FALSE));
    } else d->BX = NULL;
    CHKERRQ(MatCreateVecsEmpty(d->A,&v1,NULL));
    CHKERRQ(SlepcVecPoolCreate(v1,0,&d->auxV));
    CHKERRQ(VecDestroy(&v1));
    /* Create projected problem matrices */
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,d->eps->ncv,d->eps->ncv,NULL,&d->H));
    if (!std_probl) {
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,d->eps->ncv,d->eps->ncv,NULL,&d->G));
    } else d->G = NULL;
    if (her_probl) {
      CHKERRQ(MatSetOption(d->H,MAT_HERMITIAN,PETSC_TRUE));
      if (d->G) CHKERRQ(MatSetOption(d->G,MAT_HERMITIAN,PETSC_TRUE));
    }

    if (ind_probl) {
      CHKERRQ(PetscMalloc1(d->eps->ncv,&d->nBds));
    } else d->nBds = NULL;
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,d->eps->ncv,d->eps->ncv,NULL,&d->auxM));

    CHKERRQ(EPSDavidsonFLAdd(&d->startList,dvd_calcpairs_qz_start));
    CHKERRQ(EPSDavidsonFLAdd(&d->endList,EPSXDComputeDSConv));
    CHKERRQ(EPSDavidsonFLAdd(&d->destroyList,dvd_calcpairs_qz_d));
  }
  PetscFunctionReturn(0);
}
