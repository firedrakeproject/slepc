/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Common subroutines for all Krylov-type PEP solvers
*/

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include <slepcblaslapack.h>
#include "pepkrylov.h"

PetscErrorCode PEPExtractVectors_TOAR(PEP pep)
{
  PetscInt          i,j,nq,deg=pep->nmat-1,lds,idxcpy=0,ldds,k,ld;
  PetscScalar       *X,*er,*ei,*SS,*vals,*ivals,sone=1.0,szero=0.0,*yi,*yr,*tr,*ti,alpha,*pS0;
  const PetscScalar *S;
  PetscBLASInt      k_,nq_,lds_,one=1,ldds_,cols,info,zero=0;
  PetscBool         flg;
  PetscReal         norm,max,t,factor=1.0,done=1.0;
  Vec               xr,xi,w[4];
  PEP_TOAR          *ctx = (PEP_TOAR*)pep->data;
  Mat               S0,MS;

  PetscFunctionBegin;
  CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
  CHKERRQ(MatDenseGetArrayRead(MS,&S));
  CHKERRQ(BVGetSizes(pep->V,NULL,NULL,&ld));
  CHKERRQ(BVGetActiveColumns(pep->V,NULL,&nq));
  k = pep->nconv;
  if (k==0) PetscFunctionReturn(0);
  lds = deg*ld;
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ldds));
  CHKERRQ(PetscCalloc5(k,&er,k,&ei,nq*k,&SS,pep->nmat,&vals,pep->nmat,&ivals));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg) factor = pep->sfactor;
  for (i=0;i<k;i++) {
    er[i] = factor*pep->eigr[i];
    ei[i] = factor*pep->eigi[i];
  }
  CHKERRQ(STBackTransform(pep->st,k,er,ei));

  CHKERRQ(DSVectors(pep->ds,DS_MAT_X,NULL,NULL));
  CHKERRQ(DSGetArray(pep->ds,DS_MAT_X,&X));

  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(nq,&nq_));
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(ldds,&ldds_));

  if (pep->extract==PEP_EXTRACT_NONE || pep->refine==PEP_REFINE_MULTIPLE) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&nq_,&k_,&k_,&sone,S,&lds_,X,&ldds_,&szero,SS,&nq_));
  } else {
    switch (pep->extract) {
    case PEP_EXTRACT_NONE:
      break;
    case PEP_EXTRACT_NORM:
      for (i=0;i<k;i++) {
        CHKERRQ(PEPEvaluateBasis(pep,er[i],ei[i],vals,ivals));
        max = 1.0;
        for (j=1;j<deg;j++) {
          norm = SlepcAbsEigenvalue(vals[j],ivals[j]);
          if (max < norm) { max = norm; idxcpy = j; }
        }
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*nq,&one));
#if !defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(ei[i])!=0.0) {
          i++;
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*nq,&one));
        }
#endif
      }
      break;
    case PEP_EXTRACT_RESIDUAL:
      CHKERRQ(VecDuplicate(pep->work[0],&xr));
      CHKERRQ(VecDuplicate(pep->work[0],&w[0]));
      CHKERRQ(VecDuplicate(pep->work[0],&w[1]));
#if !defined(PETSC_USE_COMPLEX)
      CHKERRQ(VecDuplicate(pep->work[0],&w[2]));
      CHKERRQ(VecDuplicate(pep->work[0],&w[3]));
      CHKERRQ(VecDuplicate(pep->work[0],&xi));
#else
      xi = NULL;
#endif
      for (i=0;i<k;i++) {
        max = 0.0;
        for (j=0;j<deg;j++) {
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&sone,S+j*ld,&lds_,X+i*ldds,&one,&szero,SS+i*nq,&one));
          CHKERRQ(BVMultVec(pep->V,1.0,0.0,xr,SS+i*nq));
#if !defined(PETSC_USE_COMPLEX)
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&sone,S+j*ld,&lds_,X+(i+1)*ldds,&one,&szero,SS+i*nq,&one));
          CHKERRQ(BVMultVec(pep->V,1.0,0.0,xi,SS+i*nq));
#endif
          CHKERRQ(PEPComputeResidualNorm_Private(pep,er[i],ei[i],xr,xi,w,&norm));
          if (norm>max) { max = norm; idxcpy=j; }
        }
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*nq,&one));
#if !defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(ei[i])!=0.0) {
          i++;
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*nq,&one));
        }
#endif
      }
      CHKERRQ(VecDestroy(&xr));
      CHKERRQ(VecDestroy(&w[0]));
      CHKERRQ(VecDestroy(&w[1]));
#if !defined(PETSC_USE_COMPLEX)
      CHKERRQ(VecDestroy(&w[2]));
      CHKERRQ(VecDestroy(&w[3]));
      CHKERRQ(VecDestroy(&xi));
#endif
      break;
    case PEP_EXTRACT_STRUCTURED:
      CHKERRQ(PetscMalloc2(k,&tr,k,&ti));
      for (i=0;i<k;i++) {
        t = 0.0;
        CHKERRQ(PEPEvaluateBasis(pep,er[i],ei[i],vals,ivals));
        yr = X+i*ldds; yi = NULL;
#if !defined(PETSC_USE_COMPLEX)
        if (ei[i]!=0.0) { yr = tr; yi = ti; }
#endif
        for (j=0;j<deg;j++) {
          alpha = PetscConj(vals[j]);
#if !defined(PETSC_USE_COMPLEX)
          if (ei[i]!=0.0) {
            CHKERRQ(PetscArrayzero(tr,k));
            PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&k_,&vals[j],X+i*ldds,&one,tr,&one));
            PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&k_,&ivals[j],X+(i+1)*ldds,&one,tr,&one));
            CHKERRQ(PetscArrayzero(ti,k));
            PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&k_,&vals[j],X+(i+1)*ldds,&one,ti,&one));
            alpha = -ivals[j];
            PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&k_,&alpha,X+i*ldds,&one,ti,&one));
            alpha = 1.0;
          }
#endif
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&alpha,S+j*ld,&lds_,yr,&one,&sone,SS+i*nq,&one));
          t += SlepcAbsEigenvalue(vals[j],ivals[j])*SlepcAbsEigenvalue(vals[j],ivals[j]);
          if (yi) {
            PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&nq_,&k_,&alpha,S+j*ld,&lds_,yi,&one,&sone,SS+(i+1)*nq,&one));
          }
        }
        cols = yi? 2: 1;
        PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&t,&done,&nq_,&cols,SS+i*nq,&nq_,&info));
        SlepcCheckLapackInfo("lascl",info);
        if (yi) i++;
      }
      CHKERRQ(PetscFree2(tr,ti));
      break;
    }
  }

  /* update vectors V = V*S */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nq,k,NULL,&S0));
  CHKERRQ(MatDenseGetArrayWrite(S0,&pS0));
  for (i=0;i<k;i++) {
    CHKERRQ(PetscArraycpy(pS0+i*nq,SS+i*nq,nq));
  }
  CHKERRQ(MatDenseRestoreArrayWrite(S0,&pS0));
  CHKERRQ(BVMultInPlace(pep->V,S0,0,k));
  CHKERRQ(MatDestroy(&S0));
  CHKERRQ(PetscFree5(er,ei,SS,vals,ivals));
  CHKERRQ(MatDenseRestoreArrayRead(MS,&S));
  CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
  PetscFunctionReturn(0);
}

/*
   PEPKrylovConvergence - This is the analogue to EPSKrylovConvergence, but
   for polynomial Krylov methods.

   Differences:
   - Always non-symmetric
   - Does not check for STSHIFT
   - No correction factor
   - No support for true residual
*/
PetscErrorCode PEPKrylovConvergence(PEP pep,PetscBool getall,PetscInt kini,PetscInt nits,PetscReal beta,PetscInt *kout)
{
  PetscInt       k,newk,marker,inside;
  PetscScalar    re,im;
  PetscReal      resnorm;
  PetscBool      istrivial;

  PetscFunctionBegin;
  CHKERRQ(RGIsTrivial(pep->rg,&istrivial));
  marker = -1;
  if (pep->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = pep->eigr[k];
    im = pep->eigi[k];
    if (PetscUnlikely(!istrivial)) {
      CHKERRQ(STBackTransform(pep->st,1,&re,&im));
      CHKERRQ(RGCheckInside(pep->rg,1,&re,&im,&inside));
      if (marker==-1 && inside<0) marker = k;
      re = pep->eigr[k];
      im = pep->eigi[k];
    }
    newk = k;
    CHKERRQ(DSVectors(pep->ds,DS_MAT_X,&newk,&resnorm));
    resnorm *= beta;
    /* error estimate */
    CHKERRQ((*pep->converged)(pep,re,im,resnorm,&pep->errest[k],pep->convergedctx));
    if (marker==-1 && pep->errest[k] >= pep->tol) marker = k;
    if (PetscUnlikely(newk==k+1)) {
      pep->errest[k+1] = pep->errest[k];
      k++;
    }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  PetscFunctionReturn(0);
}
