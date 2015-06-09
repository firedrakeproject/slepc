/*
   Common subroutines for all Krylov-type PEP solvers.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "PEPExtractEigenPairs"
PetscErrorCode PEPExtractEigenPairs(PEP pep,PetscInt k,PetscInt sr,PetscScalar *S,PetscInt ld)
{
  PetscErrorCode ierr;
  PetscInt       i,j,deg=pep->nmat-1,lds,idxcpy=0,ldds;
  PetscScalar    *X,*er,*ei,*SS,*vals,*ivals,sone=1.0,szero=0.0,*yi,*yr,*tr,*ti,alpha,t;
  PetscBLASInt   k_,sr_,lds_,one=1;
  PetscBool      flg;
  PetscReal      norm,max;
  Vec            xr,xi,w[4];

  PetscFunctionBegin;
  if (k==0 || pep->extract==PEP_EXTRACT_NONE) PetscFunctionReturn(0);
  lds = deg*ld;
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);
  ierr = PetscMalloc5(k,&er,k,&ei,k*sr,&SS,pep->nmat,&vals,pep->nmat,&ivals);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    er[i] = pep->eigr[i];
    ei[i] = pep->eigi[i];
  }
  if (flg) {
    for (i=0;i<k;i++) {
      er[i] = pep->sfactor*pep->eigr[i];
      ei[i] = pep->sfactor*pep->eigi[i];
    }
  }
  ierr = STBackTransform(pep->st,k,er,ei);CHKERRQ(ierr);

  ierr = DSVectors(pep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);

  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);

  switch (pep->extract) {
  case PEP_EXTRACT_NONE:
    break;
  case PEP_EXTRACT_NORM:
    for (i=0;i<k;i++) {
      ierr = PEPEvaluateBasis(pep,er[i],ei[i],vals,ivals);CHKERRQ(ierr);
      max = 1.0;
      for (j=1;j<deg;j++) {
        norm = SlepcAbsEigenvalue(vals[j],ivals[j]);
        if (max < norm) { max = norm; idxcpy = j; }
      }
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*sr,&one));
#if !defined(PETSC_USE_COMPLEX)
      if (PetscRealPart(ei[i])!=0.0) {
        i++;
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*sr,&one));
      }
#endif
    }
    break;
  case PEP_EXTRACT_RESIDUAL:
    ierr = VecDuplicate(pep->work[0],&xr);CHKERRQ(ierr);
    ierr = VecDuplicate(pep->work[0],&w[0]);CHKERRQ(ierr);
    ierr = VecDuplicate(pep->work[0],&w[1]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = VecDuplicate(pep->work[0],&w[2]);CHKERRQ(ierr);
    ierr = VecDuplicate(pep->work[0],&w[3]);CHKERRQ(ierr);
    ierr = VecDuplicate(pep->work[0],&xi);CHKERRQ(ierr);
#else
    xi = NULL;
#endif
    for (i=0;i<k;i++) {
      max = 0.0;
      for (j=0;j<deg;j++) {
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&sone,S+j*ld,&lds_,X+i*ldds,&one,&szero,SS+i*sr,&one));
        ierr = BVMultVec(pep->V,1.0,0.0,xr,SS+i*sr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&sone,S+j*ld,&lds_,X+(i+1)*ldds,&one,&szero,SS+i*sr,&one));
        ierr = BVMultVec(pep->V,1.0,0.0,xi,SS+i*sr);CHKERRQ(ierr);
#endif
        ierr = PEPComputeResidualNorm_Private(pep,er[i],ei[i],xr,xi,w,&norm);CHKERRQ(ierr);
        if (norm>max) { max = norm; idxcpy=j; }
      }
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*sr,&one));
#if !defined(PETSC_USE_COMPLEX)
      if (PetscRealPart(ei[i])!=0.0) {
        i++;
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&sone,S+idxcpy*ld,&lds_,X+i*ldds,&one,&szero,SS+i*sr,&one));
      }
#endif
    }
    ierr = VecDestroy(&xr);CHKERRQ(ierr);
    ierr = VecDestroy(&w[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&w[1]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = VecDestroy(&w[2]);CHKERRQ(ierr);
    ierr = VecDestroy(&w[3]);CHKERRQ(ierr);
    ierr = VecDestroy(&xi);CHKERRQ(ierr);
#endif
    break;
  case PEP_EXTRACT_STRUCTURED:
    ierr = PetscMalloc2(sr,&tr,sr,&ti);CHKERRQ(ierr);
    for (i=0;i<k;i++) {
      ierr = PetscMemzero(SS+i*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      if (ei[i]!=0.0) {
        ierr = PetscMemzero(SS+(i+1)*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
      }
#endif
      t = 0.0;
      ierr = PEPEvaluateBasis(pep,er[i],ei[i],vals,ivals);CHKERRQ(ierr);
      yr = X+i*ldds; yi = NULL;
      for (j=0;j<deg;j++) {
        alpha = PetscConj(vals[j]);
#if !defined(PETSC_USE_COMPLEX)
        if (ei[i]!=0.0) {
          ierr = PetscMemzero(tr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
          PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&sr_,&vals[j],X+i*ldds,&one,tr,&one));
          PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&sr_,&ivals[j],X+(i+1)*ldds,&one,tr,&one));
          yr = tr;
          ierr = PetscMemzero(ti,sr*sizeof(PetscScalar));CHKERRQ(ierr);
          PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&sr_,&vals[j],X+(i+1)*ldds,&one,ti,&one));
          alpha = -ivals[j];
          PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&sr_,&alpha,X+i*ldds,&one,ti,&one));
          yi = ti;
          alpha = 1.0;
        } else { yr = X+i*ldds; yi = NULL;}
#endif
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&alpha,S+j*ld,&lds_,yr,&one,&sone,SS+i*sr,&one));
        t += SlepcAbsEigenvalue(vals[j],ivals[j])*SlepcAbsEigenvalue(vals[j],ivals[j]);
        if (yi) {
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&sr_,&k_,&alpha,S+j*ld,&lds_,yi,&one,&sone,SS+(i+1)*sr,&one));
        }
      }
      t = 1.0/t;
      PetscStackCallBLAS("BLASscal",BLASscal_(&sr_,&t,SS+i*sr,&one));
      if (yi) {
        PetscStackCallBLAS("BLASscal",BLASscal_(&sr_,&t,SS+(i+1)*sr,&one));
        i++;
      }
    }
    ierr = PetscFree2(tr,ti);CHKERRQ(ierr);
    break;
  default:
      SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Extraction not implemented in this solver");
  }
  for (i=0;i<k;i++) {
    ierr = PetscMemcpy(S+i*lds,SS+i*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscFree5(er,ei,SS,vals,ivals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

