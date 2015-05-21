/*
     This file contains some simple default routines for common PEP operations.

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

#include <slepc-private/pepimpl.h>     /*I "slepcpep.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PEPSetWorkVecs"
/*@
   PEPSetWorkVecs - Sets a number of work vectors into a PEP object.

   Collective on PEP

   Input Parameters:
+  pep - polynomial eigensolver context
-  nw  - number of work vectors to allocate

   Developers Note:
   This is PETSC_EXTERN because it may be required by user plugin PEP
   implementations.

   Level: developer
@*/
PetscErrorCode PEPSetWorkVecs(PEP pep,PetscInt nw)
{
  PetscErrorCode ierr;
  Vec            t;

  PetscFunctionBegin;
  if (pep->nwork != nw) {
    ierr = VecDestroyVecs(pep->nwork,&pep->work);CHKERRQ(ierr);
    pep->nwork = nw;
    ierr = BVGetColumn(pep->V,0,&t);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(t,nw,&pep->work);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,0,&t);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(pep,nw,pep->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPConvergedEigRelative"
/*
  PEPConvergedEigRelative - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode PEPConvergedEigRelative(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;

  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res/w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPConvergedNormRelative"
/*
  PEPConvergedNormRelative - Checks convergence relative to the matrix norms.
*/
PetscErrorCode PEPConvergedNormRelative(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      w=0.0;
  PetscScalar    t[20],*vals=t,*ivals=NULL;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    it[20];
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ivals = it;
#endif
  if (pep->nmat>20) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc2(pep->nmat,&vals,pep->nmat,&ivals);CHKERRQ(ierr);
#else
    ierr = PetscMalloc1(pep->nmat,&vals);CHKERRQ(ierr);
#endif
  }
  ierr = PEPEvaluateBasis(pep,eigr,eigi,vals,ivals);CHKERRQ(ierr);
  for (i=0;i<pep->nmat;i++) w += SlepcAbsEigenvalue(vals[i],ivals[i])*pep->nrma[i];
  *errest = res/w;
  if (pep->nmat>20) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscFree2(vals,ivals);CHKERRQ(ierr);
#else
    ierr = PetscFree(vals);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPConvergedAbsolute"
/*
  PEPConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode PEPConvergedAbsolute(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeVectors_Schur"
PetscErrorCode PEPComputeVectors_Schur(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       n,i;
  Mat            Z;
  Vec            v;
#if !defined(PETSC_USE_COMPLEX)
  Vec            v1;
  PetscScalar    tmp;
  PetscReal      norm,normi;
#endif

  PetscFunctionBegin;
  ierr = DSGetDimensions(pep->ds,&n,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSVectors(pep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetMat(pep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,n);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,Z,0,n);CHKERRQ(ierr);
  ierr = MatDestroy(&Z);CHKERRQ(ierr);

  /* Fix eigenvectors if balancing was used */
  if ((pep->scale==PEP_SCALE_DIAGONAL || pep->scale==PEP_SCALE_BOTH) && pep->Dr && (pep->refine!=PEP_REFINE_MULTIPLE)) {
    for (i=0;i<n;i++) {
      ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
    }
  }

  /* normalization */
  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (pep->eigi[i] != 0.0) {
      ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,i+1,&v1);CHKERRQ(ierr);
      ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecNorm(v1,NORM_2,&normi);CHKERRQ(ierr);
      tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
      ierr = VecScale(v,tmp);CHKERRQ(ierr);
      ierr = VecScale(v1,tmp);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i+1,&v1);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = VecNormalize(v,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeVectors_Indefinite"
PetscErrorCode PEPComputeVectors_Indefinite(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       n,i;
  Mat            Z;
  Vec            v;
#if !defined(PETSC_USE_COMPLEX)
  Vec            v1;
  PetscScalar    tmp;
  PetscReal      norm,normi;
#endif

  PetscFunctionBegin;
  ierr = DSGetDimensions(pep->ds,&n,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSVectors(pep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetMat(pep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,n);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,Z,0,n);CHKERRQ(ierr);
  ierr = MatDestroy(&Z);CHKERRQ(ierr);

  /* normalization */
  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (pep->eigi[i] != 0.0) {
      ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,i+1,&v1);CHKERRQ(ierr);
      ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecNorm(v1,NORM_2,&normi);CHKERRQ(ierr);
      tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
      ierr = VecScale(v,tmp);CHKERRQ(ierr);
      ierr = VecScale(v1,tmp);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i+1,&v1);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = VecNormalize(v,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPKrylovConvergence"
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
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,ld,inside;
  PetscScalar    re,im;
  PetscReal      resnorm;
  PetscBool      istrivial;

  PetscFunctionBegin;
  ierr = RGIsTrivial(pep->rg,&istrivial);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  marker = -1;
  if (pep->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = pep->eigr[k];
    im = pep->eigi[k];
    if (!istrivial || pep->conv==PEP_CONV_NORM) {
      ierr = STBackTransform(pep->st,1,&re,&im);CHKERRQ(ierr);
    }
    if (!istrivial) {
      ierr = RGCheckInside(pep->rg,1,&re,&im,&inside);CHKERRQ(ierr);
      if (marker==-1 && inside<=0) marker = k;
      if (!pep->conv==PEP_CONV_NORM) {  /* make sure pep->converged below uses the right value */
        re = pep->eigr[k];
        im = pep->eigi[k];
      }
    }
    newk = k;
    ierr = DSVectors(pep->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    resnorm *= beta;
    /* error estimate */
    ierr = (*pep->converged)(pep,re,im,resnorm,&pep->errest[k],pep->convergedctx);CHKERRQ(ierr);
    if (marker==-1 && pep->errest[k] >= pep->tol) marker = k;
    if (newk==k+1) {
      pep->errest[k+1] = pep->errest[k];
      k++;
    }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPBuildDiagonalScaling"
/*
  PEPBuildDiagonalScaling - compute two diagonal matrices to be applied for balancing 
  in polynomial eigenproblems.
*/
PetscErrorCode PEPBuildDiagonalScaling(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       it,i,j,k,nmat,nr,e,nz,lst,lend,nc=0,*cols,emax,emin,emaxl,eminl;
  const PetscInt *cidx,*ridx;
  Mat            M,*T,A;
  PetscMPIInt    n;
  PetscBool      cont=PETSC_TRUE,flg=PETSC_FALSE;
  PetscScalar    *array,*Dr,*Dl,t;
  PetscReal      l2,d,*rsum,*aux,*csum,w=1.0;
  MatStructure   str;
  MatInfo        info;

  PetscFunctionBegin;
  l2 = 2*PetscLogReal(2.0);
  nmat = pep->nmat;
  ierr = PetscMPIIntCast(pep->n,&n);
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = PetscMalloc1(nmat,&T);CHKERRQ(ierr);
  for (k=0;k<nmat;k++) {
    ierr = STGetTOperators(pep->st,k,&T[k]);CHKERRQ(ierr);
  }
  /* Form local auxiliar matrix M */
  ierr = PetscObjectTypeCompareAny((PetscObject)T[0],&cont,MATMPIAIJ,MATSEQAIJ);CHKERRQ(ierr);
  if (!cont) SETERRQ(PetscObjectComm((PetscObject)T[0]),PETSC_ERR_SUP,"Only for MPIAIJ or SEQAIJ matrix types");
  ierr = PetscObjectTypeCompare((PetscObject)T[0],MATMPIAIJ,&cont);CHKERRQ(ierr);
  if (cont) {
    ierr = MatMPIAIJGetLocalMat(T[0],MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
    flg = PETSC_TRUE; 
  } else {
    ierr = MatDuplicate(T[0],MAT_COPY_VALUES,&M);CHKERRQ(ierr);
  }
  ierr = MatGetInfo(M,MAT_LOCAL,&info);CHKERRQ(ierr);
  nz = info.nz_used;
  ierr = MatSeqAIJGetArray(M,&array);CHKERRQ(ierr);
  for (i=0;i<nz;i++) {
    t = PetscAbsScalar(array[i]);
    array[i] = t*t;
  }
  ierr = MatSeqAIJRestoreArray(M,&array);CHKERRQ(ierr);
  for (k=1;k<nmat;k++) {
    if (flg) {
      ierr = MatMPIAIJGetLocalMat(T[k],MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    } else {
      if (str==SAME_NONZERO_PATTERN) {
        ierr = MatCopy(T[k],A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatDuplicate(T[k],MAT_COPY_VALUES,&A);CHKERRQ(ierr);
      }
    }
    ierr = MatGetInfo(A,MAT_LOCAL,&info);CHKERRQ(ierr);
    nz = info.nz_used;
    ierr = MatSeqAIJGetArray(A,&array);CHKERRQ(ierr);
    for (i=0;i<nz;i++) {
      t = PetscAbsScalar(array[i]);
      array[i] = t*t;
    }
    ierr = MatSeqAIJRestoreArray(A,&array);CHKERRQ(ierr);
    w *= pep->slambda*pep->slambda*pep->sfactor;
    ierr = MatAXPY(M,w,A,str);CHKERRQ(ierr);
    if (flg || str!=SAME_NONZERO_PATTERN || k==nmat-2) {
      ierr = MatDestroy(&A);CHKERRQ(ierr);
    } 
  }
  ierr = MatGetRowIJ(M,0,PETSC_FALSE,PETSC_FALSE,&nr,&ridx,&cidx,&cont);CHKERRQ(ierr);
  if (!cont) SETERRQ(PetscObjectComm((PetscObject)T[0]), PETSC_ERR_SUP,"It is not possible to compute scaling diagonals for these PEP matrices");
  ierr = MatGetInfo(M,MAT_LOCAL,&info);CHKERRQ(ierr);
  nz = info.nz_used;
  ierr = VecGetOwnershipRange(pep->Dl,&lst,&lend);CHKERRQ(ierr);
  ierr = PetscMalloc4(nr,&rsum,pep->n,&csum,pep->n,&aux,PetscMin(pep->n-lend+lst,nz),&cols);CHKERRQ(ierr);
  ierr = VecSet(pep->Dr,1.0);CHKERRQ(ierr);
  ierr = VecSet(pep->Dl,1.0);CHKERRQ(ierr);
  ierr = VecGetArray(pep->Dl,&Dl);CHKERRQ(ierr);
  ierr = VecGetArray(pep->Dr,&Dr);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(M,&array);CHKERRQ(ierr);
  ierr = PetscMemzero(aux,pep->n*sizeof(PetscReal));CHKERRQ(ierr);
  for (j=0;j<nz;j++) {
    /* Search non-zero columns outsize lst-lend */
    if (aux[cidx[j]]==0 && (cidx[j]<lst || lend<=cidx[j])) cols[nc++] = cidx[j];
    /* Local column sums */
    aux[cidx[j]] += PetscAbsScalar(array[j]);
  }
  for (it=0;it<pep->sits && cont;it++) {
    emaxl = 0; eminl = 0;
    /* Column sum  */    
    if (it>0) { /* it=0 has been already done*/
      ierr = MatSeqAIJGetArray(M,&array);CHKERRQ(ierr);
      ierr = PetscMemzero(aux,pep->n*sizeof(PetscReal));CHKERRQ(ierr);
      for (j=0;j<nz;j++) aux[cidx[j]] += PetscAbsScalar(array[j]);
      ierr = MatSeqAIJRestoreArray(M,&array);CHKERRQ(ierr); 
    }
    ierr = MPI_Allreduce(aux,csum,n,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)pep->Dr));
    /* Update Dr */
    for (j=lst;j<lend;j++) {
      d = PetscLogReal(csum[j])/l2;
      e = -(PetscInt)((d < 0)?(d-0.5):(d+0.5));
      d = PetscPowReal(2.0,e);
      Dr[j-lst] *= d;
      aux[j] = d*d;
      emaxl = PetscMax(emaxl,e);
      eminl = PetscMin(eminl,e);
    }
    for (j=0;j<nc;j++) {
      d = PetscLogReal(csum[cols[j]])/l2;
      e = -(PetscInt)((d < 0)?(d-0.5):(d+0.5));
      d = PetscPowReal(2.0,e);
      aux[cols[j]] = d*d;
      emaxl = PetscMax(emaxl,e);
      eminl = PetscMin(eminl,e);
    }
    /* Scale M */
    ierr = MatSeqAIJGetArray(M,&array);CHKERRQ(ierr);
    for (j=0;j<nz;j++) {
      array[j] *= aux[cidx[j]];
    }
    ierr = MatSeqAIJRestoreArray(M,&array);CHKERRQ(ierr);
    /* Row sum */    
    ierr = PetscMemzero(rsum,nr*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = MatSeqAIJGetArray(M,&array);CHKERRQ(ierr);
    for (i=0;i<nr;i++) {
      for (j=ridx[i];j<ridx[i+1];j++) rsum[i] += PetscAbsScalar(array[j]);
      /* Update Dl */
      d = PetscLogReal(rsum[i])/l2;
      e = -(PetscInt)((d < 0)?(d-0.5):(d+0.5));
      d = PetscPowReal(2.0,e);
      Dl[i] *= d;
      /* Scale M */
      for (j=ridx[i];j<ridx[i+1];j++) array[j] *= d*d;
      emaxl = PetscMax(emaxl,e);
      eminl = PetscMin(eminl,e);      
    }
    ierr = MatSeqAIJRestoreArray(M,&array);CHKERRQ(ierr);  
    /* Compute global max and min */
    ierr = MPI_Allreduce(&emaxl,&emax,1,MPIU_INT,MPIU_MAX,PetscObjectComm((PetscObject)pep->Dl));
    ierr = MPI_Allreduce(&eminl,&emin,1,MPIU_INT,MPIU_MIN,PetscObjectComm((PetscObject)pep->Dl));
    if (emax<=emin+2) cont = PETSC_FALSE;
  }
  ierr = VecRestoreArray(pep->Dr,&Dr);CHKERRQ(ierr);
  ierr = VecRestoreArray(pep->Dl,&Dl);CHKERRQ(ierr);
  /* Free memory*/
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = PetscFree4(rsum,csum,aux,cols);CHKERRQ(ierr);
  ierr = PetscFree(T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeScaleFactor"
/*
   PEPComputeScaleFactor - compute sfactor as described in [Betcke 2008].
*/
PetscErrorCode PEPComputeScaleFactor(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      has0,has1,flg;
  PetscReal      norm0,norm1;
  Mat            T[2];
  PEPBasis       basis;

  PetscFunctionBegin;
  if (pep->scale==PEP_SCALE_NONE || pep->scale==PEP_SCALE_DIAGONAL) {  /* no scalar scaling */
    pep->sfactor = 1.0;
    PetscFunctionReturn(0);
  }
  if (pep->sfactor_set) PetscFunctionReturn(0);  /* user provided value */
  ierr = PEPGetBasis(pep,&basis);CHKERRQ(ierr);
  if (basis==PEP_BASIS_MONOMIAL) {
    ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = STGetTOperators(pep->st,0,&T[0]);CHKERRQ(ierr);
      ierr = STGetTOperators(pep->st,pep->nmat-1,&T[1]);CHKERRQ(ierr);
    } else {
      T[0] = pep->A[0];
      T[1] = pep->A[pep->nmat-1];
    }
    if (pep->nmat>2) {
      ierr = MatHasOperation(T[0],MATOP_NORM,&has0);CHKERRQ(ierr);
      ierr = MatHasOperation(T[1],MATOP_NORM,&has1);CHKERRQ(ierr);
      if (has0 && has1) {
        ierr = MatNorm(T[0],NORM_INFINITY,&norm0);CHKERRQ(ierr);
        ierr = MatNorm(T[1],NORM_INFINITY,&norm1);CHKERRQ(ierr);
        pep->sfactor = PetscPowReal(norm0/norm1,1.0/(pep->nmat-1));
      } else {
        pep->sfactor = 1.0;
      }
    }
  } else pep->sfactor = 1.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPBasisCoefficients"
/*
   PEPBasisCoefficients - compute polynomial basis coefficients
*/
PetscErrorCode PEPBasisCoefficients(PEP pep,PetscReal *pbc)
{
  PetscReal *ca,*cb,*cg;
  PetscInt  k,nmat=pep->nmat;
  
  PetscFunctionBegin;
  ca = pbc;
  cb = pbc+nmat;
  cg = pbc+2*nmat;
  switch (pep->basis) {
  case PEP_BASIS_MONOMIAL:
    for (k=0;k<nmat;k++) {
      ca[k] = 1.0; cb[k] = 0.0; cg[k] = 0.0;
    }
    break;
  case PEP_BASIS_CHEBYSHEV1:
    ca[0] = 1.0; cb[0] = 0.0; cg[0] = 0.0;
    for (k=1;k<nmat;k++) {
      ca[k] = .5; cb[k] = 0.0; cg[k] = .5;
    }
    break;
  case PEP_BASIS_CHEBYSHEV2:
    ca[0] = .5; cb[0] = 0.0; cg[0] = 0.0;
    for (k=1;k<nmat;k++) {
      ca[k] = .5; cb[k] = 0.0; cg[k] = .5;
    }    
    break;
  case PEP_BASIS_LEGENDRE:
    ca[0] = 1.0; cb[0] = 0.0; cg[0] = 0.0;
    for (k=1;k<nmat;k++) {
      ca[k] = k+1; cb[k] = -2*k; cg[k] = k;
    }
    break;
  case PEP_BASIS_LAGUERRE:
    ca[0] = -1.0; cb[0] = 0.0; cg[0] = 0.0;
    for (k=1;k<nmat;k++) {
      ca[k] = -(k+1); cb[k] = 2*k+1; cg[k] = -k;
    }
    break;
  case PEP_BASIS_HERMITE:
    ca[0] = .5; cb[0] = 0.0; cg[0] = 0.0;
    for (k=1;k<nmat;k++) {
      ca[k] = .5; cb[k] = 0.0; cg[k] = -k;
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'basis' value");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPEvaluateBasis"
/*
   PEPEvaluateBasis - evaluate the polynomial basis on a given parameter sigma
*/
PetscErrorCode PEPEvaluateBasis(PEP pep,PetscScalar sigma,PetscScalar isigma,PetscScalar *vals,PetscScalar *ivals)
{
  PetscInt   nmat=pep->nmat,k;
  PetscReal  *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat;
  
  PetscFunctionBegin;
  if (ivals) for (k=0;k<nmat;k++) ivals[k] = 0.0;
  vals[0] = 1.0;
  vals[1] = (sigma-b[0])/a[0];
#if !defined(PETSC_USE_COMPLEX)
  if (ivals) ivals[1] = isigma/a[0];
#endif
  for (k=2;k<nmat;k++) {
    vals[k] = ((sigma-b[k-1])*vals[k-1]-g[k-1]*vals[k-2])/a[k-1];
    if (ivals) vals[k] -= isigma*ivals[k-1]/a[k-1];
#if !defined(PETSC_USE_COMPLEX)
    if (ivals) ivals[k] = ((sigma-b[k-1])*ivals[k-1]+isigma*vals[k-1]-g[k-1]*ivals[k-2])/a[k-1];
#endif
  }
  PetscFunctionReturn(0);
}

