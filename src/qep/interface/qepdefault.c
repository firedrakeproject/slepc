/*
     This file contains some simple default routines for common QEP operations.  

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/qepimpl.h>     /*I "slepcqep.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "QEPDefaultGetWork"
/*
  QEPDefaultGetWork - Gets a number of work vectors.
 */
PetscErrorCode QEPDefaultGetWork(QEP qep,PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (qep->nwork != nw) {
    ierr = VecDestroyVecs(qep->nwork,&qep->work);CHKERRQ(ierr);
    qep->nwork = nw;
    ierr = VecDuplicateVecs(qep->t,nw,&qep->work);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(qep,nw,qep->work);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDefaultFreeWork"
/*
  QEPDefaultFreeWork - Free work vectors.
 */
PetscErrorCode QEPDefaultFreeWork(QEP qep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(qep->nwork,&qep->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPConvergedDefault"
/*
  QEPConvergedDefault - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode QEPConvergedDefault(QEP qep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;

  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res/w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPConvergedAbsolute"
/*
  QEPConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode QEPConvergedAbsolute(QEP qep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPComputeVectors_Schur"
PetscErrorCode QEPComputeVectors_Schur(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       n,i,ld;
  PetscBLASInt   n_,one = 1; 
  PetscScalar    *Z,tmp;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      normi;
#endif
  PetscReal      norm;
  
  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(qep->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(qep->ds,&n,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  n_ = PetscBLASIntCast(n);

  /* right eigenvectors */
  ierr = DSVectors(qep->ds,DS_MAT_X,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* normalize eigenvectors */
  ierr = DSGetArray(qep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (qep->eigi[i] != 0.0) {
      norm = BLASnrm2_(&n_,Z+i*ld,&one);
      normi = BLASnrm2_(&n_,Z+(i+1)*ld,&one);
      tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
      BLASscal_(&n_,&tmp,Z+i*ld,&one);
      BLASscal_(&n_,&tmp,Z+(i+1)*ld,&one);
      i++;     
    } else
#endif
    {
      norm = BLASnrm2_(&n_,Z+i*ld,&one);
      tmp = 1.0 / norm;
      BLASscal_(&n_,&tmp,Z+i*ld,&one);
    }
  }
  ierr = DSRestoreArray(qep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  
  /* AV = V * Z */
  ierr = DSGetArray(qep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = SlepcUpdateVectors(n,qep->V,0,n,Z,ld,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DSRestoreArray(qep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPKrylovConvergence"
/*
   QEPKrylovConvergence - This is the analogue to EPSKrylovConvergence, but
   for quadratic Krylov methods.

   Differences:
   - Always non-symmetric
   - Does not check for STSHIFT
   - No correction factor
   - No support for true residual
*/
PetscErrorCode QEPKrylovConvergence(QEP qep,PetscBool getall,PetscInt kini,PetscInt nits,PetscInt nv,PetscReal beta,PetscInt *kout)
{
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,ld;
  PetscScalar    re,im;
  PetscReal      resnorm;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(qep->ds,&ld);CHKERRQ(ierr);
  marker = -1;
  if (qep->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = qep->eigr[k];
    im = qep->eigi[k];
    newk = k;
    ierr = DSVectors(qep->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    resnorm *= beta;
    /* error estimate */
    ierr = (*qep->conv_func)(qep,re,im,resnorm,&qep->errest[k],qep->conv_ctx);CHKERRQ(ierr);
    if (marker==-1 && qep->errest[k] >= qep->tol) marker = k;
    if (newk==k+1) { qep->errest[k+1] = qep->errest[k]; k++; }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  PetscFunctionReturn(0);
}

