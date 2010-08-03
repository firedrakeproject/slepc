/*
     This file contains some simple default routines for common QEP operations.  

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/qepimpl.h"   /*I "slepcqep.h" I*/
#include "private/slepcimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "QEPDestroy_Default"
PetscErrorCode QEPDestroy_Default(QEP qep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscFree(qep->data);CHKERRQ(ierr);

  /* free work vectors */
  ierr = QEPDefaultFreeWork(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDefaultGetWork"
/*
  QEPDefaultGetWork - Gets a number of work vectors.
 */
PetscErrorCode QEPDefaultGetWork(QEP qep, PetscInt nw)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;

  if (qep->nwork != nw) {
    if (qep->nwork > 0) {
      ierr = VecDestroyVecs(qep->work,qep->nwork); CHKERRQ(ierr);
    }
    qep->nwork = nw;
    ierr = PetscMalloc(nw*sizeof(Vec),&qep->work);CHKERRQ(ierr);
    for (i=0;i<nw;i++) {
      ierr = MatGetVecs(qep->M,PETSC_NULL,qep->work+i); CHKERRQ(ierr);
    }
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
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (qep->work)  {
    ierr = VecDestroyVecs(qep->work,qep->nwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDefaultConverged"
/*
  QEPDefaultConverged - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode QEPDefaultConverged(QEP qep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;
  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res;
  if (w > res) *errest = res / w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPAbsoluteConverged"
/*
  QEPAbsoluteConverged - Checks convergence absolutely.
*/
PetscErrorCode QEPAbsoluteConverged(QEP qep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPComputeVectors_Schur"
PetscErrorCode QEPComputeVectors_Schur(QEP qep)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   ncv,nconv,mout,info,one = 1; 
  PetscScalar    *Z,*work,tmp;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else 
  PetscReal      normi;
#endif
  PetscReal      norm;
  
  PetscFunctionBegin;
  ncv = PetscBLASIntCast(qep->ncv);
  nconv = PetscBLASIntCast(qep->nconv);
 
  ierr = PetscMalloc(nconv*nconv*sizeof(PetscScalar),&Z);CHKERRQ(ierr);
  ierr = PetscMalloc(3*nconv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(nconv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

  /* right eigenvectors */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","A",PETSC_NULL,&nconv,qep->T,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,&info);
#else
  LAPACKtrevc_("R","A",PETSC_NULL,&nconv,qep->T,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,rwork,&info);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

  /* normalize eigenvectors */
  for (i=0;i<qep->nconv;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (qep->eigi[i] != 0.0) {
      norm = BLASnrm2_(&nconv,Z+i*nconv,&one);
      normi = BLASnrm2_(&nconv,Z+(i+1)*nconv,&one);
      tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
      BLASscal_(&nconv,&tmp,Z+i*nconv,&one);
      BLASscal_(&nconv,&tmp,Z+(i+1)*nconv,&one);
      i++;     
    } else
#endif
    {
      norm = BLASnrm2_(&nconv,Z+i*nconv,&one);
      tmp = 1.0 / norm;
      BLASscal_(&nconv,&tmp,Z+i*nconv,&one);
    }
  }
  
  /* AV = V * Z */
  ierr = SlepcUpdateVectors(qep->nconv,qep->V,0,qep->nconv,Z,qep->nconv,PETSC_FALSE);CHKERRQ(ierr);
   
  ierr = PetscFree(Z);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
   PetscFunctionReturn(0);
#endif 
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
PetscErrorCode QEPKrylovConvergence(QEP qep,PetscInt kini,PetscInt nits,PetscScalar *S,PetscInt lds,PetscScalar *Q,PetscInt nv,PetscReal beta,PetscInt *kout,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscInt       k,marker;
  PetscScalar    re,im,*Z,*work2;
  PetscReal      resnorm;
  PetscTruth     iscomplex;

  PetscFunctionBegin;
  Z = work; work2 = work+2*nv;
  marker = -1;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = qep->eigr[k];
    im = qep->eigi[k];
    iscomplex = PETSC_FALSE;
    if (k<nv-1 && S[k+1+k*lds] != 0.0) iscomplex = PETSC_TRUE;
    /* residual norm */
    ierr = DenseSelectedEvec(S,lds,Q,Z,k,iscomplex,nv,work2);CHKERRQ(ierr);
    if (iscomplex) resnorm = beta*SlepcAbsEigenvalue(Z[nv-1],Z[2*nv-1]);
    else resnorm = beta*PetscAbsScalar(Z[nv-1]);
    /* error estimate */
    ierr = (*qep->conv_func)(qep,re,im,resnorm,&qep->errest[k],qep->conv_ctx);CHKERRQ(ierr);
    if (marker==-1 && qep->errest[k] >= qep->tol) marker = k;
    if (iscomplex) { qep->errest[k+1] = qep->errest[k]; k++; }
    if (marker!=-1 && !qep->trackall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;

  PetscFunctionReturn(0);
}


