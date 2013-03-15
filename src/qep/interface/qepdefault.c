/*
     This file contains some simple default routines for common QEP operations.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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
#define __FUNCT__ "QEPReset_Default"
PetscErrorCode QEPReset_Default(QEP qep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(qep->nwork,&qep->work);CHKERRQ(ierr);
  qep->nwork = 0;
  ierr = QEPFreeSolution(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetWorkVecs"
/*@
   QEPSetWorkVecs - Sets a number of work vectors into a QEP object

   Collective on QEP

   Input Parameters:
+  qep - quadratic eigensolver context
-  nw  - number of work vectors to allocate

   Developers Note:
   This is PETSC_EXTERN because it may be required by user plugin QEP
   implementations.

   Level: developer
@*/
PetscErrorCode QEPSetWorkVecs(QEP qep,PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (qep->nwork != nw) {
    ierr = VecDestroyVecs(qep->nwork,&qep->work);CHKERRQ(ierr);
    qep->nwork = nw;
    ierr = VecDuplicateVecs(qep->t,nw,&qep->work);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(qep,nw,qep->work);CHKERRQ(ierr);
  }
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
  PetscInt       n,ld;
  PetscScalar    *Z;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(qep->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(qep->ds,&n,NULL,NULL,NULL);CHKERRQ(ierr);

  /* right eigenvectors */
  ierr = DSVectors(qep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);

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
    ierr = (*qep->converged)(qep,re,im,resnorm,&qep->errest[k],qep->convergedctx);CHKERRQ(ierr);
    if (marker==-1 && qep->errest[k] >= qep->tol) marker = k;
    if (newk==k+1) {
      qep->errest[k+1] = qep->errest[k];
      k++;
    }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  PetscFunctionReturn(0);
}

