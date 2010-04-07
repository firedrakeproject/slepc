/*
     This file contains some simple default routines for common QEP operations.  

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

  PetscFunctionBegin;

  if (qep->nwork != nw) {
    if (qep->nwork > 0) {
      ierr = VecDestroyVecs(qep->work,qep->nwork); CHKERRQ(ierr);
    }
    qep->nwork = nw;
    ierr = VecDuplicateVecs(qep->V[0],nw,&qep->work); CHKERRQ(ierr);
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
  QEPDefaultConverged - Checks convergence with the relative error estimate.
*/
PetscErrorCode QEPDefaultConverged(QEP qep,PetscInt n,PetscInt k,PetscScalar* eigr,PetscScalar* eigi,PetscReal* errest,PetscTruth *conv,void *ctx)
{
  PetscInt  i;
  PetscReal w;
  
  PetscFunctionBegin;
  for (i=k; i<n; i++) {
    w = SlepcAbsEigenvalue(eigr[i],eigi[i]);
    if (w > errest[i]) errest[i] = errest[i] / w;
    if (errest[i] < qep->tol) conv[i] = PETSC_TRUE;
    else conv[i] = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPAbsoluteConverged"
/*
  QEPAbsoluteConverged - Checks convergence with the absolute error estimate.
*/
PetscErrorCode QEPAbsoluteConverged(QEP qep,PetscInt n,PetscInt k,PetscScalar* eigr,PetscScalar* eigi,PetscReal* errest,PetscTruth *conv,void *ctx)
{
  PetscInt  i;
  
  PetscFunctionBegin;
  for (i=k; i<n; i++) {
    if (errest[i] < qep->tol) conv[i] = PETSC_TRUE;
    else conv[i] = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

