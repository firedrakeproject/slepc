/*
      EPS routines related to memory management.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>   /*I "slepceps.h" I*/

#undef __FUNCT__
#define __FUNCT__ "EPSAllocateSolution"
/*
  EPSAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors. The argument extra is used for methods
  that require a working basis slightly larger than ncv.
*/
PetscErrorCode EPSAllocateSolution(EPS eps,PetscInt extra)
{
  PetscErrorCode ierr;
  PetscInt       newc,cnt,requested;

  PetscFunctionBegin;
  requested = eps->ncv + extra;
  if (eps->allocated_ncv != requested) {
    newc = PetscMax(0,requested-eps->allocated_ncv);
    ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
    cnt = 0;
    ierr = PetscMalloc(requested*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(requested*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    cnt += 2*newc*sizeof(PetscScalar);
    ierr = PetscMalloc(requested*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(requested*sizeof(PetscReal),&eps->errest_left);CHKERRQ(ierr);
    cnt += 2*newc*sizeof(PetscReal);
    ierr = PetscMalloc(requested*sizeof(PetscInt),&eps->perm);CHKERRQ(ierr);
    cnt += newc*sizeof(PetscInt);
    ierr = PetscLogObjectMemory((PetscObject)eps,cnt);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(eps->t,requested,&eps->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(eps,requested,eps->V);CHKERRQ(ierr);
    if (eps->leftvecs) {
      ierr = VecDuplicateVecs(eps->t,requested,&eps->W);CHKERRQ(ierr);
      ierr = PetscLogObjectParents(eps,requested,eps->W);CHKERRQ(ierr);
    }
    eps->allocated_ncv = requested;
  }
  /* The following cannot go in the above if, to avoid crash when ncv did not change */
  if (eps->arbitrary) {
    newc = PetscMax(0,requested-eps->allocated_ncv);
    ierr = PetscFree(eps->rr);CHKERRQ(ierr);
    ierr = PetscFree(eps->ri);CHKERRQ(ierr);
    ierr = PetscMalloc(requested*sizeof(PetscScalar),&eps->rr);CHKERRQ(ierr);
    ierr = PetscMalloc(requested*sizeof(PetscScalar),&eps->ri);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,2*newc*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSFreeSolution"
/*
  EPSFreeSolution - Free memory storage. This routine is related to
  EPSAllocateSolution().
*/
PetscErrorCode EPSFreeSolution(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (eps->allocated_ncv > 0) {
    ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
    ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest_left);CHKERRQ(ierr);
    ierr = PetscFree(eps->perm);CHKERRQ(ierr);
    ierr = PetscFree(eps->rr);CHKERRQ(ierr);
    ierr = PetscFree(eps->ri);CHKERRQ(ierr);
    ierr = VecDestroyVecs(eps->allocated_ncv,&eps->V);CHKERRQ(ierr);
    ierr = VecDestroyVecs(eps->allocated_ncv,&eps->W);CHKERRQ(ierr);
    eps->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}
