/*
      EPS routines related to memory management.

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

#include <slepc-private/epsimpl.h>   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSAllocateSolution"
/*
  EPSAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors. All vectors in V (and W) share a
  contiguous chunk of memory.
*/
PetscErrorCode EPSAllocateSolution(EPS eps)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (eps->allocated_ncv != eps->ncv) {
    ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest_left);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscInt),&eps->perm);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(eps->t,eps->ncv,&eps->V);CHKERRQ(ierr);
    if (eps->leftvecs) {
      ierr = VecDuplicateVecs(eps->t,eps->ncv,&eps->W);CHKERRQ(ierr);
    }
    eps->allocated_ncv = eps->ncv;
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
    ierr = VecDestroyVecs(eps->allocated_ncv,&eps->V);CHKERRQ(ierr);
    ierr = VecDestroyVecs(eps->allocated_ncv,&eps->W);CHKERRQ(ierr);
    eps->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}
