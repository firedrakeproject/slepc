/*
      EPS routines related to memory management.

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

#include <private/epsimpl.h>   /*I "slepceps.h" I*/

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
  PetscInt       i;
  PetscScalar    *pV,*pW;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (eps->allocated_ncv != eps->ncv) {
    if (eps->allocated_ncv > 0) {
      ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
      ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
      ierr = PetscFree(eps->errest);CHKERRQ(ierr); 
      ierr = PetscFree(eps->errest_left);CHKERRQ(ierr); 
      ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
      for (i=0;i<eps->allocated_ncv;i++) {
        ierr = VecDestroy(eps->V[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pV);CHKERRQ(ierr);
      ierr = PetscFree(eps->V);CHKERRQ(ierr);
      if (eps->W) {
        ierr = VecGetArray(eps->W[0],&pW);CHKERRQ(ierr);
        for (i=0;i<eps->allocated_ncv;i++) {
          ierr = VecDestroy(eps->W[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(pW);CHKERRQ(ierr);
        ierr = PetscFree(eps->W);CHKERRQ(ierr);
      }
    }
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest_left);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->V);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*eps->nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
    for (i=0;i<eps->ncv;i++) {
      ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,pV+i*eps->nloc,&eps->V[i]);CHKERRQ(ierr);
    }
    if (eps->leftvecs) {
      ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->W);CHKERRQ(ierr);
      ierr = PetscMalloc(eps->ncv*eps->nloc*sizeof(PetscScalar),&pW);CHKERRQ(ierr);
      for (i=0;i<eps->ncv;i++) {
        ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,pW+i*eps->nloc,&eps->W[i]);CHKERRQ(ierr);
      }
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
  PetscInt       i;
  PetscScalar    *pV,*pW;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (eps->allocated_ncv > 0) {
    ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
    ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest_left);CHKERRQ(ierr);
    ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
    for (i=0;i<eps->allocated_ncv;i++) {
      ierr = VecDestroy(eps->V[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(eps->V);CHKERRQ(ierr);
    if (eps->W) {
      ierr = VecGetArray(eps->W[0],&pW);CHKERRQ(ierr);
      for (i=0;i<eps->allocated_ncv;i++) {
        ierr = VecDestroy(eps->W[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pW);CHKERRQ(ierr);
      ierr = PetscFree(eps->W);CHKERRQ(ierr);
    }
    eps->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}
