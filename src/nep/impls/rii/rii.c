/*                       
   Residual inverse iteration (RII) method for nonlinear eigenproblems.

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

#include <slepc-private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <petscblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "NEPSetUp_RII"
PetscErrorCode NEPSetUp_RII(NEP nep)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (nep->ncv) { /* ncv set */
    if (nep->ncv<nep->nev) SETERRQ(((PetscObject)nep)->comm,1,"The value of ncv must be at least nev"); 
  } else if (nep->mpd) { /* mpd set */
    nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (nep->nev<500) nep->ncv = PetscMin(nep->n,PetscMax(2*nep->nev,nep->nev+15));
    else {
      nep->mpd = 500;
      nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
    }
  }
  if (!nep->mpd) nep->mpd = nep->ncv;
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(((PetscObject)nep)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!nep->max_it) nep->max_it = PetscMax(100,2*nep->n/nep->ncv);
  if (!nep->which) nep->which = NEP_TARGET_MAGNITUDE;

  ierr = NEPAllocateSolution(nep);CHKERRQ(ierr);
  ierr = NEPDefaultGetWork(nep,1);CHKERRQ(ierr);

  ierr = DSSetType(nep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "NEPSolve_RII"
PetscErrorCode NEPSolve_RII(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       lwork,ld,nv;
  Vec            v=nep->work[0];
  PetscScalar    *work;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(nep->ds,&ld);CHKERRQ(ierr);
  lwork = 7*nep->ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  if (nep->nini>0) {
    ierr = VecCopy(nep->V[0],v);CHKERRQ(ierr);
  } else {
    ierr = SlepcVecSetRandom(v,nep->rand);CHKERRQ(ierr);
  }
  
  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    nv = ld;
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->eigi,nep->errest,nv);CHKERRQ(ierr);
  } 

  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "NEPReset_RII"
PetscErrorCode NEPReset_RII(NEP nep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NEPDefaultFreeWork(nep);CHKERRQ(ierr);
  ierr = NEPFreeSolution(nep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NEPCreate_RII"
PetscErrorCode NEPCreate_RII(NEP nep)
{
  PetscFunctionBegin;
  nep->ops->solve        = NEPSolve_RII;
  nep->ops->setup        = NEPSetUp_RII;
  nep->ops->reset        = NEPReset_RII;
  PetscFunctionReturn(0);
}
EXTERN_C_END

