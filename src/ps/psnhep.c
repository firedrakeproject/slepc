/*
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

#include <slepc-private/psimpl.h>      /*I "slepcps.h" I*/

extern PetscErrorCode EPSDenseHessenberg(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*);
extern PetscErrorCode EPSDenseSchur(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);

#undef __FUNCT__  
#define __FUNCT__ "PSAllocate_NHEP"
PetscErrorCode PSAllocate_NHEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_NHEP"
PetscErrorCode PSSolve_NHEP(PS ps,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ps->state<PS_STATE_INTERMEDIATE) { /* reduce to upper Hessenberg form */
    ierr = EPSDenseHessenberg(ps->n,ps->l,ps->mat[PS_MAT_A],ps->ld,ps->mat[PS_MAT_Q]);CHKERRQ(ierr);
    //ierr = EPSDenseHessenberg(nv,eps->nconv,T,ncv,U);CHKERRQ(ierr);
  }
  if (ps->state<PS_STATE_CONDENSED) { /* compute the (real) Schur form */
    ierr = EPSDenseSchur(ps->n,ps->l,ps->mat[PS_MAT_A],ps->ld,ps->mat[PS_MAT_Q],eigr,eigi);CHKERRQ(ierr);
    //ierr = EPSDenseSchur(nv,eps->nconv,T,ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort_NHEP"
PetscErrorCode PSSort_NHEP(PS ps,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = EPSSortDenseSchur(eps,ps->n,ps->l,ps->mat[PS_MAT_A],ps->ld,ps->mat[PS_MAT_Q],eigr,eigi);CHKERRQ(ierr);
  //ierr = EPSSortDenseSchur(eps,nv,eps->nconv,T,ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_NHEP"
PetscErrorCode PSCreate_NHEP(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_NHEP;
  //ps->ops->computevector = PSComputeVector_NHEP;
  ps->ops->solve         = PSSolve_NHEP;
  ps->ops->sort          = PSSort_NHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

