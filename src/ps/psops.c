/*
   PS operations: PSSolve(), PSSort(), etc

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

#undef __FUNCT__  
#define __FUNCT__ "PSGetLeadingDimension"
/*@
   PSGetLeadingDimension - Returns the leading dimension of the allocated
   matrices.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  ld - leading dimension (maximum allowed dimension for the matrices)

   Level: advanced

.seealso: PSAllocate(), PSSetDimensions()
@*/
PetscErrorCode PSGetLeadingDimension(PS ps,PetscInt *ld)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (ld) *ld = ps->ld;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetState"
/*@
   PSSetState - Change the state of the PS object.

   Collective on PS

   Input Parameters:
+  ps    - the projected system context
-  state - the new state

   Notes:
   The state indicates that the projected system is in an initial state (raw),
   in an intermediate state (such as tridiagonal, Hessenberg or 
   Hessenberg-triangular), in a condensed state (such as diagonal, Schur or
   generalized Schur), or in a sorted condensed state (according to a given
   sorting criterion).

   This function is normally used to return to the raw state when the
   condensed structure is destroyed.

   Level: advanced

.seealso: PSGetState()
@*/
PetscErrorCode PSSetState(PS ps,PSStateType state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ps,state,2);
  switch (state) {
    case PS_STATE_RAW:
    case PS_STATE_INTERMEDIATE:
    case PS_STATE_CONDENSED:
    case PS_STATE_SORTED:
      if (ps->state<state) { ierr = PetscInfo(ps,"PS state has been increased\n");CHKERRQ(ierr); }
      ps->state = state;
      break;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"Wrong state");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetState"
/*@
   PSGetState - Returns the current state.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  state - current state

   Level: advanced

.seealso: PSSetState()
@*/
PetscErrorCode PSGetState(PS ps,PSStateType *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (state) *state = ps->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetDimensions"
/*@
   PSSetDimensions - Resize the matrices in the PS object.

   Collective on PS

   Input Parameters:
+  ps - the projected system context
.  n  - the new size
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Note:
   The internal arrays are not reallocated.

   Level: advanced

.seealso: PSGetDimensions(), PSAllocate()
@*/
PetscErrorCode PSSetDimensions(PS ps,PetscInt n,PetscInt l,PetscInt k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ps,n,2);
  PetscValidLogicalCollectiveInt(ps,l,3);
  PetscValidLogicalCollectiveInt(ps,k,4);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (n!=PETSC_IGNORE) {
    if (n==PETSC_DECIDE || n==PETSC_DEFAULT) {
      ps->n = ps->ld;
    } else {
      if (n<1 || n>ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between 1 and ld");
      ps->n = n;
    }
  }
  if (l!=PETSC_IGNORE) {
    if (l==PETSC_DECIDE || l==PETSC_DEFAULT) {
      ps->l = 0;
    } else {
      if (l<0 || l>ps->n) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of l. Must be between 0 and n");
      ps->l = l;
    }
  }
  if (k!=PETSC_IGNORE) {
    if (k==PETSC_DECIDE || k==PETSC_DEFAULT) {
      ps->k = ps->n/2;
    } else {
      if (k<0 || k>ps->n) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of k. Must be between 0 and n");
      ps->k = k;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetDimensions"
/*@
   PSGetDimensions - Returns the current dimensions.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  state - current dimensions

   Level: advanced

.seealso: PSSetDimensions()
@*/
PetscErrorCode PSGetDimensions(PS ps,PetscInt *n,PetscInt *l,PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (n) *n = ps->n;
  if (l) *l = ps->l;
  if (k) *k = ps->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetArray"
/*@C
   PSGetArray - Returns a pointer to one of the internal arrays used to
   represent matrices. You MUST call PSRestoreArray() when you no longer
   need to access the array.

   Not Collective

   Input Parameters:
+  ps - the projected system context
-  m - the requested matrix

   Output Parameter:
.  a - pointer to the values

   Level: advanced

.seealso: PSRestoreArray(), PSGetArrayReal()
@*/
PetscErrorCode PSGetArray(PS ps,PSMatType m,PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=PS_NUM_MAT) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (!ps->mat[m]) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONGSTATE,"Requested matrix was not created in this PS");
  *a = ps->mat[m];
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSRestoreArray"
/*@C
   PSRestoreArray - Restores the matrix after PSGetArray() was called.

   Not Collective

   Input Parameters:
+  ps - the projected system context
.  m - the requested matrix
-  a - pointer to the values

   Level: advanced

.seealso: PSGetArray(), PSGetArrayReal()
@*/
PetscErrorCode PSRestoreArray(PS ps,PSMatType m,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=PS_NUM_MAT) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  CHKMEMQ;
  *a = 0;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve"
/*@
   PSSolve - Solves the problem.

   Not Collective

   Input Parameters:
+  ps   - the projected system context
.  eigr - array to store the computed eigenvalues (real part)
-  eigi - array to store the computed eigenvalues (imaginary part)

   Note:
   This call brings the projected system to condensed form. No ordering
   is enforced, call PSSort() later if you want the solution sorted.

   Level: advanced

.seealso: PSSort()
@*/
PetscErrorCode PSSolve(PS ps,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (!ps->ops->solve) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Solve,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->solve)(ps,eigr,eigi);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Solve,ps,0,0,0);CHKERRQ(ierr);
  ps->state = PS_STATE_CONDENSED;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSCond"
/*@C
   PSCond - Compute the inf-norm condition number of the first matrix
   as cond(A) = norm(A)*norm(inv(A)).

   Not Collective

   Input Parameters:
+  ps - the projected system context
-  cond - the computed condition number

   Level: advanced

.seealso: PSSolve()
@*/
PetscErrorCode PSCond(PS ps,PetscReal *cond)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!ps->ops->cond) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->cond)(ps,cond);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort"
/*@C
   PSSort - Reorders the condensed form computed by PSSolve() according to
   a given sorting criterion.

   Not Collective

   Input Parameters:
+  ps - the projected system context
.  eigr - array to store the sorted eigenvalues (real part)
-  eigi - array to store the sorted eigenvalues (imaginary part)

   Level: advanced

.seealso: PSSolve()
@*/
PetscErrorCode PSSort(PS ps,PetscScalar *eigr,PetscScalar *eigi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (ps->state!=PS_STATE_CONDENSED) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSSolve() first");
  if (!ps->ops->sort) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Sort,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->sort)(ps,eigr,eigi,comp_func,comp_ctx);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Sort,ps,0,0,0);CHKERRQ(ierr);
  ps->state = PS_STATE_SORTED;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSTranslateHarmonic"
/*@C
   PSTranslateHarmonic - Computes a translation of the projected matrix.

   Not Collective

   Input Parameters:
+  ps   - the projected system context
.  tau  - the translation amount
-  beta - last component of vector b

   Notes:
   This function is intended for use in the context of Krylov methods only.
   It computes a translation of a Krylov decomposition in order to extract
   eigenpair approximations by harmonic Rayleigh-Ritz.
   The matrix is updated as A + g*b' where g = (B-tau*eye(n))'\b and
   vector b is assumed to be beta*e_n^T.

   Level: developer

.seealso: PSRecoverHarmonic()
@*/
PetscErrorCode PSTranslateHarmonic(PS ps,PetscScalar tau,PetscScalar beta,PetscScalar *g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!ps->ops->translate) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->translate)(ps,tau,beta,g);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ps->state = PS_STATE_RAW;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

