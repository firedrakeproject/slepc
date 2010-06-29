/*
      Implements the ST class for preconditioned eigenvalue methods.

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

#include "private/stimpl.h"          /*I "slepcst.h" I*/

PetscErrorCode STDestroy_Precond(ST st);
PetscErrorCode STPrecondSetMatForPC_Precond(ST st,Mat mat);
PetscErrorCode STPrecondGetMatForPC_Precond(ST st,Mat *mat);

#undef __FUNCT__  
#define __FUNCT__ "SLEPcNotImplemented_Precond"
PetscErrorCode SLEPcNotImplemented_Precond() {
  SETERRQ(1, "STPrecond does not support some operation. Please, refer to the SLEPc Manual for more information.");
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Precond"
PetscErrorCode STSetUp_Precond(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->mat) {
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);
    CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
    ierr = MatDestroy(st->mat); CHKERRQ(ierr); st->mat = PETSC_NULL;
  } else {
    switch (st->shift_matrix) {
    case ST_MATMODE_INPLACE:
      if (st->sigma != 0.0) {
        if (st->B) {
          ierr = MatAXPY(st->A,-st->sigma,st->B,st->str);CHKERRQ(ierr); 
        } else { 
          ierr = MatShift(st->A,-st->sigma);CHKERRQ(ierr); 
        }
      }
      ierr = KSPSetOperators(st->ksp,st->A,st->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
      if (st->sigma != 0.0) {
        if (st->B) {
          ierr = MatAXPY(st->A,st->sigma,st->B,st->str);CHKERRQ(ierr); 
        } else { 
          ierr = MatShift(st->A,st->sigma);CHKERRQ(ierr); 
        }
      }
      break;
    case ST_MATMODE_SHELL:
      ierr = STMatShellCreate(st,&st->mat);CHKERRQ(ierr);
      //TODO: set the apply and apply transpose to st->mat
      ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
      ierr = MatDestroy(st->mat); CHKERRQ(ierr); st->mat = PETSC_NULL;
      break;
    default:
      if (st->sigma != 0.0) {
        ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&st->mat);CHKERRQ(ierr);
        if (st->B) { 
          ierr = MatAXPY(st->mat,-st->sigma,st->B,st->str);CHKERRQ(ierr); 
        } else { 
          ierr = MatShift(st->mat,-st->sigma);CHKERRQ(ierr); 
        }
        ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
      } else {
        ierr = KSPSetOperators(st->ksp,st->A,st->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetShift_Precond"
PetscErrorCode STSetShift_Precond(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Nothing to be done if STSetUp has not been called yet */
  if (!st->setupcalled) PetscFunctionReturn(0);
  
  st->sigma = newshift;
  if (st->shift_matrix != ST_MATMODE_SHELL) {
    ierr =  STSetUp_Precond(st); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Precond"
PetscErrorCode STCreate_Precond(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  st->data                 = 0;

  st->ops->apply           = SLEPcNotImplemented_Precond;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = SLEPcNotImplemented_Precond;
  st->ops->postsolve       = PETSC_NULL;
  st->ops->backtr          = PETSC_NULL;
  st->ops->setup           = STSetUp_Precond;
  st->ops->setshift        = STSetShift_Precond;
  st->ops->view            = STView_Default;
  st->ops->destroy         = STDestroy_Precond;
  
  st->checknullspace      = STCheckNullSpace_Default;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondGetMatForPC_C","STPrecondGetMatForPC_Precond",STPrecondGetMatForPC_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondSetMatForPC_C","STPrecondSetMatForPC_Precond",STPrecondSetMatForPC_Precond);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STDestroy_Precond"
PetscErrorCode STDestroy_Precond(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondGetMatForPC_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondSetMatForPC_C","",PETSC_NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STPrecondGetMatForPC"
/*@
   STPrecondGetMatForPC - Gets the matrix previously set by STPrecondSetMatForPC.
   This matrix will be passed as parameter in the KSPSetOperator function as
   the matrix to be used in constructing the preconditioner.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  mat - the matrix that will be used in constructing the preconditioner or
   PETSC_NULL if any previous matrix was set by STPrecondSetMatForPC.

   Level: advanced

.seealso: STPrecondSetMatForPC(), KSPSetOperator()
@*/
PetscErrorCode STPrecondGetMatForPC(ST st,Mat *mat)
{
  PetscErrorCode ierr, (*f)(ST,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STPrecondGetMatForPC_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

PetscErrorCode STPrecondGetMatForPC_Precond(ST st,Mat *mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  *mat = st->mat;

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STPrecondSetMatForPC"
/*@
   STPrecondSetMatForPC - Sets the matrix that will be passed as parameter in
   the KSPSetOperator function as the matrix to be used in constructing the
   preconditioner. If any matrix is set or mat is PETSC_NULL, A - sigma*B will
   be used, being sigma the value set by STSetShift

   Collective on ST

   Input Parameter:
+  st - the spectral transformation context
-  mat - the matrix that will be used in constructing the preconditioner

   Level: advanced

.seealso: STPrecondSetMatForPC(), KSPSetOperator()
@*/
PetscErrorCode STPrecondSetMatForPC(ST st,Mat mat)
{
  PetscErrorCode ierr, (*f)(ST,Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(mat,MAT_COOKIE,2);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STPrecondGetMatForPC_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

PetscErrorCode STPrecondSetMatForPC_Precond(ST st,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(mat,MAT_COOKIE,2);

  if (st->mat) { ierr = MatDestroy(st->mat); CHKERRQ(ierr); }
  st->mat = mat;

  PetscFunctionReturn(0);
}

