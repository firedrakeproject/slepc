/*
      Implements the ST class for preconditioned eigenvalue methods.

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

#include "private/stimpl.h"          /*I "slepcst.h" I*/

PetscErrorCode STDestroy_Precond(ST st);
PetscErrorCode STSetFromOptions_Precond(ST st); 
EXTERN_C_BEGIN
PetscErrorCode STPrecondSetMatForPC_Precond(ST st,Mat mat);
PetscErrorCode STPrecondGetMatForPC_Precond(ST st,Mat *mat);
PetscErrorCode STPrecondSetKSPHasMat_Precond(ST st,PetscTruth setmat);
PetscErrorCode STPrecondGetKSPHasMat_Precond(ST st,PetscTruth *setmat);
EXTERN_C_END

typedef struct {
  PetscTruth     setmat;
} ST_PRECOND;


#undef __FUNCT__  
#define __FUNCT__ "SLEPcNotImplemented_Precond"
PetscErrorCode SLEPcNotImplemented_Precond(ST st, Vec x, Vec y) {
  SETERRQ(1, "STPrecond does not support some operation. Please, refer to the SLEPc Manual for more information.");
}

#undef __FUNCT__  
#define __FUNCT__ "STSetFromOptions_Precond"
PetscErrorCode STSetFromOptions_Precond(ST st) 
{
  PetscErrorCode ierr;
  PC             pc;
  const PCType   pctype;
  Mat            P;
  PetscTruth     t0, t1;

  PetscFunctionBegin;

  ierr = KSPGetPC(st->ksp, &pc); CHKERRQ(ierr);
  ierr = PetscObjectGetType((PetscObject)pc, &pctype); CHKERRQ(ierr);
  ierr = STPrecondGetMatForPC(st, &P); CHKERRQ(ierr);
  if (!pctype && st->A) {
    if (P || st->shift_matrix == ST_MATMODE_SHELL) {
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    } else {
      ierr = MatHasOperation(st->A, MATOP_DUPLICATE, &t0); CHKERRQ(ierr);
      if (st->B) {
        ierr = MatHasOperation(st->A, MATOP_AXPY, &t1); CHKERRQ(ierr);
      } else {
        t1 = PETSC_TRUE;
      }
      ierr = PCSetType(pc, (t0 == PETSC_TRUE && t1 == PETSC_TRUE)?
                             PCJACOBI:PCNONE); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Precond"
PetscErrorCode STSetUp_Precond(ST st)
{
  Mat            P;
  PC             pc;
  PetscTruth     t0, setmat, destroyP, builtP;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;

  /* If pc is none and any matrix has to be set, exit */
  ierr = STSetFromOptions_Precond(st); CHKERRQ(ierr);
  ierr = KSPGetPC(st->ksp, &pc); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc, PCNONE, &t0); CHKERRQ(ierr);
  ierr = STPrecondGetKSPHasMat(st, &setmat); CHKERRQ(ierr); 
  if (t0 == PETSC_TRUE && setmat == PETSC_FALSE) PetscFunctionReturn(0);

  /* Check if a user matrix is set */
  ierr = STPrecondGetMatForPC(st, &P); CHKERRQ(ierr);

  /* If not, create A - shift*B */
  if (P) {
    builtP = PETSC_FALSE;
    destroyP = PETSC_TRUE;
  } else {
    builtP = PETSC_TRUE;

    if (st->shift_matrix == ST_MATMODE_SHELL) {
      ierr = STMatShellCreate(st,&P);CHKERRQ(ierr);
      //TODO: set the apply and apply transpose to st->mat
      destroyP = PETSC_TRUE;
    } else if (!(PetscAbsScalar(st->sigma) < PETSC_MAX) && st->B) {
      P = st->B;
      destroyP = PETSC_FALSE;
    } else if (st->sigma == 0.0) {
      P = st->A;
      destroyP = PETSC_FALSE;
    } else if (PetscAbsScalar(st->sigma) < PETSC_MAX) {
      if (st->shift_matrix == ST_MATMODE_INPLACE) {
        P = st->A;
        destroyP = PETSC_FALSE;
      } else {
        ierr = MatDuplicate(st->A, MAT_COPY_VALUES, &P); CHKERRQ(ierr);
        destroyP = PETSC_TRUE;
      } 
      if (st->B) {
        ierr = MatAXPY(P, -st->sigma, st->B, st->str); CHKERRQ(ierr); 
      } else {
        ierr = MatShift(P, -st->sigma); CHKERRQ(ierr); 
      }
    } else 
      builtP = PETSC_FALSE;
  }

  /* If P was not possible to obtain, set pc to PCNONE */
  if (!P) {
    ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);

    /* If some matrix has to be set to ksp, set ksp to KSPPREONLY */
    if (setmat == PETSC_TRUE) {
      ierr = STMatShellCreate(st, &P);CHKERRQ(ierr);
      destroyP = PETSC_TRUE;
      ierr = KSPSetType(st->ksp, KSPPREONLY); CHKERRQ(ierr);
    }
  }

  ierr = KSPSetOperators(st->ksp, setmat==PETSC_TRUE?P:PETSC_NULL, P,
                         DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

  if (destroyP == PETSC_TRUE) {
    ierr = MatDestroy(P); CHKERRQ(ierr);
  } else if (st->shift_matrix == ST_MATMODE_INPLACE && builtP == PETSC_TRUE) {
    if (st->sigma != 0.0 && PetscAbsScalar(st->sigma) < PETSC_MAX) {
      if (st->B) {
        ierr = MatAXPY(st->A,st->sigma,st->B,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(st->A,st->sigma);CHKERRQ(ierr); 
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
  ST_PRECOND     *data;

  PetscFunctionBegin;

  ierr = PetscNew(ST_PRECOND, &data); CHKERRQ(ierr);
  st->data                 = data;

  st->ops->apply           = SLEPcNotImplemented_Precond;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = SLEPcNotImplemented_Precond;
  st->ops->postsolve       = PETSC_NULL;
  st->ops->backtr          = PETSC_NULL;
  st->ops->setup           = STSetUp_Precond;
  st->ops->setshift        = STSetShift_Precond;
  st->ops->view            = STView_Default;
  st->ops->destroy         = STDestroy_Precond;
  st->ops->setfromoptions  = STSetFromOptions_Precond;
  
  st->checknullspace       = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondGetMatForPC_C","STPrecondGetMatForPC_Precond",STPrecondGetMatForPC_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondSetMatForPC_C","STPrecondSetMatForPC_Precond",STPrecondSetMatForPC_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondGetKSPHasMat_C","STPrecondGetKSPHasMat_Precond",STPrecondGetKSPHasMat_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondSetKSPHasMat_C","STPrecondSetKSPHasMat_Precond",STPrecondSetKSPHasMat_Precond);CHKERRQ(ierr);

  ierr = STPrecondSetKSPHasMat_Precond(st, PETSC_TRUE); CHKERRQ(ierr);
  ierr = KSPSetType(st->ksp, KSPPREONLY); CHKERRQ(ierr);

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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondGetKSPHasMat_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STPrecondSetKSPHasMat_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscFree(st->data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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

EXTERN_C_BEGIN
PetscErrorCode STPrecondGetMatForPC_Precond(ST st,Mat *mat)
{
  PetscErrorCode ierr;
  PC             pc;
  PetscTruth     flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  ierr = KSPGetPC(st->ksp, &pc); CHKERRQ(ierr);
  ierr = PCGetOperatorsSet(pc, PETSC_NULL, &flag); CHKERRQ(ierr);
  if (flag == PETSC_TRUE) {
    ierr = PCGetOperators(pc, PETSC_NULL, mat, PETSC_NULL); CHKERRQ(ierr);
  } else
    *mat = PETSC_NULL;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STPrecondSetMatForPC"
/*@
   STPrecondSetMatForPC - Sets the matrix that will be passed as parameter in
   the KSPSetOperators function as the matrix to be used in constructing the
   preconditioner. If any matrix is set or mat is PETSC_NULL, A - sigma*B will
   be used, being sigma the value set by STSetShift

   Collective on ST

   Input Parameter:
+  st - the spectral transformation context
-  mat - the matrix that will be used in constructing the preconditioner

   Level: advanced

.seealso: STPrecondSetMatForPC(), KSPSetOperators()
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

EXTERN_C_BEGIN
PetscErrorCode STPrecondSetMatForPC_Precond(ST st,Mat mat)
{
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(mat,MAT_COOKIE,2);

  ierr = KSPGetPC(st->ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetOperators(pc, PETSC_NULL, mat, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "STPrecondSetKSPHasMat"
/*@
   STPrecondSetKSPHasMat - Sets if during the STSetUp the KSP matrix associated
   to the linear system is set with the matrix for building the preconditioner.

   Collective on ST

   Input Parameter:
+  st - the spectral transformation context
-  setmat - if true, the KSP matrix associated to linear system is set with
   the matrix for building the preconditioner

   Level: developer

.seealso: STPrecondGetKSPHasMat(), TSetShift(), KSPSetOperators()
@*/
PetscErrorCode STPrecondSetKSPHasMat(ST st,PetscTruth setmat)
{
  PetscErrorCode ierr, (*f)(ST,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STPrecondSetKSPHasMat_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,setmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPrecondGetKSPHasMat"
/*@
   STPrecondGetKSPHasMat - Gets if during the STSetUp the KSP matrix associated
   to the linear system is set with the matrix for building the preconditioner.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  setmat - if true, the KSP matrix associated to linear system is set with
   the matrix for building the preconditioner

   Level: developer

.seealso: STPrecondSetKSPHasMat(), STSetShift(), KSPSetOperators()
@*/
PetscErrorCode STPrecondGetKSPHasMat(ST st,PetscTruth *setmat)
{
  PetscErrorCode ierr, (*f)(ST,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STPrecondGetKSPHasMat_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,setmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode STPrecondSetKSPHasMat_Precond(ST st,PetscTruth setmat)
{
  ST_PRECOND     *data = (ST_PRECOND*)st->data;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  data->setmat = setmat;

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
PetscErrorCode STPrecondGetKSPHasMat_Precond(ST st,PetscTruth *setmat)
{
  ST_PRECOND     *data = (ST_PRECOND*)st->data;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  *setmat = data->setmat;

  PetscFunctionReturn(0);
}
EXTERN_C_END

