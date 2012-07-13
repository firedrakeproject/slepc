/*
   PS operations: PSSolve(), PSVectors(), etc

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

   Logically Collective on PS

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

   Logically Collective on PS

   Input Parameters:
+  ps - the projected system context
.  n  - the new size
.  m  - the new column size (only for SVD)
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Notes:
   The internal arrays are not reallocated.

   The value m is not used except in the case of PSSVD, pass PETSC_IGNORE
   otherwise. PETSC_IGNORE can also be used in any of the other parameters
   to leave the value unchanged.

   Level: advanced

.seealso: PSGetDimensions(), PSAllocate()
@*/
PetscErrorCode PSSetDimensions(PS ps,PetscInt n,PetscInt m,PetscInt l,PetscInt k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ps,n,2);
  PetscValidLogicalCollectiveInt(ps,m,3);
  PetscValidLogicalCollectiveInt(ps,l,4);
  PetscValidLogicalCollectiveInt(ps,k,5);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (n!=PETSC_IGNORE) {
    if (n==PETSC_DECIDE || n==PETSC_DEFAULT) {
      ps->n = ps->ld;
    } else {
      if (n<1 || n>ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between 1 and ld");
      if (ps->extrarow && n+1>ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"A value of n equal to ld leaves no room for extra row");
      ps->n = n;
    }
  }
  if (m!=PETSC_IGNORE) {
    if (m==PETSC_DECIDE || m==PETSC_DEFAULT) {
      ps->m = ps->ld;
    } else {
      if (m<1 || m>ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of m. Must be between 1 and ld");
      ps->m = m;
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
+  n  - the current size
.  m  - the current column size (only for SVD)
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Level: advanced

.seealso: PSSetDimensions()
@*/
PetscErrorCode PSGetDimensions(PS ps,PetscInt *n,PetscInt *m,PetscInt *l,PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (n) *n = ps->n;
  if (m) *m = ps->m;
  if (l) *l = ps->l;
  if (k) *k = ps->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSTruncate"
/*@
   PSTruncate - Truncates the system represented in the PS object.

   Logically Collective on PS

   Input Parameters:
+  ps - the projected system context
-  n  - the new size

   Note:
   The new size is set to n. In cases where the extra row is meaningful,
   the first n elements are kept as the extra row for the new system.

   Level: developer

.seealso: PSSetDimensions(), PSSetExtraRow()
@*/
PetscErrorCode PSTruncate(PS ps,PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ps,n,2);
  if (!ps->ops->truncate) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (n<ps->l || n>ps->n) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between l and n");
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->truncate)(ps,n);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ps->state = PS_STATE_RAW;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
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
#define __FUNCT__ "PSGetArrayReal"
/*@C
   PSGetArrayReal - Returns a pointer to one of the internal arrays used to
   represent real matrices. You MUST call PSRestoreArray() when you no longer
   need to access the array.

   Not Collective

   Input Parameters:
+  ps - the projected system context
-  m - the requested matrix

   Output Parameter:
.  a - pointer to the values

   Level: advanced

.seealso: PSRestoreArrayReal(), PSGetArray()
@*/
PetscErrorCode PSGetArrayReal(PS ps,PSMatType m,PetscReal *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=PS_NUM_MAT) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (!ps->rmat[m]) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONGSTATE,"Requested matrix was not created in this PS");
  *a = ps->rmat[m];
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSRestoreArrayReal"
/*@C
   PSRestoreArrayReal - Restores the matrix after PSGetArrayReal() was called.

   Not Collective

   Input Parameters:
+  ps - the projected system context
.  m - the requested matrix
-  a - pointer to the values

   Level: advanced

.seealso: PSGetArrayReal(), PSGetArray()
@*/
PetscErrorCode PSRestoreArrayReal(PS ps,PSMatType m,PetscReal *a[])
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

   Logically Collective on PS

   Input Parameters:
+  ps   - the projected system context
.  eigr - array to store the computed eigenvalues (real part)
-  eigi - array to store the computed eigenvalues (imaginary part)

   Note:
   This call brings the projected system to condensed form. No ordering
   of the eigenvalues is enforced unless a comparison function has been
   provided with PSSetEigenvalueComparison().

   Level: advanced

.seealso: PSSetEigenvalueComparison()
@*/
PetscErrorCode PSSolve(PS ps,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidPointer(eigr,2);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (!ps->ops->solve) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  if (ps->state>=PS_STATE_CONDENSED) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PS_Solve,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  if (!ps->ops->solve[ps->method]) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The specified method number does not exist for this PS");
  ierr = (*ps->ops->solve[ps->method])(ps,eigr,eigi);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Solve,ps,0,0,0);CHKERRQ(ierr);
  if (ps->comp_fun) ps->state = PS_STATE_SORTED;
  else ps->state = PS_STATE_CONDENSED;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors"
/*@
   PSVectors - Compute vectors associated to the projected system such
   as eigenvectors.

   Logically Collective on PS

   Input Parameters:
+  ps  - the projected system context
-  mat - the matrix, used to indicate which vectors are required

   Input/Output Parameter:
-  j   - (optional) index of vector to be computed

   Output Parameter:
.  rnorm - (optional) computed residual norm

   Notes:
   Allowed values for mat are PS_MAT_X, PS_MAT_Y, PS_MAT_U and PS_MAT_VT, to
   compute right or left eigenvectors, or left or right singular vectors,
   respectively.

   If PETSC_NULL is passed in argument j then all vectors are computed,
   otherwise j indicates which vector must be computed. In real non-symmetric
   problems, on exit the index j will be incremented when a complex conjugate
   pair is found.

   This function can be invoked after the projected problem has been solved,
   to get the residual norm estimate of the associated Ritz pair. In that
   case, the relevant information is returned in rnorm.

   For computing eigenvectors, LAPACK's _trevc is used so the matrix must
   be in (quasi-)triangular form, or call PSSolve() first.

   Level: advanced

.seealso: PSSolve()
@*/
PetscErrorCode PSVectors(PS ps,PSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ps,mat,2);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (!ps->ops->vectors) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  if (rnorm && !j) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must give a value of j");
  ierr = PSAllocateMat_Private(ps,mat);CHKERRQ(ierr); 
  ierr = PetscLogEventBegin(PS_Vectors,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->vectors)(ps,mat,j,rnorm);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Vectors,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSNormalize"
/*@
   PSNormalize - Normalize a column or all the columns of a matrix. Considers
   the case when the columns represent the real and the imaginary part of a vector.          

   Logically Collective on PS

   Input Parameter:
+  ps  - the projected system context
.  mat - the matrix to be modified
-  col - the column to normalize or -1 to normalize all of them

   Notes:
   The columns are normalized with respect to the 2-norm.

   If col and col+1 (or col-1 and col) represent the real and the imaginary
   part of a vector, both columns are scaled.

   Level: advanced
@*/
PetscErrorCode PSNormalize(PS ps,PSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ps,mat,2);
  PetscValidLogicalCollectiveInt(ps,col,3);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (!ps->ops->normalize) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  if (col<-1) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"col should be at least minus one");
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->normalize)(ps,mat,col);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
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
  PetscValidPointer(cond,2);
  if (!ps->ops->cond) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->cond)(ps,cond);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSTranslateHarmonic"
/*@C
   PSTranslateHarmonic - Computes a translation of the projected matrix.

   Logically Collective on PS

   Input Parameters:
+  ps      - the projected system context
.  tau     - the translation amount
.  beta    - last component of vector b
-  recover - boolean flag to indicate whether to recover or not

   Output Parameters:
+  g       - the computed vector (optional)
-  gamma   - scale factor (optional)

   Notes:
   This function is intended for use in the context of Krylov methods only.
   It computes a translation of a Krylov decomposition in order to extract
   eigenpair approximations by harmonic Rayleigh-Ritz.
   The matrix is updated as A + g*b' where g = (A-tau*eye(n))'\b and
   vector b is assumed to be beta*e_n^T.

   The gamma factor is defined as sqrt(1+g'*g) and can be interpreted as
   the factor by which the residual of the Krylov decomposition is scaled.

   If the recover flag is activated, the computed translation undoes the
   translation done previously. In that case, parameter tau is ignored.

   Level: developer
@*/
PetscErrorCode PSTranslateHarmonic(PS ps,PetscScalar tau,PetscReal beta,PetscBool recover,PetscScalar *g,PetscReal *gamma)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!ps->ops->transharm) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->transharm)(ps,tau,beta,recover,g,gamma);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ps->state = PS_STATE_RAW;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSTranslateRKS"
/*@C
   PSTranslateRKS - Computes a modification of the projected matrix corresponding
   to an update of the shift in a rational Krylov method.

   Logically Collective on PS

   Input Parameters:
+  ps    - the projected system context
-  alpha - the translation amount

   Notes:
   This function is intended for use in the context of Krylov methods only.
   It takes the leading (k+1,k) submatrix of A, containing the truncated
   Rayleigh quotient of a Krylov-Schur relation computed from a shift
   sigma1 and transforms it to obtain a Krylov relation as if computed 
   from a different shift sigma2. The new matrix is computed as
   1.0/alpha*(eye(k)-Q*inv(R)), where [Q,R]=qr(eye(k)-alpha*A) and
   alpha = sigma1-sigma2.

   Matrix Q is placed in PS_MAT_Q so that it can be used to update the
   Krylov basis.

   Level: developer
@*/
PetscErrorCode PSTranslateRKS(PS ps,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!ps->ops->transrks) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP,"PS type %s",((PetscObject)ps)->type_name);
  ierr = PetscLogEventBegin(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ps->ops->transrks)(ps,alpha);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PS_Other,ps,0,0,0);CHKERRQ(ierr);
  ps->state   = PS_STATE_RAW;
  ps->compact = PETSC_FALSE;
  ierr = PetscObjectStateIncrease((PetscObject)ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

