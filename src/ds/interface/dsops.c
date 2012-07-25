/*
   DS operations: DSSolve(), DSVectors(), etc

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

#include <slepc-private/dsimpl.h>      /*I "slepcds.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "DSGetLeadingDimension"
/*@
   DSGetLeadingDimension - Returns the leading dimension of the allocated
   matrices.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  ld - leading dimension (maximum allowed dimension for the matrices)

   Level: advanced

.seealso: DSAllocate(), DSSetDimensions()
@*/
PetscErrorCode DSGetLeadingDimension(DS ds,PetscInt *ld)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (ld) *ld = ds->ld;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetState"
/*@
   DSSetState - Change the state of the DS object.

   Logically Collective on DS

   Input Parameters:
+  ds    - the direct solver context
-  state - the new state

   Notes:
   The state indicates that the dense system is in an initial state (raw),
   in an intermediate state (such as tridiagonal, Hessenberg or 
   Hessenberg-triangular), in a condensed state (such as diagonal, Schur or
   generalized Schur), or in a sorted condensed state (according to a given
   sorting criterion).

   This function is normally used to return to the raw state when the
   condensed structure is destroyed.

   Level: advanced

.seealso: DSGetState()
@*/
PetscErrorCode DSSetState(DS ds,DSStateType state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,state,2);
  switch (state) {
    case DS_STATE_RAW:
    case DS_STATE_INTERMEDIATE:
    case DS_STATE_CONDENSED:
    case DS_STATE_SORTED:
      if (ds->state<state) { ierr = PetscInfo(ds,"DS state has been increased\n");CHKERRQ(ierr); }
      ds->state = state;
      break;
    default:
      SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Wrong state");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetState"
/*@
   DSGetState - Returns the current state.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  state - current state

   Level: advanced

.seealso: DSSetState()
@*/
PetscErrorCode DSGetState(DS ds,DSStateType *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (state) *state = ds->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetDimensions"
/*@
   DSSetDimensions - Resize the matrices in the DS object.

   Logically Collective on DS

   Input Parameters:
+  ds - the direct solver context
.  n  - the new size
.  m  - the new column size (only for SVD)
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Notes:
   The internal arrays are not reallocated.

   The value m is not used except in the case of DSSVD, pass PETSC_IGNORE
   otherwise.

   Level: advanced

.seealso: DSGetDimensions(), DSAllocate()
@*/
PetscErrorCode DSSetDimensions(DS ds,PetscInt n,PetscInt m,PetscInt l,PetscInt k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,n,2);
  PetscValidLogicalCollectiveInt(ds,m,3);
  PetscValidLogicalCollectiveInt(ds,l,4);
  PetscValidLogicalCollectiveInt(ds,k,5);
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (n!=PETSC_IGNORE) {
    if (n==PETSC_DECIDE || n==PETSC_DEFAULT) {
      ds->n = ds->ld;
    } else {
      if (n<1 || n>ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between 1 and ld");
      if (ds->extrarow && n+1>ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"A value of n equal to ld leaves no room for extra row");
      ds->n = n;
    }
  }
  if (m!=PETSC_IGNORE) {
    if (m==PETSC_DECIDE || m==PETSC_DEFAULT) {
      ds->m = ds->ld;
    } else {
      if (m<1 || m>ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of m. Must be between 1 and ld");
      ds->m = m;
    }
  }
  if (l==PETSC_DECIDE || l==PETSC_DEFAULT) {
    ds->l = 0;
  } else {
    if (l<0 || l>ds->n) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of l. Must be between 0 and n");
    ds->l = l;
  }
  if (k==PETSC_DECIDE || k==PETSC_DEFAULT) {
    ds->k = ds->n/2;
  } else {
    if (k<0 || k>ds->n) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of k. Must be between 0 and n");
    ds->k = k;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetDimensions"
/*@
   DSGetDimensions - Returns the current dimensions.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
+  n  - the current size
.  m  - the current column size (only for SVD)
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Level: advanced

.seealso: DSSetDimensions()
@*/
PetscErrorCode DSGetDimensions(DS ds,PetscInt *n,PetscInt *m,PetscInt *l,PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (n) *n = ds->n;
  if (m) *m = ds->m;
  if (l) *l = ds->l;
  if (k) *k = ds->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSTruncate"
/*@
   DSTruncate - Truncates the system represented in the DS object.

   Logically Collective on DS

   Input Parameters:
+  ds - the direct solver context
-  n  - the new size

   Note:
   The new size is set to n. In cases where the extra row is meaningful,
   the first n elements are kept as the extra row for the new system.

   Level: developer

.seealso: DSSetDimensions(), DSSetExtraRow()
@*/
PetscErrorCode DSTruncate(DS ds,PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,n,2);
  if (!ds->ops->truncate) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (n<ds->l || n>ds->n) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between l and n");
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->truncate)(ds,n);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ds->state = DS_STATE_RAW;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetArray"
/*@C
   DSGetArray - Returns a pointer to one of the internal arrays used to
   represent matrices. You MUST call DSRestoreArray() when you no longer
   need to access the array.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  m - the requested matrix

   Output Parameter:
.  a - pointer to the values

   Level: advanced

.seealso: DSRestoreArray(), DSGetArrayReal()
@*/
PetscErrorCode DSGetArray(DS ds,DSMatType m,PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=DS_NUM_MAT) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (!ds->mat[m]) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONGSTATE,"Requested matrix was not created in this DS");
  *a = ds->mat[m];
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSRestoreArray"
/*@C
   DSRestoreArray - Restores the matrix after DSGetArray() was called.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
.  m - the requested matrix
-  a - pointer to the values

   Level: advanced

.seealso: DSGetArray(), DSGetArrayReal()
@*/
PetscErrorCode DSRestoreArray(DS ds,DSMatType m,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=DS_NUM_MAT) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  CHKMEMQ;
  *a = 0;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetArrayReal"
/*@C
   DSGetArrayReal - Returns a pointer to one of the internal arrays used to
   represent real matrices. You MUST call DSRestoreArray() when you no longer
   need to access the array.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  m - the requested matrix

   Output Parameter:
.  a - pointer to the values

   Level: advanced

.seealso: DSRestoreArrayReal(), DSGetArray()
@*/
PetscErrorCode DSGetArrayReal(DS ds,DSMatType m,PetscReal *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=DS_NUM_MAT) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (!ds->rmat[m]) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONGSTATE,"Requested matrix was not created in this DS");
  *a = ds->rmat[m];
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSRestoreArrayReal"
/*@C
   DSRestoreArrayReal - Restores the matrix after DSGetArrayReal() was called.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
.  m - the requested matrix
-  a - pointer to the values

   Level: advanced

.seealso: DSGetArrayReal(), DSGetArray()
@*/
PetscErrorCode DSRestoreArrayReal(DS ds,DSMatType m,PetscReal *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(a,2);
  if (m<0 || m>=DS_NUM_MAT) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Invalid matrix");
  CHKMEMQ;
  *a = 0;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSolve"
/*@
   DSSolve - Solves the problem.

   Logically Collective on DS

   Input Parameters:
+  ds   - the direct solver context
.  eigr - array to store the computed eigenvalues (real part)
-  eigi - array to store the computed eigenvalues (imaginary part)

   Note:
   This call brings the dense system to condensed form. No ordering
   of the eigenvalues is enforced (for this, call DSSort() afterwards).

   Level: advanced

.seealso: DSSort()
@*/
PetscErrorCode DSSolve(DS ds,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(eigr,2);
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (!ds->ops->solve) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (ds->state>=DS_STATE_CONDENSED) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(DS_Solve,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  if (!ds->ops->solve[ds->method]) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The specified method number does not exist for this DS");
  ierr = (*ds->ops->solve[ds->method])(ds,eigr,eigi);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Solve,ds,0,0,0);CHKERRQ(ierr);
  ds->state = DS_STATE_CONDENSED;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSort"
/*@
   DSSort - Sorts the result of DSSolve() according to a given sorting
   criterion.

   Logically Collective on DS

   Input Parameters:
+  ds   - the direct solver context
.  eigr - array containing the computed eigenvalues (real part)
.  eigi - array containing the computed eigenvalues (imaginary part)
.  rr   - (optional) array containing auxiliary values (real part)
-  ri   - (optional) array containing auxiliary values (imaginary part)

   Input/Output Parameter:
.  k    - (optional) number of elements in the leading group

   Notes:
   This routine sorts the arrays provided in eigr and eigi, and also
   sorts the dense system stored inside ds (assumed to be in condensed form).
   The sorting criterion is specified with DSSetEigenvalueComparison().

   If arrays rr and ri are provided, then a (partial) reordering based on these
   values rather than on the eigenvalues is performed. In symmetric problems
   a total order is obtained (parameter k is ignored), but otherwise the result
   is sorted only partially. In this latter case, it is only guaranteed that
   all the first k elements satisfy the comparison with any of the last n-k
   elements. The output value of parameter k is the final number of elements in
   the first set.

   Level: advanced

.seealso: DSSolve(), DSSetEigenvalueComparison()
@*/
PetscErrorCode DSSort(DS ds,PetscScalar *eigr,PetscScalar *eigi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(eigr,2);
  if (rr) PetscValidPointer(rr,4);
  if (ds->state<DS_STATE_CONDENSED) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSSolve() first");
  if (ds->state==DS_STATE_SORTED) PetscFunctionReturn(0);
  if (!ds->ops->sort) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (!ds->comp_fun) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must provide a sorting criterion with DSSetEigenvalueComparison() first");
  if (k && !rr) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Argument k can only be used together with rr");
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->sort)(ds,eigr,eigi,rr,ri,k);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ds->state = DS_STATE_SORTED;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSVectors"
/*@
   DSVectors - Compute vectors associated to the dense system such
   as eigenvectors.

   Logically Collective on DS

   Input Parameters:
+  ds  - the direct solver context
-  mat - the matrix, used to indicate which vectors are required

   Input/Output Parameter:
-  j   - (optional) index of vector to be computed

   Output Parameter:
.  rnorm - (optional) computed residual norm

   Notes:
   Allowed values for mat are DS_MAT_X, DS_MAT_Y, DS_MAT_U and DS_MAT_VT, to
   compute right or left eigenvectors, or left or right singular vectors,
   respectively.

   If PETSC_NULL is passed in argument j then all vectors are computed,
   otherwise j indicates which vector must be computed. In real non-symmetric
   problems, on exit the index j will be incremented when a complex conjugate
   pair is found.

   This function can be invoked after the dense problem has been solved,
   to get the residual norm estimate of the associated Ritz pair. In that
   case, the relevant information is returned in rnorm.

   For computing eigenvectors, LAPACK's _trevc is used so the matrix must
   be in (quasi-)triangular form, or call DSSolve() first.

   Level: advanced

.seealso: DSSolve()
@*/
PetscErrorCode DSVectors(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (!ds->ops->vectors) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (rnorm && !j) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must give a value of j");
  ierr = DSAllocateMat_Private(ds,mat);CHKERRQ(ierr); 
  ierr = PetscLogEventBegin(DS_Vectors,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->vectors)(ds,mat,j,rnorm);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Vectors,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSNormalize"
/*@
   DSNormalize - Normalize a column or all the columns of a matrix. Considers
   the case when the columns represent the real and the imaginary part of a vector.          

   Logically Collective on DS

   Input Parameter:
+  ds  - the direct solver context
.  mat - the matrix to be modified
-  col - the column to normalize or -1 to normalize all of them

   Notes:
   The columns are normalized with respect to the 2-norm.

   If col and col+1 (or col-1 and col) represent the real and the imaginary
   part of a vector, both columns are scaled.

   Level: advanced
@*/
PetscErrorCode DSNormalize(DS ds,DSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,mat,2);
  PetscValidLogicalCollectiveInt(ds,col,3);
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  if (!ds->ops->normalize) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (col<-1) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"col should be at least minus one");
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->normalize)(ds,mat,col);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSUpdateExtraRow"
/*@C
   DSUpdateExtraRow - Performs all necessary operations so that the extra
   row gets up-to-date after a call to DSSolve().

   Not Collective

   Input Parameters:
.  ds - the direct solver context

   Level: advanced

.seealso: DSSolve(), DSSetExtraRow()
@*/
PetscErrorCode DSUpdateExtraRow(DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (!ds->ops->update) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (!ds->extrarow) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONGSTATE,"Should have called DSSetExtraRow");
  if (!ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Must call DSAllocate() first");
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->update)(ds);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSCond"
/*@C
   DSCond - Compute the inf-norm condition number of the first matrix
   as cond(A) = norm(A)*norm(inv(A)).

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  cond - the computed condition number

   Level: advanced

.seealso: DSSolve()
@*/
PetscErrorCode DSCond(DS ds,PetscReal *cond)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(cond,2);
  if (!ds->ops->cond) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->cond)(ds,cond);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSTranslateHarmonic"
/*@C
   DSTranslateHarmonic - Computes a translation of the dense system.

   Logically Collective on DS

   Input Parameters:
+  ds      - the direct solver context
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
PetscErrorCode DSTranslateHarmonic(DS ds,PetscScalar tau,PetscReal beta,PetscBool recover,PetscScalar *g,PetscReal *gamma)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (!ds->ops->transharm) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->transharm)(ds,tau,beta,recover,g,gamma);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ds->state = DS_STATE_RAW;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSTranslateRKS"
/*@C
   DSTranslateRKS - Computes a modification of the dense system corresponding
   to an update of the shift in a rational Krylov method.

   Logically Collective on DS

   Input Parameters:
+  ds    - the direct solver context
-  alpha - the translation amount

   Notes:
   This function is intended for use in the context of Krylov methods only.
   It takes the leading (k+1,k) submatrix of A, containing the truncated
   Rayleigh quotient of a Krylov-Schur relation computed from a shift
   sigma1 and transforms it to obtain a Krylov relation as if computed 
   from a different shift sigma2. The new matrix is computed as
   1.0/alpha*(eye(k)-Q*inv(R)), where [Q,R]=qr(eye(k)-alpha*A) and
   alpha = sigma1-sigma2.

   Matrix Q is placed in DS_MAT_Q so that it can be used to update the
   Krylov basis.

   Level: developer
@*/
PetscErrorCode DSTranslateRKS(DS ds,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (!ds->ops->transrks) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  ierr = (*ds->ops->transrks)(ds,alpha);CHKERRQ(ierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ds->state   = DS_STATE_RAW;
  ds->compact = PETSC_FALSE;
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

