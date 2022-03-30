/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   DS operations: DSSolve(), DSVectors(), etc
*/

#include <slepc/private/dsimpl.h>      /*I "slepcds.h" I*/

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
  PetscValidIntPointer(ld,2);
  *ld = ds->ld;
  PetscFunctionReturn(0);
}

/*@
   DSSetState - Change the state of the DS object.

   Logically Collective on ds

   Input Parameters:
+  ds    - the direct solver context
-  state - the new state

   Notes:
   The state indicates that the dense system is in an initial state (raw),
   in an intermediate state (such as tridiagonal, Hessenberg or
   Hessenberg-triangular), in a condensed state (such as diagonal, Schur or
   generalized Schur), or in a truncated state.

   This function is normally used to return to the raw state when the
   condensed structure is destroyed.

   Level: advanced

.seealso: DSGetState()
@*/
PetscErrorCode DSSetState(DS ds,DSStateType state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,state,2);
  switch (state) {
    case DS_STATE_RAW:
    case DS_STATE_INTERMEDIATE:
    case DS_STATE_CONDENSED:
    case DS_STATE_TRUNCATED:
      if (ds->state!=state) PetscCall(PetscInfo(ds,"State has changed from %s to %s\n",DSStateTypes[ds->state],DSStateTypes[state]));
      ds->state = state;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Wrong state");
  }
  PetscFunctionReturn(0);
}

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
  PetscValidPointer(state,2);
  *state = ds->state;
  PetscFunctionReturn(0);
}

/*@
   DSSetDimensions - Resize the matrices in the DS object.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
.  n  - the new size
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Notes:
   The internal arrays are not reallocated.

   Some DS types have additional dimensions, e.g. the number of columns
   in DSSVD. For these, you should call a specific interface function.

   Level: intermediate

.seealso: DSGetDimensions(), DSAllocate(), DSSVDSetDimensions()
@*/
PetscErrorCode DSSetDimensions(DS ds,PetscInt n,PetscInt l,PetscInt k)
{
  PetscInt       on,ol,ok;
  PetscBool      issvd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveInt(ds,n,2);
  PetscValidLogicalCollectiveInt(ds,l,3);
  PetscValidLogicalCollectiveInt(ds,k,4);
  on = ds->n; ol = ds->l; ok = ds->k;
  if (n==PETSC_DECIDE || n==PETSC_DEFAULT) {
    ds->n = ds->ld;
  } else {
    PetscCheck(n>=0 && n<=ds->ld,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between 0 and ld");
    PetscCall(PetscObjectTypeCompareAny((PetscObject)ds,&issvd,DSSVD,DSGSVD,""));  /* SVD and GSVD have extra column instead of extra row */
    PetscCheck(!ds->extrarow || n<ds->ld || issvd,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"A value of n equal to ld leaves no room for extra row");
    ds->n = n;
  }
  ds->t = ds->n;   /* truncated length equal to the new dimension */
  if (l==PETSC_DECIDE || l==PETSC_DEFAULT) {
    ds->l = 0;
  } else {
    PetscCheck(l>=0 && l<=ds->n,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of l. Must be between 0 and n");
    ds->l = l;
  }
  if (k==PETSC_DECIDE || k==PETSC_DEFAULT) {
    ds->k = ds->n/2;
  } else {
    PetscCheck(k>=0 || k<=ds->n,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of k. Must be between 0 and n");
    ds->k = k;
  }
  if (on!=ds->n || ol!=ds->l || ok!=ds->k) PetscCall(PetscInfo(ds,"New dimensions are: n=%" PetscInt_FMT ", l=%" PetscInt_FMT ", k=%" PetscInt_FMT "\n",ds->n,ds->l,ds->k));
  PetscFunctionReturn(0);
}

/*@
   DSGetDimensions - Returns the current dimensions.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
+  n  - the current size
.  l  - number of locked (inactive) leading columns
.  k  - intermediate dimension (e.g., position of arrow)
-  t  - truncated length

   Note:
   The t parameter makes sense only if DSTruncate() has been called.
   Otherwise its value equals n.

   Some DS types have additional dimensions, e.g. the number of columns
   in DSSVD. For these, you should call a specific interface function.

   Level: intermediate

.seealso: DSSetDimensions(), DSTruncate(), DSSVDGetDimensions()
@*/
PetscErrorCode DSGetDimensions(DS ds,PetscInt *n,PetscInt *l,PetscInt *k,PetscInt *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  if (n) *n = ds->n;
  if (l) *l = ds->l;
  if (k) *k = ds->k;
  if (t) *t = ds->t;
  PetscFunctionReturn(0);
}

/*@
   DSTruncate - Truncates the system represented in the DS object.

   Logically Collective on ds

   Input Parameters:
+  ds   - the direct solver context
.  n    - the new size
-  trim - a flag to indicate if the factorization must be trimmed

   Note:
   The new size is set to n. Note that in some cases the new size could
   be n+1 or n-1 to avoid breaking a 2x2 diagonal block (e.g. in real
   Schur form). In cases where the extra row is meaningful, the first
   n elements are kept as the extra row for the new system.

   If the flag trim is turned on, it resets the locked and intermediate
   dimensions to zero, see DSSetDimensions(), and sets the state to RAW.
   It also cleans the extra row if being used.

   The typical usage of trim=true is to truncate the Schur decomposition
   at the end of a Krylov iteration. In this case, the state must be
   changed to RAW so that DSVectors() computes eigenvectors from scratch.

   Level: advanced

.seealso: DSSetDimensions(), DSSetExtraRow(), DSStateType
@*/
PetscErrorCode DSTruncate(DS ds,PetscInt n,PetscBool trim)
{
  DSStateType    old;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveInt(ds,n,2);
  PetscValidLogicalCollectiveBool(ds,trim,3);
  PetscCheck(ds->ops->truncate,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  PetscCheck(n>=ds->l && n<=ds->n,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n (%" PetscInt_FMT "). Must be between l (%" PetscInt_FMT ") and n (%" PetscInt_FMT ")",n,ds->l,ds->n);
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->truncate)(ds,n,trim));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscCall(PetscInfo(ds,"Decomposition %s to size n=%" PetscInt_FMT "\n",trim?"trimmed":"truncated",ds->n));
  old = ds->state;
  ds->state = trim? DS_STATE_RAW: DS_STATE_TRUNCATED;
  if (old!=ds->state) PetscCall(PetscInfo(ds,"State has changed from %s to %s\n",DSStateTypes[old],DSStateTypes[ds->state]));
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@
   DSMatGetSize - Returns the numbers of rows and columns of one of the DS matrices.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  t  - the requested matrix

   Output Parameters:
+  n  - the number of rows
-  m  - the number of columns

   Note:
   This is equivalent to MatGetSize() on a matrix obtained with DSGetMat().

   Level: developer

.seealso: DSSetDimensions(), DSGetMat()
@*/
PetscErrorCode DSMatGetSize(DS ds,DSMatType t,PetscInt *m,PetscInt *n)
{
  PetscInt       rows,cols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckValidMat(ds,t,2);
  if (ds->ops->matgetsize) PetscCall((*ds->ops->matgetsize)(ds,t,&rows,&cols));
  else {
    if (ds->state==DS_STATE_TRUNCATED && t>=DS_MAT_Q) rows = ds->t;
    else rows = (t==DS_MAT_A && ds->extrarow)? ds->n+1: ds->n;
    cols = ds->n;
  }
  if (m) *m = rows;
  if (n) *n = cols;
  PetscFunctionReturn(0);
}

/*@
   DSMatIsHermitian - Checks if one of the DS matrices is known to be Hermitian.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  t  - the requested matrix

   Output Parameter:
.  flg - the Hermitian flag

   Note:
   Does not check the matrix values directly. The flag is set according to the
   problem structure. For instance, in DSHEP matrix A is Hermitian.

   Level: developer

.seealso: DSGetMat()
@*/
PetscErrorCode DSMatIsHermitian(DS ds,DSMatType t,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckValidMat(ds,t,2);
  PetscValidBoolPointer(flg,3);
  if (ds->ops->hermitian) PetscCall((*ds->ops->hermitian)(ds,t,flg));
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode DSGetTruncateSize_Default(DS ds,PetscInt l,PetscInt n,PetscInt *k)
{
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar *A = ds->mat[DS_MAT_A];
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  if (A[l+(*k)+(l+(*k)-1)*ds->ld] != 0.0) {
    if (l+(*k)<n-1) (*k)++;
    else (*k)--;
  }
#endif
  PetscFunctionReturn(0);
}

/*@
   DSGetTruncateSize - Gets the correct size to be used in DSTruncate()
   to avoid breaking a 2x2 block.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
.  l  - the size of the locked part (set to 0 to use ds->l)
.  n  - the total matrix size (set to 0 to use ds->n)
-  k  - the wanted truncation size

   Output Parameter:
.  k  - the possibly modified value of the truncation size

   Notes:
   This should be called before DSTruncate() to make sure that the truncation
   does not break a 2x2 block corresponding to a complex conjugate eigenvalue.

   The total size is n (either user-provided or ds->n if 0 is passed). The
   size where the truncation is intended is equal to l+k (where l can be
   equal to the locked size ds->l if set to 0). Then if there is a 2x2 block
   at the l+k limit, the value of k is increased (or decreased) by 1.

   Level: advanced

.seealso: DSTruncate(), DSSetDimensions()
@*/
PetscErrorCode DSGetTruncateSize(DS ds,PetscInt l,PetscInt n,PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveInt(ds,l,2);
  PetscValidLogicalCollectiveInt(ds,n,3);
  PetscValidIntPointer(k,4);
  if (ds->ops->gettruncatesize) PetscCall((*ds->ops->gettruncatesize)(ds,l?l:ds->l,n?n:ds->n,k));
  PetscFunctionReturn(0);
}

/*@
   DSGetMat - Returns a sequential dense Mat object containing the requested
   matrix.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  m  - the requested matrix

   Output Parameter:
.  A  - Mat object

   Notes:
   The Mat is created with sizes equal to the current DS dimensions (nxm),
   then it is filled with the values that would be obtained with DSGetArray()
   (not DSGetArrayReal()). If the DS was truncated, then the number of rows
   is equal to the dimension prior to truncation, see DSTruncate().
   The communicator is always PETSC_COMM_SELF.

   When no longer needed, the user can either destroy the matrix or call
   DSRestoreMat(). The latter will copy back the modified values.

   Level: advanced

.seealso: DSRestoreMat(), DSSetDimensions(), DSGetArray(), DSGetArrayReal(), DSTruncate()
@*/
PetscErrorCode DSGetMat(DS ds,DSMatType m,Mat *A)
{
  PetscInt       j,rows,cols,arows,acols;
  PetscBool      create=PETSC_FALSE,flg;
  PetscScalar    *pA,*M;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  DSCheckValidMat(ds,m,2);
  PetscValidPointer(A,3);
  PetscCheck(m!=DS_MAT_T && m!=DS_MAT_D,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Not implemented for T or D matrices");

  PetscCall(DSMatGetSize(ds,m,&rows,&cols));
  if (!ds->omat[m]) create=PETSC_TRUE;
  else {
    PetscCall(MatGetSize(ds->omat[m],&arows,&acols));
    if (arows!=rows || acols!=cols) {
      PetscCall(MatDestroy(&ds->omat[m]));
      create=PETSC_TRUE;
    }
  }
  if (create) PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,rows,cols,NULL,&ds->omat[m]));

  /* set Hermitian flag */
  PetscCall(DSMatIsHermitian(ds,m,&flg));
  PetscCall(MatSetOption(ds->omat[m],MAT_HERMITIAN,flg));

  /* copy entries */
  PetscCall(PetscObjectReference((PetscObject)ds->omat[m]));
  *A = ds->omat[m];
  M  = ds->mat[m];
  PetscCall(MatDenseGetArray(*A,&pA));
  for (j=0;j<cols;j++) PetscCall(PetscArraycpy(pA+j*rows,M+j*ds->ld,rows));
  PetscCall(MatDenseRestoreArray(*A,&pA));
  PetscFunctionReturn(0);
}

/*@
   DSRestoreMat - Restores the matrix after DSGetMat() was called.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
.  m  - the requested matrix
-  A  - the fetched Mat object

   Notes:
   A call to this function must match a previous call of DSGetMat().
   The effect is that the contents of the Mat are copied back to the
   DS internal array, and the matrix is destroyed.

   It is not compulsory to call this function, the matrix obtained with
   DSGetMat() can simply be destroyed if entries need not be copied back.

   Level: advanced

.seealso: DSGetMat(), DSRestoreArray(), DSRestoreArrayReal()
@*/
PetscErrorCode DSRestoreMat(DS ds,DSMatType m,Mat *A)
{
  PetscInt       j,rows,cols;
  PetscScalar    *pA,*M;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  DSCheckValidMat(ds,m,2);
  PetscValidPointer(A,3);
  PetscCheck(ds->omat[m],PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONGSTATE,"DSRestoreMat must match a previous call to DSGetMat");
  PetscCheck(ds->omat[m]==*A,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONGSTATE,"Mat argument is not the same as the one obtained with DSGetMat");

  PetscCall(MatGetSize(*A,&rows,&cols));
  M  = ds->mat[m];
  PetscCall(MatDenseGetArray(*A,&pA));
  for (j=0;j<cols;j++) PetscCall(PetscArraycpy(M+j*ds->ld,pA+j*rows,rows));
  PetscCall(MatDenseRestoreArray(*A,&pA));
  PetscCall(MatDestroy(A));
  PetscFunctionReturn(0);
}

/*@C
   DSGetArray - Returns a pointer to one of the internal arrays used to
   represent matrices. You MUST call DSRestoreArray() when you no longer
   need to access the array.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  m  - the requested matrix

   Output Parameter:
.  a  - pointer to the values

   Level: advanced

.seealso: DSRestoreArray(), DSGetArrayReal()
@*/
PetscErrorCode DSGetArray(DS ds,DSMatType m,PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  DSCheckValidMat(ds,m,2);
  PetscValidPointer(a,3);
  *a = ds->mat[m];
  CHKMEMQ;
  PetscFunctionReturn(0);
}

/*@C
   DSRestoreArray - Restores the matrix after DSGetArray() was called.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
.  m  - the requested matrix
-  a  - pointer to the values

   Level: advanced

.seealso: DSGetArray(), DSGetArrayReal()
@*/
PetscErrorCode DSRestoreArray(DS ds,DSMatType m,PetscScalar *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  DSCheckValidMat(ds,m,2);
  PetscValidPointer(a,3);
  CHKMEMQ;
  *a = 0;
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@C
   DSGetArrayReal - Returns a pointer to one of the internal arrays used to
   represent real matrices. You MUST call DSRestoreArrayReal() when you no longer
   need to access the array.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
-  m  - the requested matrix

   Output Parameter:
.  a  - pointer to the values

   Level: advanced

.seealso: DSRestoreArrayReal(), DSGetArray()
@*/
PetscErrorCode DSGetArrayReal(DS ds,DSMatType m,PetscReal *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  DSCheckValidMatReal(ds,m,2);
  PetscValidPointer(a,3);
  *a = ds->rmat[m];
  CHKMEMQ;
  PetscFunctionReturn(0);
}

/*@C
   DSRestoreArrayReal - Restores the matrix after DSGetArrayReal() was called.

   Not Collective

   Input Parameters:
+  ds - the direct solver context
.  m  - the requested matrix
-  a  - pointer to the values

   Level: advanced

.seealso: DSGetArrayReal(), DSGetArray()
@*/
PetscErrorCode DSRestoreArrayReal(DS ds,DSMatType m,PetscReal *a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  DSCheckValidMatReal(ds,m,2);
  PetscValidPointer(a,3);
  CHKMEMQ;
  *a = 0;
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@
   DSSolve - Solves the problem.

   Logically Collective on ds

   Input Parameters:
+  ds   - the direct solver context
.  eigr - array to store the computed eigenvalues (real part)
-  eigi - array to store the computed eigenvalues (imaginary part)

   Note:
   This call brings the dense system to condensed form. No ordering
   of the eigenvalues is enforced (for this, call DSSort() afterwards).

   Level: intermediate

.seealso: DSSort(), DSStateType
@*/
PetscErrorCode DSSolve(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscValidScalarPointer(eigr,2);
  if (ds->state>=DS_STATE_CONDENSED) PetscFunctionReturn(0);
  PetscCheck(ds->ops->solve[ds->method],PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The specified method number does not exist for this DS");
  PetscCall(PetscInfo(ds,"Starting solve with problem sizes: n=%" PetscInt_FMT ", l=%" PetscInt_FMT ", k=%" PetscInt_FMT "\n",ds->n,ds->l,ds->k));
  PetscCall(PetscLogEventBegin(DS_Solve,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->solve[ds->method])(ds,eigr,eigi));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Solve,ds,0,0,0));
  PetscCall(PetscInfo(ds,"State has changed from %s to CONDENSED\n",DSStateTypes[ds->state]));
  ds->state = DS_STATE_CONDENSED;
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@C
   DSSort - Sorts the result of DSSolve() according to a given sorting
   criterion.

   Logically Collective on ds

   Input Parameters:
+  ds   - the direct solver context
.  rr   - (optional) array containing auxiliary values (real part)
-  ri   - (optional) array containing auxiliary values (imaginary part)

   Input/Output Parameters:
+  eigr - array containing the computed eigenvalues (real part)
.  eigi - array containing the computed eigenvalues (imaginary part)
-  k    - (optional) number of elements in the leading group

   Notes:
   This routine sorts the arrays provided in eigr and eigi, and also
   sorts the dense system stored inside ds (assumed to be in condensed form).
   The sorting criterion is specified with DSSetSlepcSC().

   If arrays rr and ri are provided, then a (partial) reordering based on these
   values rather than on the eigenvalues is performed. In symmetric problems
   a total order is obtained (parameter k is ignored), but otherwise the result
   is sorted only partially. In this latter case, it is only guaranteed that
   all the first k elements satisfy the comparison with any of the last n-k
   elements. The output value of parameter k is the final number of elements in
   the first set.

   Level: intermediate

.seealso: DSSolve(), DSSetSlepcSC(), DSSortWithPermutation()
@*/
PetscErrorCode DSSort(DS ds,PetscScalar *eigr,PetscScalar *eigi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckSolved(ds,1);
  PetscValidScalarPointer(eigr,2);
  if (rr) PetscValidScalarPointer(rr,4);
  PetscCheck(ds->state<DS_STATE_TRUNCATED,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"Cannot sort a truncated DS");
  PetscCheck(ds->ops->sort,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  PetscCheck(ds->sc,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"Must provide a sorting criterion first");
  PetscCheck(!k || rr,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Argument k can only be used together with rr");

  for (i=0;i<ds->n;i++) ds->perm[i] = i;   /* initialize to trivial permutation */
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->sort)(ds,eigr,eigi,rr,ri,k));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscCall(PetscInfo(ds,"Finished sorting\n"));
  PetscFunctionReturn(0);
}

/*@C
   DSSortWithPermutation - Reorders the result of DSSolve() according to a given
   permutation.

   Logically Collective on ds

   Input Parameters:
+  ds   - the direct solver context
-  perm - permutation that indicates the new ordering

   Input/Output Parameters:
+  eigr - array with the reordered eigenvalues (real part)
-  eigi - array with the reordered eigenvalues (imaginary part)

   Notes:
   This routine reorders the arrays provided in eigr and eigi, and also the dense
   system stored inside ds (assumed to be in condensed form). There is no sorting
   criterion, as opposed to DSSort(). Instead, the new ordering is given in argument perm.

   Level: advanced

.seealso: DSSolve(), DSSort()
@*/
PetscErrorCode DSSortWithPermutation(DS ds,PetscInt *perm,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckSolved(ds,1);
  PetscValidIntPointer(perm,2);
  PetscValidScalarPointer(eigr,3);
  PetscCheck(ds->state<DS_STATE_TRUNCATED,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"Cannot sort a truncated DS");
  PetscCheck(ds->ops->sortperm,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);

  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->sortperm)(ds,perm,eigr,eigi));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscCall(PetscInfo(ds,"Finished sorting\n"));
  PetscFunctionReturn(0);
}

/*@
   DSSynchronize - Make sure that all processes have the same data, performing
   communication if necessary.

   Collective on ds

   Input Parameter:
.  ds   - the direct solver context

   Input/Output Parameters:
+  eigr - (optional) array with the computed eigenvalues (real part)
-  eigi - (optional) array with the computed eigenvalues (imaginary part)

   Notes:
   When the DS has been created with a communicator with more than one process,
   the internal data, especially the computed matrices, may diverge in the
   different processes. This happens when using multithreaded BLAS and may
   cause numerical issues in some ill-conditioned problems. This function
   performs the necessary communication among the processes so that the
   internal data is exactly equal in all of them.

   Depending on the parallel mode as set with DSSetParallel(), this function
   will either do nothing or synchronize the matrices computed by DSSolve()
   and DSSort(). The arguments eigr and eigi are typically those used in the
   calls to DSSolve() and DSSort().

   Level: developer

.seealso: DSSetParallel(), DSSolve(), DSSort()
@*/
PetscErrorCode DSSynchronize(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ds),&size));
  if (size>1 && ds->pmode==DS_PARALLEL_SYNCHRONIZED) {
    PetscCall(PetscLogEventBegin(DS_Synchronize,ds,0,0,0));
    if (ds->ops->synchronize) PetscCall((*ds->ops->synchronize)(ds,eigr,eigi));
    PetscCall(PetscLogEventEnd(DS_Synchronize,ds,0,0,0));
    PetscCall(PetscObjectStateIncrease((PetscObject)ds));
    PetscCall(PetscInfo(ds,"Synchronization completed (%s)\n",DSParallelTypes[ds->pmode]));
  }
  PetscFunctionReturn(0);
}

/*@C
   DSVectors - Compute vectors associated to the dense system such
   as eigenvectors.

   Logically Collective on ds

   Input Parameters:
+  ds  - the direct solver context
-  mat - the matrix, used to indicate which vectors are required

   Input/Output Parameter:
.  j   - (optional) index of vector to be computed

   Output Parameter:
.  rnorm - (optional) computed residual norm

   Notes:
   Allowed values for mat are DS_MAT_X, DS_MAT_Y, DS_MAT_U and DS_MAT_V, to
   compute right or left eigenvectors, or left or right singular vectors,
   respectively.

   If NULL is passed in argument j then all vectors are computed,
   otherwise j indicates which vector must be computed. In real non-symmetric
   problems, on exit the index j will be incremented when a complex conjugate
   pair is found.

   This function can be invoked after the dense problem has been solved,
   to get the residual norm estimate of the associated Ritz pair. In that
   case, the relevant information is returned in rnorm.

   For computing eigenvectors, LAPACK's _trevc is used so the matrix must
   be in (quasi-)triangular form, or call DSSolve() first.

   Level: intermediate

.seealso: DSSolve()
@*/
PetscErrorCode DSVectors(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  PetscCheck(mat<DS_NUM_MAT,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Invalid matrix");
  PetscCheck(ds->ops->vectors,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  PetscCheck(!rnorm || j,PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"Must give a value of j");
  if (!ds->mat[mat]) PetscCall(DSAllocateMat_Private(ds,mat));
  if (!j) PetscCall(PetscInfo(ds,"Computing all vectors on %s\n",DSMatName[mat]));
  PetscCall(PetscLogEventBegin(DS_Vectors,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->vectors)(ds,mat,j,rnorm));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Vectors,ds,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscCheck(ds->ops->update,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  PetscCheck(ds->extrarow,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONGSTATE,"Should have called DSSetExtraRow");
  PetscCall(PetscInfo(ds,"Updating extra row\n"));
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->update)(ds));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscFunctionReturn(0);
}

/*@
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscValidRealPointer(cond,2);
  PetscCheck(ds->ops->cond,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->cond)(ds,cond));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscCall(PetscInfo(ds,"Computed condition number = %g\n",(double)*cond));
  PetscFunctionReturn(0);
}

/*@C
   DSTranslateHarmonic - Computes a translation of the dense system.

   Logically Collective on ds

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

.seealso: DSTranslateRKS()
@*/
PetscErrorCode DSTranslateHarmonic(DS ds,PetscScalar tau,PetscReal beta,PetscBool recover,PetscScalar *g,PetscReal *gamma)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscCheck(ds->ops->transharm,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  if (recover) PetscCall(PetscInfo(ds,"Undoing the translation\n"));
  else PetscCall(PetscInfo(ds,"Computing the translation\n"));
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->transharm)(ds,tau,beta,recover,g,gamma));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  ds->state = DS_STATE_RAW;
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@
   DSTranslateRKS - Computes a modification of the dense system corresponding
   to an update of the shift in a rational Krylov method.

   Logically Collective on ds

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

.seealso: DSTranslateHarmonic()
@*/
PetscErrorCode DSTranslateRKS(DS ds,PetscScalar alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidType(ds,1);
  DSCheckAlloc(ds,1);
  PetscCheck(ds->ops->transrks,PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"DS type %s",((PetscObject)ds)->type_name);
  PetscCall(PetscInfo(ds,"Translating with alpha=%g\n",(double)PetscRealPart(alpha)));
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall((*ds->ops->transrks)(ds,alpha));
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  ds->state   = DS_STATE_RAW;
  ds->compact = PETSC_FALSE;
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*@
   DSCopyMat - Copies the contents of a sequential dense Mat object to
   the indicated DS matrix, or vice versa.

   Not Collective

   Input Parameters:
+  ds   - the direct solver context
.  m    - the requested matrix
.  mr   - first row of m to be considered
.  mc   - first column of m to be considered
.  A    - Mat object
.  Ar   - first row of A to be considered
.  Ac   - first column of A to be considered
.  rows - number of rows to copy
.  cols - number of columns to copy
-  out  - whether the data is copied out of the DS

   Note:
   If out=true, the values of the DS matrix m are copied to A, otherwise
   the entries of A are copied to the DS.

   Level: developer

.seealso: DSGetMat()
@*/
PetscErrorCode DSCopyMat(DS ds,DSMatType m,PetscInt mr,PetscInt mc,Mat A,PetscInt Ar,PetscInt Ac,PetscInt rows,PetscInt cols,PetscBool out)
{
  PetscInt       j,mrows,mcols,arows,acols;
  PetscScalar    *pA,*M;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveEnum(ds,m,2);
  DSCheckValidMat(ds,m,2);
  PetscValidLogicalCollectiveInt(ds,mr,3);
  PetscValidLogicalCollectiveInt(ds,mc,4);
  PetscValidHeaderSpecific(A,MAT_CLASSID,5);
  PetscValidLogicalCollectiveInt(ds,Ar,6);
  PetscValidLogicalCollectiveInt(ds,Ac,7);
  PetscValidLogicalCollectiveInt(ds,rows,8);
  PetscValidLogicalCollectiveInt(ds,cols,9);
  PetscValidLogicalCollectiveBool(ds,out,10);
  if (!rows || !cols) PetscFunctionReturn(0);

  PetscCall(DSMatGetSize(ds,m,&mrows,&mcols));
  PetscCall(MatGetSize(A,&arows,&acols));
  PetscCheck(m!=DS_MAT_T && m!=DS_MAT_D,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Not implemented for T or D matrices");
  PetscCheck(mr>=0 && mr<mrows,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid initial row in m");
  PetscCheck(mc>=0 && mc<mcols,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid initial column in m");
  PetscCheck(Ar>=0 && Ar<arows,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid initial row in A");
  PetscCheck(Ac>=0 && Ac<acols,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid initial column in A");
  PetscCheck(mr+rows<=mrows && Ar+rows<=arows,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid number of rows");
  PetscCheck(mc+cols<=mcols && Ac+cols<=acols,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid number of columns");

  M  = ds->mat[m];
  PetscCall(MatDenseGetArray(A,&pA));
  for (j=0;j<cols;j++) {
    if (out) PetscCall(PetscArraycpy(pA+(Ac+j)*arows+Ar,M+(mc+j)*ds->ld+mr,rows));
    else PetscCall(PetscArraycpy(M+(mc+j)*ds->ld+mr,pA+(Ac+j)*arows+Ar,rows));
  }
  PetscCall(MatDenseRestoreArray(A,&pA));
  PetscFunctionReturn(0);
}
