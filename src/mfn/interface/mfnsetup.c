/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   MFN routines related to problem setup
*/

#include <slepc/private/mfnimpl.h>       /*I "slepcmfn.h" I*/

/*@
   MFNSetUp - Sets up all the internal data structures necessary for the
   execution of the matrix function solver.

   Collective on mfn

   Input Parameter:
.  mfn   - matrix function context

   Notes:
   This function need not be called explicitly in most cases, since MFNSolve()
   calls it. It can be useful when one wants to measure the set-up time
   separately from the solve time.

   Level: developer

.seealso: MFNCreate(), MFNSolve(), MFNDestroy()
@*/
PetscErrorCode MFNSetUp(MFN mfn)
{
  PetscInt       N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);

  /* reset the convergence flag from the previous solves */
  mfn->reason = MFN_CONVERGED_ITERATING;

  if (mfn->setupcalled) PetscFunctionReturn(0);
  CHKERRQ(PetscLogEventBegin(MFN_SetUp,mfn,0,0,0));

  /* Set default solver type (MFNSetFromOptions was not called) */
  if (!((PetscObject)mfn)->type_name) CHKERRQ(MFNSetType(mfn,MFNKRYLOV));
  if (!mfn->fn) CHKERRQ(MFNGetFN(mfn,&mfn->fn));
  if (!((PetscObject)mfn->fn)->type_name) CHKERRQ(FNSetFromOptions(mfn->fn));

  /* Check problem dimensions */
  PetscCheck(mfn->A,PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_WRONGSTATE,"MFNSetOperator must be called first");
  CHKERRQ(MatGetSize(mfn->A,&N,NULL));
  if (mfn->ncv > N) mfn->ncv = N;

  /* call specific solver setup */
  CHKERRQ((*mfn->ops->setup)(mfn));

  /* set tolerance if not yet set */
  if (mfn->tol==PETSC_DEFAULT) mfn->tol = SLEPC_DEFAULT_TOL;

  CHKERRQ(PetscLogEventEnd(MFN_SetUp,mfn,0,0,0));
  mfn->setupcalled = 1;
  PetscFunctionReturn(0);
}

/*@
   MFNSetOperator - Sets the matrix for which the matrix function is to be computed.

   Collective on mfn

   Input Parameters:
+  mfn - the matrix function context
-  A   - the problem matrix

   Notes:
   It must be called before MFNSetUp(). If it is called again after MFNSetUp() then
   the MFN object is reset.

   Level: beginner

.seealso: MFNSolve(), MFNSetUp(), MFNReset()
@*/
PetscErrorCode MFNSetOperator(MFN mfn,Mat A)
{
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(mfn,1,A,2);

  CHKERRQ(MatGetSize(A,&m,&n));
  PetscCheck(m==n,PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_WRONG,"A is a non-square matrix");
  CHKERRQ(PetscObjectReference((PetscObject)A));
  if (mfn->setupcalled) CHKERRQ(MFNReset(mfn));
  else CHKERRQ(MatDestroy(&mfn->A));
  mfn->A = A;
  mfn->setupcalled = 0;
  PetscFunctionReturn(0);
}

/*@
   MFNGetOperator - Gets the matrix associated with the MFN object.

   Collective on mfn

   Input Parameter:
.  mfn - the MFN context

   Output Parameters:
.  A  - the matrix for which the matrix function is to be computed

   Level: intermediate

.seealso: MFNSolve(), MFNSetOperator()
@*/
PetscErrorCode MFNGetOperator(MFN mfn,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(A,2);
  *A = mfn->A;
  PetscFunctionReturn(0);
}

/*@
   MFNAllocateSolution - Allocate memory storage for common variables such
   as the basis vectors.

   Collective on mfn

   Input Parameters:
+  mfn   - matrix function context
-  extra - number of additional positions, used for methods that require a
           working basis slightly larger than ncv

   Developer Notes:
   This is SLEPC_EXTERN because it may be required by user plugin MFN
   implementations.

   Level: developer

.seealso: MFNSetUp()
@*/
PetscErrorCode MFNAllocateSolution(MFN mfn,PetscInt extra)
{
  PetscInt       oldsize,requested;
  Vec            t;

  PetscFunctionBegin;
  requested = mfn->ncv + extra;

  /* oldsize is zero if this is the first time setup is called */
  CHKERRQ(BVGetSizes(mfn->V,NULL,NULL,&oldsize));

  /* allocate basis vectors */
  if (!mfn->V) CHKERRQ(MFNGetBV(mfn,&mfn->V));
  if (!oldsize) {
    if (!((PetscObject)(mfn->V))->type_name) CHKERRQ(BVSetType(mfn->V,BVSVEC));
    CHKERRQ(MatCreateVecsEmpty(mfn->A,&t,NULL));
    CHKERRQ(BVSetSizesFromVec(mfn->V,t,requested));
    CHKERRQ(VecDestroy(&t));
  } else CHKERRQ(BVResize(mfn->V,requested,PETSC_FALSE));
  PetscFunctionReturn(0);
}
