/*
      MFN routines related to problem setup.

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

#include <slepc-private/mfnimpl.h>       /*I "slepcmfn.h" I*/
#include <slepc-private/ipimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "MFNSetUp"
/*@
   MFNSetUp - Sets up all the internal data structures necessary for the
   execution of the matrix function solver.

   Collective on MFN

   Input Parameter:
.  mfn   - matrix function context

   Notes:
   This function need not be called explicitly in most cases, since MFNSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   Level: advanced

.seealso: MFNCreate(), MFNSolve(), MFNDestroy()
@*/
PetscErrorCode MFNSetUp(MFN mfn)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (mfn->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(MFN_SetUp,mfn,0,0,0);CHKERRQ(ierr);

  /* reset the convergence flag from the previous solves */
  mfn->reason = MFN_CONVERGED_ITERATING;

  /* Set default solver type (MFNSetFromOptions was not called) */
  if (!((PetscObject)mfn)->type_name) {
    ierr = MFNSetType(mfn,MFNKRYLOV);CHKERRQ(ierr);
  }
  if (!mfn->ip) { ierr = MFNGetIP(mfn,&mfn->ip);CHKERRQ(ierr); }
  if (!((PetscObject)mfn->ip)->type_name) {
    ierr = IPSetDefaultType_Private(mfn->ip);CHKERRQ(ierr);
  }
  ierr = IPSetMatrix(mfn->ip,NULL);CHKERRQ(ierr);
  if (!mfn->ds) { ierr = MFNGetDS(mfn,&mfn->ds);CHKERRQ(ierr); }
  ierr = DSReset(mfn->ds);CHKERRQ(ierr);
  if (!((PetscObject)mfn->rand)->type_name) {
    ierr = PetscRandomSetFromOptions(mfn->rand);CHKERRQ(ierr);
  }
  
  /* Set problem dimensions */
  if (!mfn->A) SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_ARG_WRONGSTATE,"MFNSetOperator must be called first"); 
  ierr = MatGetSize(mfn->A,&mfn->n,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mfn->A,&mfn->nloc,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&mfn->t);CHKERRQ(ierr);
  ierr = SlepcMatGetVecsTemplate(mfn->A,&mfn->t,NULL);CHKERRQ(ierr);

  /* Set default function */
  if (!mfn->function) {
    ierr = MFNSetFunction(mfn,SLEPC_FUNCTION_EXP);CHKERRQ(ierr);
  }
  
  if (mfn->ncv > mfn->n) mfn->ncv = mfn->n;

  /* call specific solver setup */
  ierr = (*mfn->ops->setup)(mfn);CHKERRQ(ierr);

  /* set tolerance if not yet set */
  if (mfn->tol==PETSC_DEFAULT) mfn->tol = SLEPC_DEFAULT_TOL;

  ierr = PetscLogEventEnd(MFN_SetUp,mfn,0,0,0);CHKERRQ(ierr);
  mfn->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNSetOperator"
/*@
   MFNSetOperator - Sets the matrix for which the matrix function is to be computed.

   Collective on MFN and Mat

   Input Parameters:
+  mfn - the matrix function context
-  A   - the problem matrix

   Notes: 
   It must be called after MFNSetUp(). If it is called again after MFNSetUp() then
   the MFN object is reset.

   Level: beginner

.seealso: MFNSolve(), MFNSetUp(), MFNReset()
@*/
PetscErrorCode MFNSetOperator(MFN mfn,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(mfn,1,A,2);

  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_ARG_WRONG,"A is a non-square matrix");
  if (mfn->setupcalled) { ierr = MFNReset(mfn);CHKERRQ(ierr); }
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = MatDestroy(&mfn->A);CHKERRQ(ierr);
  mfn->A = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetOperator"
/*@
   MFNGetOperator - Gets the matrix associated with the MFN object.

   Collective on MFN and Mat

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

