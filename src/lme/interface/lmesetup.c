/*
      LME routines related to problem setup.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/lmeimpl.h>       /*I "slepclme.h" I*/

#undef __FUNCT__
#define __FUNCT__ "LMESetUp"
/*@
   LMESetUp - Sets up all the internal data structures necessary for the
   execution of the linear matrix equation solver.

   Collective on LME

   Input Parameter:
.  lme   - linear matrix equation solver context

   Notes:
   This function need not be called explicitly in most cases, since LMESolve()
   calls it. It can be useful when one wants to measure the set-up time
   separately from the solve time.

   Level: developer

.seealso: LMECreate(), LMESolve(), LMEDestroy()
@*/
PetscErrorCode LMESetUp(LME lme)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);

  /* reset the convergence flag from the previous solves */
  lme->reason = LME_CONVERGED_ITERATING;

  if (lme->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(LME_SetUp,lme,0,0,0);CHKERRQ(ierr);

  /* Set default solver type (LMESetFromOptions was not called) */
  if (!((PetscObject)lme)->type_name) {
    ierr = LMESetType(lme,LMEKRYLOV);CHKERRQ(ierr);
  }

  /* Check problem dimensions */
  if (!lme->A) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_WRONGSTATE,"LMESetCoefficients must be called first");
  ierr = MatGetSize(lme->A,&N,NULL);CHKERRQ(ierr);
  if (lme->ncv > N) lme->ncv = N;

  /* call specific solver setup */
  ierr = (*lme->ops->setup)(lme);CHKERRQ(ierr);

  /* set tolerance if not yet set */
  if (lme->tol==PETSC_DEFAULT) lme->tol = SLEPC_DEFAULT_TOL;

  ierr = PetscLogEventEnd(LME_SetUp,lme,0,0,0);CHKERRQ(ierr);
  lme->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMESetCoefficients"
/*@
   LMESetCoefficients - Sets the coefficient matrices that define the linear matrix
   equation to be solved.

   Collective on LME and Mat

   Input Parameters:
+  lme - the matrix function context
.  A   - first coefficient matrix
.  B   - second coefficient matrix
.  D   - third coefficient matrix
-  E   - fourth coefficient matrix

   Notes:
   The matrix equation takes the general form A*X*E+D*X*B=C, where matrix C is not
   provided here but with LMESetRHS(). Not all four matrices must be passed, some
   can be NULL instead, see LMESetProblemType() for details.

   It must be called before LMESetUp(). If it is called again after LMESetUp() then
   the LME object is reset.

   Level: beginner

.seealso: LMESolve(), LMESetUp(), LMESetRHS(), LMESetProblemType()
@*/
PetscErrorCode LMESetCoefficients(LME lme,Mat A,Mat B,Mat D,Mat E)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(lme,1,A,2);
  if (B) {
    PetscValidHeaderSpecific(B,MAT_CLASSID,3);
    PetscCheckSameComm(lme,1,B,3);
  }
  if (D) {
    PetscValidHeaderSpecific(D,MAT_CLASSID,4);
    PetscCheckSameComm(lme,1,D,4);
  }
  if (E) {
    PetscValidHeaderSpecific(E,MAT_CLASSID,5);
    PetscCheckSameComm(lme,1,E,5);
  }

  if (lme->setupcalled) { ierr = LMEReset(lme);CHKERRQ(ierr); }

  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_WRONG,"A is a non-square matrix");
  if (!lme->setupcalled && lme->A) { ierr = MatDestroy(&lme->A);CHKERRQ(ierr); }
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  lme->A = A;
  if (B) {
    ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr);
    if (m!=n) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_WRONG,"B is a non-square matrix");
    if (!lme->setupcalled && lme->B) { ierr = MatDestroy(&lme->B);CHKERRQ(ierr); }
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    lme->B = B;
  }
  if (D) {
    ierr = MatGetSize(D,&m,&n);CHKERRQ(ierr);
    if (m!=n) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_WRONG,"D is a non-square matrix");
    if (!lme->setupcalled && lme->D) { ierr = MatDestroy(&lme->D);CHKERRQ(ierr); }
    ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr);
    lme->D = D;
  }
  if (E) {
    ierr = MatGetSize(E,&m,&n);CHKERRQ(ierr);
    if (m!=n) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_WRONG,"E is a non-square matrix");
    if (!lme->setupcalled && lme->E) { ierr = MatDestroy(&lme->E);CHKERRQ(ierr); }
    ierr = PetscObjectReference((PetscObject)E);CHKERRQ(ierr);
    lme->E = E;
  }

  lme->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMEGetCoefficients"
/*@
   LMEGetCoefficients - Gets the coefficient matrices of the matrix equation.

   Collective on LME and Mat

   Input Parameter:
.  lme - the LME context

   Output Parameters:
+  A   - first coefficient matrix
.  B   - second coefficient matrix
.  D   - third coefficient matrix
-  E   - fourth coefficient matrix

   Level: intermediate

.seealso: LMESolve(), LMESetCoefficients()
@*/
PetscErrorCode LMEGetCoefficients(LME lme,Mat *A,Mat *B,Mat *D,Mat *E)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  if (A) *A = lme->A;
  if (B) *B = lme->B;
  if (D) *D = lme->D;
  if (E) *E = lme->E;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMEAllocateSolution"
/*@
   LMEAllocateSolution - Allocate memory storage for common variables such
   as the basis vectors.

   Collective on LME

   Input Parameters:
.  lme   - linear matrix equation solver context

   Developers Note:
   This is PETSC_EXTERN because it may be required by user plugin LME
   implementations.

   Level: developer
@*/
PetscErrorCode LMEAllocateSolution(LME lme)
{
  PetscErrorCode ierr;
  PetscInt       oldsize,requested;
  Vec            t;

  PetscFunctionBegin;
  requested = lme->ncv;

  /* oldsize is zero if this is the first time setup is called */
  ierr = BVGetSizes(lme->V,NULL,NULL,&oldsize);CHKERRQ(ierr);

  /* allocate basis vectors */
  if (!lme->V) { ierr = LMEGetBV(lme,&lme->V);CHKERRQ(ierr); }
  if (!oldsize) {
    if (!((PetscObject)(lme->V))->type_name) {
      ierr = BVSetType(lme->V,BVSVEC);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(lme->A,&t,NULL);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(lme->V,t,requested);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
  } else {
    ierr = BVResize(lme->V,requested,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

