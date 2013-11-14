/*
      PEP routines related to problem setup.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/pepimpl.h>       /*I "slepcpep.h" I*/
#include <slepc-private/ipimpl.h>
#include <slepc-private/stimpl.h>

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp"
/*@
   PEPSetUp - Sets up all the internal data structures necessary for the
   execution of the PEP solver.

   Collective on PEP

   Input Parameter:
.  pep   - solver context

   Notes:
   This function need not be called explicitly in most cases, since PEPSolve()
   calls it. It can be useful when one wants to measure the set-up time
   separately from the solve time.

   Level: advanced

.seealso: PEPCreate(), PEPSolve(), PEPDestroy()
@*/
PetscErrorCode PEPSetUp(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      islinear,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (pep->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PEP_SetUp,pep,0,0,0);CHKERRQ(ierr);

  /* reset the convergence flag from the previous solves */
  pep->reason = PEP_CONVERGED_ITERATING;

  /* Set default solver type (PEPSetFromOptions was not called) */
  if (!((PetscObject)pep)->type_name) {
    ierr = PEPSetType(pep,PEPLINEAR);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&islinear);CHKERRQ(ierr);
  if (!islinear) {
    if (!pep->st) { ierr = PEPGetST(pep,&pep->st);CHKERRQ(ierr); }
    if (!((PetscObject)pep->st)->type_name) {
      ierr = STSetType(pep->st,STSHIFT);CHKERRQ(ierr);
    }
  }
  if (!pep->ip) { ierr = PEPGetIP(pep,&pep->ip);CHKERRQ(ierr); }
  if (!((PetscObject)pep->ip)->type_name) {
    ierr = IPSetType_Default(pep->ip);CHKERRQ(ierr);
  }
  if (!pep->ds) { ierr = PEPGetDS(pep,&pep->ds);CHKERRQ(ierr); }
  ierr = DSReset(pep->ds);CHKERRQ(ierr);
  if (!((PetscObject)pep->rand)->type_name) {
    ierr = PetscRandomSetFromOptions(pep->rand);CHKERRQ(ierr);
  }

  /* Check matrices, transfer them to ST */
  /*if (!pep->M || !pep->C || !pep->K) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONGSTATE,"PEPSetOperators must be called first");
  if (!islinear) {
    mat[0] = pep->K;
    mat[1] = pep->C;
    mat[2] = pep->M;
    ierr = STSetOperators(pep->st,3,mat);CHKERRQ(ierr);
  }*/

  /* Set problem dimensions */
  ierr = MatGetSize(pep->A[0],&pep->n,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pep->A[0],&pep->nloc,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&pep->t);CHKERRQ(ierr);
  ierr = SlepcMatGetVecsTemplate(pep->A[0],&pep->t,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->t);CHKERRQ(ierr);

  /* Set default problem type */
  if (!pep->problem_type) {
    ierr = PEPSetProblemType(pep,PEP_GENERAL);CHKERRQ(ierr);
  }

 /* Call specific solver setup */
  ierr = (*pep->ops->setup)(pep);CHKERRQ(ierr);

  /* set tolerance if not yet set */
  if (pep->tol==PETSC_DEFAULT) pep->tol = SLEPC_DEFAULT_TOL;

  /* set eigenvalue comparison */
  switch (pep->which) {
    case PEP_LARGEST_MAGNITUDE:
      pep->comparison    = SlepcCompareLargestMagnitude;
      pep->comparisonctx = NULL;
      break;
    case PEP_SMALLEST_MAGNITUDE:
      pep->comparison    = SlepcCompareSmallestMagnitude;
      pep->comparisonctx = NULL;
      break;
    case PEP_LARGEST_REAL:
      pep->comparison    = SlepcCompareLargestReal;
      pep->comparisonctx = NULL;
      break;
    case PEP_SMALLEST_REAL:
      pep->comparison    = SlepcCompareSmallestReal;
      pep->comparisonctx = NULL;
      break;
    case PEP_LARGEST_IMAGINARY:
      pep->comparison    = SlepcCompareLargestImaginary;
      pep->comparisonctx = NULL;
      break;
    case PEP_SMALLEST_IMAGINARY:
      pep->comparison    = SlepcCompareSmallestImaginary;
      pep->comparisonctx = NULL;
      break;
    case PEP_TARGET_MAGNITUDE:
      pep->comparison    = SlepcCompareTargetMagnitude;
      pep->comparisonctx = &pep->target;
      break;
    case PEP_TARGET_REAL:
      pep->comparison    = SlepcCompareTargetReal;
      pep->comparisonctx = &pep->target;
      break;
    case PEP_TARGET_IMAGINARY:
      pep->comparison    = SlepcCompareTargetImaginary;
      pep->comparisonctx = &pep->target;
      break;
  }

  if (pep->ncv > 2*pep->n) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"ncv must be twice the problem size at most");
  if (pep->nev > pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"nev bigger than ncv");

   /* Setup ST */
  if (!islinear) {
    ierr = PetscObjectTypeCompareAny((PetscObject)pep->st,&flg,STSHIFT,STSINVERT,"");CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Only STSHIFT and STSINVERT spectral transformations can be used in PEP");
    pep->st->userscale = pep->sfactor_set;
    if (pep->sfactor_set) {
      pep->st->gamma = pep->sfactor;
      pep->st->delta = 1.0;
    }
    ierr = STSetUp(pep->st);CHKERRQ(ierr);
    if (!pep->sfactor_set) pep->sfactor = pep->st->gamma;
  }

  /* process initial vectors */
  if (pep->nini<0) {
    pep->nini = -pep->nini;
    if (pep->nini>pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),1,"The number of initial vectors is larger than ncv");
    ierr = IPOrthonormalizeBasis_Private(pep->ip,&pep->nini,&pep->IS,pep->V);CHKERRQ(ierr);
  }
  if (pep->ninil<0) {
    if (!pep->leftvecs) { ierr = PetscInfo(pep,"Ignoring initial left vectors\n");CHKERRQ(ierr); }
    else {
      pep->ninil = -pep->ninil;
      if (pep->ninil>pep->ncv) SETERRQ(PetscObjectComm((PetscObject)pep),1,"The number of initial left vectors is larger than ncv");
      ierr = IPOrthonormalizeBasis_Private(pep->ip,&pep->ninil,&pep->ISL,pep->W);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(PEP_SetUp,pep,0,0,0);CHKERRQ(ierr);
  pep->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetOperators"
/*@
   PEPSetOperators - Sets the matrices associated with the polynomial eigenvalue problem.

   Collective on PEP and Mat

   Input Parameters:
+  pep - the eigenproblem solver context
.  M   - the first coefficient matrix
.  C   - the second coefficient matrix
-  K   - the third coefficient matrix

   Notes:
   The polynomial eigenproblem is defined as (l^2*M + l*C + K)*x = 0, where l is
   the eigenvalue and x is the eigenvector.

   Level: beginner

.seealso: PEPSolve(), PEPGetOperators()
@*/
PetscErrorCode PEPSetOperators(PEP pep,PetscInt nmat,Mat A[])
{
/*  PetscErrorCode ierr;
  PetscInt       m,n,m0;*/

  PetscFunctionBegin;
#if 0
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(M,MAT_CLASSID,2);
  PetscValidHeaderSpecific(C,MAT_CLASSID,3);
  PetscValidHeaderSpecific(K,MAT_CLASSID,4);
  PetscCheckSameComm(pep,1,M,2);
  PetscCheckSameComm(pep,1,C,3);
  PetscCheckSameComm(pep,1,K,4);

  /* Check for square matrices */
  ierr = MatGetSize(M,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"M is a non-square matrix");
  m0=m;
  ierr = MatGetSize(C,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"C is a non-square matrix");
  if (m!=m0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_INCOMP,"Dimensions of M and C do not match");
  ierr = MatGetSize(K,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"K is a non-square matrix");
  if (m!=m0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_INCOMP,"Dimensions of M and K do not match");

  /* Store a copy of the matrices */
  if (pep->setupcalled) { ierr = PEPReset(pep);CHKERRQ(ierr); }
  ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr);
  ierr = MatDestroy(&pep->M);CHKERRQ(ierr);
  pep->M = M;
  ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
  ierr = MatDestroy(&pep->C);CHKERRQ(ierr);
  pep->C = C;
  ierr = PetscObjectReference((PetscObject)K);CHKERRQ(ierr);
  ierr = MatDestroy(&pep->K);CHKERRQ(ierr);
  pep->K = K;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetOperators"
/*@
   PEPGetOperators - Gets the matrices associated with the polynomial eigensystem.

   Collective on PEP and Mat

   Input Parameter:
.  pep - the PEP context

   Output Parameters:
+  M   - the first coefficient matrix
.  C   - the second coefficient matrix
-  K   - the third coefficient matrix

   Level: intermediate

.seealso: PEPSolve(), PEPSetOperators()
@*/
PetscErrorCode PEPGetOperators(PEP pep,PetscInt k,Mat *A)
{
  PetscFunctionBegin;
#if 0
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (M) { PetscValidPointer(M,2); *M = pep->M; }
  if (C) { PetscValidPointer(C,3); *C = pep->C; }
  if (K) { PetscValidPointer(K,4); *K = pep->K; }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetInitialSpace"
/*@
   PEPSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on PEP and Vec

   Input Parameter:
+  pep   - the polynomial eigensolver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one PEPSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: PEPSetInitialSpaceLeft()
@*/
PetscErrorCode PEPSetInitialSpace(PEP pep,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative");
  ierr = SlepcBasisReference_Private(n,is,&pep->nini,&pep->IS);CHKERRQ(ierr);
  if (n>0) pep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetInitialSpaceLeft"
/*@
   PEPSetInitialSpaceLeft - Specify a basis of vectors that constitute the initial
   left space, that is, the subspace from which the solver starts to iterate for
   building the left subspace (in methods that work with two subspaces).

   Collective on PEP and Vec

   Input Parameter:
+  pep   - the polynomial eigensolver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial left space

   Notes:
   Some solvers start to iterate on a single vector (initial left vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one PEPSolve() call to the other, so the
   initial left space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted left eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: PEPSetInitialSpace()
@*/
PetscErrorCode PEPSetInitialSpaceLeft(PEP pep,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative");
  ierr = SlepcBasisReference_Private(n,is,&pep->ninil,&pep->ISL);CHKERRQ(ierr);
  if (n>0) pep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPAllocateSolution"
/*
  PEPAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors. The argument extra is used for methods
  that require a working basis slightly larger than ncv.
*/
PetscErrorCode PEPAllocateSolution(PEP pep,PetscInt extra)
{
  PetscErrorCode ierr;
  PetscInt       newc,cnt,requested;

  PetscFunctionBegin;
  requested = pep->ncv + extra;
  if (pep->allocated_ncv != requested) {
    newc = PetscMax(0,requested-pep->allocated_ncv);
    ierr = PEPFreeSolution(pep);CHKERRQ(ierr);
    cnt = 0;
    ierr = PetscMalloc(requested*sizeof(PetscScalar),&pep->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(requested*sizeof(PetscScalar),&pep->eigi);CHKERRQ(ierr);
    cnt += 2*newc*sizeof(PetscScalar);
    ierr = PetscMalloc(requested*sizeof(PetscReal),&pep->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(requested*sizeof(PetscInt),&pep->perm);CHKERRQ(ierr);
    cnt += 2*newc*sizeof(PetscReal);
    ierr = PetscLogObjectMemory((PetscObject)pep,cnt);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(pep->t,requested,&pep->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(pep,requested,pep->V);CHKERRQ(ierr);
    pep->allocated_ncv = requested;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPFreeSolution"
/*
  PEPFreeSolution - Free memory storage. This routine is related to
  PEPAllocateSolution().
*/
PetscErrorCode PEPFreeSolution(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pep->allocated_ncv > 0) {
    ierr = PetscFree(pep->eigr);CHKERRQ(ierr);
    ierr = PetscFree(pep->eigi);CHKERRQ(ierr);
    ierr = PetscFree(pep->errest);CHKERRQ(ierr);
    ierr = PetscFree(pep->perm);CHKERRQ(ierr);
    ierr = VecDestroyVecs(pep->allocated_ncv,&pep->V);CHKERRQ(ierr);
    pep->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}

