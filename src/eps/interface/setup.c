/*
      EPS routines related to problem setup.

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

#include <slepc-private/epsimpl.h>       /*I "slepceps.h" I*/
#include <slepc-private/ipimpl.h>

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp"
/*@
   EPSSetUp - Sets up all the internal data structures necessary for the
   execution of the eigensolver. Then calls STSetUp() for any set-up
   operations associated to the ST object.

   Collective on EPS

   Input Parameter:
.  eps   - eigenproblem solver context

   Notes:
   This function need not be called explicitly in most cases, since EPSSolve()
   calls it. It can be useful when one wants to measure the set-up time
   separately from the solve time.

   Level: advanced

.seealso: EPSCreate(), EPSSolve(), EPSDestroy(), STSetUp(), EPSSetInitialSpace()
@*/
PetscErrorCode EPSSetUp(EPS eps)
{
  PetscErrorCode ierr;
  Mat            A,B;
  PetscInt       i,k,nmat;
  PetscBool      flg,lindep;
  Vec            *newDS;
  PetscReal      norm;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    sigma;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (eps->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);

  /* reset the convergence flag from the previous solves */
  eps->reason = EPS_CONVERGED_ITERATING;

  /* Set default solver type (EPSSetFromOptions was not called) */
  if (!((PetscObject)eps)->type_name) {
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  }
  if (!eps->st) { ierr = EPSGetST(eps,&eps->st);CHKERRQ(ierr); }
  if (!((PetscObject)eps->st)->type_name) {
    ierr = PetscObjectTypeCompareAny((PetscObject)eps,&flg,EPSGD,EPSJD,EPSRQCG,EPSBLOPEX,EPSPRIMME,"");CHKERRQ(ierr);
    ierr = STSetType(eps->st,flg?STPRECOND:STSHIFT);CHKERRQ(ierr);
  }
  if (!eps->ip) { ierr = EPSGetIP(eps,&eps->ip);CHKERRQ(ierr); }
  if (!((PetscObject)eps->ip)->type_name) {
    ierr = IPSetType_Default(eps->ip);CHKERRQ(ierr);
  }
  if (!eps->ds) { ierr = EPSGetDS(eps,&eps->ds);CHKERRQ(ierr); }
  ierr = DSReset(eps->ds);CHKERRQ(ierr);
  if (!((PetscObject)eps->rand)->type_name) {
    ierr = PetscRandomSetFromOptions(eps->rand);CHKERRQ(ierr);
  }

  /* Set problem dimensions */
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  if (!nmat) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"EPSSetOperators must be called first");
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&eps->n,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&eps->nloc,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&eps->t);CHKERRQ(ierr);
  ierr = SlepcMatGetVecsTemplate(A,&eps->t,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->t);CHKERRQ(ierr);

  /* Set default problem type */
  if (!eps->problem_type) {
    if (nmat==1) {
      ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
    } else {
      ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
    }
  } else if (nmat==1 && eps->isgeneralized) {
    ierr = PetscInfo(eps,"Eigenproblem set as generalized but no matrix B was provided; reverting to a standard eigenproblem\n");CHKERRQ(ierr);
    eps->isgeneralized = PETSC_FALSE;
    eps->problem_type = eps->ishermitian? EPS_HEP: EPS_NHEP;
  } else if (nmat>1 && !eps->isgeneralized) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_INCOMP,"Inconsistent EPS state");
#if defined(PETSC_USE_COMPLEX)
  ierr = STGetShift(eps->st,&sigma);CHKERRQ(ierr);
  if (eps->ishermitian && PetscImaginaryPart(sigma) != 0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Hermitian problems are not compatible with complex shifts");
#endif
  if (eps->ishermitian && eps->leftvecs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Requesting left eigenvectors not allowed in Hermitian problems");

  if (eps->ispositive || (eps->isgeneralized && eps->ishermitian)) {
    ierr = STGetBilinearForm(eps->st,&B);CHKERRQ(ierr);
    ierr = IPSetMatrix(eps->ip,B);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    if (!eps->ispositive) {
      ierr = IPSetType(eps->ip,IPINDEFINITE);CHKERRQ(ierr);
    }
  } else {
    ierr = IPSetMatrix(eps->ip,NULL);CHKERRQ(ierr);
  }

  if (eps->nev > eps->n) eps->nev = eps->n;
  if (eps->ncv > eps->n) eps->ncv = eps->n;

  /* initialization of matrix norms */
  if (eps->nrma == PETSC_DETERMINE) {
    ierr = MatHasOperation(A,MATOP_NORM,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatNorm(A,NORM_INFINITY,&eps->nrma);CHKERRQ(ierr);
    } else eps->nrma = 1.0;
  }
  if (eps->nrmb == PETSC_DETERMINE) {
    if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
    ierr = MatHasOperation(B,MATOP_NORM,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatNorm(B,NORM_INFINITY,&eps->nrmb);CHKERRQ(ierr);
    } else eps->nrmb = 1.0;
  }

  if (!eps->balance) eps->balance = EPS_BALANCE_NONE;

  /* call specific solver setup */
  ierr = (*eps->ops->setup)(eps);CHKERRQ(ierr);

  /* check extraction */
  ierr = PetscObjectTypeCompareAny((PetscObject)eps->st,&flg,STPRECOND,STSHIFT,"");CHKERRQ(ierr);
  if (!flg && eps->extraction && eps->extraction!=EPS_RITZ) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Cannot use a spectral transformation combined with harmonic extraction");

  /* set tolerance if not yet set */
  if (eps->tol==PETSC_DEFAULT) eps->tol = SLEPC_DEFAULT_TOL;

  /* set eigenvalue comparison */
  switch (eps->which) {
    case EPS_LARGEST_MAGNITUDE:
      eps->comparison    = SlepcCompareLargestMagnitude;
      eps->comparisonctx = NULL;
      break;
    case EPS_SMALLEST_MAGNITUDE:
      eps->comparison    = SlepcCompareSmallestMagnitude;
      eps->comparisonctx = NULL;
      break;
    case EPS_LARGEST_REAL:
      eps->comparison    = SlepcCompareLargestReal;
      eps->comparisonctx = NULL;
      break;
    case EPS_SMALLEST_REAL:
      eps->comparison    = SlepcCompareSmallestReal;
      eps->comparisonctx = NULL;
      break;
    case EPS_LARGEST_IMAGINARY:
      eps->comparison    = SlepcCompareLargestImaginary;
      eps->comparisonctx = NULL;
      break;
    case EPS_SMALLEST_IMAGINARY:
      eps->comparison    = SlepcCompareSmallestImaginary;
      eps->comparisonctx = NULL;
      break;
    case EPS_TARGET_MAGNITUDE:
      eps->comparison    = SlepcCompareTargetMagnitude;
      eps->comparisonctx = &eps->target;
      break;
    case EPS_TARGET_REAL:
      eps->comparison    = SlepcCompareTargetReal;
      eps->comparisonctx = &eps->target;
      break;
    case EPS_TARGET_IMAGINARY:
      eps->comparison    = SlepcCompareTargetImaginary;
      eps->comparisonctx = &eps->target;
      break;
    case EPS_ALL:
      eps->comparison    = SlepcCompareSmallestReal;
      eps->comparisonctx = NULL;
      break;
    case EPS_WHICH_USER:
      break;
  }

  /* Build balancing matrix if required */
  if (!eps->ishermitian && (eps->balance==EPS_BALANCE_ONESIDE || eps->balance==EPS_BALANCE_TWOSIDE)) {
    if (!eps->D) {
      ierr = VecDuplicate(eps->V[0],&eps->D);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->D);CHKERRQ(ierr);
    } else {
      ierr = VecSet(eps->D,1.0);CHKERRQ(ierr);
    }
    ierr = EPSBuildBalance_Krylov(eps);CHKERRQ(ierr);
    ierr = STSetBalanceMatrix(eps->st,eps->D);CHKERRQ(ierr);
  }

  /* Setup ST */
  ierr = STSetUp(eps->st);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STCAYLEY,&flg);CHKERRQ(ierr);
  if (flg && eps->problem_type == EPS_PGNHEP) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Cayley spectral transformation is not compatible with PGNHEP");

  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STFOLD,&flg);CHKERRQ(ierr);
  if (flg && !eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Fold spectral transformation requires a Hermitian problem");

  if (eps->nds>0) {
    if (!eps->ds_ortho) {
      /* allocate memory and copy deflation basis vectors into defl */
      ierr = VecDuplicateVecs(eps->t,eps->nds,&newDS);CHKERRQ(ierr);
      for (i=0;i<eps->nds;i++) {
        ierr = VecCopy(eps->defl[i],newDS[i]);CHKERRQ(ierr);
        ierr = VecDestroy(&eps->defl[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(eps->defl);CHKERRQ(ierr);
      eps->defl = newDS;
      ierr = PetscLogObjectParents(eps,eps->nds,eps->defl);CHKERRQ(ierr);
      /* orthonormalize vectors in defl */
      k = 0;
      for (i=0;i<eps->nds;i++) {
        ierr = IPOrthogonalize(eps->ip,0,NULL,k,NULL,eps->defl,eps->defl[k],NULL,&norm,&lindep);CHKERRQ(ierr);
        if (norm==0.0 || lindep) {
          ierr = PetscInfo(eps,"Linearly dependent deflation vector found, removing...\n");CHKERRQ(ierr);
        } else {
          ierr = VecScale(eps->defl[k],1.0/norm);CHKERRQ(ierr);
          k++;
        }
      }
      for (i=k;i<eps->nds;i++) { ierr = VecDestroy(&eps->defl[i]);CHKERRQ(ierr); }
      eps->nds = k;
      eps->ds_ortho = PETSC_TRUE;
    }
  }
  ierr = STCheckNullSpace(eps->st,eps->nds,eps->defl);CHKERRQ(ierr);

  /* process initial vectors */
  if (eps->nini<0) {
    eps->nini = -eps->nini;
    if (eps->nini>eps->ncv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The number of initial vectors is larger than ncv");
    ierr = IPOrthonormalizeBasis_Private(eps->ip,&eps->nini,&eps->IS,eps->V);CHKERRQ(ierr);
  }
  if (eps->ninil<0) {
    if (!eps->leftvecs) {
      ierr = PetscInfo(eps,"Ignoring initial left vectors\n");CHKERRQ(ierr);
    } else {
      eps->ninil = -eps->ninil;
      if (eps->ninil>eps->ncv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The number of initial left vectors is larger than ncv");
      ierr = IPOrthonormalizeBasis_Private(eps->ip,&eps->ninil,&eps->ISL,eps->W);CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);
  eps->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetOperators"
/*@
   EPSSetOperators - Sets the matrices associated with the eigenvalue problem.

   Collective on EPS and Mat

   Input Parameters:
+  eps - the eigenproblem solver context
.  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes:
   To specify a standard eigenproblem, use NULL for parameter B.

   It must be called after EPSSetUp(). If it is called again after EPSSetUp() then
   the EPS object is reset.

   Level: beginner

.seealso: EPSSolve(), EPSSetUp(), EPSReset(), EPSGetST(), STGetOperators()
@*/
PetscErrorCode EPSSetOperators(EPS eps,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       m,n,m0,nmat;
  Mat            mat[2];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscCheckSameComm(eps,1,A,2);
  if (B) PetscCheckSameComm(eps,1,B,3);

  /* Check for square matrices */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"A is a non-square matrix");
  if (B) {
    ierr = MatGetSize(B,&m0,&n);CHKERRQ(ierr);
    if (m0!=n) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"B is a non-square matrix");
    if (m!=m0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_INCOMP,"Dimensions of A and B do not match");
  }

  if (eps->setupcalled) { ierr = EPSReset(eps);CHKERRQ(ierr); }
  if (!eps->st) { ierr = EPSGetST(eps,&eps->st);CHKERRQ(ierr); }
  mat[0] = A;
  if (B) {
    mat[1] = B;
    nmat = 2;
  } else nmat = 1;
  ierr = STSetOperators(eps->st,nmat,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetOperators"
/*@C
   EPSGetOperators - Gets the matrices associated with the eigensystem.

   Collective on EPS and Mat

   Input Parameter:
.  eps - the EPS context

   Output Parameters:
+  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Level: intermediate

.seealso: EPSSolve(), EPSGetST(), STGetOperators(), STSetOperators()
@*/
PetscErrorCode EPSGetOperators(EPS eps,Mat *A,Mat *B)
{
  PetscErrorCode ierr;
  ST             st;
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  if (A) { ierr = STGetOperators(st,0,A);CHKERRQ(ierr); }
  if (B) {
    ierr = STGetNumMatrices(st,&k);CHKERRQ(ierr);
    if (k==1) B = NULL;
    else {
      ierr = STGetOperators(st,1,B);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetDeflationSpace"
/*@
   EPSSetDeflationSpace - Specify a basis of vectors that constitute
   the deflation space.

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  v     - set of basis vectors of the deflation space

   Notes:
   When a deflation space is given, the eigensolver seeks the eigensolution
   in the restriction of the problem to the orthogonal complement of this
   space. This can be used for instance in the case that an invariant
   subspace is known beforehand (such as the nullspace of the matrix).

   Basis vectors set by a previous call to EPSSetDeflationSpace() are
   replaced.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   These vectors persist from one EPSSolve() call to the other, use
   EPSRemoveDeflationSpace() to eliminate them.

   Level: intermediate

.seealso: EPSRemoveDeflationSpace()
@*/
PetscErrorCode EPSSetDeflationSpace(EPS eps,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Argument n out of range");

  /* free previous vectors */
  ierr = EPSRemoveDeflationSpace(eps);CHKERRQ(ierr);

  /* get references of passed vectors */
  if (n>0) {
    ierr = PetscMalloc(n*sizeof(Vec),&eps->defl);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,n*sizeof(Vec));CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscObjectReference((PetscObject)v[i]);CHKERRQ(ierr);
      eps->defl[i] = v[i];
    }
    eps->setupcalled = 0;
    eps->ds_ortho = PETSC_FALSE;
  }

  eps->nds = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSRemoveDeflationSpace"
/*@
   EPSRemoveDeflationSpace - Removes the deflation space.

   Collective on EPS

   Input Parameter:
.  eps   - the eigenproblem solver context

   Level: intermediate

.seealso: EPSSetDeflationSpace()
@*/
PetscErrorCode EPSRemoveDeflationSpace(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = VecDestroyVecs(eps->nds,&eps->defl);CHKERRQ(ierr);
  eps->nds = 0;
  eps->setupcalled = 0;
  eps->ds_ortho = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetInitialSpace"
/*@
   EPSSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   In contrast to EPSSetDeflationSpace(), these vectors do not persist from one
   EPSSolve() call to the other, so the initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: EPSSetInitialSpaceLeft(), EPSSetDeflationSpace()
@*/
PetscErrorCode EPSSetInitialSpace(EPS eps,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative");
  ierr = SlepcBasisReference_Private(n,is,&eps->nini,&eps->IS);CHKERRQ(ierr);
  if (n>0) eps->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetInitialSpaceLeft"
/*@
   EPSSetInitialSpaceLeft - Specify a basis of vectors that constitute the initial
   left space, that is, the subspace from which the solver starts to iterate for
   building the left subspace (in methods that work with two subspaces).

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial left space

   Notes:
   Some solvers start to iterate on a single vector (initial left vector). In that case,
   the other vectors are ignored.

   In contrast to EPSSetDeflationSpace(), these vectors do not persist from one
   EPSSolve() call to the other, so the initial left space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted left eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: EPSSetInitialSpace(), EPSSetDeflationSpace()
@*/
PetscErrorCode EPSSetInitialSpaceLeft(EPS eps,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative");
  ierr = SlepcBasisReference_Private(n,is,&eps->ninil,&eps->ISL);CHKERRQ(ierr);
  if (n>0) eps->setupcalled = 0;
  PetscFunctionReturn(0);
}

