/*
     SVD routines for setting up the solver.

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

#include <slepc/private/svdimpl.h>      /*I "slepcsvd.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SVDSetOperator"
/*@
   SVDSetOperator - Set the matrix associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd - the singular value solver context
-  A  - the matrix associated with the singular value problem

   Level: beginner

.seealso: SVDSolve(), SVDGetOperator()
@*/
PetscErrorCode SVDSetOperator(SVD svd,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscCheckSameComm(svd,1,mat,2);
  if (svd->state) { ierr = SVDReset(svd);CHKERRQ(ierr); }
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&svd->OP);CHKERRQ(ierr);
  svd->OP = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDGetOperator"
/*@
   SVDGetOperator - Get the matrix associated with the singular value problem.

   Not collective, though parallel Mats are returned if the SVD is parallel

   Input Parameter:
.  svd - the singular value solver context

   Output Parameters:
.  A    - the matrix associated with the singular value problem

   Level: advanced

.seealso: SVDSolve(), SVDSetOperator()
@*/
PetscErrorCode SVDGetOperator(SVD svd,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(A,2);
  *A = svd->OP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetUp"
/*@
   SVDSetUp - Sets up all the internal data structures necessary for the
   execution of the singular value solver.

   Collective on SVD

   Input Parameter:
.  svd   - singular value solver context

   Notes:
   This function need not be called explicitly in most cases, since SVDSolve()
   calls it. It can be useful when one wants to measure the set-up time
   separately from the solve time.

   Level: developer

.seealso: SVDCreate(), SVDSolve(), SVDDestroy()
@*/
PetscErrorCode SVDSetUp(SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      expltrans,flg;
  PetscInt       M,N,k;
  SlepcSC        sc;
  Vec            *T;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->state) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);

  /* reset the convergence flag from the previous solves */
  svd->reason = SVD_CONVERGED_ITERATING;

  /* Set default solver type (SVDSetFromOptions was not called) */
  if (!((PetscObject)svd)->type_name) {
    ierr = SVDSetType(svd,SVDCROSS);CHKERRQ(ierr);
  }
  if (!svd->ds) { ierr = SVDGetDS(svd,&svd->ds);CHKERRQ(ierr); }
  ierr = DSReset(svd->ds);CHKERRQ(ierr);

  /* check matrix */
  if (!svd->OP) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONGSTATE,"SVDSetOperator must be called first");

  /* determine how to handle the transpose */
  expltrans = PETSC_TRUE;
  if (svd->impltrans) expltrans = PETSC_FALSE;
  else {
    ierr = MatHasOperation(svd->OP,MATOP_TRANSPOSE,&flg);CHKERRQ(ierr);
    if (!flg) expltrans = PETSC_FALSE;
    else {
      ierr = PetscObjectTypeCompare((PetscObject)svd,SVDLAPACK,&flg);CHKERRQ(ierr);
      if (flg) expltrans = PETSC_FALSE;
    }
  }

  /* build transpose matrix */
  ierr = MatDestroy(&svd->A);CHKERRQ(ierr);
  ierr = MatDestroy(&svd->AT);CHKERRQ(ierr);
  ierr = MatGetSize(svd->OP,&M,&N);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)svd->OP);CHKERRQ(ierr);
  if (expltrans) {
    if (M>=N) {
      svd->A = svd->OP;
      ierr = MatTranspose(svd->OP,MAT_INITIAL_MATRIX,&svd->AT);CHKERRQ(ierr);
      ierr = MatConjugate(svd->AT);CHKERRQ(ierr);
    } else {
      ierr = MatTranspose(svd->OP,MAT_INITIAL_MATRIX,&svd->A);CHKERRQ(ierr);
      ierr = MatConjugate(svd->A);CHKERRQ(ierr);
      svd->AT = svd->OP;
    }
  } else {
    if (M>=N) {
      svd->A = svd->OP;
      svd->AT = NULL;
    } else {
      svd->A = NULL;
      svd->AT = svd->OP;
    }
  }

  /* swap initial vectors if necessary */
  if (M<N) {
    T=svd->ISL; svd->ISL=svd->IS; svd->IS=T;
    k=svd->ninil; svd->ninil=svd->nini; svd->nini=k;
  }

  if (svd->ncv > PetscMin(M,N)) svd->ncv = PetscMin(M,N);
  if (svd->nsv > PetscMin(M,N)) svd->nsv = PetscMin(M,N);
  if (svd->ncv && svd->nsv > svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"nsv bigger than ncv");

  /* call specific solver setup */
  ierr = (*svd->ops->setup)(svd);CHKERRQ(ierr);

  /* set tolerance if not yet set */
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  /* fill sorting criterion context */
  ierr = DSGetSlepcSC(svd->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = (svd->which==SVD_LARGEST)? SlepcCompareLargestReal: SlepcCompareSmallestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;

  /* process initial vectors */
  if (svd->nini<0) {
    k = -svd->nini;
    if (k>svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),1,"The number of initial vectors is larger than ncv");
    ierr = BVInsertVecs(svd->V,0,&k,svd->IS,PETSC_TRUE);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->nini,&svd->IS);CHKERRQ(ierr);
    svd->nini = k;
  }
  if (svd->ninil<0) {
    k = 0;
    if (svd->leftbasis) {
      k = -svd->ninil;
      if (k>svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),1,"The number of left initial vectors is larger than ncv");
      ierr = BVInsertVecs(svd->U,0,&k,svd->ISL,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo(svd,"Ignoring initial left vectors\n");CHKERRQ(ierr);
    }
    ierr = SlepcBasisDestroy_Private(&svd->ninil,&svd->ISL);CHKERRQ(ierr);
    svd->ninil = k;
  }

  ierr = PetscLogEventEnd(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);
  svd->state = SVD_STATE_SETUP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetInitialSpace"
/*@
   SVDSetInitialSpace - Specify a basis of vectors that constitute the initial
   (right) space, that is, a rough approximation to the right singular subspace
   from which the solver starts to iterate.

   Collective on SVD and Vec

   Input Parameter:
+  svd   - the singular value solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one SVDSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted singular space. Then, convergence may be faster.

   Level: intermediate

.seealso: SVDSetInitialSpaceLeft()
@*/
PetscErrorCode SVDSetInitialSpace(SVD svd,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative");
  ierr = SlepcBasisReference_Private(n,is,&svd->nini,&svd->IS);CHKERRQ(ierr);
  if (n>0) svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetInitialSpaceLeft"
/*@
   SVDSetInitialSpaceLeft - Specify a basis of vectors that constitute the initial
   left space, that is, a rough approximation to the left singular subspace
   from which the solver starts to iterate.

   Collective on SVD and Vec

   Input Parameter:
+  svd   - the singular value solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one SVDSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted singular space. Then, convergence may be faster.

   Level: intermediate

.seealso: SVDSetInitialSpace()
@*/
PetscErrorCode SVDSetInitialSpaceLeft(SVD svd,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,n,2);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative");
  ierr = SlepcBasisReference_Private(n,is,&svd->ninil,&svd->ISL);CHKERRQ(ierr);
  if (n>0) svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetDimensions_Default"
/*
  SVDSetDimensions_Default - Set reasonable values for ncv, mpd if not set
  by the user. This is called at setup.
 */
PetscErrorCode SVDSetDimensions_Default(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,NULL,&N);CHKERRQ(ierr);
  if (svd->ncv) { /* ncv set */
    if (svd->ncv<svd->nsv) SETERRQ(PetscObjectComm((PetscObject)svd),1,"The value of ncv must be at least nsv");
  } else if (svd->mpd) { /* mpd set */
    svd->ncv = PetscMin(N,svd->nsv+svd->mpd);
  } else { /* neither set: defaults depend on nsv being small or large */
    if (svd->nsv<500) svd->ncv = PetscMin(N,PetscMax(2*svd->nsv,10));
    else {
      svd->mpd = 500;
      svd->ncv = PetscMin(N,svd->nsv+svd->mpd);
    }
  }
  if (!svd->mpd) svd->mpd = svd->ncv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDAllocateSolution"
/*@
   SVDAllocateSolution - Allocate memory storage for common variables such
   as the singular values and the basis vectors.

   Collective on SVD

   Input Parameters:
+  svd   - eigensolver context
-  extra - number of additional positions, used for methods that require a
           working basis slightly larger than ncv

   Developers Notes:
   This is PETSC_EXTERN because it may be required by user plugin SVD
   implementations.

   This is called at setup after setting the value of ncv and the flag leftbasis.

   Level: developer
@*/
PetscErrorCode SVDAllocateSolution(SVD svd,PetscInt extra)
{
  PetscErrorCode ierr;
  PetscInt       oldsize,requested;
  Vec            tr,tl;

  PetscFunctionBegin;
  requested = svd->ncv + extra;

  /* oldsize is zero if this is the first time setup is called */
  ierr = BVGetSizes(svd->V,NULL,NULL,&oldsize);CHKERRQ(ierr);

  /* allocate sigma */
  if (requested != oldsize || !svd->sigma) {
    if (oldsize) {
      ierr = PetscFree3(svd->sigma,svd->perm,svd->errest);CHKERRQ(ierr);
    }
    ierr = PetscMalloc3(requested,&svd->sigma,requested,&svd->perm,requested,&svd->errest);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)svd,PetscMax(0,requested-oldsize)*(2*sizeof(PetscReal)+sizeof(PetscInt)));CHKERRQ(ierr);
  }
  /* allocate V */
  if (!svd->V) { ierr = SVDGetBV(svd,&svd->V,NULL);CHKERRQ(ierr); }
  if (!oldsize) {
    if (!((PetscObject)(svd->V))->type_name) {
      ierr = BVSetType(svd->V,BVSVEC);CHKERRQ(ierr);
    }
    ierr = SVDMatCreateVecs(svd,&tr,NULL);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(svd->V,tr,requested);CHKERRQ(ierr);
    ierr = VecDestroy(&tr);CHKERRQ(ierr);
  } else {
    ierr = BVResize(svd->V,requested,PETSC_FALSE);CHKERRQ(ierr);
  }
  /* allocate U */
  if (svd->leftbasis) {
    if (!svd->U) { ierr = SVDGetBV(svd,NULL,&svd->U);CHKERRQ(ierr); }
    if (!oldsize) {
      if (!((PetscObject)(svd->U))->type_name) {
        ierr = BVSetType(svd->U,BVSVEC);CHKERRQ(ierr);
      }
      ierr = SVDMatCreateVecs(svd,NULL,&tl);CHKERRQ(ierr);
      ierr = BVSetSizesFromVec(svd->U,tl,requested);CHKERRQ(ierr);
      ierr = VecDestroy(&tl);CHKERRQ(ierr);
    } else {
      ierr = BVResize(svd->U,requested,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

