/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SVD routines for setting up the solver
*/

#include <slepc/private/svdimpl.h>      /*I "slepcsvd.h" I*/

/*@
   SVDSetOperators - Set the matrices associated with the singular value problem.

   Collective on svd

   Input Parameters:
+  svd - the singular value solver context
.  A   - the matrix associated with the singular value problem
-  B   - the second matrix in the case of GSVD

   Level: beginner

.seealso: SVDSolve(), SVDGetOperators()
@*/
PetscErrorCode SVDSetOperators(SVD svd,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       Ma,Na,Mb,Nb,ma,na,mb,nb,M0,N0,m0,n0;
  PetscBool      samesize=PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscCheckSameComm(svd,1,A,2);
  if (B) PetscCheckSameComm(svd,1,B,3);

  /* Check matrix sizes */
  ierr = MatGetSize(A,&Ma,&Na);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&ma,&na);CHKERRQ(ierr);
  if (svd->OP) {
    ierr = MatGetSize(svd->OP,&M0,&N0);CHKERRQ(ierr);
    ierr = MatGetLocalSize(svd->OP,&m0,&n0);CHKERRQ(ierr);
    if (M0!=Ma || N0!=Na || m0!=ma || n0!=na) samesize = PETSC_FALSE;
  }
  if (B) {
    ierr = MatGetSize(B,&Mb,&Nb);CHKERRQ(ierr);
    ierr = MatGetLocalSize(B,&mb,&nb);CHKERRQ(ierr);
    if (Na!=Nb) SETERRQ2(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Different number of columns in A (%D) and B (%D)",Na,Nb);
    if (na!=nb) SETERRQ2(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Different local column size in A (%D) and B (%D)",na,nb);
    if (svd->OPb) {
      ierr = MatGetSize(svd->OPb,&M0,&N0);CHKERRQ(ierr);
      ierr = MatGetLocalSize(svd->OPb,&m0,&n0);CHKERRQ(ierr);
      if (M0!=Mb || N0!=Nb || m0!=mb || n0!=nb) samesize = PETSC_FALSE;
    }
  }

  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  if (B) { ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr); }
  if (svd->state && !samesize) {
    ierr = SVDReset(svd);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&svd->OP);CHKERRQ(ierr);
    ierr = MatDestroy(&svd->OPb);CHKERRQ(ierr);
    ierr = MatDestroy(&svd->A);CHKERRQ(ierr);
    ierr = MatDestroy(&svd->B);CHKERRQ(ierr);
    ierr = MatDestroy(&svd->AT);CHKERRQ(ierr);
    ierr = MatDestroy(&svd->BT);CHKERRQ(ierr);
  }
  svd->nrma = 0.0;
  svd->nrmb = 0.0;
  svd->OP   = A;
  svd->OPb  = B;
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   SVDGetOperators - Get the matrices associated with the singular value problem.

   Collective on svd

   Input Parameter:
.  svd - the singular value solver context

   Output Parameters:
+  A  - the matrix associated with the singular value problem
-  B  - the second matrix in the case of GSVD

   Level: intermediate

.seealso: SVDSolve(), SVDSetOperators()
@*/
PetscErrorCode SVDGetOperators(SVD svd,Mat *A,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (A) *A = svd->OP;
  if (B) *B = svd->OPb;
  PetscFunctionReturn(0);
}

/*@
   SVDSetUp - Sets up all the internal data structures necessary for the
   execution of the singular value solver.

   Collective on svd

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
  PetscBool      flg;
  PetscInt       M,N,P=0,k,maxnsol;
  SlepcSC        sc;
  Vec            *T;
  BV             bv;

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

  /* check matrices */
  if (!svd->OP) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONGSTATE,"SVDSetOperators() must be called first");

  /* Set default problem type */
  if (!svd->problem_type) {
    if (svd->OPb) {
      ierr = SVDSetProblemType(svd,SVD_GENERALIZED);CHKERRQ(ierr);
    } else {
      ierr = SVDSetProblemType(svd,SVD_STANDARD);CHKERRQ(ierr);
    }
  } else if (!svd->OPb && svd->isgeneralized) {
    ierr = PetscInfo(svd,"Problem type set as generalized but no matrix B was provided; reverting to a standard singular value problem\n");CHKERRQ(ierr);
    svd->isgeneralized = PETSC_FALSE;
    svd->problem_type = SVD_STANDARD;
  } else if (svd->OPb && !svd->isgeneralized) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_INCOMP,"Inconsistent SVD state: the problem type does not match the number of matrices");

  /* determine how to handle the transpose */
  svd->expltrans = PETSC_TRUE;
  if (svd->impltrans) svd->expltrans = PETSC_FALSE;
  else {
    ierr = MatHasOperation(svd->OP,MATOP_TRANSPOSE,&flg);CHKERRQ(ierr);
    if (!flg) svd->expltrans = PETSC_FALSE;
    else {
      ierr = PetscObjectTypeCompareAny((PetscObject)svd,&flg,SVDLAPACK,SVDSCALAPACK,SVDELEMENTAL,"");CHKERRQ(ierr);
      if (flg) svd->expltrans = PETSC_FALSE;
    }
  }

  /* get matrix dimensions */
  ierr = MatGetSize(svd->OP,&M,&N);CHKERRQ(ierr);
  if (svd->isgeneralized) {
    ierr = MatGetSize(svd->OPb,&P,NULL);CHKERRQ(ierr);
    if (M+P<N) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"The case when [A;B] has less rows than columns is not supported");
  }

  /* build transpose matrix */
  ierr = MatDestroy(&svd->A);CHKERRQ(ierr);
  ierr = MatDestroy(&svd->AT);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)svd->OP);CHKERRQ(ierr);
  if (svd->expltrans) {
    if (svd->isgeneralized || M>=N) {
      svd->A = svd->OP;
      ierr = MatHermitianTranspose(svd->OP,MAT_INITIAL_MATRIX,&svd->AT);CHKERRQ(ierr);
    } else {
      ierr = MatHermitianTranspose(svd->OP,MAT_INITIAL_MATRIX,&svd->A);CHKERRQ(ierr);
      svd->AT = svd->OP;
    }
  } else {
    if (svd->isgeneralized || M>=N) {
      svd->A = svd->OP;
      ierr = MatCreateHermitianTranspose(svd->OP,&svd->AT);CHKERRQ(ierr);
    } else {
      ierr = MatCreateHermitianTranspose(svd->OP,&svd->A);CHKERRQ(ierr);
      svd->AT = svd->OP;
    }
  }

  /* build transpose matrix B for GSVD */
  if (svd->isgeneralized) {
    ierr = MatDestroy(&svd->B);CHKERRQ(ierr);
    ierr = MatDestroy(&svd->BT);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)svd->OPb);CHKERRQ(ierr);
    if (svd->expltrans) {
      svd->B = svd->OPb;
      ierr = MatHermitianTranspose(svd->OPb,MAT_INITIAL_MATRIX,&svd->BT);CHKERRQ(ierr);
    } else {
      svd->B = svd->OPb;
      ierr = MatCreateHermitianTranspose(svd->OPb,&svd->BT);CHKERRQ(ierr);
    }
  }

  if (!svd->isgeneralized && M<N) {
    /* swap initial vectors */
    if (svd->nini || svd->ninil) {
      T=svd->ISL; svd->ISL=svd->IS; svd->IS=T;
      k=svd->ninil; svd->ninil=svd->nini; svd->nini=k;
    }
    /* swap basis vectors */
    if (!svd->swapped) {  /* only the first time in case of multiple calls */
      bv=svd->V; svd->V=svd->U; svd->U=bv;
      svd->swapped = PETSC_TRUE;
    }
  }

  maxnsol = svd->isgeneralized? PetscMin(PetscMin(M,N),P): PetscMin(M,N);
  svd->ncv = PetscMin(svd->ncv,maxnsol);
  svd->nsv = PetscMin(svd->nsv,maxnsol);
  if (svd->ncv!=PETSC_DEFAULT && svd->nsv > svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"nsv bigger than ncv");

  /* initialization of matrix norms */
  if (svd->conv==SVD_CONV_NORM) {
    if (!svd->nrma) {
      ierr = MatNorm(svd->OP,NORM_INFINITY,&svd->nrma);CHKERRQ(ierr);
    }
    if (svd->isgeneralized && !svd->nrmb) {
      ierr = MatNorm(svd->OPb,NORM_INFINITY,&svd->nrmb);CHKERRQ(ierr);
    }
  }

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
    if (k>svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The number of initial vectors is larger than ncv");
    ierr = BVInsertVecs(svd->V,0,&k,svd->IS,PETSC_TRUE);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&svd->nini,&svd->IS);CHKERRQ(ierr);
    svd->nini = k;
  }
  if (svd->ninil<0) {
    k = 0;
    if (svd->leftbasis) {
      k = -svd->ninil;
      if (k>svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The number of left initial vectors is larger than ncv");
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

/*@C
   SVDSetInitialSpaces - Specify two basis of vectors that constitute the initial
   right and/or left spaces, that is, a rough approximation to the right and/or
   left singular subspaces from which the solver starts to iterate.

   Collective on svd

   Input Parameters:
+  svd   - the singular value solver context
.  nr    - number of right vectors
.  isr   - set of basis vectors of the right initial space
.  nl    - number of left vectors
-  isl   - set of basis vectors of the left initial space

   Notes:
   It is not necessary to provide both sets of vectors.

   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one SVDSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted singular space. Then, convergence may be faster.

   Level: intermediate
@*/
PetscErrorCode SVDSetInitialSpaces(SVD svd,PetscInt nr,Vec isr[],PetscInt nl,Vec isl[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,nr,2);
  PetscValidLogicalCollectiveInt(svd,nl,4);
  if (nr<0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Argument nr cannot be negative");
  if (nr>0) {
    PetscValidPointer(isr,3);
    PetscValidHeaderSpecific(*isr,VEC_CLASSID,3);
  }
  if (nl<0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Argument nl cannot be negative");
  if (nl>0) {
    PetscValidPointer(isl,5);
    PetscValidHeaderSpecific(*isl,VEC_CLASSID,5);
  }
  ierr = SlepcBasisReference_Private(nr,isr,&svd->nini,&svd->IS);CHKERRQ(ierr);
  ierr = SlepcBasisReference_Private(nl,isl,&svd->ninil,&svd->ISL);CHKERRQ(ierr);
  if (nr>0 || nl>0) svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*
  SVDSetDimensions_Default - Set reasonable values for ncv, mpd if not set
  by the user. This is called at setup.
 */
PetscErrorCode SVDSetDimensions_Default(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       N,M,P,maxnsol;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->OP,&M,&N);CHKERRQ(ierr);
  maxnsol = PetscMin(M,N);
  if (svd->isgeneralized) {
    ierr = MatGetSize(svd->OPb,&P,NULL);CHKERRQ(ierr);
    maxnsol = PetscMin(maxnsol,P);
  }
  if (svd->ncv!=PETSC_DEFAULT) { /* ncv set */
    if (svd->ncv<svd->nsv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must be at least nsv");
  } else if (svd->mpd!=PETSC_DEFAULT) { /* mpd set */
    svd->ncv = PetscMin(maxnsol,svd->nsv+svd->mpd);
  } else { /* neither set: defaults depend on nsv being small or large */
    if (svd->nsv<500) svd->ncv = PetscMin(maxnsol,PetscMax(2*svd->nsv,10));
    else {
      svd->mpd = 500;
      svd->ncv = PetscMin(maxnsol,svd->nsv+svd->mpd);
    }
  }
  if (svd->mpd==PETSC_DEFAULT) svd->mpd = svd->ncv;
  PetscFunctionReturn(0);
}

/*@
   SVDAllocateSolution - Allocate memory storage for common variables such
   as the singular values and the basis vectors.

   Collective on svd

   Input Parameters:
+  svd   - eigensolver context
-  extra - number of additional positions, used for methods that require a
           working basis slightly larger than ncv

   Developers Notes:
   This is SLEPC_EXTERN because it may be required by user plugin SVD
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
    ierr = PetscFree3(svd->sigma,svd->perm,svd->errest);CHKERRQ(ierr);
    ierr = PetscMalloc3(requested,&svd->sigma,requested,&svd->perm,requested,&svd->errest);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)svd,PetscMax(0,requested-oldsize)*(2*sizeof(PetscReal)+sizeof(PetscInt)));CHKERRQ(ierr);
  }
  /* allocate V */
  if (!svd->V) { ierr = SVDGetBV(svd,&svd->V,NULL);CHKERRQ(ierr); }
  if (!oldsize) {
    if (!((PetscObject)(svd->V))->type_name) {
      ierr = BVSetType(svd->V,BVSVEC);CHKERRQ(ierr);
    }
    ierr = MatCreateVecsEmpty(svd->A,&tr,NULL);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(svd->V,tr,requested);CHKERRQ(ierr);
    ierr = VecDestroy(&tr);CHKERRQ(ierr);
  } else {
    ierr = BVResize(svd->V,requested,PETSC_FALSE);CHKERRQ(ierr);
  }
  /* allocate U */
  if (svd->leftbasis && !svd->isgeneralized) {
    if (!svd->U) { ierr = SVDGetBV(svd,NULL,&svd->U);CHKERRQ(ierr); }
    if (!oldsize) {
      if (!((PetscObject)(svd->U))->type_name) {
        ierr = BVSetType(svd->U,BVSVEC);CHKERRQ(ierr);
      }
      ierr = MatCreateVecsEmpty(svd->A,NULL,&tl);CHKERRQ(ierr);
      ierr = BVSetSizesFromVec(svd->U,tl,requested);CHKERRQ(ierr);
      ierr = VecDestroy(&tl);CHKERRQ(ierr);
    } else {
      ierr = BVResize(svd->U,requested,PETSC_FALSE);CHKERRQ(ierr);
    }
  } else if (svd->isgeneralized) {  /* left basis for the GSVD */
    if (!svd->U) { ierr = SVDGetBV(svd,NULL,&svd->U);CHKERRQ(ierr); }
    if (!oldsize) {
      if (!((PetscObject)(svd->U))->type_name) {
        ierr = BVSetType(svd->U,BVSVEC);CHKERRQ(ierr);
      }
      ierr = SVDCreateLeftTemplate(svd,&tl);CHKERRQ(ierr);
      ierr = BVSetSizesFromVec(svd->U,tl,requested);CHKERRQ(ierr);
      ierr = VecDestroy(&tl);CHKERRQ(ierr);
    } else {
      ierr = BVResize(svd->U,requested,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

