/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscInt       Ma,Na,Mb,Nb,ma,na,mb,nb,M0,N0,m0,n0;
  PetscBool      samesize=PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscCheckSameComm(svd,1,A,2);
  if (B) PetscCheckSameComm(svd,1,B,3);

  /* Check matrix sizes */
  PetscCall(MatGetSize(A,&Ma,&Na));
  PetscCall(MatGetLocalSize(A,&ma,&na));
  if (svd->OP) {
    PetscCall(MatGetSize(svd->OP,&M0,&N0));
    PetscCall(MatGetLocalSize(svd->OP,&m0,&n0));
    if (M0!=Ma || N0!=Na || m0!=ma || n0!=na) samesize = PETSC_FALSE;
  }
  if (B) {
    PetscCall(MatGetSize(B,&Mb,&Nb));
    PetscCall(MatGetLocalSize(B,&mb,&nb));
    PetscCheck(Na==Nb,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Different number of columns in A (%" PetscInt_FMT ") and B (%" PetscInt_FMT ")",Na,Nb);
    PetscCheck(na==nb,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Different local column size in A (%" PetscInt_FMT ") and B (%" PetscInt_FMT ")",na,nb);
    if (svd->OPb) {
      PetscCall(MatGetSize(svd->OPb,&M0,&N0));
      PetscCall(MatGetLocalSize(svd->OPb,&m0,&n0));
      if (M0!=Mb || N0!=Nb || m0!=mb || n0!=nb) samesize = PETSC_FALSE;
    }
  }

  PetscCall(PetscObjectReference((PetscObject)A));
  if (B) PetscCall(PetscObjectReference((PetscObject)B));
  if (svd->state && !samesize) PetscCall(SVDReset(svd));
  else {
    PetscCall(MatDestroy(&svd->OP));
    PetscCall(MatDestroy(&svd->OPb));
    PetscCall(MatDestroy(&svd->A));
    PetscCall(MatDestroy(&svd->B));
    PetscCall(MatDestroy(&svd->AT));
    PetscCall(MatDestroy(&svd->BT));
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
   SVDSetSignature - Set the signature matrix defining a hyperbolic singular value problem.

   Collective on svd

   Input Parameters:
+  svd   - the singular value solver context
-  omega - a vector containing the diagonal elements of the signature matrix (or NULL)

   Notes:
   The signature matrix is relevant only for hyperbolic problems (HSVD).
   Use NULL to reset a previously set signature.

   Level: intermediate

.seealso: SVDSetProblemType(), SVDSetOperators(), SVDGetSignature()
@*/
PetscErrorCode SVDSetSignature(SVD svd,Vec omega)
{
  PetscInt N,Ma,n,ma;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (omega) {
    PetscValidHeaderSpecific(omega,VEC_CLASSID,2);
    PetscCheckSameComm(svd,1,omega,2);
  }

  if (omega && svd->OP) {  /* Check sizes */
    PetscCall(VecGetSize(omega,&N));
    PetscCall(VecGetLocalSize(omega,&n));
    PetscCall(MatGetSize(svd->OP,&Ma,NULL));
    PetscCall(MatGetLocalSize(svd->OP,&ma,NULL));
    PetscCheck(N==Ma,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Global size of signature (%" PetscInt_FMT ") does not match the row size of A (%" PetscInt_FMT ")",N,Ma);
    PetscCheck(n==ma,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Local size of signature (%" PetscInt_FMT ") does not match the local row size of A (%" PetscInt_FMT ")",n,ma);
  }

  if (omega) PetscCall(PetscObjectReference((PetscObject)omega));
  PetscCall(VecDestroy(&svd->omega));
  svd->omega = omega;
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   SVDGetSignature - Get the signature matrix defining a hyperbolic singular value problem.

   Collective on svd

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  omega - a vector containing the diagonal elements of the signature matrix

   Level: intermediate

.seealso: SVDSetSignature()
@*/
PetscErrorCode SVDGetSignature(SVD svd,Vec *omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(omega,2);
  *omega = svd->omega;
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
  PetscBool      flg;
  PetscInt       M,N,P=0,k,maxnsol;
  SlepcSC        sc;
  Vec            *T;
  BV             bv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->state) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(SVD_SetUp,svd,0,0,0));

  /* reset the convergence flag from the previous solves */
  svd->reason = SVD_CONVERGED_ITERATING;

  /* set default solver type (SVDSetFromOptions was not called) */
  if (!((PetscObject)svd)->type_name) PetscCall(SVDSetType(svd,SVDCROSS));
  if (!svd->ds) PetscCall(SVDGetDS(svd,&svd->ds));

  /* check matrices */
  PetscCheck(svd->OP,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONGSTATE,"SVDSetOperators() must be called first");

  if (!svd->problem_type) {  /* set default problem type */
    if (svd->OPb) {
      PetscCheck(!svd->omega,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"There is no support yet for generalized hyperbolic problems");
      PetscCall(SVDSetProblemType(svd,SVD_GENERALIZED));
    } else {
      if (svd->omega) PetscCall(SVDSetProblemType(svd,SVD_HYPERBOLIC));
      else PetscCall(SVDSetProblemType(svd,SVD_STANDARD));
    }
  } else {  /* check consistency of problem type set by user */
    if (svd->OPb) {
      PetscCheck(svd->isgeneralized,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_INCOMP,"Inconsistent SVD state: the problem type does not match the number of matrices");
      PetscCheck(!svd->omega,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"There is no support yet for generalized hyperbolic problems");
    } else {
      PetscCheck(!svd->isgeneralized,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_INCOMP,"Inconsistent SVD state: the problem type does not match the number of matrices");
      if (svd->omega) PetscCheck(svd->ishyperbolic,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_INCOMP,"Inconsistent SVD state: the problem type must be set to hyperbolic when passing a signature with SVDSetSignature()");
      else PetscCheck(!svd->ishyperbolic,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_INCOMP,"Inconsistent SVD state: a hyperbolic problem requires passing a signature with SVDSetSignature()");
    }
  }

  /* determine how to handle the transpose */
  svd->expltrans = PETSC_TRUE;
  if (svd->impltrans) svd->expltrans = PETSC_FALSE;
  else {
    PetscCall(MatHasOperation(svd->OP,MATOP_TRANSPOSE,&flg));
    if (!flg) svd->expltrans = PETSC_FALSE;
    else {
      PetscCall(PetscObjectTypeCompareAny((PetscObject)svd,&flg,SVDLAPACK,SVDSCALAPACK,SVDELEMENTAL,""));
      if (flg) svd->expltrans = PETSC_FALSE;
    }
  }

  /* get matrix dimensions */
  PetscCall(MatGetSize(svd->OP,&M,&N));
  if (svd->isgeneralized) {
    PetscCall(MatGetSize(svd->OPb,&P,NULL));
    PetscCheck(M+P>=N,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"The case when [A;B] has less rows than columns is not supported");
  }

  /* build transpose matrix */
  PetscCall(MatDestroy(&svd->A));
  PetscCall(MatDestroy(&svd->AT));
  PetscCall(PetscObjectReference((PetscObject)svd->OP));
  if (svd->expltrans) {
    if (svd->isgeneralized || M>=N) {
      svd->A = svd->OP;
      PetscCall(MatHermitianTranspose(svd->OP,MAT_INITIAL_MATRIX,&svd->AT));
    } else {
      PetscCall(MatHermitianTranspose(svd->OP,MAT_INITIAL_MATRIX,&svd->A));
      svd->AT = svd->OP;
    }
  } else {
    if (svd->isgeneralized || M>=N) {
      svd->A = svd->OP;
      PetscCall(MatCreateHermitianTranspose(svd->OP,&svd->AT));
    } else {
      PetscCall(MatCreateHermitianTranspose(svd->OP,&svd->A));
      svd->AT = svd->OP;
    }
  }

  /* build transpose matrix B for GSVD */
  if (svd->isgeneralized) {
    PetscCall(MatDestroy(&svd->B));
    PetscCall(MatDestroy(&svd->BT));
    PetscCall(PetscObjectReference((PetscObject)svd->OPb));
    if (svd->expltrans) {
      svd->B = svd->OPb;
      PetscCall(MatHermitianTranspose(svd->OPb,MAT_INITIAL_MATRIX,&svd->BT));
    } else {
      svd->B = svd->OPb;
      PetscCall(MatCreateHermitianTranspose(svd->OPb,&svd->BT));
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
  PetscCheck(svd->ncv==PETSC_DEFAULT || svd->nsv<=svd->ncv,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"nsv bigger than ncv");

  /* relative convergence criterion is not allowed in GSVD */
  if (svd->conv==(SVDConv)-1) PetscCall(SVDSetConvergenceTest(svd,svd->isgeneralized?SVD_CONV_NORM:SVD_CONV_REL));
  PetscCheck(!svd->isgeneralized || svd->conv!=SVD_CONV_REL,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Relative convergence criterion is not allowed in GSVD");

  /* initialization of matrix norm (stardard case only, for GSVD it is done inside setup()) */
  if (!svd->isgeneralized && svd->conv==SVD_CONV_NORM && !svd->nrma) PetscCall(MatNorm(svd->OP,NORM_INFINITY,&svd->nrma));

  /* call specific solver setup */
  PetscUseTypeMethod(svd,setup);

  /* set tolerance if not yet set */
  if (svd->tol==PETSC_DEFAULT) svd->tol = SLEPC_DEFAULT_TOL;

  /* fill sorting criterion context */
  PetscCall(DSGetSlepcSC(svd->ds,&sc));
  sc->comparison    = (svd->which==SVD_LARGEST)? SlepcCompareLargestReal: SlepcCompareSmallestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;

  /* process initial vectors */
  if (svd->nini<0) {
    k = -svd->nini;
    PetscCheck(k<=svd->ncv,PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The number of initial vectors is larger than ncv");
    PetscCall(BVInsertVecs(svd->V,0,&k,svd->IS,PETSC_TRUE));
    PetscCall(SlepcBasisDestroy_Private(&svd->nini,&svd->IS));
    svd->nini = k;
  }
  if (svd->ninil<0) {
    k = 0;
    if (svd->leftbasis) {
      k = -svd->ninil;
      PetscCheck(k<=svd->ncv,PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The number of left initial vectors is larger than ncv");
      PetscCall(BVInsertVecs(svd->U,0,&k,svd->ISL,PETSC_TRUE));
    } else PetscCall(PetscInfo(svd,"Ignoring initial left vectors\n"));
    PetscCall(SlepcBasisDestroy_Private(&svd->ninil,&svd->ISL));
    svd->ninil = k;
  }

  PetscCall(PetscLogEventEnd(SVD_SetUp,svd,0,0,0));
  svd->state = SVD_STATE_SETUP;
  PetscFunctionReturn(0);
}

/*@C
   SVDSetInitialSpaces - Specify two basis of vectors that constitute the initial
   right and/or left spaces.

   Collective on svd

   Input Parameters:
+  svd   - the singular value solver context
.  nr    - number of right vectors
.  isr   - set of basis vectors of the right initial space
.  nl    - number of left vectors
-  isl   - set of basis vectors of the left initial space

   Notes:
   The initial right and left spaces are rough approximations to the right and/or
   left singular subspaces from which the solver starts to iterate.
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

.seealso: SVDSetUp()
@*/
PetscErrorCode SVDSetInitialSpaces(SVD svd,PetscInt nr,Vec isr[],PetscInt nl,Vec isl[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,nr,2);
  PetscValidLogicalCollectiveInt(svd,nl,4);
  PetscCheck(nr>=0,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Argument nr cannot be negative");
  if (nr>0) {
    PetscValidPointer(isr,3);
    PetscValidHeaderSpecific(*isr,VEC_CLASSID,3);
  }
  PetscCheck(nl>=0,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Argument nl cannot be negative");
  if (nl>0) {
    PetscValidPointer(isl,5);
    PetscValidHeaderSpecific(*isl,VEC_CLASSID,5);
  }
  PetscCall(SlepcBasisReference_Private(nr,isr,&svd->nini,&svd->IS));
  PetscCall(SlepcBasisReference_Private(nl,isl,&svd->ninil,&svd->ISL));
  if (nr>0 || nl>0) svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*
  SVDSetDimensions_Default - Set reasonable values for ncv, mpd if not set
  by the user. This is called at setup.
 */
PetscErrorCode SVDSetDimensions_Default(SVD svd)
{
  PetscInt       N,M,P,maxnsol;

  PetscFunctionBegin;
  PetscCall(MatGetSize(svd->OP,&M,&N));
  maxnsol = PetscMin(M,N);
  if (svd->isgeneralized) {
    PetscCall(MatGetSize(svd->OPb,&P,NULL));
    maxnsol = PetscMin(maxnsol,P);
  }
  if (svd->ncv!=PETSC_DEFAULT) { /* ncv set */
    PetscCheck(svd->ncv>=svd->nsv,PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must be at least nsv");
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

   Developer Notes:
   This is SLEPC_EXTERN because it may be required by user plugin SVD
   implementations.

   This is called at setup after setting the value of ncv and the flag leftbasis.

   Level: developer

.seealso: SVDSetUp()
@*/
PetscErrorCode SVDAllocateSolution(SVD svd,PetscInt extra)
{
  PetscInt       oldsize,requested;
  Vec            tr,tl;

  PetscFunctionBegin;
  requested = svd->ncv + extra;

  /* oldsize is zero if this is the first time setup is called */
  PetscCall(BVGetSizes(svd->V,NULL,NULL,&oldsize));

  /* allocate sigma */
  if (requested != oldsize || !svd->sigma) {
    PetscCall(PetscFree3(svd->sigma,svd->perm,svd->errest));
    if (svd->sign) PetscCall(PetscFree(svd->sign));
    PetscCall(PetscMalloc3(requested,&svd->sigma,requested,&svd->perm,requested,&svd->errest));
    if (svd->ishyperbolic) PetscCall(PetscMalloc1(requested,&svd->sign));
  }
  /* allocate V */
  if (!svd->V) PetscCall(SVDGetBV(svd,&svd->V,NULL));
  if (!oldsize) {
    if (!((PetscObject)(svd->V))->type_name) PetscCall(BVSetType(svd->V,BVSVEC));
    PetscCall(MatCreateVecsEmpty(svd->A,&tr,NULL));
    PetscCall(BVSetSizesFromVec(svd->V,tr,requested));
    PetscCall(VecDestroy(&tr));
  } else PetscCall(BVResize(svd->V,requested,PETSC_FALSE));
  /* allocate U */
  if (svd->leftbasis && !svd->isgeneralized) {
    if (!svd->U) PetscCall(SVDGetBV(svd,NULL,&svd->U));
    if (!oldsize) {
      if (!((PetscObject)(svd->U))->type_name) PetscCall(BVSetType(svd->U,((PetscObject)(svd->V))->type_name));
      PetscCall(MatCreateVecsEmpty(svd->A,NULL,&tl));
      PetscCall(BVSetSizesFromVec(svd->U,tl,requested));
      PetscCall(VecDestroy(&tl));
    } else PetscCall(BVResize(svd->U,requested,PETSC_FALSE));
  } else if (svd->isgeneralized) {  /* left basis for the GSVD */
    if (!svd->U) PetscCall(SVDGetBV(svd,NULL,&svd->U));
    if (!oldsize) {
      if (!((PetscObject)(svd->U))->type_name) PetscCall(BVSetType(svd->U,((PetscObject)(svd->V))->type_name));
      PetscCall(SVDCreateLeftTemplate(svd,&tl));
      PetscCall(BVSetSizesFromVec(svd->U,tl,requested));
      PetscCall(VecDestroy(&tl));
    } else PetscCall(BVResize(svd->U,requested,PETSC_FALSE));
  }
  PetscFunctionReturn(0);
}
