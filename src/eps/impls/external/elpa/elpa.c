/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to eigensolvers in ELPA.
*/

#include <slepc/private/epsimpl.h>    /*I "slepceps.h" I*/
#include <slepc/private/slepcscalapack.h>
#include <elpa/elpa.h>

#define CHKERRELPA(func, ...) do {                                                   \
    PetscErrorCode elpa_ierr_; \
    PetscStackPush(PetscStringize(func));                                   \
    func(__VA_ARGS__,&elpa_ierr_);                                              \
    PetscStackPop;                                                                             \
    PetscCheck(!elpa_ierr_,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling %s: error code %d",PetscStringize(func(__VA_ARGS__,&elpa_ierr)),elpa_ierr_); \
  } while (0)

#define CHKERRELPARET(func, ...) do {                                                   \
    PetscStackPush(PetscStringize(func));                                                      \
    PetscErrorCode elpa_ierr_ = func(__VA_ARGS__);                                              \
    PetscStackPop;                                                                             \
    PetscCheck(!elpa_ierr_,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling %s: error code %d",PetscStringize(func(__VA_ARGS__)),elpa_ierr_); \
  } while (0)

#define CHKERRELPANOARG(func) do {                                                   \
    PetscErrorCode elpa_ierr_; \
    PetscStackPush(PetscStringize(func));                                   \
    func(&elpa_ierr_);                                              \
    PetscStackPop;                                                                             \
    PetscCheck(!elpa_ierr_,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling %s: error code %d",PetscStringize(func(&elpa_ierr)),elpa_ierr_); \
  } while (0)

typedef struct {
  Mat As,Bs;        /* converted matrices */
} EPS_ELPA;

PetscErrorCode EPSSetUp_ELPA(EPS eps)
{
  EPS_ELPA       *ctx = (EPS_ELPA*)eps->data;
  Mat            A,B;
  PetscInt       nmat;
  PetscBool      isshift;
  PetscScalar    shift;

  PetscFunctionBegin;
  EPSCheckHermitianDefinite(eps);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift));
  PetscCheck(isshift,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");
  eps->ncv = eps->n;
  if (eps->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(eps,"Warning: parameter mpd ignored\n"));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  PetscCheck(eps->which!=EPS_ALL || eps->inta==eps->intb,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support interval computation");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);
  CHKERRQ(EPSAllocateSolution(eps,0));

  /* convert matrices */
  CHKERRQ(MatDestroy(&ctx->As));
  CHKERRQ(MatDestroy(&ctx->Bs));
  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  CHKERRQ(MatConvert(A,MATSCALAPACK,MAT_INITIAL_MATRIX,&ctx->As));
  if (nmat>1) {
    CHKERRQ(STGetMatrix(eps->st,1,&B));
    CHKERRQ(MatConvert(B,MATSCALAPACK,MAT_INITIAL_MATRIX,&ctx->Bs));
  }
  CHKERRQ(STGetShift(eps->st,&shift));
  if (shift != 0.0) {
    if (nmat>1) CHKERRQ(MatAXPY(ctx->As,-shift,ctx->Bs,SAME_NONZERO_PATTERN));
    else CHKERRQ(MatShift(ctx->As,-shift));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_ELPA(EPS eps)
{
  EPS_ELPA       *ctx = (EPS_ELPA*)eps->data;
  Mat            A = ctx->As,B = ctx->Bs,Q,V;
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data,*b,*q;
  PetscReal      *w = eps->errest;  /* used to store real eigenvalues */
  PetscInt       i;
  elpa_t         handle;

  PetscFunctionBegin;
  CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Q));
  q = (Mat_ScaLAPACK*)Q->data;

  CHKERRELPARET(elpa_init,20200417);    /* 20171201 */
  CHKERRELPANOARG(handle = elpa_allocate);

  /* set parameters of the matrix and its MPI distribution */
  CHKERRELPA(elpa_set,handle,"na",a->N);                         /* matrix size */
  CHKERRELPA(elpa_set,handle,"nev",a->N);                        /* number of eigenvectors to be computed (1<=nev<=na) */
  CHKERRELPA(elpa_set,handle,"local_nrows",a->locr);             /* number of local rows of the distributed matrix on this MPI task  */
  CHKERRELPA(elpa_set,handle,"local_ncols",a->locc);             /* number of local columns of the distributed matrix on this MPI task */
  CHKERRELPA(elpa_set,handle,"nblk",a->mb);                      /* size of the BLACS block cyclic distribution */
  CHKERRELPA(elpa_set,handle,"mpi_comm_parent",MPI_Comm_c2f(PetscObjectComm((PetscObject)eps)));
  CHKERRELPA(elpa_set,handle,"process_row",a->grid->myrow);      /* row coordinate of MPI process */
  CHKERRELPA(elpa_set,handle,"process_col",a->grid->mycol);      /* column coordinate of MPI process */
  if (B) CHKERRELPA(elpa_set,handle,"blacs_context",a->grid->ictxt);

  /* setup and set tunable run-time options */
  CHKERRELPARET(elpa_setup,handle);
  CHKERRELPA(elpa_set,handle,"solver",ELPA_SOLVER_2STAGE);
  /* CHKERRELPA(elpa_print_settings,handle); */

  /* solve the eigenvalue problem */
  if (B) {
    b = (Mat_ScaLAPACK*)B->data;
    CHKERRELPA(elpa_generalized_eigenvectors,handle,a->loc,b->loc,w,q->loc,0);
  } else CHKERRELPA(elpa_eigenvectors,handle,a->loc,w,q->loc);

  /* cleanup */
  CHKERRELPA(elpa_deallocate,handle);
  CHKERRELPANOARG(elpa_uninit);

  for (i=0;i<eps->ncv;i++) {
    eps->eigr[i]   = eps->errest[i];
    eps->errest[i] = PETSC_MACHINE_EPSILON;
  }

  CHKERRQ(BVGetMat(eps->V,&V));
  CHKERRQ(MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&V));
  CHKERRQ(BVRestoreMat(eps->V,&V));
  CHKERRQ(MatDestroy(&Q));

  eps->nconv  = eps->ncv;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_ELPA(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_ELPA(EPS eps)
{
  EPS_ELPA       *ctx = (EPS_ELPA*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&ctx->As));
  CHKERRQ(MatDestroy(&ctx->Bs));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_ELPA(EPS eps)
{
  EPS_ELPA       *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
  eps->data = (void*)ctx;

  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_ELPA;
  eps->ops->setup          = EPSSetUp_ELPA;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->destroy        = EPSDestroy_ELPA;
  eps->ops->reset          = EPSReset_ELPA;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;
  PetscFunctionReturn(0);
}
