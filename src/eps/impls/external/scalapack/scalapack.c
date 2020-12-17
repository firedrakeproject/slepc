/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to eigensolvers in ScaLAPACK.
*/

#include <slepc/private/epsimpl.h>    /*I "slepceps.h" I*/
#include <slepc/private/slepcscalapack.h>

typedef struct {
  Mat As,Bs;        /* converted matrices */
} EPS_ScaLAPACK;

PetscErrorCode EPSSetUp_ScaLAPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ScaLAPACK  *ctx = (EPS_ScaLAPACK*)eps->data;
  Mat            A,B;
  PetscInt       nmat;
  PetscBool      isshift;
  PetscScalar    shift;

  PetscFunctionBegin;
  EPSCheckHermitianDefinite(eps);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (!isshift) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");
  eps->ncv = eps->n;
  if (eps->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->which==EPS_ALL && eps->inta!=eps->intb) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support interval computation");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);

  /* convert matrices */
  ierr = MatDestroy(&ctx->As);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Bs);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,MATSCALAPACK,MAT_INITIAL_MATRIX,&ctx->As);CHKERRQ(ierr);
  if (nmat>1) {
    ierr = STGetMatrix(eps->st,1,&B);CHKERRQ(ierr);
    ierr = MatConvert(B,MATSCALAPACK,MAT_INITIAL_MATRIX,&ctx->Bs);CHKERRQ(ierr);
  }
  ierr = STGetShift(eps->st,&shift);CHKERRQ(ierr);
  if (shift != 0.0) {
    if (nmat>1) {
      ierr = MatAXPY(ctx->As,-shift,ctx->Bs,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      ierr = MatShift(ctx->As,-shift);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_ScaLAPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ScaLAPACK  *ctx = (EPS_ScaLAPACK*)eps->data;
  Mat            A = ctx->As,B = ctx->Bs,Q,V;
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data,*b,*q;
  PetscReal      rdummy=0.0,abstol=0.0,*gap=NULL,orfac=-1.0,*w = eps->errest;  /* used to store real eigenvalues */
  PetscScalar    *work,minlwork[3];
  PetscBLASInt   i,m,info,idummy=0,lwork=-1,liwork=-1,minliwork,*iwork,*ifail=NULL,*iclustr=NULL,one=1;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork,minlrwork[3];
  PetscBLASInt   lrwork=-1;
#endif

  PetscFunctionBegin;
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Q);CHKERRQ(ierr);
  q = (Mat_ScaLAPACK*)Q->data;

  if (B) {

    b = (Mat_ScaLAPACK*)B->data;
    ierr = PetscMalloc3(a->grid->nprow*a->grid->npcol,&gap,a->N,&ifail,2*a->grid->nprow*a->grid->npcol,&iclustr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    /* allocate workspace */
    PetscStackCallBLAS("SCALAPACKsygvx",SCALAPACKsygvx_(&one,"V","A","L",&a->N,a->loc,&one,&one,a->desc,b->loc,&one,&one,b->desc,&rdummy,&rdummy,&idummy,&idummy,&abstol,&m,&idummy,w,&orfac,q->loc,&one,&one,q->desc,minlwork,&lwork,&minliwork,&liwork,ifail,iclustr,gap,&info));
    PetscCheckScaLapackInfo("sygvx",info);
    ierr = PetscBLASIntCast((PetscInt)minlwork[0],&lwork);CHKERRQ(ierr);
    liwork = minliwork;
    /* call computational routine */
    ierr = PetscMalloc2(lwork,&work,liwork,&iwork);CHKERRQ(ierr);
    PetscStackCallBLAS("SCALAPACKsygvx",SCALAPACKsygvx_(&one,"V","A","L",&a->N,a->loc,&one,&one,a->desc,b->loc,&one,&one,b->desc,&rdummy,&rdummy,&idummy,&idummy,&abstol,&m,&idummy,w,&orfac,q->loc,&one,&one,q->desc,work,&lwork,iwork,&liwork,ifail,iclustr,gap,&info));
    PetscCheckScaLapackInfo("sygvx",info);
    ierr = PetscFree2(work,iwork);CHKERRQ(ierr);
#else
    /* allocate workspace */
    PetscStackCallBLAS("SCALAPACKsygvx",SCALAPACKsygvx_(&one,"V","A","L",&a->N,a->loc,&one,&one,a->desc,b->loc,&one,&one,b->desc,&rdummy,&rdummy,&idummy,&idummy,&abstol,&m,&idummy,w,&orfac,q->loc,&one,&one,q->desc,minlwork,&lwork,minlrwork,&lrwork,&minliwork,&liwork,ifail,iclustr,gap,&info));
    PetscCheckScaLapackInfo("sygvx",info);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(minlwork[0]),&lwork);CHKERRQ(ierr);
    ierr = PetscBLASIntCast((PetscInt)minlrwork[0],&lrwork);CHKERRQ(ierr);
    lrwork += a->N*a->N;
    liwork = minliwork;
    /* call computational routine */
    ierr = PetscMalloc3(lwork,&work,lrwork,&rwork,liwork,&iwork);CHKERRQ(ierr);
    PetscStackCallBLAS("SCALAPACKsygvx",SCALAPACKsygvx_(&one,"V","A","L",&a->N,a->loc,&one,&one,a->desc,b->loc,&one,&one,b->desc,&rdummy,&rdummy,&idummy,&idummy,&abstol,&m,&idummy,w,&orfac,q->loc,&one,&one,q->desc,work,&lwork,rwork,&lrwork,iwork,&liwork,ifail,iclustr,gap,&info));
    PetscCheckScaLapackInfo("sygvx",info);
    ierr = PetscFree3(work,rwork,iwork);CHKERRQ(ierr);
#endif
    ierr = PetscFree3(gap,ifail,iclustr);CHKERRQ(ierr);

  } else {

#if !defined(PETSC_USE_COMPLEX)
    /* allocate workspace */
    PetscStackCallBLAS("SCALAPACKsyev",SCALAPACKsyev_("V","L",&a->N,a->loc,&one,&one,a->desc,w,q->loc,&one,&one,q->desc,minlwork,&lwork,&info));
    PetscCheckScaLapackInfo("syev",info);
    ierr = PetscBLASIntCast((PetscInt)minlwork[0],&lwork);CHKERRQ(ierr);
    ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
    /* call computational routine */
    PetscStackCallBLAS("SCALAPACKsyev",SCALAPACKsyev_("V","L",&a->N,a->loc,&one,&one,a->desc,w,q->loc,&one,&one,q->desc,work,&lwork,&info));
    PetscCheckScaLapackInfo("syev",info);
    ierr = PetscFree(work);CHKERRQ(ierr);
#else
    /* allocate workspace */
    PetscStackCallBLAS("SCALAPACKsyev",SCALAPACKsyev_("V","L",&a->N,a->loc,&one,&one,a->desc,w,q->loc,&one,&one,q->desc,minlwork,&lwork,minlrwork,&lrwork,&info));
    PetscCheckScaLapackInfo("syev",info);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(minlwork[0]),&lwork);CHKERRQ(ierr);
    lrwork = 4*a->N;  /* ierr = PetscBLASIntCast((PetscInt)minlrwork[0],&lrwork);CHKERRQ(ierr); */
    ierr = PetscMalloc2(lwork,&work,lrwork,&rwork);CHKERRQ(ierr);
    /* call computational routine */
    PetscStackCallBLAS("SCALAPACKsyev",SCALAPACKsyev_("V","L",&a->N,a->loc,&one,&one,a->desc,w,q->loc,&one,&one,q->desc,work,&lwork,rwork,&lrwork,&info));
    PetscCheckScaLapackInfo("syev",info);
    ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
#endif

  }

  for (i=0;i<eps->ncv;i++) {
    eps->eigr[i]   = eps->errest[i];
    eps->errest[i] = PETSC_MACHINE_EPSILON;
  }

  ierr = BVGetMat(eps->V,&V);CHKERRQ(ierr);
  ierr = MatConvert(Q,MATDENSE,MAT_REUSE_MATRIX,&V);CHKERRQ(ierr);
  ierr = BVRestoreMat(eps->V,&V);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

  eps->nconv  = eps->ncv;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_ScaLAPACK(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_ScaLAPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ScaLAPACK  *ctx = (EPS_ScaLAPACK*)eps->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->As);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_ScaLAPACK(EPS eps)
{
  EPS_ScaLAPACK  *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_ScaLAPACK;
  eps->ops->setup          = EPSSetUp_ScaLAPACK;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->destroy        = EPSDestroy_ScaLAPACK;
  eps->ops->reset          = EPSReset_ScaLAPACK;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;
  PetscFunctionReturn(0);
}

