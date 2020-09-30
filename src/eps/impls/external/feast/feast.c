/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to the FEAST solver in MKL
*/

#include <petscsys.h>
#if defined(PETSC_HAVE_MKL_INTEL_ILP64)
#define MKL_ILP64
#endif
#include <mkl.h>
#include <slepc/private/epsimpl.h>        /*I "slepceps.h" I*/

#if defined(PETSC_USE_COMPLEX)
#  if defined(PETSC_USE_REAL_SINGLE)
#    define FEAST_RCI cfeast_hrci
#    define SCALAR_CAST (MKL_Complex8*)
#  else
#    define FEAST_RCI zfeast_hrci
#    define SCALAR_CAST (MKL_Complex16*)
#  endif
#else
#  if defined(PETSC_USE_REAL_SINGLE)
#    define FEAST_RCI sfeast_srci
#  else
#    define FEAST_RCI dfeast_srci
#  endif
#  define SCALAR_CAST
#endif

typedef struct {
  PetscInt      npoints;          /* number of contour points */
  PetscScalar   *work1,*Aq,*Bq;   /* workspace */
#if defined(PETSC_USE_REAL_SINGLE)
  MKL_Complex8  *work2;
#else
  MKL_Complex16 *work2;
#endif
} EPS_FEAST;

PetscErrorCode EPSSetUp_FEAST(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       ncv;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size);CHKERRQ(ierr);
  if (size!=1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The FEAST interface is supported for sequential runs only");
  EPSCheckSinvertCayley(eps);
  if (eps->ncv!=PETSC_DEFAULT) {
    if (eps->ncv<eps->nev+2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The value of ncv must be at least nev+2");
  } else eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),eps->n); /* set default value of ncv */
  if (eps->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 20;
  if (!eps->which) eps->which = EPS_ALL;
  if (eps->which!=EPS_ALL || eps->inta==eps->intb) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver must be used with a computational interval");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_CONVERGENCE | EPS_FEATURE_STOPPING | EPS_FEATURE_TWOSIDED);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION);

  if (!ctx->npoints) ctx->npoints = 8;

  ncv = eps->ncv;
  ierr = PetscFree4(ctx->work1,ctx->work2,ctx->Aq,ctx->Bq);CHKERRQ(ierr);
  ierr = PetscMalloc4(eps->nloc*ncv,&ctx->work1,eps->nloc*ncv,&ctx->work2,ncv*ncv,&ctx->Aq,ncv*ncv,&ctx->Bq);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,(2*eps->nloc*ncv+2*ncv*ncv)*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_FEAST(EPS eps)
{
  PetscErrorCode ierr;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  MKL_INT        fpm[128],ijob,n,ncv,nconv,loop,info;
  PetscReal      *evals,epsout=0.0;
  PetscInt       i,k,nmat;
  PetscScalar    *pV,*pz;
  Vec            x,y,w=eps->work[0],z=eps->work[1];
  Mat            A,B;
#if defined(PETSC_USE_REAL_SINGLE)
  MKL_Complex8   Ze;
#else
  MKL_Complex16  Ze;
#endif

  PetscFunctionBegin;
  ncv = eps->ncv;
  n   = eps->nloc;

  /* parameters */
  feastinit(fpm);
  fpm[0] = (eps->numbermonitors>0)? 1: 0;   /* runtime comments */
  fpm[1] = ctx->npoints;                    /* contour points */
#if !defined(PETSC_USE_REAL_SINGLE)
  fpm[2] = -PetscLog10Real(eps->tol);       /* tolerance for trace */
#endif
  fpm[3] = eps->max_it;                     /* refinement loops */
  fpm[5] = 1;                               /* second stopping criterion */
#if defined(PETSC_USE_REAL_SINGLE)
  fpm[6] = -PetscLog10Real(eps->tol);       /* tolerance for trace */
#endif

  ierr = PetscMalloc1(eps->ncv,&evals);CHKERRQ(ierr);
  ierr = BVGetArray(eps->V,&pV);CHKERRQ(ierr);

  ijob = -1;           /* first call to reverse communication interface */
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetMatrix(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;
  ierr = MatCreateVecsEmpty(A,&x,&y);CHKERRQ(ierr);

  do {

    FEAST_RCI(&ijob,&n,&Ze,SCALAR_CAST ctx->work1,ctx->work2,SCALAR_CAST ctx->Aq,SCALAR_CAST ctx->Bq,fpm,&epsout,&loop,&eps->inta,&eps->intb,&ncv,evals,SCALAR_CAST pV,&nconv,eps->errest,&info);

    if (ncv!=eps->ncv) SETERRQ1(PetscObjectComm((PetscObject)eps),1,"FEAST changed value of ncv to %d",ncv);
    if (ijob == 10) {
      /* set new quadrature point */
      ierr = STSetShift(eps->st,Ze.real);CHKERRQ(ierr);
    } else if (ijob == 20) {
      /* use same quadrature point and factorization for transpose solve */
    } else if (ijob == 11 || ijob == 21) {
      /* linear solve (A-sigma*B)\work2, overwrite work2 */
      for (k=0;k<ncv;k++) {
        ierr = VecGetArray(z,&pz);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        for (i=0;i<eps->nloc;i++) pz[i] = PetscCMPLX(ctx->work2[eps->nloc*k+i].real,ctx->work2[eps->nloc*k+i].imag);
#else
        for (i=0;i<eps->nloc;i++) pz[i] = ctx->work2[eps->nloc*k+i].real;
#endif
        ierr = VecRestoreArray(z,&pz);CHKERRQ(ierr);
        if (ijob == 11) {
          ierr = STMatSolve(eps->st,z,w);CHKERRQ(ierr);
        } else {
          ierr = VecConjugate(z);CHKERRQ(ierr);
          ierr = STMatSolveTranspose(eps->st,z,w);CHKERRQ(ierr);
          ierr = VecConjugate(w);CHKERRQ(ierr);
        }
        ierr = VecGetArray(w,&pz);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        for (i=0;i<eps->nloc;i++) {
          ctx->work2[eps->nloc*k+i].real = PetscRealPart(pz[i]);
          ctx->work2[eps->nloc*k+i].imag = PetscImaginaryPart(pz[i]);
        }
#else
        for (i=0;i<eps->nloc;i++) ctx->work2[eps->nloc*k+i].real = pz[i];
#endif
        ierr = VecRestoreArray(w,&pz);CHKERRQ(ierr);
      }
    } else if (ijob == 30 || ijob == 40) {
      /* multiplication A*V or B*V, result in work1 */
      for (k=fpm[23]-1;k<fpm[23]+fpm[24]-1;k++) {
        ierr = VecPlaceArray(x,&pV[k*eps->nloc]);CHKERRQ(ierr);
        ierr = VecPlaceArray(y,&ctx->work1[k*eps->nloc]);CHKERRQ(ierr);
        if (ijob == 30) {
          ierr = MatMult(A,x,y);CHKERRQ(ierr);
        } else if (nmat>1) {
          ierr = MatMult(B,x,y);CHKERRQ(ierr);
        } else {
          ierr = VecCopy(x,y);CHKERRQ(ierr);
        }
        ierr = VecResetArray(x);CHKERRQ(ierr);
        ierr = VecResetArray(y);CHKERRQ(ierr);
      }
    } else if (ijob && ijob!=-2) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Internal error in FEAST reverse comunication interface (ijob=%d)",ijob);

  } while (ijob);

  eps->reason = EPS_CONVERGED_TOL;
  eps->its    = loop;
  eps->nconv  = nconv;
  if (info) {
    if (info==1) { /* No eigenvalue has been found in the proposed search interval */
      eps->nconv = 0;
    } else if (info==2) { /* FEAST did not converge "yet" */
      eps->reason = EPS_DIVERGED_ITS;
    } else SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by FEAST (%d)",info);
  }

  for (i=0;i<eps->nconv;i++) eps->eigr[i] = evals[i];

  ierr = BVRestoreArray(eps->V,&pV);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFree(evals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_FEAST(EPS eps)
{
  PetscErrorCode ierr;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;

  PetscFunctionBegin;
  ierr = PetscFree4(ctx->work1,ctx->work2,ctx->Aq,ctx->Bq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_FEAST(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTSetNumPoints_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTGetNumPoints_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_FEAST(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscInt       n;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS FEAST Options");CHKERRQ(ierr);

    n = ctx->npoints;
    ierr = PetscOptionsInt("-eps_feast_num_points","Number of contour integration points","EPSFEASTSetNumPoints",n,&n,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSFEASTSetNumPoints(eps,n);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_FEAST(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  number of contour integration points=%D\n",ctx->npoints);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_FEAST(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSFEASTSetNumPoints_FEAST(EPS eps,PetscInt npoints)
{
  EPS_FEAST *ctx = (EPS_FEAST*)eps->data;

  PetscFunctionBegin;
  if (npoints == PETSC_DEFAULT) ctx->npoints = 8;
  else ctx->npoints = npoints;
  PetscFunctionReturn(0);
}

/*@
   EPSFEASTSetNumPoints - Sets the number of contour integration points for
   the FEAST package.

   Collective on EPS

   Input Parameters:
+  eps     - the eigenproblem solver context
-  npoints - number of contour integration points

   Options Database Key:
.  -eps_feast_num_points - Sets the number of points

   Level: advanced

.seealso: EPSFEASTGetNumPoints()
@*/
PetscErrorCode EPSFEASTSetNumPoints(EPS eps,PetscInt npoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,npoints,2);
  ierr = PetscTryMethod(eps,"EPSFEASTSetNumPoints_C",(EPS,PetscInt),(eps,npoints));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSFEASTGetNumPoints_FEAST(EPS eps,PetscInt *npoints)
{
  EPS_FEAST *ctx = (EPS_FEAST*)eps->data;

  PetscFunctionBegin;
  *npoints = ctx->npoints;
  PetscFunctionReturn(0);
}

/*@
   EPSFEASTGetNumPoints - Gets the number of contour integration points for
   the FEAST package.

   Collective on EPS

   Input Parameter:
.  eps     - the eigenproblem solver context

   Output Parameter:
-  npoints - number of contour integration points

   Level: advanced

.seealso: EPSFEASTSetNumPoints()
@*/
PetscErrorCode EPSFEASTGetNumPoints(EPS eps,PetscInt *npoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(npoints,2);
  ierr = PetscUseMethod(eps,"EPSFEASTGetNumPoints_C",(EPS,PetscInt*),(eps,npoints));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_FEAST(EPS eps)
{
  EPS_FEAST      *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->categ = EPS_CATEGORY_CONTOUR;

  eps->ops->solve          = EPSSolve_FEAST;
  eps->ops->setup          = EPSSetUp_FEAST;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->setfromoptions = EPSSetFromOptions_FEAST;
  eps->ops->destroy        = EPSDestroy_FEAST;
  eps->ops->reset          = EPSReset_FEAST;
  eps->ops->view           = EPSView_FEAST;
  eps->ops->setdefaultst   = EPSSetDefaultST_FEAST;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTSetNumPoints_C",EPSFEASTSetNumPoints_FEAST);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTGetNumPoints_C",EPSFEASTGetNumPoints_FEAST);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

