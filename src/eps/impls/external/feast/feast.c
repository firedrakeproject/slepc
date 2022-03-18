/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  PetscInt       ncv;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size));
  PetscCheck(size==1,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The FEAST interface is supported for sequential runs only");
  EPSCheckSinvertCayley(eps);
  if (eps->ncv!=PETSC_DEFAULT) {
    PetscCheck(eps->ncv>=eps->nev+2,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The value of ncv must be at least nev+2");
  } else eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),eps->n); /* set default value of ncv */
  if (eps->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(eps,"Warning: parameter mpd ignored\n"));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 20;
  if (!eps->which) eps->which = EPS_ALL;
  PetscCheck(eps->which==EPS_ALL && eps->inta!=eps->intb,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver must be used with a computational interval");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_CONVERGENCE | EPS_FEATURE_STOPPING | EPS_FEATURE_TWOSIDED);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION);

  if (!ctx->npoints) ctx->npoints = 8;

  ncv = eps->ncv;
  CHKERRQ(PetscFree4(ctx->work1,ctx->work2,ctx->Aq,ctx->Bq));
  CHKERRQ(PetscMalloc4(eps->nloc*ncv,&ctx->work1,eps->nloc*ncv,&ctx->work2,ncv*ncv,&ctx->Aq,ncv*ncv,&ctx->Bq));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,(2*eps->nloc*ncv+2*ncv*ncv)*sizeof(PetscScalar)));

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(EPSSetWorkVecs(eps,2));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_FEAST(EPS eps)
{
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

  CHKERRQ(PetscMalloc1(eps->ncv,&evals));
  CHKERRQ(BVGetArray(eps->V,&pV));

  ijob = -1;           /* first call to reverse communication interface */
  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  if (nmat>1) CHKERRQ(STGetMatrix(eps->st,1,&B));
  else B = NULL;
  CHKERRQ(MatCreateVecsEmpty(A,&x,&y));

  do {

    FEAST_RCI(&ijob,&n,&Ze,SCALAR_CAST ctx->work1,ctx->work2,SCALAR_CAST ctx->Aq,SCALAR_CAST ctx->Bq,fpm,&epsout,&loop,&eps->inta,&eps->intb,&ncv,evals,SCALAR_CAST pV,&nconv,eps->errest,&info);

    PetscCheck(ncv==eps->ncv,PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"FEAST changed value of ncv to %d",ncv);
    if (ijob == 10) {
      /* set new quadrature point */
      CHKERRQ(STSetShift(eps->st,Ze.real));
    } else if (ijob == 20) {
      /* use same quadrature point and factorization for transpose solve */
    } else if (ijob == 11 || ijob == 21) {
      /* linear solve (A-sigma*B)\work2, overwrite work2 */
      for (k=0;k<ncv;k++) {
        CHKERRQ(VecGetArray(z,&pz));
#if defined(PETSC_USE_COMPLEX)
        for (i=0;i<eps->nloc;i++) pz[i] = PetscCMPLX(ctx->work2[eps->nloc*k+i].real,ctx->work2[eps->nloc*k+i].imag);
#else
        for (i=0;i<eps->nloc;i++) pz[i] = ctx->work2[eps->nloc*k+i].real;
#endif
        CHKERRQ(VecRestoreArray(z,&pz));
        if (ijob == 11) {
          CHKERRQ(STMatSolve(eps->st,z,w));
        } else {
          CHKERRQ(VecConjugate(z));
          CHKERRQ(STMatSolveTranspose(eps->st,z,w));
          CHKERRQ(VecConjugate(w));
        }
        CHKERRQ(VecGetArray(w,&pz));
#if defined(PETSC_USE_COMPLEX)
        for (i=0;i<eps->nloc;i++) {
          ctx->work2[eps->nloc*k+i].real = PetscRealPart(pz[i]);
          ctx->work2[eps->nloc*k+i].imag = PetscImaginaryPart(pz[i]);
        }
#else
        for (i=0;i<eps->nloc;i++) ctx->work2[eps->nloc*k+i].real = pz[i];
#endif
        CHKERRQ(VecRestoreArray(w,&pz));
      }
    } else if (ijob == 30 || ijob == 40) {
      /* multiplication A*V or B*V, result in work1 */
      for (k=fpm[23]-1;k<fpm[23]+fpm[24]-1;k++) {
        CHKERRQ(VecPlaceArray(x,&pV[k*eps->nloc]));
        CHKERRQ(VecPlaceArray(y,&ctx->work1[k*eps->nloc]));
        if (ijob == 30) {
          CHKERRQ(MatMult(A,x,y));
        } else if (nmat>1) {
          CHKERRQ(MatMult(B,x,y));
        } else {
          CHKERRQ(VecCopy(x,y));
        }
        CHKERRQ(VecResetArray(x));
        CHKERRQ(VecResetArray(y));
      }
    } else PetscCheck(ijob==0 || ijob==-2,PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Internal error in FEAST reverse communication interface (ijob=%d)",ijob);

  } while (ijob);

  eps->reason = EPS_CONVERGED_TOL;
  eps->its    = loop;
  eps->nconv  = nconv;
  if (info) {
    switch (info) {
      case 1:  /* No eigenvalue has been found in the proposed search interval */
        eps->nconv = 0;
        break;
      case 2:   /* FEAST did not converge "yet" */
        eps->reason = EPS_DIVERGED_ITS;
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by FEAST (%d)",info);
    }
  }

  for (i=0;i<eps->nconv;i++) eps->eigr[i] = evals[i];

  CHKERRQ(BVRestoreArray(eps->V,&pV));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFree(evals));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_FEAST(EPS eps)
{
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree4(ctx->work1,ctx->work2,ctx->Aq,ctx->Bq));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_FEAST(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTSetNumPoints_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTGetNumPoints_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_FEAST(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscInt       n;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS FEAST Options"));

    n = ctx->npoints;
    CHKERRQ(PetscOptionsInt("-eps_feast_num_points","Number of contour integration points","EPSFEASTSetNumPoints",n,&n,&flg));
    if (flg) CHKERRQ(EPSFEASTSetNumPoints(eps,n));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_FEAST(EPS eps,PetscViewer viewer)
{
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of contour integration points=%" PetscInt_FMT "\n",ctx->npoints));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_FEAST(EPS eps)
{
  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) {
    CHKERRQ(STSetType(eps->st,STSINVERT));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,npoints,2);
  CHKERRQ(PetscTryMethod(eps,"EPSFEASTSetNumPoints_C",(EPS,PetscInt),(eps,npoints)));
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
.  npoints - number of contour integration points

   Level: advanced

.seealso: EPSFEASTSetNumPoints()
@*/
PetscErrorCode EPSFEASTGetNumPoints(EPS eps,PetscInt *npoints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(npoints,2);
  CHKERRQ(PetscUseMethod(eps,"EPSFEASTGetNumPoints_C",(EPS,PetscInt*),(eps,npoints)));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_FEAST(EPS eps)
{
  EPS_FEAST      *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
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

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTSetNumPoints_C",EPSFEASTSetNumPoints_FEAST));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSFEASTGetNumPoints_C",EPSFEASTGetNumPoints_FEAST));
  PetscFunctionReturn(0);
}
