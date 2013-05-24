/*

   SLEPc eigensolver: "ciss"

   Method: Contour Integral Spectral Slicing

   Algorithm:

       Contour integral based on Sakurai-Sugiura method to construct a
       subspace, with various eigenpair extractions (Rayleigh-Ritz,
       explicit moment).

   Based on code contributed by Tetsuya Sakurai.

   References:

       [1] T. Sakurai and H. Sugiura, "A projection method for generalized
           eigenvalue problems", J. Comput. Appl. Math. 159:119-128, 2003.

       [2] T. Sakurai and H. Tadano, "CIRR: a Rayleigh-Ritz type method with
           contour integral for generalized eigenvalue problems", Hokkaido
           Math. J. 36:745-757, 2007.

   Last update: Jun 2013

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode EPSSolve_CISS(EPS);

typedef struct {
  PetscScalar center;     /* center of the region where to find eigenpairs (0.0) */
  PetscReal   radius;     /* radius of the region (1.0) */
  PetscReal   vscale;     /* vertical scale of the region (1.0) */
  PetscInt    N;          /* number of integration points (32) */
  PetscInt    L;          /* block size (4) */
  PetscInt    M;          /* moment degree (N/4 = 8) */
  PetscReal   delta;      /* threshold of singular value (1e-12) */
  PetscBool   useconj;    /* (false) */
  PetscInt    npart;      /* number of partitions of the matrix (1) */
  PetscInt    Lmax;       /* (256) */
} EPS_CISS;

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_CISS"
PetscErrorCode EPSSetUp_CISS(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       nmat;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"CISS only works for complex scalars");
#endif
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must be at least nev");
  }
  else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
  }
  else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
    else {
      eps->mpd = 500;
      eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
    }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
//  ierr = VecDuplicateVecs(eps->t,eps->mpd,&ctx->AV);CHKERRQ(ierr);
//  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
//  if (nmat>1) {
//    ierr = VecDuplicateVecs(eps->t,eps->mpd,&ctx->BV);CHKERRQ(ierr);
//  }
//  ierr = VecDuplicateVecs(eps->t,eps->mpd,&ctx->P);CHKERRQ(ierr);
//  ierr = VecDuplicateVecs(eps->t,eps->mpd,&ctx->G);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_CISS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_CISS"
PetscErrorCode EPSSolve_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,k,ld,off,nv,ncv = eps->ncv,kini,nmat;
  PetscScalar    *C,*Y;
  PetscReal      resnorm,norm;
  PetscBool      breakdown;
  Mat            A,B;
  Vec            w=eps->work[0];

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;
//  ierr = PetscMalloc(eps->mpd*sizeof(PetscScalar),&gamma);CHKERRQ(ierr);

  ierr = SlepcVecSetRandom(eps->V[0],eps->rand);CHKERRQ(ierr);

  k = 0;
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,0);CHKERRQ(ierr);

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    eps->nconv = k;
    eps->reason = EPS_DIVERGED_ITS;
  }

  /* truncate Schur decomposition and change the state to raw so that
     PSVectors() computes eigenvectors from scratch */
  //ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  //ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetRegion_CISS"
static PetscErrorCode EPSCISSSetRegion_CISS(EPS eps,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ctx->center = center;
  if (radius<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The radius argument must > 0.0");
  ctx->radius = radius;
  if (vscale<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The vscale argument must > 0.0");
  ctx->vscale = vscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetRegion"
/*@
   EPSCISSSetRegion - Sets the parameters defining the region where eigenvalues
   must be computed.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Options Database Keys:
+  -eps_ciss_center - Sets the center
.  -eps_ciss_radius - Sets the radius
-  -eps_ciss_vscale - Sets the vertical scale

   Level: advanced

.seealso: EPSCISSGetRegion()
@*/
PetscErrorCode EPSCISSSetRegion(EPS eps,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveScalar(eps,center,2);
  PetscValidLogicalCollectiveReal(eps,radius,3);
  PetscValidLogicalCollectiveReal(eps,vscale,4);
  ierr = PetscTryMethod(eps,"EPSCISSSetRegion_C",(EPS,PetscScalar,PetscReal,PetscReal),(eps,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetRegion_CISS"
static PetscErrorCode EPSCISSGetRegion_CISS(EPS eps,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (center) *center = ctx->center;
  if (radius) *radius = ctx->radius;
  if (vscale) *vscale = ctx->vscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetRegion"
/*@
   EPSCISSGetRegion - Gets the parameters that define the region where eigenvalues
   must be computed.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Level: advanced

.seealso: EPSCISSSetRegion()
@*/
PetscErrorCode EPSCISSGetRegion(EPS eps,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSCISSGetRegion_C",(EPS,PetscScalar*,PetscReal*,PetscReal*),(eps,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_CISS"
PetscErrorCode EPSReset_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
//  ierr = VecDestroyVecs(eps->mpd,&ctx->AV);CHKERRQ(ierr);
//  ierr = VecDestroyVecs(eps->mpd,&ctx->BV);CHKERRQ(ierr);
//  ierr = VecDestroyVecs(eps->mpd,&ctx->P);CHKERRQ(ierr);
//  ierr = VecDestroyVecs(eps->mpd,&ctx->G);CHKERRQ(ierr);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_CISS"
PetscErrorCode EPSSetFromOptions_CISS(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscScalar    s;
  PetscReal      r1,r2;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS CISS Options");CHKERRQ(ierr);
  ierr = EPSCISSGetRegion(eps,&s,&r1,&r2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_ciss_radius","CISS radius of region","EPSCISSSetRegion",r1,&r1,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-eps_ciss_center","CISS center of region","EPSCISSSetRegion",s,&s,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_ciss_vscale","CISS vertical scale of region","EPSCISSSetRegion",r2,&r2,&flg);CHKERRQ(ierr);
  ierr = EPSCISSSetRegion(eps,s,r1,r2);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_CISS"
PetscErrorCode EPSDestroy_CISS(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRegion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRegion_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_CISS"
PetscErrorCode EPSView_CISS(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->center,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: region { center: %s, radius: %g, vscale: %g }\n",str,ctx->radius,ctx->vscale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_CISS"
PETSC_EXTERN PetscErrorCode EPSCreate_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_CISS,&ctx);CHKERRQ(ierr);
  eps->data = ctx;
  eps->ops->setup          = EPSSetUp_CISS;
  eps->ops->setfromoptions = EPSSetFromOptions_CISS;
  eps->ops->destroy        = EPSDestroy_CISS;
  eps->ops->reset          = EPSReset_CISS;
  eps->ops->view           = EPSView_CISS;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Default;
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRegion_C",EPSCISSSetRegion_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRegion_C",EPSCISSGetRegion_CISS);CHKERRQ(ierr);
  /* set default values of parameters */
  if (!ctx->radius) ctx->radius = 1.0;
  if (!ctx->vscale) ctx->vscale = 1.0;
  if (!ctx->N)      ctx->N      = 32;
  if (!ctx->L)      ctx->L      = 4;
  if (!ctx->M)      ctx->M      = ctx->N/4;
  if (!ctx->delta)  ctx->delta  = 1e-12;
  if (!ctx->npart)  ctx->npart  = 1;
  if (!ctx->Lmax)   ctx->Lmax   = 256;
  PetscFunctionReturn(0);
}

