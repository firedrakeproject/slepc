/*

   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi

   Algorithm:

       Arnoldi method with explicit restart and deflation.

   References:

       [1] "Arnoldi Methods in SLEPc", SLEPc Technical Report STR-4,
           available at http://slepc.upv.es.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

PetscErrorCode EPSSolve_Arnoldi(EPS);

typedef struct {
  PetscBool delayed;
} EPS_ARNOLDI;

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_Arnoldi"
PetscErrorCode EPSSetUp_Arnoldi(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  if (eps->ncv>eps->nev+eps->mpd) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must not be larger than nev+mpd");
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY)) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  }
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  ierr = EPSAllocateSolution(eps,1);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
  if (eps->extraction==EPS_REFINED || eps->extraction==EPS_REFINED_HARMONIC) {
    ierr = DSSetRefined(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = DSSetExtraRow(eps->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->ncv+1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->isgeneralized && eps->ishermitian && !eps->ispositive) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Requested method does not work for indefinite problems");
  eps->ops->solve = EPSSolve_Arnoldi;
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "EPSDelayedArnoldi"
/*
   EPSDelayedArnoldi - This function is equivalent to EPSBasicArnoldi but
   performs the computation in a different way. The main idea is that
   reorthogonalization is delayed to the next Arnoldi step. This version is
   more scalable but in some cases convergence may stagnate.
*/
PetscErrorCode EPSDelayedArnoldi(EPS eps,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M;
  Vec            u,t;
  PetscScalar    shh[100],*lhh,dot,dot2;
  PetscReal      norm1=0.0,norm2;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else {
    ierr = PetscMalloc1(m,&lhh);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&t);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->st,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,0,NULL,eps->nds,NULL,eps->defl,f,NULL,NULL,NULL);CHKERRQ(ierr);

    ierr = IPMInnerProductBegin(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = IPMInnerProductBegin(eps->ip,V[j],j,V,lhh);CHKERRQ(ierr);
      ierr = IPInnerProductBegin(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr);
    }
    if (j>k+1) {
      ierr = IPNormBegin(eps->ip,u,&norm2);CHKERRQ(ierr);
      ierr = VecDotBegin(u,V[j-2],&dot2);CHKERRQ(ierr);
    }

    ierr = IPMInnerProductEnd(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = IPMInnerProductEnd(eps->ip,V[j],j,V,lhh);CHKERRQ(ierr);
      ierr = IPInnerProductEnd(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr);
    }
    if (j>k+1) {
      ierr = IPNormEnd(eps->ip,u,&norm2);CHKERRQ(ierr);
      ierr = VecDotEnd(u,V[j-2],&dot2);CHKERRQ(ierr);
      if (PetscAbsScalar(dot2/norm2) > PETSC_MACHINE_EPSILON) {
        *breakdown = PETSC_TRUE;
        *M = j-1;
        *beta = norm2;

        if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
        ierr = VecDestroy(&u);CHKERRQ(ierr);
        ierr = VecDestroy(&t);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }

    if (j>k) {
      norm1 = PetscSqrtReal(PetscRealPart(dot));
      for (i=0;i<j;i++)
        H[ldh*j+i] = H[ldh*j+i]/norm1;
      H[ldh*j+j] = H[ldh*j+j]/dot;

      ierr = VecCopy(V[j],t);CHKERRQ(ierr);
      ierr = VecScale(V[j],1.0/norm1);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm1);CHKERRQ(ierr);
    }

    ierr = SlepcVecMAXPBY(f,1.0,-1.0,j+1,H+ldh*j,V);CHKERRQ(ierr);

    if (j>k) {
      ierr = SlepcVecMAXPBY(t,1.0,-1.0,j,lhh,V);CHKERRQ(ierr);
      for (i=0;i<j;i++)
        H[ldh*(j-1)+i] += lhh[i];
    }

    if (j>k+1) {
      ierr = VecCopy(u,V[j-1]);CHKERRQ(ierr);
      ierr = VecScale(V[j-1],1.0/norm2);CHKERRQ(ierr);
      H[ldh*(j-2)+j-1] = norm2;
    }

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
      ierr = VecCopy(t,u);CHKERRQ(ierr);
    }
  }

  ierr = IPNorm(eps->ip,t,&norm2);CHKERRQ(ierr);
  ierr = VecScale(t,1.0/norm2);CHKERRQ(ierr);
  ierr = VecCopy(t,V[m-1]);CHKERRQ(ierr);
  H[ldh*(m-2)+m-1] = norm2;

  ierr = IPMInnerProduct(eps->ip,f,m,V,lhh);CHKERRQ(ierr);

  ierr = SlepcVecMAXPBY(f,1.0,-1.0,m,lhh,V);CHKERRQ(ierr);
  for (i=0;i<m;i++)
    H[ldh*(m-1)+i] += lhh[i];

  ierr = IPNorm(eps->ip,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDelayedArnoldi1"
/*
   EPSDelayedArnoldi1 - This function is similar to EPSDelayedArnoldi,
   but without reorthogonalization (only delayed normalization).
*/
PetscErrorCode EPSDelayedArnoldi1(EPS eps,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M;
  PetscScalar    dot;
  PetscReal      norm=0.0;

  PetscFunctionBegin;
  for (j=k;j<m;j++) {
    ierr = STApply(eps->st,V[j],f);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,0,NULL,eps->nds,NULL,eps->defl,f,NULL,NULL,NULL);CHKERRQ(ierr);

    ierr = IPMInnerProductBegin(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = IPInnerProductBegin(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr);
    }

    ierr = IPMInnerProductEnd(eps->ip,f,j+1,V,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = IPInnerProductEnd(eps->ip,V[j],V[j],&dot);CHKERRQ(ierr);
    }

    if (j>k) {
      norm = PetscSqrtReal(PetscRealPart(dot));
      ierr = VecScale(V[j],1.0/norm);CHKERRQ(ierr);
      H[ldh*(j-1)+j] = norm;

      for (i=0;i<j;i++)
        H[ldh*j+i] = H[ldh*j+i]/norm;
      H[ldh*j+j] = H[ldh*j+j]/dot;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
    }

    ierr = SlepcVecMAXPBY(f,1.0,-1.0,j+1,H+ldh*j,V);CHKERRQ(ierr);

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }

  ierr = IPNorm(eps->ip,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_Arnoldi"
PetscErrorCode EPSSolve_Arnoldi(EPS eps)
{
  PetscErrorCode     ierr;
  PetscInt           k,nv,ld;
  Mat                U;
  PetscScalar        *H,*X;
  PetscReal          beta,gamma=1.0;
  PetscBool          breakdown,harmonic,refined;
  BVOrthogRefineType orthog_ref;
  EPS_ARNOLDI        *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetRefined(eps->ds,&refined);CHKERRQ(ierr);
  harmonic = (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  ierr = BVGetOrthogonalization(eps->V,NULL,&orthog_ref,NULL);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,0);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    if (!arnoldi->delayed) {
      ierr = EPSBasicArnoldi(eps,PETSC_FALSE,H,ld,eps->nconv,&nv,&beta,&breakdown);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)eps),1,"Not implemented");
    /*if (orthog_ref == BV_ORTHOG_REFINE_NEVER) {
      ierr = EPSDelayedArnoldi1(eps,H,ld,eps->V,eps->nconv,&nv,f,&beta,&breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSDelayedArnoldi(eps,H,ld,eps->V,eps->nconv,&nv,f,&beta,&breakdown);CHKERRQ(ierr);
    }*/
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,eps->nconv,nv);CHKERRQ(ierr);

    /* Compute translation of Krylov decomposition if harmonic extraction used */
    if (harmonic) {
      ierr = DSTranslateHarmonic(eps->ds,eps->target,beta,PETSC_FALSE,NULL,&gamma);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(eps->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta,gamma,&k);CHKERRQ(ierr);
    if (refined) {
      ierr = DSGetArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultColumn(eps->V,1.0,0.0,k,X+k*ld);CHKERRQ(ierr);
      ierr = DSRestoreArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
      ierr = BVMultInPlace(eps->V,U,eps->nconv,nv);CHKERRQ(ierr);
      ierr = MatDestroy(&U);CHKERRQ(ierr);
      ierr = BVOrthogonalizeColumn(eps->V,k,NULL,NULL,NULL);CHKERRQ(ierr);
    } else {
      ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
      ierr = BVMultInPlace(eps->V,U,eps->nconv,nv);CHKERRQ(ierr);
      ierr = MatDestroy(&U);CHKERRQ(ierr);
    }
    eps->nconv = k;

    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    if (breakdown && k<eps->nev) {
      ierr = PetscInfo2(eps,"Breakdown in Arnoldi method (it=%D norm=%g)\n",eps->its,(double)beta);CHKERRQ(ierr);
      ierr = EPSGetStartVector(eps,k,&breakdown);CHKERRQ(ierr);
      if (breakdown) {
        eps->reason = EPS_DIVERGED_BREAKDOWN;
        ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
      }
    }
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }

  /* truncate Schur decomposition and change the state to raw so that
     PSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_Arnoldi"
PetscErrorCode EPSSetFromOptions_Arnoldi(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS Arnoldi Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_arnoldi_delayed","Arnoldi with delayed reorthogonalization","EPSArnoldiSetDelayed",arnoldi->delayed,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = EPSArnoldiSetDelayed(eps,val);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSArnoldiSetDelayed_Arnoldi"
static PetscErrorCode EPSArnoldiSetDelayed_Arnoldi(EPS eps,PetscBool delayed)
{
  EPS_ARNOLDI *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  arnoldi->delayed = delayed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSArnoldiSetDelayed"
/*@
   EPSArnoldiSetDelayed - Activates or deactivates delayed reorthogonalization
   in the Arnoldi iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  delayed - boolean flag

   Options Database Key:
.  -eps_arnoldi_delayed - Activates delayed reorthogonalization in Arnoldi

   Note:
   Delayed reorthogonalization is an aggressive optimization for the Arnoldi
   eigensolver than may provide better scalability, but sometimes makes the
   solver converge less than the default algorithm.

   Level: advanced

.seealso: EPSArnoldiGetDelayed()
@*/
PetscErrorCode EPSArnoldiSetDelayed(EPS eps,PetscBool delayed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,delayed,2);
  ierr = PetscTryMethod(eps,"EPSArnoldiSetDelayed_C",(EPS,PetscBool),(eps,delayed));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSArnoldiGetDelayed_Arnoldi"
static PetscErrorCode EPSArnoldiGetDelayed_Arnoldi(EPS eps,PetscBool *delayed)
{
  EPS_ARNOLDI *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  *delayed = arnoldi->delayed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSArnoldiGetDelayed"
/*@C
   EPSArnoldiGetDelayed - Gets the type of reorthogonalization used during the Arnoldi
   iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  delayed - boolean flag indicating if delayed reorthogonalization has been enabled

   Level: advanced

.seealso: EPSArnoldiSetDelayed()
@*/
PetscErrorCode EPSArnoldiGetDelayed(EPS eps,PetscBool *delayed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(delayed,2);
  ierr = PetscTryMethod(eps,"EPSArnoldiGetDelayed_C",(EPS,PetscBool*),(eps,delayed));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_Arnoldi"
PetscErrorCode EPSDestroy_Arnoldi(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiSetDelayed_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiGetDelayed_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_Arnoldi"
PetscErrorCode EPSView_Arnoldi(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI*)eps->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (arnoldi->delayed) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Arnoldi: using delayed reorthogonalization\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_Arnoldi"
PETSC_EXTERN PetscErrorCode EPSCreate_Arnoldi(EPS eps)
{
  EPS_ARNOLDI    *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->setup                = EPSSetUp_Arnoldi;
  eps->ops->setfromoptions       = EPSSetFromOptions_Arnoldi;
  eps->ops->destroy              = EPSDestroy_Arnoldi;
  eps->ops->view                 = EPSView_Arnoldi;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiSetDelayed_C",EPSArnoldiSetDelayed_Arnoldi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSArnoldiGetDelayed_C",EPSArnoldiGetDelayed_Arnoldi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

