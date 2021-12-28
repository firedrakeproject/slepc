/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "ciss"

   Method: Contour Integral Spectral Slicing

   Algorithm:

       Contour integral based on Sakurai-Sugiura method to construct a
       subspace, with various eigenpair extractions (Rayleigh-Ritz,
       explicit moment).

   Based on code contributed by Y. Maeda, T. Sakurai.

   References:

       [1] T. Sakurai and H. Sugiura, "A projection method for generalized
           eigenvalue problems", J. Comput. Appl. Math. 159:119-128, 2003.

       [2] T. Sakurai and H. Tadano, "CIRR: a Rayleigh-Ritz type method with
           contour integral for generalized eigenvalue problems", Hokkaido
           Math. J. 36:745-757, 2007.
*/

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepc/private/slepccontour.h>
#include <slepcblaslapack.h>

typedef struct {
  /* user parameters */
  PetscInt          N;          /* number of integration points (32) */
  PetscInt          L;          /* block size (16) */
  PetscInt          M;          /* moment degree (N/4 = 4) */
  PetscReal         delta;      /* threshold of singular value (1e-12) */
  PetscInt          L_max;      /* maximum number of columns of the source matrix V */
  PetscReal         spurious_threshold; /* discard spurious eigenpairs */
  PetscBool         isreal;     /* A and B are real */
  PetscInt          npart;      /* number of partitions */
  PetscInt          refine_inner;
  PetscInt          refine_blocksize;
  EPSCISSQuadRule   quad;
  EPSCISSExtraction extraction;
  PetscBool         usest;
  /* private data */
  SlepcContourData  contour;
  PetscReal         *sigma;     /* threshold for numerical rank */
  PetscScalar       *weight;
  PetscScalar       *omega;
  PetscScalar       *pp;
  BV                V;
  BV                S;
  BV                pV;
  BV                Y;
  PetscBool         useconj;
  PetscBool         usest_set;  /* whether the user set the usest flag or not */
  PetscObjectId     rgid;
  PetscObjectState  rgstate;
} EPS_CISS;

static PetscErrorCode EPSCISSSolveSystem(EPS eps,Mat A,Mat B,BV V,PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode   ierr;
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscInt         i,p_id;
  Mat              Fz,kspMat,MV,BMV=NULL,MC;
  KSP              ksp;
  const char       *prefix;

  PetscFunctionBegin;
  if (!ctx->contour || !ctx->contour->ksp) { ierr = EPSCISSGetKSPs(eps,NULL,NULL);CHKERRQ(ierr); }
  contour = ctx->contour;
  if (ctx->usest) {
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Fz);CHKERRQ(ierr);
  }
  ierr = BVSetActiveColumns(V,L_start,L_end);CHKERRQ(ierr);
  ierr = BVGetMat(V,&MV);CHKERRQ(ierr);
  if (B) {
    ierr = MatProductCreate(B,MV,NULL,&BMV);CHKERRQ(ierr);
    ierr = MatProductSetType(BMV,MATPRODUCT_AB);CHKERRQ(ierr);
    ierr = MatProductSetFromOptions(BMV);CHKERRQ(ierr);
    ierr = MatProductSymbolic(BMV);CHKERRQ(ierr);
  }
  for (i=0;i<contour->npoints;i++) {
    p_id = i*contour->subcomm->n + contour->subcomm->color;
    if (!ctx->usest && initksp) {
      ierr = MatDuplicate(A,MAT_COPY_VALUES,&kspMat);CHKERRQ(ierr);
      if (B) {
        ierr = MatAXPY(kspMat,-ctx->omega[p_id],B,UNKNOWN_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatShift(kspMat,-ctx->omega[p_id]);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(contour->ksp[i],kspMat,kspMat);CHKERRQ(ierr);
      /* set Mat prefix to be the same as KSP to enable setting command-line options (e.g. MUMPS) */
      ierr = KSPGetOptionsPrefix(contour->ksp[i],&prefix);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(kspMat,prefix);CHKERRQ(ierr);
      ierr = MatDestroy(&kspMat);CHKERRQ(ierr);
    } else if (ctx->usest) {
      ierr = STSetShift(eps->st,ctx->omega[p_id]);CHKERRQ(ierr);
      ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(ctx->Y,i*ctx->L_max+L_start,i*ctx->L_max+L_end);CHKERRQ(ierr);
    ierr = BVGetMat(ctx->Y,&MC);CHKERRQ(ierr);
    if (B) {
      ierr = MatProductNumeric(BMV);CHKERRQ(ierr);
      if (ctx->usest) {
        ierr = KSPMatSolve(ksp,BMV,MC);CHKERRQ(ierr);
      } else {
        ierr = KSPMatSolve(contour->ksp[i],BMV,MC);CHKERRQ(ierr);
      }
    } else {
      if (ctx->usest) {
        ierr = KSPMatSolve(ksp,MV,MC);CHKERRQ(ierr);
      } else {
        ierr = KSPMatSolve(contour->ksp[i],MV,MC);CHKERRQ(ierr);
      }
    }
    if (ctx->usest && i<contour->npoints-1) { ierr = KSPReset(ksp);CHKERRQ(ierr); }
    ierr = BVRestoreMat(ctx->Y,&MC);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&BMV);CHKERRQ(ierr);
  ierr = BVRestoreMat(V,&MV);CHKERRQ(ierr);
  if (ctx->usest) { ierr = MatDestroy(&Fz);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode rescale_eig(EPS eps,PetscInt nv)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i;
  PetscScalar    center;
  PetscReal      radius,a,b,c,d,rgscale;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      start_ang,end_ang,vscale,theta;
#endif
  PetscBool      isring,isellipse,isinterval;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGRING,&isring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGINTERVAL,&isinterval);CHKERRQ(ierr);
  ierr = RGGetScale(eps->rg,&rgscale);CHKERRQ(ierr);
  if (isinterval) {
    ierr = RGIntervalGetEndpoints(eps->rg,NULL,NULL,&c,&d);CHKERRQ(ierr);
    if (c==d) {
      for (i=0;i<nv;i++) {
#if defined(PETSC_USE_COMPLEX)
        eps->eigr[i] = PetscRealPart(eps->eigr[i]);
#else
        eps->eigi[i] = 0;
#endif
      }
    }
  }
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
    if (isellipse) {
      ierr = RGEllipseGetParameters(eps->rg,&center,&radius,NULL);CHKERRQ(ierr);
      for (i=0;i<nv;i++) eps->eigr[i] = rgscale*(center + radius*eps->eigr[i]);
    } else if (isinterval) {
      ierr = RGIntervalGetEndpoints(eps->rg,&a,&b,&c,&d);CHKERRQ(ierr);
      if (ctx->quad == EPS_CISS_QUADRULE_CHEBYSHEV) {
        for (i=0;i<nv;i++) {
          if (c==d) eps->eigr[i] = ((b-a)*(eps->eigr[i]+1.0)/2.0+a)*rgscale;
          if (a==b) {
#if defined(PETSC_USE_COMPLEX)
            eps->eigr[i] = ((d-c)*(eps->eigr[i]+1.0)/2.0+c)*rgscale*PETSC_i;
#else
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Integration points on a vertical line require complex arithmetic");
#endif
          }
        }
      } else {
        center = (b+a)/2.0+(d+c)/2.0*PETSC_PI;
        radius = PetscSqrtReal(PetscPowRealInt((b-a)/2.0,2)+PetscPowRealInt((d-c)/2.0,2));
        for (i=0;i<nv;i++) eps->eigr[i] = center + radius*eps->eigr[i];
      }
    } else if (isring) {  /* only supported in complex scalars */
#if defined(PETSC_USE_COMPLEX)
      ierr = RGRingGetParameters(eps->rg,&center,&radius,&vscale,&start_ang,&end_ang,NULL);CHKERRQ(ierr);
      if (ctx->quad == EPS_CISS_QUADRULE_CHEBYSHEV) {
        for (i=0;i<nv;i++) {
          theta = (start_ang*2.0+(end_ang-start_ang)*(PetscRealPart(eps->eigr[i])+1.0))*PETSC_PI;
          eps->eigr[i] = rgscale*center + (rgscale*radius+PetscImaginaryPart(eps->eigr[i]))*PetscCMPLX(PetscCosReal(theta),vscale*PetscSinReal(theta));
        }
      } else {
        for (i=0;i<nv;i++) eps->eigr[i] = rgscale*(center + radius*eps->eigr[i]);
      }
#endif
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUp_CISS(EPS eps)
{
  PetscErrorCode   ierr;
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscBool        istrivial,isring,isellipse,isinterval,flg;
  PetscReal        c,d;
  PetscRandom      rand;
  PetscObjectId    id;
  PetscObjectState state;
  Mat              A[2];
  Vec              v0;

  PetscFunctionBegin;
  if (eps->ncv==PETSC_DEFAULT) {
    eps->ncv = ctx->L_max*ctx->M;
    if (eps->ncv>eps->n) {
      eps->ncv = eps->n;
      ctx->L_max = eps->ncv/ctx->M;
      if (!ctx->L_max) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Cannot adjust solver parameters, try setting a smaller value of M (moment size)");
    }
  } else {
    ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
    ctx->L_max = eps->ncv/ctx->M;
    if (!ctx->L_max) {
      ctx->L_max = 1;
      eps->ncv = ctx->L_max*ctx->M;
    }
  }
  ctx->L = PetscMin(ctx->L,ctx->L_max);
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 5;
  if (eps->mpd==PETSC_DEFAULT) eps->mpd = eps->ncv;
  if (!eps->which) eps->which = EPS_ALL;
  if (eps->which!=EPS_ALL) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only computing all eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_EXTRACTION | EPS_FEATURE_STOPPING | EPS_FEATURE_TWOSIDED);

  /* check region */
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"CISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  ierr = RGGetComplement(eps->rg,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"A region with complement flag set is not allowed");
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGRING,&isring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGINTERVAL,&isinterval);CHKERRQ(ierr);
  if (!isellipse && !isring && !isinterval) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Currently only implemented for interval, elliptic or ring regions");

  /* if the region has changed, then reset contour data */
  ierr = PetscObjectGetId((PetscObject)eps->rg,&id);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)eps->rg,&state);CHKERRQ(ierr);
  if (ctx->rgid && (id != ctx->rgid || state != ctx->rgstate)) {
    ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
    ierr = PetscInfo(eps,"Resetting the contour data structure due to a change of region\n");CHKERRQ(ierr);
    ctx->rgid = id; ctx->rgstate = state;
  }

#if !defined(PETSC_USE_COMPLEX)
  if (isring) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Ring region only supported for complex scalars");
#endif
  if (isinterval) {
    ierr = RGIntervalGetEndpoints(eps->rg,NULL,NULL,&c,&d);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (c!=d || c!=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"In real scalars, endpoints of the imaginary axis must be both zero");
#endif
    if (!ctx->quad && c==d) ctx->quad = EPS_CISS_QUADRULE_CHEBYSHEV;
  }
  if (!ctx->quad) ctx->quad = EPS_CISS_QUADRULE_TRAPEZOIDAL;

  /* create contour data structure */
  if (!ctx->contour) {
    ierr = RGCanUseConjugates(eps->rg,ctx->isreal,&ctx->useconj);CHKERRQ(ierr);
    ierr = SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)eps,&ctx->contour);CHKERRQ(ierr);
  }

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = BVGetRandomContext(eps->V,&rand);CHKERRQ(ierr);  /* make sure the random context is available when duplicating */
  if (ctx->weight) { ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr); }
  ierr = PetscMalloc4(ctx->N,&ctx->weight,ctx->N+1,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* allocate basis vectors */
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->L_max*ctx->M,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->L_max,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->V);CHKERRQ(ierr);

  ierr = STGetMatrix(eps->st,0,&A[0]);CHKERRQ(ierr);
  ierr = MatIsShell(A[0],&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Matrix type shell is not supported in this solver");
  if (eps->isgeneralized) { ierr = STGetMatrix(eps->st,1,&A[1]);CHKERRQ(ierr); }

  if (!ctx->usest_set) ctx->usest = (ctx->npart>1)? PETSC_FALSE: PETSC_TRUE;
  if (ctx->usest && ctx->npart>1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The usest flag is not supported when partitions > 1");

  contour = ctx->contour;
  ierr = SlepcContourRedundantMat(contour,eps->isgeneralized?2:1,A);CHKERRQ(ierr);
  if (contour->pA) {
    ierr = BVGetColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = SlepcContourScatterCreate(contour,v0);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
    ierr = BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->pV);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->pV,contour->xsub,eps->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->pV);CHKERRQ(ierr);
    ierr = BVResize(ctx->pV,ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->pV);CHKERRQ(ierr);
  }

  EPSCheckDefinite(eps);
  EPSCheckSinvertCondition(eps,ctx->usest," (with the usest flag set)");

  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  if (contour->pA) {
    ierr = BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->Y);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->Y,contour->xsub,eps->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->Y);CHKERRQ(ierr);
    ierr = BVResize(ctx->Y,contour->npoints*ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicateResize(eps->V,contour->npoints*ctx->L_max,&ctx->Y);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->Y);CHKERRQ(ierr);

  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
    ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
  } else if (eps->isgeneralized) {
    if (eps->ishermitian && eps->ispositive) {
      ierr = DSSetType(eps->ds,DSGHEP);CHKERRQ(ierr);
    } else {
      ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
    }
  } else {
    if (eps->ishermitian) {
      ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
    } else {
      ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
    }
  }
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  ierr = EPSSetWorkVecs(eps,3);CHKERRQ(ierr);
  if (!eps->ishermitian) { ierr = PetscInfo(eps,"Warning: complex eigenvalues are not calculated exactly without --with-scalar-type=complex in PETSc\n");CHKERRQ(ierr); }
#else
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUpSort_CISS(EPS eps)
{
  PetscErrorCode ierr;
  SlepcSC        sc;

  PetscFunctionBegin;
  /* fill sorting criterion context */
  eps->sc->comparison    = SlepcCompareSmallestReal;
  eps->sc->comparisonctx = NULL;
  eps->sc->map           = NULL;
  eps->sc->mapobj        = NULL;

  /* fill sorting criterion for DS */
  ierr = DSGetSlepcSC(eps->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_CISS(EPS eps)
{
  PetscErrorCode   ierr;
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour = ctx->contour;
  Mat              A,B,X,M,pA,pB;
  PetscInt         i,j,ld,nmat,L_add=0,nv=0,L_base=ctx->L,inner,nlocal,*inside;
  PetscScalar      *Mu,*H0,*H1=NULL,*rr,*temp;
  PetscReal        error,max_error,norm;
  PetscBool        *fl1;
  Vec              si,si1=NULL,w[3];
  PetscRandom      rand;
#if defined(PETSC_USE_COMPLEX)
  PetscBool        isellipse;
  PetscReal        est_eig,eta;
#else
  PetscReal        normi;
#endif

  PetscFunctionBegin;
  w[0] = eps->work[0];
#if defined(PETSC_USE_COMPLEX)
  w[1] = NULL;
#else
  w[1] = eps->work[2];
#endif
  w[2] = eps->work[1];
  ierr = VecGetLocalSize(w[0],&nlocal);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetMatrix(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;
  ierr = RGComputeQuadrature(eps->rg,ctx->quad==EPS_CISS_QUADRULE_CHEBYSHEV?RG_QUADRULE_CHEBYSHEV:RG_QUADRULE_TRAPEZOIDAL,ctx->N,ctx->omega,ctx->pp,ctx->weight);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
  ierr = BVGetRandomContext(ctx->V,&rand);CHKERRQ(ierr);

  if (contour->pA) {
    ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
    ierr = EPSCISSSolveSystem(eps,contour->pA[0],contour->pA[1],ctx->pV,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = EPSCISSSolveSystem(eps,A,B,ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = BVTraceQuadrature(ctx->Y,ctx->V,ctx->L,ctx->L_max,ctx->weight,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj,&est_eig);CHKERRQ(ierr);
    ierr = PetscInfo1(eps,"Estimated eigenvalue count: %f\n",(double)est_eig);CHKERRQ(ierr);
    eta = PetscPowReal(10.0,-PetscLog10Real(eps->tol)/ctx->N);
    L_add = PetscMax(0,(PetscInt)PetscCeilReal((est_eig*eta)/ctx->M)-ctx->L);
    if (L_add>ctx->L_max-ctx->L) {
      ierr = PetscInfo(eps,"Number of eigenvalues inside the contour path may be too large\n");CHKERRQ(ierr);
      L_add = ctx->L_max-ctx->L;
    }
  }
#endif
  if (L_add>0) {
    ierr = PetscInfo2(eps,"Changing L %D -> %D by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (contour->pA) {
      ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
      ierr = EPSCISSSolveSystem(eps,contour->pA[0],contour->pA[1],ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = EPSCISSSolveSystem(eps,A,B,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
  }
  ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  for (i=0;i<ctx->refine_blocksize;i++) {
    ierr = BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
    ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(EPS_CISS_SVD,eps,0,0,0);CHKERRQ(ierr);
    ierr = SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(EPS_CISS_SVD,eps,0,0,0);CHKERRQ(ierr);
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    ierr = PetscInfo2(eps,"Changing L %D -> %D by SVD(H0)\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (contour->pA) {
      ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
      ierr = EPSCISSSolveSystem(eps,contour->pA[0],contour->pA[1],ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = EPSCISSSolveSystem(eps,A,B,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
    if (L_add) {
      ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
      ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
    }
  }
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
    ierr = PetscMalloc1(ctx->L*ctx->M*ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  }

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
        ierr = BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(EPS_CISS_SVD,eps,0,0,0);CHKERRQ(ierr);
        ierr = SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(EPS_CISS_SVD,eps,0,0,0);CHKERRQ(ierr);
        break;
      } else {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        ierr = BVSVDAndRank(ctx->S,ctx->M,ctx->L,ctx->delta,BV_SVD_METHOD_REFINE,H0,ctx->sigma,&nv);CHKERRQ(ierr);
        if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
          if (contour->pA) {
            ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
            ierr = EPSCISSSolveSystem(eps,contour->pA[0],contour->pA[1],ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
          } else {
            ierr = EPSCISSSolveSystem(eps,A,B,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
          }
        } else break;
      }
    }
    eps->nconv = 0;
    if (nv == 0) eps->reason = EPS_CONVERGED_TOL;
    else {
      ierr = DSSetDimensions(eps->ds,nv,0,0);CHKERRQ(ierr);
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);

      if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
        ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
        ierr = CISS_BlockHankel(Mu,1,ctx->L,ctx->M,H1);CHKERRQ(ierr);
        ierr = DSGetArray(eps->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        for (j=0;j<nv;j++) {
          for (i=0;i<nv;i++) {
            temp[i+j*ld] = H1[i+j*ctx->L*ctx->M];
          }
        }
        ierr = DSRestoreArray(eps->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        ierr = DSGetArray(eps->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
        for (j=0;j<nv;j++) {
          for (i=0;i<nv;i++) {
            temp[i+j*ld] = H0[i+j*ctx->L*ctx->M];
          }
        }
        ierr = DSRestoreArray(eps->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
      } else {
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = DSGetMat(eps->ds,DS_MAT_A,&pA);CHKERRQ(ierr);
        ierr = MatZeroEntries(pA);CHKERRQ(ierr);
        ierr = BVMatProject(ctx->S,A,ctx->S,pA);CHKERRQ(ierr);
        ierr = DSRestoreMat(eps->ds,DS_MAT_A,&pA);CHKERRQ(ierr);
        if (B) {
          ierr = DSGetMat(eps->ds,DS_MAT_B,&pB);CHKERRQ(ierr);
          ierr = MatZeroEntries(pB);CHKERRQ(ierr);
          ierr = BVMatProject(ctx->S,B,ctx->S,pB);CHKERRQ(ierr);
          ierr = DSRestoreMat(eps->ds,DS_MAT_B,&pB);CHKERRQ(ierr);
        }
      }

      ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
      ierr = DSSynchronize(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);

      ierr = PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr);CHKERRQ(ierr);
      ierr = rescale_eig(eps,nv);CHKERRQ(ierr);
      ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
      ierr = DSGetMat(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = SlepcCISS_isGhost(X,nv,ctx->sigma,ctx->spurious_threshold,fl1);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      ierr = RGCheckInside(eps->rg,nv,eps->eigr,eps->eigi,inside);CHKERRQ(ierr);
      for (i=0;i<nv;i++) {
        if (fl1[i] && inside[i]>=0) {
          rr[i] = 1.0;
          eps->nconv++;
        } else rr[i] = 0.0;
      }
      ierr = DSSort(eps->ds,eps->eigr,eps->eigi,rr,NULL,&eps->nconv);CHKERRQ(ierr);
      ierr = DSSynchronize(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
      ierr = rescale_eig(eps,nv);CHKERRQ(ierr);
      ierr = PetscFree3(fl1,inside,rr);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
      if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
      }
      ierr = BVCopy(ctx->S,eps->V);CHKERRQ(ierr);

      ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
      ierr = DSGetMat(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(ctx->S,X,0,eps->nconv);CHKERRQ(ierr);
      if (eps->ishermitian) {
        ierr = BVMultInPlace(eps->V,X,0,eps->nconv);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      max_error = 0.0;
      for (i=0;i<eps->nconv;i++) {
        ierr = BVGetColumn(ctx->S,i,&si);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        if (eps->eigi[i]!=0.0) { ierr = BVGetColumn(ctx->S,i+1,&si1);CHKERRQ(ierr); }
#endif
        ierr = EPSComputeResidualNorm_Private(eps,PETSC_FALSE,eps->eigr[i],eps->eigi[i],si,si1,w,&error);CHKERRQ(ierr);
        if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {  /* vector is not normalized */
          ierr = VecNorm(si,NORM_2,&norm);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
          if (eps->eigi[i]!=0.0) {
            ierr = VecNorm(si1,NORM_2,&normi);CHKERRQ(ierr);
            norm = SlepcAbsEigenvalue(norm,normi);
          }
#endif
          error /= norm;
        }
        ierr = (*eps->converged)(eps,eps->eigr[i],eps->eigi[i],error,&error,eps->convergedctx);CHKERRQ(ierr);
        ierr = BVRestoreColumn(ctx->S,i,&si);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        if (eps->eigi[i]!=0.0) {
          ierr = BVRestoreColumn(ctx->S,i+1,&si1);CHKERRQ(ierr);
          i++;
        }
#endif
        max_error = PetscMax(max_error,error);
      }

      if (max_error <= eps->tol) eps->reason = EPS_CONVERGED_TOL;
      else if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
      else {
        if (eps->nconv > ctx->L) nv = eps->nconv;
        else if (ctx->L > nv) nv = ctx->L;
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,nv,ctx->L,NULL,&M);CHKERRQ(ierr);
        ierr = MatSetRandom(M,rand);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = BVMultInPlace(ctx->S,M,0,ctx->L);CHKERRQ(ierr);
        ierr = MatDestroy(&M);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        if (contour->pA) {
          ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
          ierr = EPSCISSSolveSystem(eps,contour->pA[0],contour->pA[1],ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = EPSCISSSolveSystem(eps,A,B,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        }
      }
    }
  }
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
    ierr = PetscFree(H1);CHKERRQ(ierr);
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSComputeVectors_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       n;
  Mat            Z,B=NULL;

  PetscFunctionBegin;
  if (eps->ishermitian) {
    if (eps->isgeneralized && !eps->ispositive) {
      ierr = EPSComputeVectors_Indefinite(eps);CHKERRQ(ierr);
    } else {
      ierr = EPSComputeVectors_Hermitian(eps);CHKERRQ(ierr);
    }
    if (eps->isgeneralized && eps->ispositive && ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
      /* normalize to have unit B-norm */
      ierr = STGetMatrix(eps->st,1,&B);CHKERRQ(ierr);
      ierr = BVSetMatrix(eps->V,B,PETSC_FALSE);CHKERRQ(ierr);
      ierr = BVNormalize(eps->V,NULL);CHKERRQ(ierr);
      ierr = BVSetMatrix(eps->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = DSGetDimensions(eps->ds,&n,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(eps->V,0,n);CHKERRQ(ierr);

  /* right eigenvectors */
  ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);

  /* V = V * Z */
  ierr = DSGetMat(eps->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = BVMultInPlace(eps->V,Z,0,n);CHKERRQ(ierr);
  ierr = MatDestroy(&Z);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(eps->V,0,eps->nconv);CHKERRQ(ierr);

  /* normalize */
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
    ierr = BVNormalize(eps->V,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetSizes_CISS(EPS eps,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       oN,oL,oM,oLmax,onpart;

  PetscFunctionBegin;
  oN = ctx->N;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
    if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
  } else {
    if (ip<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
    if (ip%2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
    if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
  }
  oL = ctx->L;
  if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
    ctx->L = 16;
  } else {
    if (bs<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
    ctx->L = bs;
  }
  oM = ctx->M;
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    if (ms<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = ms;
  }
  onpart = ctx->npart;
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->npart = 1;
  } else {
    if (npart<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
    ctx->npart = npart;
  }
  oLmax = ctx->L_max;
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L_max = 64;
  } else {
    if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    ctx->L_max = PetscMax(bsmax,ctx->L);
  }
  if (onpart != ctx->npart || oN != ctx->N || realmats != ctx->isreal) {
    ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
    ierr = PetscInfo(eps,"Resetting the contour data structure due to a change of parameters\n");CHKERRQ(ierr);
    eps->state = EPS_STATE_INITIAL;
  }
  ctx->isreal = realmats;
  if (oL != ctx->L || oM != ctx->M || oLmax != ctx->L_max) eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSSetSizes - Sets the values of various size parameters in the CISS solver.

   Logically Collective on eps

   Input Parameters:
+  eps   - the eigenproblem solver context
.  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - A and B are real

   Options Database Keys:
+  -eps_ciss_integration_points - Sets the number of integration points
.  -eps_ciss_blocksize - Sets the block size
.  -eps_ciss_moments - Sets the moment size
.  -eps_ciss_partitions - Sets the number of partitions
.  -eps_ciss_maxblocksize - Sets the maximum block size
-  -eps_ciss_realmats - A and B are real

   Note:
   The default number of partitions is 1. This means the internal KSP object is shared
   among all processes of the EPS communicator. Otherwise, the communicator is split
   into npart communicators, so that npart KSP solves proceed simultaneously.

   Level: advanced

.seealso: EPSCISSGetSizes()
@*/
PetscErrorCode EPSCISSSetSizes(EPS eps,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,ip,2);
  PetscValidLogicalCollectiveInt(eps,bs,3);
  PetscValidLogicalCollectiveInt(eps,ms,4);
  PetscValidLogicalCollectiveInt(eps,npart,5);
  PetscValidLogicalCollectiveInt(eps,bsmax,6);
  PetscValidLogicalCollectiveBool(eps,realmats,7);
  ierr = PetscTryMethod(eps,"EPSCISSSetSizes_C",(EPS,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(eps,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetSizes_CISS(EPS eps,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (ip) *ip = ctx->N;
  if (bs) *bs = ctx->L;
  if (ms) *ms = ctx->M;
  if (npart) *npart = ctx->npart;
  if (bsmax) *bsmax = ctx->L_max;
  if (realmats) *realmats = ctx->isreal;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSGetSizes - Gets the values of various size parameters in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - A and B are real

   Level: advanced

.seealso: EPSCISSSetSizes()
@*/
PetscErrorCode EPSCISSGetSizes(EPS eps,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSCISSGetSizes_C",(EPS,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(eps,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetThreshold_CISS(EPS eps,PetscReal delta,PetscReal spur)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (delta == PETSC_DEFAULT) {
    ctx->delta = SLEPC_DEFAULT_TOL*1e-4;
  } else {
    if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
    ctx->delta = delta;
  }
  if (spur == PETSC_DEFAULT) {
    ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  } else {
    if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
    ctx->spurious_threshold = spur;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSCISSSetThreshold - Sets the values of various threshold parameters in
   the CISS solver.

   Logically Collective on eps

   Input Parameters:
+  eps   - the eigenproblem solver context
.  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Options Database Keys:
+  -eps_ciss_delta - Sets the delta
-  -eps_ciss_spurious_threshold - Sets the spurious threshold

   Level: advanced

.seealso: EPSCISSGetThreshold()
@*/
PetscErrorCode EPSCISSSetThreshold(EPS eps,PetscReal delta,PetscReal spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,delta,2);
  PetscValidLogicalCollectiveReal(eps,spur,3);
  ierr = PetscTryMethod(eps,"EPSCISSSetThreshold_C",(EPS,PetscReal,PetscReal),(eps,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetThreshold_CISS(EPS eps,PetscReal *delta,PetscReal *spur)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (delta) *delta = ctx->delta;
  if (spur)  *spur = ctx->spurious_threshold;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSGetThreshold - Gets the values of various threshold parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Level: advanced

.seealso: EPSCISSSetThreshold()
@*/
PetscErrorCode EPSCISSGetThreshold(EPS eps,PetscReal *delta,PetscReal *spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSCISSGetThreshold_C",(EPS,PetscReal*,PetscReal*),(eps,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetRefinement_CISS(EPS eps,PetscInt inner,PetscInt blsize)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (inner == PETSC_DEFAULT) {
    ctx->refine_inner = 0;
  } else {
    if (inner<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine inner argument must be >= 0");
    ctx->refine_inner = inner;
  }
  if (blsize == PETSC_DEFAULT) {
    ctx->refine_blocksize = 0;
  } else {
    if (blsize<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine blocksize argument must be >= 0");
    ctx->refine_blocksize = blsize;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSCISSSetRefinement - Sets the values of various refinement parameters
   in the CISS solver.

   Logically Collective on eps

   Input Parameters:
+  eps    - the eigenproblem solver context
.  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Options Database Keys:
+  -eps_ciss_refine_inner - Sets number of inner iterations
-  -eps_ciss_refine_blocksize - Sets number of blocksize iterations

   Level: advanced

.seealso: EPSCISSGetRefinement()
@*/
PetscErrorCode EPSCISSSetRefinement(EPS eps,PetscInt inner,PetscInt blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,inner,2);
  PetscValidLogicalCollectiveInt(eps,blsize,3);
  ierr = PetscTryMethod(eps,"EPSCISSSetRefinement_C",(EPS,PetscInt,PetscInt),(eps,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetRefinement_CISS(EPS eps,PetscInt *inner,PetscInt *blsize)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (inner)  *inner = ctx->refine_inner;
  if (blsize) *blsize = ctx->refine_blocksize;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSGetRefinement - Gets the values of various refinement parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Level: advanced

.seealso: EPSCISSSetRefinement()
@*/
PetscErrorCode EPSCISSGetRefinement(EPS eps, PetscInt *inner, PetscInt *blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSCISSGetRefinement_C",(EPS,PetscInt*,PetscInt*),(eps,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetUseST_CISS(EPS eps,PetscBool usest)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ctx->usest     = usest;
  ctx->usest_set = PETSC_TRUE;
  eps->state     = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSSetUseST - Sets a flag indicating that the CISS solver will
   use the ST object for the linear solves.

   Logically Collective on eps

   Input Parameters:
+  eps    - the eigenproblem solver context
-  usest  - boolean flag to use the ST object or not

   Options Database Keys:
.  -eps_ciss_usest <bool> - whether the ST object will be used or not

   Level: advanced

.seealso: EPSCISSGetUseST()
@*/
PetscErrorCode EPSCISSSetUseST(EPS eps,PetscBool usest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,usest,2);
  ierr = PetscTryMethod(eps,"EPSCISSSetUseST_C",(EPS,PetscBool),(eps,usest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetUseST_CISS(EPS eps,PetscBool *usest)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  *usest = ctx->usest;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSGetUseST - Gets the flag for using the ST object
   in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  usest - boolean flag indicating if the ST object is being used

   Level: advanced

.seealso: EPSCISSSetUseST()
@*/
PetscErrorCode EPSCISSGetUseST(EPS eps,PetscBool *usest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidBoolPointer(usest,2);
  ierr = PetscUseMethod(eps,"EPSCISSGetUseST_C",(EPS,PetscBool*),(eps,usest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetQuadRule_CISS(EPS eps,EPSCISSQuadRule quad)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (ctx->quad != quad) {
    ctx->quad  = quad;
    eps->state = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSCISSSetQuadRule - Sets the quadrature rule used in the CISS solver.

   Logically Collective on eps

   Input Parameters:
+  eps  - the eigenproblem solver context
-  quad - the quadrature rule

   Options Database Key:
.  -eps_ciss_quadrule - Sets the quadrature rule (either 'trapezoidal' or
                           'chebyshev')

   Notes:
   By default, the trapezoidal rule is used (EPS_CISS_QUADRULE_TRAPEZOIDAL).

   If the 'chebyshev' option is specified (EPS_CISS_QUADRULE_CHEBYSHEV), then
   Chebyshev points are used as quadrature points.

   Level: advanced

.seealso: EPSCISSGetQuadRule(), EPSCISSQuadRule
@*/
PetscErrorCode EPSCISSSetQuadRule(EPS eps,EPSCISSQuadRule quad)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,quad,2);
  ierr = PetscTryMethod(eps,"EPSCISSSetQuadRule_C",(EPS,EPSCISSQuadRule),(eps,quad));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetQuadRule_CISS(EPS eps,EPSCISSQuadRule *quad)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  *quad = ctx->quad;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSGetQuadRule - Gets the quadrature rule used in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  quad - quadrature rule

   Level: advanced

.seealso: EPSCISSSetQuadRule() EPSCISSQuadRule
@*/
PetscErrorCode EPSCISSGetQuadRule(EPS eps,EPSCISSQuadRule *quad)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(quad,2);
  ierr = PetscUseMethod(eps,"EPSCISSGetQuadRule_C",(EPS,EPSCISSQuadRule*),(eps,quad));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetExtraction_CISS(EPS eps,EPSCISSExtraction extraction)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (ctx->extraction != extraction) {
    ctx->extraction = extraction;
    eps->state      = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSCISSSetExtraction - Sets the extraction technique used in the CISS solver.

   Logically Collective on eps

   Input Parameters:
+  eps        - the eigenproblem solver context
-  extraction - the extraction technique

   Options Database Key:
.  -eps_ciss_extraction - Sets the extraction technique (either 'ritz' or
                           'hankel')

   Notes:
   By default, the Rayleigh-Ritz extraction is used (EPS_CISS_EXTRACTION_RITZ).

   If the 'hankel' option is specified (EPS_CISS_EXTRACTION_HANKEL), then
   the Block Hankel method is used for extracting eigenpairs.

   Level: advanced

.seealso: EPSCISSGetExtraction(), EPSCISSExtraction
@*/
PetscErrorCode EPSCISSSetExtraction(EPS eps,EPSCISSExtraction extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,extraction,2);
  ierr = PetscTryMethod(eps,"EPSCISSSetExtraction_C",(EPS,EPSCISSExtraction),(eps,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetExtraction_CISS(EPS eps,EPSCISSExtraction *extraction)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  *extraction = ctx->extraction;
  PetscFunctionReturn(0);
}

/*@
   EPSCISSGetExtraction - Gets the extraction technique used in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  extraction - extraction technique

   Level: advanced

.seealso: EPSCISSSetExtraction() EPSCISSExtraction
@*/
PetscErrorCode EPSCISSGetExtraction(EPS eps,EPSCISSExtraction *extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(extraction,2);
  ierr = PetscUseMethod(eps,"EPSCISSGetExtraction_C",(EPS,EPSCISSExtraction*),(eps,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetKSPs_CISS(EPS eps,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode   ierr;
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscInt         i;
  PC               pc;

  PetscFunctionBegin;
  if (!ctx->contour) {  /* initialize contour data structure first */
    ierr = RGCanUseConjugates(eps->rg,ctx->isreal,&ctx->useconj);CHKERRQ(ierr);
    ierr = SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)eps,&ctx->contour);CHKERRQ(ierr);
  }
  contour = ctx->contour;
  if (!contour->ksp) {
    ierr = PetscMalloc1(contour->npoints,&contour->ksp);CHKERRQ(ierr);
    for (i=0;i<contour->npoints;i++) {
      ierr = KSPCreate(PetscSubcommChild(contour->subcomm),&contour->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)contour->ksp[i],(PetscObject)eps,1);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(contour->ksp[i],((PetscObject)eps)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(contour->ksp[i],"eps_ciss_");CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)contour->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptions((PetscObject)contour->ksp[i],((PetscObject)eps)->options);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(contour->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(contour->ksp[i],SlepcDefaultTol(eps->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPGetPC(contour->ksp[i],&pc);CHKERRQ(ierr);
      ierr = KSPSetType(contour->ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    }
  }
  if (nsolve) *nsolve = contour->npoints;
  if (ksp)    *ksp    = contour->ksp;
  PetscFunctionReturn(0);
}

/*@C
   EPSCISSGetKSPs - Retrieve the array of linear solver objects associated with
   the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver solver

   Output Parameters:
+  nsolve - number of solver objects
-  ksp - array of linear solver object

   Notes:
   The number of KSP solvers is equal to the number of integration points divided by
   the number of partitions. This value is halved in the case of real matrices with
   a region centered at the real axis.

   Level: advanced

.seealso: EPSCISSSetSizes()
@*/
PetscErrorCode EPSCISSGetKSPs(EPS eps,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSCISSGetKSPs_C",(EPS,PetscInt*,KSP**),(eps,nsolve,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  if (!ctx->usest) {
    ierr = SlepcContourDataReset(ctx->contour);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_CISS(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode    ierr;
  PetscReal         r3,r4;
  PetscInt          i,i1,i2,i3,i4,i5,i6,i7;
  PetscBool         b1,b2,flg,flg2,flg3,flg4,flg5,flg6;
  EPS_CISS          *ctx = (EPS_CISS*)eps->data;
  EPSCISSQuadRule   quad;
  EPSCISSExtraction extraction;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS CISS Options");CHKERRQ(ierr);

    ierr = EPSCISSGetSizes(eps,&i1,&i2,&i3,&i4,&i5,&b1);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_integration_points","Number of integration points","EPSCISSSetSizes",i1,&i1,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_blocksize","Block size","EPSCISSSetSizes",i2,&i2,&flg2);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_moments","Moment size","EPSCISSSetSizes",i3,&i3,&flg3);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_partitions","Number of partitions","EPSCISSSetSizes",i4,&i4,&flg4);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_maxblocksize","Maximum block size","EPSCISSSetSizes",i5,&i5,&flg5);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-eps_ciss_realmats","True if A and B are real","EPSCISSSetSizes",b1,&b1,&flg6);CHKERRQ(ierr);
    if (flg || flg2 || flg3 || flg4 || flg5 || flg6) { ierr = EPSCISSSetSizes(eps,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr); }

    ierr = EPSCISSGetThreshold(eps,&r3,&r4);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_ciss_delta","Threshold for numerical rank","EPSCISSSetThreshold",r3,&r3,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_ciss_spurious_threshold","Threshold for the spurious eigenpairs","EPSCISSSetThreshold",r4,&r4,&flg2);CHKERRQ(ierr);
    if (flg || flg2) { ierr = EPSCISSSetThreshold(eps,r3,r4);CHKERRQ(ierr); }

    ierr = EPSCISSGetRefinement(eps,&i6,&i7);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_refine_inner","Number of inner iterative refinement iterations","EPSCISSSetRefinement",i6,&i6,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ciss_refine_blocksize","Number of blocksize iterative refinement iterations","EPSCISSSetRefinement",i7,&i7,&flg2);CHKERRQ(ierr);
    if (flg || flg2) { ierr = EPSCISSSetRefinement(eps,i6,i7);CHKERRQ(ierr); }

    ierr = EPSCISSGetUseST(eps,&b2);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-eps_ciss_usest","Use ST for linear solves","EPSCISSSetUseST",b2,&b2,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSCISSSetUseST(eps,b2);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-eps_ciss_quadrule","Quadrature rule","EPSCISSSetQuadRule",EPSCISSQuadRules,(PetscEnum)ctx->quad,(PetscEnum*)&quad,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSCISSSetQuadRule(eps,quad);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-eps_ciss_extraction","Extraction technique","EPSCISSSetExtraction",EPSCISSExtractions,(PetscEnum)ctx->extraction,(PetscEnum*)&extraction,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSCISSSetExtraction(eps,extraction);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!eps->rg) { ierr = EPSGetRG(eps,&eps->rg);CHKERRQ(ierr); }
  ierr = RGSetFromOptions(eps->rg);CHKERRQ(ierr); /* this is necessary here to set useconj */
  if (!ctx->contour || !ctx->contour->ksp) { ierr = EPSCISSGetKSPs(eps,NULL,NULL);CHKERRQ(ierr); }
  for (i=0;i<ctx->contour->npoints;i++) {
    ierr = KSPSetFromOptions(ctx->contour->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscSubcommSetFromOptions(ctx->contour->subcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
  ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetUseST_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetUseST_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetQuadRule_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetQuadRule_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetKSPs_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_CISS(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      isascii;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  sizes { integration points: %D, block size: %D, moment size: %D, partitions: %D, maximum block size: %D }\n",ctx->N,ctx->L,ctx->M,ctx->npart,ctx->L_max);CHKERRQ(ierr);
    if (ctx->isreal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  exploiting symmetry of integration points\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  threshold { delta: %g, spurious threshold: %g }\n",(double)ctx->delta,(double)ctx->spurious_threshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  iterative refinement { inner: %D, blocksize: %D }\n",ctx->refine_inner, ctx->refine_blocksize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  extraction: %s\n",EPSCISSExtractions[ctx->extraction]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  quadrature rule: %s\n",EPSCISSQuadRules[ctx->quad]);CHKERRQ(ierr);
    if (ctx->usest) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using ST for linear solves\n");CHKERRQ(ierr);
    } else {
      if (!ctx->contour || !ctx->contour->ksp) { ierr = EPSCISSGetKSPs(eps,NULL,NULL);CHKERRQ(ierr); }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      if (ctx->npart>1 && ctx->contour->subcomm) {
        ierr = PetscViewerGetSubViewer(viewer,ctx->contour->subcomm->child,&sviewer);CHKERRQ(ierr);
        if (!ctx->contour->subcomm->color) {
          ierr = KSPView(ctx->contour->ksp[0],sviewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,ctx->contour->subcomm->child,&sviewer);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        /* extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
        ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      } else {
        ierr = KSPView(ctx->contour->ksp[0],viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      usest = ctx->usest;
  KSP            ksp;
  PC             pc;

  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) {
    if (!ctx->usest_set) usest = (ctx->npart>1)? PETSC_FALSE: PETSC_TRUE;
    if (usest) {
      ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
    } else {
      /* we are not going to use ST, so avoid factorizing the matrix */
      ierr = STSetType(eps->st,STSHIFT);CHKERRQ(ierr);
      if (eps->isgeneralized) {
        ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = ctx;

  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_CONTOUR;

  eps->ops->solve          = EPSSolve_CISS;
  eps->ops->setup          = EPSSetUp_CISS;
  eps->ops->setupsort      = EPSSetUpSort_CISS;
  eps->ops->setfromoptions = EPSSetFromOptions_CISS;
  eps->ops->destroy        = EPSDestroy_CISS;
  eps->ops->reset          = EPSReset_CISS;
  eps->ops->view           = EPSView_CISS;
  eps->ops->computevectors = EPSComputeVectors_CISS;
  eps->ops->setdefaultst   = EPSSetDefaultST_CISS;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",EPSCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",EPSCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",EPSCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",EPSCISSGetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",EPSCISSSetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",EPSCISSGetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetUseST_C",EPSCISSSetUseST_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetUseST_C",EPSCISSGetUseST_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetQuadRule_C",EPSCISSSetQuadRule_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetQuadRule_C",EPSCISSGetQuadRule_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetExtraction_C",EPSCISSSetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetExtraction_C",EPSCISSGetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetKSPs_C",EPSCISSGetKSPs_CISS);CHKERRQ(ierr);

  /* set default values of parameters */
  ctx->N                  = 32;
  ctx->L                  = 16;
  ctx->M                  = ctx->N/4;
  ctx->delta              = SLEPC_DEFAULT_TOL*1e-4;
  ctx->L_max              = 64;
  ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  ctx->usest              = PETSC_TRUE;
  ctx->usest_set          = PETSC_FALSE;
  ctx->isreal             = PETSC_FALSE;
  ctx->refine_inner       = 0;
  ctx->refine_blocksize   = 0;
  ctx->npart              = 1;
  ctx->quad               = (EPSCISSQuadRule)0;
  ctx->extraction         = EPS_CISS_EXTRACTION_RITZ;
  PetscFunctionReturn(0);
}

