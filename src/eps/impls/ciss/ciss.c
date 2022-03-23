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

/*
  Set up KSP solvers for every integration point, only called if !ctx->usest
*/
static PetscErrorCode EPSCISSSetUp(EPS eps,Mat A,Mat B,Mat Pa,Mat Pb)
{
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscInt         i,p_id,nsplit;
  Mat              Amat,Pmat;
  MatStructure     str,strp;

  PetscFunctionBegin;
  if (!ctx->contour || !ctx->contour->ksp) CHKERRQ(EPSCISSGetKSPs(eps,NULL,NULL));
  contour = ctx->contour;
  CHKERRQ(STGetMatStructure(eps->st,&str));
  CHKERRQ(STGetSplitPreconditionerInfo(eps->st,&nsplit,&strp));
  for (i=0;i<contour->npoints;i++) {
    p_id = i*contour->subcomm->n + contour->subcomm->color;
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&Amat));
    if (B) CHKERRQ(MatAXPY(Amat,-ctx->omega[p_id],B,str));
    else CHKERRQ(MatShift(Amat,-ctx->omega[p_id]));
    if (nsplit) {
      CHKERRQ(MatDuplicate(Pa,MAT_COPY_VALUES,&Pmat));
      if (Pb) CHKERRQ(MatAXPY(Pmat,-ctx->omega[p_id],Pb,strp));
      else CHKERRQ(MatShift(Pmat,-ctx->omega[p_id]));
    } else Pmat = Amat;
    CHKERRQ(EPS_KSPSetOperators(contour->ksp[i],Amat,Amat));
    CHKERRQ(MatDestroy(&Amat));
    if (nsplit) CHKERRQ(MatDestroy(&Pmat));
  }
  PetscFunctionReturn(0);
}

/*
  Y_i = (A-z_i B)^{-1}BV for every integration point, Y=[Y_i] is in the context
*/
static PetscErrorCode EPSCISSSolve(EPS eps,Mat B,BV V,PetscInt L_start,PetscInt L_end)
{
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscInt         i,p_id;
  Mat              MV,BMV=NULL,MC;
  KSP              ksp;

  PetscFunctionBegin;
  if (!ctx->contour || !ctx->contour->ksp) CHKERRQ(EPSCISSGetKSPs(eps,NULL,NULL));
  contour = ctx->contour;
  CHKERRQ(BVSetActiveColumns(V,L_start,L_end));
  CHKERRQ(BVGetMat(V,&MV));
  for (i=0;i<contour->npoints;i++) {
    p_id = i*contour->subcomm->n + contour->subcomm->color;
    if (ctx->usest)  {
      CHKERRQ(STSetShift(eps->st,ctx->omega[p_id]));
      CHKERRQ(STGetKSP(eps->st,&ksp));
    } else ksp = contour->ksp[i];
    CHKERRQ(BVSetActiveColumns(ctx->Y,i*ctx->L+L_start,i*ctx->L+L_end));
    CHKERRQ(BVGetMat(ctx->Y,&MC));
    if (B) {
      if (!i) {
        CHKERRQ(MatProductCreate(B,MV,NULL,&BMV));
        CHKERRQ(MatProductSetType(BMV,MATPRODUCT_AB));
        CHKERRQ(MatProductSetFromOptions(BMV));
        CHKERRQ(MatProductSymbolic(BMV));
      }
      CHKERRQ(MatProductNumeric(BMV));
      CHKERRQ(KSPMatSolve(ksp,BMV,MC));
    } else CHKERRQ(KSPMatSolve(ksp,MV,MC));
    CHKERRQ(BVRestoreMat(ctx->Y,&MC));
    if (ctx->usest && i<contour->npoints-1) CHKERRQ(KSPReset(ksp));
  }
  CHKERRQ(MatDestroy(&BMV));
  CHKERRQ(BVRestoreMat(V,&MV));
  PetscFunctionReturn(0);
}

static PetscErrorCode rescale_eig(EPS eps,PetscInt nv)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i;
  PetscScalar    center;
  PetscReal      radius,a,b,c,d,rgscale;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      start_ang,end_ang,vscale,theta;
#endif
  PetscBool      isring,isellipse,isinterval;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGRING,&isring));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGINTERVAL,&isinterval));
  CHKERRQ(RGGetScale(eps->rg,&rgscale));
  if (isinterval) {
    CHKERRQ(RGIntervalGetEndpoints(eps->rg,NULL,NULL,&c,&d));
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
      CHKERRQ(RGEllipseGetParameters(eps->rg,&center,&radius,NULL));
      for (i=0;i<nv;i++) eps->eigr[i] = rgscale*(center + radius*eps->eigr[i]);
    } else if (isinterval) {
      CHKERRQ(RGIntervalGetEndpoints(eps->rg,&a,&b,&c,&d));
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
      CHKERRQ(RGRingGetParameters(eps->rg,&center,&radius,&vscale,&start_ang,&end_ang,NULL));
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
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscBool        istrivial,isring,isellipse,isinterval,flg;
  PetscReal        c,d;
  PetscInt         nsplit;
  PetscRandom      rand;
  PetscObjectId    id;
  PetscObjectState state;
  Mat              A[2],Psplit[2];
  Vec              v0;

  PetscFunctionBegin;
  if (eps->ncv==PETSC_DEFAULT) {
    eps->ncv = ctx->L_max*ctx->M;
    if (eps->ncv>eps->n) {
      eps->ncv = eps->n;
      ctx->L_max = eps->ncv/ctx->M;
      PetscCheck(ctx->L_max,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Cannot adjust solver parameters, try setting a smaller value of M (moment size)");
    }
  } else {
    CHKERRQ(EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd));
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
  PetscCheck(eps->which==EPS_ALL,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only computing all eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_EXTRACTION | EPS_FEATURE_STOPPING | EPS_FEATURE_TWOSIDED);

  /* check region */
  CHKERRQ(RGIsTrivial(eps->rg,&istrivial));
  PetscCheck(!istrivial,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"CISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  CHKERRQ(RGGetComplement(eps->rg,&flg));
  PetscCheck(!flg,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"A region with complement flag set is not allowed");
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGRING,&isring));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGINTERVAL,&isinterval));
  PetscCheck(isellipse || isring || isinterval,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Currently only implemented for interval, elliptic or ring regions");

  /* if the region has changed, then reset contour data */
  CHKERRQ(PetscObjectGetId((PetscObject)eps->rg,&id));
  CHKERRQ(PetscObjectStateGet((PetscObject)eps->rg,&state));
  if (ctx->rgid && (id != ctx->rgid || state != ctx->rgstate)) {
    CHKERRQ(SlepcContourDataDestroy(&ctx->contour));
    CHKERRQ(PetscInfo(eps,"Resetting the contour data structure due to a change of region\n"));
    ctx->rgid = id; ctx->rgstate = state;
  }

#if !defined(PETSC_USE_COMPLEX)
  PetscCheck(!isring,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Ring region only supported for complex scalars");
#endif
  if (isinterval) {
    CHKERRQ(RGIntervalGetEndpoints(eps->rg,NULL,NULL,&c,&d));
#if !defined(PETSC_USE_COMPLEX)
    PetscCheck(c==d && c==0.0,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"In real scalars, endpoints of the imaginary axis must be both zero");
#endif
    if (!ctx->quad && c==d) ctx->quad = EPS_CISS_QUADRULE_CHEBYSHEV;
  }
  if (!ctx->quad) ctx->quad = EPS_CISS_QUADRULE_TRAPEZOIDAL;

  /* create contour data structure */
  if (!ctx->contour) {
    CHKERRQ(RGCanUseConjugates(eps->rg,ctx->isreal,&ctx->useconj));
    CHKERRQ(SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)eps,&ctx->contour));
  }

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(BVGetRandomContext(eps->V,&rand));  /* make sure the random context is available when duplicating */
  if (ctx->weight) CHKERRQ(PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma));
  CHKERRQ(PetscMalloc4(ctx->N,&ctx->weight,ctx->N+1,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal)));

  /* allocate basis vectors */
  CHKERRQ(BVDestroy(&ctx->S));
  CHKERRQ(BVDuplicateResize(eps->V,ctx->L*ctx->M,&ctx->S));
  CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->S));
  CHKERRQ(BVDestroy(&ctx->V));
  CHKERRQ(BVDuplicateResize(eps->V,ctx->L,&ctx->V));
  CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->V));

  CHKERRQ(STGetMatrix(eps->st,0,&A[0]));
  CHKERRQ(MatIsShell(A[0],&flg));
  PetscCheck(!flg,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Matrix type shell is not supported in this solver");
  if (eps->isgeneralized) CHKERRQ(STGetMatrix(eps->st,1,&A[1]));

  if (!ctx->usest_set) ctx->usest = (ctx->npart>1)? PETSC_FALSE: PETSC_TRUE;
  PetscCheck(!ctx->usest || ctx->npart==1,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The usest flag is not supported when partitions > 1");

  /* check if a user-defined split preconditioner has been set */
  CHKERRQ(STGetSplitPreconditionerInfo(eps->st,&nsplit,NULL));
  if (nsplit) {
    CHKERRQ(STGetSplitPreconditionerTerm(eps->st,0,&Psplit[0]));
    if (eps->isgeneralized) CHKERRQ(STGetSplitPreconditionerTerm(eps->st,1,&Psplit[1]));
  }

  contour = ctx->contour;
  CHKERRQ(SlepcContourRedundantMat(contour,eps->isgeneralized?2:1,A,nsplit?Psplit:NULL));
  if (contour->pA) {
    CHKERRQ(BVGetColumn(ctx->V,0,&v0));
    CHKERRQ(SlepcContourScatterCreate(contour,v0));
    CHKERRQ(BVRestoreColumn(ctx->V,0,&v0));
    CHKERRQ(BVDestroy(&ctx->pV));
    CHKERRQ(BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->pV));
    CHKERRQ(BVSetSizesFromVec(ctx->pV,contour->xsub,eps->n));
    CHKERRQ(BVSetFromOptions(ctx->pV));
    CHKERRQ(BVResize(ctx->pV,ctx->L,PETSC_FALSE));
    CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->pV));
  }

  EPSCheckDefinite(eps);
  EPSCheckSinvertCondition(eps,ctx->usest," (with the usest flag set)");

  CHKERRQ(BVDestroy(&ctx->Y));
  if (contour->pA) {
    CHKERRQ(BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->Y));
    CHKERRQ(BVSetSizesFromVec(ctx->Y,contour->xsub,eps->n));
    CHKERRQ(BVSetFromOptions(ctx->Y));
    CHKERRQ(BVResize(ctx->Y,contour->npoints*ctx->L,PETSC_FALSE));
  } else CHKERRQ(BVDuplicateResize(eps->V,contour->npoints*ctx->L,&ctx->Y));
  CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->Y));

  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) CHKERRQ(DSSetType(eps->ds,DSGNHEP));
  else if (eps->isgeneralized) {
    if (eps->ishermitian && eps->ispositive) CHKERRQ(DSSetType(eps->ds,DSGHEP));
    else CHKERRQ(DSSetType(eps->ds,DSGNHEP));
  } else {
    if (eps->ishermitian) CHKERRQ(DSSetType(eps->ds,DSHEP));
    else CHKERRQ(DSSetType(eps->ds,DSNHEP));
  }
  CHKERRQ(DSAllocate(eps->ds,eps->ncv));

#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(EPSSetWorkVecs(eps,3));
  if (!eps->ishermitian) CHKERRQ(PetscInfo(eps,"Warning: complex eigenvalues are not calculated exactly without --with-scalar-type=complex in PETSc\n"));
#else
  CHKERRQ(EPSSetWorkVecs(eps,2));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUpSort_CISS(EPS eps)
{
  SlepcSC        sc;

  PetscFunctionBegin;
  /* fill sorting criterion context */
  eps->sc->comparison    = SlepcCompareSmallestReal;
  eps->sc->comparisonctx = NULL;
  eps->sc->map           = NULL;
  eps->sc->mapobj        = NULL;

  /* fill sorting criterion for DS */
  CHKERRQ(DSGetSlepcSC(eps->ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_CISS(EPS eps)
{
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour = ctx->contour;
  Mat              A,B,X,M,pA,pB,T,J,Pa=NULL,Pb=NULL;
  BV               V;
  PetscInt         i,j,ld,nmat,L_add=0,nv=0,L_base=ctx->L,inner,nlocal,*inside,nsplit;
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
  CHKERRQ(VecGetLocalSize(w[0],&nlocal));
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(RGComputeQuadrature(eps->rg,ctx->quad==EPS_CISS_QUADRULE_CHEBYSHEV?RG_QUADRULE_CHEBYSHEV:RG_QUADRULE_TRAPEZOIDAL,ctx->N,ctx->omega,ctx->pp,ctx->weight));
  CHKERRQ(STGetNumMatrices(eps->st,&nmat));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  if (nmat>1) CHKERRQ(STGetMatrix(eps->st,1,&B));
  else B = NULL;
  J = (contour->pA && nmat>1)? contour->pA[1]: B;
  V = contour->pA? ctx->pV: ctx->V;
  if (!ctx->usest) {
    T = contour->pA? contour->pA[0]: A;
    CHKERRQ(STGetSplitPreconditionerInfo(eps->st,&nsplit,NULL));
    if (nsplit) {
      if (contour->pA) {
        Pa = contour->pP[0];
        if (nsplit>1) Pb = contour->pP[1];
      } else {
        CHKERRQ(STGetSplitPreconditionerTerm(eps->st,0,&Pa));
        if (nsplit>1) CHKERRQ(STGetSplitPreconditionerTerm(eps->st,1,&Pb));
      }
    }
    CHKERRQ(EPSCISSSetUp(eps,T,J,Pa,Pb));
  }
  CHKERRQ(BVSetActiveColumns(ctx->V,0,ctx->L));
  CHKERRQ(BVSetRandomSign(ctx->V));
  CHKERRQ(BVGetRandomContext(ctx->V,&rand));

  if (contour->pA) CHKERRQ(BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup));
  CHKERRQ(EPSCISSSolve(eps,J,V,0,ctx->L));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse));
  if (isellipse) {
    CHKERRQ(BVTraceQuadrature(ctx->Y,ctx->V,ctx->L,ctx->L,ctx->weight,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj,&est_eig));
    CHKERRQ(PetscInfo(eps,"Estimated eigenvalue count: %f\n",(double)est_eig));
    eta = PetscPowReal(10.0,-PetscLog10Real(eps->tol)/ctx->N);
    L_add = PetscMax(0,(PetscInt)PetscCeilReal((est_eig*eta)/ctx->M)-ctx->L);
    if (L_add>ctx->L_max-ctx->L) {
      CHKERRQ(PetscInfo(eps,"Number of eigenvalues inside the contour path may be too large\n"));
      L_add = ctx->L_max-ctx->L;
    }
  }
#endif
  if (L_add>0) {
    CHKERRQ(PetscInfo(eps,"Changing L %" PetscInt_FMT " -> %" PetscInt_FMT " by Estimate #Eig\n",ctx->L,ctx->L+L_add));
    CHKERRQ(BVCISSResizeBases(ctx->S,contour->pA?ctx->pV:ctx->V,ctx->Y,ctx->L,ctx->L+L_add,ctx->M,contour->npoints));
    CHKERRQ(BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add));
    CHKERRQ(BVSetRandomSign(ctx->V));
    if (contour->pA) CHKERRQ(BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup));
    ctx->L += L_add;
    CHKERRQ(EPSCISSSolve(eps,J,V,ctx->L-L_add,ctx->L));
  }
  CHKERRQ(PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0));
  for (i=0;i<ctx->refine_blocksize;i++) {
    CHKERRQ(BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj));
    CHKERRQ(CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0));
    CHKERRQ(PetscLogEventBegin(EPS_CISS_SVD,eps,0,0,0));
    CHKERRQ(SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv));
    CHKERRQ(PetscLogEventEnd(EPS_CISS_SVD,eps,0,0,0));
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    CHKERRQ(PetscInfo(eps,"Changing L %" PetscInt_FMT " -> %" PetscInt_FMT " by SVD(H0)\n",ctx->L,ctx->L+L_add));
    CHKERRQ(BVCISSResizeBases(ctx->S,contour->pA?ctx->pV:ctx->V,ctx->Y,ctx->L,ctx->L+L_add,ctx->M,contour->npoints));
    CHKERRQ(BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add));
    CHKERRQ(BVSetRandomSign(ctx->V));
    if (contour->pA) CHKERRQ(BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup));
    ctx->L += L_add;
    CHKERRQ(EPSCISSSolve(eps,J,V,ctx->L-L_add,ctx->L));
    if (L_add) {
      CHKERRQ(PetscFree2(Mu,H0));
      CHKERRQ(PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0));
    }
  }
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) CHKERRQ(PetscMalloc1(ctx->L*ctx->M*ctx->L*ctx->M,&H1));

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
        CHKERRQ(BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj));
        CHKERRQ(CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0));
        CHKERRQ(PetscLogEventBegin(EPS_CISS_SVD,eps,0,0,0));
        CHKERRQ(SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv));
        CHKERRQ(PetscLogEventEnd(EPS_CISS_SVD,eps,0,0,0));
        break;
      } else {
        CHKERRQ(BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj));
        CHKERRQ(BVSetActiveColumns(ctx->S,0,ctx->L));
        CHKERRQ(BVSetActiveColumns(ctx->V,0,ctx->L));
        CHKERRQ(BVCopy(ctx->S,ctx->V));
        CHKERRQ(BVSVDAndRank(ctx->S,ctx->M,ctx->L,ctx->delta,BV_SVD_METHOD_REFINE,H0,ctx->sigma,&nv));
        if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
          if (contour->pA) CHKERRQ(BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup));
          CHKERRQ(EPSCISSSolve(eps,J,V,0,ctx->L));
        } else break;
      }
    }
    eps->nconv = 0;
    if (nv == 0) eps->reason = EPS_CONVERGED_TOL;
    else {
      CHKERRQ(DSSetDimensions(eps->ds,nv,0,0));
      CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));

      if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
        CHKERRQ(CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0));
        CHKERRQ(CISS_BlockHankel(Mu,1,ctx->L,ctx->M,H1));
        CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&temp));
        for (j=0;j<nv;j++) {
          for (i=0;i<nv;i++) {
            temp[i+j*ld] = H1[i+j*ctx->L*ctx->M];
          }
        }
        CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&temp));
        CHKERRQ(DSGetArray(eps->ds,DS_MAT_B,&temp));
        for (j=0;j<nv;j++) {
          for (i=0;i<nv;i++) {
            temp[i+j*ld] = H0[i+j*ctx->L*ctx->M];
          }
        }
        CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_B,&temp));
      } else {
        CHKERRQ(BVSetActiveColumns(ctx->S,0,nv));
        CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&pA));
        CHKERRQ(MatZeroEntries(pA));
        CHKERRQ(BVMatProject(ctx->S,A,ctx->S,pA));
        CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&pA));
        if (B) {
          CHKERRQ(DSGetMat(eps->ds,DS_MAT_B,&pB));
          CHKERRQ(MatZeroEntries(pB));
          CHKERRQ(BVMatProject(ctx->S,B,ctx->S,pB));
          CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_B,&pB));
        }
      }

      CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
      CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

      CHKERRQ(PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr));
      CHKERRQ(rescale_eig(eps,nv));
      CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&X));
      CHKERRQ(SlepcCISS_isGhost(X,nv,ctx->sigma,ctx->spurious_threshold,fl1));
      CHKERRQ(MatDestroy(&X));
      CHKERRQ(RGCheckInside(eps->rg,nv,eps->eigr,eps->eigi,inside));
      for (i=0;i<nv;i++) {
        if (fl1[i] && inside[i]>=0) {
          rr[i] = 1.0;
          eps->nconv++;
        } else rr[i] = 0.0;
      }
      CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,rr,NULL,&eps->nconv));
      CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));
      CHKERRQ(rescale_eig(eps,nv));
      CHKERRQ(PetscFree3(fl1,inside,rr));
      CHKERRQ(BVSetActiveColumns(eps->V,0,nv));
      if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
        CHKERRQ(BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj));
        CHKERRQ(BVSetActiveColumns(ctx->S,0,ctx->L));
        CHKERRQ(BVCopy(ctx->S,ctx->V));
        CHKERRQ(BVSetActiveColumns(ctx->S,0,nv));
      }
      CHKERRQ(BVCopy(ctx->S,eps->V));

      CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&X));
      CHKERRQ(BVMultInPlace(ctx->S,X,0,eps->nconv));
      if (eps->ishermitian) CHKERRQ(BVMultInPlace(eps->V,X,0,eps->nconv));
      CHKERRQ(MatDestroy(&X));
      max_error = 0.0;
      for (i=0;i<eps->nconv;i++) {
        CHKERRQ(BVGetColumn(ctx->S,i,&si));
#if !defined(PETSC_USE_COMPLEX)
        if (eps->eigi[i]!=0.0) CHKERRQ(BVGetColumn(ctx->S,i+1,&si1));
#endif
        CHKERRQ(EPSComputeResidualNorm_Private(eps,PETSC_FALSE,eps->eigr[i],eps->eigi[i],si,si1,w,&error));
        if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {  /* vector is not normalized */
          CHKERRQ(VecNorm(si,NORM_2,&norm));
#if !defined(PETSC_USE_COMPLEX)
          if (eps->eigi[i]!=0.0) {
            CHKERRQ(VecNorm(si1,NORM_2,&normi));
            norm = SlepcAbsEigenvalue(norm,normi);
          }
#endif
          error /= norm;
        }
        CHKERRQ((*eps->converged)(eps,eps->eigr[i],eps->eigi[i],error,&error,eps->convergedctx));
        CHKERRQ(BVRestoreColumn(ctx->S,i,&si));
#if !defined(PETSC_USE_COMPLEX)
        if (eps->eigi[i]!=0.0) {
          CHKERRQ(BVRestoreColumn(ctx->S,i+1,&si1));
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
        nv = PetscMin(nv,ctx->L*ctx->M);
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nv,ctx->L,NULL,&M));
        CHKERRQ(MatSetRandom(M,rand));
        CHKERRQ(BVSetActiveColumns(ctx->S,0,nv));
        CHKERRQ(BVMultInPlace(ctx->S,M,0,ctx->L));
        CHKERRQ(MatDestroy(&M));
        CHKERRQ(BVSetActiveColumns(ctx->S,0,ctx->L));
        CHKERRQ(BVSetActiveColumns(ctx->V,0,ctx->L));
        CHKERRQ(BVCopy(ctx->S,ctx->V));
        if (contour->pA) CHKERRQ(BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup));
        CHKERRQ(EPSCISSSolve(eps,J,V,0,ctx->L));
      }
    }
  }
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) CHKERRQ(PetscFree(H1));
  CHKERRQ(PetscFree2(Mu,H0));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSComputeVectors_CISS(EPS eps)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       n;
  Mat            Z,B=NULL;

  PetscFunctionBegin;
  if (eps->ishermitian) {
    if (eps->isgeneralized && !eps->ispositive) CHKERRQ(EPSComputeVectors_Indefinite(eps));
    else CHKERRQ(EPSComputeVectors_Hermitian(eps));
    if (eps->isgeneralized && eps->ispositive && ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) {
      /* normalize to have unit B-norm */
      CHKERRQ(STGetMatrix(eps->st,1,&B));
      CHKERRQ(BVSetMatrix(eps->V,B,PETSC_FALSE));
      CHKERRQ(BVNormalize(eps->V,NULL));
      CHKERRQ(BVSetMatrix(eps->V,NULL,PETSC_FALSE));
    }
    PetscFunctionReturn(0);
  }
  CHKERRQ(DSGetDimensions(eps->ds,&n,NULL,NULL,NULL));
  CHKERRQ(BVSetActiveColumns(eps->V,0,n));

  /* right eigenvectors */
  CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));

  /* V = V * Z */
  CHKERRQ(DSGetMat(eps->ds,DS_MAT_X,&Z));
  CHKERRQ(BVMultInPlace(eps->V,Z,0,n));
  CHKERRQ(MatDestroy(&Z));
  CHKERRQ(BVSetActiveColumns(eps->V,0,eps->nconv));

  /* normalize */
  if (ctx->extraction == EPS_CISS_EXTRACTION_HANKEL) CHKERRQ(BVNormalize(eps->V,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetSizes_CISS(EPS eps,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       oN,oL,oM,oLmax,onpart;
  PetscMPIInt    size;

  PetscFunctionBegin;
  oN = ctx->N;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
    if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
  } else {
    PetscCheck(ip>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
    PetscCheck(ip%2==0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
    if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
  }
  oL = ctx->L;
  if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
    ctx->L = 16;
  } else {
    PetscCheck(bs>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
    ctx->L = bs;
  }
  oM = ctx->M;
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    PetscCheck(ms>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    PetscCheck(ms<=ctx->N,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = ms;
  }
  onpart = ctx->npart;
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->npart = 1;
  } else {
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size));
    PetscCheck(npart>0 && npart<=size,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of npart");
    ctx->npart = npart;
  }
  oLmax = ctx->L_max;
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L_max = 64;
  } else {
    PetscCheck(bsmax>0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    ctx->L_max = PetscMax(bsmax,ctx->L);
  }
  if (onpart != ctx->npart || oN != ctx->N || realmats != ctx->isreal) {
    CHKERRQ(SlepcContourDataDestroy(&ctx->contour));
    CHKERRQ(PetscInfo(eps,"Resetting the contour data structure due to a change of parameters\n"));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,ip,2);
  PetscValidLogicalCollectiveInt(eps,bs,3);
  PetscValidLogicalCollectiveInt(eps,ms,4);
  PetscValidLogicalCollectiveInt(eps,npart,5);
  PetscValidLogicalCollectiveInt(eps,bsmax,6);
  PetscValidLogicalCollectiveBool(eps,realmats,7);
  CHKERRQ(PetscTryMethod(eps,"EPSCISSSetSizes_C",(EPS,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(eps,ip,bs,ms,npart,bsmax,realmats)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetSizes_C",(EPS,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(eps,ip,bs,ms,npart,bsmax,realmats)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetThreshold_CISS(EPS eps,PetscReal delta,PetscReal spur)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (delta == PETSC_DEFAULT) {
    ctx->delta = SLEPC_DEFAULT_TOL*1e-4;
  } else {
    PetscCheck(delta>0.0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
    ctx->delta = delta;
  }
  if (spur == PETSC_DEFAULT) {
    ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  } else {
    PetscCheck(spur>0.0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,delta,2);
  PetscValidLogicalCollectiveReal(eps,spur,3);
  CHKERRQ(PetscTryMethod(eps,"EPSCISSSetThreshold_C",(EPS,PetscReal,PetscReal),(eps,delta,spur)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetThreshold_C",(EPS,PetscReal*,PetscReal*),(eps,delta,spur)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSSetRefinement_CISS(EPS eps,PetscInt inner,PetscInt blsize)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (inner == PETSC_DEFAULT) {
    ctx->refine_inner = 0;
  } else {
    PetscCheck(inner>=0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine inner argument must be >= 0");
    ctx->refine_inner = inner;
  }
  if (blsize == PETSC_DEFAULT) {
    ctx->refine_blocksize = 0;
  } else {
    PetscCheck(blsize>=0,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine blocksize argument must be >= 0");
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,inner,2);
  PetscValidLogicalCollectiveInt(eps,blsize,3);
  CHKERRQ(PetscTryMethod(eps,"EPSCISSSetRefinement_C",(EPS,PetscInt,PetscInt),(eps,inner,blsize)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetRefinement_C",(EPS,PetscInt*,PetscInt*),(eps,inner,blsize)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,usest,2);
  CHKERRQ(PetscTryMethod(eps,"EPSCISSSetUseST_C",(EPS,PetscBool),(eps,usest)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidBoolPointer(usest,2);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetUseST_C",(EPS,PetscBool*),(eps,usest)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,quad,2);
  CHKERRQ(PetscTryMethod(eps,"EPSCISSSetQuadRule_C",(EPS,EPSCISSQuadRule),(eps,quad)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(quad,2);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetQuadRule_C",(EPS,EPSCISSQuadRule*),(eps,quad)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,extraction,2);
  CHKERRQ(PetscTryMethod(eps,"EPSCISSSetExtraction_C",(EPS,EPSCISSExtraction),(eps,extraction)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(extraction,2);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetExtraction_C",(EPS,EPSCISSExtraction*),(eps,extraction)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSCISSGetKSPs_CISS(EPS eps,PetscInt *nsolve,KSP **ksp)
{
  EPS_CISS         *ctx = (EPS_CISS*)eps->data;
  SlepcContourData contour;
  PetscInt         i,nsplit;
  PC               pc;
  MPI_Comm         child;

  PetscFunctionBegin;
  if (!ctx->contour) {  /* initialize contour data structure first */
    CHKERRQ(RGCanUseConjugates(eps->rg,ctx->isreal,&ctx->useconj));
    CHKERRQ(SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)eps,&ctx->contour));
  }
  contour = ctx->contour;
  if (!contour->ksp) {
    CHKERRQ(PetscMalloc1(contour->npoints,&contour->ksp));
    CHKERRQ(EPSGetST(eps,&eps->st));
    CHKERRQ(STGetSplitPreconditionerInfo(eps->st,&nsplit,NULL));
    CHKERRQ(PetscSubcommGetChild(contour->subcomm,&child));
    for (i=0;i<contour->npoints;i++) {
      CHKERRQ(KSPCreate(child,&contour->ksp[i]));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)contour->ksp[i],(PetscObject)eps,1));
      CHKERRQ(KSPSetOptionsPrefix(contour->ksp[i],((PetscObject)eps)->prefix));
      CHKERRQ(KSPAppendOptionsPrefix(contour->ksp[i],"eps_ciss_"));
      CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)contour->ksp[i]));
      CHKERRQ(PetscObjectSetOptions((PetscObject)contour->ksp[i],((PetscObject)eps)->options));
      CHKERRQ(KSPSetErrorIfNotConverged(contour->ksp[i],PETSC_TRUE));
      CHKERRQ(KSPSetTolerances(contour->ksp[i],SlepcDefaultTol(eps->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
      CHKERRQ(KSPGetPC(contour->ksp[i],&pc));
      if (nsplit) {
        CHKERRQ(KSPSetType(contour->ksp[i],KSPBCGS));
        CHKERRQ(PCSetType(pc,PCBJACOBI));
      } else {
        CHKERRQ(KSPSetType(contour->ksp[i],KSPPREONLY));
        CHKERRQ(PCSetType(pc,PCLU));
      }
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSCISSGetKSPs_C",(EPS,PetscInt*,KSP**),(eps,nsolve,ksp)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_CISS(EPS eps)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(BVDestroy(&ctx->S));
  CHKERRQ(BVDestroy(&ctx->V));
  CHKERRQ(BVDestroy(&ctx->Y));
  if (!ctx->usest) CHKERRQ(SlepcContourDataReset(ctx->contour));
  CHKERRQ(BVDestroy(&ctx->pV));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_CISS(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscReal         r3,r4;
  PetscInt          i,i1,i2,i3,i4,i5,i6,i7;
  PetscBool         b1,b2,flg,flg2,flg3,flg4,flg5,flg6;
  EPS_CISS          *ctx = (EPS_CISS*)eps->data;
  EPSCISSQuadRule   quad;
  EPSCISSExtraction extraction;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS CISS Options"));

    CHKERRQ(EPSCISSGetSizes(eps,&i1,&i2,&i3,&i4,&i5,&b1));
    CHKERRQ(PetscOptionsInt("-eps_ciss_integration_points","Number of integration points","EPSCISSSetSizes",i1,&i1,&flg));
    CHKERRQ(PetscOptionsInt("-eps_ciss_blocksize","Block size","EPSCISSSetSizes",i2,&i2,&flg2));
    CHKERRQ(PetscOptionsInt("-eps_ciss_moments","Moment size","EPSCISSSetSizes",i3,&i3,&flg3));
    CHKERRQ(PetscOptionsInt("-eps_ciss_partitions","Number of partitions","EPSCISSSetSizes",i4,&i4,&flg4));
    CHKERRQ(PetscOptionsInt("-eps_ciss_maxblocksize","Maximum block size","EPSCISSSetSizes",i5,&i5,&flg5));
    CHKERRQ(PetscOptionsBool("-eps_ciss_realmats","True if A and B are real","EPSCISSSetSizes",b1,&b1,&flg6));
    if (flg || flg2 || flg3 || flg4 || flg5 || flg6) CHKERRQ(EPSCISSSetSizes(eps,i1,i2,i3,i4,i5,b1));

    CHKERRQ(EPSCISSGetThreshold(eps,&r3,&r4));
    CHKERRQ(PetscOptionsReal("-eps_ciss_delta","Threshold for numerical rank","EPSCISSSetThreshold",r3,&r3,&flg));
    CHKERRQ(PetscOptionsReal("-eps_ciss_spurious_threshold","Threshold for the spurious eigenpairs","EPSCISSSetThreshold",r4,&r4,&flg2));
    if (flg || flg2) CHKERRQ(EPSCISSSetThreshold(eps,r3,r4));

    CHKERRQ(EPSCISSGetRefinement(eps,&i6,&i7));
    CHKERRQ(PetscOptionsInt("-eps_ciss_refine_inner","Number of inner iterative refinement iterations","EPSCISSSetRefinement",i6,&i6,&flg));
    CHKERRQ(PetscOptionsInt("-eps_ciss_refine_blocksize","Number of blocksize iterative refinement iterations","EPSCISSSetRefinement",i7,&i7,&flg2));
    if (flg || flg2) CHKERRQ(EPSCISSSetRefinement(eps,i6,i7));

    CHKERRQ(EPSCISSGetUseST(eps,&b2));
    CHKERRQ(PetscOptionsBool("-eps_ciss_usest","Use ST for linear solves","EPSCISSSetUseST",b2,&b2,&flg));
    if (flg) CHKERRQ(EPSCISSSetUseST(eps,b2));

    CHKERRQ(PetscOptionsEnum("-eps_ciss_quadrule","Quadrature rule","EPSCISSSetQuadRule",EPSCISSQuadRules,(PetscEnum)ctx->quad,(PetscEnum*)&quad,&flg));
    if (flg) CHKERRQ(EPSCISSSetQuadRule(eps,quad));

    CHKERRQ(PetscOptionsEnum("-eps_ciss_extraction","Extraction technique","EPSCISSSetExtraction",EPSCISSExtractions,(PetscEnum)ctx->extraction,(PetscEnum*)&extraction,&flg));
    if (flg) CHKERRQ(EPSCISSSetExtraction(eps,extraction));

  CHKERRQ(PetscOptionsTail());

  if (!eps->rg) CHKERRQ(EPSGetRG(eps,&eps->rg));
  CHKERRQ(RGSetFromOptions(eps->rg)); /* this is necessary here to set useconj */
  if (!ctx->contour || !ctx->contour->ksp) CHKERRQ(EPSCISSGetKSPs(eps,NULL,NULL));
  for (i=0;i<ctx->contour->npoints;i++) CHKERRQ(KSPSetFromOptions(ctx->contour->ksp[i]));
  CHKERRQ(PetscSubcommSetFromOptions(ctx->contour->subcomm));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_CISS(EPS eps)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(SlepcContourDataDestroy(&ctx->contour));
  CHKERRQ(PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma));
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetUseST_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetUseST_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetQuadRule_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetQuadRule_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetExtraction_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetExtraction_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetKSPs_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_CISS(EPS eps,PetscViewer viewer)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      isascii;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  sizes { integration points: %" PetscInt_FMT ", block size: %" PetscInt_FMT ", moment size: %" PetscInt_FMT ", partitions: %" PetscInt_FMT ", maximum block size: %" PetscInt_FMT " }\n",ctx->N,ctx->L,ctx->M,ctx->npart,ctx->L_max));
    if (ctx->isreal) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  exploiting symmetry of integration points\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  threshold { delta: %g, spurious threshold: %g }\n",(double)ctx->delta,(double)ctx->spurious_threshold));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  iterative refinement { inner: %" PetscInt_FMT ", blocksize: %" PetscInt_FMT " }\n",ctx->refine_inner, ctx->refine_blocksize));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  extraction: %s\n",EPSCISSExtractions[ctx->extraction]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  quadrature rule: %s\n",EPSCISSQuadRules[ctx->quad]));
    if (ctx->usest) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using ST for linear solves\n"));
    else {
      if (!ctx->contour || !ctx->contour->ksp) CHKERRQ(EPSCISSGetKSPs(eps,NULL,NULL));
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      if (ctx->npart>1 && ctx->contour->subcomm) {
        CHKERRQ(PetscViewerGetSubViewer(viewer,ctx->contour->subcomm->child,&sviewer));
        if (!ctx->contour->subcomm->color) CHKERRQ(KSPView(ctx->contour->ksp[0],sviewer));
        CHKERRQ(PetscViewerFlush(sviewer));
        CHKERRQ(PetscViewerRestoreSubViewer(viewer,ctx->contour->subcomm->child,&sviewer));
        CHKERRQ(PetscViewerFlush(viewer));
        /* extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
        CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
      } else CHKERRQ(KSPView(ctx->contour->ksp[0],viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_CISS(EPS eps)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      usest = ctx->usest;
  KSP            ksp;
  PC             pc;

  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) {
    if (!ctx->usest_set) usest = (ctx->npart>1)? PETSC_FALSE: PETSC_TRUE;
    if (usest) CHKERRQ(STSetType(eps->st,STSINVERT));
    else {
      /* we are not going to use ST, so avoid factorizing the matrix */
      CHKERRQ(STSetType(eps->st,STSHIFT));
      if (eps->isgeneralized) {
        CHKERRQ(STGetKSP(eps->st,&ksp));
        CHKERRQ(KSPGetPC(ksp,&pc));
        CHKERRQ(PCSetType(pc,PCNONE));
      }
    }
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_CISS(EPS eps)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
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

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",EPSCISSSetSizes_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",EPSCISSGetSizes_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",EPSCISSSetThreshold_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",EPSCISSGetThreshold_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",EPSCISSSetRefinement_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",EPSCISSGetRefinement_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetUseST_C",EPSCISSSetUseST_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetUseST_C",EPSCISSGetUseST_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetQuadRule_C",EPSCISSSetQuadRule_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetQuadRule_C",EPSCISSGetQuadRule_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetExtraction_C",EPSCISSSetExtraction_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetExtraction_C",EPSCISSGetExtraction_CISS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetKSPs_C",EPSCISSGetKSPs_CISS));

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
