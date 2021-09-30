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

       [1] J. Asakura, T. Sakurai, H. Tadano, T. Ikegami, K. Kimura, "A
           numerical method for nonlinear eigenvalue problems using contour
           integrals", JSIAM Lett. 1:52-55, 2009.

       [2] S. Yokota and T. Sakurai, "A projection method for nonlinear
           eigenvalue problems using contour integrals", JSIAM Lett.
           5:41-44, 2013.
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepc/private/slepccontour.h>

typedef struct _n_nep_ciss_project *NEP_CISS_PROJECT;
typedef struct {
  /* parameters */
  PetscInt          N;             /* number of integration points (32) */
  PetscInt          L;             /* block size (16) */
  PetscInt          M;             /* moment degree (N/4 = 4) */
  PetscReal         delta;         /* threshold of singular value (1e-12) */
  PetscInt          L_max;         /* maximum number of columns of the source matrix V */
  PetscReal         spurious_threshold; /* discard spurious eigenpairs */
  PetscBool         isreal;        /* T(z) is real for real z */
  PetscInt          npart;         /* number of partitions */
  PetscInt          refine_inner;
  PetscInt          refine_blocksize;
  NEPCISSExtraction extraction;
  /* private data */
  SlepcContourData  contour;
  PetscReal         *sigma;        /* threshold for numerical rank */
  PetscScalar       *weight;
  PetscScalar       *omega;
  PetscScalar       *pp;
  BV                V;
  BV                S;
  BV                Y;
  PetscBool         useconj;
  Mat               T,J;           /* auxiliary matrices when using subcomm */
  BV                pV;
  NEP_CISS_PROJECT  dsctxf;
  PetscObjectId     rgid;
  PetscObjectState  rgstate;
} NEP_CISS;

struct _n_nep_ciss_project {
  NEP  nep;
  BV   Q;
};

static PetscErrorCode NEPContourDSComputeMatrix(DS ds,PetscScalar lambda,PetscBool deriv,DSMatType mat,void *ctx)
{
  NEP_CISS_PROJECT proj = (NEP_CISS_PROJECT)ctx;
  PetscErrorCode   ierr;
  Mat              M,fun;

  PetscFunctionBegin;
  if (!deriv) {
    ierr = NEPComputeFunction(proj->nep,lambda,proj->nep->function,proj->nep->function);CHKERRQ(ierr);
    fun = proj->nep->function;
  } else {
    ierr = NEPComputeJacobian(proj->nep,lambda,proj->nep->jacobian);CHKERRQ(ierr);
    fun = proj->nep->jacobian;
  }
  ierr = DSGetMat(ds,mat,&M);CHKERRQ(ierr);
  ierr = BVMatProject(proj->Q,fun,proj->Q,M);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,mat,&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPComputeFunctionSubcomm(NEP nep,PetscScalar lambda,Mat T,PetscBool deriv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  switch (nep->fui) {
  case NEP_USER_INTERFACE_CALLBACK:
    SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Partitions are not supported with callbacks");
  case NEP_USER_INTERFACE_SPLIT:
    ierr = MatZeroEntries(T);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      if (!deriv) {
        ierr = FNEvaluateFunction(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      } else {
        ierr = FNEvaluateDerivative(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      }
      ierr = MatAXPY(T,alpha,ctx->contour->pA[i],nep->mstr);CHKERRQ(ierr);
    }
    break;
  }
  PetscFunctionReturn(0);
}

/*
  Y_i = F(z_i)^{-1}Fp(z_i)V for every integration point, Y=[Y_i] is in the context
*/
static PetscErrorCode NEPCISSSolveSystem(NEP nep,Mat T,Mat dT,BV V,PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode   ierr;
  NEP_CISS         *ctx = (NEP_CISS*)nep->data;
  SlepcContourData contour = ctx->contour;
  PetscInt         i,p_id;
  Mat              kspMat,MV,BMV=NULL,MC;

  PetscFunctionBegin;
  if (!ctx->contour || !ctx->contour->ksp) { ierr = NEPCISSGetKSPs(nep,NULL,NULL);CHKERRQ(ierr); }
  ierr = BVSetActiveColumns(V,L_start,L_end);CHKERRQ(ierr);
  ierr = BVGetMat(V,&MV);CHKERRQ(ierr);
  for (i=0;i<contour->npoints;i++) {
    p_id = i*contour->subcomm->n + contour->subcomm->color;
    if (initksp) {
      if (contour->subcomm->n == 1) {
        ierr = NEPComputeFunction(nep,ctx->omega[p_id],T,T);CHKERRQ(ierr);
      } else {
        ierr = NEPComputeFunctionSubcomm(nep,ctx->omega[p_id],T,PETSC_FALSE);CHKERRQ(ierr);
      }
      ierr = MatDuplicate(T,MAT_COPY_VALUES,&kspMat);CHKERRQ(ierr);
      ierr = KSPSetOperators(contour->ksp[i],kspMat,kspMat);CHKERRQ(ierr);
      ierr = MatDestroy(&kspMat);CHKERRQ(ierr);
    }
    if (contour->subcomm->n==1) {
      ierr = NEPComputeJacobian(nep,ctx->omega[p_id],dT);CHKERRQ(ierr);
    } else {
      ierr = NEPComputeFunctionSubcomm(nep,ctx->omega[p_id],dT,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(ctx->Y,i*ctx->L_max+L_start,i*ctx->L_max+L_end);CHKERRQ(ierr);
    ierr = BVGetMat(ctx->Y,&MC);CHKERRQ(ierr);
    if (!i) {
      ierr = MatProductCreate(dT,MV,NULL,&BMV);CHKERRQ(ierr);
      ierr = MatProductSetType(BMV,MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFromOptions(BMV);CHKERRQ(ierr);
      ierr = MatProductSymbolic(BMV);CHKERRQ(ierr);
    }
    ierr = MatProductNumeric(BMV);CHKERRQ(ierr);
    ierr = KSPMatSolve(contour->ksp[i],BMV,MC);CHKERRQ(ierr);
    ierr = BVRestoreMat(ctx->Y,&MC);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&BMV);CHKERRQ(ierr);
  ierr = BVRestoreMat(V,&MV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetUp_CISS(NEP nep)
{
  PetscErrorCode   ierr;
  NEP_CISS         *ctx = (NEP_CISS*)nep->data;
  SlepcContourData contour;
  PetscInt         nwork;
  PetscBool        istrivial,isellipse,flg;
  NEP_CISS_PROJECT dsctxf;
  PetscObjectId    id;
  PetscObjectState state;
  Vec              v0;

  PetscFunctionBegin;
  if (nep->ncv==PETSC_DEFAULT) nep->ncv = ctx->L_max*ctx->M;
  else {
    ctx->L_max = nep->ncv/ctx->M;
    if (!ctx->L_max) {
      ctx->L_max = 1;
      nep->ncv = ctx->L_max*ctx->M;
    }
  }
  ctx->L = PetscMin(ctx->L,ctx->L_max);
  if (nep->max_it==PETSC_DEFAULT) nep->max_it = 5;
  if (nep->mpd==PETSC_DEFAULT) nep->mpd = nep->ncv;
  if (!nep->which) nep->which = NEP_ALL;
  if (nep->which!=NEP_ALL) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver supports only computing all eigenvalues");
  NEPCheckUnsupported(nep,NEP_FEATURE_STOPPING | NEP_FEATURE_TWOSIDED);

  /* check region */
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"CISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  ierr = RGGetComplement(nep->rg,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"A region with complement flag set is not allowed");
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (!isellipse) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Currently only implemented for elliptic regions");

  /* if the region has changed, then reset contour data */
  ierr = PetscObjectGetId((PetscObject)nep->rg,&id);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)nep->rg,&state);CHKERRQ(ierr);
  if (ctx->rgid && (id != ctx->rgid || state != ctx->rgstate)) {
    ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
    ierr = PetscInfo(nep,"Resetting the contour data structure due to a change of region\n");CHKERRQ(ierr);
    ctx->rgid = id; ctx->rgstate = state;
  }

  /* create contour data structure */
  if (!ctx->contour) {
    ierr = RGCanUseConjugates(nep->rg,ctx->isreal,&ctx->useconj);CHKERRQ(ierr);
    ierr = SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)nep,&ctx->contour);CHKERRQ(ierr);
  }

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  if (ctx->weight) { ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr); }
  ierr = PetscMalloc4(ctx->N,&ctx->weight,ctx->N,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* allocate basis vectors */
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDuplicateResize(nep->V,ctx->L_max*ctx->M,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDuplicateResize(nep->V,ctx->L_max,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->V);CHKERRQ(ierr);

  contour = ctx->contour;
  ierr = SlepcContourRedundantMat(contour,nep->nt,nep->A);CHKERRQ(ierr);
  if (contour->pA) {
    if (!ctx->T) {
      ierr = MatDuplicate(contour->pA[0],MAT_DO_NOT_COPY_VALUES,&ctx->T);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->T);CHKERRQ(ierr);
    }
    if (!ctx->J) {
      ierr = MatDuplicate(contour->pA[0],MAT_DO_NOT_COPY_VALUES,&ctx->J);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->J);CHKERRQ(ierr);
    }
    ierr = BVGetColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = SlepcContourScatterCreate(contour,v0);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
    ierr = BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->pV);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->pV,contour->xsub,nep->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->pV);CHKERRQ(ierr);
    ierr = BVResize(ctx->pV,ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->pV);CHKERRQ(ierr);
  }

  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  if (contour->pA) {
    ierr = BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->Y);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->Y,contour->xsub,nep->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->Y);CHKERRQ(ierr);
    ierr = BVResize(ctx->Y,contour->npoints*ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicateResize(nep->V,contour->npoints*ctx->L_max,&ctx->Y);CHKERRQ(ierr);
  }

  if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
    ierr = DSSetType(nep->ds,DSGNHEP);CHKERRQ(ierr);
  } else if (ctx->extraction == NEP_CISS_EXTRACTION_CAA) {
    ierr = DSSetType(nep->ds,DSNHEP);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(nep->ds,DSNEP);CHKERRQ(ierr);
    ierr = DSSetMethod(nep->ds,1);CHKERRQ(ierr);
    ierr = DSNEPSetRG(nep->ds,nep->rg);CHKERRQ(ierr);
    if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
      ierr = DSNEPSetFN(nep->ds,nep->nt,nep->f);CHKERRQ(ierr);
    } else {
      ierr = PetscNew(&dsctxf);CHKERRQ(ierr);
      ierr = DSNEPSetComputeMatrixFunction(nep->ds,NEPContourDSComputeMatrix,dsctxf);CHKERRQ(ierr);
      dsctxf->nep = nep;
      ctx->dsctxf = dsctxf;
    }
  }
  ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
  nwork = (nep->fui==NEP_USER_INTERFACE_SPLIT)? 2: 1;
  ierr = NEPSetWorkVecs(nep,nwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_CISS(NEP nep)
{
  PetscErrorCode   ierr;
  NEP_CISS         *ctx = (NEP_CISS*)nep->data;
  SlepcContourData contour = ctx->contour;
  Mat              X,M,E;
  PetscInt         i,j,ld,L_add=0,nv=0,L_base=ctx->L,inner,*inside;
  PetscScalar      *Mu,*H0,*H1,*rr,*temp,center;
  PetscReal        error,max_error,radius,rgscale,est_eig,eta;
  PetscBool        isellipse,*fl1;
  Vec              si;
  SlepcSC          sc;
  PetscRandom      rand;

  PetscFunctionBegin;
  ierr = DSSetFromOptions(nep->ds);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(nep->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSGetLeadingDimension(nep->ds,&ld);CHKERRQ(ierr);
  ierr = RGComputeQuadrature(nep->rg,RG_QUADRULE_TRAPEZOIDAL,ctx->N,ctx->omega,ctx->pp,ctx->weight);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
  ierr = BVGetRandomContext(ctx->V,&rand);CHKERRQ(ierr);
  if (contour->pA) {
    ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
    ierr = NEPCISSSolveSystem(nep,ctx->T,ctx->J,ctx->pV,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = NEPCISSSolveSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = BVTraceQuadrature(ctx->Y,ctx->V,ctx->L,ctx->L_max,ctx->weight,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj,&est_eig);CHKERRQ(ierr);
    ierr = PetscInfo1(nep,"Estimated eigenvalue count: %f\n",(double)est_eig);CHKERRQ(ierr);
    eta = PetscPowReal(10.0,-PetscLog10Real(nep->tol)/ctx->N);
    L_add = PetscMax(0,(PetscInt)PetscCeilReal((est_eig*eta)/ctx->M)-ctx->L);
    if (L_add>ctx->L_max-ctx->L) {
      ierr = PetscInfo(nep,"Number of eigenvalues inside the contour path may be too large\n");CHKERRQ(ierr);
      L_add = ctx->L_max-ctx->L;
    }
  }
  /* Updates L after estimate the number of eigenvalue */
  if (L_add>0) {
    ierr = PetscInfo2(nep,"Changing L %D -> %D by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (contour->pA) {
      ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
      ierr = NEPCISSSolveSystem(nep,ctx->T,ctx->J,ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = NEPCISSSolveSystem(nep,nep->function,nep->jacobian,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
  }

  ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  for (i=0;i<ctx->refine_blocksize;i++) {
    ierr = BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
    ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(NEP_CISS_SVD,nep,0,0,0);CHKERRQ(ierr);
    ierr = SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(NEP_CISS_SVD,nep,0,0,0);CHKERRQ(ierr);
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    ierr = PetscInfo2(nep,"Changing L %D -> %D by SVD(H0)\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (contour->pA) {
      ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
      ierr = NEPCISSSolveSystem(nep,ctx->T,ctx->J,ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = NEPCISSSolveSystem(nep,nep->function,nep->jacobian,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
    if (L_add) {
      ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
      ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
    }
  }

  ierr = RGGetScale(nep->rg,&rgscale);CHKERRQ(ierr);
  ierr = RGEllipseGetParameters(nep->rg,&center,&radius,NULL);CHKERRQ(ierr);

  if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
    ierr = PetscMalloc1(ctx->L*ctx->M*ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  }

  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        ierr = BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(NEP_CISS_SVD,nep,0,0,0);CHKERRQ(ierr);
        ierr = SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(NEP_CISS_SVD,nep,0,0,0);CHKERRQ(ierr);
      } else {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        /* compute SVD of S */
        ierr = BVSVDAndRank(ctx->S,ctx->M,ctx->L,ctx->delta,(ctx->extraction==NEP_CISS_EXTRACTION_CAA)?BV_SVD_METHOD_QR_CAA:BV_SVD_METHOD_QR,H0,ctx->sigma,&nv);CHKERRQ(ierr);
      }
      if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        if (contour->pA) {
          ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
          ierr = NEPCISSSolveSystem(nep,ctx->T,ctx->J,ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = NEPCISSSolveSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        }
      } else break;
    }
    nep->nconv = 0;
    if (nv == 0) { nep->reason = NEP_CONVERGED_TOL; break; }
    else {
      /* Extracting eigenpairs */
      ierr = DSSetDimensions(nep->ds,nv,0,0);CHKERRQ(ierr);
      ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
      if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
        ierr = CISS_BlockHankel(Mu,1,ctx->L,ctx->M,H1);CHKERRQ(ierr);
        ierr = DSGetArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        for (j=0;j<nv;j++)
          for (i=0;i<nv;i++)
            temp[i+j*ld] = H1[i+j*ctx->L*ctx->M];
        ierr = DSRestoreArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        ierr = DSGetArray(nep->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
        for (j=0;j<nv;j++)
          for (i=0;i<nv;i++)
            temp[i+j*ld] = H0[i+j*ctx->L*ctx->M];
        ierr = DSRestoreArray(nep->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
      } else if (ctx->extraction == NEP_CISS_EXTRACTION_CAA) {
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = DSGetArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        for (i=0;i<nv;i++) {
          ierr = PetscArraycpy(temp+i*ld,H0+i*nv,nv);CHKERRQ(ierr);
        }
        ierr = DSRestoreArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
      } else {
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
          for (i=0;i<nep->nt;i++) {
            ierr = DSGetMat(nep->ds,DSMatExtra[i],&E);CHKERRQ(ierr);
            ierr = BVMatProject(ctx->S,nep->A[i],ctx->S,E);CHKERRQ(ierr);
            ierr = DSRestoreMat(nep->ds,DSMatExtra[i],&E);CHKERRQ(ierr);
          }
        } else { ctx->dsctxf->Q = ctx->S; }
      }
      ierr = DSSolve(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
      ierr = DSSynchronize(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
      ierr = DSGetDimensions(nep->ds,NULL,NULL,NULL,&nv);CHKERRQ(ierr);
      if (ctx->extraction == NEP_CISS_EXTRACTION_CAA || ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        for (i=0;i<nv;i++) {
          nep->eigr[i] = (nep->eigr[i]*radius+center)*rgscale;
        }
      }
      ierr = PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr);CHKERRQ(ierr);
      ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
      ierr = DSGetMat(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = SlepcCISS_isGhost(X,nv,ctx->sigma,ctx->spurious_threshold,fl1);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      ierr = RGCheckInside(nep->rg,nv,nep->eigr,nep->eigi,inside);CHKERRQ(ierr);
      for (i=0;i<nv;i++) {
        if (fl1[i] && inside[i]>=0) {
          rr[i] = 1.0;
          nep->nconv++;
        } else rr[i] = 0.0;
      }
      ierr = DSSort(nep->ds,nep->eigr,nep->eigi,rr,NULL,&nep->nconv);CHKERRQ(ierr);
      ierr = DSSynchronize(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
      if (ctx->extraction == NEP_CISS_EXTRACTION_CAA || ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        for (i=0;i<nv;i++) nep->eigr[i] = (nep->eigr[i]*radius+center)*rgscale;
      }
      ierr = PetscFree3(fl1,inside,rr);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(nep->V,0,nv);CHKERRQ(ierr);
      ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
      if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,nep->V);CHKERRQ(ierr);
        ierr = DSGetMat(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
        ierr = BVMultInPlace(ctx->S,X,0,nep->nconv);CHKERRQ(ierr);
        ierr = BVMultInPlace(nep->V,X,0,nep->nconv);CHKERRQ(ierr);
        ierr = MatDestroy(&X);CHKERRQ(ierr);
      } else {
        ierr = DSGetMat(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
        ierr = BVMultInPlace(ctx->S,X,0,nep->nconv);CHKERRQ(ierr);
        ierr = MatDestroy(&X);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,nep->V);CHKERRQ(ierr);
      }
      max_error = 0.0;
      for (i=0;i<nep->nconv;i++) {
        ierr = BVGetColumn(nep->V,i,&si);CHKERRQ(ierr);
        ierr = VecNormalize(si,NULL);CHKERRQ(ierr);
        ierr = NEPComputeResidualNorm_Private(nep,PETSC_FALSE,nep->eigr[i],si,nep->work,&error);CHKERRQ(ierr);
        ierr = (*nep->converged)(nep,nep->eigr[i],0,error,&error,nep->convergedctx);CHKERRQ(ierr);
        ierr = BVRestoreColumn(nep->V,i,&si);CHKERRQ(ierr);
        max_error = PetscMax(max_error,error);
      }
      if (max_error <= nep->tol) nep->reason = NEP_CONVERGED_TOL;
      else if (nep->its > nep->max_it) nep->reason = NEP_DIVERGED_ITS;
      else {
        if (nep->nconv > ctx->L) nv = nep->nconv;
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
          ierr = NEPCISSSolveSystem(nep,ctx->T,ctx->J,ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = NEPCISSSolveSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
  if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
    ierr = PetscFree(H1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSSetSizes_CISS(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       oN,oL,oM,oLmax,onpart;

  PetscFunctionBegin;
  oN = ctx->N;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
    if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
  } else {
    if (ip<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
    if (ip%2) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
    if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
  }
  oL = ctx->L;
  if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
    ctx->L = 16;
  } else {
    if (bs<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
    ctx->L = bs;
  }
  oM = ctx->M;
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    if (ms<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = PetscMax(ms,2);
  }
  onpart = ctx->npart;
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->npart = 1;
  } else {
    if (npart<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
    ctx->npart = npart;
  }
  oLmax = ctx->L_max;
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L_max = 64;
  } else {
    if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    ctx->L_max = PetscMax(bsmax,ctx->L);
  }
  if (onpart != ctx->npart || oN != ctx->N || realmats != ctx->isreal) {
    ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
    ierr = PetscInfo(nep,"Resetting the contour data structure due to a change of parameters\n");CHKERRQ(ierr);
    nep->state = NEP_STATE_INITIAL;
  }
  ctx->isreal = realmats;
  if (oL != ctx->L || oM != ctx->M || oLmax != ctx->L_max) nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSSetSizes - Sets the values of various size parameters in the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep   - the nonlinear eigensolver context
.  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - T(z) is real for real z

   Options Database Keys:
+  -nep_ciss_integration_points - Sets the number of integration points
.  -nep_ciss_blocksize - Sets the block size
.  -nep_ciss_moments - Sets the moment size
.  -nep_ciss_partitions - Sets the number of partitions
.  -nep_ciss_maxblocksize - Sets the maximum block size
-  -nep_ciss_realmats - T(z) is real for real z

   Notes:
   The default number of partitions is 1. This means the internal KSP object is shared
   among all processes of the NEP communicator. Otherwise, the communicator is split
   into npart communicators, so that npart KSP solves proceed simultaneously.

   The realmats flag can be set to true when T(.) is guaranteed to be real
   when the argument is a real value, for example, when all matrices in
   the split form are real. When set to true, the solver avoids some computations.

   Level: advanced

.seealso: NEPCISSGetSizes()
@*/
PetscErrorCode NEPCISSSetSizes(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,ip,2);
  PetscValidLogicalCollectiveInt(nep,bs,3);
  PetscValidLogicalCollectiveInt(nep,ms,4);
  PetscValidLogicalCollectiveInt(nep,npart,5);
  PetscValidLogicalCollectiveInt(nep,bsmax,6);
  PetscValidLogicalCollectiveBool(nep,realmats,7);
  ierr = PetscTryMethod(nep,"NEPCISSSetSizes_C",(NEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(nep,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetSizes_CISS(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

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
   NEPCISSGetSizes - Gets the values of various size parameters in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - T(z) is real for real z

   Level: advanced

.seealso: NEPCISSSetSizes()
@*/
PetscErrorCode NEPCISSGetSizes(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetSizes_C",(NEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(nep,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSSetThreshold_CISS(NEP nep,PetscReal delta,PetscReal spur)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (delta == PETSC_DEFAULT) {
    ctx->delta = SLEPC_DEFAULT_TOL*1e-4;
  } else {
    if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
    ctx->delta = delta;
  }
  if (spur == PETSC_DEFAULT) {
    ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  } else {
    if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
    ctx->spurious_threshold = spur;
  }
  PetscFunctionReturn(0);
}

/*@
   NEPCISSSetThreshold - Sets the values of various threshold parameters in
   the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep   - the nonlinear eigensolver context
.  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Options Database Keys:
+  -nep_ciss_delta - Sets the delta
-  -nep_ciss_spurious_threshold - Sets the spurious threshold

   Level: advanced

.seealso: NEPCISSGetThreshold()
@*/
PetscErrorCode NEPCISSSetThreshold(NEP nep,PetscReal delta,PetscReal spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,delta,2);
  PetscValidLogicalCollectiveReal(nep,spur,3);
  ierr = PetscTryMethod(nep,"NEPCISSSetThreshold_C",(NEP,PetscReal,PetscReal),(nep,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetThreshold_CISS(NEP nep,PetscReal *delta,PetscReal *spur)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (delta) *delta = ctx->delta;
  if (spur)  *spur = ctx->spurious_threshold;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetThreshold - Gets the values of various threshold parameters in
   the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Level: advanced

.seealso: NEPCISSSetThreshold()
@*/
PetscErrorCode NEPCISSGetThreshold(NEP nep,PetscReal *delta,PetscReal *spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetThreshold_C",(NEP,PetscReal*,PetscReal*),(nep,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSSetRefinement_CISS(NEP nep,PetscInt inner,PetscInt blsize)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (inner == PETSC_DEFAULT) {
    ctx->refine_inner = 0;
  } else {
    if (inner<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The refine inner argument must be >= 0");
    ctx->refine_inner = inner;
  }
  if (blsize == PETSC_DEFAULT) {
    ctx->refine_blocksize = 0;
  } else {
    if (blsize<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The refine blocksize argument must be >= 0");
    ctx->refine_blocksize = blsize;
  }
  PetscFunctionReturn(0);
}

/*@
   NEPCISSSetRefinement - Sets the values of various refinement parameters
   in the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Options Database Keys:
+  -nep_ciss_refine_inner - Sets number of inner iterations
-  -nep_ciss_refine_blocksize - Sets number of blocksize iterations

   Level: advanced

.seealso: NEPCISSGetRefinement()
@*/
PetscErrorCode NEPCISSSetRefinement(NEP nep,PetscInt inner,PetscInt blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,inner,2);
  PetscValidLogicalCollectiveInt(nep,blsize,3);
  ierr = PetscTryMethod(nep,"NEPCISSSetRefinement_C",(NEP,PetscInt,PetscInt),(nep,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetRefinement_CISS(NEP nep,PetscInt *inner,PetscInt *blsize)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (inner)  *inner = ctx->refine_inner;
  if (blsize) *blsize = ctx->refine_blocksize;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetRefinement - Gets the values of various refinement parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Level: advanced

.seealso: NEPCISSSetRefinement()
@*/
PetscErrorCode NEPCISSGetRefinement(NEP nep, PetscInt *inner, PetscInt *blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetRefinement_C",(NEP,PetscInt*,PetscInt*),(nep,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSSetExtraction_CISS(NEP nep,NEPCISSExtraction extraction)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ctx->extraction != extraction) {
    ctx->extraction = extraction;
    nep->state      = NEP_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   NEPCISSSetExtraction - Sets the extraction technique used in the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep        - the nonlinear eigensolver context
-  extraction - the extraction technique

   Options Database Key:
.  -nep_ciss_extraction - Sets the extraction technique (either 'ritz', 'hankel' or 'caa')

   Notes:
   By default, the Rayleigh-Ritz extraction is used (NEP_CISS_EXTRACTION_RITZ).

   If the 'hankel' or the 'caa' option is specified (NEP_CISS_EXTRACTION_HANKEL or
   NEP_CISS_EXTRACTION_CAA), then the Block Hankel method, or the Communication-avoiding
   Arnoldi method, respectively, is used for extracting eigenpairs.

   Level: advanced

.seealso: NEPCISSGetExtraction(), NEPCISSExtraction
@*/
PetscErrorCode NEPCISSSetExtraction(NEP nep,NEPCISSExtraction extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,extraction,2);
  ierr = PetscTryMethod(nep,"NEPCISSSetExtraction_C",(NEP,NEPCISSExtraction),(nep,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetExtraction_CISS(NEP nep,NEPCISSExtraction *extraction)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  *extraction = ctx->extraction;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetExtraction - Gets the extraction technique used in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
.  extraction - extraction technique

   Level: advanced

.seealso: NEPCISSSetExtraction() NEPCISSExtraction
@*/
PetscErrorCode NEPCISSGetExtraction(NEP nep,NEPCISSExtraction *extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(extraction,2);
  ierr = PetscUseMethod(nep,"NEPCISSGetExtraction_C",(NEP,NEPCISSExtraction*),(nep,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetKSPs_CISS(NEP nep,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode   ierr;
  NEP_CISS         *ctx = (NEP_CISS*)nep->data;
  SlepcContourData contour;
  PetscInt         i;
  PC               pc;

  PetscFunctionBegin;
  if (!ctx->contour) {  /* initialize contour data structure first */
    ierr = RGCanUseConjugates(nep->rg,ctx->isreal,&ctx->useconj);CHKERRQ(ierr);
    ierr = SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)nep,&ctx->contour);CHKERRQ(ierr);
  }
  contour = ctx->contour;
  if (!contour->ksp) {
    ierr = PetscMalloc1(contour->npoints,&contour->ksp);CHKERRQ(ierr);
    for (i=0;i<contour->npoints;i++) {
      ierr = KSPCreate(PetscSubcommChild(contour->subcomm),&contour->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)contour->ksp[i],(PetscObject)nep,1);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(contour->ksp[i],((PetscObject)nep)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(contour->ksp[i],"nep_ciss_");CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)contour->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptions((PetscObject)contour->ksp[i],((PetscObject)nep)->options);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(contour->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(contour->ksp[i],SlepcDefaultTol(nep->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
   NEPCISSGetKSPs - Retrieve the array of linear solver objects associated with
   the CISS solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameters:
+  nsolve - number of solver objects
-  ksp - array of linear solver object

   Notes:
   The number of KSP solvers is equal to the number of integration points divided by
   the number of partitions. This value is halved in the case of real matrices with
   a region centered at the real axis.

   Level: advanced

.seealso: NEPCISSSetSizes()
@*/
PetscErrorCode NEPCISSGetKSPs(NEP nep,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetKSPs_C",(NEP,PetscInt*,KSP**),(nep,nsolve,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPReset_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  ierr = SlepcContourDataReset(ctx->contour);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->T);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
  if (ctx->extraction == NEP_CISS_EXTRACTION_RITZ && nep->fui==NEP_USER_INTERFACE_CALLBACK) {
    ierr = PetscFree(ctx->dsctxf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetFromOptions_CISS(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode    ierr;
  NEP_CISS          *ctx = (NEP_CISS*)nep->data;
  PetscReal         r1,r2;
  PetscInt          i,i1,i2,i3,i4,i5,i6,i7;
  PetscBool         b1,flg,flg2,flg3,flg4,flg5,flg6;
  NEPCISSExtraction extraction;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP CISS Options");CHKERRQ(ierr);

    ierr = NEPCISSGetSizes(nep,&i1,&i2,&i3,&i4,&i5,&b1);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_integration_points","Number of integration points","NEPCISSSetSizes",i1,&i1,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_blocksize","Block size","NEPCISSSetSizes",i2,&i2,&flg2);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_moments","Moment size","NEPCISSSetSizes",i3,&i3,&flg3);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_partitions","Number of partitions","NEPCISSSetSizes",i4,&i4,&flg4);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_maxblocksize","Maximum block size","NEPCISSSetSizes",i5,&i5,&flg5);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-nep_ciss_realmats","True if T(z) is real for real z","NEPCISSSetSizes",b1,&b1,&flg6);CHKERRQ(ierr);
    if (flg || flg2 || flg3 || flg4 || flg5 || flg6) { ierr = NEPCISSSetSizes(nep,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr); }

    ierr = NEPCISSGetThreshold(nep,&r1,&r2);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-nep_ciss_delta","Threshold for numerical rank","NEPCISSSetThreshold",r1,&r1,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-nep_ciss_spurious_threshold","Threshold for the spurious eigenpairs","NEPCISSSetThreshold",r2,&r2,&flg2);CHKERRQ(ierr);
    if (flg || flg2) { ierr = NEPCISSSetThreshold(nep,r1,r2);CHKERRQ(ierr); }

    ierr = NEPCISSGetRefinement(nep,&i6,&i7);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_refine_inner","Number of inner iterative refinement iterations","NEPCISSSetRefinement",i6,&i6,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_refine_blocksize","Number of blocksize iterative refinement iterations","NEPCISSSetRefinement",i7,&i7,&flg2);CHKERRQ(ierr);
    if (flg || flg2) { ierr = NEPCISSSetRefinement(nep,i6,i7);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-nep_ciss_extraction","Extraction technique","NEPCISSSetExtraction",NEPCISSExtractions,(PetscEnum)ctx->extraction,(PetscEnum*)&extraction,&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPCISSSetExtraction(nep,extraction);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGSetFromOptions(nep->rg);CHKERRQ(ierr); /* this is necessary here to set useconj */
  if (!ctx->contour || !ctx->contour->ksp) { ierr = NEPCISSGetKSPs(nep,NULL,NULL);CHKERRQ(ierr); }
  for (i=0;i<ctx->contour->npoints;i++) {
    ierr = KSPSetFromOptions(ctx->contour->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscSubcommSetFromOptions(ctx->contour->subcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDestroy_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
  ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetKSPs_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPView_CISS(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
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
    ierr = PetscViewerASCIIPrintf(viewer,"  iterative refinement  { inner: %D, blocksize: %D }\n",ctx->refine_inner, ctx->refine_blocksize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  extraction: %s\n",NEPCISSExtractions[ctx->extraction]);CHKERRQ(ierr);
    if (!ctx->contour || !ctx->contour->ksp) { ierr = NEPCISSGetKSPs(nep,NULL,NULL);CHKERRQ(ierr); }
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
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode NEPCreate_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = ctx;
  /* set default values of parameters */
  ctx->N                  = 32;
  ctx->L                  = 16;
  ctx->M                  = ctx->N/4;
  ctx->delta              = SLEPC_DEFAULT_TOL*1e-4;
  ctx->L_max              = 64;
  ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  ctx->isreal             = PETSC_FALSE;
  ctx->npart              = 1;

  nep->useds = PETSC_TRUE;

  nep->ops->solve          = NEPSolve_CISS;
  nep->ops->setup          = NEPSetUp_CISS;
  nep->ops->setfromoptions = NEPSetFromOptions_CISS;
  nep->ops->reset          = NEPReset_CISS;
  nep->ops->destroy        = NEPDestroy_CISS;
  nep->ops->view           = NEPView_CISS;

  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NEPCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NEPCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NEPCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NEPCISSGetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRefinement_C",NEPCISSSetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRefinement_C",NEPCISSGetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetExtraction_C",NEPCISSSetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetExtraction_C",NEPCISSGetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetKSPs_C",NEPCISSGetKSPs_CISS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

