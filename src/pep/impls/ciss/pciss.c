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
           numerical method for polynomial eigenvalue problems using contour
           integral", Japan J. Indust. Appl. Math. 27:73-90, 2010.
*/

#include <slepc/private/pepimpl.h>         /*I "slepcpep.h" I*/
#include <slepc/private/slepccontour.h>

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
  PEPCISSExtraction extraction;
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
  Mat               T,J;           /* auxiliary matrices */
  BV                pV;
  PetscObjectId     rgid;
  PetscObjectState  rgstate;
} PEP_CISS;

static PetscErrorCode PEPComputeFunction(PEP pep,PetscScalar lambda,Mat T,PetscBool deriv)
{
  PetscErrorCode   ierr;
  PetscInt         i;
  PetscScalar      *coeff;
  Mat              *A;
  MatStructure     str;
  PEP_CISS         *ctx = (PEP_CISS*)pep->data;
  SlepcContourData contour = ctx->contour;

  PetscFunctionBegin;
  A = (contour->pA)?contour->pA:pep->A;
  ierr = PetscMalloc1(pep->nmat,&coeff);CHKERRQ(ierr);
  if (deriv) {
    ierr = PEPEvaluateBasisDerivative(pep,lambda,0,coeff,NULL);CHKERRQ(ierr);
  } else {
    ierr = PEPEvaluateBasis(pep,lambda,0,coeff,NULL);CHKERRQ(ierr);
  }
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = MatZeroEntries(T);CHKERRQ(ierr);
  i = deriv?1:0;
  for (;i<pep->nmat;i++) {
    ierr = MatAXPY(T,coeff[i],A[i],str);CHKERRQ(ierr);
  }
  ierr = PetscFree(coeff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Y_i = F(z_i)^{-1}Fp(z_i)V for every integration point, Y=[Y_i] is in the context
*/
static PetscErrorCode PEPCISSSolveSystem(PEP pep,Mat T,Mat dT,BV V,PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode   ierr;
  PEP_CISS         *ctx = (PEP_CISS*)pep->data;
  SlepcContourData contour = ctx->contour;
  PetscInt         i,p_id;
  Mat              kspMat,MV,BMV=NULL,MC;

  PetscFunctionBegin;
  if (!ctx->contour || !ctx->contour->ksp) { ierr = PEPCISSGetKSPs(pep,NULL,NULL);CHKERRQ(ierr); }
  ierr = BVSetActiveColumns(V,L_start,L_end);CHKERRQ(ierr);
  ierr = BVGetMat(V,&MV);CHKERRQ(ierr);
  for (i=0;i<contour->npoints;i++) {
    p_id = i*contour->subcomm->n + contour->subcomm->color;
    if (initksp) {
      ierr = PEPComputeFunction(pep,ctx->omega[p_id],T,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatDuplicate(T,MAT_COPY_VALUES,&kspMat);CHKERRQ(ierr);
      ierr = KSPSetOperators(contour->ksp[i],kspMat,kspMat);CHKERRQ(ierr);
      ierr = MatDestroy(&kspMat);CHKERRQ(ierr);
    }
    ierr = PEPComputeFunction(pep,ctx->omega[p_id],dT,PETSC_TRUE);CHKERRQ(ierr);
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

PetscErrorCode PEPSetUp_CISS(PEP pep)
{
  PetscErrorCode   ierr;
  PEP_CISS         *ctx = (PEP_CISS*)pep->data;
  SlepcContourData contour;
  PetscInt         nwork;
  PetscBool        istrivial,isellipse,flg;
  PetscObjectId    id;
  PetscObjectState state;
  Vec              v0;

  PetscFunctionBegin;
  if (pep->ncv==PETSC_DEFAULT) pep->ncv = ctx->L_max*ctx->M;
  else {
    ctx->L_max = pep->ncv/ctx->M;
    if (!ctx->L_max) {
      ctx->L_max = 1;
      pep->ncv = ctx->L_max*ctx->M;
    }
  }
  ctx->L = PetscMin(ctx->L,ctx->L_max);
  if (pep->max_it==PETSC_DEFAULT) pep->max_it = 5;
  if (pep->mpd==PETSC_DEFAULT) pep->mpd = pep->ncv;
  if (!pep->which) pep->which = PEP_ALL;
  if (pep->which!=PEP_ALL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"This solver supports only computing all eigenvalues");
  PEPCheckUnsupported(pep,PEP_FEATURE_STOPPING);
  PEPCheckIgnored(pep,PEP_FEATURE_SCALE);

  /* check region */
  ierr = RGIsTrivial(pep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"CISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  ierr = RGGetComplement(pep->rg,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"A region with complement flag set is not allowed");
  ierr = PetscObjectTypeCompare((PetscObject)pep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (!isellipse) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Currently only implemented for elliptic regions");

  /* if the region has changed, then reset contour data */
  ierr = PetscObjectGetId((PetscObject)pep->rg,&id);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)pep->rg,&state);CHKERRQ(ierr);
  if (ctx->rgid && (id != ctx->rgid || state != ctx->rgstate)) {
    ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
    ierr = PetscInfo(pep,"Resetting the contour data structure due to a change of region\n");CHKERRQ(ierr);
    ctx->rgid = id; ctx->rgstate = state;
  }

  /* create contour data structure */
  if (!ctx->contour) {
    ierr = RGCanUseConjugates(pep->rg,ctx->isreal,&ctx->useconj);CHKERRQ(ierr);
    ierr = SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)pep,&ctx->contour);CHKERRQ(ierr);
  }

  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  if (ctx->weight) { ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr); }
  ierr = PetscMalloc4(ctx->N,&ctx->weight,ctx->N,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)pep,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* allocate basis vectors */
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDuplicateResize(pep->V,ctx->L_max*ctx->M,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDuplicateResize(pep->V,ctx->L_max,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->V);CHKERRQ(ierr);

  contour = ctx->contour;
  ierr = SlepcContourRedundantMat(contour,pep->nmat,pep->A);CHKERRQ(ierr);
  if (!ctx->T) {
    ierr = MatDuplicate(contour->pA?contour->pA[0]:pep->A[0],MAT_DO_NOT_COPY_VALUES,&ctx->T);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->T);CHKERRQ(ierr);
  }
  if (!ctx->J) {
    ierr = MatDuplicate(contour->pA?contour->pA[0]:pep->A[0],MAT_DO_NOT_COPY_VALUES,&ctx->J);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->J);CHKERRQ(ierr);
  }
  if (contour->pA) {
    ierr = BVGetColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = SlepcContourScatterCreate(contour,v0);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
    ierr = BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->pV);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->pV,contour->xsub,pep->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->pV);CHKERRQ(ierr);
    ierr = BVResize(ctx->pV,ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->pV);CHKERRQ(ierr);
  }

  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  if (contour->pA) {
    ierr = BVCreate(PetscObjectComm((PetscObject)contour->xsub),&ctx->Y);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->Y,contour->xsub,pep->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->Y);CHKERRQ(ierr);
    ierr = BVResize(ctx->Y,contour->npoints*ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicateResize(pep->V,contour->npoints*ctx->L_max,&ctx->Y);CHKERRQ(ierr);
  }

  if (ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
    ierr = DSSetType(pep->ds,DSGNHEP);CHKERRQ(ierr);
  } else if (ctx->extraction == PEP_CISS_EXTRACTION_CAA) {
    ierr = DSSetType(pep->ds,DSNHEP);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(pep->ds,DSPEP);CHKERRQ(ierr);
    ierr = DSPEPSetDegree(pep->ds,pep->nmat-1);CHKERRQ(ierr);
    ierr = DSPEPSetCoefficients(pep->ds,pep->pbc);CHKERRQ(ierr);
  }
  ierr = DSAllocate(pep->ds,pep->ncv);CHKERRQ(ierr);
  nwork = 2;
  ierr = PEPSetWorkVecs(pep,nwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSolve_CISS(PEP pep)
{
  PetscErrorCode   ierr;
  PEP_CISS         *ctx = (PEP_CISS*)pep->data;
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
  ierr = DSSetFromOptions(pep->ds);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(pep->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  ierr = RGComputeQuadrature(pep->rg,RG_QUADRULE_TRAPEZOIDAL,ctx->N,ctx->omega,ctx->pp,ctx->weight);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
  ierr = BVGetRandomContext(ctx->V,&rand);CHKERRQ(ierr);
  if (contour->pA) {
    ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
  }
  ierr = PEPCISSSolveSystem(pep,ctx->T,ctx->J,(contour->pA)?ctx->pV:ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = BVTraceQuadrature(ctx->Y,ctx->V,ctx->L,ctx->L_max,ctx->weight,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj,&est_eig);CHKERRQ(ierr);
    ierr = PetscInfo1(pep,"Estimated eigenvalue count: %f\n",(double)est_eig);CHKERRQ(ierr);
    eta = PetscPowReal(10.0,-PetscLog10Real(pep->tol)/ctx->N);
    L_add = PetscMax(0,(PetscInt)PetscCeilReal((est_eig*eta)/ctx->M)-ctx->L);
    if (L_add>ctx->L_max-ctx->L) {
      ierr = PetscInfo(pep,"Number of eigenvalues inside the contour path may be too large\n");CHKERRQ(ierr);
      L_add = ctx->L_max-ctx->L;
    }
  }
  /* Updates L after estimate the number of eigenvalue */
  if (L_add>0) {
    ierr = PetscInfo2(pep,"Changing L %D -> %D by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (contour->pA) {
      ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
    }
    ierr = PEPCISSSolveSystem(pep,ctx->T,ctx->J,(contour->pA)?ctx->pV:ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    ctx->L += L_add;
  }

  ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  for (i=0;i<ctx->refine_blocksize;i++) {
    ierr = BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
    ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(PEP_CISS_SVD,pep,0,0,0);CHKERRQ(ierr);
    ierr = SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PEP_CISS_SVD,pep,0,0,0);CHKERRQ(ierr);
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    ierr = PetscInfo2(pep,"Changing L %D -> %D by SVD(H0)\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (contour->pA) {
      ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
    }
    ierr = PEPCISSSolveSystem(pep,ctx->T,ctx->J,(contour->pA)?ctx->pV:ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    ctx->L += L_add;
    if (L_add) {
      ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
      ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
    }
  }

  ierr = RGGetScale(pep->rg,&rgscale);CHKERRQ(ierr);
  ierr = RGEllipseGetParameters(pep->rg,&center,&radius,NULL);CHKERRQ(ierr);

  if (ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
    ierr = PetscMalloc1(ctx->L*ctx->M*ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  }

  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      if (ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
        ierr = BVDotQuadrature(ctx->Y,(contour->pA)?ctx->pV:ctx->V,Mu,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(PEP_CISS_SVD,pep,0,0,0);CHKERRQ(ierr);
        ierr = SlepcCISS_BH_SVD(H0,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(PEP_CISS_SVD,pep,0,0,0);CHKERRQ(ierr);
      } else {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        /* compute SVD of S */
        ierr = BVSVDAndRank(ctx->S,ctx->M,ctx->L,ctx->delta,(ctx->extraction==PEP_CISS_EXTRACTION_CAA)?BV_SVD_METHOD_QR_CAA:BV_SVD_METHOD_QR,H0,ctx->sigma,&nv);CHKERRQ(ierr);
      }
      if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        if (contour->pA) {
          ierr = BVScatter(ctx->V,ctx->pV,contour->scatterin,contour->xdup);CHKERRQ(ierr);
        }
        ierr = PEPCISSSolveSystem(pep,ctx->T,ctx->J,(contour->pA)?ctx->pV:ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
      } else break;
    }
    pep->nconv = 0;
    if (nv == 0) { pep->reason = PEP_CONVERGED_TOL; break; }
    else {
      /* Extracting eigenpairs */
      ierr = DSSetDimensions(pep->ds,nv,0,0);CHKERRQ(ierr);
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
      if (ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
        ierr = CISS_BlockHankel(Mu,0,ctx->L,ctx->M,H0);CHKERRQ(ierr);
        ierr = CISS_BlockHankel(Mu,1,ctx->L,ctx->M,H1);CHKERRQ(ierr);
        ierr = DSGetArray(pep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        for (j=0;j<nv;j++)
          for (i=0;i<nv;i++)
            temp[i+j*ld] = H1[i+j*ctx->L*ctx->M];
        ierr = DSRestoreArray(pep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        ierr = DSGetArray(pep->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
        for (j=0;j<nv;j++)
          for (i=0;i<nv;i++)
            temp[i+j*ld] = H0[i+j*ctx->L*ctx->M];
        ierr = DSRestoreArray(pep->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
      } else if (ctx->extraction == PEP_CISS_EXTRACTION_CAA) {
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = DSGetArray(pep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        for (i=0;i<nv;i++) {
          ierr = PetscArraycpy(temp+i*ld,H0+i*nv,nv);CHKERRQ(ierr);
        }
        ierr = DSRestoreArray(pep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
      } else {
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        for (i=0;i<pep->nmat;i++) {
          ierr = DSGetMat(pep->ds,DSMatExtra[i],&E);CHKERRQ(ierr);
          ierr = BVMatProject(ctx->S,pep->A[i],ctx->S,E);CHKERRQ(ierr);
          ierr = DSRestoreMat(pep->ds,DSMatExtra[i],&E);CHKERRQ(ierr);
        }
        nv = (pep->nmat-1)*nv;
      }
      ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
      ierr = DSSynchronize(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
      if (ctx->extraction == PEP_CISS_EXTRACTION_CAA || ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
        for (i=0;i<nv;i++) {
          pep->eigr[i] = (pep->eigr[i]*radius+center)*rgscale;
        }
      }
      ierr = PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr);CHKERRQ(ierr);
      ierr = DSVectors(pep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = SlepcCISS_isGhost(X,nv,ctx->sigma,ctx->spurious_threshold,fl1);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      ierr = RGCheckInside(pep->rg,nv,pep->eigr,pep->eigi,inside);CHKERRQ(ierr);
      for (i=0;i<nv;i++) {
        if (fl1[i] && inside[i]>=0) {
          rr[i] = 1.0;
          pep->nconv++;
        } else rr[i] = 0.0;
      }
      ierr = DSSort(pep->ds,pep->eigr,pep->eigi,rr,NULL,&pep->nconv);CHKERRQ(ierr);
      ierr = DSSynchronize(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
      if (ctx->extraction == PEP_CISS_EXTRACTION_CAA || ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
        for (i=0;i<nv;i++) pep->eigr[i] = (pep->eigr[i]*radius+center)*rgscale;
      }
      ierr = PetscFree3(fl1,inside,rr);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(pep->V,0,nv);CHKERRQ(ierr);
      ierr = DSVectors(pep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
      if (ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
        ierr = BVSumQuadrature(ctx->S,ctx->Y,ctx->M,ctx->L,ctx->L_max,ctx->weight,ctx->pp,contour->scatterin,contour->subcomm,contour->npoints,ctx->useconj);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,pep->V);CHKERRQ(ierr);
        ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
        ierr = BVMultInPlace(ctx->S,X,0,pep->nconv);CHKERRQ(ierr);
        ierr = BVMultInPlace(pep->V,X,0,pep->nconv);CHKERRQ(ierr);
        ierr = MatDestroy(&X);CHKERRQ(ierr);
      } else {
        ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
        ierr = BVMultInPlace(ctx->S,X,0,pep->nconv);CHKERRQ(ierr);
        ierr = MatDestroy(&X);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,pep->V);CHKERRQ(ierr);
      }
      max_error = 0.0;
      for (i=0;i<pep->nconv;i++) {
        ierr = BVGetColumn(pep->V,i,&si);CHKERRQ(ierr);
        ierr = VecNormalize(si,NULL);CHKERRQ(ierr);
        ierr = PEPComputeResidualNorm_Private(pep,pep->eigr[i],0,si,NULL,pep->work,&error);CHKERRQ(ierr);
        ierr = (*pep->converged)(pep,pep->eigr[i],0,error,&error,pep->convergedctx);CHKERRQ(ierr);
        ierr = BVRestoreColumn(pep->V,i,&si);CHKERRQ(ierr);
        max_error = PetscMax(max_error,error);
      }
      if (max_error <= pep->tol) pep->reason = PEP_CONVERGED_TOL;
      else if (pep->its > pep->max_it) pep->reason = PEP_DIVERGED_ITS;
      else {
        if (pep->nconv > ctx->L) nv = pep->nconv;
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
        }
        ierr = PEPCISSSolveSystem(pep,ctx->T,ctx->J,(contour->pA)?ctx->pV:ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
  if (ctx->extraction == PEP_CISS_EXTRACTION_HANKEL) {
    ierr = PetscFree(H1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSSetSizes_CISS(PEP pep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;
  PEP_CISS       *ctx = (PEP_CISS*)pep->data;
  PetscInt       oN,oL,oM,oLmax,onpart;

  PetscFunctionBegin;
  oN = ctx->N;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
    if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
  } else {
    if (ip<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
    if (ip%2) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
    if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
  }
  oL = ctx->L;
  if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
    ctx->L = 16;
  } else {
    if (bs<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
    ctx->L = bs;
  }
  oM = ctx->M;
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    if (ms<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = PetscMax(ms,2);
  }
  onpart = ctx->npart;
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->npart = 1;
  } else {
    if (npart<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
    ctx->npart = npart;
  }
  oLmax = ctx->L_max;
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L_max = 64;
  } else {
    if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    ctx->L_max = PetscMax(bsmax,ctx->L);
  }
  if (onpart != ctx->npart || oN != ctx->N || realmats != ctx->isreal) {
    ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
    ierr = PetscInfo(pep,"Resetting the contour data structure due to a change of parameters\n");CHKERRQ(ierr);
    pep->state = PEP_STATE_INITIAL;
  }
  ctx->isreal = realmats;
  if (oL != ctx->L || oM != ctx->M || oLmax != ctx->L_max) pep->state = PEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   PEPCISSSetSizes - Sets the values of various size parameters in the CISS solver.

   Logically Collective on pep

   Input Parameters:
+  pep   - the polynomial eigensolver context
.  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - all coefficient matrices of P(.) are real

   Options Database Keys:
+  -pep_ciss_integration_points - Sets the number of integration points
.  -pep_ciss_blocksize - Sets the block size
.  -pep_ciss_moments - Sets the moment size
.  -pep_ciss_partitions - Sets the number of partitions
.  -pep_ciss_maxblocksize - Sets the maximum block size
-  -pep_ciss_realmats - all coefficient matrices of P(.) are real

   Notes:
   The default number of partitions is 1. This means the internal KSP object is shared
   among all processes of the PEP communicator. Otherwise, the communicator is split
   into npart communicators, so that npart KSP solves proceed simultaneously.

   Level: advanced

.seealso: PEPCISSGetSizes()
@*/
PetscErrorCode PEPCISSSetSizes(PEP pep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,ip,2);
  PetscValidLogicalCollectiveInt(pep,bs,3);
  PetscValidLogicalCollectiveInt(pep,ms,4);
  PetscValidLogicalCollectiveInt(pep,npart,5);
  PetscValidLogicalCollectiveInt(pep,bsmax,6);
  PetscValidLogicalCollectiveBool(pep,realmats,7);
  ierr = PetscTryMethod(pep,"PEPCISSSetSizes_C",(PEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(pep,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSGetSizes_CISS(PEP pep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

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
   PEPCISSGetSizes - Gets the values of various size parameters in the CISS solver.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
+  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - all coefficient matrices of P(.) are real

   Level: advanced

.seealso: PEPCISSSetSizes()
@*/
PetscErrorCode PEPCISSGetSizes(PEP pep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  ierr = PetscUseMethod(pep,"PEPCISSGetSizes_C",(PEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(pep,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSSetThreshold_CISS(PEP pep,PetscReal delta,PetscReal spur)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  if (delta == PETSC_DEFAULT) {
    ctx->delta = SLEPC_DEFAULT_TOL*1e-4;
  } else {
    if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
    ctx->delta = delta;
  }
  if (spur == PETSC_DEFAULT) {
    ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  } else {
    if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
    ctx->spurious_threshold = spur;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPCISSSetThreshold - Sets the values of various threshold parameters in
   the CISS solver.

   Logically Collective on pep

   Input Parameters:
+  pep   - the polynomial eigensolver context
.  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Options Database Keys:
+  -pep_ciss_delta - Sets the delta
-  -pep_ciss_spurious_threshold - Sets the spurious threshold

   Level: advanced

.seealso: PEPCISSGetThreshold()
@*/
PetscErrorCode PEPCISSSetThreshold(PEP pep,PetscReal delta,PetscReal spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,delta,2);
  PetscValidLogicalCollectiveReal(pep,spur,3);
  ierr = PetscTryMethod(pep,"PEPCISSSetThreshold_C",(PEP,PetscReal,PetscReal),(pep,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSGetThreshold_CISS(PEP pep,PetscReal *delta,PetscReal *spur)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  if (delta) *delta = ctx->delta;
  if (spur)  *spur = ctx->spurious_threshold;
  PetscFunctionReturn(0);
}

/*@
   PEPCISSGetThreshold - Gets the values of various threshold parameters in
   the CISS solver.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
+  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Level: advanced

.seealso: PEPCISSSetThreshold()
@*/
PetscErrorCode PEPCISSGetThreshold(PEP pep,PetscReal *delta,PetscReal *spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  ierr = PetscUseMethod(pep,"PEPCISSGetThreshold_C",(PEP,PetscReal*,PetscReal*),(pep,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSSetRefinement_CISS(PEP pep,PetscInt inner,PetscInt blsize)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  if (inner == PETSC_DEFAULT) {
    ctx->refine_inner = 0;
  } else {
    if (inner<0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The refine inner argument must be >= 0");
    ctx->refine_inner = inner;
  }
  if (blsize == PETSC_DEFAULT) {
    ctx->refine_blocksize = 0;
  } else {
    if (blsize<0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The refine blocksize argument must be >= 0");
    ctx->refine_blocksize = blsize;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPCISSSetRefinement - Sets the values of various refinement parameters
   in the CISS solver.

   Logically Collective on pep

   Input Parameters:
+  pep    - the polynomial eigensolver context
.  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Options Database Keys:
+  -pep_ciss_refine_inner - Sets number of inner iterations
-  -pep_ciss_refine_blocksize - Sets number of blocksize iterations

   Level: advanced

.seealso: PEPCISSGetRefinement()
@*/
PetscErrorCode PEPCISSSetRefinement(PEP pep,PetscInt inner,PetscInt blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,inner,2);
  PetscValidLogicalCollectiveInt(pep,blsize,3);
  ierr = PetscTryMethod(pep,"PEPCISSSetRefinement_C",(PEP,PetscInt,PetscInt),(pep,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSGetRefinement_CISS(PEP pep,PetscInt *inner,PetscInt *blsize)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  if (inner)  *inner = ctx->refine_inner;
  if (blsize) *blsize = ctx->refine_blocksize;
  PetscFunctionReturn(0);
}

/*@
   PEPCISSGetRefinement - Gets the values of various refinement parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
+  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Level: advanced

.seealso: PEPCISSSetRefinement()
@*/
PetscErrorCode PEPCISSGetRefinement(PEP pep, PetscInt *inner, PetscInt *blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  ierr = PetscUseMethod(pep,"PEPCISSGetRefinement_C",(PEP,PetscInt*,PetscInt*),(pep,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSSetExtraction_CISS(PEP pep,PEPCISSExtraction extraction)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  if (ctx->extraction != extraction) {
    ctx->extraction = extraction;
    pep->state      = PEP_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPCISSSetExtraction - Sets the extraction technique used in the CISS solver.

   Logically Collective on pep

   Input Parameters:
+  pep        - the polynomial eigensolver context
-  extraction - the extraction technique

   Options Database Key:
.  -pep_ciss_extraction - Sets the extraction technique (either 'ritz', 'hankel' or 'caa')

   Notes:
   By default, the Rayleigh-Ritz extraction is used (PEP_CISS_EXTRACTION_RITZ).

   If the 'hankel' or the 'caa' option is specified (PEP_CISS_EXTRACTION_HANKEL or
   PEP_CISS_EXTRACTION_CAA), then the Block Hankel method, or the Communication-avoiding
   Arnoldi method, respectively, is used for extracting eigenpairs.

   Level: advanced

.seealso: PEPCISSGetExtraction(), PEPCISSExtraction
@*/
PetscErrorCode PEPCISSSetExtraction(PEP pep,PEPCISSExtraction extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pep,extraction,2);
  ierr = PetscTryMethod(pep,"PEPCISSSetExtraction_C",(PEP,PEPCISSExtraction),(pep,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSGetExtraction_CISS(PEP pep,PEPCISSExtraction *extraction)
{
  PEP_CISS *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  *extraction = ctx->extraction;
  PetscFunctionReturn(0);
}

/*@
   PEPCISSGetExtraction - Gets the extraction technique used in the CISS solver.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
.  extraction - extraction technique

   Level: advanced

.seealso: PEPCISSSetExtraction() PEPCISSExtraction
@*/
PetscErrorCode PEPCISSGetExtraction(PEP pep,PEPCISSExtraction *extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(extraction,2);
  ierr = PetscUseMethod(pep,"PEPCISSGetExtraction_C",(PEP,PEPCISSExtraction*),(pep,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPCISSGetKSPs_CISS(PEP pep,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode   ierr;
  PEP_CISS         *ctx = (PEP_CISS*)pep->data;
  SlepcContourData contour;
  PetscInt         i;
  PC               pc;

  PetscFunctionBegin;
  if (!ctx->contour) {  /* initialize contour data structure first */
    ierr = RGCanUseConjugates(pep->rg,ctx->isreal,&ctx->useconj);CHKERRQ(ierr);
    ierr = SlepcContourDataCreate(ctx->useconj?ctx->N/2:ctx->N,ctx->npart,(PetscObject)pep,&ctx->contour);CHKERRQ(ierr);
  }
  contour = ctx->contour;
  if (!contour->ksp) {
    ierr = PetscMalloc1(contour->npoints,&contour->ksp);CHKERRQ(ierr);
    for (i=0;i<contour->npoints;i++) {
      ierr = KSPCreate(PetscSubcommChild(contour->subcomm),&contour->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)contour->ksp[i],(PetscObject)pep,1);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(contour->ksp[i],((PetscObject)pep)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(contour->ksp[i],"pep_ciss_");CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)contour->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptions((PetscObject)contour->ksp[i],((PetscObject)pep)->options);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(contour->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(contour->ksp[i],SlepcDefaultTol(pep->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
   PEPCISSGetKSPs - Retrieve the array of linear solver objects associated with
   the CISS solver.

   Not Collective

   Input Parameter:
.  pep - polynomial eigenvalue solver

   Output Parameters:
+  nsolve - number of solver objects
-  ksp - array of linear solver object

   Notes:
   The number of KSP solvers is equal to the number of integration points divided by
   the number of partitions. This value is halved in the case of real matrices with
   a region centered at the real axis.

   Level: advanced

.seealso: PEPCISSSetSizes()
@*/
PetscErrorCode PEPCISSGetKSPs(PEP pep,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  ierr = PetscUseMethod(pep,"PEPCISSGetKSPs_C",(PEP,PetscInt*,KSP**),(pep,nsolve,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPReset_CISS(PEP pep)
{
  PetscErrorCode ierr;
  PEP_CISS       *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  ierr = SlepcContourDataReset(ctx->contour);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->T);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetFromOptions_CISS(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode    ierr;
  PEP_CISS          *ctx = (PEP_CISS*)pep->data;
  PetscReal         r1,r2;
  PetscInt          i,i1,i2,i3,i4,i5,i6,i7;
  PetscBool         b1,flg,flg2,flg3,flg4,flg5,flg6;
  PEPCISSExtraction extraction;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP CISS Options");CHKERRQ(ierr);

    ierr = PEPCISSGetSizes(pep,&i1,&i2,&i3,&i4,&i5,&b1);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_integration_points","Number of integration points","PEPCISSSetSizes",i1,&i1,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_blocksize","Block size","PEPCISSSetSizes",i2,&i2,&flg2);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_moments","Moment size","PEPCISSSetSizes",i3,&i3,&flg3);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_partitions","Number of partitions","PEPCISSSetSizes",i4,&i4,&flg4);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_maxblocksize","Maximum block size","PEPCISSSetSizes",i5,&i5,&flg5);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-pep_ciss_realmats","True if all coefficient matrices of P(.) are real","PEPCISSSetSizes",b1,&b1,&flg6);CHKERRQ(ierr);
    if (flg || flg2 || flg3 || flg4 || flg5 || flg6) { ierr = PEPCISSSetSizes(pep,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr); }

    ierr = PEPCISSGetThreshold(pep,&r1,&r2);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pep_ciss_delta","Threshold for numerical rank","PEPCISSSetThreshold",r1,&r1,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pep_ciss_spurious_threshold","Threshold for the spurious eigenpairs","PEPCISSSetThreshold",r2,&r2,&flg2);CHKERRQ(ierr);
    if (flg || flg2) { ierr = PEPCISSSetThreshold(pep,r1,r2);CHKERRQ(ierr); }

    ierr = PEPCISSGetRefinement(pep,&i6,&i7);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_refine_inner","Number of inner iterative refinement iterations","PEPCISSSetRefinement",i6,&i6,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ciss_refine_blocksize","Number of blocksize iterative refinement iterations","PEPCISSSetRefinement",i7,&i7,&flg2);CHKERRQ(ierr);
    if (flg || flg2) { ierr = PEPCISSSetRefinement(pep,i6,i7);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-pep_ciss_extraction","Extraction technique","PEPCISSSetExtraction",PEPCISSExtractions,(PetscEnum)ctx->extraction,(PetscEnum*)&extraction,&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPCISSSetExtraction(pep,extraction);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!pep->rg) { ierr = PEPGetRG(pep,&pep->rg);CHKERRQ(ierr); }
  ierr = RGSetFromOptions(pep->rg);CHKERRQ(ierr); /* this is necessary here to set useconj */
  if (!ctx->contour || !ctx->contour->ksp) { ierr = PEPCISSGetKSPs(pep,NULL,NULL);CHKERRQ(ierr); }
  for (i=0;i<ctx->contour->npoints;i++) {
    ierr = KSPSetFromOptions(ctx->contour->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscSubcommSetFromOptions(ctx->contour->subcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPDestroy_CISS(PEP pep)
{
  PetscErrorCode ierr;
  PEP_CISS       *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  ierr = SlepcContourDataDestroy(&ctx->contour);CHKERRQ(ierr);
  ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr);
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetKSPs_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPView_CISS(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_CISS       *ctx = (PEP_CISS*)pep->data;
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
    ierr = PetscViewerASCIIPrintf(viewer,"  extraction: %s\n",PEPCISSExtractions[ctx->extraction]);CHKERRQ(ierr);
    if (!ctx->contour || !ctx->contour->ksp) { ierr = PEPCISSGetKSPs(pep,NULL,NULL);CHKERRQ(ierr); }
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

SLEPC_EXTERN PetscErrorCode PEPCreate_CISS(PEP pep)
{
  PetscErrorCode ierr;
  PEP_CISS       *ctx = (PEP_CISS*)pep->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  pep->data = ctx;
  /* set default values of parameters */
  ctx->N                  = 32;
  ctx->L                  = 16;
  ctx->M                  = ctx->N/4;
  ctx->delta              = SLEPC_DEFAULT_TOL*1e-4;
  ctx->L_max              = 64;
  ctx->spurious_threshold = PetscSqrtReal(SLEPC_DEFAULT_TOL);
  ctx->isreal             = PETSC_FALSE;
  ctx->npart              = 1;

  pep->ops->solve          = PEPSolve_CISS;
  pep->ops->setup          = PEPSetUp_CISS;
  pep->ops->setfromoptions = PEPSetFromOptions_CISS;
  pep->ops->reset          = PEPReset_CISS;
  pep->ops->destroy        = PEPDestroy_CISS;
  pep->ops->view           = PEPView_CISS;

  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetSizes_C",PEPCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetSizes_C",PEPCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetThreshold_C",PEPCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetThreshold_C",PEPCISSGetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetRefinement_C",PEPCISSSetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetRefinement_C",PEPCISSGetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSSetExtraction_C",PEPCISSSetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetExtraction_C",PEPCISSGetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPCISSGetKSPs_C",PEPCISSGetKSPs_CISS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

