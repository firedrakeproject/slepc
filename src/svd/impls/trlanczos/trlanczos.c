/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "trlanczos"

   Method: Thick-restart Lanczos

   Algorithm:

       Golub-Kahan-Lanczos bidiagonalization with thick-restart.

   References:

       [1] G.H. Golub and W. Kahan, "Calculating the singular values
           and pseudo-inverse of a matrix", SIAM J. Numer. Anal. Ser.
           B 2:205-224, 1965.

       [2] V. Hernandez, J.E. Roman, and A. Tomas, "A robust and
           efficient parallel SVD solver based on restarted Lanczos
           bidiagonalization", Elec. Trans. Numer. Anal. 31:68-85,
           2008.
*/

#include <slepc/private/svdimpl.h>          /*I "slepcsvd.h" I*/

static PetscBool  cited = PETSC_FALSE,citedg = PETSC_FALSE;
static const char citation[] =
  "@Article{slepc-svd,\n"
  "   author = \"V. Hern{\\'a}ndez and J. E. Rom{\\'a}n and A. Tom{\\'a}s\",\n"
  "   title = \"A robust and efficient parallel {SVD} solver based on restarted {Lanczos} bidiagonalization\",\n"
  "   journal = \"Electron. Trans. Numer. Anal.\",\n"
  "   volume = \"31\",\n"
  "   pages = \"68--85\",\n"
  "   year = \"2008\"\n"
  "}\n";
static const char citationg[] =
  "@Article{slepc-gsvd,\n"
  "   author = \"F. Alvarruiz and C. Campos and J. E. Roman\",\n"
  "   title = \"Thick-restarted {Lanczos} bidiagonalization methods for the {GSVD}\",\n"
  "   note = \"In preparation\",\n"
  "   year = \"2021\"\n"
  "}\n";

typedef struct {
  /* user parameters */
  PetscBool           oneside;   /* one-sided variant */
  PetscReal           keep;      /* restart parameter */
  PetscBool           lock;      /* locking/non-locking variant */
  KSP                 ksp;       /* solver for least-squares problem in GSVD */
  SVDTRLanczosGBidiag bidiag;    /* bidiagonalization variant for GSVD */
  PetscBool           explicitmatrix;
  /* auxiliary variables */
  Mat                 Z;         /* aux matrix for GSVD, Z=[A;B] */
} SVD_TRLANCZOS;

/* Context for shell matrix [A; B] */
typedef struct {
  Mat      A,B,AT,BT;
  Vec      y1,y2,y;
  PetscInt m;
} MatZData;

static PetscErrorCode MatZCreateContext(SVD svd,MatZData **zdata)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(zdata));
  (*zdata)->A = svd->A;
  (*zdata)->B = svd->B;
  (*zdata)->AT = svd->AT;
  (*zdata)->BT = svd->BT;
  PetscCall(MatCreateVecsEmpty(svd->A,NULL,&(*zdata)->y1));
  PetscCall(MatCreateVecsEmpty(svd->B,NULL,&(*zdata)->y2));
  PetscCall(VecGetLocalSize((*zdata)->y1,&(*zdata)->m));
  PetscCall(BVCreateVec(svd->U,&(*zdata)->y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Z(Mat Z)
{
  MatZData       *zdata;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Z,&zdata));
  PetscCall(VecDestroy(&zdata->y1));
  PetscCall(VecDestroy(&zdata->y2));
  PetscCall(VecDestroy(&zdata->y));
  PetscCall(PetscFree(zdata));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Z(Mat Z,Vec x,Vec y)
{
  MatZData       *zdata;
  PetscScalar    *y_elems;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Z,&zdata));
  PetscCall(VecGetArray(y,&y_elems));
  PetscCall(VecPlaceArray(zdata->y1,y_elems));
  PetscCall(VecPlaceArray(zdata->y2,y_elems+zdata->m));

  PetscCall(MatMult(zdata->A,x,zdata->y1));
  PetscCall(MatMult(zdata->B,x,zdata->y2));

  PetscCall(VecResetArray(zdata->y1));
  PetscCall(VecResetArray(zdata->y2));
  PetscCall(VecRestoreArray(y,&y_elems));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Z(Mat Z,Vec y,Vec x)
{
  MatZData          *zdata;
  const PetscScalar *y_elems;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Z,&zdata));
  PetscCall(VecGetArrayRead(y,&y_elems));
  PetscCall(VecPlaceArray(zdata->y1,y_elems));
  PetscCall(VecPlaceArray(zdata->y2,y_elems+zdata->m));

  PetscCall(MatMult(zdata->AT,zdata->y1,x));
  PetscCall(MatMultAdd(zdata->BT,zdata->y2,x,x));

  PetscCall(VecResetArray(zdata->y1));
  PetscCall(VecResetArray(zdata->y2));
  PetscCall(VecRestoreArrayRead(y,&y_elems));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateVecs_Z(Mat Z,Vec *right,Vec *left)
{
  MatZData       *zdata;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Z,&zdata));
  if (right) PetscCall(MatCreateVecs(zdata->A,right,NULL));
  if (left) PetscCall(VecDuplicate(zdata->y,left));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetUp_TRLanczos(SVD svd)
{
  PetscInt       N,m,n,p;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  DSType         dstype;
  MatZData       *zdata;
  Mat            mats[2],normal;
  MatType        Atype;
  PetscBool      sametype;

  PetscFunctionBegin;
  PetscCall(MatGetSize(svd->A,NULL,&N));
  PetscCall(SVDSetDimensions_Default(svd));
  PetscCheck(svd->ncv<=svd->nsv+svd->mpd,PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nsv+mpd");
  PetscCheck(lanczos->lock || svd->mpd>=svd->ncv,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  if (!lanczos->keep) lanczos->keep = 0.5;
  svd->leftbasis = PETSC_TRUE;
  PetscCall(SVDAllocateSolution(svd,1));
  dstype = DSSVD;
  if (svd->isgeneralized) {
    if (lanczos->bidiag==SVD_TRLANCZOS_GBIDIAG_UPPER || lanczos->bidiag==SVD_TRLANCZOS_GBIDIAG_LOWER) dstype = DSGSVD;
    PetscCall(SVDSetWorkVecs(svd,1,1));

    if (svd->conv==SVD_CONV_ABS) {  /* Residual norms are multiplied by matrix norms */
      if (!svd->nrma) PetscCall(MatNorm(svd->OP,NORM_INFINITY,&svd->nrma));
      if (!svd->nrmb) PetscCall(MatNorm(svd->OPb,NORM_INFINITY,&svd->nrmb));
    }

    /* Create the matrix Z=[A;B] */
    PetscCall(MatDestroy(&lanczos->Z));
    PetscCall(MatGetLocalSize(svd->A,&m,&n));
    PetscCall(MatGetLocalSize(svd->B,&p,NULL));
    if (lanczos->explicitmatrix) {
      mats[0] = svd->A;
      mats[1] = svd->B;
      PetscCall(MatCreateNest(PetscObjectComm((PetscObject)svd),2,NULL,1,NULL,mats,&lanczos->Z));
      PetscCall(MatGetType(svd->A,&Atype));
      PetscCall(PetscObjectTypeCompare((PetscObject)svd->B,Atype,&sametype));
      if (!sametype) Atype = MATAIJ;
      PetscCall(MatConvert(lanczos->Z,Atype,MAT_INPLACE_MATRIX,&lanczos->Z));
    } else {
      PetscCall(MatZCreateContext(svd,&zdata));
      PetscCall(MatCreateShell(PetscObjectComm((PetscObject)svd),m+p,n,PETSC_DECIDE,PETSC_DECIDE,zdata,&lanczos->Z));
      PetscCall(MatShellSetOperation(lanczos->Z,MATOP_MULT,(void(*)(void))MatMult_Z));
#if defined(PETSC_USE_COMPLEX)
      PetscCall(MatShellSetOperation(lanczos->Z,MATOP_MULT_HERMITIAN_TRANSPOSE,(void(*)(void))MatMultTranspose_Z));
#else
      PetscCall(MatShellSetOperation(lanczos->Z,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Z));
#endif
      PetscCall(MatShellSetOperation(lanczos->Z,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_Z));
      PetscCall(MatShellSetOperation(lanczos->Z,MATOP_DESTROY,(void(*)(void))MatDestroy_Z));
    }
    PetscCall(PetscLogObjectParent((PetscObject)svd,(PetscObject)lanczos->Z));

    /* create normal equations matrix, to build the preconditioner in LSQR */
    PetscCall(MatCreateNormalHermitian(lanczos->Z,&normal));

    if (!lanczos->ksp) PetscCall(SVDTRLanczosGetKSP(svd,&lanczos->ksp));
    PetscCall(SVD_KSPSetOperators(lanczos->ksp,lanczos->Z,normal));
    PetscCall(KSPSetUp(lanczos->ksp));
    PetscCall(MatDestroy(&normal));

    if (lanczos->oneside) PetscCall(PetscInfo(svd,"Warning: one-side option is ignored in GSVD\n"));
  }
  PetscCall(DSSetType(svd->ds,dstype));
  PetscCall(DSSetCompact(svd->ds,PETSC_TRUE));
  PetscCall(DSSetExtraRow(svd->ds,PETSC_TRUE));
  PetscCall(DSAllocate(svd->ds,svd->ncv+1));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDOneSideTRLanczosMGS(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscReal      a,b;
  PetscInt       i,k=nconv+l;
  Vec            ui,ui1,vi;

  PetscFunctionBegin;
  PetscCall(BVGetColumn(V,k,&vi));
  PetscCall(BVGetColumn(U,k,&ui));
  PetscCall(MatMult(svd->A,vi,ui));
  PetscCall(BVRestoreColumn(V,k,&vi));
  PetscCall(BVRestoreColumn(U,k,&ui));
  if (l>0) {
    PetscCall(BVSetActiveColumns(U,nconv,n));
    for (i=0;i<l;i++) work[i]=beta[i+nconv];
    PetscCall(BVMultColumn(U,-1.0,1.0,k,work));
  }
  PetscCall(BVNormColumn(U,k,NORM_2,&a));
  PetscCall(BVScaleColumn(U,k,1.0/a));
  alpha[k] = a;

  for (i=k+1;i<n;i++) {
    PetscCall(BVGetColumn(V,i,&vi));
    PetscCall(BVGetColumn(U,i-1,&ui1));
    PetscCall(MatMult(svd->AT,ui1,vi));
    PetscCall(BVRestoreColumn(V,i,&vi));
    PetscCall(BVRestoreColumn(U,i-1,&ui1));
    PetscCall(BVOrthonormalizeColumn(V,i,PETSC_FALSE,&b,NULL));
    beta[i-1] = b;

    PetscCall(BVGetColumn(V,i,&vi));
    PetscCall(BVGetColumn(U,i,&ui));
    PetscCall(MatMult(svd->A,vi,ui));
    PetscCall(BVRestoreColumn(V,i,&vi));
    PetscCall(BVGetColumn(U,i-1,&ui1));
    PetscCall(VecAXPY(ui,-b,ui1));
    PetscCall(BVRestoreColumn(U,i-1,&ui1));
    PetscCall(BVRestoreColumn(U,i,&ui));
    PetscCall(BVNormColumn(U,i,NORM_2,&a));
    PetscCall(BVScaleColumn(U,i,1.0/a));
    alpha[i] = a;
  }

  PetscCall(BVGetColumn(V,n,&vi));
  PetscCall(BVGetColumn(U,n-1,&ui1));
  PetscCall(MatMult(svd->AT,ui1,vi));
  PetscCall(BVRestoreColumn(V,n,&vi));
  PetscCall(BVRestoreColumn(U,n-1,&ui1));
  PetscCall(BVOrthogonalizeColumn(V,n,NULL,&b,NULL));
  beta[n-1] = b;
  PetscFunctionReturn(0);
}

/*
  Custom CGS orthogonalization, preprocess after first orthogonalization
*/
static PetscErrorCode SVDOrthogonalizeCGS(BV V,PetscInt i,PetscScalar* h,PetscReal a,BVOrthogRefineType refine,PetscReal eta,PetscReal *norm)
{
  PetscReal      sum,onorm;
  PetscScalar    dot;
  PetscInt       j;

  PetscFunctionBegin;
  switch (refine) {
  case BV_ORTHOG_REFINE_NEVER:
    PetscCall(BVNormColumn(V,i,NORM_2,norm));
    break;
  case BV_ORTHOG_REFINE_ALWAYS:
    PetscCall(BVSetActiveColumns(V,0,i));
    PetscCall(BVDotColumn(V,i,h));
    PetscCall(BVMultColumn(V,-1.0,1.0,i,h));
    PetscCall(BVNormColumn(V,i,NORM_2,norm));
    break;
  case BV_ORTHOG_REFINE_IFNEEDED:
    dot = h[i];
    onorm = PetscSqrtReal(PetscRealPart(dot)) / a;
    sum = 0.0;
    for (j=0;j<i;j++) {
      sum += PetscRealPart(h[j] * PetscConj(h[j]));
    }
    *norm = PetscRealPart(dot)/(a*a) - sum;
    if (*norm>0.0) *norm = PetscSqrtReal(*norm);
    else PetscCall(BVNormColumn(V,i,NORM_2,norm));
    if (*norm < eta*onorm) {
      PetscCall(BVSetActiveColumns(V,0,i));
      PetscCall(BVDotColumn(V,i,h));
      PetscCall(BVMultColumn(V,-1.0,1.0,i,h));
      PetscCall(BVNormColumn(V,i,NORM_2,norm));
    }
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDOneSideTRLanczosCGS(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscReal          a,b,eta;
  PetscInt           i,j,k=nconv+l;
  Vec                ui,ui1,vi;
  BVOrthogRefineType refine;

  PetscFunctionBegin;
  PetscCall(BVGetColumn(V,k,&vi));
  PetscCall(BVGetColumn(U,k,&ui));
  PetscCall(MatMult(svd->A,vi,ui));
  PetscCall(BVRestoreColumn(V,k,&vi));
  PetscCall(BVRestoreColumn(U,k,&ui));
  if (l>0) {
    PetscCall(BVSetActiveColumns(U,nconv,n));
    for (i=0;i<l;i++) work[i]=beta[i+nconv];
    PetscCall(BVMultColumn(U,-1.0,1.0,k,work));
  }
  PetscCall(BVGetOrthogonalization(V,NULL,&refine,&eta,NULL));

  for (i=k+1;i<n;i++) {
    PetscCall(BVGetColumn(V,i,&vi));
    PetscCall(BVGetColumn(U,i-1,&ui1));
    PetscCall(MatMult(svd->AT,ui1,vi));
    PetscCall(BVRestoreColumn(V,i,&vi));
    PetscCall(BVRestoreColumn(U,i-1,&ui1));
    PetscCall(BVNormColumnBegin(U,i-1,NORM_2,&a));
    if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
      PetscCall(BVSetActiveColumns(V,0,i+1));
      PetscCall(BVGetColumn(V,i,&vi));
      PetscCall(BVDotVecBegin(V,vi,work));
    } else {
      PetscCall(BVSetActiveColumns(V,0,i));
      PetscCall(BVDotColumnBegin(V,i,work));
    }
    PetscCall(BVNormColumnEnd(U,i-1,NORM_2,&a));
    if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
      PetscCall(BVDotVecEnd(V,vi,work));
      PetscCall(BVRestoreColumn(V,i,&vi));
      PetscCall(BVSetActiveColumns(V,0,i));
    } else PetscCall(BVDotColumnEnd(V,i,work));

    PetscCall(BVScaleColumn(U,i-1,1.0/a));
    for (j=0;j<i;j++) work[j] = work[j] / a;
    PetscCall(BVMultColumn(V,-1.0,1.0/a,i,work));
    PetscCall(SVDOrthogonalizeCGS(V,i,work,a,refine,eta,&b));
    PetscCall(BVScaleColumn(V,i,1.0/b));
    PetscCheck(PetscAbsReal(b)>10*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)svd),PETSC_ERR_PLIB,"Recurrence generated a zero vector; use a two-sided variant");

    PetscCall(BVGetColumn(V,i,&vi));
    PetscCall(BVGetColumn(U,i,&ui));
    PetscCall(BVGetColumn(U,i-1,&ui1));
    PetscCall(MatMult(svd->A,vi,ui));
    PetscCall(VecAXPY(ui,-b,ui1));
    PetscCall(BVRestoreColumn(V,i,&vi));
    PetscCall(BVRestoreColumn(U,i,&ui));
    PetscCall(BVRestoreColumn(U,i-1,&ui1));

    alpha[i-1] = a;
    beta[i-1] = b;
  }

  PetscCall(BVGetColumn(V,n,&vi));
  PetscCall(BVGetColumn(U,n-1,&ui1));
  PetscCall(MatMult(svd->AT,ui1,vi));
  PetscCall(BVRestoreColumn(V,n,&vi));
  PetscCall(BVRestoreColumn(U,n-1,&ui1));

  PetscCall(BVNormColumnBegin(svd->U,n-1,NORM_2,&a));
  if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
    PetscCall(BVSetActiveColumns(V,0,n+1));
    PetscCall(BVGetColumn(V,n,&vi));
    PetscCall(BVDotVecBegin(V,vi,work));
  } else {
    PetscCall(BVSetActiveColumns(V,0,n));
    PetscCall(BVDotColumnBegin(V,n,work));
  }
  PetscCall(BVNormColumnEnd(svd->U,n-1,NORM_2,&a));
  if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
    PetscCall(BVDotVecEnd(V,vi,work));
    PetscCall(BVRestoreColumn(V,n,&vi));
  } else PetscCall(BVDotColumnEnd(V,n,work));

  PetscCall(BVScaleColumn(U,n-1,1.0/a));
  for (j=0;j<n;j++) work[j] = work[j] / a;
  PetscCall(BVMultColumn(V,-1.0,1.0/a,n,work));
  PetscCall(SVDOrthogonalizeCGS(V,n,work,a,refine,eta,&b));
  PetscCall(BVSetActiveColumns(V,nconv,n));
  alpha[n-1] = a;
  beta[n-1] = b;
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_TRLanczos(SVD svd)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta;
  PetscScalar    *swork=NULL,*w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,V;
  PetscBool      breakdown=PETSC_FALSE;
  BVOrthogType   orthog;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation,&cited));
  /* allocate working space */
  PetscCall(DSGetLeadingDimension(svd->ds,&ld));
  PetscCall(BVGetOrthogonalization(svd->V,&orthog,NULL,NULL,NULL));
  PetscCall(PetscMalloc1(ld,&w));
  if (lanczos->oneside) PetscCall(PetscMalloc1(svd->ncv+1,&swork));

  /* normalize start vector */
  if (!svd->nini) {
    PetscCall(BVSetRandomColumn(svd->V,0));
    PetscCall(BVOrthonormalizeColumn(svd->V,0,PETSC_TRUE,NULL,NULL));
  }

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    beta = alpha + ld;
    if (lanczos->oneside) {
      if (orthog == BV_ORTHOG_MGS) PetscCall(SVDOneSideTRLanczosMGS(svd,alpha,beta,svd->V,svd->U,svd->nconv,l,nv,swork));
      else PetscCall(SVDOneSideTRLanczosCGS(svd,alpha,beta,svd->V,svd->U,svd->nconv,l,nv,swork));
    } else PetscCall(SVDTwoSideLanczos(svd,alpha,beta,svd->V,svd->U,svd->nconv+l,&nv,&breakdown));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));
    PetscCall(BVScaleColumn(svd->V,nv,1.0/beta[nv-1]));
    PetscCall(BVSetActiveColumns(svd->V,svd->nconv,nv));
    PetscCall(BVSetActiveColumns(svd->U,svd->nconv,nv));

    /* solve projected problem */
    PetscCall(DSSetDimensions(svd->ds,nv,svd->nconv,svd->nconv+l));
    PetscCall(DSSVDSetDimensions(svd->ds,nv));
    PetscCall(DSSetState(svd->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    PetscCall(DSSolve(svd->ds,w,NULL));
    PetscCall(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
    PetscCall(DSUpdateExtraRow(svd->ds));
    PetscCall(DSSynchronize(svd->ds,w,NULL));
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    PetscCall(SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,1.0,&k));
    PetscCall((*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx));

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) PetscCall(PetscInfo(svd,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (PetscUnlikely(breakdown || k==nv)) {
        /* Start a new bidiagonalization */
        PetscCall(PetscInfo(svd,"Breakdown in bidiagonalization (it=%" PetscInt_FMT ")\n",svd->its));
        if (k<svd->nsv) {
          PetscCall(BVSetRandomColumn(svd->V,k));
          PetscCall(BVOrthonormalizeColumn(svd->V,k,PETSC_FALSE,NULL,&breakdown));
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            PetscCall(PetscInfo(svd,"Unable to generate more start vectors\n"));
          }
        }
      } else PetscCall(DSTruncate(svd->ds,k+l,PETSC_FALSE));
    }

    /* compute converged singular vectors and restart vectors */
    PetscCall(DSGetMat(svd->ds,DS_MAT_V,&V));
    PetscCall(BVMultInPlace(svd->V,V,svd->nconv,k+l));
    PetscCall(MatDestroy(&V));
    PetscCall(DSGetMat(svd->ds,DS_MAT_U,&U));
    PetscCall(BVMultInPlace(svd->U,U,svd->nconv,k+l));
    PetscCall(MatDestroy(&U));

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) PetscCall(BVCopyColumn(svd->V,nv,k+l));

    svd->nconv = k;
    PetscCall(SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv));
  }

  /* orthonormalize U columns in one side method */
  if (lanczos->oneside) {
    for (i=0;i<svd->nconv;i++) PetscCall(BVOrthonormalizeColumn(svd->U,i,PETSC_FALSE,NULL,NULL));
  }

  /* free working space */
  PetscCall(PetscFree(w));
  if (swork) PetscCall(PetscFree(swork));
  PetscCall(DSTruncate(svd->ds,svd->nconv,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTwoSideLanczosGSingle(SVD svd,PetscReal *alpha,PetscReal *beta,Mat Z,BV V,BV U,KSP ksp,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscInt          i,j,m;
  const PetscScalar *carr;
  PetscScalar       *arr;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0],v1;
  PetscBool         lindep=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatCreateVecsEmpty(svd->A,NULL,&v1));
  PetscCall(BVGetColumn(V,k,&v));
  PetscCall(BVGetColumn(U,k,&u));

  /* Form ut=[u;0] */
  PetscCall(VecZeroEntries(ut));
  PetscCall(VecGetLocalSize(u,&m));
  PetscCall(VecGetArrayRead(u,&carr));
  PetscCall(VecGetArray(ut,&arr));
  for (j=0; j<m; j++) arr[j] = carr[j];
  PetscCall(VecRestoreArrayRead(u,&carr));
  PetscCall(VecRestoreArray(ut,&arr));

  /* Solve least squares problem */
  PetscCall(KSPSolve(ksp,ut,x));

  PetscCall(MatMult(Z,x,v));

  PetscCall(BVRestoreColumn(U,k,&u));
  PetscCall(BVRestoreColumn(V,k,&v));
  PetscCall(BVOrthonormalizeColumn(V,k,PETSC_FALSE,alpha+k,&lindep));
  if (PetscUnlikely(lindep)) {
    *n = k;
    if (breakdown) *breakdown = lindep;
    PetscFunctionReturn(0);
  }

  for (i=k+1; i<*n; i++) {

    /* Compute vector i of BV U */
    PetscCall(BVGetColumn(V,i-1,&v));
    PetscCall(VecGetArray(v,&arr));
    PetscCall(VecPlaceArray(v1,arr));
    PetscCall(VecRestoreArray(v,&arr));
    PetscCall(BVRestoreColumn(V,i-1,&v));
    PetscCall(BVInsertVec(U,i,v1));
    PetscCall(VecResetArray(v1));
    PetscCall(BVOrthonormalizeColumn(U,i,PETSC_FALSE,beta+i-1,&lindep));
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }

    /* Compute vector i of BV V */

    PetscCall(BVGetColumn(V,i,&v));
    PetscCall(BVGetColumn(U,i,&u));

    /* Form ut=[u;0] */
    PetscCall(VecGetArrayRead(u,&carr));
    PetscCall(VecGetArray(ut,&arr));
    for (j=0; j<m; j++) arr[j] = carr[j];
    PetscCall(VecRestoreArrayRead(u,&carr));
    PetscCall(VecRestoreArray(ut,&arr));

    /* Solve least squares problem */
    PetscCall(KSPSolve(ksp,ut,x));

    PetscCall(MatMult(Z,x,v));

    PetscCall(BVRestoreColumn(U,i,&u));
    PetscCall(BVRestoreColumn(V,i,&v));
    PetscCall(BVOrthonormalizeColumn(V,i,PETSC_FALSE,alpha+i,&lindep));
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }
  }

  /* Compute vector n of BV U */
  if (!lindep) {
    PetscCall(BVGetColumn(V,*n-1,&v));
    PetscCall(VecGetArray(v,&arr));
    PetscCall(VecPlaceArray(v1,arr));
    PetscCall(VecRestoreArray(v,&arr));
    PetscCall(BVRestoreColumn(V,*n-1,&v));
    PetscCall(BVInsertVec(U,*n,v1));
    PetscCall(VecResetArray(v1));
    PetscCall(BVOrthonormalizeColumn(U,*n,PETSC_FALSE,beta+*n-1,&lindep));
  }
  if (breakdown) *breakdown = lindep;
  PetscCall(VecDestroy(&v1));
  PetscFunctionReturn(0);
}

/* solve generalized problem with single bidiagonalization of Q_A */
PetscErrorCode SVDSolve_TRLanczosGSingle(SVD svd,BV U1,BV V)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta,normr;
  PetscScalar    *w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,VV;
  PetscBool      breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DSGetLeadingDimension(svd->ds,&ld));
  PetscCall(PetscMalloc1(ld,&w));
  normr = (svd->conv==SVD_CONV_ABS)? PetscMax(svd->nrma,svd->nrmb): 1.0;

  /* normalize start vector */
  if (!svd->ninil) {
    PetscCall(BVSetRandomColumn(U1,0));
    PetscCall(BVOrthonormalizeColumn(U1,0,PETSC_TRUE,NULL,NULL));
  }

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    beta = alpha + ld;
    PetscCall(SVDTwoSideLanczosGSingle(svd,alpha,beta,lanczos->Z,V,U1,lanczos->ksp,svd->nconv+l,&nv,&breakdown));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));
    PetscCall(BVSetActiveColumns(V,svd->nconv,nv));
    PetscCall(BVSetActiveColumns(U1,svd->nconv,nv));

    /* solve projected problem */
    PetscCall(DSSetDimensions(svd->ds,nv,svd->nconv,svd->nconv+l));
    PetscCall(DSSVDSetDimensions(svd->ds,nv));
    PetscCall(DSSetState(svd->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    PetscCall(DSSolve(svd->ds,w,NULL));
    PetscCall(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
    PetscCall(DSUpdateExtraRow(svd->ds));
    PetscCall(DSSynchronize(svd->ds,w,NULL));
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    PetscCall(SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,normr,&k));
    PetscCall((*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx));

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) PetscCall(PetscInfo(svd,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (PetscUnlikely(breakdown || k==nv)) {
        /* Start a new bidiagonalization */
        PetscCall(PetscInfo(svd,"Breakdown in bidiagonalization (it=%" PetscInt_FMT ")\n",svd->its));
        if (k<svd->nsv) {
          PetscCall(BVSetRandomColumn(U1,k));
          PetscCall(BVOrthonormalizeColumn(U1,k,PETSC_FALSE,NULL,&breakdown));
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            PetscCall(PetscInfo(svd,"Unable to generate more start vectors\n"));
          }
        }
      } else PetscCall(DSTruncate(svd->ds,k+l,PETSC_FALSE));
    }

    /* compute converged singular vectors and restart vectors */
    PetscCall(DSGetMat(svd->ds,DS_MAT_U,&U));
    PetscCall(BVMultInPlace(V,U,svd->nconv,k+l));
    PetscCall(MatDestroy(&U));
    PetscCall(DSGetMat(svd->ds,DS_MAT_V,&VV));
    PetscCall(BVMultInPlace(U1,VV,svd->nconv,k+l));
    PetscCall(MatDestroy(&VV));

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) PetscCall(BVCopyColumn(U1,nv,k+l));

    svd->nconv = k;
    PetscCall(SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv));
  }

  PetscCall(PetscFree(w));
  PetscFunctionReturn(0);
}

/* Move generalized left singular vectors (0..nconv) from U1 and U2 to its final destination svd->U (single variant) */
static inline PetscErrorCode SVDLeftSingularVectors_Single(SVD svd,BV U1,BV U2)
{
  PetscInt          i,k,m,p;
  Vec               u,u1,u2;
  PetscScalar       *ua,*u2a;
  const PetscScalar *u1a;
  PetscReal         s;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(svd->A,&m,NULL));
  PetscCall(MatGetLocalSize(svd->B,&p,NULL));
  for (i=0;i<svd->nconv;i++) {
    PetscCall(BVGetColumn(U1,i,&u1));
    PetscCall(BVGetColumn(U2,i,&u2));
    PetscCall(BVGetColumn(svd->U,i,&u));
    PetscCall(VecGetArrayRead(u1,&u1a));
    PetscCall(VecGetArray(u,&ua));
    PetscCall(VecGetArray(u2,&u2a));
    /* Copy column from U1 to upper part of u */
    for (k=0;k<m;k++) ua[k] = u1a[k];
    /* Copy column from lower part of U to U2. Orthogonalize column in U2 and copy back to U */
    for (k=0;k<p;k++) u2a[k] = ua[m+k];
    PetscCall(VecRestoreArray(u2,&u2a));
    PetscCall(BVRestoreColumn(U2,i,&u2));
    PetscCall(BVOrthonormalizeColumn(U2,i,PETSC_FALSE,&s,NULL));
    PetscCall(BVGetColumn(U2,i,&u2));
    PetscCall(VecGetArray(u2,&u2a));
    for (k=0;k<p;k++) ua[m+k] = u2a[k];
    /* Update singular value */
    svd->sigma[i] /= s;
    PetscCall(VecRestoreArrayRead(u1,&u1a));
    PetscCall(VecRestoreArray(u,&ua));
    PetscCall(VecRestoreArray(u2,&u2a));
    PetscCall(BVRestoreColumn(U1,i,&u1));
    PetscCall(BVRestoreColumn(U2,i,&u2));
    PetscCall(BVRestoreColumn(svd->U,i,&u));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTwoSideLanczosGUpper(SVD svd,PetscReal *alpha,PetscReal *beta,PetscReal *alphah,PetscReal *betah,Mat Z,BV U1,BV U2,BV V,KSP ksp,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscInt          i,j,m,p;
  const PetscScalar *carr;
  PetscScalar       *arr,*u2arr;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0],v1,u2;
  PetscBool         lindep=PETSC_FALSE,lindep1=PETSC_FALSE,lindep2=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatCreateVecsEmpty(svd->A,NULL,&v1));
  PetscCall(MatGetLocalSize(svd->A,&m,NULL));
  PetscCall(MatGetLocalSize(svd->B,&p,NULL));

  for (i=k; i<*n; i++) {
    /* Compute vector i of BV U1 */
    PetscCall(BVGetColumn(V,i,&v));
    PetscCall(VecGetArrayRead(v,&carr));
    PetscCall(VecPlaceArray(v1,carr));
    PetscCall(BVInsertVec(U1,i,v1));
    PetscCall(VecResetArray(v1));
    PetscCall(BVOrthonormalizeColumn(U1,i,PETSC_FALSE,alpha+i,&lindep1));

    /* Compute vector i of BV U2 */
    PetscCall(BVGetColumn(U2,i,&u2));
    PetscCall(VecGetArray(u2,&u2arr));
    if (i%2) {
      for (j=0; j<p; j++) u2arr[j] = -carr[m+j];
    } else {
      for (j=0; j<p; j++) u2arr[j] = carr[m+j];
    }
    PetscCall(VecRestoreArray(u2,&u2arr));
    PetscCall(BVRestoreColumn(U2,i,&u2));
    PetscCall(VecRestoreArrayRead(v,&carr));
    PetscCall(BVRestoreColumn(V,i,&v));
    PetscCall(BVOrthonormalizeColumn(U2,i,PETSC_FALSE,alphah+i,&lindep2));
    if (i%2) alphah[i] = -alphah[i];
    if (PetscUnlikely(lindep1 || lindep2)) {
      lindep = PETSC_TRUE;
      *n = i;
      break;
    }

    /* Compute vector i+1 of BV V */
    PetscCall(BVGetColumn(V,i+1,&v));
    /* Form ut=[u;0] */
    PetscCall(BVGetColumn(U1,i,&u));
    PetscCall(VecZeroEntries(ut));
    PetscCall(VecGetArrayRead(u,&carr));
    PetscCall(VecGetArray(ut,&arr));
    for (j=0; j<m; j++) arr[j] = carr[j];
    PetscCall(VecRestoreArrayRead(u,&carr));
    PetscCall(VecRestoreArray(ut,&arr));
    /* Solve least squares problem */
    PetscCall(KSPSolve(ksp,ut,x));
    PetscCall(MatMult(Z,x,v));
    PetscCall(BVRestoreColumn(U1,i,&u));
    PetscCall(BVRestoreColumn(V,i+1,&v));
    PetscCall(BVOrthonormalizeColumn(V,i+1,PETSC_FALSE,beta+i,&lindep));
    betah[i] = -alpha[i]*beta[i]/alphah[i];
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }
  }
  if (breakdown) *breakdown = lindep;
  PetscCall(VecDestroy(&v1));
  PetscFunctionReturn(0);
}

/* generate random initial vector in column k for joint upper-upper bidiagonalization */
static inline PetscErrorCode SVDInitialVectorGUpper(SVD svd,BV V,BV U1,PetscInt k,PetscBool *breakdown)
{
  SVD_TRLANCZOS     *lanczos = (SVD_TRLANCZOS*)svd->data;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0];
  PetscInt          m,j;
  PetscScalar       *arr;
  const PetscScalar *carr;

  PetscFunctionBegin;
  /* Form ut=[u;0] where u is the k-th column of U1 */
  PetscCall(VecZeroEntries(ut));
  PetscCall(BVGetColumn(U1,k,&u));
  PetscCall(VecGetLocalSize(u,&m));
  PetscCall(VecGetArrayRead(u,&carr));
  PetscCall(VecGetArray(ut,&arr));
  for (j=0; j<m; j++) arr[j] = carr[j];
  PetscCall(VecRestoreArrayRead(u,&carr));
  PetscCall(VecRestoreArray(ut,&arr));
  PetscCall(BVRestoreColumn(U1,k,&u));
  /* Solve least squares problem Z*x=ut for x. Then set v=Z*x */
  PetscCall(KSPSolve(lanczos->ksp,ut,x));
  PetscCall(BVGetColumn(V,k,&v));
  PetscCall(MatMult(lanczos->Z,x,v));
  PetscCall(BVRestoreColumn(V,k,&v));
  if (breakdown) PetscCall(BVOrthonormalizeColumn(V,k,PETSC_FALSE,NULL,breakdown));
  else PetscCall(BVOrthonormalizeColumn(V,k,PETSC_TRUE,NULL,NULL));
  PetscFunctionReturn(0);
}

/* solve generalized problem with joint upper-upper bidiagonalization */
PetscErrorCode SVDSolve_TRLanczosGUpper(SVD svd,BV U1,BV U2,BV V)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta,*alphah,*betah,normr;
  PetscScalar    *w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,Vmat,X;
  PetscBool      breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DSGetLeadingDimension(svd->ds,&ld));
  PetscCall(PetscMalloc1(ld,&w));
  normr = (svd->conv==SVD_CONV_ABS)? PetscMax(svd->nrma,svd->nrmb): 1.0;

  /* normalize start vector */
  if (!svd->ninil) PetscCall(BVSetRandomColumn(U1,0));
  PetscCall(SVDInitialVectorGUpper(svd,V,U1,0,NULL));

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_D,&alphah));
    beta = alpha + ld;
    betah = alpha + 2*ld;
    PetscCall(SVDTwoSideLanczosGUpper(svd,alpha,beta,alphah,betah,lanczos->Z,U1,U2,V,lanczos->ksp,svd->nconv+l,&nv,&breakdown));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_D,&alphah));
    PetscCall(BVSetActiveColumns(V,svd->nconv,nv));
    PetscCall(BVSetActiveColumns(U1,svd->nconv,nv));
    PetscCall(BVSetActiveColumns(U2,svd->nconv,nv));

    /* solve projected problem */
    PetscCall(DSSetDimensions(svd->ds,nv,svd->nconv,svd->nconv+l));
    PetscCall(DSGSVDSetDimensions(svd->ds,nv,nv));
    PetscCall(DSSetState(svd->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    PetscCall(DSSolve(svd->ds,w,NULL));
    PetscCall(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
    PetscCall(DSUpdateExtraRow(svd->ds));
    PetscCall(DSSynchronize(svd->ds,w,NULL));
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    PetscCall(SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,normr,&k));
    PetscCall((*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx));

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) PetscCall(PetscInfo(svd,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (PetscUnlikely(breakdown || k==nv)) {
        /* Start a new bidiagonalization */
        PetscCall(PetscInfo(svd,"Breakdown in bidiagonalization (it=%" PetscInt_FMT ")\n",svd->its));
        if (k<svd->nsv) {
          PetscCall(BVSetRandomColumn(U1,k));
          PetscCall(SVDInitialVectorGUpper(svd,V,U1,k,&breakdown));
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            PetscCall(PetscInfo(svd,"Unable to generate more start vectors\n"));
          }
        }
      } else PetscCall(DSTruncate(svd->ds,k+l,PETSC_FALSE));
    }
    /* compute converged singular vectors and restart vectors */
    PetscCall(DSGetMat(svd->ds,DS_MAT_X,&X));
    PetscCall(BVMultInPlace(V,X,svd->nconv,k+l));
    PetscCall(MatDestroy(&X));
    PetscCall(DSGetMat(svd->ds,DS_MAT_U,&U));
    PetscCall(BVMultInPlace(U1,U,svd->nconv,k+l));
    PetscCall(MatDestroy(&U));
    PetscCall(DSGetMat(svd->ds,DS_MAT_V,&Vmat));
    PetscCall(BVMultInPlace(U2,Vmat,svd->nconv,k+l));
    PetscCall(MatDestroy(&Vmat));

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) PetscCall(BVCopyColumn(V,nv,k+l));

    svd->nconv = k;
    PetscCall(SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv));
  }

  PetscCall(PetscFree(w));
  PetscFunctionReturn(0);
}

/* Move generalized left singular vectors (0..nconv) from U1 and U2 to its final destination svd->U (upper and lower variants) */
static inline PetscErrorCode SVDLeftSingularVectors(SVD svd,BV U1,BV U2)
{
  PetscInt          i,k,m,p;
  Vec               u,u1,u2;
  PetscScalar       *ua;
  const PetscScalar *u1a,*u2a;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(svd->A,&m,NULL));
  PetscCall(MatGetLocalSize(svd->B,&p,NULL));
  for (i=0;i<svd->nconv;i++) {
    PetscCall(BVGetColumn(U1,i,&u1));
    PetscCall(BVGetColumn(U2,i,&u2));
    PetscCall(BVGetColumn(svd->U,i,&u));
    PetscCall(VecGetArrayRead(u1,&u1a));
    PetscCall(VecGetArrayRead(u2,&u2a));
    PetscCall(VecGetArray(u,&ua));
    /* Copy column from u1 to upper part of u */
    for (k=0;k<m;k++) ua[k] = u1a[k];
    /* Copy column from u2 to lower part of u */
    for (k=0;k<p;k++) ua[m+k] = u2a[k];
    PetscCall(VecRestoreArrayRead(u1,&u1a));
    PetscCall(VecRestoreArrayRead(u2,&u2a));
    PetscCall(VecRestoreArray(u,&ua));
    PetscCall(BVRestoreColumn(U1,i,&u1));
    PetscCall(BVRestoreColumn(U2,i,&u2));
    PetscCall(BVRestoreColumn(svd->U,i,&u));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTwoSideLanczosGLower(SVD svd,PetscReal *alpha,PetscReal *beta,PetscReal *alphah,PetscReal *betah,Mat Z,BV U1,BV U2,BV V,KSP ksp,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscInt          i,j,m,p;
  const PetscScalar *carr;
  PetscScalar       *arr,*u2arr;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0],v1,u2;
  PetscBool         lindep=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatCreateVecsEmpty(svd->A,NULL,&v1));
  PetscCall(MatGetLocalSize(svd->A,&m,NULL));
  PetscCall(MatGetLocalSize(svd->B,&p,NULL));

  for (i=k; i<*n; i++) {
    /* Compute vector i of BV U2 */
    PetscCall(BVGetColumn(V,i,&v));
    PetscCall(VecGetArrayRead(v,&carr));
    PetscCall(BVGetColumn(U2,i,&u2));
    PetscCall(VecGetArray(u2,&u2arr));
    if (i%2) {
      for (j=0; j<p; j++) u2arr[j] = -carr[m+j];
    } else {
      for (j=0; j<p; j++) u2arr[j] = carr[m+j];
    }
    PetscCall(VecRestoreArray(u2,&u2arr));
    PetscCall(BVRestoreColumn(U2,i,&u2));
    PetscCall(BVOrthonormalizeColumn(U2,i,PETSC_FALSE,alphah+i,&lindep));
    if (i%2) alphah[i] = -alphah[i];
    if (PetscUnlikely(lindep)) {
      PetscCall(BVRestoreColumn(V,i,&v));
      *n = i;
      break;
    }

    /* Compute vector i+1 of BV U1 */
    PetscCall(VecPlaceArray(v1,carr));
    PetscCall(BVInsertVec(U1,i+1,v1));
    PetscCall(VecResetArray(v1));
    PetscCall(BVOrthonormalizeColumn(U1,i+1,PETSC_FALSE,beta+i,&lindep));
    PetscCall(VecRestoreArrayRead(v,&carr));
    PetscCall(BVRestoreColumn(V,i,&v));
    if (PetscUnlikely(lindep)) {
      *n = i+1;
      break;
    }

    /* Compute vector i+1 of BV V */
    PetscCall(BVGetColumn(V,i+1,&v));
    /* Form ut=[u;0] where u is column i+1 of BV U1 */
    PetscCall(BVGetColumn(U1,i+1,&u));
    PetscCall(VecZeroEntries(ut));
    PetscCall(VecGetArrayRead(u,&carr));
    PetscCall(VecGetArray(ut,&arr));
    for (j=0; j<m; j++) arr[j] = carr[j];
    PetscCall(VecRestoreArrayRead(u,&carr));
    PetscCall(VecRestoreArray(ut,&arr));
    /* Solve least squares problem */
    PetscCall(KSPSolve(ksp,ut,x));
    PetscCall(MatMult(Z,x,v));
    PetscCall(BVRestoreColumn(U1,i+1,&u));
    PetscCall(BVRestoreColumn(V,i+1,&v));
    PetscCall(BVOrthonormalizeColumn(V,i+1,PETSC_FALSE,alpha+i+1,&lindep));
    betah[i] = -alpha[i+1]*beta[i]/alphah[i];
    if (PetscUnlikely(lindep)) {
      *n = i+1;
      break;
    }
  }
  if (breakdown) *breakdown = lindep;
  PetscCall(VecDestroy(&v1));
  PetscFunctionReturn(0);
}

/* generate random initial vector in column k for joint lower-upper bidiagonalization */
static inline PetscErrorCode SVDInitialVectorGLower(SVD svd,BV V,BV U1,BV U2,PetscInt k,PetscBool *breakdown)
{
  SVD_TRLANCZOS     *lanczos = (SVD_TRLANCZOS*)svd->data;
  const PetscScalar *carr;
  PetscScalar       *arr;
  PetscReal         *alpha;
  PetscInt          j,m,p;
  Vec               u,uh,v,ut=svd->workl[0],x=svd->workr[0];

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(svd->A,&m,NULL));
  PetscCall(MatGetLocalSize(svd->B,&p,NULL));
  /* Form ut=[0;uh], where uh is the k-th column of U2 */
  PetscCall(BVGetColumn(U2,k,&uh));
  PetscCall(VecZeroEntries(ut));
  PetscCall(VecGetArrayRead(uh,&carr));
  PetscCall(VecGetArray(ut,&arr));
  for (j=0; j<p; j++) arr[m+j] = carr[j];
  PetscCall(VecRestoreArrayRead(uh,&carr));
  PetscCall(VecRestoreArray(ut,&arr));
  PetscCall(BVRestoreColumn(U2,k,&uh));
  /* Solve least squares problem Z*x=ut for x. Then set ut=Z*x */
  PetscCall(KSPSolve(lanczos->ksp,ut,x));
  PetscCall(MatMult(lanczos->Z,x,ut));
  /* Form u, column k of BV U1, as the upper part of ut and orthonormalize */
  PetscCall(MatCreateVecsEmpty(svd->A,NULL,&u));
  PetscCall(VecGetArrayRead(ut,&carr));
  PetscCall(VecPlaceArray(u,carr));
  PetscCall(BVInsertVec(U1,k,u));
  PetscCall(VecResetArray(u));
  PetscCall(VecRestoreArrayRead(ut,&carr));
  PetscCall(VecDestroy(&u));
  if (breakdown) PetscCall(BVOrthonormalizeColumn(U1,k,PETSC_FALSE,NULL,breakdown));
  else PetscCall(BVOrthonormalizeColumn(U1,k,PETSC_TRUE,NULL,NULL));

  if (!breakdown || !*breakdown) {
    PetscCall(MatGetLocalSize(svd->A,&m,NULL));
    /* Compute k-th vector of BV V */
    PetscCall(BVGetColumn(V,k,&v));
    /* Form ut=[u;0] where u is the 1st column of U1 */
    PetscCall(BVGetColumn(U1,k,&u));
    PetscCall(VecZeroEntries(ut));
    PetscCall(VecGetArrayRead(u,&carr));
    PetscCall(VecGetArray(ut,&arr));
    for (j=0; j<m; j++) arr[j] = carr[j];
    PetscCall(VecRestoreArrayRead(u,&carr));
    PetscCall(VecRestoreArray(ut,&arr));
    /* Solve least squares problem */
    PetscCall(KSPSolve(lanczos->ksp,ut,x));
    PetscCall(MatMult(lanczos->Z,x,v));
    PetscCall(BVRestoreColumn(U1,k,&u));
    PetscCall(BVRestoreColumn(V,k,&v));
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    if (breakdown) PetscCall(BVOrthonormalizeColumn(V,k,PETSC_FALSE,alpha+k,breakdown));
    else PetscCall(BVOrthonormalizeColumn(V,k,PETSC_TRUE,alpha+k,NULL));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));
  }
  PetscFunctionReturn(0);
}

/* solve generalized problem with joint lower-upper bidiagonalization */
PetscErrorCode SVDSolve_TRLanczosGLower(SVD svd,BV U1,BV U2,BV V)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta,*alphah,*betah,normr;
  PetscScalar    *w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,Vmat,X;
  PetscBool      breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DSGetLeadingDimension(svd->ds,&ld));
  PetscCall(PetscMalloc1(ld,&w));
  normr = (svd->conv==SVD_CONV_ABS)? PetscMax(svd->nrma,svd->nrmb): 1.0;

  /* normalize start vector */
  if (!svd->ninil) PetscCall(BVSetRandomColumn(U2,0));
  PetscCall(SVDInitialVectorGLower(svd,V,U1,U2,0,NULL));

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    PetscCall(DSGetArrayReal(svd->ds,DS_MAT_D,&alphah));
    beta = alpha + ld;
    betah = alpha + 2*ld;
    PetscCall(SVDTwoSideLanczosGLower(svd,alpha,beta,alphah,betah,lanczos->Z,U1,U2,V,lanczos->ksp,svd->nconv+l,&nv,&breakdown));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));
    PetscCall(DSRestoreArrayReal(svd->ds,DS_MAT_D,&alphah));
    PetscCall(BVSetActiveColumns(V,svd->nconv,nv));
    PetscCall(BVSetActiveColumns(U1,svd->nconv,nv+1));
    PetscCall(BVSetActiveColumns(U2,svd->nconv,nv));

    /* solve projected problem */
    PetscCall(DSSetDimensions(svd->ds,nv+1,svd->nconv,svd->nconv+l));
    PetscCall(DSGSVDSetDimensions(svd->ds,nv,nv));
    PetscCall(DSSetState(svd->ds,l?DS_STATE_RAW:DS_STATE_INTERMEDIATE));
    PetscCall(DSSolve(svd->ds,w,NULL));
    PetscCall(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
    PetscCall(DSUpdateExtraRow(svd->ds));
    PetscCall(DSSynchronize(svd->ds,w,NULL));
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    PetscCall(SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,normr,&k));
    PetscCall((*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx));

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) PetscCall(PetscInfo(svd,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (PetscUnlikely(breakdown || k==nv)) {
        /* Start a new bidiagonalization */
        PetscCall(PetscInfo(svd,"Breakdown in bidiagonalization (it=%" PetscInt_FMT ")\n",svd->its));
        if (k<svd->nsv) {
          PetscCall(BVSetRandomColumn(U2,k));
          PetscCall(SVDInitialVectorGLower(svd,V,U1,U2,k,&breakdown));
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            PetscCall(PetscInfo(svd,"Unable to generate more start vectors\n"));
          }
        }
      } else PetscCall(DSTruncate(svd->ds,k+l,PETSC_FALSE));
    }

    /* compute converged singular vectors and restart vectors */
    PetscCall(DSGetMat(svd->ds,DS_MAT_X,&X));
    PetscCall(BVMultInPlace(V,X,svd->nconv,k+l));
    PetscCall(MatDestroy(&X));
    PetscCall(DSGetMat(svd->ds,DS_MAT_U,&U));
    PetscCall(BVMultInPlace(U1,U,svd->nconv,k+l+1));
    PetscCall(MatDestroy(&U));
    PetscCall(DSGetMat(svd->ds,DS_MAT_V,&Vmat));
    PetscCall(BVMultInPlace(U2,Vmat,svd->nconv,k+l));
    PetscCall(MatDestroy(&Vmat));

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) PetscCall(BVCopyColumn(V,nv,k+l));

    svd->nconv = k;
    PetscCall(SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv));
  }

  PetscCall(PetscFree(w));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_TRLanczos_GSVD(SVD svd)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscInt       k,m,p;
  PetscBool      convchg=PETSC_FALSE;
  BV             U1,U2;
  BVType         type;
  Mat            U,V;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citationg,&citedg));

  if (svd->converged==SVDConvergedNorm) {  /* override temporarily since computed residual is already relative to the norms */
    svd->converged = SVDConvergedAbsolute;
    convchg = PETSC_TRUE;
  }
  PetscCall(MatGetLocalSize(svd->A,&m,NULL));
  PetscCall(MatGetLocalSize(svd->B,&p,NULL));

  /* Create BV for U1 */
  PetscCall(BVCreate(PetscObjectComm((PetscObject)svd),&U1));
  PetscCall(PetscLogObjectParent((PetscObject)svd,(PetscObject)U1));
  PetscCall(BVGetType(svd->U,&type));
  PetscCall(BVSetType(U1,type));
  PetscCall(BVGetSizes(svd->U,NULL,NULL,&k));
  PetscCall(BVSetSizes(U1,m,PETSC_DECIDE,k));

  /* Create BV for U2 */
  PetscCall(BVCreate(PetscObjectComm((PetscObject)svd),&U2));
  PetscCall(PetscLogObjectParent((PetscObject)svd,(PetscObject)U2));
  PetscCall(BVSetType(U2,type));
  PetscCall(BVSetSizes(U2,p,PETSC_DECIDE,k));

  /* Copy initial vectors from svd->U to U1 and U2 */
  if (svd->ninil) {
    Vec u, uh, nest, aux[2];
    PetscCall(BVGetColumn(U1,0,&u));
    PetscCall(BVGetColumn(U2,0,&uh));
    aux[0] = u;
    aux[1] = uh;
    PetscCall(VecCreateNest(PetscObjectComm((PetscObject)svd),2,NULL,aux,&nest));
    PetscCall(BVCopyVec(svd->U,0,nest));
    PetscCall(BVRestoreColumn(U1,0,&u));
    PetscCall(BVRestoreColumn(U2,0,&uh));
    PetscCall(VecDestroy(&nest));
  }

  switch (lanczos->bidiag) {
    case SVD_TRLANCZOS_GBIDIAG_SINGLE:
      PetscCall(SVDSolve_TRLanczosGSingle(svd,U1,svd->U));
      break;
    case SVD_TRLANCZOS_GBIDIAG_UPPER:
      PetscCall(SVDSolve_TRLanczosGUpper(svd,U1,U2,svd->U));
      break;
    case SVD_TRLANCZOS_GBIDIAG_LOWER:
      PetscCall(SVDSolve_TRLanczosGLower(svd,U1,U2,svd->U));
      break;
  }

  /* Compute converged right singular vectors */
  PetscCall(BVSetActiveColumns(svd->U,0,svd->nconv));
  PetscCall(BVSetActiveColumns(svd->V,0,svd->nconv));
  PetscCall(BVGetMat(svd->U,&U));
  PetscCall(BVGetMat(svd->V,&V));
  PetscCall(KSPMatSolve(lanczos->ksp,U,V));
  PetscCall(BVRestoreMat(svd->U,&U));
  PetscCall(BVRestoreMat(svd->V,&V));

  /* Finish computing left singular vectors and move them to its place */
  switch (lanczos->bidiag) {
    case SVD_TRLANCZOS_GBIDIAG_SINGLE:
      PetscCall(SVDLeftSingularVectors_Single(svd,U1,U2));
      break;
    case SVD_TRLANCZOS_GBIDIAG_UPPER:
    case SVD_TRLANCZOS_GBIDIAG_LOWER:
      PetscCall(SVDLeftSingularVectors(svd,U1,U2));
      break;
  }

  PetscCall(BVDestroy(&U2));
  PetscCall(BVDestroy(&U1));
  PetscCall(DSTruncate(svd->ds,svd->nconv,PETSC_TRUE));
  if (convchg) svd->converged = SVDConvergedNorm;
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_TRLanczos(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  SVD_TRLANCZOS       *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscBool           flg,val,lock;
  PetscReal           keep;
  SVDTRLanczosGBidiag bidiag;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"SVD TRLanczos Options"));

    PetscCall(PetscOptionsBool("-svd_trlanczos_oneside","Use one-side reorthogonalization","SVDTRLanczosSetOneSide",lanczos->oneside,&val,&flg));
    if (flg) PetscCall(SVDTRLanczosSetOneSide(svd,val));

    PetscCall(PetscOptionsReal("-svd_trlanczos_restart","Proportion of vectors kept after restart","SVDTRLanczosSetRestart",0.5,&keep,&flg));
    if (flg) PetscCall(SVDTRLanczosSetRestart(svd,keep));

    PetscCall(PetscOptionsBool("-svd_trlanczos_locking","Choose between locking and non-locking variants","SVDTRLanczosSetLocking",PETSC_TRUE,&lock,&flg));
    if (flg) PetscCall(SVDTRLanczosSetLocking(svd,lock));

    PetscCall(PetscOptionsEnum("-svd_trlanczos_gbidiag","Bidiagonalization choice for Generalized Problem","SVDTRLanczosSetGBidiag",SVDTRLanczosGBidiags,(PetscEnum)lanczos->bidiag,(PetscEnum*)&bidiag,&flg));
    if (flg) PetscCall(SVDTRLanczosSetGBidiag(svd,bidiag));

    PetscCall(PetscOptionsBool("-svd_trlanczos_explicitmatrix","Build explicit matrix for KSP solver","SVDTRLanczosSetExplicitMatrix",lanczos->explicitmatrix,&val,&flg));
    if (flg) PetscCall(SVDTRLanczosSetExplicitMatrix(svd,val));

  PetscCall(PetscOptionsTail());

  if (svd->OPb) {
    if (!lanczos->ksp) PetscCall(SVDTRLanczosGetKSP(svd,&lanczos->ksp));
    PetscCall(KSPSetFromOptions(lanczos->ksp));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetOneSide_TRLanczos(SVD svd,PetscBool oneside)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (lanczos->oneside != oneside) {
    lanczos->oneside = oneside;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetOneSide - Indicate if the variant of the Lanczos method
   to be used is one-sided or two-sided.

   Logically Collective on svd

   Input Parameters:
+  svd     - singular value solver
-  oneside - boolean flag indicating if the method is one-sided or not

   Options Database Key:
.  -svd_trlanczos_oneside <boolean> - Indicates the boolean flag

   Note:
   By default, a two-sided variant is selected, which is sometimes slightly
   more robust. However, the one-sided variant is faster because it avoids
   the orthogonalization associated to left singular vectors.

   Level: advanced

.seealso: SVDLanczosSetOneSide()
@*/
PetscErrorCode SVDTRLanczosSetOneSide(SVD svd,PetscBool oneside)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,oneside,2);
  PetscTryMethod(svd,"SVDTRLanczosSetOneSide_C",(SVD,PetscBool),(svd,oneside));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetOneSide_TRLanczos(SVD svd,PetscBool *oneside)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *oneside = lanczos->oneside;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetOneSide - Gets if the variant of the Lanczos method
   to be used is one-sided or two-sided.

   Not Collective

   Input Parameters:
.  svd     - singular value solver

   Output Parameters:
.  oneside - boolean flag indicating if the method is one-sided or not

   Level: advanced

.seealso: SVDTRLanczosSetOneSide()
@*/
PetscErrorCode SVDTRLanczosGetOneSide(SVD svd,PetscBool *oneside)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(oneside,2);
  PetscUseMethod(svd,"SVDTRLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetGBidiag_TRLanczos(SVD svd,SVDTRLanczosGBidiag bidiag)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  switch (bidiag) {
    case SVD_TRLANCZOS_GBIDIAG_SINGLE:
    case SVD_TRLANCZOS_GBIDIAG_UPPER:
    case SVD_TRLANCZOS_GBIDIAG_LOWER:
      if (lanczos->bidiag != bidiag) {
        lanczos->bidiag = bidiag;
        svd->state = SVD_STATE_INITIAL;
      }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Invalid bidiagonalization choice");
  }
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetGBidiag - Sets the bidiagonalization choice to use in
   the GSVD TRLanczos solver.

   Logically Collective on svd

   Input Parameters:
+  svd - the singular value solver
-  bidiag - the bidiagonalization choice

   Options Database Key:
.  -svd_trlanczos_gbidiag - Sets the bidiagonalization choice (either 's' or 'juu'
   or 'jlu')

   Level: advanced

.seealso: SVDTRLanczosGetGBidiag(), SVDTRLanczosGBidiag
@*/
PetscErrorCode SVDTRLanczosSetGBidiag(SVD svd,SVDTRLanczosGBidiag bidiag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svd,bidiag,2);
  PetscTryMethod(svd,"SVDTRLanczosSetGBidiag_C",(SVD,SVDTRLanczosGBidiag),(svd,bidiag));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetGBidiag_TRLanczos(SVD svd,SVDTRLanczosGBidiag *bidiag)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *bidiag = lanczos->bidiag;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetGBidiag - Gets the bidiagonalization choice used in the GSVD
   TRLanczos solver.

   Not Collective

   Input Parameter:
.  svd - the singular value solver

   Output Parameter:
.  bidiag - the bidiagonalization choice

   Level: advanced

.seealso: SVDTRLanczosSetGBidiag(), SVDTRLanczosGBidiag
@*/
PetscErrorCode SVDTRLanczosGetGBidiag(SVD svd,SVDTRLanczosGBidiag *bidiag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(bidiag,2);
  PetscUseMethod(svd,"SVDTRLanczosGetGBidiag_C",(SVD,SVDTRLanczosGBidiag*),(svd,bidiag));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetKSP_TRLanczos(SVD svd,KSP ksp)
{
  SVD_TRLANCZOS  *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&ctx->ksp));
  ctx->ksp = ksp;
  PetscCall(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->ksp));
  svd->state = SVD_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetKSP - Associate a linear solver object (KSP) to the SVD solver.

   Collective on svd

   Input Parameters:
+  svd - SVD solver
-  ksp - the linear solver object

   Note:
   Only used for the GSVD problem.

   Level: advanced

.seealso: SVDTRLanczosGetKSP()
@*/
PetscErrorCode SVDTRLanczosSetKSP(SVD svd,KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(svd,1,ksp,2);
  PetscTryMethod(svd,"SVDTRLanczosSetKSP_C",(SVD,KSP),(svd,ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetKSP_TRLanczos(SVD svd,KSP *ksp)
{
  SVD_TRLANCZOS  *ctx = (SVD_TRLANCZOS*)svd->data;
  PC             pc;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    /* Create linear solver */
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)svd),&ctx->ksp));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)svd,1));
    PetscCall(KSPSetOptionsPrefix(ctx->ksp,((PetscObject)svd)->prefix));
    PetscCall(KSPAppendOptionsPrefix(ctx->ksp,"svd_trlanczos_"));
    PetscCall(PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->ksp));
    PetscCall(PetscObjectSetOptions((PetscObject)ctx->ksp,((PetscObject)svd)->options));
    PetscCall(KSPSetType(ctx->ksp,KSPLSQR));
    PetscCall(KSPGetPC(ctx->ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));
    PetscCall(KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE));
    PetscCall(KSPSetTolerances(ctx->ksp,SlepcDefaultTol(svd->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetKSP - Retrieve the linear solver object (KSP) associated with
   the SVD solver.

   Not Collective

   Input Parameter:
.  svd - SVD solver

   Output Parameter:
.  ksp - the linear solver object

   Level: advanced

.seealso: SVDTRLanczosSetKSP()
@*/
PetscErrorCode SVDTRLanczosGetKSP(SVD svd,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(ksp,2);
  PetscUseMethod(svd,"SVDTRLanczosGetKSP_C",(SVD,KSP*),(svd,ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetRestart_TRLanczos(SVD svd,PetscReal keep)
{
  SVD_TRLANCZOS *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    PetscCheck(keep>=0.1 && keep<=0.9,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument %g must be in the range [0.1,0.9]",(double)keep);
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetRestart - Sets the restart parameter for the thick-restart
   Lanczos method, in particular the proportion of basis vectors that must be
   kept after restart.

   Logically Collective on svd

   Input Parameters:
+  svd  - the singular value solver
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -svd_trlanczos_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: SVDTRLanczosGetRestart()
@*/
PetscErrorCode SVDTRLanczosSetRestart(SVD svd,PetscReal keep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveReal(svd,keep,2);
  PetscTryMethod(svd,"SVDTRLanczosSetRestart_C",(SVD,PetscReal),(svd,keep));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetRestart_TRLanczos(SVD svd,PetscReal *keep)
{
  SVD_TRLANCZOS *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetRestart - Gets the restart parameter used in the thick-restart
   Lanczos method.

   Not Collective

   Input Parameter:
.  svd - the singular value solver

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: SVDTRLanczosSetRestart()
@*/
PetscErrorCode SVDTRLanczosGetRestart(SVD svd,PetscReal *keep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidRealPointer(keep,2);
  PetscUseMethod(svd,"SVDTRLanczosGetRestart_C",(SVD,PetscReal*),(svd,keep));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetLocking_TRLanczos(SVD svd,PetscBool lock)
{
  SVD_TRLANCZOS *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetLocking - Choose between locking and non-locking variants of
   the thick-restart Lanczos method.

   Logically Collective on svd

   Input Parameters:
+  svd  - the singular value solver
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -svd_trlanczos_locking - Sets the locking flag

   Notes:
   The default is to lock converged singular triplets when the method restarts.
   This behaviour can be changed so that all directions are kept in the
   working subspace even if already converged to working accuracy (the
   non-locking variant).

   Level: advanced

.seealso: SVDTRLanczosGetLocking()
@*/
PetscErrorCode SVDTRLanczosSetLocking(SVD svd,PetscBool lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,lock,2);
  PetscTryMethod(svd,"SVDTRLanczosSetLocking_C",(SVD,PetscBool),(svd,lock));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetLocking_TRLanczos(SVD svd,PetscBool *lock)
{
  SVD_TRLANCZOS *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetLocking - Gets the locking flag used in the thick-restart
   Lanczos method.

   Not Collective

   Input Parameter:
.  svd - the singular value solver

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: SVDTRLanczosSetLocking()
@*/
PetscErrorCode SVDTRLanczosGetLocking(SVD svd,PetscBool *lock)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(lock,2);
  PetscUseMethod(svd,"SVDTRLanczosGetLocking_C",(SVD,PetscBool*),(svd,lock));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetExplicitMatrix_TRLanczos(SVD svd,PetscBool explicitmat)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (lanczos->explicitmatrix != explicitmat) {
    lanczos->explicitmatrix = explicitmat;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetExplicitMatrix - Indicate if the matrix Z=[A;B] must
   be built explicitly.

   Logically Collective on svd

   Input Parameters:
+  svd         - singular value solver
-  explicitmat - Boolean flag indicating if Z=[A;B] is built explicitly

   Options Database Key:
.  -svd_trlanczos_explicitmatrix <boolean> - Indicates the boolean flag

   Notes:
   This option is relevant for the GSVD case only.
   Z is the coefficient matrix of the KSP solver used internally.

   Level: advanced

.seealso: SVDTRLanczosGetExplicitMatrix()
@*/
PetscErrorCode SVDTRLanczosSetExplicitMatrix(SVD svd,PetscBool explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmat,2);
  PetscTryMethod(svd,"SVDTRLanczosSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmat));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetExplicitMatrix_TRLanczos(SVD svd,PetscBool *explicitmat)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *explicitmat = lanczos->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetExplicitMatrix - Returns the flag indicating if Z=[A;B] is built explicitly.

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicitmat - the mode flag

   Level: advanced

.seealso: SVDTRLanczosSetExplicitMatrix()
@*/
PetscErrorCode SVDTRLanczosGetExplicitMatrix(SVD svd,PetscBool *explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(explicitmat,2);
  PetscUseMethod(svd,"SVDTRLanczosGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmat));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_TRLanczos(SVD svd)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (svd->isgeneralized || (!svd->problem_type && svd->OPb)) {
    PetscCall(KSPReset(lanczos->ksp));
    PetscCall(MatDestroy(&lanczos->Z));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_TRLanczos(SVD svd)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (svd->isgeneralized || (!svd->problem_type && svd->OPb)) PetscCall(KSPDestroy(&lanczos->ksp));
  PetscCall(PetscFree(svd->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetOneSide_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetGBidiag_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetGBidiag_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetKSP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetKSP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetRestart_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetRestart_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetLocking_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetLocking_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetExplicitMatrix_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetExplicitMatrix_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_TRLanczos(SVD svd,PetscViewer viewer)
{
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %d%% of basis vectors kept after restart\n",(int)(100*lanczos->keep)));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  using the %slocking variant\n",lanczos->lock?"":"non-"));
    if (svd->isgeneralized) {
      const char *bidiag="";

      switch (lanczos->bidiag) {
        case SVD_TRLANCZOS_GBIDIAG_SINGLE: bidiag = "single"; break;
        case SVD_TRLANCZOS_GBIDIAG_UPPER:  bidiag = "joint upper-upper"; break;
        case SVD_TRLANCZOS_GBIDIAG_LOWER:  bidiag = "joint lower-upper"; break;
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"  bidiagonalization choice: %s\n",bidiag));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  %s matrix\n",lanczos->explicitmatrix?"explicit":"implicit"));
      if (!lanczos->ksp) PetscCall(SVDTRLanczosGetKSP(svd,&lanczos->ksp));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(KSPView(lanczos->ksp,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    } else PetscCall(PetscViewerASCIIPrintf(viewer,"  %s-sided reorthogonalization\n",lanczos->oneside? "one": "two"));
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_TRLanczos(SVD svd)
{
  SVD_TRLANCZOS  *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(svd,&ctx));
  svd->data = (void*)ctx;

  ctx->lock   = PETSC_TRUE;
  ctx->bidiag = SVD_TRLANCZOS_GBIDIAG_LOWER;

  svd->ops->setup          = SVDSetUp_TRLanczos;
  svd->ops->solve          = SVDSolve_TRLanczos;
  svd->ops->solveg         = SVDSolve_TRLanczos_GSVD;
  svd->ops->destroy        = SVDDestroy_TRLanczos;
  svd->ops->reset          = SVDReset_TRLanczos;
  svd->ops->setfromoptions = SVDSetFromOptions_TRLanczos;
  svd->ops->view           = SVDView_TRLanczos;
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C",SVDTRLanczosSetOneSide_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetOneSide_C",SVDTRLanczosGetOneSide_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetGBidiag_C",SVDTRLanczosSetGBidiag_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetGBidiag_C",SVDTRLanczosGetGBidiag_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetKSP_C",SVDTRLanczosSetKSP_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetKSP_C",SVDTRLanczosGetKSP_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetRestart_C",SVDTRLanczosSetRestart_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetRestart_C",SVDTRLanczosGetRestart_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetLocking_C",SVDTRLanczosSetLocking_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetLocking_C",SVDTRLanczosGetLocking_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetExplicitMatrix_C",SVDTRLanczosSetExplicitMatrix_TRLanczos));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetExplicitMatrix_C",SVDTRLanczosGetExplicitMatrix_TRLanczos));
  PetscFunctionReturn(0);
}
