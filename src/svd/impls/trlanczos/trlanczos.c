/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  "   title = \"Thick-restarted {Lanczos} bidigonalization methods for the {GSVD}\",\n"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(zdata);CHKERRQ(ierr);
  (*zdata)->A = svd->A;
  (*zdata)->B = svd->B;
  (*zdata)->AT = svd->AT;
  (*zdata)->BT = svd->BT;
  ierr = MatCreateVecsEmpty(svd->A,NULL,&(*zdata)->y1);CHKERRQ(ierr);
  ierr = MatCreateVecsEmpty(svd->B,NULL,&(*zdata)->y2);CHKERRQ(ierr);
  ierr = VecGetLocalSize((*zdata)->y1,&(*zdata)->m);CHKERRQ(ierr);
  ierr = BVCreateVec(svd->U,&(*zdata)->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Z(Mat Z)
{
  PetscErrorCode ierr;
  MatZData       *zdata;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Z,&zdata);CHKERRQ(ierr);
  ierr = VecDestroy(&zdata->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&zdata->y2);CHKERRQ(ierr);
  ierr = VecDestroy(&zdata->y);CHKERRQ(ierr);
  ierr = PetscFree(zdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Z(Mat Z,Vec x,Vec y)
{
  MatZData       *zdata;
  PetscScalar    *y_elems;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Z,&zdata);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_elems);CHKERRQ(ierr);
  ierr = VecPlaceArray(zdata->y1,y_elems);CHKERRQ(ierr);
  ierr = VecPlaceArray(zdata->y2,y_elems+zdata->m);CHKERRQ(ierr);

  ierr = MatMult(zdata->A,x,zdata->y1);CHKERRQ(ierr);
  ierr = MatMult(zdata->B,x,zdata->y2);CHKERRQ(ierr);

  ierr = VecResetArray(zdata->y1);CHKERRQ(ierr);
  ierr = VecResetArray(zdata->y2);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_elems);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Z(Mat Z,Vec y,Vec x)
{
  MatZData          *zdata;
  const PetscScalar *y_elems;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Z,&zdata);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y,&y_elems);CHKERRQ(ierr);
  ierr = VecPlaceArray(zdata->y1,y_elems);CHKERRQ(ierr);
  ierr = VecPlaceArray(zdata->y2,y_elems+zdata->m);CHKERRQ(ierr);

  ierr = MatMult(zdata->AT,zdata->y1,x);CHKERRQ(ierr);
  ierr = MatMultAdd(zdata->BT,zdata->y2,x,x);CHKERRQ(ierr);

  ierr = VecResetArray(zdata->y1);CHKERRQ(ierr);
  ierr = VecResetArray(zdata->y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(y,&y_elems);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateVecs_Z(Mat Z,Vec *right,Vec *left)
{
  MatZData       *zdata;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Z,&zdata);CHKERRQ(ierr);
  if (right) {
    ierr = MatCreateVecs(zdata->A,right,NULL);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecDuplicate(zdata->y,left);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetUp_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       N,m,n,p;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  DSType         dstype;
  MatZData       *zdata;
  Mat            mats[2],normal;
  MatType        Atype;
  PetscBool      sametype;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,NULL,&N);CHKERRQ(ierr);
  ierr = SVDSetDimensions_Default(svd);CHKERRQ(ierr);
  if (svd->ncv>svd->nsv+svd->mpd) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nsv+mpd");
  if (!lanczos->lock && svd->mpd<svd->ncv) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Should not use mpd parameter in non-locking variant");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  if (!lanczos->keep) lanczos->keep = 0.5;
  svd->leftbasis = PETSC_TRUE;
  ierr = SVDAllocateSolution(svd,1);CHKERRQ(ierr);
  dstype = DSSVD;
  if (svd->isgeneralized) {
    if (lanczos->bidiag==SVD_TRLANCZOS_GBIDIAG_UPPER || lanczos->bidiag==SVD_TRLANCZOS_GBIDIAG_LOWER) dstype = DSGSVD;
    ierr = SVDSetWorkVecs(svd,1,1);CHKERRQ(ierr);

    /* Create the matrix Z=[A;B] */
    ierr = MatDestroy(&lanczos->Z);CHKERRQ(ierr);
    ierr = MatGetLocalSize(svd->A,&m,&n);CHKERRQ(ierr);
    ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);
    if (lanczos->explicitmatrix) {
      mats[0] = svd->A;
      mats[1] = svd->B;
      ierr = MatCreateNest(PetscObjectComm((PetscObject)svd),2,NULL,1,NULL,mats,&lanczos->Z);CHKERRQ(ierr);
      ierr = MatGetType(svd->A,&Atype);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)svd->B,Atype,&sametype);CHKERRQ(ierr);
      if (!sametype) Atype = MATAIJ;
      ierr = MatConvert(lanczos->Z,Atype,MAT_INPLACE_MATRIX,&lanczos->Z);CHKERRQ(ierr);
    } else {
      ierr = MatZCreateContext(svd,&zdata);CHKERRQ(ierr);
      ierr = MatCreateShell(PetscObjectComm((PetscObject)svd),m+p,n,PETSC_DECIDE,PETSC_DECIDE,zdata,&lanczos->Z);CHKERRQ(ierr);
      ierr = MatShellSetOperation(lanczos->Z,MATOP_MULT,(void(*)(void))MatMult_Z);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = MatShellSetOperation(lanczos->Z,MATOP_MULT_HERMITIAN_TRANSPOSE,(void(*)(void))MatMultTranspose_Z);CHKERRQ(ierr);
#else
      ierr = MatShellSetOperation(lanczos->Z,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Z);CHKERRQ(ierr);
#endif
      ierr = MatShellSetOperation(lanczos->Z,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_Z);CHKERRQ(ierr);
      ierr = MatShellSetOperation(lanczos->Z,MATOP_DESTROY,(void(*)(void))MatDestroy_Z);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)lanczos->Z);CHKERRQ(ierr);

    /* create normal equations matrix, to build the preconditioner in LSQR */
    ierr = MatCreateNormalHermitian(lanczos->Z,&normal);CHKERRQ(ierr);

    if (!lanczos->ksp) { ierr = SVDTRLanczosGetKSP(svd,&lanczos->ksp);CHKERRQ(ierr); }
    ierr = KSPSetOperators(lanczos->ksp,lanczos->Z,normal);CHKERRQ(ierr);
    ierr = KSPSetUp(lanczos->ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&normal);CHKERRQ(ierr);

    if (lanczos->oneside) { ierr = PetscInfo(svd,"Warning: one-side option is ignored in GSVD\n");CHKERRQ(ierr); }
  }
  ierr = DSSetType(svd->ds,dstype);CHKERRQ(ierr);
  ierr = DSSetCompact(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSSetExtraRow(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDOneSideTRLanczosMGS(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscReal      a,b;
  PetscInt       i,k=nconv+l;
  Vec            ui,ui1,vi;

  PetscFunctionBegin;
  ierr = BVGetColumn(V,k,&vi);CHKERRQ(ierr);
  ierr = BVGetColumn(U,k,&ui);CHKERRQ(ierr);
  ierr = MatMult(svd->A,vi,ui);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,k,&vi);CHKERRQ(ierr);
  ierr = BVRestoreColumn(U,k,&ui);CHKERRQ(ierr);
  if (l>0) {
    ierr = BVSetActiveColumns(U,nconv,n);CHKERRQ(ierr);
    for (i=0;i<l;i++) work[i]=beta[i+nconv];
    ierr = BVMultColumn(U,-1.0,1.0,k,work);CHKERRQ(ierr);
  }
  ierr = BVNormColumn(U,k,NORM_2,&a);CHKERRQ(ierr);
  ierr = BVScaleColumn(U,k,1.0/a);CHKERRQ(ierr);
  alpha[k] = a;

  for (i=k+1;i<n;i++) {
    ierr = BVGetColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = MatMult(svd->AT,ui1,vi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(V,i,PETSC_FALSE,&b,NULL);CHKERRQ(ierr);
    beta[i-1] = b;

    ierr = BVGetColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i,&ui);CHKERRQ(ierr);
    ierr = MatMult(svd->A,vi,ui);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = VecAXPY(ui,-b,ui1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,i,&ui);CHKERRQ(ierr);
    ierr = BVNormColumn(U,i,NORM_2,&a);CHKERRQ(ierr);
    ierr = BVScaleColumn(U,i,1.0/a);CHKERRQ(ierr);
    alpha[i] = a;
  }

  ierr = BVGetColumn(V,n,&vi);CHKERRQ(ierr);
  ierr = BVGetColumn(U,n-1,&ui1);CHKERRQ(ierr);
  ierr = MatMult(svd->AT,ui1,vi);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,n,&vi);CHKERRQ(ierr);
  ierr = BVRestoreColumn(U,n-1,&ui1);CHKERRQ(ierr);
  ierr = BVOrthogonalizeColumn(V,n,NULL,&b,NULL);CHKERRQ(ierr);
  beta[n-1] = b;
  PetscFunctionReturn(0);
}

/*
  Custom CGS orthogonalization, preprocess after first orthogonalization
*/
static PetscErrorCode SVDOrthogonalizeCGS(BV V,PetscInt i,PetscScalar* h,PetscReal a,BVOrthogRefineType refine,PetscReal eta,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscReal      sum,onorm;
  PetscScalar    dot;
  PetscInt       j;

  PetscFunctionBegin;
  switch (refine) {
  case BV_ORTHOG_REFINE_NEVER:
    ierr = BVNormColumn(V,i,NORM_2,norm);CHKERRQ(ierr);
    break;
  case BV_ORTHOG_REFINE_ALWAYS:
    ierr = BVSetActiveColumns(V,0,i);CHKERRQ(ierr);
    ierr = BVDotColumn(V,i,h);CHKERRQ(ierr);
    ierr = BVMultColumn(V,-1.0,1.0,i,h);CHKERRQ(ierr);
    ierr = BVNormColumn(V,i,NORM_2,norm);CHKERRQ(ierr);
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
    else {
      ierr = BVNormColumn(V,i,NORM_2,norm);CHKERRQ(ierr);
    }
    if (*norm < eta*onorm) {
      ierr = BVSetActiveColumns(V,0,i);CHKERRQ(ierr);
      ierr = BVDotColumn(V,i,h);CHKERRQ(ierr);
      ierr = BVMultColumn(V,-1.0,1.0,i,h);CHKERRQ(ierr);
      ierr = BVNormColumn(V,i,NORM_2,norm);CHKERRQ(ierr);
    }
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDOneSideTRLanczosCGS(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscErrorCode     ierr;
  PetscReal          a,b,eta;
  PetscInt           i,j,k=nconv+l;
  Vec                ui,ui1,vi;
  BVOrthogRefineType refine;

  PetscFunctionBegin;
  ierr = BVGetColumn(V,k,&vi);CHKERRQ(ierr);
  ierr = BVGetColumn(U,k,&ui);CHKERRQ(ierr);
  ierr = MatMult(svd->A,vi,ui);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,k,&vi);CHKERRQ(ierr);
  ierr = BVRestoreColumn(U,k,&ui);CHKERRQ(ierr);
  if (l>0) {
    ierr = BVSetActiveColumns(U,nconv,n);CHKERRQ(ierr);
    for (i=0;i<l;i++) work[i]=beta[i+nconv];
    ierr = BVMultColumn(U,-1.0,1.0,k,work);CHKERRQ(ierr);
  }
  ierr = BVGetOrthogonalization(V,NULL,&refine,&eta,NULL);CHKERRQ(ierr);

  for (i=k+1;i<n;i++) {
    ierr = BVGetColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = MatMult(svd->AT,ui1,vi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = BVNormColumnBegin(U,i-1,NORM_2,&a);CHKERRQ(ierr);
    if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
      ierr = BVSetActiveColumns(V,0,i+1);CHKERRQ(ierr);
      ierr = BVGetColumn(V,i,&vi);CHKERRQ(ierr);
      ierr = BVDotVecBegin(V,vi,work);CHKERRQ(ierr);
    } else {
      ierr = BVSetActiveColumns(V,0,i);CHKERRQ(ierr);
      ierr = BVDotColumnBegin(V,i,work);CHKERRQ(ierr);
    }
    ierr = BVNormColumnEnd(U,i-1,NORM_2,&a);CHKERRQ(ierr);
    if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
      ierr = BVDotVecEnd(V,vi,work);CHKERRQ(ierr);
      ierr = BVRestoreColumn(V,i,&vi);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(V,0,i);CHKERRQ(ierr);
    } else {
      ierr = BVDotColumnEnd(V,i,work);CHKERRQ(ierr);
    }

    ierr = BVScaleColumn(U,i-1,1.0/a);CHKERRQ(ierr);
    for (j=0;j<i;j++) work[j] = work[j] / a;
    ierr = BVMultColumn(V,-1.0,1.0/a,i,work);CHKERRQ(ierr);
    ierr = SVDOrthogonalizeCGS(V,i,work,a,refine,eta,&b);CHKERRQ(ierr);
    ierr = BVScaleColumn(V,i,1.0/b);CHKERRQ(ierr);
    if (PetscAbsReal(b)<10*PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_PLIB,"Recurrence generated a zero vector; use a two-sided variant");

    ierr = BVGetColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i,&ui);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i-1,&ui1);CHKERRQ(ierr);
    ierr = MatMult(svd->A,vi,ui);CHKERRQ(ierr);
    ierr = VecAXPY(ui,-b,ui1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&vi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,i,&ui);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U,i-1,&ui1);CHKERRQ(ierr);

    alpha[i-1] = a;
    beta[i-1] = b;
  }

  ierr = BVGetColumn(V,n,&vi);CHKERRQ(ierr);
  ierr = BVGetColumn(U,n-1,&ui1);CHKERRQ(ierr);
  ierr = MatMult(svd->AT,ui1,vi);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,n,&vi);CHKERRQ(ierr);
  ierr = BVRestoreColumn(U,n-1,&ui1);CHKERRQ(ierr);

  ierr = BVNormColumnBegin(svd->U,n-1,NORM_2,&a);CHKERRQ(ierr);
  if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
    ierr = BVSetActiveColumns(V,0,n+1);CHKERRQ(ierr);
    ierr = BVGetColumn(V,n,&vi);CHKERRQ(ierr);
    ierr = BVDotVecBegin(V,vi,work);CHKERRQ(ierr);
  } else {
    ierr = BVSetActiveColumns(V,0,n);CHKERRQ(ierr);
    ierr = BVDotColumnBegin(V,n,work);CHKERRQ(ierr);
  }
  ierr = BVNormColumnEnd(svd->U,n-1,NORM_2,&a);CHKERRQ(ierr);
  if (refine == BV_ORTHOG_REFINE_IFNEEDED) {
    ierr = BVDotVecEnd(V,vi,work);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,n,&vi);CHKERRQ(ierr);
  } else {
    ierr = BVDotColumnEnd(V,n,work);CHKERRQ(ierr);
  }

  ierr = BVScaleColumn(U,n-1,1.0/a);CHKERRQ(ierr);
  for (j=0;j<n;j++) work[j] = work[j] / a;
  ierr = BVMultColumn(V,-1.0,1.0/a,n,work);CHKERRQ(ierr);
  ierr = SVDOrthogonalizeCGS(V,n,work,a,refine,eta,&b);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(V,nconv,n);CHKERRQ(ierr);
  alpha[n-1] = a;
  beta[n-1] = b;
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta;
  PetscScalar    *swork=NULL,*w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,V;
  PetscBool      breakdown=PETSC_FALSE;
  BVOrthogType   orthog;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  /* allocate working space */
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = BVGetOrthogonalization(svd->V,&orthog,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&w);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = PetscMalloc1(svd->ncv+1,&swork);CHKERRQ(ierr);
  }

  /* normalize start vector */
  if (!svd->nini) {
    ierr = BVSetRandomColumn(svd->V,0);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(svd->V,0,PETSC_TRUE,NULL,NULL);CHKERRQ(ierr);
  }

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    if (lanczos->oneside) {
      if (orthog == BV_ORTHOG_MGS) {
        ierr = SVDOneSideTRLanczosMGS(svd,alpha,beta,svd->V,svd->U,svd->nconv,l,nv,swork);CHKERRQ(ierr);
      } else {
        ierr = SVDOneSideTRLanczosCGS(svd,alpha,beta,svd->V,svd->U,svd->nconv,l,nv,swork);CHKERRQ(ierr);
      }
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,svd->U,svd->nconv+l,&nv,&breakdown);CHKERRQ(ierr);
    }
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = BVScaleColumn(svd->V,nv,1.0/beta[nv-1]);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->V,svd->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->U,svd->nconv,nv);CHKERRQ(ierr);

    /* solve projected problem */
    ierr = DSSetDimensions(svd->ds,nv,svd->nconv,svd->nconv+l);CHKERRQ(ierr);
    ierr = DSSVDSetDimensions(svd->ds,nv);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(svd->ds);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    ierr = SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,&k);CHKERRQ(ierr);
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) { ierr = PetscInfo1(svd,"Preparing to restart keeping l=%D vectors\n",l);CHKERRQ(ierr); }

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (breakdown || k==nv) {
        /* Start a new bidiagonalization */
        ierr = PetscInfo1(svd,"Breakdown in bidiagonalization (it=%D)\n",svd->its);CHKERRQ(ierr);
        if (k<svd->nsv) {
          ierr = BVSetRandomColumn(svd->V,k);CHKERRQ(ierr);
          ierr = BVOrthonormalizeColumn(svd->V,k,PETSC_FALSE,NULL,&breakdown);CHKERRQ(ierr);
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(svd,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          }
        }
      } else {
        ierr = DSTruncate(svd->ds,k+l,PETSC_FALSE);CHKERRQ(ierr);
      }
    }

    /* compute converged singular vectors and restart vectors */
    ierr = DSGetMat(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->V,V,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&V);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->U,U,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(svd->V,nv,k+l);CHKERRQ(ierr);
    }

    svd->nconv = k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }

  /* orthonormalize U columns in one side method */
  if (lanczos->oneside) {
    for (i=0;i<svd->nconv;i++) {
      ierr = BVOrthonormalizeColumn(svd->U,i,PETSC_FALSE,NULL,NULL);CHKERRQ(ierr);
    }
  }

  /* free working space */
  ierr = PetscFree(w);CHKERRQ(ierr);
  if (swork) { ierr = PetscFree(swork);CHKERRQ(ierr); }
  ierr = DSTruncate(svd->ds,svd->nconv,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTwoSideLanczosGSingle(SVD svd,PetscReal *alpha,PetscReal *beta,Mat Z,BV V,BV U,KSP ksp,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,m;
  const PetscScalar *carr;
  PetscScalar       *arr;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0],v1;
  PetscBool         lindep=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatCreateVecsEmpty(svd->A,NULL,&v1);CHKERRQ(ierr);
  ierr = BVGetColumn(V,k,&v);CHKERRQ(ierr);
  ierr = BVGetColumn(U,k,&u);CHKERRQ(ierr);

  /* Form ut=[u;0] */
  ierr = VecZeroEntries(ut);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&m);CHKERRQ(ierr);
  ierr = VecGetArrayRead(u,&carr);CHKERRQ(ierr);
  ierr = VecGetArray(ut,&arr);CHKERRQ(ierr);
  for (j=0; j<m; j++) arr[j] = carr[j];
  ierr = VecRestoreArrayRead(u,&carr);CHKERRQ(ierr);
  ierr = VecRestoreArray(ut,&arr);CHKERRQ(ierr);

  /* Solve least squares problem */
  ierr = KSPSolve(ksp,ut,x);CHKERRQ(ierr);

  ierr = MatMult(Z,x,v);CHKERRQ(ierr);

  ierr = BVRestoreColumn(U,k,&u);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,k,&v);CHKERRQ(ierr);
  ierr = BVOrthonormalizeColumn(V,k,PETSC_FALSE,alpha+k,&lindep);CHKERRQ(ierr);
  if (lindep) {
    *n = k;
    if (breakdown) *breakdown = lindep;
    PetscFunctionReturn(0);
  }

  for (i=k+1; i<*n; i++) {

    /* Compute vector i of BV U */
    ierr = BVGetColumn(V,i-1,&v);CHKERRQ(ierr);
    ierr = VecGetArray(v,&arr);CHKERRQ(ierr);
    ierr = VecPlaceArray(v1,arr);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&arr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i-1,&v);CHKERRQ(ierr);
    ierr = BVInsertVec(U,i,v1);CHKERRQ(ierr);
    ierr = VecResetArray(v1);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U,i,PETSC_FALSE,beta+i-1,&lindep);CHKERRQ(ierr);
    if (lindep) {
      *n = i;
      break;
    }

    /* Compute vector i of BV V */

    ierr = BVGetColumn(V,i,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(U,i,&u);CHKERRQ(ierr);

    /* Form ut=[u;0] */
    ierr = VecGetArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecGetArray(ut,&arr);CHKERRQ(ierr);
    for (j=0; j<m; j++) arr[j] = carr[j];
    ierr = VecRestoreArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecRestoreArray(ut,&arr);CHKERRQ(ierr);

    /* Solve least squares problem */
    ierr = KSPSolve(ksp,ut,x);CHKERRQ(ierr);

    ierr = MatMult(Z,x,v);CHKERRQ(ierr);

    ierr = BVRestoreColumn(U,i,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&v);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(V,i,PETSC_FALSE,alpha+i,&lindep);CHKERRQ(ierr);
    if (lindep) {
      *n = i;
      break;
    }
  }

  /* Compute vector n of BV U */
  if (!lindep) {
    ierr = BVGetColumn(V,*n-1,&v);CHKERRQ(ierr);
    ierr = VecGetArray(v,&arr);CHKERRQ(ierr);
    ierr = VecPlaceArray(v1,arr);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&arr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,*n-1,&v);CHKERRQ(ierr);
    ierr = BVInsertVec(U,*n,v1);CHKERRQ(ierr);
    ierr = VecResetArray(v1);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U,*n,PETSC_FALSE,beta+*n-1,&lindep);CHKERRQ(ierr);
  }
  if (breakdown) *breakdown = lindep;
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* solve generalized problem with single bidiagonalization of Q_A */
PetscErrorCode SVDSolve_TRLanczosGSingle(SVD svd,BV U1,BV V)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta;
  PetscScalar    *w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,VV;
  PetscBool      breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&w);CHKERRQ(ierr);

  /* normalize start vector */
  if (!svd->nini) {
    ierr = BVSetRandomColumn(U1,0);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U1,0,PETSC_TRUE,NULL,NULL);CHKERRQ(ierr);
  }

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    ierr = SVDTwoSideLanczosGSingle(svd,alpha,beta,lanczos->Z,V,U1,lanczos->ksp,svd->nconv+l,&nv,&breakdown);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(V,svd->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(U1,svd->nconv,nv);CHKERRQ(ierr);

    /* solve projected problem */
    ierr = DSSetDimensions(svd->ds,nv,svd->nconv,svd->nconv+l);CHKERRQ(ierr);
    ierr = DSSVDSetDimensions(svd->ds,nv);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(svd->ds);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    ierr = SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,&k);CHKERRQ(ierr);
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) { ierr = PetscInfo1(svd,"Preparing to restart keeping l=%D vectors\n",l);CHKERRQ(ierr); }

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (breakdown || k==nv) {
        /* Start a new bidiagonalization */
        ierr = PetscInfo1(svd,"Breakdown in bidiagonalization (it=%D)\n",svd->its);CHKERRQ(ierr);
        if (k<svd->nsv) {
          ierr = BVSetRandomColumn(U1,k);CHKERRQ(ierr);
          ierr = BVOrthonormalizeColumn(U1,k,PETSC_FALSE,NULL,&breakdown);CHKERRQ(ierr);
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(svd,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          }
        }
      } else {
        ierr = DSTruncate(svd->ds,k+l,PETSC_FALSE);CHKERRQ(ierr);
      }
    }

    /* compute converged singular vectors and restart vectors */
    ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(V,U,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_V,&VV);CHKERRQ(ierr);
    ierr = BVMultInPlace(U1,VV,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&VV);CHKERRQ(ierr);

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(U1,nv,k+l);CHKERRQ(ierr);
    }

    svd->nconv = k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }

  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Move generalized left singular vectors (0..nconv) from U1 and U2 to its final destination svd->U (single variant) */
PETSC_STATIC_INLINE PetscErrorCode SVDLeftSingularVectors_Single(SVD svd,BV U1,BV U2)
{
  PetscErrorCode    ierr;
  PetscInt          i,k,m,p;
  Vec               u,u1,u2;
  PetscScalar       *ua,*u2a;
  const PetscScalar *u1a;
  PetscReal         s;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = BVGetColumn(U1,i,&u1);CHKERRQ(ierr);
    ierr = BVGetColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i,&u);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u1,&u1a);CHKERRQ(ierr);
    ierr = VecGetArray(u,&ua);CHKERRQ(ierr);
    ierr = VecGetArray(u2,&u2a);CHKERRQ(ierr);
    /* Copy column from U1 to upper part of u */
    for (k=0;k<m;k++) ua[k] = u1a[k];
    /* Copy column from lower part of U to U2. Orthogonalize column in U2 and copy back to U */
    for (k=0;k<p;k++) u2a[k] = ua[m+k];
    ierr = VecRestoreArray(u2,&u2a);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U2,i,PETSC_FALSE,&s,NULL);CHKERRQ(ierr);
    ierr = BVGetColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = VecGetArray(u2,&u2a);CHKERRQ(ierr);
    for (k=0;k<p;k++) ua[m+k] = u2a[k];
    /* Update singular value */
    svd->sigma[i] /= s;
    ierr = VecRestoreArrayRead(u1,&u1a);CHKERRQ(ierr);
    ierr = VecRestoreArray(u,&ua);CHKERRQ(ierr);
    ierr = VecRestoreArray(u2,&u2a);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U1,i,&u1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i,&u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTwoSideLanczosGUpper(SVD svd,PetscReal *alpha,PetscReal *beta,PetscReal *alphah,PetscReal *betah,Mat Z,BV U1,BV U2,BV V,KSP ksp,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,m,p;
  const PetscScalar *carr;
  PetscScalar       *arr,*u2arr;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0],v1,u2;
  PetscBool         lindep=PETSC_FALSE,lindep1=PETSC_FALSE,lindep2=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatCreateVecsEmpty(svd->A,NULL,&v1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);

  for (i=k; i<*n; i++) {
    /* Compute vector i of BV U1 */
    ierr = BVGetColumn(V,i,&v);CHKERRQ(ierr);
    ierr = VecGetArrayRead(v,&carr);CHKERRQ(ierr);
    ierr = VecPlaceArray(v1,carr);CHKERRQ(ierr);
    ierr = BVInsertVec(U1,i,v1);CHKERRQ(ierr);
    ierr = VecResetArray(v1);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U1,i,PETSC_FALSE,alpha+i,&lindep1);CHKERRQ(ierr);

    /* Compute vector i of BV U2 */
    ierr = BVGetColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = VecGetArray(u2,&u2arr);CHKERRQ(ierr);
    if (i%2) {
      for (j=0; j<p; j++) u2arr[j] = -carr[m+j];
    } else {
      for (j=0; j<p; j++) u2arr[j] = carr[m+j];
    }
    ierr = VecRestoreArray(u2,&u2arr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v,&carr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&v);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U2,i,PETSC_FALSE,alphah+i,&lindep2);CHKERRQ(ierr);
    if (i%2) alphah[i] = -alphah[i];
    if (lindep1 || lindep2) {
      lindep = PETSC_TRUE;
      *n = i;
      break;
    }

    /* Compute vector i+1 of BV V */
    ierr = BVGetColumn(V,i+1,&v);CHKERRQ(ierr);
    /* Form ut=[u;0] */
    ierr = BVGetColumn(U1,i,&u);CHKERRQ(ierr);
    ierr = VecZeroEntries(ut);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecGetArray(ut,&arr);CHKERRQ(ierr);
    for (j=0; j<m; j++) arr[j] = carr[j];
    ierr = VecRestoreArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecRestoreArray(ut,&arr);CHKERRQ(ierr);
    /* Solve least squares problem */
    ierr = KSPSolve(ksp,ut,x);CHKERRQ(ierr);
    ierr = MatMult(Z,x,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U1,i,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i+1,&v);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(V,i+1,PETSC_FALSE,beta+i,&lindep);CHKERRQ(ierr);
    betah[i] = -alpha[i]*beta[i]/alphah[i];
    if (lindep) {
      *n = i;
      break;
    }
  }
  if (breakdown) *breakdown = lindep;
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* generate random initial vector in column k for joint upper-upper bidiagonalization */
PETSC_STATIC_INLINE PetscErrorCode SVDInitialVectorGUpper(SVD svd,BV V,BV U1,PetscInt k,PetscBool *breakdown)
{
  PetscErrorCode    ierr;
  SVD_TRLANCZOS     *lanczos = (SVD_TRLANCZOS*)svd->data;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0];
  PetscInt          m,j;
  PetscRandom       rand;
  PetscScalar       *arr;
  const PetscScalar *carr;

  PetscFunctionBegin;
  ierr = BVCreateVec(U1,&u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&m);CHKERRQ(ierr);
  ierr = BVGetRandomContext(U1,&rand);CHKERRQ(ierr);
  ierr = VecSetRandom(u,rand);CHKERRQ(ierr);
  /* Form ut=[u;0] */
  ierr = VecZeroEntries(ut);CHKERRQ(ierr);
  ierr = VecGetArrayRead(u,&carr);CHKERRQ(ierr);
  ierr = VecGetArray(ut,&arr);CHKERRQ(ierr);
  for (j=0; j<m; j++) arr[j] = carr[j];
  ierr = VecRestoreArrayRead(u,&carr);CHKERRQ(ierr);
  ierr = VecRestoreArray(ut,&arr);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  /* Solve least squares problem and premultiply the result by Z */
  ierr = KSPSolve(lanczos->ksp,ut,x);CHKERRQ(ierr);
  ierr = BVGetColumn(V,k,&v);CHKERRQ(ierr);
  ierr = MatMult(lanczos->Z,x,v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,k,&v);CHKERRQ(ierr);
  if (breakdown) { ierr = BVOrthonormalizeColumn(V,k,PETSC_FALSE,NULL,breakdown);CHKERRQ(ierr); }
  else { ierr = BVOrthonormalizeColumn(V,k,PETSC_TRUE,NULL,NULL);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* solve generalized problem with joint upper-upper bidiagonalization */
PetscErrorCode SVDSolve_TRLanczosGUpper(SVD svd,BV U1,BV U2,BV V)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta,*alphah,*betah;
  PetscScalar    *w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,Vmat,X;
  PetscBool      breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&w);CHKERRQ(ierr);

  /* normalize start vector */
  if (!svd->nini) {
    ierr = SVDInitialVectorGUpper(svd,V,U1,0,NULL);CHKERRQ(ierr);
  }

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_D,&alphah);CHKERRQ(ierr);
    beta = alpha + ld;
    betah = alpha + 2*ld;
    ierr = SVDTwoSideLanczosGUpper(svd,alpha,beta,alphah,betah,lanczos->Z,U1,U2,V,lanczos->ksp,svd->nconv+l,&nv,&breakdown);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_D,&alphah);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(V,svd->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(U1,svd->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(U2,svd->nconv,nv);CHKERRQ(ierr);

    /* solve projected problem */
    ierr = DSSetDimensions(svd->ds,nv,svd->nconv,svd->nconv+l);CHKERRQ(ierr);
    ierr = DSGSVDSetDimensions(svd->ds,nv,nv);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(svd->ds);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    ierr = SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,&k);CHKERRQ(ierr);
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) { ierr = PetscInfo1(svd,"Preparing to restart keeping l=%D vectors\n",l);CHKERRQ(ierr); }

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (breakdown || k==nv) {
        /* Start a new bidiagonalization */
        ierr = PetscInfo1(svd,"Breakdown in bidiagonalization (it=%D)\n",svd->its);CHKERRQ(ierr);
        if (k<svd->nsv) {
          ierr = SVDInitialVectorGUpper(svd,V,U1,k,&breakdown);CHKERRQ(ierr);
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(svd,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          }
        }
      } else {
        ierr = DSTruncate(svd->ds,k+l,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
    /* compute converged singular vectors and restart vectors */
    ierr = DSGetMat(svd->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = BVMultInPlace(V,X,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(U1,U,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_V,&Vmat);CHKERRQ(ierr);
    ierr = BVMultInPlace(U2,Vmat,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&Vmat);CHKERRQ(ierr);

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(V,nv,k+l);CHKERRQ(ierr);
    }

    svd->nconv = k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }

  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Move generalized left singular vectors (0..nconv) from U1 and U2 to its final destination svd->U (upper and lower variants) */
PETSC_STATIC_INLINE PetscErrorCode SVDLeftSingularVectors(SVD svd,BV U1,BV U2)
{
  PetscErrorCode    ierr;
  PetscInt          i,k,m,p;
  Vec               u,u1,u2;
  PetscScalar       *ua;
  const PetscScalar *u1a,*u2a;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = BVGetColumn(U1,i,&u1);CHKERRQ(ierr);
    ierr = BVGetColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i,&u);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u1,&u1a);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u2,&u2a);CHKERRQ(ierr);
    ierr = VecGetArray(u,&ua);CHKERRQ(ierr);
    /* Copy column from u1 to upper part of u */
    for (k=0;k<m;k++) ua[k] = u1a[k];
    /* Copy column from u2 to lower part of u */
    for (k=0;k<p;k++) ua[m+k] = u2a[k];
    ierr = VecRestoreArrayRead(u1,&u1a);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u2,&u2a);CHKERRQ(ierr);
    ierr = VecRestoreArray(u,&ua);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U1,i,&u1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i,&u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTwoSideLanczosGLower(SVD svd,PetscReal *alpha,PetscReal *beta,PetscReal *alphah,PetscReal *betah,Mat Z,BV U1,BV U2,BV V,KSP ksp,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,m,p;
  const PetscScalar *carr;
  PetscScalar       *arr,*u2arr;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0],v1,u2;
  PetscBool         lindep=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatCreateVecsEmpty(svd->A,NULL,&v1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);

  for (i=k; i<*n; i++) {
    /* Compute vector i of BV U2 */
    ierr = BVGetColumn(V,i,&v);CHKERRQ(ierr);
    ierr = VecGetArrayRead(v,&carr);CHKERRQ(ierr);
    ierr = BVGetColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = VecGetArray(u2,&u2arr);CHKERRQ(ierr);
    if (i%2) {
      for (j=0; j<p; j++) u2arr[j] = -carr[m+j];
    } else {
      for (j=0; j<p; j++) u2arr[j] = carr[m+j];
    }
    ierr = VecRestoreArray(u2,&u2arr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U2,i,&u2);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U2,i,PETSC_FALSE,alphah+i,&lindep);CHKERRQ(ierr);
    if (i%2) alphah[i] = -alphah[i];
    if (lindep) {
      ierr = BVRestoreColumn(V,i,&v);CHKERRQ(ierr);
      *n = i;
      break;
    }

    /* Compute vector i+1 of BV U1 */
    ierr = VecPlaceArray(v1,carr);CHKERRQ(ierr);
    ierr = BVInsertVec(U1,i+1,v1);CHKERRQ(ierr);
    ierr = VecResetArray(v1);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(U1,i+1,PETSC_FALSE,beta+i,&lindep);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v,&carr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&v);CHKERRQ(ierr);
    if (lindep) {
      *n = i+1;
      break;
    }

    /* Compute vector i+1 of BV V */
    ierr = BVGetColumn(V,i+1,&v);CHKERRQ(ierr);
    /* Form ut=[u;0] where u is column i+1 of BV U1 */
    ierr = BVGetColumn(U1,i+1,&u);CHKERRQ(ierr);
    ierr = VecZeroEntries(ut);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecGetArray(ut,&arr);CHKERRQ(ierr);
    for (j=0; j<m; j++) arr[j] = carr[j];
    ierr = VecRestoreArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecRestoreArray(ut,&arr);CHKERRQ(ierr);
    /* Solve least squares problem */
    ierr = KSPSolve(ksp,ut,x);CHKERRQ(ierr);
    ierr = MatMult(Z,x,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U1,i+1,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i+1,&v);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(V,i+1,PETSC_FALSE,alpha+i+1,&lindep);CHKERRQ(ierr);
    betah[i] = -alpha[i+1]*beta[i]/alphah[i];
    if (lindep) {
      *n = i+1;
      break;
    }
  }
  if (breakdown) *breakdown = lindep;
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* generate random initial vector in column k for joint lower-upper bidiagonalization */
PETSC_STATIC_INLINE PetscErrorCode SVDInitialVectorGLower(SVD svd,BV V,BV U1,PetscInt k,PetscBool *breakdown)
{
  PetscErrorCode    ierr;
  SVD_TRLANCZOS     *lanczos = (SVD_TRLANCZOS*)svd->data;
  const PetscScalar *carr;
  PetscScalar       *arr;
  PetscReal         *alpha;
  PetscInt          j,m;
  Vec               u,v,ut=svd->workl[0],x=svd->workr[0];

  PetscFunctionBegin;
  ierr = BVSetRandomColumn(U1,k);CHKERRQ(ierr);
  if (breakdown) { ierr = BVOrthonormalizeColumn(U1,k,PETSC_FALSE,NULL,breakdown);CHKERRQ(ierr); }
  else { ierr = BVOrthonormalizeColumn(U1,k,PETSC_TRUE,NULL,NULL);CHKERRQ(ierr); }

  if (!breakdown || !*breakdown) {
    ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
    /* Compute k-th vector of BV V */
    ierr = BVGetColumn(V,k,&v);CHKERRQ(ierr);
    /* Form ut=[u;0] where u is the 1st column of U1 */
    ierr = BVGetColumn(U1,k,&u);CHKERRQ(ierr);
    ierr = VecZeroEntries(ut);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecGetArray(ut,&arr);CHKERRQ(ierr);
    for (j=0; j<m; j++) arr[j] = carr[j];
    ierr = VecRestoreArrayRead(u,&carr);CHKERRQ(ierr);
    ierr = VecRestoreArray(ut,&arr);CHKERRQ(ierr);
    /* Solve least squares problem */
    ierr = KSPSolve(lanczos->ksp,ut,x);CHKERRQ(ierr);
    ierr = MatMult(lanczos->Z,x,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(U1,k,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,k,&v);CHKERRQ(ierr);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    if (breakdown) { ierr = BVOrthonormalizeColumn(V,k,PETSC_FALSE,alpha+k,breakdown);CHKERRQ(ierr); }
    else { ierr = BVOrthonormalizeColumn(V,k,PETSC_TRUE,alpha+k,NULL);CHKERRQ(ierr); }
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* solve generalized problem with joint lower-upper bidiagonalization */
PetscErrorCode SVDSolve_TRLanczosGLower(SVD svd,BV U1,BV U2,BV V)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta,*alphah,*betah;
  PetscScalar    *w;
  PetscInt       i,k,l,nv,ld;
  Mat            U,Vmat,X;
  PetscBool      breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&w);CHKERRQ(ierr);

  /* normalize start vector */
  if (!svd->nini) {
    ierr = SVDInitialVectorGLower(svd,V,U1,0,NULL);CHKERRQ(ierr);
  }

  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_D,&alphah);CHKERRQ(ierr);
    beta = alpha + ld;
    betah = alpha + 2*ld;
    ierr = SVDTwoSideLanczosGLower(svd,alpha,beta,alphah,betah,lanczos->Z,U1,U2,V,lanczos->ksp,svd->nconv+l,&nv,&breakdown);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_D,&alphah);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(V,svd->nconv,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(U1,svd->nconv,nv+1);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(U2,svd->nconv,nv);CHKERRQ(ierr);

    /* solve projected problem */
    ierr = DSSetDimensions(svd->ds,nv+1,svd->nconv,svd->nconv+l);CHKERRQ(ierr);
    ierr = DSGSVDSetDimensions(svd->ds,nv,nv);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(svd->ds);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    ierr = SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,&k);CHKERRQ(ierr);
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);

    /* update l */
    if (svd->reason != SVD_CONVERGED_ITERATING || breakdown || k==nv) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*lanczos->keep));
    if (!lanczos->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged triplets */
    if (l) { ierr = PetscInfo1(svd,"Preparing to restart keeping l=%D vectors\n",l);CHKERRQ(ierr); }

    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (breakdown || k==nv) {
        /* Start a new bidiagonalization */
        ierr = PetscInfo1(svd,"Breakdown in bidiagonalization (it=%D)\n",svd->its);CHKERRQ(ierr);
        if (k<svd->nsv) {
          ierr = SVDInitialVectorGLower(svd,V,U1,k,&breakdown);CHKERRQ(ierr);
          if (breakdown) {
            svd->reason = SVD_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(svd,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          }
        }
      } else {
        ierr = DSTruncate(svd->ds,k+l,PETSC_FALSE);CHKERRQ(ierr);
      }
    }

    /* compute converged singular vectors and restart vectors */
    ierr = DSGetMat(svd->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = BVMultInPlace(V,X,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(U1,U,svd->nconv,k+l+1);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_V,&Vmat);CHKERRQ(ierr);
    ierr = BVMultInPlace(U2,Vmat,svd->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&Vmat);CHKERRQ(ierr);

    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(V,nv,k+l);CHKERRQ(ierr);
    }

    svd->nconv = k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }

  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_TRLanczos_GSVD(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscInt       k,m,p;
  BV             U1,U2;
  BVType         type;
  Mat            U,V;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citationg,&citedg);CHKERRQ(ierr);

  ierr = MatGetLocalSize(svd->A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->B,&p,NULL);CHKERRQ(ierr);

  /* Create BV for U1 */
  ierr = BVCreate(PetscObjectComm((PetscObject)svd),&U1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)U1);CHKERRQ(ierr);
  ierr = BVGetType(svd->U,&type);CHKERRQ(ierr);
  ierr = BVSetType(U1,type);CHKERRQ(ierr);
  ierr = BVGetSizes(svd->U,NULL,NULL,&k);CHKERRQ(ierr);
  ierr = BVSetSizes(U1,m,PETSC_DECIDE,k);CHKERRQ(ierr);

  /* Create BV for U2 */
  ierr = BVCreate(PetscObjectComm((PetscObject)svd),&U2);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)U2);CHKERRQ(ierr);
  ierr = BVSetType(U2,type);CHKERRQ(ierr);
  ierr = BVSetSizes(U2,p,PETSC_DECIDE,k);CHKERRQ(ierr);

  switch (lanczos->bidiag) {
    case SVD_TRLANCZOS_GBIDIAG_SINGLE:
      ierr = SVDSolve_TRLanczosGSingle(svd,U1,svd->U);CHKERRQ(ierr);
      break;
    case SVD_TRLANCZOS_GBIDIAG_UPPER:
      ierr = SVDSolve_TRLanczosGUpper(svd,U1,U2,svd->U);CHKERRQ(ierr);
      break;
    case SVD_TRLANCZOS_GBIDIAG_LOWER:
      ierr = SVDSolve_TRLanczosGLower(svd,U1,U2,svd->U);CHKERRQ(ierr);
      break;
  }

  /* Compute converged right singular vectors */
  ierr = BVSetActiveColumns(svd->U,0,svd->nconv);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(svd->V,0,svd->nconv);CHKERRQ(ierr);
  ierr = BVGetMat(svd->U,&U);CHKERRQ(ierr);
  ierr = BVGetMat(svd->V,&V);CHKERRQ(ierr);
  ierr = KSPMatSolve(lanczos->ksp,U,V);CHKERRQ(ierr);
  ierr = BVRestoreMat(svd->U,&U);CHKERRQ(ierr);
  ierr = BVRestoreMat(svd->V,&V);CHKERRQ(ierr);

  /* Finish computing left singular vectors and move them to its place */
  switch (lanczos->bidiag) {
    case SVD_TRLANCZOS_GBIDIAG_SINGLE:
      ierr = SVDLeftSingularVectors_Single(svd,U1,U2);CHKERRQ(ierr);
      break;
    case SVD_TRLANCZOS_GBIDIAG_UPPER:
    case SVD_TRLANCZOS_GBIDIAG_LOWER:
      ierr = SVDLeftSingularVectors(svd,U1,U2);CHKERRQ(ierr);
      break;
  }

  ierr = BVDestroy(&U2);CHKERRQ(ierr);
  ierr = BVDestroy(&U1);CHKERRQ(ierr);
  ierr = DSTruncate(svd->ds,svd->nconv,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_TRLanczos(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode      ierr;
  SVD_TRLANCZOS       *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscBool           flg,val,lock;
  PetscReal           keep;
  SVDTRLanczosGBidiag bidiag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD TRLanczos Options");CHKERRQ(ierr);

    ierr = PetscOptionsBool("-svd_trlanczos_oneside","Use one-side reorthogonalization","SVDTRLanczosSetOneSide",lanczos->oneside,&val,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDTRLanczosSetOneSide(svd,val);CHKERRQ(ierr); }

    ierr = PetscOptionsReal("-svd_trlanczos_restart","Proportion of vectors kept after restart","SVDTRLanczosSetRestart",0.5,&keep,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDTRLanczosSetRestart(svd,keep);CHKERRQ(ierr); }

    ierr = PetscOptionsBool("-svd_trlanczos_locking","Choose between locking and non-locking variants","SVDTRLanczosSetLocking",PETSC_TRUE,&lock,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDTRLanczosSetLocking(svd,lock);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-svd_trlanczos_gbidiag","Bidiagonalization choice for Generalized Problem","SVDTRLanczosSetGBidiag",SVDTRLanczosGBidiags,(PetscEnum)lanczos->bidiag,(PetscEnum*)&bidiag,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDTRLanczosSetGBidiag(svd,bidiag);CHKERRQ(ierr); }

    ierr = PetscOptionsBool("-svd_trlanczos_explicitmatrix","Build explicit matrix for KSP solver","SVDTRLanczosSetExplicitMatrix",lanczos->explicitmatrix,&val,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDTRLanczosSetExplicitMatrix(svd,val);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (svd->OPb) {
    if (!lanczos->ksp) { ierr = SVDTRLanczosGetKSP(svd,&lanczos->ksp);CHKERRQ(ierr); }
    ierr = KSPSetFromOptions(lanczos->ksp);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,oneside,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetOneSide_C",(SVD,PetscBool),(svd,oneside));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(oneside,2);
  ierr = PetscUseMethod(svd,"SVDTRLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svd,bidiag,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetGBidiag_C",(SVD,SVDTRLanczosGBidiag),(svd,bidiag));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(bidiag,2);
  ierr = PetscUseMethod(svd,"SVDTRLanczosGetGBidiag_C",(SVD,SVDTRLanczosGBidiag*),(svd,bidiag));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetKSP_TRLanczos(SVD svd,KSP ksp)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ctx->ksp = ksp;
  ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->ksp);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(svd,1,ksp,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetKSP_C",(SVD,KSP),(svd,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetKSP_TRLanczos(SVD svd,KSP *ksp)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *ctx = (SVD_TRLANCZOS*)svd->data;
  PC             pc;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    /* Create linear solver */
    ierr = KSPCreate(PetscObjectComm((PetscObject)svd),&ctx->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)svd,1);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ctx->ksp,((PetscObject)svd)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(ctx->ksp,"svd_trlanczos_");CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ctx->ksp);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)ctx->ksp,((PetscObject)svd)->options);CHKERRQ(ierr);
    ierr = KSPSetType(ctx->ksp,KSPLSQR);CHKERRQ(ierr);
    ierr = KSPGetPC(ctx->ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(ctx->ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ctx->ksp,SlepcDefaultTol(svd->tol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(svd,"SVDTRLanczosGetKSP_C",(SVD,KSP*),(svd,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetRestart_TRLanczos(SVD svd,PetscReal keep)
{
  SVD_TRLANCZOS *ctx = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ1(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument %g must be in the range [0.1,0.9]",keep);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveReal(svd,keep,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetRestart_C",(SVD,PetscReal),(svd,keep));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidRealPointer(keep,2);
  ierr = PetscUseMethod(svd,"SVDTRLanczosGetRestart_C",(SVD,PetscReal*),(svd,keep));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,lock,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetLocking_C",(SVD,PetscBool),(svd,lock));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(lock,2);
  ierr = PetscUseMethod(svd,"SVDTRLanczosGetLocking_C",(SVD,PetscBool*),(svd,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosSetExplicitMatrix_TRLanczos(SVD svd,PetscBool explicitmatrix)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (lanczos->explicitmatrix != explicitmatrix) {
    lanczos->explicitmatrix = explicitmatrix;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosSetExplicitMatrix - Indicate if the matrix Z=[A;B] must
   be built explicitly.

   Logically Collective on svd

   Input Parameters:
+  svd      - singular value solver
-  explicit - Boolean flag indicating if Z=[A;B] is built explicitly

   Options Database Key:
.  -svd_trlanczos_explicitmatrix <boolean> - Indicates the boolean flag

   Notes:
   This option is relevant for the GSVD case only.
   Z is the coefficient matrix of the KSP solver used internally.

   Level: advanced

.seealso: SVDTRLanczosGetExplicitMatrix()
@*/
PetscErrorCode SVDTRLanczosSetExplicitMatrix(SVD svd,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmatrix,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDTRLanczosGetExplicitMatrix_TRLanczos(SVD svd,PetscBool *explicitmatrix)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *explicitmatrix = lanczos->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   SVDTRLanczosGetExplicitMatrix - Returns the flag indicating if Z=[A;B] is built explicitly.

   Not Collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicit - the mode flag

   Level: advanced

.seealso: SVDTRLanczosSetExplicitMatrix()
@*/
PetscErrorCode SVDTRLanczosGetExplicitMatrix(SVD svd,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(explicitmatrix,2);
  ierr = PetscUseMethod(svd,"SVDTRLanczosGetExplicitMatrix_C",(SVD,PetscBool*),(svd,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDReset_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (svd->isgeneralized) {
    ierr = KSPReset(lanczos->ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&lanczos->Z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  if (svd->isgeneralized) {
    ierr = KSPDestroy(&lanczos->ksp);CHKERRQ(ierr);
  }
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetOneSide_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetGBidiag_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetGBidiag_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_TRLanczos(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  %d%% of basis vectors kept after restart\n",(int)(100*lanczos->keep));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  using the %slocking variant\n",lanczos->lock?"":"non-");CHKERRQ(ierr);
    if (svd->isgeneralized) {
      const char *bidiag="";

      switch (lanczos->bidiag) {
        case SVD_TRLANCZOS_GBIDIAG_SINGLE: bidiag = "single"; break;
        case SVD_TRLANCZOS_GBIDIAG_UPPER:  bidiag = "joint upper-upper"; break;
        case SVD_TRLANCZOS_GBIDIAG_LOWER:  bidiag = "joint lower-upper"; break;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  bidiagonalization choice: %s\n",bidiag);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  %s matrix\n",lanczos->explicitmatrix?"explicit":"implicit");CHKERRQ(ierr);
      if (!lanczos->ksp) { ierr = SVDTRLanczosGetKSP(svd,&lanczos->ksp);CHKERRQ(ierr); }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(lanczos->ksp,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  %s-sided reorthogonalization\n",lanczos->oneside? "one": "two");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&ctx);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C",SVDTRLanczosSetOneSide_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetOneSide_C",SVDTRLanczosGetOneSide_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetGBidiag_C",SVDTRLanczosSetGBidiag_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetGBidiag_C",SVDTRLanczosGetGBidiag_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetKSP_C",SVDTRLanczosSetKSP_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetKSP_C",SVDTRLanczosGetKSP_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetRestart_C",SVDTRLanczosSetRestart_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetRestart_C",SVDTRLanczosGetRestart_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetLocking_C",SVDTRLanczosSetLocking_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetLocking_C",SVDTRLanczosGetLocking_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetExplicitMatrix_C",SVDTRLanczosSetExplicitMatrix_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetExplicitMatrix_C",SVDTRLanczosGetExplicitMatrix_TRLanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

