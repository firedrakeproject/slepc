/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to the LAPACK eigenvalue subroutines.
   Generalized problems are transformed to standard ones only if necessary.
*/

#include <slepc/private/epsimpl.h>

PetscErrorCode EPSSetUp_LAPACK(EPS eps)
{
  PetscErrorCode ierra,ierrb;
  PetscBool      isshift,flg,denseok=PETSC_FALSE;
  Mat            A,B,OP,shell,Ar,Br,Adense=NULL,Bdense=NULL;
  PetscScalar    shift,*Ap,*Bp;
  PetscInt       i,ld,nmat;
  KSP            ksp;
  PC             pc;
  Vec            v;

  PetscFunctionBegin;
  eps->ncv = eps->n;
  if (eps->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(eps,"Warning: parameter mpd ignored\n"));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  PetscCheck(eps->which!=EPS_ALL || eps->inta==eps->intb,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support interval computation");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);
  CHKERRQ(EPSAllocateSolution(eps,0));

  /* attempt to get dense representations of A and B separately */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift));
  if (isshift) {
    CHKERRQ(STGetNumMatrices(eps->st,&nmat));
    CHKERRQ(STGetMatrix(eps->st,0,&A));
    CHKERRQ(MatHasOperation(A,MATOP_CREATE_SUBMATRICES,&flg));
    if (flg) {
      PetscPushErrorHandler(PetscReturnErrorHandler,NULL);
      ierra  = MatCreateRedundantMatrix(A,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);
      if (!ierra) ierra |= MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&Adense);
      ierra |= MatDestroy(&Ar);
      PetscPopErrorHandler();
    } else ierra = 1;
    if (nmat>1) {
      CHKERRQ(STGetMatrix(eps->st,1,&B));
      CHKERRQ(MatHasOperation(B,MATOP_CREATE_SUBMATRICES,&flg));
      if (flg) {
        PetscPushErrorHandler(PetscReturnErrorHandler,NULL);
        ierrb  = MatCreateRedundantMatrix(B,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Br);
        if (!ierrb) ierrb |= MatConvert(Br,MATSEQDENSE,MAT_INITIAL_MATRIX,&Bdense);
        ierrb |= MatDestroy(&Br);
        PetscPopErrorHandler();
      } else ierrb = 1;
    } else ierrb = 0;
    denseok = PetscNot(ierra || ierrb);
  }

  /* setup DS */
  if (denseok) {
    if (eps->isgeneralized) {
      if (eps->ishermitian) {
        if (eps->ispositive) CHKERRQ(DSSetType(eps->ds,DSGHEP));
        else CHKERRQ(DSSetType(eps->ds,DSGNHEP)); /* TODO: should be DSGHIEP */
      } else CHKERRQ(DSSetType(eps->ds,DSGNHEP));
    } else {
      if (eps->ishermitian) CHKERRQ(DSSetType(eps->ds,DSHEP));
      else CHKERRQ(DSSetType(eps->ds,DSNHEP));
    }
  } else CHKERRQ(DSSetType(eps->ds,DSNHEP));
  CHKERRQ(DSAllocate(eps->ds,eps->ncv));
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(DSSetDimensions(eps->ds,eps->ncv,0,0));

  if (denseok) {
    CHKERRQ(STGetShift(eps->st,&shift));
    if (shift != 0.0) {
      if (nmat>1) CHKERRQ(MatAXPY(Adense,-shift,Bdense,SAME_NONZERO_PATTERN));
      else CHKERRQ(MatShift(Adense,-shift));
    }
    /* use dummy pc and ksp to avoid problems when B is not positive definite */
    CHKERRQ(STGetKSP(eps->st,&ksp));
    CHKERRQ(KSPSetType(ksp,KSPPREONLY));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));
  } else {
    CHKERRQ(PetscInfo(eps,"Using slow explicit operator\n"));
    CHKERRQ(STGetOperator(eps->st,&shell));
    CHKERRQ(MatComputeOperator(shell,MATDENSE,&OP));
    CHKERRQ(STRestoreOperator(eps->st,&shell));
    CHKERRQ(MatDestroy(&Adense));
    CHKERRQ(MatCreateRedundantMatrix(OP,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Adense));
    CHKERRQ(MatDestroy(&OP));
  }

  /* fill DS matrices */
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,ld,NULL,&v));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&Ap));
  for (i=0;i<ld;i++) {
    CHKERRQ(VecPlaceArray(v,Ap+i*ld));
    CHKERRQ(MatGetColumnVector(Adense,v,i));
    CHKERRQ(VecResetArray(v));
  }
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&Ap));
  if (denseok && eps->isgeneralized) {
    CHKERRQ(DSGetArray(eps->ds,DS_MAT_B,&Bp));
    for (i=0;i<ld;i++) {
      CHKERRQ(VecPlaceArray(v,Bp+i*ld));
      CHKERRQ(MatGetColumnVector(Bdense,v,i));
      CHKERRQ(VecResetArray(v));
    }
    CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_B,&Bp));
  }
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
  CHKERRQ(MatDestroy(&Adense));
  CHKERRQ(MatDestroy(&Bdense));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_LAPACK(EPS eps)
{
  PetscInt       n=eps->n,i,low,high;
  PetscScalar    *array,*pX,*pY;
  Vec            v,w;

  PetscFunctionBegin;
  CHKERRQ(DSSolve(eps->ds,eps->eigr,eps->eigi));
  CHKERRQ(DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL));
  CHKERRQ(DSSynchronize(eps->ds,eps->eigr,eps->eigi));

  /* right eigenvectors */
  CHKERRQ(DSVectors(eps->ds,DS_MAT_X,NULL,NULL));
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_X,&pX));
  for (i=0;i<eps->ncv;i++) {
    CHKERRQ(BVGetColumn(eps->V,i,&v));
    CHKERRQ(VecGetOwnershipRange(v,&low,&high));
    CHKERRQ(VecGetArray(v,&array));
    CHKERRQ(PetscArraycpy(array,pX+i*n+low,high-low));
    CHKERRQ(VecRestoreArray(v,&array));
    CHKERRQ(BVRestoreColumn(eps->V,i,&v));
  }
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_X,&pX));

  /* left eigenvectors */
  if (eps->twosided) {
    CHKERRQ(DSVectors(eps->ds,DS_MAT_Y,NULL,NULL));
    CHKERRQ(DSGetArray(eps->ds,DS_MAT_Y,&pY));
    for (i=0;i<eps->ncv;i++) {
      CHKERRQ(BVGetColumn(eps->W,i,&w));
      CHKERRQ(VecGetOwnershipRange(w,&low,&high));
      CHKERRQ(VecGetArray(w,&array));
      CHKERRQ(PetscArraycpy(array,pY+i*n+low,high-low));
      CHKERRQ(VecRestoreArray(w,&array));
      CHKERRQ(BVRestoreColumn(eps->W,i,&w));
    }
    CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_Y,&pY));
  }

  eps->nconv  = eps->ncv;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_LAPACK(EPS eps)
{
  PetscFunctionBegin;
  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_LAPACK;
  eps->ops->setup          = EPSSetUp_LAPACK;
  eps->ops->setupsort      = EPSSetUpSort_Default;
  eps->ops->backtransform  = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}
