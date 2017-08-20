/*
   This file implements a wrapper to the LAPACK eigenvalue subroutines.
   Generalized problems are transformed to standard ones only if necessary.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/epsimpl.h>

PetscErrorCode EPSSetUp_LAPACK(EPS eps)
{
  PetscErrorCode ierr,ierra,ierrb;
  PetscBool      isshift,flg,denseok=PETSC_FALSE;
  Mat            A,B,OP,shell,Ar,Br,Adense=NULL,Bdense=NULL;
  PetscScalar    shift,*Ap,*Bp;
  PetscInt       i,ld,nmat;
  KSP            ksp;
  PC             pc;
  Vec            v;

  PetscFunctionBegin;
  eps->ncv = eps->n;
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->balance!=EPS_BALANCE_NONE) { ierr = PetscInfo(eps,"Warning: balancing ignored\n");CHKERRQ(ierr); }
  if (eps->stopping!=EPSStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"User-defined stopping test not supported");
  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);

  /* attempt to get dense representations of A and B separately */
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (isshift) {
    ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
    ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
    ierr = MatHasOperation(A,MATOP_CREATE_SUBMATRICES,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscPushErrorHandler(PetscIgnoreErrorHandler,NULL);
      ierra  = MatCreateRedundantMatrix(A,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);
      if (!ierra) { ierra |= MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&Adense); }
      ierra |= MatDestroy(&Ar);
      PetscPopErrorHandler();
    } else ierra = 1;
    if (nmat>1) {
      ierr = STGetMatrix(eps->st,1,&B);CHKERRQ(ierr);
      ierr = MatHasOperation(B,MATOP_CREATE_SUBMATRICES,&flg);CHKERRQ(ierr);
      if (flg) {
        PetscPushErrorHandler(PetscIgnoreErrorHandler,NULL);
        ierrb  = MatCreateRedundantMatrix(B,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Br);
        if (!ierrb) { ierrb |= MatConvert(Br,MATSEQDENSE,MAT_INITIAL_MATRIX,&Bdense); }
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
        if (eps->ispositive) {
          ierr = DSSetType(eps->ds,DSGHEP);CHKERRQ(ierr);
        } else {
          ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr); /* TODO: should be DSGHIEP */
        }
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
  } else {
    ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
  }
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(eps->ds,eps->ncv,0,0,0);CHKERRQ(ierr);

  if (denseok) {
    ierr = STGetShift(eps->st,&shift);CHKERRQ(ierr);
    if (shift != 0.0) {
      ierr = MatShift(Adense,shift);CHKERRQ(ierr);
    }
    /* use dummy pc and ksp to avoid problems when B is not positive definite */
    ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(eps,"Using slow explicit operator\n");CHKERRQ(ierr);
    ierr = STGetOperator(eps->st,&shell);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(shell,&OP);CHKERRQ(ierr);
    ierr = MatDestroy(&shell);CHKERRQ(ierr);
    ierr = MatDestroy(&Adense);CHKERRQ(ierr);
    ierr = MatCreateRedundantMatrix(OP,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);CHKERRQ(ierr);
    ierr = MatDestroy(&OP);CHKERRQ(ierr);
    ierr = MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
    ierr = MatDestroy(&Ar);CHKERRQ(ierr);
  }

  /* fill DS matrices */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ld,NULL,&v);CHKERRQ(ierr);
  ierr = DSGetArray(eps->ds,DS_MAT_A,&Ap);CHKERRQ(ierr);
  for (i=0;i<ld;i++) {
    ierr = VecPlaceArray(v,Ap+i*ld);CHKERRQ(ierr);
    ierr = MatGetColumnVector(Adense,v,i);CHKERRQ(ierr);
    ierr = VecResetArray(v);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(eps->ds,DS_MAT_A,&Ap);CHKERRQ(ierr);
  if (denseok && eps->isgeneralized) {
    ierr = DSGetArray(eps->ds,DS_MAT_B,&Bp);CHKERRQ(ierr);
    for (i=0;i<ld;i++) {
      ierr = VecPlaceArray(v,Bp+i*ld);CHKERRQ(ierr);
      ierr = MatGetColumnVector(Bdense,v,i);CHKERRQ(ierr);
      ierr = VecResetArray(v);CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_B,&Bp);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = MatDestroy(&Adense);CHKERRQ(ierr);
  ierr = MatDestroy(&Bdense);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       n=eps->n,i,low,high;
  PetscScalar    *array,*pX;
  Vec            v;

  PetscFunctionBegin;
  ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
  ierr = DSSort(eps->ds,eps->eigr,eps->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

  /* right eigenvectors */
  ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  for (i=0;i<eps->ncv;i++) {
    ierr = BVGetColumn(eps->V,i,&v);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v,&low,&high);CHKERRQ(ierr);
    ierr = VecGetArray(v,&array);CHKERRQ(ierr);
    ierr = PetscMemcpy(array,pX+i*n+low,(high-low)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,i,&v);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);

  eps->nconv  = eps->ncv;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EPSCreate_LAPACK(EPS eps)
{
  PetscFunctionBegin;
  eps->useds = PETSC_TRUE;
  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_LAPACK;
  eps->ops->setup          = EPSSetUp_LAPACK;
  eps->ops->backtransform  = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}

