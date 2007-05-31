/*
     SVD routines for accessing the problem matrix.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/svd/svdimpl.h"      /*I "slepcsvd.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SVDMatMult"
PetscErrorCode SVDMatMult(SVD svd,PetscTruth trans,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidHeaderSpecific(y,VEC_COOKIE,4);
  svd->matvecs++;
  if (trans) {
    if (svd->AT) {
      ierr = MatMult(svd->AT,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(svd->A,x,y);CHKERRQ(ierr);
    }
  } else {
    if (svd->A) {
      ierr = MatMult(svd->A,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(svd->AT,x,y);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDMatGetVecs"
PetscErrorCode SVDMatGetVecs(SVD svd,Vec *x,Vec *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (svd->A) {
    ierr = MatGetVecs(svd->A,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatGetVecs(svd->AT,y,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDMatGetSize"
PetscErrorCode SVDMatGetSize(SVD svd,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (svd->A) {
    ierr = MatGetSize(svd->A,m,n);CHKERRQ(ierr);
  } else {
    ierr = MatGetSize(svd->AT,n,m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDMatGetLocalSize"
PetscErrorCode SVDMatGetLocalSize(SVD svd,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (svd->A) {
    ierr = MatGetLocalSize(svd->A,m,n);CHKERRQ(ierr);
  } else {
    ierr = MatGetLocalSize(svd->AT,n,m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
