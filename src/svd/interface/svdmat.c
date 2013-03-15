/*
     SVD routines for accessing the problem matrix.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/svdimpl.h>      /*I "slepcsvd.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SVDMatMult"
PetscErrorCode SVDMatMult(SVD svd,PetscBool trans,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  svd->matvecs++;
  if (trans) {
    if (svd->AT) {
      ierr = MatMult(svd->AT,x,y);CHKERRQ(ierr);
    } else {
#if defined(PETSC_USE_COMPLEX)
      ierr = MatMultHermitianTranspose(svd->A,x,y);CHKERRQ(ierr);
#else
      ierr = MatMultTranspose(svd->A,x,y);CHKERRQ(ierr);
#endif
    }
  } else {
    if (svd->A) {
      ierr = MatMult(svd->A,x,y);CHKERRQ(ierr);
    } else {
#if defined(PETSC_USE_COMPLEX)
      ierr = MatMultHermitianTranspose(svd->AT,x,y);CHKERRQ(ierr);
#else
      ierr = MatMultTranspose(svd->AT,x,y);CHKERRQ(ierr);
#endif
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
  if (svd->A) {
    ierr = MatGetLocalSize(svd->A,m,n);CHKERRQ(ierr);
  } else {
    ierr = MatGetLocalSize(svd->AT,n,m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
