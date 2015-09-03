/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#if !defined(_FNIMPL)
#define _FNIMPL

#include <slepcfn.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool FNRegisterAllCalled;
PETSC_EXTERN PetscErrorCode FNRegisterAll(void);
PETSC_EXTERN PetscLogEvent FN_Evaluate;

typedef struct _FNOps *FNOps;

struct _FNOps {
  PetscErrorCode (*evaluatefunction)(FN,PetscScalar,PetscScalar*);
  PetscErrorCode (*evaluatederivative)(FN,PetscScalar,PetscScalar*);
  PetscErrorCode (*evaluatefunctionmat)(FN,Mat,Mat);
  PetscErrorCode (*evaluatefunctionmatsym)(FN,Mat,Mat);
  PetscErrorCode (*setfromoptions)(PetscOptions*,FN);
  PetscErrorCode (*view)(FN,PetscViewer);
  PetscErrorCode (*duplicate)(FN,MPI_Comm,FN*);
  PetscErrorCode (*destroy)(FN);
};

struct _p_FN {
  PETSCHEADER(struct _FNOps);
  /*------------------------- User parameters --------------------------*/
  PetscScalar        alpha;   /* inner scaling (argument) */
  PetscScalar        beta;    /* outer scaling (result) */

  /*---------------------- Cached data and workspace -------------------*/
  Mat                W;       /* workspace matrix */
  void               *data;
};

#undef __FUNCT__
#define __FUNCT__ "FN_AllocateWorkMat"
/*
  FN_AllocateWorkMat - Allocate a working Mat of appropriate size if not available already.
*/
PETSC_STATIC_INLINE PetscErrorCode FN_AllocateWorkMat(FN fn,Mat A)
{
  PetscErrorCode ierr;
  PetscInt       n,na;
  PetscBool      create=PETSC_FALSE;

  PetscFunctionBegin;
  if (!fn->W) create=PETSC_TRUE;
  else {
    ierr = MatGetSize(fn->W,&n,NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A,&na,NULL);CHKERRQ(ierr);
    if (n!=na) {
      ierr = MatDestroy(&fn->W);CHKERRQ(ierr);
      create=PETSC_TRUE;
    }
  }
  if (create) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&fn->W);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)fn,(PetscObject)fn->W);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(A,fn->W,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#endif
