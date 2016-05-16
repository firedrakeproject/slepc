/*
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
  PetscErrorCode (*evaluatefunctionmatvec)(FN,Mat,Vec);
  PetscErrorCode (*evaluatefunctionmatvecsym)(FN,Mat,Vec);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,FN);
  PetscErrorCode (*view)(FN,PetscViewer);
  PetscErrorCode (*duplicate)(FN,MPI_Comm,FN*);
  PetscErrorCode (*destroy)(FN);
};

#define FN_MAX_W 6

struct _p_FN {
  PETSCHEADER(struct _FNOps);
  /*------------------------- User parameters --------------------------*/
  PetscScalar alpha;          /* inner scaling (argument) */
  PetscScalar beta;           /* outer scaling (result) */

  /*---------------------- Cached data and workspace -------------------*/
  Mat         W[FN_MAX_W];    /* workspace matrices */
  PetscInt    nw;             /* number of allocated W matrices */
  PetscInt    cw;             /* current W matrix */
  void        *data;
};

#undef __FUNCT__
#define __FUNCT__ "FN_AllocateWorkMat"
/*
  FN_AllocateWorkMat - Allocate a work Mat of the same dimension of A and copy
  its contents. The work matrix is returned in M and should be freed with
  FN_FreeWorkMat().
*/
PETSC_STATIC_INLINE PetscErrorCode FN_AllocateWorkMat(FN fn,Mat A,Mat *M)
{
  PetscErrorCode ierr;
  PetscInt       n,na;
  PetscBool      create=PETSC_FALSE;

  PetscFunctionBegin;
  *M = NULL;
  if (fn->cw==FN_MAX_W) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Too many requested work matrices %D",fn->cw);
  if (fn->nw<=fn->cw) {
    create=PETSC_TRUE;
    fn->nw++;
  } else {
    ierr = MatGetSize(fn->W[fn->cw],&n,NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A,&na,NULL);CHKERRQ(ierr);
    if (n!=na) {
      ierr = MatDestroy(&fn->W[fn->cw]);CHKERRQ(ierr);
      create=PETSC_TRUE;
    }
  }
  if (create) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&fn->W[fn->cw]);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)fn,(PetscObject)fn->W[fn->cw]);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(A,fn->W[fn->cw],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  *M = fn->W[fn->cw];
  fn->cw++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FN_FreeWorkMat"
/*
  FN_FreeWorkMat - Release a work matrix created with FN_AllocateWorkMat().
*/
PETSC_STATIC_INLINE PetscErrorCode FN_FreeWorkMat(FN fn,Mat *M)
{
  PetscFunctionBegin;
  if (!fn->cw) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"There are no work matrices");
  fn->cw--;
  if (fn->W[fn->cw]!=*M) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Work matrices must be freed in the reverse order of their creation");
  *M = NULL;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode SlepcMatDenseSqrt(PetscBLASInt,PetscScalar*,PetscBLASInt);
PETSC_INTERN PetscErrorCode SlepcSchurParlettSqrt(PetscBLASInt,PetscScalar*,PetscBLASInt,PetscBool);

#endif
