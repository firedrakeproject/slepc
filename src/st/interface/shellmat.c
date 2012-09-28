/*
      This file contains the subroutines which implement various operations
      of the matrix associated to the shift-and-invert technique for eigenvalue
      problems, and also a subroutine to create it.

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

#include <slepc-private/stimpl.h>

typedef struct {
  PetscScalar alpha;
  ST          st;
  Vec         z;
} ST_SHELLMAT;

#undef __FUNCT__
#define __FUNCT__ "STMatShellShift"
PetscErrorCode STMatShellShift(Mat A,PetscScalar alpha)
{
  PetscErrorCode ierr;
  ST_SHELLMAT    *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ctx->alpha = alpha;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatShellMult"
static PetscErrorCode STMatShellMult(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST_SHELLMAT    *ctx;
  ST             st;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  st = ctx->st;
  ierr = MatMult(st->A[0],x,y);CHKERRQ(ierr);
  if (ctx->alpha != 0.0) { 
    if (st->nmat>1) {  /* y = (A + alpha*B) x */
      ierr = MatMult(st->A[1],x,ctx->z);CHKERRQ(ierr);
      ierr = VecAXPY(y,ctx->alpha,ctx->z);CHKERRQ(ierr); 
    } else {    /* y = (A + alpha*I) x */
      ierr = VecAXPY(y,ctx->alpha,x);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatShellMultTranspose"
static PetscErrorCode STMatShellMultTranspose(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST_SHELLMAT    *ctx;
  ST             st;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  st = ctx->st;
  ierr = MatMultTranspose(st->A[0],x,y);CHKERRQ(ierr);
  if (ctx->alpha != 0.0) { 
    if (st->nmat>1) {  /* y = (A + alpha*B) x */
      ierr = MatMultTranspose(st->A[1],x,ctx->z);CHKERRQ(ierr);
      ierr = VecAXPY(y,ctx->alpha,ctx->z);CHKERRQ(ierr); 
    } else {    /* y = (A + alpha*I) x */
      ierr = VecAXPY(y,ctx->alpha,x);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatShellGetDiagonal"
static PetscErrorCode STMatShellGetDiagonal(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  ST_SHELLMAT    *ctx;
  ST             st;
  Vec            diagb;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  st = ctx->st;
  ierr = MatGetDiagonal(st->A[0],diag);CHKERRQ(ierr);
  if (ctx->alpha != 0.0) { 
    if (st->nmat>1) {
      ierr = VecDuplicate(diag,&diagb);CHKERRQ(ierr);
      ierr = MatGetDiagonal(st->A[1],diagb);CHKERRQ(ierr);
      ierr = VecAXPY(diag,ctx->alpha,diagb);CHKERRQ(ierr);
      ierr = VecDestroy(&diagb);CHKERRQ(ierr);
    } else {
      ierr = VecShift(diag,ctx->alpha);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatShellDestroy"
static PetscErrorCode STMatShellDestroy(Mat A)
{
  PetscErrorCode ierr;
  ST_SHELLMAT    *ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->z);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STMatShellCreate"
PetscErrorCode STMatShellCreate(ST st,PetscScalar alpha,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       n,m,N,M;
  PetscBool      hasA,hasB;
  ST_SHELLMAT    *ctx;

  PetscFunctionBegin;
  if (st->nmat>2) SETERRQ(((PetscObject)st)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not implemented for polynomial problems");
  ierr = MatGetSize(st->A[0],&M,&N);CHKERRQ(ierr);  
  ierr = MatGetLocalSize(st->A[0],&m,&n);CHKERRQ(ierr);  
  ierr = PetscNew(ST_SHELLMAT,&ctx);CHKERRQ(ierr);
  ctx->st = st;
  ctx->alpha = alpha;
  ierr = MatGetVecs(st->A[0],&ctx->z,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreateShell(((PetscObject)st)->comm,m,n,M,N,(void*)ctx,mat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*mat,MATOP_MULT,(void(*)(void))STMatShellMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*mat,MATOP_MULT_TRANSPOSE,(void(*)(void))STMatShellMultTranspose);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*mat,MATOP_DESTROY,(void(*)(void))STMatShellDestroy);CHKERRQ(ierr);

  ierr = MatHasOperation(st->A[0],MATOP_GET_DIAGONAL,&hasA);CHKERRQ(ierr);
  if (st->nmat>1) { ierr = MatHasOperation(st->A[1],MATOP_GET_DIAGONAL,&hasB);CHKERRQ(ierr); }
  if ((hasA && st->nmat==1) || (hasA && hasB)) {
    ierr = MatShellSetOperation(*mat,MATOP_GET_DIAGONAL,(void(*)(void))STMatShellGetDiagonal);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

