/*
      This file contains the subroutines which implement various operations
      of the matrix associated to the shift-and-invert technique for eigenvalue
      problems, and also a subroutine to create it.
*/

#include "src/st/stimpl.h"
#include "sinvert.h"  

#undef __FUNCT__
#define __FUNCT__ "MatSinvert_Mult"
static int MatSinvert_Mult(Mat A,Vec x,Vec y)
{
  int         ierr;
  CTX_SINV    *ctx;
  PetscScalar alpha;

  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  alpha = -ctx->sigma;

  ierr = MatMult(ctx->A,x,y);CHKERRQ(ierr);
  if (alpha != 0.0) { 
    if (ctx->B) {  /* y = (A - sB) x */
      ierr = MatMult(ctx->B,x,ctx->w);CHKERRQ(ierr);
      ierr = VecAXPY(&alpha,ctx->w,y);CHKERRQ(ierr); 
    }
    else {    /* y = (A - sI) x */
    ierr = VecAXPY(&alpha,x,y);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSinvert_GetDiagonal"
static int MatSinvert_GetDiagonal(Mat A,Vec diag)
{
  int         ierr;
  CTX_SINV    *ctx;
  PetscScalar alpha;
  Vec         diagb;

  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  alpha = -ctx->sigma;

  ierr = MatGetDiagonal(ctx->A,diag);CHKERRQ(ierr);
  if (alpha != 0.0) { 
    if (ctx->B) {
      ierr = VecDuplicate(diag,&diagb);CHKERRQ(ierr);
      ierr = MatGetDiagonal(ctx->B,diagb);CHKERRQ(ierr);
      ierr = VecAXPY(&alpha,diagb,diag);CHKERRQ(ierr);
      ierr = VecDestroy(diagb);CHKERRQ(ierr);
    }
    else {
      ierr = VecShift(&alpha,diag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSinvert_Destroy"
static int MatSinvert_Destroy(Mat A)
{
  CTX_SINV   *ctx;
  int        ierr;

  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  if (ctx->B) { ierr = VecDestroy(ctx->w);CHKERRQ(ierr); }
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMatSinvert"
int MatCreateMatSinvert(ST st,Mat *mat)
{
  int          n, m, N, M, ierr;
  PetscTruth   hasA, hasB;
  CTX_SINV     *ctx;

  PetscFunctionBegin;
  ierr = PetscNew(CTX_SINV,&ctx);CHKERRQ(ierr);
  PetscMemzero(ctx,sizeof(CTX_SINV));
  PetscLogObjectMemory(st,sizeof(CTX_SINV));
  ctx->A = st->A;
  ctx->B = st->B;
  ctx->sigma = st->sigma;
  if (st->B) { ierr = VecDuplicate(st->vec,&ctx->w);CHKERRQ(ierr); }
  ierr = MatGetSize(st->A,&M,&N);CHKERRQ(ierr);  
  ierr = MatGetLocalSize(st->A,&m,&n);CHKERRQ(ierr);  
  ierr = MatCreateShell(st->comm,m,n,M,N,(void*)ctx,mat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*mat,MATOP_MULT,(void(*)(void))MatSinvert_Mult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*mat,MATOP_DESTROY,(void(*)(void))MatSinvert_Destroy);CHKERRQ(ierr);

  ierr = MatHasOperation(st->A,MATOP_GET_DIAGONAL,&hasA);CHKERRQ(ierr);
  if (st->B) { ierr = MatHasOperation(st->B,MATOP_GET_DIAGONAL,&hasB);CHKERRQ(ierr); }
  if ( (hasA && !st->B) || (hasA && hasB) ) {
    ierr = MatShellSetOperation(*mat,MATOP_GET_DIAGONAL,(void(*)(void))MatSinvert_GetDiagonal);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

