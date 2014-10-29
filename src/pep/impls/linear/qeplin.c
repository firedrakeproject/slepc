/*

   Various types of linearization for quadratic eigenvalue problem.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/pepimpl.h>
#include "linearp.h"

/*
    Given the quadratic problem (l^2*M + l*C + K)*x = 0 the linearization is
    A*z = l*B*z for z = [  x  ] and A,B defined as follows:
                        [ l*x ]

            N1:
                     A = [  0   I ]     B = [ I  0 ]
                         [ -K  -C ]         [ 0  M ]

            N2:
                     A = [ -K   0 ]     B = [ C  M ]
                         [  0   I ]         [ I  0 ]

            S1:
                     A = [  0  -K ]     B = [-K  0 ]
                         [ -K  -C ]         [ 0  M ]

            S2:
                     A = [ -K   0 ]     B = [ C  M ]
                         [  0   M ]         [ M  0 ]

            H1:
                     A = [  K   0 ]     B = [ 0  K ]
                         [  C   K ]         [-M  0 ]

            H2:
                     A = [  0  -K ]     B = [ M  C ]
                         [  M   0 ]         [ 0  M ]
 */

/* - - - N1 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_N1A"
PetscErrorCode MatMult_Linear_N1A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y2 = -(K*x1 + C*x2) */
  ierr = MatMult(ctx->K,x1,y2);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,x2,y1);CHKERRQ(ierr);
  ierr = VecAXPY(y2,ctx->sfactor,y1);CHKERRQ(ierr);
  ierr = VecScale(y2,-1.0);CHKERRQ(ierr);
  /* y1 = x2 */
  ierr = VecCopy(x2,y1);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_N1B"
PetscErrorCode MatMult_Linear_N1B(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = x1 */
  ierr = VecCopy(x1,y1);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,x2,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_N1A"
PetscErrorCode MatGetDiagonal_Linear_N1A(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(x1,0.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,x2);CHKERRQ(ierr);
  ierr = VecScale(x2,-ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_N1B"
PetscErrorCode MatGetDiagonal_Linear_N1B(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(x1,1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,x2);CHKERRQ(ierr);
  ierr = VecScale(x2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N1A"
PetscErrorCode MatCreateExplicit_Linear_N1A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(Id,1.0);CHKERRQ(ierr);
  ierr = SlepcMatTile(0.0,Id,1.0,Id,-1.0,ctx->K,-ctx->sfactor,ctx->C,A);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N1B"
PetscErrorCode MatCreateExplicit_Linear_N1B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(Id,1.0);CHKERRQ(ierr);
  ierr = SlepcMatTile(1.0,Id,0.0,Id,0.0,Id,ctx->sfactor*ctx->sfactor,ctx->M,B);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* - - - N2 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_N2A"
PetscErrorCode MatMult_Linear_N2A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x1 */
  ierr = MatMult(ctx->K,x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-1.0);CHKERRQ(ierr);
  /* y2 = x2 */
  ierr = VecCopy(x2,y2);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_N2B"
PetscErrorCode MatMult_Linear_N2B(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = C*x1 + M*x2 */
  ierr = MatMult(ctx->C,x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,ctx->sfactor);CHKERRQ(ierr);
  ierr = MatMult(ctx->M,x2,y2);CHKERRQ(ierr);
  ierr = VecAXPY(y1,ctx->sfactor*ctx->sfactor,y2);CHKERRQ(ierr);
  /* y2 = x1 */
  ierr = VecCopy(x1,y2);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_N2A"
PetscErrorCode MatGetDiagonal_Linear_N2A(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,x1);CHKERRQ(ierr);
  ierr = VecScale(x1,-1.0);CHKERRQ(ierr);
  ierr = VecSet(x2,1.0);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_N2B"
PetscErrorCode MatGetDiagonal_Linear_N2B(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,x1);CHKERRQ(ierr);
  ierr = VecScale(x1,ctx->sfactor);CHKERRQ(ierr);
  ierr = VecSet(x2,0.0);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N2A"
PetscErrorCode MatCreateExplicit_Linear_N2A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(Id,1.0);CHKERRQ(ierr);
  ierr = SlepcMatTile(-1.0,ctx->K,0.0,Id,0.0,Id,1.0,Id,A);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N2B"
PetscErrorCode MatCreateExplicit_Linear_N2B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(Id,1.0);CHKERRQ(ierr);
  ierr = SlepcMatTile(ctx->sfactor,ctx->C,ctx->sfactor*ctx->sfactor,ctx->M,1.0,Id,0.0,Id,B);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* - - - S1 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_S1A"
PetscErrorCode MatMult_Linear_S1A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y2 = -(K*x1 + C*x2) */
  ierr = MatMult(ctx->K,x1,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,-1.0);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,x2,y1);CHKERRQ(ierr);
  ierr = VecAXPY(y2,-ctx->sfactor,y1);CHKERRQ(ierr);
  /* y1 = -K*x2 */
  ierr = MatMult(ctx->K,x2,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-1.0);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_S1B"
PetscErrorCode MatMult_Linear_S1B(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x1 */
  ierr = MatMult(ctx->K,x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-1.0);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,x2,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_S1A"
PetscErrorCode MatGetDiagonal_Linear_S1A(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(x1,0.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,x2);CHKERRQ(ierr);
  ierr = VecScale(x2,-ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_S1B"
PetscErrorCode MatGetDiagonal_Linear_S1B(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,x1);CHKERRQ(ierr);
  ierr = VecScale(x1,-1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,x2);CHKERRQ(ierr);
  ierr = VecScale(x2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_S1A"
PetscErrorCode MatCreateExplicit_Linear_S1A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(0.0,ctx->K,-1.0,ctx->K,-1.0,ctx->K,-ctx->sfactor,ctx->C,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_S1B"
PetscErrorCode MatCreateExplicit_Linear_S1B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(-1.0,ctx->K,0.0,ctx->M,0.0,ctx->M,ctx->sfactor*ctx->sfactor,ctx->M,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* - - - S2 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_S2A"
PetscErrorCode MatMult_Linear_S2A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x1 */
  ierr = MatMult(ctx->K,x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-1.0);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,x2,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_S2B"
PetscErrorCode MatMult_Linear_S2B(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = C*x1 + M*x2 */
  ierr = MatMult(ctx->C,x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,ctx->sfactor);CHKERRQ(ierr);
  ierr = MatMult(ctx->M,x2,y2);CHKERRQ(ierr);
  ierr = VecAXPY(y1,ctx->sfactor*ctx->sfactor,y2);CHKERRQ(ierr);
  /* y2 = M*x1 */
  ierr = MatMult(ctx->M,x1,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_S2A"
PetscErrorCode MatGetDiagonal_Linear_S2A(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,x1);CHKERRQ(ierr);
  ierr = VecScale(x1,-1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,x2);CHKERRQ(ierr);
  ierr = VecScale(x2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_S2B"
PetscErrorCode MatGetDiagonal_Linear_S2B(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,x1);CHKERRQ(ierr);
  ierr = VecScale(x1,ctx->sfactor);CHKERRQ(ierr);
  ierr = VecSet(x2,0.0);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_S2A"
PetscErrorCode MatCreateExplicit_Linear_S2A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(-1.0,ctx->K,0.0,ctx->M,0.0,ctx->M,ctx->sfactor*ctx->sfactor,ctx->M,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_S2B"
PetscErrorCode MatCreateExplicit_Linear_S2B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(ctx->sfactor,ctx->C,ctx->sfactor*ctx->sfactor,ctx->M,ctx->sfactor*ctx->sfactor,ctx->M,0.0,ctx->M,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* - - - H1 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_H1A"
PetscErrorCode MatMult_Linear_H1A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y2 = C*x1 + K*x2 */
  ierr = MatMult(ctx->C,x1,y1);CHKERRQ(ierr);
  ierr = MatMult(ctx->K,x2,y2);CHKERRQ(ierr);
  ierr = VecAXPY(y2,ctx->sfactor,y1);CHKERRQ(ierr);
  /* y1 = K*x1 */
  ierr = MatMult(ctx->K,x1,y1);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_H1B"
PetscErrorCode MatMult_Linear_H1B(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = K*x2 */
  ierr = MatMult(ctx->K,x2,y1);CHKERRQ(ierr);
  /* y2 = -M*x1 */
  ierr = MatMult(ctx->M,x1,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,-ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_H1A"
PetscErrorCode MatGetDiagonal_Linear_H1A(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,x1);CHKERRQ(ierr);
  ierr = VecCopy(x1,x2);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_H1B"
PetscErrorCode MatGetDiagonal_Linear_H1B(Mat B,Vec diag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(diag,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_H1A"
PetscErrorCode MatCreateExplicit_Linear_H1A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(1.0,ctx->K,0.0,ctx->K,ctx->sfactor,ctx->C,1.0,ctx->K,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_H1B"
PetscErrorCode MatCreateExplicit_Linear_H1B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(0.0,ctx->K,1.0,ctx->K,-ctx->sfactor*ctx->sfactor,ctx->M,0.0,ctx->K,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* - - - H2 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_H2A"
PetscErrorCode MatMult_Linear_H2A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x2 */
  ierr = MatMult(ctx->K,x2,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-1.0);CHKERRQ(ierr);
  /* y2 = M*x1 */
  ierr = MatMult(ctx->M,x1,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_H2B"
PetscErrorCode MatMult_Linear_H2B(Mat B,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  Vec               x1,x2,y1,y2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];y1=ctx->w[2];y2=ctx->w[3];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(y2,py+m);CHKERRQ(ierr);
  /* y1 = M*x1 + C*x2 */
  ierr = MatMult(ctx->M,x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,x2,y2);CHKERRQ(ierr);
  ierr = VecAXPY(y1,ctx->sfactor,y2);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,x2,y2);CHKERRQ(ierr);
  ierr = VecScale(y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_H2A"
PetscErrorCode MatGetDiagonal_Linear_H2A(Mat A,Vec diag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(diag,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Linear_H2B"
PetscErrorCode MatGetDiagonal_Linear_H2B(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  Vec            x1,x2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  x1=ctx->w[0];x2=ctx->w[1];
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,x1);CHKERRQ(ierr);
  ierr = VecScale(x1,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecCopy(x1,x2);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_H2A"
PetscErrorCode MatCreateExplicit_Linear_H2A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(0.0,ctx->K,-1.0,ctx->K,ctx->sfactor*ctx->sfactor,ctx->M,0.0,ctx->K,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_H2B"
PetscErrorCode MatCreateExplicit_Linear_H2B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcMatTile(ctx->sfactor*ctx->sfactor,ctx->M,ctx->sfactor,ctx->C,0.0,ctx->C,ctx->sfactor*ctx->sfactor,ctx->M,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

