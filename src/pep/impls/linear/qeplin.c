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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y2 = -(K*x1 + C*x2) */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y2,ctx->sfactor,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,-1.0);CHKERRQ(ierr);
  /* y1 = x2 */
  ierr = VecCopy(ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = x1 */
  ierr = VecCopy(ctx->x1,ctx->y1);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(ctx->x1,0.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,ctx->x2);CHKERRQ(ierr);
  ierr = VecScale(ctx->x2,-ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(ctx->x1,1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,ctx->x2);CHKERRQ(ierr);
  ierr = VecScale(ctx->x2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N1A"
PetscErrorCode MatCreateExplicit_Linear_N1A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,i,Istart,Iend;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Id,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(Id,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = SlepcMatTile(0.0,Id,1.0,Id,-1.0,ctx->K,-ctx->sfactor,ctx->C,A);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N1B"
PetscErrorCode MatCreateExplicit_Linear_N1B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,i,Istart,Iend;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Id,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(Id,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x1 */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,-1.0);CHKERRQ(ierr);
  /* y2 = x2 */
  ierr = VecCopy(ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = C*x1 + M*x2 */
  ierr = MatMult(ctx->C,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,ctx->sfactor);CHKERRQ(ierr);
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y1,ctx->sfactor*ctx->sfactor,ctx->y2);CHKERRQ(ierr);
  /* y2 = x1 */
  ierr = VecCopy(ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,ctx->x1);CHKERRQ(ierr);
  ierr = VecScale(ctx->x1,-1.0);CHKERRQ(ierr);
  ierr = VecSet(ctx->x2,1.0);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,ctx->x1);CHKERRQ(ierr);
  ierr = VecScale(ctx->x1,ctx->sfactor);CHKERRQ(ierr);
  ierr = VecSet(ctx->x2,0.0);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N2A"
PetscErrorCode MatCreateExplicit_Linear_N2A(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,i,Istart,Iend;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Id,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(Id,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = SlepcMatTile(-1.0,ctx->K,0.0,Id,0.0,Id,1.0,Id,A);CHKERRQ(ierr);
  ierr = MatDestroy(&Id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExplicit_Linear_N2B"
PetscErrorCode MatCreateExplicit_Linear_N2B(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,i,Istart,Iend;
  Mat            Id;

  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ctx->M),&Id);CHKERRQ(ierr);
  ierr = MatSetSizes(Id,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Id);CHKERRQ(ierr);
  ierr = MatSetUp(Id);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Id,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(Id,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Id,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y2 = -(K*x1 + C*x2) */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,-1.0);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y2,-ctx->sfactor,ctx->y1);CHKERRQ(ierr);
  /* y1 = -K*x2 */
  ierr = MatMult(ctx->K,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,-1.0);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x1 */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,-1.0);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(ctx->x1,0.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,ctx->x2);CHKERRQ(ierr);
  ierr = VecScale(ctx->x2,-ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,ctx->x1);CHKERRQ(ierr);
  ierr = VecScale(ctx->x1,-1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,ctx->x2);CHKERRQ(ierr);
  ierr = VecScale(ctx->x2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x1 */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,-1.0);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = C*x1 + M*x2 */
  ierr = MatMult(ctx->C,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,ctx->sfactor);CHKERRQ(ierr);
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y1,ctx->sfactor*ctx->sfactor,ctx->y2);CHKERRQ(ierr);
  /* y2 = M*x1 */
  ierr = MatMult(ctx->M,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,ctx->x1);CHKERRQ(ierr);
  ierr = VecScale(ctx->x1,-1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,ctx->x2);CHKERRQ(ierr);
  ierr = VecScale(ctx->x2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,ctx->x1);CHKERRQ(ierr);
  ierr = VecScale(ctx->x1,ctx->sfactor);CHKERRQ(ierr);
  ierr = VecSet(ctx->x2,0.0);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y2 = C*x1 + K*x2 */
  ierr = MatMult(ctx->C,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = MatMult(ctx->K,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y2,ctx->sfactor,ctx->y1);CHKERRQ(ierr);
  /* y1 = K*x1 */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = K*x2 */
  ierr = MatMult(ctx->K,ctx->x2,ctx->y1);CHKERRQ(ierr);
  /* y2 = -M*x1 */
  ierr = MatMult(ctx->M,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,-ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->K,ctx->x1);CHKERRQ(ierr);
  ierr = VecCopy(ctx->x1,ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = -K*x2 */
  ierr = MatMult(ctx->K,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,-1.0);CHKERRQ(ierr);
  /* y2 = M*x1 */
  ierr = MatMult(ctx->M,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = M*x1 + C*x2 */
  ierr = MatMult(ctx->M,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y1,ctx->sfactor,ctx->y2);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,ctx->x1);CHKERRQ(ierr);
  ierr = VecScale(ctx->x1,ctx->sfactor*ctx->sfactor);CHKERRQ(ierr);
  ierr = VecCopy(ctx->x1,ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
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

