/*                       

   Linearization for gyroscopic QEP, companion form 1.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/qepimpl.h>         /*I "slepcqep.h" I*/
#include "linearp.h"

/*
    Given the quadratic problem (l^2*M + l*C + K)*x = 0 the following
    linearization is employed:

      A*z = l*B*z   where   A = [  K   0 ]     B = [ 0  K ]     z = [  x  ]
                                [  C   K ]         [-M  0 ]         [ l*x ]
 */

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Linear_H1A"
PetscErrorCode MatMult_Linear_H1A(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  QEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,PETSC_NULL);CHKERRQ(ierr);
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
  QEP_LINEAR        *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,PETSC_NULL);CHKERRQ(ierr);
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
  QEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode MatCreateExplicit_Linear_H1A(MPI_Comm comm,QEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = SlepcMatTile(1.0,ctx->K,0.0,ctx->K,ctx->sfactor,ctx->C,1.0,ctx->K,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateExplicit_Linear_H1B"
PetscErrorCode MatCreateExplicit_Linear_H1B(MPI_Comm comm,QEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = SlepcMatTile(0.0,ctx->K,1.0,ctx->K,-ctx->sfactor*ctx->sfactor,ctx->M,0.0,ctx->K,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

