/*                       

   Straightforward linearization for quadratic eigenproblems.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/qepimpl.h"         /*I "slepcqep.h" I*/
#include "slepceps.h"

/*
    Given the quadratic problem (l^2*M + l*C + K)*x = 0 the following
    linearization is employed:

      A*z = l*B*z   where   A = [  0   I ]     B = [ I  0 ]     z = [  x  ]
                                [ -K  -C ]         [ 0  M ]         [ l*x ]
 */

typedef struct {
  PetscTruth explicitmatrix;
  Mat        A,B;             /* matrices of generalized eigenproblem */
  EPS        eps;             /* linear eigensolver for Az=lBz */
  Mat        M,C,K;           /* copy of QEP coefficient matrices */
  Vec        x1,x2,y1,y2;     /* work vectors */
} QEP_LINEAR;

#undef __FUNCT__  
#define __FUNCT__ "MatMult_QEPLINEAR_A"
PetscErrorCode MatMult_QEPLINEAR_A(Mat A,Vec x, Vec y)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx;
  PetscScalar    *px,*py;
  PetscInt       m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y2 = -(K*x1 + C*x2) */
  ierr = MatMult(ctx->K,ctx->x1,ctx->y2);CHKERRQ(ierr);
  ierr = MatMult(ctx->C,ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y2,1.0,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,-1.0);CHKERRQ(ierr);
  /* y1 = x2 */
  ierr = VecCopy(ctx->x2,ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_QEPLINEAR_B"
PetscErrorCode MatMult_QEPLINEAR_B(Mat B,Vec x, Vec y)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx;
  PetscScalar    *px,*py;
  PetscInt       m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+m);CHKERRQ(ierr);
  /* y1 = x1 */
  ierr = VecCopy(ctx->x1,ctx->y1);CHKERRQ(ierr);
  /* y2 = M*x2 */
  ierr = MatMult(ctx->M,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_QEPLINEAR_A"
PetscErrorCode MatGetDiagonal_QEPLINEAR_A(Mat A,Vec diag)
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
  ierr = VecSet(ctx->x1,0.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->C,ctx->x2);CHKERRQ(ierr);
  ierr = VecScale(ctx->x2,-1.0);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_QEPLINEAR_B"
PetscErrorCode MatGetDiagonal_QEPLINEAR_B(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx;
  PetscScalar    *pd;
  PetscInt       m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,pd);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,pd+m);CHKERRQ(ierr);
  ierr = VecSet(ctx->x1,1.0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->M,ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateExplicit_QEPLINEAR_A"
PetscErrorCode MatCreateExplicit_QEPLINEAR_A(MPI_Comm comm,QEP_LINEAR *ctx,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,i,j,row,start,end,ncols,*pos;
  const PetscInt    *cols;
  const PetscScalar *vals;
  
  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m+n,m+n,M+N,M+N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*n,&pos);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(ctx->M,&start,&end);CHKERRQ(ierr);
  for (i=start;i<end;i++) {
    row = i + M;
    ierr = MatSetValue(*A,i,i+M,-1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatGetRow(ctx->K,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(*A,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(ctx->K,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatGetRow(ctx->C,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0;j<ncols;j++) 
      pos[j] = cols[j] + M;
    ierr = MatSetValues(*A,1,&i,ncols,pos,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(ctx->C,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = PetscFree(pos);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatScale(*A,-1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateExplicit_QEPLINEAR_B"
PetscErrorCode MatCreateExplicit_QEPLINEAR_B(MPI_Comm comm,QEP_LINEAR *ctx,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n,i,j,row,start,end,ncols,*pos;
  const PetscInt    *cols;
  const PetscScalar *vals;
  
  PetscFunctionBegin;
  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(comm,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,m+n,m+n,M+N,M+N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*B);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*n,&pos);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(ctx->M,&start,&end);CHKERRQ(ierr);
  for (i=start;i<end;i++) {
    row = i + M;
    ierr = MatSetValue(*B,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatGetRow(ctx->M,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0;j<ncols;j++) 
      pos[j] = cols[j] + M;
    ierr = MatSetValues(*B,1,&row,ncols,pos,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(ctx->M,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = PetscFree(pos);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetUp_LINEAR"
PetscErrorCode QEPSetUp_LINEAR(QEP qep)
{
  PetscErrorCode    ierr;
  QEP_LINEAR        *ctx = (QEP_LINEAR *)qep->data;
  PetscInt          M,N,m,n;
  EPSWhich          which;

  PetscFunctionBegin;
  
  if (ctx->A) { 
    ierr = MatDestroy(ctx->A);CHKERRQ(ierr);
    ierr = MatDestroy(ctx->B);CHKERRQ(ierr);
  }
  if (ctx->x1) { 
    ierr = VecDestroy(ctx->x1);CHKERRQ(ierr); 
    ierr = VecDestroy(ctx->x2);CHKERRQ(ierr); 
    ierr = VecDestroy(ctx->y1);CHKERRQ(ierr); 
    ierr = VecDestroy(ctx->y2);CHKERRQ(ierr); 
  }

  ierr = MatGetSize(ctx->M,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->M,&m,&n);CHKERRQ(ierr);
  if (ctx->explicitmatrix) {
    ctx->x1 = ctx->x2 = ctx->y1 = ctx->y2 = PETSC_NULL;
    ierr = MatCreateExplicit_QEPLINEAR_A(((PetscObject)qep)->comm,ctx,&ctx->A);CHKERRQ(ierr);
    ierr = MatCreateExplicit_QEPLINEAR_B(((PetscObject)qep)->comm,ctx,&ctx->B);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(((PetscObject)qep)->comm,m,M,PETSC_NULL,&ctx->x1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)qep)->comm,n,N,PETSC_NULL,&ctx->x2);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)qep)->comm,m,M,PETSC_NULL,&ctx->y1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)qep)->comm,n,N,PETSC_NULL,&ctx->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(((PetscObject)qep)->comm,m+n,m+n,M+N,M+N,ctx,&ctx->A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_QEPLINEAR_A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->A,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_QEPLINEAR_A);CHKERRQ(ierr);
    ierr = MatCreateShell(((PetscObject)qep)->comm,m+n,m+n,M+N,M+N,ctx,&ctx->B);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->B,MATOP_MULT,(void(*)(void))MatMult_QEPLINEAR_B);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->B,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_QEPLINEAR_B);CHKERRQ(ierr);
  }

  ierr = EPSSetOperators(ctx->eps,ctx->A,ctx->B);CHKERRQ(ierr);
  ierr = EPSSetProblemType(ctx->eps,EPS_GNHEP);CHKERRQ(ierr);
  switch (qep->which) {
      case QEP_LARGEST_MAGNITUDE:  which = EPS_LARGEST_MAGNITUDE; break;
      case QEP_SMALLEST_MAGNITUDE: which = EPS_SMALLEST_MAGNITUDE; break;
      case QEP_LARGEST_REAL:       which = EPS_LARGEST_REAL; break;
      case QEP_SMALLEST_REAL:      which = EPS_SMALLEST_REAL; break;
      case QEP_LARGEST_IMAGINARY:  which = EPS_LARGEST_IMAGINARY; break;
      case QEP_SMALLEST_IMAGINARY: which = EPS_SMALLEST_IMAGINARY; break;
  }
  ierr = EPSSetWhichEigenpairs(ctx->eps,which);CHKERRQ(ierr);
  ierr = EPSSetDimensions(ctx->eps,qep->nev,qep->ncv,qep->mpd);CHKERRQ(ierr);
  ierr = EPSSetTolerances(ctx->eps,qep->tol,qep->max_it);CHKERRQ(ierr);
  ierr = EPSSetUp(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(ctx->eps,PETSC_NULL,&qep->ncv,&qep->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(ctx->eps,&qep->tol,&qep->max_it);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSolve_LINEAR"
PetscErrorCode QEPSolve_LINEAR(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR *)qep->data;
  PetscInt       i,M,start,end;
  PetscScalar    *pv,tmp;
  PetscReal      norm;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      normi;
#endif
  Vec            xr,xi;
  IS             isV;
  VecScatter     vsV;

  PetscFunctionBegin;
  ierr = EPSSolve(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(ctx->eps,&qep->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(ctx->eps,&qep->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&qep->reason);CHKERRQ(ierr);
  ierr = MatGetVecs(ctx->A,&xr,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(ctx->A,&xi,PETSC_NULL);CHKERRQ(ierr);
  if (ctx->explicitmatrix) {
    ierr = MatGetSize(ctx->M,&M,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(qep->V[0],&start,&end);CHKERRQ(ierr);
    ierr = ISCreateBlock(((PetscObject)qep)->comm,end-start,1,&start,&isV);CHKERRQ(ierr);      
    ierr = VecScatterCreate(xr,isV,qep->V[0],PETSC_NULL,&vsV);CHKERRQ(ierr);
    for (i=0;i<qep->nconv;i++) {
      ierr = EPSGetEigenpair(ctx->eps,i,&qep->eigr[i],&qep->eigi[i],xr,xi);CHKERRQ(ierr);
      ierr = VecScatterBegin(vsV,xr,qep->V[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      //ierr = VecScatterBegin(vsV,xi,qep->V[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(vsV,xr,qep->V[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      //ierr = VecScatterEnd(vsV,xi,qep->V[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    ierr = ISDestroy(isV);CHKERRQ(ierr);
    ierr = VecScatterDestroy(vsV);CHKERRQ(ierr);
  } else {
    for (i=0;i<qep->nconv;i++) {
      ierr = EPSGetEigenpair(ctx->eps,i,&qep->eigr[i],&qep->eigi[i],xr,xi);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      if (qep->eigi[i]>0) {   /* first eigenvalue of a complex conjugate pair */
        ierr = VecGetArray(xr,&pv);CHKERRQ(ierr);
        ierr = VecPlaceArray(ctx->x1,pv);CHKERRQ(ierr);
        ierr = VecCopy(ctx->x1,qep->V[i]);CHKERRQ(ierr);
        ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
        ierr = VecRestoreArray(xr,&pv);CHKERRQ(ierr);
        ierr = VecGetArray(xi,&pv);CHKERRQ(ierr);
        ierr = VecPlaceArray(ctx->x1,pv);CHKERRQ(ierr);
        ierr = VecCopy(ctx->x1,qep->V[i+1]);CHKERRQ(ierr);
        ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
        ierr = VecRestoreArray(xi,&pv);CHKERRQ(ierr);
      }
      else if (qep->eigi[i]==0)   /* real eigenvalue */
#endif
      {
        ierr = VecGetArray(xr,&pv);CHKERRQ(ierr);
        ierr = VecPlaceArray(ctx->x1,pv);CHKERRQ(ierr);
        ierr = VecCopy(ctx->x1,qep->V[i]);CHKERRQ(ierr);
        ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
        ierr = VecRestoreArray(xr,&pv);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecDestroy(xr);CHKERRQ(ierr);
  ierr = VecDestroy(xi);CHKERRQ(ierr);

  /* Normalize eigenvector */
  for (i=0;i<qep->nconv;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (qep->eigi[i] != 0.0) {
      ierr = VecNorm(qep->V[i],NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecNorm(qep->V[i+1],NORM_2,&normi);CHKERRQ(ierr);
      tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
      ierr = VecScale(qep->V[i],tmp);CHKERRQ(ierr);
      ierr = VecScale(qep->V[i+1],tmp);CHKERRQ(ierr);
      i++;     
    } else
#endif
    {
      ierr = VecNormalize(qep->V[i],PETSC_NULL);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSMonitor_QEP_LINEAR"
PetscErrorCode EPSMonitor_QEP_LINEAR(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt   i;
  QEP        qep = (QEP)ctx;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0;i<nest;i++) {
    qep->eigr[i] = eigr[i];
    qep->eigi[i] = eigi[i];
    qep->errest[i] = errest[i];
    if (errest[i] < qep->tol) nconv++;
  }
  QEPMonitor(qep,its,nconv,qep->eigr,qep->eigi,qep->errest,nest);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetFromOptions_LINEAR"
PetscErrorCode QEPSetFromOptions_LINEAR(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR *)qep->data;
  ST             st;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)qep)->comm,((PetscObject)qep)->prefix,"LINEAR Quadratic Eigenvalue Problem solver Options","QEP");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-qep_linear_explicitmatrix","Use explicit matrix in linearization","QEPLinearSetExplicitMatrix",PETSC_FALSE,&ctx->explicitmatrix,PETSC_NULL);CHKERRQ(ierr);
  if (!ctx->explicitmatrix) {
    /* use as default an ST with shell matrix and Jacobi */ 
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,STMATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = EPSSetFromOptions(ctx->eps);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPLinearSetExplicitMatrix_LINEAR"
PetscErrorCode QEPLinearSetExplicitMatrix_LINEAR(QEP qep,PetscTruth explicitmatrix)
{
  QEP_LINEAR *ctx = (QEP_LINEAR *)qep->data;

  PetscFunctionBegin;
  ctx->explicitmatrix = explicitmatrix;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSetExplicitMatrix"
/*@
   QEPLinearSetExplicitMatrix - Indicate if the matrices A and B for the linearization
   of the quadratic problem must be built explicitly.

   Collective on QEP

   Input Parameters:
+  qep      - quadratic eigenvalue solver
-  explicit - boolean flag indicating if the matrices are built explicitly

   Options Database Key:
.  -qep_linear_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: QEPLinearGetExplicitMatrix()
@*/
PetscErrorCode QEPLinearSetExplicitMatrix(QEP qep,PetscTruth explicitmatrix)
{
  PetscErrorCode ierr, (*f)(QEP,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)qep,"QEPLinearSetExplicitMatrix_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(qep,explicitmatrix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPLinearGetExplicitMatrix_LINEAR"
PetscErrorCode QEPLinearGetExplicitMatrix_LINEAR(QEP qep,PetscTruth *explicitmatrix)
{
  QEP_LINEAR *ctx = (QEP_LINEAR *)qep->data;

  PetscFunctionBegin;
  PetscValidPointer(explicitmatrix,2);
  *explicitmatrix = ctx->explicitmatrix;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetExplicitMatrix"
/*@C
   QEPLinearGetExplicitMatrix - Returns the flag indicating if the matrices A and B
   for the linearization of the quadratic problem are built explicitly

   Not collective

   Input Parameter:
.  qep  - quadratic eigenvalue solver

   Output Parameter:
.  explicit - the mode flag

   Level: advanced

.seealso: QEPLinearSetExplicitMatrix()
@*/
PetscErrorCode QEPLinearGetExplicitMatrix(QEP qep,PetscTruth *explicitmatrix)
{
  PetscErrorCode ierr, (*f)(QEP,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)qep,"QEPLinearGetExplicitMatrix_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(qep,explicitmatrix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPLinearSetEPS_LINEAR"
PetscErrorCode QEPLinearSetEPS_LINEAR(QEP qep,EPS eps)
{
  PetscErrorCode  ierr;
  QEP_LINEAR *ctx = (QEP_LINEAR *)qep->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,2);
  PetscCheckSameComm(qep,1,eps,2);
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(ctx->eps);CHKERRQ(ierr);  
  ctx->eps = eps;
  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "QEPLinearSetEPS"
/*@
   QEPLinearSetEPS - Associate an eigensolver object (EPS) to the
   quadratic eigenvalue solver. 

   Collective on QEP

   Input Parameters:
+  qep - quadratic eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: QEPLinearGetEPS()
@*/
PetscErrorCode QEPLinearSetEPS(QEP qep,EPS eps)
{
  PetscErrorCode ierr, (*f)(QEP,EPS eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)qep,"QEPLinearSetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(qep,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPLinearGetEPS_LINEAR"
PetscErrorCode QEPLinearGetEPS_LINEAR(QEP qep,EPS *eps)
{
  QEP_LINEAR *ctx = (QEP_LINEAR *)qep->data;

  PetscFunctionBegin;
  PetscValidPointer(eps,2);
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "QEPLinearGetEPS"
/*@C
   QEPLinearGetEPS - Retrieve the eigensolver object (EPS) associated
   to the quadratic eigenvalue solver.

   Not Collective

   Input Parameter:
.  qep - quadratic eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: QEPLinearSetEPS()
@*/
PetscErrorCode QEPLinearGetEPS(QEP qep,EPS *eps)
{
  PetscErrorCode ierr, (*f)(QEP,EPS *eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)qep,"QEPLinearGetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(qep,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPView_LINEAR"
PetscErrorCode QEPView_LINEAR(QEP qep,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  QEP_LINEAR *ctx = (QEP_LINEAR *)qep->data;

  PetscFunctionBegin;
  if (ctx->explicitmatrix) {
    ierr = PetscViewerASCIIPrintf(viewer,"linearized matrices: explicit\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"linearized matrices: implicit\n");CHKERRQ(ierr);
  }
  ierr = EPSView(ctx->eps,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDestroy_LINEAR"
PetscErrorCode QEPDestroy_LINEAR(QEP qep)
{
  PetscErrorCode  ierr;
  QEP_LINEAR *ctx = (QEP_LINEAR *)qep->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(ctx->eps);CHKERRQ(ierr);
  if (ctx->A) {
    ierr = MatDestroy(ctx->A);CHKERRQ(ierr);
    ierr = MatDestroy(ctx->B);CHKERRQ(ierr);
  }
  if (ctx->x1) { 
    ierr = VecDestroy(ctx->x1);CHKERRQ(ierr);
    ierr = VecDestroy(ctx->x2);CHKERRQ(ierr);
    ierr = VecDestroy(ctx->y1);CHKERRQ(ierr);
    ierr = VecDestroy(ctx->y2);CHKERRQ(ierr);
  }
  ierr = PetscFree(qep->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPCreate_LINEAR"
PetscErrorCode QEPCreate_LINEAR(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx;

  PetscFunctionBegin;
  ierr = PetscNew(QEP_LINEAR,&ctx);CHKERRQ(ierr);
  PetscLogObjectMemory(qep,sizeof(QEP_LINEAR));
  qep->data                      = (void *)ctx;
  qep->ops->solve                = QEPSolve_LINEAR;
  qep->ops->setup                = QEPSetUp_LINEAR;
  qep->ops->setfromoptions       = QEPSetFromOptions_LINEAR;
  qep->ops->destroy              = QEPDestroy_LINEAR;
  qep->ops->view                 = QEPView_LINEAR;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)qep,"QEPLinearSetEPS_C","QEPLinearSetEPS_LINEAR",QEPLinearSetEPS_LINEAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)qep,"QEPLinearGetEPS_C","QEPLinearGetEPS_LINEAR",QEPLinearGetEPS_LINEAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)qep,"QEPLinearSetExplicitMatrix_C","QEPLinearSetExplicitMatrix_LINEAR",QEPLinearSetExplicitMatrix_LINEAR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)qep,"QEPLinearGetExplicitMatrix_C","QEPLinearGetExplicitMatrix_LINEAR",QEPLinearGetExplicitMatrix_LINEAR);CHKERRQ(ierr);

  ierr = EPSCreate(((PetscObject)qep)->comm,&ctx->eps);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(ctx->eps,((PetscObject)qep)->prefix);CHKERRQ(ierr);
  ierr = EPSAppendOptionsPrefix(ctx->eps,"qep_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)qep,1);CHKERRQ(ierr);  
  PetscLogObjectParent(qep,ctx->eps);
  ierr = EPSSetIP(ctx->eps,qep->ip);CHKERRQ(ierr);
  ierr = EPSMonitorSet(ctx->eps,EPSMonitor_QEP_LINEAR,qep,PETSC_NULL);CHKERRQ(ierr);
  ctx->M = qep->M;
  ctx->C = qep->C;
  ctx->K = qep->K;
  ctx->explicitmatrix = PETSC_FALSE;
  ctx->A = PETSC_NULL;
  ctx->B = PETSC_NULL;
  ctx->x1 = PETSC_NULL;
  ctx->x2 = PETSC_NULL;
  ctx->y1 = PETSC_NULL;
  ctx->y2 = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END

