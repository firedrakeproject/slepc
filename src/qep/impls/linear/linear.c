/*
   Straightforward linearization for quadratic eigenproblems.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/qepimpl.h>         /*I "slepcqep.h" I*/
#include <slepcvec.h>
#include "linearp.h"

#undef __FUNCT__
#define __FUNCT__ "QEPSetUp_Linear"
PetscErrorCode QEPSetUp_Linear(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;
  PetscInt       i=0;
  EPSWhich       which;
  PetscBool      trackall,khas,mhas;
  PetscScalar    sigma;
  PetscReal      knorm,mnorm;
  /* function tables */
  PetscErrorCode (*fcreate[][2])(MPI_Comm,QEP_LINEAR*,Mat*) = {
    { MatCreateExplicit_QLinear_N1A, MatCreateExplicit_QLinear_N1B },   /* N1 */
    { MatCreateExplicit_QLinear_N2A, MatCreateExplicit_QLinear_N2B },   /* N2 */
    { MatCreateExplicit_QLinear_S1A, MatCreateExplicit_QLinear_S1B },   /* S1 */
    { MatCreateExplicit_QLinear_S2A, MatCreateExplicit_QLinear_S2B },   /* S2 */
    { MatCreateExplicit_QLinear_H1A, MatCreateExplicit_QLinear_H1B },   /* H1 */
    { MatCreateExplicit_QLinear_H2A, MatCreateExplicit_QLinear_H2B }    /* H2 */
  };
  PetscErrorCode (*fmult[][2])(Mat,Vec,Vec) = {
    { MatMult_QLinear_N1A, MatMult_QLinear_N1B },
    { MatMult_QLinear_N2A, MatMult_QLinear_N2B },
    { MatMult_QLinear_S1A, MatMult_QLinear_S1B },
    { MatMult_QLinear_S2A, MatMult_QLinear_S2B },
    { MatMult_QLinear_H1A, MatMult_QLinear_H1B },
    { MatMult_QLinear_H2A, MatMult_QLinear_H2B }
  };
  PetscErrorCode (*fgetdiagonal[][2])(Mat,Vec) = {
    { MatGetDiagonal_QLinear_N1A, MatGetDiagonal_QLinear_N1B },
    { MatGetDiagonal_QLinear_N2A, MatGetDiagonal_QLinear_N2B },
    { MatGetDiagonal_QLinear_S1A, MatGetDiagonal_QLinear_S1B },
    { MatGetDiagonal_QLinear_S2A, MatGetDiagonal_QLinear_S2B },
    { MatGetDiagonal_QLinear_H1A, MatGetDiagonal_QLinear_H1B },
    { MatGetDiagonal_QLinear_H2A, MatGetDiagonal_QLinear_H2B }
  };

  PetscFunctionBegin;
  if (!ctx->cform) ctx->cform = 1;
  if (!qep->which) qep->which = QEP_LARGEST_MAGNITUDE;
  ctx->M = qep->M;
  ctx->C = qep->C;
  ctx->K = qep->K;

  /* Compute scaling factor if not set by user */
  if (qep->sfactor==0.0) {
    ierr = MatHasOperation(qep->K,MATOP_NORM,&khas);CHKERRQ(ierr);
    ierr = MatHasOperation(qep->M,MATOP_NORM,&mhas);CHKERRQ(ierr);
    if (khas && mhas) {
      ierr = MatNorm(qep->K,NORM_INFINITY,&knorm);CHKERRQ(ierr);
      ierr = MatNorm(qep->M,NORM_INFINITY,&mnorm);CHKERRQ(ierr);
      qep->sfactor = PetscSqrtReal(knorm/mnorm);
    } else qep->sfactor = 1.0;
  }
  ctx->sfactor = qep->sfactor;

  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->B);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);

  switch (qep->problem_type) {
    case QEP_GENERAL:    i = 0; break;
    case QEP_HERMITIAN:  i = 2; break;
    case QEP_GYROSCOPIC: i = 4; break;
    default: SETERRQ(PetscObjectComm((PetscObject)qep),1,"Wrong value of qep->problem_type");
  }
  i += ctx->cform-1;

  if (ctx->explicitmatrix) {
    ctx->x1 = ctx->x2 = ctx->y1 = ctx->y2 = NULL;
    ierr = (*fcreate[i][0])(PetscObjectComm((PetscObject)qep),ctx,&ctx->A);CHKERRQ(ierr);
    ierr = (*fcreate[i][1])(PetscObjectComm((PetscObject)qep),ctx,&ctx->B);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&ctx->x1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&ctx->x2);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&ctx->y1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&ctx->y2);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->x1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->x2);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->y1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)qep),2*qep->nloc,2*qep->nloc,2*qep->n,2*qep->n,ctx,&ctx->A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))fmult[i][0]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->A,MATOP_GET_DIAGONAL,(void(*)(void))fgetdiagonal[i][0]);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)qep),2*qep->nloc,2*qep->nloc,2*qep->n,2*qep->n,ctx,&ctx->B);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->B,MATOP_MULT,(void(*)(void))fmult[i][1]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->B,MATOP_GET_DIAGONAL,(void(*)(void))fgetdiagonal[i][1]);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->B);CHKERRQ(ierr);

  if (!ctx->eps) { ierr = QEPLinearGetEPS(qep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSSetOperators(ctx->eps,ctx->A,ctx->B);CHKERRQ(ierr);
  if (qep->problem_type==QEP_HERMITIAN) {
    ierr = EPSSetProblemType(ctx->eps,EPS_GHIEP);CHKERRQ(ierr);
  } else {
    ierr = EPSSetProblemType(ctx->eps,EPS_GNHEP);CHKERRQ(ierr);
  }
  switch (qep->which) {
      case QEP_LARGEST_MAGNITUDE:  which = EPS_LARGEST_MAGNITUDE; break;
      case QEP_SMALLEST_MAGNITUDE: which = EPS_SMALLEST_MAGNITUDE; break;
      case QEP_LARGEST_REAL:       which = EPS_LARGEST_REAL; break;
      case QEP_SMALLEST_REAL:      which = EPS_SMALLEST_REAL; break;
      case QEP_LARGEST_IMAGINARY:  which = EPS_LARGEST_IMAGINARY; break;
      case QEP_SMALLEST_IMAGINARY: which = EPS_SMALLEST_IMAGINARY; break;
      default: SETERRQ(PetscObjectComm((PetscObject)qep),1,"Wrong value of which");
  }
  ierr = EPSSetWhichEigenpairs(ctx->eps,which);CHKERRQ(ierr);
  ierr = EPSSetDimensions(ctx->eps,qep->nev,qep->ncv,qep->mpd);CHKERRQ(ierr);
  ierr = EPSSetTolerances(ctx->eps,qep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:qep->tol/10.0,qep->max_it);CHKERRQ(ierr);
  /* Transfer the trackall option from qep to eps */
  ierr = QEPGetTrackAll(qep,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(ctx->eps,trackall);CHKERRQ(ierr);
  if (ctx->setfromoptionscalled) {
    ierr = EPSSetFromOptions(ctx->eps);CHKERRQ(ierr);
    ctx->setfromoptionscalled = PETSC_FALSE;
  }
  /* temporary change of target */
  if (qep->sfactor!=1.0) {
    ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
    ierr = EPSSetTarget(ctx->eps,sigma/qep->sfactor);CHKERRQ(ierr);
  }
  ierr = EPSSetUp(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(ctx->eps,NULL,&qep->ncv,&qep->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(ctx->eps,NULL,&qep->max_it);CHKERRQ(ierr);
  if (qep->nini>0) { ierr = PetscInfo(qep,"Ignoring initial vectors\n");CHKERRQ(ierr); }
  ierr = QEPAllocateSolution(qep,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSelect_Norm"
/*
   QEPLinearSelect_Norm - Auxiliary routine that copies the solution of the
   linear eigenproblem to the QEP object. The eigenvector of the generalized
   problem is supposed to be
                               z = [  x  ]
                                   [ l*x ]
   The eigenvector is taken from z(1:n) or z(n+1:2*n) depending on the explicitly
   computed residual norm.
   Finally, x is normalized so that ||x||_2 = 1.
*/
static PetscErrorCode QEPLinearSelect_Norm(QEP qep,EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *px;
  PetscReal      rn1,rn2;
  Vec            xr,xi,wr,wi;
  Mat            A;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *py;
#endif

  PetscFunctionBegin;
  ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&xr,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(xr,&xi);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&wr);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&wi);CHKERRQ(ierr);
  for (i=0;i<qep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,&qep->eigr[i],&qep->eigi[i],xr,xi);CHKERRQ(ierr);
    qep->eigr[i] *= qep->sfactor;
    qep->eigi[i] *= qep->sfactor;
#if !defined(PETSC_USE_COMPLEX)
    if (qep->eigi[i]>0.0) {   /* first eigenvalue of a complex conjugate pair */
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArray(xi,&py);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px);CHKERRQ(ierr);
      ierr = VecPlaceArray(wi,py);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,wi,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = QEPComputeResidualNorm_Private(qep,qep->eigr[i],qep->eigi[i],wr,wi,&rn1);CHKERRQ(ierr);
      ierr = BVInsertVec(qep->V,i,wr);CHKERRQ(ierr);
      ierr = BVInsertVec(qep->V,i+1,wi);CHKERRQ(ierr);
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecResetArray(wi);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px+qep->nloc);CHKERRQ(ierr);
      ierr = VecPlaceArray(wi,py+qep->nloc);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,wi,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = QEPComputeResidualNorm_Private(qep,qep->eigr[i],qep->eigi[i],wr,wi,&rn2);CHKERRQ(ierr);
      if (rn1>rn2) {
        ierr = BVInsertVec(qep->V,i,wr);CHKERRQ(ierr);
        ierr = BVInsertVec(qep->V,i+1,wi);CHKERRQ(ierr);
      }
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecResetArray(wi);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
      ierr = VecRestoreArray(xi,&py);CHKERRQ(ierr);
    } else if (qep->eigi[i]==0.0)   /* real eigenvalue */
#endif
    {
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = QEPComputeResidualNorm_Private(qep,qep->eigr[i],qep->eigi[i],wr,NULL,&rn1);CHKERRQ(ierr);
      ierr = BVInsertVec(qep->V,i,wr);CHKERRQ(ierr);
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px+qep->nloc);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = QEPComputeResidualNorm_Private(qep,qep->eigr[i],qep->eigi[i],wr,NULL,&rn2);CHKERRQ(ierr);
      if (rn1>rn2) {
        ierr = BVInsertVec(qep->V,i,wr);CHKERRQ(ierr);
      }
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&wr);CHKERRQ(ierr);
  ierr = VecDestroy(&wi);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSelect_Simple"
/*
   QEPLinearSelect_Simple - Auxiliary routine that copies the solution of the
   linear eigenproblem to the QEP object. The eigenvector of the generalized
   problem is supposed to be
                               z = [  x  ]
                                   [ l*x ]
   If |l|<1.0, the eigenvector is taken from z(1:n), otherwise from z(n+1:2*n).
   Finally, x is normalized so that ||x||_2 = 1.
*/
static PetscErrorCode QEPLinearSelect_Simple(QEP qep,EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,offset;
  PetscScalar    *px;
  Vec            xr,xi,w,vi;
#if !defined(PETSC_USE_COMPLEX)
  Vec            vi1;
#endif
  Mat            A;

  PetscFunctionBegin;
  ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&xr,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(xr,&xi);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)qep),1,qep->nloc,qep->n,NULL,&w);CHKERRQ(ierr);
  for (i=0;i<qep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,&qep->eigr[i],&qep->eigi[i],xr,xi);CHKERRQ(ierr);
    qep->eigr[i] *= qep->sfactor;
    qep->eigi[i] *= qep->sfactor;
    if (SlepcAbsEigenvalue(qep->eigr[i],qep->eigi[i])>1.0) offset = qep->nloc;
    else offset = 0;
#if !defined(PETSC_USE_COMPLEX)
    if (qep->eigi[i]>0.0) {   /* first eigenvalue of a complex conjugate pair */
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(qep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArray(xi,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(qep->V,i+1,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArray(xi,&px);CHKERRQ(ierr);
      ierr = BVGetColumn(qep->V,i,&vi);CHKERRQ(ierr);
      ierr = BVGetColumn(qep->V,i+1,&vi1);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(vi,vi1,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(qep->V,i,&vi);CHKERRQ(ierr);
      ierr = BVRestoreColumn(qep->V,i+1,&vi1);CHKERRQ(ierr);
    } else if (qep->eigi[i]==0.0)   /* real eigenvalue */
#endif
    {
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(qep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
      ierr = BVGetColumn(qep->V,i,&vi);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(vi,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(qep->V,i,&vi);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSolve_Linear"
PetscErrorCode QEPSolve_Linear(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;
  PetscBool      flg=PETSC_FALSE;
  PetscScalar    sigma;

  PetscFunctionBegin;
  ierr = EPSSolve(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(ctx->eps,&qep->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(ctx->eps,&qep->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&qep->reason);CHKERRQ(ierr);
  /* restore target */
  ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
  ierr = EPSSetTarget(ctx->eps,sigma*qep->sfactor);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(((PetscObject)qep)->prefix,"-qep_linear_select_simple",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = QEPLinearSelect_Simple(qep,ctx->eps);CHKERRQ(ierr);
  } else {
    ierr = QEPLinearSelect_Norm(qep,ctx->eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSMonitor_Linear"
static PetscErrorCode EPSMonitor_Linear(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i;
  QEP            qep = (QEP)ctx;
  ST             st;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0;i<PetscMin(nest,qep->ncv);i++) {
    qep->eigr[i] = eigr[i];
    qep->eigi[i] = eigi[i];
    qep->errest[i] = errest[i];
    if (0.0 < errest[i] && errest[i] < qep->tol) nconv++;
  }
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STBackTransform(st,nest,qep->eigr,qep->eigi);CHKERRQ(ierr);
  ierr = QEPMonitor(qep,its,nconv,qep->eigr,qep->eigi,qep->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetFromOptions_Linear"
PetscErrorCode QEPSetFromOptions_Linear(QEP qep)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  PetscInt       i;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;
  ST             st;

  PetscFunctionBegin;
  ctx->setfromoptionscalled = PETSC_TRUE;
  ierr = PetscOptionsHead("QEP Linear Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qep_linear_cform","Number of the companion form","QEPLinearSetCompanionForm",ctx->cform,&i,&set);CHKERRQ(ierr);
  if (set) {
    ierr = QEPLinearSetCompanionForm(qep,i);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-qep_linear_explicitmatrix","Use explicit matrix in linearization","QEPLinearSetExplicitMatrix",ctx->explicitmatrix,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = QEPLinearSetExplicitMatrix(qep,val);CHKERRQ(ierr);
  }
  if (!ctx->explicitmatrix) {
    /* use as default an ST with shell matrix and Jacobi */
    if (!ctx->eps) { ierr = QEPLinearGetEPS(qep,&ctx->eps);CHKERRQ(ierr); }
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSetCompanionForm_Linear"
static PetscErrorCode QEPLinearSetCompanionForm_Linear(QEP qep,PetscInt cform)
{
  QEP_LINEAR *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  if (!cform) PetscFunctionReturn(0);
  if (cform==PETSC_DECIDE || cform==PETSC_DEFAULT) ctx->cform = 1;
  else {
    if (cform!=1 && cform!=2) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid value of argument 'cform'");
    ctx->cform = cform;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSetCompanionForm"
/*@
   QEPLinearSetCompanionForm - Choose between the two companion forms available
   for the linearization of the quadratic problem.

   Logically Collective on QEP

   Input Parameters:
+  qep   - quadratic eigenvalue solver
-  cform - 1 or 2 (first or second companion form)

   Options Database Key:
.  -qep_linear_cform <int> - Choose the companion form

   Level: advanced

.seealso: QEPLinearGetCompanionForm()
@*/
PetscErrorCode QEPLinearSetCompanionForm(QEP qep,PetscInt cform)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(qep,cform,2);
  ierr = PetscTryMethod(qep,"QEPLinearSetCompanionForm_C",(QEP,PetscInt),(qep,cform));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetCompanionForm_Linear"
static PetscErrorCode QEPLinearGetCompanionForm_Linear(QEP qep,PetscInt *cform)
{
  QEP_LINEAR *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  *cform = ctx->cform;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetCompanionForm"
/*@
   QEPLinearGetCompanionForm - Returns the number of the companion form that
   will be used for the linearization of the quadratic problem.

   Not Collective

   Input Parameter:
.  qep  - quadratic eigenvalue solver

   Output Parameter:
.  cform - the companion form number (1 or 2)

   Level: advanced

.seealso: QEPLinearSetCompanionForm()
@*/
PetscErrorCode QEPLinearGetCompanionForm(QEP qep,PetscInt *cform)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidIntPointer(cform,2);
  ierr = PetscTryMethod(qep,"QEPLinearGetCompanionForm_C",(QEP,PetscInt*),(qep,cform));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSetExplicitMatrix_Linear"
static PetscErrorCode QEPLinearSetExplicitMatrix_Linear(QEP qep,PetscBool explicitmatrix)
{
  QEP_LINEAR *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  ctx->explicitmatrix = explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSetExplicitMatrix"
/*@
   QEPLinearSetExplicitMatrix - Indicate if the matrices A and B for the linearization
   of the quadratic problem must be built explicitly.

   Logically Collective on QEP

   Input Parameters:
+  qep      - quadratic eigenvalue solver
-  explicit - boolean flag indicating if the matrices are built explicitly

   Options Database Key:
.  -qep_linear_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: QEPLinearGetExplicitMatrix()
@*/
PetscErrorCode QEPLinearSetExplicitMatrix(QEP qep,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(qep,explicitmatrix,2);
  ierr = PetscTryMethod(qep,"QEPLinearSetExplicitMatrix_C",(QEP,PetscBool),(qep,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetExplicitMatrix_Linear"
static PetscErrorCode QEPLinearGetExplicitMatrix_Linear(QEP qep,PetscBool *explicitmatrix)
{
  QEP_LINEAR *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  *explicitmatrix = ctx->explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetExplicitMatrix"
/*@
   QEPLinearGetExplicitMatrix - Returns the flag indicating if the matrices A and B
   for the linearization of the quadratic problem are built explicitly.

   Not Collective

   Input Parameter:
.  qep  - quadratic eigenvalue solver

   Output Parameter:
.  explicitmatrix - the mode flag

   Level: advanced

.seealso: QEPLinearSetExplicitMatrix()
@*/
PetscErrorCode QEPLinearGetExplicitMatrix(QEP qep,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(explicitmatrix,2);
  ierr = PetscTryMethod(qep,"QEPLinearGetExplicitMatrix_C",(QEP,PetscBool*),(qep,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearSetEPS_Linear"
static PetscErrorCode QEPLinearSetEPS_Linear(QEP qep,EPS eps)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ctx->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->eps);CHKERRQ(ierr);
  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(qep,1,eps,2);
  ierr = PetscTryMethod(qep,"QEPLinearSetEPS_C",(QEP,EPS),(qep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetEPS_Linear"
static PetscErrorCode QEPLinearGetEPS_Linear(QEP qep,EPS *eps)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;
  ST             st;

  PetscFunctionBegin;
  if (!ctx->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)qep),&ctx->eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(ctx->eps,((PetscObject)qep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(ctx->eps,"qep_");CHKERRQ(ierr);
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetOptionsPrefix(st,((PetscObject)ctx->eps)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)qep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)qep,(PetscObject)ctx->eps);CHKERRQ(ierr);
    ierr = EPSMonitorSet(ctx->eps,EPSMonitor_Linear,qep,NULL);CHKERRQ(ierr);
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPLinearGetEPS"
/*@
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscTryMethod(qep,"QEPLinearGetEPS_C",(QEP,EPS*),(qep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPView_Linear"
PetscErrorCode QEPView_Linear(QEP qep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = QEPLinearGetEPS(qep,&ctx->eps);CHKERRQ(ierr); }
  ierr = PetscViewerASCIIPrintf(viewer,"  Linear: %s matrices\n",ctx->explicitmatrix? "explicit": "implicit");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Linear: %s companion form\n",ctx->cform==1? "1st": "2nd");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = EPSView(ctx->eps,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPReset_Linear"
PetscErrorCode QEPReset_Linear(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = EPSReset(ctx->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->B);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);
  ierr = QEPReset_Default(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPDestroy_Linear"
PetscErrorCode QEPDestroy_Linear(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx = (QEP_LINEAR*)qep->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ierr = PetscFree(qep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearSetCompanionForm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearGetCompanionForm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearGetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPCreate_Linear"
PETSC_EXTERN PetscErrorCode QEPCreate_Linear(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(qep,&ctx);CHKERRQ(ierr);
  qep->data = (void*)ctx;

  qep->ops->solve                = QEPSolve_Linear;
  qep->ops->setup                = QEPSetUp_Linear;
  qep->ops->setfromoptions       = QEPSetFromOptions_Linear;
  qep->ops->destroy              = QEPDestroy_Linear;
  qep->ops->reset                = QEPReset_Linear;
  qep->ops->view                 = QEPView_Linear;
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearSetCompanionForm_C",QEPLinearSetCompanionForm_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearGetCompanionForm_C",QEPLinearGetCompanionForm_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearSetEPS_C",QEPLinearSetEPS_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearGetEPS_C",QEPLinearGetEPS_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearSetExplicitMatrix_C",QEPLinearSetExplicitMatrix_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)qep,"QEPLinearGetExplicitMatrix_C",QEPLinearGetExplicitMatrix_Linear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

