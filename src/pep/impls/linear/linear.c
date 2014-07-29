/*
   Straightforward linearization for quadratic eigenproblems.

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

#include <slepc-private/pepimpl.h>         /*I "slepcpep.h" I*/
#include <slepcvec.h>
#include "linearp.h"

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_Linear"
PetscErrorCode PEPSetUp_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  PetscInt       i=0;
  EPSWhich       which;
  PetscBool      trackall,istrivial,flg;
  PetscScalar    sigma;
  /* function tables */
  PetscErrorCode (*fcreate[][2])(MPI_Comm,PEP_LINEAR*,Mat*) = {
    { MatCreateExplicit_Linear_N1A, MatCreateExplicit_Linear_N1B },   /* N1 */
    { MatCreateExplicit_Linear_N2A, MatCreateExplicit_Linear_N2B },   /* N2 */
    { MatCreateExplicit_Linear_S1A, MatCreateExplicit_Linear_S1B },   /* S1 */
    { MatCreateExplicit_Linear_S2A, MatCreateExplicit_Linear_S2B },   /* S2 */
    { MatCreateExplicit_Linear_H1A, MatCreateExplicit_Linear_H1B },   /* H1 */
    { MatCreateExplicit_Linear_H2A, MatCreateExplicit_Linear_H2B }    /* H2 */
  };
  PetscErrorCode (*fmult[][2])(Mat,Vec,Vec) = {
    { MatMult_Linear_N1A, MatMult_Linear_N1B },
    { MatMult_Linear_N2A, MatMult_Linear_N2B },
    { MatMult_Linear_S1A, MatMult_Linear_S1B },
    { MatMult_Linear_S2A, MatMult_Linear_S2B },
    { MatMult_Linear_H1A, MatMult_Linear_H1B },
    { MatMult_Linear_H2A, MatMult_Linear_H2B }
  };
  PetscErrorCode (*fgetdiagonal[][2])(Mat,Vec) = {
    { MatGetDiagonal_Linear_N1A, MatGetDiagonal_Linear_N1B },
    { MatGetDiagonal_Linear_N2A, MatGetDiagonal_Linear_N2B },
    { MatGetDiagonal_Linear_S1A, MatGetDiagonal_Linear_S1B },
    { MatGetDiagonal_Linear_S2A, MatGetDiagonal_Linear_S2B },
    { MatGetDiagonal_Linear_H1A, MatGetDiagonal_Linear_H1B },
    { MatGetDiagonal_Linear_H2A, MatGetDiagonal_Linear_H2B }
  };

  PetscFunctionBegin;
  if (!ctx->cform) ctx->cform = 1;
  if (!pep->which) pep->which = PEP_LARGEST_MAGNITUDE;
  if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver not implemented for non-monomial bases");
  if (pep->nmat!=3) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver only available for quadratic problems");
  if (pep->scale==PEP_SCALE_DIAGONAL || pep->scale==PEP_SCALE_BOTH) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Diagonal scaling not allowed in PEP linear solver");
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"ST transformation flag not allowed for PEP linear solver");

  /* compute scale factor if no set by user */
  ierr = PEPComputeScaleFactor(pep);CHKERRQ(ierr);
 
  ierr = STGetOperators(pep->st,0,&ctx->K);CHKERRQ(ierr);
  ierr = STGetOperators(pep->st,1,&ctx->C);CHKERRQ(ierr);
  ierr = STGetOperators(pep->st,2,&ctx->M);CHKERRQ(ierr);
  ctx->sfactor = pep->sfactor;

  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->B);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);

  switch (pep->problem_type) {
    case PEP_GENERAL:    i = 0; break;
    case PEP_HERMITIAN:  i = 2; break;
    case PEP_GYROSCOPIC: i = 4; break;
    default: SETERRQ(PetscObjectComm((PetscObject)pep),1,"Wrong value of pep->problem_type");
  }
  i += ctx->cform-1;

  if (ctx->explicitmatrix) {
    ctx->x1 = ctx->x2 = ctx->y1 = ctx->y2 = NULL;
    ierr = (*fcreate[i][0])(PetscObjectComm((PetscObject)pep),ctx,&ctx->A);CHKERRQ(ierr);
    ierr = (*fcreate[i][1])(PetscObjectComm((PetscObject)pep),ctx,&ctx->B);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->x1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->x2);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->y1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->y2);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->x1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->x2);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->y1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)pep),2*pep->nloc,2*pep->nloc,2*pep->n,2*pep->n,ctx,&ctx->A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))fmult[i][0]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->A,MATOP_GET_DIAGONAL,(void(*)(void))fgetdiagonal[i][0]);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)pep),2*pep->nloc,2*pep->nloc,2*pep->n,2*pep->n,ctx,&ctx->B);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->B,MATOP_MULT,(void(*)(void))fmult[i][1]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ctx->B,MATOP_GET_DIAGONAL,(void(*)(void))fgetdiagonal[i][1]);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->B);CHKERRQ(ierr);

  if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSSetOperators(ctx->eps,ctx->A,ctx->B);CHKERRQ(ierr);
  if (pep->problem_type==PEP_HERMITIAN) {
    ierr = EPSSetProblemType(ctx->eps,EPS_GHIEP);CHKERRQ(ierr);
  } else {
    ierr = EPSSetProblemType(ctx->eps,EPS_GNHEP);CHKERRQ(ierr);
  }
  switch (pep->which) {
      case PEP_LARGEST_MAGNITUDE:  which = EPS_LARGEST_MAGNITUDE; break;
      case PEP_SMALLEST_MAGNITUDE: which = EPS_SMALLEST_MAGNITUDE; break;
      case PEP_LARGEST_REAL:       which = EPS_LARGEST_REAL; break;
      case PEP_SMALLEST_REAL:      which = EPS_SMALLEST_REAL; break;
      case PEP_LARGEST_IMAGINARY:  which = EPS_LARGEST_IMAGINARY; break;
      case PEP_SMALLEST_IMAGINARY: which = EPS_SMALLEST_IMAGINARY; break;
      case PEP_TARGET_MAGNITUDE:   which = EPS_TARGET_MAGNITUDE; break;
      case PEP_TARGET_REAL:        which = EPS_TARGET_REAL; break;
      case PEP_TARGET_IMAGINARY:   which = EPS_TARGET_IMAGINARY; break;
      default: SETERRQ(PetscObjectComm((PetscObject)pep),1,"Wrong value of which");
  }
  ierr = EPSSetWhichEigenpairs(ctx->eps,which);CHKERRQ(ierr);
  ierr = EPSSetDimensions(ctx->eps,pep->nev,pep->ncv?pep->ncv:PETSC_DEFAULT,pep->mpd?pep->mpd:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(ctx->eps,pep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/10.0:pep->tol/10.0,pep->max_it?pep->max_it:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = RGIsTrivial(pep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) { ierr = EPSSetRG(ctx->eps,pep->rg);CHKERRQ(ierr); }
  /* Transfer the trackall option from pep to eps */
  ierr = PEPGetTrackAll(pep,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(ctx->eps,trackall);CHKERRQ(ierr);

  /* temporary change of target */
  if (pep->sfactor!=1.0) {
    ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
    ierr = EPSSetTarget(ctx->eps,sigma/pep->sfactor);CHKERRQ(ierr);
  }
  ierr = EPSSetUp(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(ctx->eps,NULL,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(ctx->eps,NULL,&pep->max_it);CHKERRQ(ierr);
  if (pep->nini>0) { ierr = PetscInfo(pep,"Ignoring initial vectors\n");CHKERRQ(ierr); }
  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearExtract_Residual"
/*
   PEPLinearExtract_Residual - Auxiliary routine that copies the solution of the
   linear eigenproblem to the PEP object. The eigenvector of the generalized
   problem is supposed to be
                               z = [  x  ]
                                   [ l*x ]
   The eigenvector is taken from z(1:n) or z(n+1:2*n) depending on the explicitly
   computed residual norm.
   Finally, x is normalized so that ||x||_2 = 1.
*/
static PetscErrorCode PEPLinearExtract_Residual(PEP pep,EPS eps)
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
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&wr);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&wi);CHKERRQ(ierr);
  for (i=0;i<pep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,&pep->eigr[i],&pep->eigi[i],xr,xi);CHKERRQ(ierr);
    pep->eigr[i] *= pep->sfactor;
    pep->eigi[i] *= pep->sfactor;
#if !defined(PETSC_USE_COMPLEX)
    if (pep->eigi[i]>0.0) {   /* first eigenvalue of a complex conjugate pair */
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArray(xi,&py);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px);CHKERRQ(ierr);
      ierr = VecPlaceArray(wi,py);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,wi,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = PEPComputeResidualNorm_Private(pep,pep->eigr[i],pep->eigi[i],wr,wi,&rn1);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i+1,wi);CHKERRQ(ierr);
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecResetArray(wi);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px+pep->nloc);CHKERRQ(ierr);
      ierr = VecPlaceArray(wi,py+pep->nloc);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,wi,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = PEPComputeResidualNorm_Private(pep,pep->eigr[i],pep->eigi[i],wr,wi,&rn2);CHKERRQ(ierr);
      if (rn1>rn2) {
        ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
        ierr = BVInsertVec(pep->V,i+1,wi);CHKERRQ(ierr);
      }
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecResetArray(wi);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
      ierr = VecRestoreArray(xi,&py);CHKERRQ(ierr);
    } else if (pep->eigi[i]==0.0)   /* real eigenvalue */
#endif
    {
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = PEPComputeResidualNorm_Private(pep,pep->eigr[i],pep->eigi[i],wr,NULL,&rn1);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px+pep->nloc);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = PEPComputeResidualNorm_Private(pep,pep->eigr[i],pep->eigi[i],wr,NULL,&rn2);CHKERRQ(ierr);
      if (rn1>rn2) {
        ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
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
#define __FUNCT__ "PEPLinearExtract_Norm"
/*
   PEPLinearExtract_Norm - Auxiliary routine that copies the solution of the
   linear eigenproblem to the PEP object. The eigenvector of the generalized
   problem is supposed to be
                               z = [  x  ]
                                   [ l*x ]
   If |l|<1.0, the eigenvector is taken from z(1:n), otherwise from z(n+1:2*n).
   Finally, x is normalized so that ||x||_2 = 1.
*/
static PetscErrorCode PEPLinearExtract_Norm(PEP pep,EPS eps)
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
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&w);CHKERRQ(ierr);
  for (i=0;i<pep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,&pep->eigr[i],&pep->eigi[i],xr,xi);CHKERRQ(ierr);
    pep->eigr[i] *= pep->sfactor;
    pep->eigi[i] *= pep->sfactor;
    if (SlepcAbsEigenvalue(pep->eigr[i],pep->eigi[i])>1.0) offset = pep->nloc;
    else offset = 0;
#if !defined(PETSC_USE_COMPLEX)
    if (pep->eigi[i]>0.0) {   /* first eigenvalue of a complex conjugate pair */
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArray(xi,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i+1,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArray(xi,&px);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,i,&vi);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,i+1,&vi1);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(vi,vi1,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&vi);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i+1,&vi1);CHKERRQ(ierr);
    } else if (pep->eigi[i]==0.0)   /* real eigenvalue */
#endif
    {
      ierr = VecGetArray(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArray(xr,&px);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,i,&vi);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(vi,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&vi);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_Linear"
PetscErrorCode PEPSolve_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  PetscScalar    sigma;

  PetscFunctionBegin;
  ierr = EPSSolve(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(ctx->eps,&pep->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(ctx->eps,&pep->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&pep->reason);CHKERRQ(ierr);
  /* restore target */
  ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
  ierr = EPSSetTarget(ctx->eps,sigma*pep->sfactor);CHKERRQ(ierr);

  switch (pep->extract) {
  case PEP_EXTRACT_NORM:
    ierr = PEPLinearExtract_Norm(pep,ctx->eps);CHKERRQ(ierr);
    break;
  case PEP_EXTRACT_RESIDUAL:
    ierr = PEPLinearExtract_Residual(pep,ctx->eps);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Extraction not implemented in this solver");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSMonitor_Linear"
static PetscErrorCode EPSMonitor_Linear(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i;
  PEP            pep = (PEP)ctx;
  ST             st;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<PetscMin(nest,pep->ncv);i++) {
    pep->eigr[i] = eigr[i];
    pep->eigi[i] = eigi[i];
    pep->errest[i] = errest[i];
  }
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STBackTransform(st,nest,pep->eigr,pep->eigi);CHKERRQ(ierr);
  ierr = PEPMonitor(pep,its,nconv,pep->eigr,pep->eigi,pep->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_Linear"
PetscErrorCode PEPSetFromOptions_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  PetscInt       i;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  ST             st;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSSetFromOptions(ctx->eps);CHKERRQ(ierr);
  ierr = PetscOptionsHead("PEP Linear Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pep_linear_cform","Number of the companion form","PEPLinearSetCompanionForm",ctx->cform,&i,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PEPLinearSetCompanionForm(pep,i);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-pep_linear_explicitmatrix","Use explicit matrix in linearization","PEPLinearSetExplicitMatrix",ctx->explicitmatrix,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PEPLinearSetExplicitMatrix(pep,val);CHKERRQ(ierr);
  }
  if (!ctx->explicitmatrix) {
    /* use as default an ST with shell matrix and Jacobi */
    if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetCompanionForm_Linear"
static PetscErrorCode PEPLinearSetCompanionForm_Linear(PEP pep,PetscInt cform)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!cform) PetscFunctionReturn(0);
  if (cform==PETSC_DECIDE || cform==PETSC_DEFAULT) ctx->cform = 1;
  else {
    if (cform!=1 && cform!=2) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid value of argument 'cform'");
    ctx->cform = cform;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetCompanionForm"
/*@
   PEPLinearSetCompanionForm - Choose between the two companion forms available
   for the linearization of a quadratic eigenproblem.

   Logically Collective on PEP

   Input Parameters:
+  pep   - polynomial eigenvalue solver
-  cform - 1 or 2 (first or second companion form)

   Options Database Key:
.  -pep_linear_cform <int> - Choose the companion form

   Level: advanced

.seealso: PEPLinearGetCompanionForm()
@*/
PetscErrorCode PEPLinearSetCompanionForm(PEP pep,PetscInt cform)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,cform,2);
  ierr = PetscTryMethod(pep,"PEPLinearSetCompanionForm_C",(PEP,PetscInt),(pep,cform));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetCompanionForm_Linear"
static PetscErrorCode PEPLinearGetCompanionForm_Linear(PEP pep,PetscInt *cform)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  *cform = ctx->cform;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetCompanionForm"
/*@
   PEPLinearGetCompanionForm - Returns the number of the companion form that
   will be used for the linearization of a quadratic eigenproblem.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameter:
.  cform - the companion form number (1 or 2)

   Level: advanced

.seealso: PEPLinearSetCompanionForm()
@*/
PetscErrorCode PEPLinearGetCompanionForm(PEP pep,PetscInt *cform)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidIntPointer(cform,2);
  ierr = PetscTryMethod(pep,"PEPLinearGetCompanionForm_C",(PEP,PetscInt*),(pep,cform));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetExplicitMatrix_Linear"
static PetscErrorCode PEPLinearSetExplicitMatrix_Linear(PEP pep,PetscBool explicitmatrix)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ctx->explicitmatrix = explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetExplicitMatrix"
/*@
   PEPLinearSetExplicitMatrix - Indicate if the matrices A and B for the
   linearization of the problem must be built explicitly.

   Logically Collective on PEP

   Input Parameters:
+  pep      - polynomial eigenvalue solver
-  explicit - boolean flag indicating if the matrices are built explicitly

   Options Database Key:
.  -pep_linear_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: PEPLinearGetExplicitMatrix()
@*/
PetscErrorCode PEPLinearSetExplicitMatrix(PEP pep,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,explicitmatrix,2);
  ierr = PetscTryMethod(pep,"PEPLinearSetExplicitMatrix_C",(PEP,PetscBool),(pep,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetExplicitMatrix_Linear"
static PetscErrorCode PEPLinearGetExplicitMatrix_Linear(PEP pep,PetscBool *explicitmatrix)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  *explicitmatrix = ctx->explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetExplicitMatrix"
/*@
   PEPLinearGetExplicitMatrix - Returns the flag indicating if the matrices
   A and B for the linearization are built explicitly.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameter:
.  explicitmatrix - the mode flag

   Level: advanced

.seealso: PEPLinearSetExplicitMatrix()
@*/
PetscErrorCode PEPLinearGetExplicitMatrix(PEP pep,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(explicitmatrix,2);
  ierr = PetscTryMethod(pep,"PEPLinearGetExplicitMatrix_C",(PEP,PetscBool*),(pep,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetEPS_Linear"
static PetscErrorCode PEPLinearSetEPS_Linear(PEP pep,EPS eps)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ctx->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->eps);CHKERRQ(ierr);
  pep->state = PEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetEPS"
/*@
   PEPLinearSetEPS - Associate an eigensolver object (EPS) to the
   polynomial eigenvalue solver.

   Collective on PEP

   Input Parameters:
+  pep - polynomial eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: PEPLinearGetEPS()
@*/
PetscErrorCode PEPLinearSetEPS(PEP pep,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(pep,1,eps,2);
  ierr = PetscTryMethod(pep,"PEPLinearSetEPS_C",(PEP,EPS),(pep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetEPS_Linear"
static PetscErrorCode PEPLinearGetEPS_Linear(PEP pep,EPS *eps)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  ST             st;

  PetscFunctionBegin;
  if (!ctx->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)pep),&ctx->eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(ctx->eps,((PetscObject)pep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(ctx->eps,"pep_");CHKERRQ(ierr);
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetOptionsPrefix(st,((PetscObject)ctx->eps)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)pep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->eps);CHKERRQ(ierr);
    ierr = EPSMonitorSet(ctx->eps,EPSMonitor_Linear,pep,NULL);CHKERRQ(ierr);
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetEPS"
/*@
   PEPLinearGetEPS - Retrieve the eigensolver object (EPS) associated
   to the polynomial eigenvalue solver.

   Not Collective

   Input Parameter:
.  pep - polynomial eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: PEPLinearSetEPS()
@*/
PetscErrorCode PEPLinearGetEPS(PEP pep,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscTryMethod(pep,"PEPLinearGetEPS_C",(PEP,EPS*),(pep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_Linear"
PetscErrorCode PEPView_Linear(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
  ierr = PetscViewerASCIIPrintf(viewer,"  Linear: %s matrices\n",ctx->explicitmatrix? "explicit": "implicit");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Linear: %s companion form\n",ctx->cform==1? "1st": "2nd");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = EPSView(ctx->eps,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPReset_Linear"
PetscErrorCode PEPReset_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = EPSReset(ctx->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->B);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_Linear"
PetscErrorCode PEPDestroy_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetCompanionForm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetCompanionForm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_Linear"
PETSC_EXTERN PetscErrorCode PEPCreate_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  ctx->explicitmatrix = PETSC_TRUE;
  pep->data = (void*)ctx;

  pep->ops->solve                = PEPSolve_Linear;
  pep->ops->setup                = PEPSetUp_Linear;
  pep->ops->setfromoptions       = PEPSetFromOptions_Linear;
  pep->ops->destroy              = PEPDestroy_Linear;
  pep->ops->reset                = PEPReset_Linear;
  pep->ops->view                 = PEPView_Linear;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetCompanionForm_C",PEPLinearSetCompanionForm_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetCompanionForm_C",PEPLinearGetCompanionForm_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetEPS_C",PEPLinearSetEPS_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetEPS_C",PEPLinearGetEPS_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetExplicitMatrix_C",PEPLinearSetExplicitMatrix_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetExplicitMatrix_C",PEPLinearGetExplicitMatrix_Linear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

