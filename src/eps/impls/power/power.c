/*

   SLEPc eigensolver: "power"

   Method: Power Iteration

   Algorithm:

       This solver implements the power iteration for finding dominant
       eigenpairs. It also includes the following well-known methods:
       - Inverse Iteration: when used in combination with shift-and-invert
         spectral transformation.
       - Rayleigh Quotient Iteration (RQI): also with shift-and-invert plus
         a variable shift.

   References:

       [1] "Single Vector Iteration Methods in SLEPc", SLEPc Technical Report
           STR-2, available at http://slepc.upv.es.

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

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  EPSPowerShiftType shift_type;
} EPS_POWER;

PetscErrorCode EPSSetUp_Power(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER*)eps->data;
  PetscBool      flg,istrivial;
  STMatMode      mode;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must be at least nev");
  } else eps->ncv = eps->nev;
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(1000*eps->nev,100*eps->n);
  if (!eps->which) { ierr = EPSSetWhichEigenpairs_Default(eps);CHKERRQ(ierr); }
  if (eps->which!=EPS_LARGEST_MAGNITUDE && eps->which !=EPS_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
  if (power->shift_type != EPS_POWER_SHIFT_CONSTANT) {
    ierr = PetscObjectTypeCompareAny((PetscObject)eps->st,&flg,STSINVERT,STCAYLEY,"");CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Variable shifts only allowed in shift-and-invert or Cayley ST");
    ierr = STGetMatMode(eps->st,&mode);CHKERRQ(ierr);
    if (mode == ST_MATMODE_INPLACE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"ST matrix mode inplace does not work with variable shifts");
  }
  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  if (eps->balance!=EPS_BALANCE_NONE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Balancing not supported in this solver");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->nev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Normalize a vector x with respect to a given norm as well as the
   sign of the first entry.
*/
static PetscErrorCode Normalize(Vec x,PetscReal norm)
{
  PetscErrorCode    ierr;
  PetscScalar       alpha = 1.0;
  PetscMPIInt       rank;
  PetscReal         absv;
  const PetscScalar *xx;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)x),&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    absv = PetscAbsScalar(*xx);
    if (absv>10*PETSC_MACHINE_EPSILON) alpha = *xx/absv;
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(&alpha,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)x));CHKERRQ(ierr);
  alpha *= norm;
  ierr = VecScale(x,1.0/alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_Power(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_LAEV2)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LAEV2 - Lapack routine is unavailable");
#else
  PetscErrorCode     ierr;
  EPS_POWER          *power = (EPS_POWER*)eps->data;
  PetscInt           k,ld;
  Vec                v,y,e;
  Mat                A;
  KSP                ksp;
  PetscReal          relerr,norm,norm1,rt1,rt2,cs1;
  PetscScalar        theta,rho,delta,sigma,alpha2,beta1,sn1,*T;
  PetscBool          breakdown;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  y = eps->work[1];
  e = eps->work[0];

  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  ierr = STGetShift(eps->st,&sigma);CHKERRQ(ierr);    /* original shift */
  rho = sigma;

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    k = eps->nconv;

    /* y = OP v */
    ierr = BVGetColumn(eps->V,k,&v);CHKERRQ(ierr);
    ierr = STApply(eps->st,v,y);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,k,&v);CHKERRQ(ierr);

    /* purge previously converged eigenvectors */
    ierr = DSGetArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,0,k);CHKERRQ(ierr);
    ierr = BVOrthogonalizeVec(eps->V,y,T+k*ld,&norm,NULL);CHKERRQ(ierr);

    /* theta = (v,y)_B */
    ierr = BVSetActiveColumns(eps->V,k,k+1);CHKERRQ(ierr);
    ierr = BVDotVec(eps->V,y,&theta);CHKERRQ(ierr);
    T[k+k*ld] = theta;
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&T);CHKERRQ(ierr);

    if (power->shift_type == EPS_POWER_SHIFT_CONSTANT) { /* direct & inverse iteration */

      /* approximate eigenvalue is the Rayleigh quotient */
      eps->eigr[eps->nconv] = theta;

      /* compute relative error as ||y-theta v||_2/|theta| */
      ierr = VecCopy(y,e);CHKERRQ(ierr);
      ierr = BVGetColumn(eps->V,k,&v);CHKERRQ(ierr);
      ierr = VecAXPY(e,-theta,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,k,&v);CHKERRQ(ierr);
      ierr = VecNorm(e,NORM_2,&relerr);CHKERRQ(ierr);
      relerr /= PetscAbsScalar(theta);

    } else {  /* RQI */

      /* delta = ||y||_B */
      delta = norm;

      /* compute relative error */
      if (rho == 0.0) relerr = PETSC_MAX_REAL;
      else relerr = 1.0 / (norm*PetscAbsScalar(rho));

      /* approximate eigenvalue is the shift */
      eps->eigr[eps->nconv] = rho;

      /* compute new shift */
      if (relerr<eps->tol) {
        rho = sigma;  /* if converged, restore original shift */
        ierr = STSetShift(eps->st,rho);CHKERRQ(ierr);
      } else {
        rho = rho + theta/(delta*delta);  /* Rayleigh quotient R(v) */
        if (power->shift_type == EPS_POWER_SHIFT_WILKINSON) {
          /* beta1 is the norm of the residual associated with R(v) */
          ierr = BVGetColumn(eps->V,k,&v);CHKERRQ(ierr);
          ierr = VecAXPY(v,-theta/(delta*delta),y);CHKERRQ(ierr);
          ierr = BVRestoreColumn(eps->V,k,&v);CHKERRQ(ierr);
          ierr = BVScaleColumn(eps->V,k,1.0/delta);CHKERRQ(ierr);
          ierr = BVNormColumn(eps->V,k,NORM_2,&norm1);CHKERRQ(ierr);
          beta1 = norm1;

          /* alpha2 = (e'*A*e)/(beta1*beta1), where e is the residual */
          ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
          ierr = BVGetColumn(eps->V,k,&v);CHKERRQ(ierr);
          ierr = MatMult(A,v,e);CHKERRQ(ierr);
          ierr = VecDot(v,e,&alpha2);CHKERRQ(ierr);
          ierr = BVRestoreColumn(eps->V,k,&v);CHKERRQ(ierr);
          alpha2 = alpha2 / (beta1 * beta1);

          /* choose the eigenvalue of [rho beta1; beta1 alpha2] closest to rho */
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKlaev2",LAPACKlaev2_(&rho,&beta1,&alpha2,&rt1,&rt2,&cs1,&sn1));
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          if (PetscAbsScalar(rt1-rho) < PetscAbsScalar(rt2-rho)) rho = rt1;
          else rho = rt2;
        }
        /* update operator according to new shift */
        ierr = KSPSetErrorIfNotConverged(ksp,PETSC_FALSE);CHKERRQ(ierr);
        ierr = STSetShift(eps->st,rho);CHKERRQ(ierr);
        ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
        if (reason) {
          ierr = PetscInfo(eps,"Factorization failed, repeat with a perturbed shift\n");CHKERRQ(ierr);
          rho *= 1+PETSC_MACHINE_EPSILON;
          ierr = STSetShift(eps->st,rho);CHKERRQ(ierr);
          ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
          if (reason) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"Second factorization failed");
        }
        ierr = KSPSetErrorIfNotConverged(ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    eps->errest[eps->nconv] = relerr;

    /* normalize */
    ierr = Normalize(y,norm);CHKERRQ(ierr);
    ierr = BVInsertVec(eps->V,k,y);CHKERRQ(ierr);

    /* if relerr<tol, accept eigenpair */
    if (relerr<eps->tol) {
      eps->nconv = eps->nconv + 1;
      if (eps->nconv<eps->nev) {
        ierr = EPSGetStartVector(eps,eps->nconv,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          break;
        }
      }
    }
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1);CHKERRQ(ierr);
    ierr = (*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx);CHKERRQ(ierr);
  }

  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

PetscErrorCode EPSBackTransform_Power(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER*)eps->data;

  PetscFunctionBegin;
  if (power->shift_type == EPS_POWER_SHIFT_CONSTANT) {
    ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_Power(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode    ierr;
  EPS_POWER         *power = (EPS_POWER*)eps->data;
  PetscBool         flg;
  EPSPowerShiftType shift;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS Power Options");CHKERRQ(ierr);

    ierr = PetscOptionsEnum("-eps_power_shift_type","Shift type","EPSPowerSetShiftType",EPSPowerShiftTypes,(PetscEnum)power->shift_type,(PetscEnum*)&shift,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSPowerSetShiftType(eps,shift);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSPowerSetShiftType_Power(EPS eps,EPSPowerShiftType shift)
{
  EPS_POWER *power = (EPS_POWER*)eps->data;

  PetscFunctionBegin;
  switch (shift) {
    case EPS_POWER_SHIFT_CONSTANT:
    case EPS_POWER_SHIFT_RAYLEIGH:
    case EPS_POWER_SHIFT_WILKINSON:
      power->shift_type = shift;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid shift type");
  }
  PetscFunctionReturn(0);
}

/*@
   EPSPowerSetShiftType - Sets the type of shifts used during the power
   iteration. This can be used to emulate the Rayleigh Quotient Iteration
   (RQI) method.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  shift - the type of shift

   Options Database Key:
.  -eps_power_shift_type - Sets the shift type (either 'constant' or
                           'rayleigh' or 'wilkinson')

   Notes:
   By default, shifts are constant (EPS_POWER_SHIFT_CONSTANT) and the iteration
   is the simple power method (or inverse iteration if a shift-and-invert
   transformation is being used).

   A variable shift can be specified (EPS_POWER_SHIFT_RAYLEIGH or
   EPS_POWER_SHIFT_WILKINSON). In this case, the iteration behaves rather like
   a cubic converging method such as RQI.

   Level: advanced

.seealso: EPSPowerGetShiftType(), STSetShift(), EPSPowerShiftType
@*/
PetscErrorCode EPSPowerSetShiftType(EPS eps,EPSPowerShiftType shift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,shift,2);
  ierr = PetscTryMethod(eps,"EPSPowerSetShiftType_C",(EPS,EPSPowerShiftType),(eps,shift));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSPowerGetShiftType_Power(EPS eps,EPSPowerShiftType *shift)
{
  EPS_POWER *power = (EPS_POWER*)eps->data;

  PetscFunctionBegin;
  *shift = power->shift_type;
  PetscFunctionReturn(0);
}

/*@
   EPSPowerGetShiftType - Gets the type of shifts used during the power
   iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  shift - the type of shift

   Level: advanced

.seealso: EPSPowerSetShiftType(), EPSPowerShiftType
@*/
PetscErrorCode EPSPowerGetShiftType(EPS eps,EPSPowerShiftType *shift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(shift,2);
  ierr = PetscUseMethod(eps,"EPSPowerGetShiftType_C",(EPS,EPSPowerShiftType*),(eps,shift));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_Power(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPowerSetShiftType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPowerGetShiftType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_Power(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Power: %s shifts\n",EPSPowerShiftTypes[power->shift_type]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EPSCreate_Power(EPS eps)
{
  EPS_POWER      *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->solve          = EPSSolve_Power;
  eps->ops->setup          = EPSSetUp_Power;
  eps->ops->setfromoptions = EPSSetFromOptions_Power;
  eps->ops->destroy        = EPSDestroy_Power;
  eps->ops->view           = EPSView_Power;
  eps->ops->backtransform  = EPSBackTransform_Power;
  eps->ops->computevectors = EPSComputeVectors_Schur;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPowerSetShiftType_C",EPSPowerSetShiftType_Power);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPowerGetShiftType_C",EPSPowerGetShiftType_Power);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

