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

       [1] "Single Vector Iteration Methods in SLEPc", SLEPc Technical Report STR-2, 
           available at http://www.grycap.upv.es/slepc.

   Last update: Feb 2009

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode EPSSolve_Power(EPS);
PetscErrorCode EPSSolve_TS_Power(EPS);

typedef struct {
  EPSPowerShiftType shift_type;
} EPS_POWER;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_Power"
PetscErrorCode EPSSetUp_Power(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscBool      flg;
  STMatMode      mode;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(2000,100*eps->n);
  if (!eps->which) { ierr = EPSDefaultSetWhich(eps);CHKERRQ(ierr); }
  if (eps->which!=EPS_LARGEST_MAGNITUDE && eps->which !=EPS_TARGET_MAGNITUDE) SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");
  if (power->shift_type != EPS_POWER_SHIFT_CONSTANT) {
    ierr = PetscObjectTypeCompareAny((PetscObject)eps->OP,&flg,STSINVERT,STCAYLEY,"");CHKERRQ(ierr);
    if (!flg) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Variable shifts only allowed in shift-and-invert or Cayley ST");
    ierr = STGetMatMode(eps->OP,&mode);CHKERRQ(ierr); 
    if (mode == ST_MATMODE_INPLACE) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"ST matrix mode inplace does not work with variable shifts");
  }
  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  if (eps->balance!=EPS_BALANCE_NONE) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Balancing not supported in this solver");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->leftvecs) {
    ierr = EPSDefaultGetWork(eps,3);CHKERRQ(ierr);
  } else {
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  }

  /* dispatch solve method */
  if (eps->leftvecs) eps->ops->solve = EPSSolve_TS_Power;
  else eps->ops->solve = EPSSolve_Power;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_Power"
PetscErrorCode EPSSolve_Power(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_LAEV2)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LAEV2 - Lapack routine is unavailable.");
#else 
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscInt       i;
  Vec            v,y,e;
  Mat            A;
  PetscReal      relerr,norm,rt1,rt2,cs1,anorm;
  PetscScalar    theta,rho,delta,sigma,alpha2,beta1,sn1;
  PetscBool      breakdown,*select = PETSC_NULL,hasnorm;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->work[1];
  e = eps->work[0];

  /* prepare for selective orthogonalization of converged vectors */
  if (power->shift_type != EPS_POWER_SHIFT_CONSTANT && eps->nev>1) {
    ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatHasOperation(A,MATOP_NORM,&hasnorm);CHKERRQ(ierr);
    if (hasnorm) {
      ierr = MatNorm(A,NORM_INFINITY,&anorm);CHKERRQ(ierr);
      ierr = PetscMalloc(eps->nev*sizeof(PetscBool),&select);CHKERRQ(ierr);
    }
  }

  ierr = EPSGetStartVector(eps,0,v,PETSC_NULL);CHKERRQ(ierr);
  ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);    /* original shift */
  rho = sigma;

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its = eps->its + 1;

    /* y = OP v */
    ierr = STApply(eps->OP,v,y);CHKERRQ(ierr);

    /* theta = (v,y)_B */
    ierr = IPInnerProduct(eps->ip,v,y,&theta);CHKERRQ(ierr);

    if (power->shift_type == EPS_POWER_SHIFT_CONSTANT) { /* direct & inverse iteration */

      /* approximate eigenvalue is the Rayleigh quotient */
      eps->eigr[eps->nconv] = theta;

      /* compute relative error as ||y-theta v||_2/|theta| */
      ierr = VecCopy(y,e);CHKERRQ(ierr);
      ierr = VecAXPY(e,-theta,v);CHKERRQ(ierr);
      ierr = VecNorm(e,NORM_2,&norm);CHKERRQ(ierr);
      relerr = norm / PetscAbsScalar(theta);

    } else {  /* RQI */

      /* delta = ||y||_B */
      ierr = IPNorm(eps->ip,y,&norm);CHKERRQ(ierr);
      delta = norm;

      /* compute relative error */
      if (rho == 0.0) relerr = PETSC_MAX_REAL;
      else relerr = 1.0 / (norm*PetscAbsScalar(rho));

      /* approximate eigenvalue is the shift */
      eps->eigr[eps->nconv] = rho;

      /* compute new shift */
      if (relerr<eps->tol) {
        rho = sigma; /* if converged, restore original shift */ 
        ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);
      } else {
        rho = rho + theta/(delta*delta);  /* Rayleigh quotient R(v) */
        if (power->shift_type == EPS_POWER_SHIFT_WILKINSON) {
          /* beta1 is the norm of the residual associated to R(v) */
          ierr = VecAXPY(v,-theta/(delta*delta),y);CHKERRQ(ierr);
          ierr = VecScale(v,1.0/delta);CHKERRQ(ierr);
          ierr = IPNorm(eps->ip,v,&norm);CHKERRQ(ierr);
          beta1 = norm;
    
          /* alpha2 = (e'*A*e)/(beta1*beta1), where e is the residual */
          ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatMult(A,v,e);CHKERRQ(ierr);
          ierr = VecDot(v,e,&alpha2);CHKERRQ(ierr);
          alpha2 = alpha2 / (beta1 * beta1);

          /* choose the eigenvalue of [rho beta1; beta1 alpha2] closest to rho */
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          LAPACKlaev2_(&rho,&beta1,&alpha2,&rt1,&rt2,&cs1,&sn1);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          if (PetscAbsScalar(rt1-rho) < PetscAbsScalar(rt2-rho)) rho = rt1;
          else rho = rt2;
        }
        /* update operator according to new shift */
        PetscPushErrorHandler(PetscIgnoreErrorHandler,PETSC_NULL);
        ierr = STSetShift(eps->OP,rho);
        PetscPopErrorHandler();
        if (ierr) {
          eps->eigr[eps->nconv] = rho;
          relerr = PETSC_MACHINE_EPSILON;
          rho = sigma;
          ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);
        }
      }
    }

    eps->errest[eps->nconv] = relerr;
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1);CHKERRQ(ierr);

    /* purge previously converged eigenvectors */
    if (select) {
      for (i=0;i<eps->nconv;i++) {
        if(PetscAbsScalar(rho-eps->eigr[i])>eps->its*anorm/1000) select[i] = PETSC_TRUE;
        else select[i] = PETSC_FALSE;
      }
      ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,eps->nconv,select,eps->V,y,PETSC_NULL,&norm,PETSC_NULL);CHKERRQ(ierr);
    } else {
      ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,eps->nconv,PETSC_NULL,eps->V,y,PETSC_NULL,&norm,PETSC_NULL);CHKERRQ(ierr);
    }

    /* v = y/||y||_B */
    ierr = VecCopy(y,v);CHKERRQ(ierr);
    ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);

    /* if relerr<tol, accept eigenpair */
    if (relerr<eps->tol) {
      eps->nconv = eps->nconv + 1;
      if (eps->nconv==eps->nev) eps->reason = EPS_CONVERGED_TOL;
      else {
        v = eps->V[eps->nconv];
        ierr = EPSGetStartVector(eps,eps->nconv,v,&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      }
    }
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
  }
  ierr = PetscFree(select);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_TS_Power"
PetscErrorCode EPSSolve_TS_Power(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_LAEV2)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LAEV2 - Lapack routine is unavailable.");
#else 
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  Vec            v,w,y,z,e;
  Mat            A;
  PetscReal      relerr,norm,rt1,rt2,cs1;
  PetscScalar    theta,alpha,beta,rho,delta,sigma,alpha2,beta1,sn1;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->work[1];
  e = eps->work[0];
  w = eps->W[0];
  z = eps->work[2];

  ierr = EPSGetStartVector(eps,0,v,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSGetStartVectorLeft(eps,0,w,PETSC_NULL);CHKERRQ(ierr);
  ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);    /* original shift */
  rho = sigma;

  while (eps->its<eps->max_it) {
    eps->its++;
    
    /* y = OP v, z = OP' w */
    ierr = STApply(eps->OP,v,y);CHKERRQ(ierr);
    ierr = STApplyTranspose(eps->OP,w,z);CHKERRQ(ierr);

    /* theta = (v,z)_B */
    ierr = IPInnerProduct(eps->ip,v,z,&theta);CHKERRQ(ierr);

    if (power->shift_type == EPS_POWER_SHIFT_CONSTANT) { /* direct & inverse iteration */

      /* approximate eigenvalue is the Rayleigh quotient */
      eps->eigr[eps->nconv] = theta;

      /* compute relative errors (right and left) */
      ierr = VecCopy(y,e);CHKERRQ(ierr);
      ierr = VecAXPY(e,-theta,v);CHKERRQ(ierr);
      ierr = VecNorm(e,NORM_2,&norm);CHKERRQ(ierr);
      relerr = norm / PetscAbsScalar(theta);
      eps->errest[eps->nconv] = relerr;
      ierr = VecCopy(z,e);CHKERRQ(ierr);
      ierr = VecAXPY(e,-theta,w);CHKERRQ(ierr);
      ierr = VecNorm(e,NORM_2,&norm);CHKERRQ(ierr);
      relerr = norm / PetscAbsScalar(theta);
      eps->errest_left[eps->nconv] = relerr;

    } else {  /* RQI */

      /* delta = sqrt(y,z)_B */
      ierr = IPInnerProduct(eps->ip,y,z,&alpha);CHKERRQ(ierr);
      if (alpha==0.0) SETERRQ(((PetscObject)eps)->comm,1,"Breakdown in two-sided Power/RQI");
      delta = PetscSqrtScalar(alpha);

      /* compute relative error */
      if (rho == 0.0) relerr = PETSC_MAX_REAL;
      else relerr = 1.0 / (PetscAbsScalar(delta*rho));
      eps->errest[eps->nconv] = relerr;
      eps->errest_left[eps->nconv] = relerr;

      /* approximate eigenvalue is the shift */
      eps->eigr[eps->nconv] = rho;

      /* compute new shift */
      if (eps->errest[eps->nconv]<eps->tol && eps->errest_left[eps->nconv]<eps->tol) {
        rho = sigma; /* if converged, restore original shift */ 
        ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);
      } else {
        rho = rho + theta/(delta*delta);  /* Rayleigh quotient R(v,w) */
        if (power->shift_type == EPS_POWER_SHIFT_WILKINSON) {
          /* beta1 is the norm of the residual associated to R(v,w) */
          ierr = VecAXPY(v,-theta/(delta*delta),y);CHKERRQ(ierr);
          ierr = VecScale(v,1.0/delta);CHKERRQ(ierr);
          ierr = IPNorm(eps->ip,v,&norm);CHKERRQ(ierr);
          beta1 = norm;
    
          /* alpha2 = (e'*A*e)/(beta1*beta1), where e is the residual */
          ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatMult(A,v,e);CHKERRQ(ierr);
          ierr = VecDot(v,e,&alpha2);CHKERRQ(ierr);
          alpha2 = alpha2 / (beta1 * beta1);

          /* choose the eigenvalue of [rho beta1; beta1 alpha2] closest to rho */
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          LAPACKlaev2_(&rho,&beta1,&alpha2,&rt1,&rt2,&cs1,&sn1);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          if (PetscAbsScalar(rt1-rho) < PetscAbsScalar(rt2-rho)) rho = rt1;
          else rho = rt2;
        }
        /* update operator according to new shift */
        PetscPushErrorHandler(PetscIgnoreErrorHandler,PETSC_NULL);
        ierr = STSetShift(eps->OP,rho);
        PetscPopErrorHandler();
        if (ierr) {
          eps->eigr[eps->nconv] = rho;
          eps->errest[eps->nconv] = PETSC_MACHINE_EPSILON;
          eps->errest_left[eps->nconv] = PETSC_MACHINE_EPSILON;
          rho = sigma;
          ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);
        }
      }
    }

    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1);CHKERRQ(ierr);
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest_left,eps->nconv+1);CHKERRQ(ierr);

    /* purge previously converged eigenvectors */
    ierr = IPBiOrthogonalize(eps->ip,eps->nconv,eps->V,eps->W,z,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPBiOrthogonalize(eps->ip,eps->nconv,eps->W,eps->V,y,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* normalize so that (y,z)_B=1  */
    ierr = VecCopy(y,v);CHKERRQ(ierr);
    ierr = VecCopy(z,w);CHKERRQ(ierr);
    ierr = IPInnerProduct(eps->ip,y,z,&alpha);CHKERRQ(ierr);
    if (alpha==0.0) SETERRQ(((PetscObject)eps)->comm,1,"Breakdown in two-sided Power/RQI");
    delta = PetscSqrtScalar(PetscAbsScalar(alpha));
    beta = 1.0/PetscConj(alpha/delta);
    delta = 1.0/delta;
    ierr = VecScale(w,beta);CHKERRQ(ierr);
    ierr = VecScale(v,delta);CHKERRQ(ierr);

    /* if relerr<tol (both right and left), accept eigenpair */
    if (eps->errest[eps->nconv]<eps->tol && eps->errest_left[eps->nconv]<eps->tol) {
      eps->nconv = eps->nconv + 1;
      if (eps->nconv==eps->nev) break;
      v = eps->V[eps->nconv];
      ierr = EPSGetStartVector(eps,eps->nconv,v,PETSC_NULL);CHKERRQ(ierr);
      w = eps->W[eps->nconv];
      ierr = EPSGetStartVectorLeft(eps,eps->nconv,w,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  if (eps->nconv == eps->nev) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_Power"
PetscErrorCode EPSBackTransform_Power(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;

  PetscFunctionBegin;
  if (power->shift_type == EPS_POWER_SHIFT_CONSTANT) {
    ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_Power"
PetscErrorCode EPSSetFromOptions_Power(EPS eps)
{
  PetscErrorCode    ierr;
  EPS_POWER         *power = (EPS_POWER *)eps->data;
  PetscBool         flg;
  EPSPowerShiftType shift;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS Power Options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-eps_power_shift_type","Shift type","EPSPowerSetShiftType",EPSPowerShiftTypes,(PetscEnum)power->shift_type,(PetscEnum*)&shift,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSPowerSetShiftType(eps,shift);CHKERRQ(ierr); }
  if (power->shift_type != EPS_POWER_SHIFT_CONSTANT) {
    ierr = STSetType(eps->OP,STSINVERT);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPowerSetShiftType_Power"
PetscErrorCode EPSPowerSetShiftType_Power(EPS eps,EPSPowerShiftType shift)
{
  EPS_POWER *power = (EPS_POWER *)eps->data;

  PetscFunctionBegin;
  switch (shift) {
    case EPS_POWER_SHIFT_CONSTANT:
    case EPS_POWER_SHIFT_RAYLEIGH:
    case EPS_POWER_SHIFT_WILKINSON:
      power->shift_type = shift;
      break;
    default:
      SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid shift type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPowerSetShiftType"
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
   a cubic converging method as RQI. See the users manual for details.
   
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPowerGetShiftType_Power"
PetscErrorCode EPSPowerGetShiftType_Power(EPS eps,EPSPowerShiftType *shift)
{
  EPS_POWER  *power = (EPS_POWER *)eps->data;

  PetscFunctionBegin;
  *shift = power->shift_type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPowerGetShiftType"
/*@C
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
  ierr = PetscTryMethod(eps,"EPSPowerGetShiftType_C",(EPS,EPSPowerShiftType*),(eps,shift));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_Power"
PetscErrorCode EPSDestroy_Power(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPowerSetShiftType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPowerGetShiftType_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_Power"
PetscErrorCode EPSView_Power(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ1(((PetscObject)eps)->comm,1,"Viewer type %s not supported for EPS Power",((PetscObject)viewer)->type_name);
  ierr = PetscViewerASCIIPrintf(viewer,"  Power: %s shifts\n",EPSPowerShiftTypes[power->shift_type]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_Power"
PetscErrorCode EPSCreate_Power(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_POWER,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_Power;
  eps->ops->setfromoptions       = EPSSetFromOptions_Power;
  eps->ops->destroy              = EPSDestroy_Power;
  eps->ops->reset                = EPSReset_Default;
  eps->ops->view                 = EPSView_Power;
  eps->ops->backtransform        = EPSBackTransform_Power;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPowerSetShiftType_C","EPSPowerSetShiftType_Power",EPSPowerSetShiftType_Power);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPowerGetShiftType_C","EPSPowerGetShiftType_Power",EPSPowerGetShiftType_Power);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

