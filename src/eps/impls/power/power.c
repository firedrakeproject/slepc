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

   Last update: June 2005

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  EPSPowerShiftType shift_type;
} EPS_POWER;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_POWER"
PetscErrorCode EPSSetUp_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscInt       N;
  PetscTruth     flg;
  STMatMode      mode;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (!eps->max_it) eps->max_it = PetscMax(2000,100*N);
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
    ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&flg);CHKERRQ(ierr);
    if (!flg) 
      SETERRQ(PETSC_ERR_SUP,"Variable shifts only allowed in shift-and-invert ST");
    ierr = STGetMatMode(eps->OP,&mode);CHKERRQ(ierr); 
    if (mode == STMATMODE_INPLACE)
      SETERRQ(PETSC_ERR_SUP,"ST matrix mode inplace does not work with variable shifts");
  }
  if (eps->projection) {
     ierr = PetscInfo(eps,"Warning: projection type ignored\n");CHKERRQ(ierr);
  }
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  } else {
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_POWER"
PetscErrorCode EPSSolve_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscInt       i, nsv;
  Vec            v, y, e, *SV;
  Mat            A;
  PetscReal      relerr, norm, rt1, rt2, cs1, anorm;
  PetscScalar    theta, rho, delta, sigma, alpha2, beta1, sn1;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->AV[0];
  e = eps->work[0];

  /* prepare for selective orthogonalization of converged vectors */
  if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
    ierr = PetscMalloc(eps->nev*sizeof(Vec),&SV);CHKERRQ(ierr);
    for (i=0;i<eps->nds;i++) SV[i]=eps->DS[i];
    if (eps->nev>1) {
      ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatNorm(A,NORM_INFINITY,&anorm);CHKERRQ(ierr);
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

    if (power->shift_type == EPSPOWER_SHIFT_CONSTANT) { /* direct & inverse iteration */

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
      if (rho == 0.0) relerr = PETSC_MAX;
      else relerr = 1.0 / (norm*PetscAbsScalar(rho));

      /* approximate eigenvalue is the shift */
      eps->eigr[eps->nconv] = rho;

      /* compute new shift */
      if (relerr<eps->tol) {
        rho = sigma; /* if converged, restore original shift */ 
        ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);
      } else {
        rho = rho + theta/(delta*delta);  /* Rayleigh quotient R(v) */
        if (power->shift_type == EPSPOWER_SHIFT_WILKINSON) {
#if defined(SLEPC_MISSING_LAPACK_LAEV2)
          SETERRQ(PETSC_ERR_SUP,"LAEV2 - Lapack routine is unavailable.");
#else 
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
          LAPACKlaev2_(&rho,&beta1,&alpha2,&rt1,&rt2,&cs1,&sn1);
          if (PetscAbsScalar(rt1-rho) < PetscAbsScalar(rt2-rho)) rho = rt1;
          else rho = rt2;
#endif 
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
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1); 

    /* purge previously converged eigenvectors */
    if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
      nsv = eps->nds;
      for (i=0;i<eps->nconv;i++) {
        if(PetscAbsScalar(rho-eps->eigr[i])>eps->its*anorm/1000) SV[nsv++]=eps->V[i];
      }
      ierr = IPOrthogonalize(eps->ip,nsv,PETSC_NULL,SV,y,PETSC_NULL,&norm,PETSC_NULL,eps->work[1]);CHKERRQ(ierr);
    } else {
      ierr = IPOrthogonalize(eps->ip,eps->nds+eps->nconv,PETSC_NULL,eps->DSV,y,PETSC_NULL,&norm,PETSC_NULL,eps->work[1]);CHKERRQ(ierr);
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
	  PetscInfo(eps,"Unable to generate more start vectors\n");
	}
      }
    }

    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
  }

  if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
    ierr = PetscFree(SV);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_TS_POWER"
PetscErrorCode EPSSolve_TS_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  Vec            v, w, y, z, e;
  Mat            A;
  PetscReal      relerr, norm, rt1, rt2, cs1;
  PetscScalar    theta, alpha, beta, rho, delta, sigma, alpha2, beta1, sn1;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->AV[0];
  e = eps->work[0];
  w = eps->W[0];
  z = eps->AW[0];

  ierr = EPSGetStartVector(eps,0,v,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSGetLeftStartVector(eps,0,w);CHKERRQ(ierr);
  ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);    /* original shift */
  rho = sigma;

  while (eps->its<eps->max_it) {
    eps->its++;
    
    /* y = OP v, z = OP' w */
    ierr = STApply(eps->OP,v,y);CHKERRQ(ierr);
    ierr = STApplyTranspose(eps->OP,w,z);CHKERRQ(ierr);

    /* theta = (v,z)_B */
    ierr = IPInnerProduct(eps->ip,v,z,&theta);CHKERRQ(ierr);

    if (power->shift_type == EPSPOWER_SHIFT_CONSTANT) { /* direct & inverse iteration */

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
      if (alpha==0.0) SETERRQ(1,"Breakdown in two-sided Power/RQI");
      delta = PetscSqrtScalar(alpha);

      /* compute relative error */
      if (rho == 0.0) relerr = PETSC_MAX;
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
        if (power->shift_type == EPSPOWER_SHIFT_WILKINSON) {
#if defined(SLEPC_MISSING_LAPACK_LAEV2)
          SETERRQ(PETSC_ERR_SUP,"LAEV2 - Lapack routine is unavailable.");
#else 
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
          LAPACKlaev2_(&rho,&beta1,&alpha2,&rt1,&rt2,&cs1,&sn1);
          if (PetscAbsScalar(rt1-rho) < PetscAbsScalar(rt2-rho)) rho = rt1;
          else rho = rt2;
#endif 
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

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1); 
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest_left,eps->nconv+1); 

    /* purge previously converged eigenvectors */
    ierr = IPBiOrthogonalize(eps->ip,eps->nconv,eps->V,eps->W,z,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPBiOrthogonalize(eps->ip,eps->nconv,eps->W,eps->V,y,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* normalize so that (y,z)_B=1  */
    ierr = VecCopy(y,v);CHKERRQ(ierr);
    ierr = VecCopy(z,w);CHKERRQ(ierr);
    ierr = IPInnerProduct(eps->ip,y,z,&alpha);CHKERRQ(ierr);
    if (alpha==0.0) SETERRQ(1,"Breakdown in two-sided Power/RQI");
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
      ierr = EPSGetLeftStartVector(eps,eps->nconv,w);CHKERRQ(ierr);
    }
  }

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_POWER"
PetscErrorCode EPSBackTransform_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER *power = (EPS_POWER *)eps->data;

  PetscFunctionBegin;
  if (power->shift_type == EPSPOWER_SHIFT_CONSTANT) {
    ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_POWER"
PetscErrorCode EPSSetFromOptions_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscTruth     flg;
  PetscInt       i;
  const char     *shift_list[3] = { "constant", "rayleigh", "wilkinson" };

  PetscFunctionBegin;
  ierr = PetscOptionsHead("POWER options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_power_shift_type","Shift type","EPSPowerSetShiftType",shift_list,3,shift_list[power->shift_type],&i,&flg);CHKERRQ(ierr);
  if (flg ) power->shift_type = (EPSPowerShiftType)i;
  if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
    ierr = STSetType(eps->OP,STSINV);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPowerSetShiftType_POWER"
PetscErrorCode EPSPowerSetShiftType_POWER(EPS eps,EPSPowerShiftType shift)
{
  EPS_POWER *power = (EPS_POWER *)eps->data;

  PetscFunctionBegin;
  switch (shift) {
    case EPSPOWER_SHIFT_CONSTANT:
    case EPSPOWER_SHIFT_RAYLEIGH:
    case EPSPOWER_SHIFT_WILKINSON:
      power->shift_type = shift;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid shift type");
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

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  shift - the type of shift

   Options Database Key:
.  -eps_power_shift_type - Sets the shift type (either 'constant' or 
                           'rayleigh' or 'wilkinson')

   Notes:
   By default, shifts are constant (EPSPOWER_SHIFT_CONSTANT) and the iteration
   is the simple power method (or inverse iteration if a shift-and-invert
   transformation is being used).

   A variable shift can be specified (EPSPOWER_SHIFT_RAYLEIGH or
   EPSPOWER_SHIFT_WILKINSON). In this case, the iteration behaves rather like
   a cubic converging method as RQI. See the users manual for details.
   
   Level: advanced

.seealso: EPSPowerGetShiftType(), STSetShift(), EPSPowerShiftType
@*/
PetscErrorCode EPSPowerSetShiftType(EPS eps,EPSPowerShiftType shift)
{
  PetscErrorCode ierr, (*f)(EPS,EPSPowerShiftType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPowerSetShiftType_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,shift);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPowerGetShiftType_POWER"
PetscErrorCode EPSPowerGetShiftType_POWER(EPS eps,EPSPowerShiftType *shift)
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

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  shift - the type of shift

   Level: advanced

.seealso: EPSPowerSetShiftType(), EPSPowerShiftType
@*/
PetscErrorCode EPSPowerGetShiftType(EPS eps,EPSPowerShiftType *shift)
{
  PetscErrorCode ierr, (*f)(EPS,EPSPowerShiftType*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPowerGetShiftType_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,shift);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_POWER"
PetscErrorCode EPSView_POWER(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  PetscTruth     isascii;
  const char     *shift_list[3] = { "constant", "rayleigh", "wilkinson" };

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSPOWER",((PetscObject)viewer)->type_name);
  }  
  ierr = PetscViewerASCIIPrintf(viewer,"shift type: %s\n",shift_list[power->shift_type]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_POWER"
PetscErrorCode EPSCreate_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_POWER,&power);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_POWER));
  eps->data                      = (void *) power;
  eps->ops->solve                = EPSSolve_POWER;
  eps->ops->solvets              = EPSSolve_TS_POWER;
  eps->ops->setup                = EPSSetUp_POWER;
  eps->ops->setfromoptions       = EPSSetFromOptions_POWER;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_POWER;
  eps->ops->backtransform        = EPSBackTransform_POWER;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  power->shift_type              = EPSPOWER_SHIFT_CONSTANT;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPowerSetShiftType_C","EPSPowerSetShiftType_POWER",EPSPowerSetShiftType_POWER);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPowerGetShiftType_C","EPSPowerGetShiftType_POWER",EPSPowerGetShiftType_POWER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

