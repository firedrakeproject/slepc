/*                       

   SLEPc eigensolver: "power"

   Method: Power Iteration

   Description:

       This solver implements the power iteration for finding dominant
       eigenpairs. It also includes the following well-known methods:
       - Inverse Iteration: when used in combination with shift-and-invert
         spectral transformation.
       - Rayleigh Quotient Iteration (RQI): also with shift-and-invert plus
         a variable shift.

   Algorithm:

       The implemented algorithm is the simple power iteration working with
       OP, the operator provided by the ST object. Converged eigenpairs are
       deflated by restriction, so that several eigenpairs can be sought. 
       Symmetry is preserved in symmetric definite pencils. See the SLEPc
       users guide for details.

       Variable shifts can be used. There are two possible strategies for
       updating shift: Rayleigh quotients and Wilkinson shifts.

   References:

       [1] B.N. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics in 
       Applied Mathematics (1998), pp 61-80 and 159-165.

   Last update: June 2004

*/
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
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
  int            N;
  PetscTruth     flg;
  STMatMode      mode;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (!eps->max_it) eps->max_it = PetscMax(2000,100*N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
    ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&flg);CHKERRQ(ierr);
    if (flg) 
      SETERRQ(PETSC_ERR_SUP,"Shift spectral transformation does not work with variable shifts");
    ierr = STGetMatMode(eps->OP,&mode);CHKERRQ(ierr); 
    if (mode == STMATMODE_INPLACE)
      SETERRQ(PETSC_ERR_SUP,"ST matrix mode inplace does not work with variable shifts");
  }
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPowerUpdateShift"
/*
   EPSPowerUpdateShift - Computes the new shift to be used in the next
   iteration of the power method. This function is invoked only when using
   the option of variable shifts (see EPSPowerSetShiftType).
*/
static PetscErrorCode EPSPowerUpdateShift(EPS eps,Vec v,PetscScalar* shift)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  Vec            e, w;
  Mat            A;
  PetscReal      norm, rt1, rt2, cs1;
  PetscScalar    alpha, alpha1, alpha2, beta1, sn1;

  PetscFunctionBegin;
  e = eps->work[0];
  w = eps->work[1];
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);

  /* compute the Rayleigh quotient R(v) assuming v is B-normalized */
  ierr = MatMult(A,v,e);CHKERRQ(ierr);
  ierr = VecDot(v,e,&alpha1);CHKERRQ(ierr);

  /* in the case of Wilkinson the shift is improved */
  if (power->shift_type == EPSPOWER_SHIFT_WILKINSON) {
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
    SETERRQ(PETSC_ERR_SUP,"LAEV2 - Lapack routine is unavailable.");
#endif 
    /* beta1 is the norm of the residual associated to R(v) */
    alpha = -alpha1;
    ierr = VecAXPY(&alpha,v,e);CHKERRQ(ierr);
    ierr = STNorm(eps->OP,e,&norm);CHKERRQ(ierr);
    beta1 = norm;
    
    /* alfa2 = (e'*A*e)/(beta1*beta1), where e is the residual */
    ierr = MatMult(A,e,w);CHKERRQ(ierr);
    ierr = VecDot(e,w,&alpha2);CHKERRQ(ierr);
    alpha2 = alpha2 / (beta1 * beta1);

    /* choose the eigenvalue of [alfa1 beta1; beta1 alfa2] closest to alpha1 */
    LAlaev2_(&alpha1,&beta1,&alpha2,&rt1,&rt2,&cs1,&sn1);
    if (PetscAbsScalar(rt1-alpha1) < PetscAbsScalar(rt2-alpha1)) {
      *shift = rt1;
    } else {
      *shift = rt2;
    }
  }
  else *shift = alpha1;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_POWER"
PetscErrorCode EPSSolve_POWER(EPS eps)
{
  PetscErrorCode ierr;
  EPS_POWER      *power = (EPS_POWER *)eps->data;
  int            i;
  Vec            v, y, e;
  PetscReal      relerr, norm;
  PetscScalar    theta, alpha, rho;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->AV[0];
  e = eps->work[0];

  ierr = VecCopy(eps->vec_initial,y);CHKERRQ(ierr);

  eps->nconv = 0;
  eps->its = 0;

  for (i=0;i<eps->ncv;i++) eps->eigi[i]=0.0;

  while (eps->its<eps->max_it) {

    eps->its = eps->its + 1;

    /* deflation of converged eigenvectors */
    ierr = EPSPurge(eps,y);

    /* v = y/||y||_B */
    ierr = VecCopy(y,v);CHKERRQ(ierr);
    ierr = STNorm(eps->OP,y,&norm);CHKERRQ(ierr);
    alpha = 1.0/norm;
    ierr = VecScale(&alpha,v);CHKERRQ(ierr);

    /* y = OP v */
    ierr = STApply(eps->OP,v,y);CHKERRQ(ierr);

    /* theta = (y,v)_B */
    ierr = STInnerProduct(eps->OP,y,v,&theta);CHKERRQ(ierr);

    /* compute residual norm */
    ierr = VecCopy(y,e);CHKERRQ(ierr);
    alpha = -theta;
    ierr = VecAXPY(&alpha,v,e);CHKERRQ(ierr);
    ierr = VecNorm(e,NORM_2,&relerr);CHKERRQ(ierr);
    relerr = relerr / PetscAbsScalar(theta);
    eps->errest[eps->nconv] = relerr;

    /* update eigenvalue and shift */
    if (power->shift_type != EPSPOWER_SHIFT_CONSTANT) {
        ierr = EPSPowerUpdateShift(eps,v,&rho);CHKERRQ(ierr);
        /* change the shift only if rho is not too close to an eigenvalue */
        if (relerr > 1000*eps->tol) {
          ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);
        }
        eps->eigr[eps->nconv] = rho;
    } else {
      eps->eigr[eps->nconv] = theta;
    }

    /* if ||y-theta v||_2 / |theta| < tol, accept eigenpair */
    if (relerr<eps->tol) {
      eps->nconv = eps->nconv + 1;
      if (eps->nconv==eps->nev) break;
      v = eps->V[eps->nconv];
    }

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1); 
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
  const char     *shift_list[3] = { "constant", "rayleigh", "wilkinson" };

  PetscFunctionBegin;
  ierr = PetscOptionsHead("POWER options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_power_shift_type","Shift type","EPSPowerSetShiftType",shift_list,3,shift_list[power->shift_type],(int*)&power->shift_type,&flg);CHKERRQ(ierr);
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

.seealso: EPSGetShiftType(), STSetShift()
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
/*@
   EPSPowerGetShiftType - Gets the type of shifts used during the power
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  shift - the type of shift

   Level: advanced

.seealso: EPSSetShiftType()
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
  PetscMemzero(power,sizeof(EPS_POWER));
  PetscLogObjectMemory(eps,sizeof(EPS_POWER));
  eps->data                      = (void *) power;
  eps->ops->solve                = EPSSolve_POWER;
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

