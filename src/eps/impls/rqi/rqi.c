
/*                       
       This implements the Rayleigh Quotient Iteration method.
*/
#include "src/eps/epsimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_RQI"
static int EPSSetUp_RQI(EPS eps)
{
  int         ierr, N;
  PetscTruth  isSinv;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;

  ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&isSinv);CHKERRQ(ierr);
  if (!isSinv) SETERRQ(1,"A shift-and-invert ST must be specified in order to use RQI");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_RQI"
static int  EPSSolve_RQI(EPS eps)
{
  int         ierr, i, maxit=eps->max_it;
  Vec         v, w, y, e;
  PetscReal   relerr, tol=1.0/PetscSqrtScalar(eps->tol);
  PetscScalar theta, alpha, eta, rho;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->work[0];
  w = eps->work[1];
  e = eps->work[2];
  eps->nconv = 0;

  /* initial shift, rho_1 */
  ierr = STGetShift(eps->OP,&rho);CHKERRQ(ierr);

  /* w = B v, normalize v so that v^* w = 1 */
  ierr = VecCopy(eps->vec_initial,v);CHKERRQ(ierr);
  ierr = STApplyB(eps->OP,v,w);CHKERRQ(ierr);
  ierr = VecDot(w,v,&eta);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (eta<0.0) SETERRQ(1,"Negative value of eta");
#endif
  eta = PetscSqrtScalar(eta);
  if (eta==0.0) SETERRQ(1,"Zero value of eta");
  alpha = 1.0/eta;
  ierr = VecScale(&alpha,v);CHKERRQ(ierr);
  ierr = VecScale(&alpha,w);CHKERRQ(ierr);

  for (i=0;i<eps->ncv;i++) eps->eigi[i]=0.0;

  for (i=0;i<maxit;i++) {
    eps->its = i;

    /* y = OP w */
    ierr = STApplyNoB(eps->OP,w,y);CHKERRQ(ierr);

    /* theta = w^* y */
    ierr = VecDot(y,w,&theta);CHKERRQ(ierr);

    /* w = B y */
    ierr = STApplyB(eps->OP,y,w);CHKERRQ(ierr);

    /* eta = ||y||_B */
    ierr = VecDot(w,y,&eta);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (eta<0.0) SETERRQ(1,"Negative value of eta");
#endif
    eta = PetscSqrtScalar(eta);

    /* normalize y and w */
    ierr = VecCopy(y,v);CHKERRQ(ierr);
    alpha = 1.0/eta;
    ierr = VecScale(&alpha,v);CHKERRQ(ierr);
    ierr = VecScale(&alpha,w);CHKERRQ(ierr);

    /* rho_{k+1} = rho_{k} + theta/eta^2 */
    rho = rho + theta/(eta*eta);
    ierr = STSetShift(eps->OP,rho);CHKERRQ(ierr);

    /* if |theta| > tol^-1/2, stop */
    relerr = PetscAbsScalar(theta);
    eps->errest[eps->nconv] = 1/(relerr*relerr);
    eps->eigr[eps->nconv] = rho;
    EPSMonitor(eps,i+1,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1); 
    if (relerr>tol) {
      eps->nconv = eps->nconv + 1;
      break;
    }

  }

  if( i==maxit ) i--;
  eps->its = i+1;
  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_RQI"
int EPSCreate_RQI(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->setup                = EPSSetUp_RQI;
  eps->ops->solve                = EPSSolve_RQI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = 0;
  eps->ops->backtransform        = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
