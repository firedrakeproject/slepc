
/*                       
       This implements the power iteration for finding the eigenpair
       corresponding to the eigenvalue with largest magnitude.
*/
#include "src/eps/epsimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_POWER"
static int EPSSetUp_POWER(EPS eps)
{
  int      ierr;
  
  PetscFunctionBegin;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSDefaultGetWork(eps,3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDefaults_POWER"
static int EPSSetDefaults_POWER(EPS eps)
{
  int      ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (!eps->max_it) eps->max_it = PetscMax(2000,100*N);
  if (!eps->tol) eps->tol = 1.e-7;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_POWER"
static int  EPSSolve_POWER(EPS eps)
{
  int         ierr, i, k, maxit=eps->max_it;
  Vec         v, w, y, e;
  PetscReal   relerr, norm, tol=eps->tol;
  PetscScalar theta, alpha, eta;
  PetscTruth  isSinv;

  PetscFunctionBegin;
  v = eps->V[0];
  y = eps->work[0];
  w = eps->work[1];
  e = eps->work[2];

  ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&isSinv);CHKERRQ(ierr);

  ierr = VecCopy(eps->vec_initial,y);CHKERRQ(ierr);

  eps->nconv = 0;
  eps->its = 0;

  while (eps->its<maxit) {

    eps->its = eps->its + 1;

    if (isSinv) {
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
      if (eta==0.0) SETERRQ(1,"Zero value of eta");
      alpha = 1.0/eta;
      ierr = VecScale(&alpha,v);CHKERRQ(ierr);
      ierr = VecScale(&alpha,w);CHKERRQ(ierr);

      /* y = OP w */
      ierr = STApplyNoB(eps->OP,w,y);CHKERRQ(ierr);

      /* deflation of converged eigenvectors */
      if (eps->nconv>0) {
        ierr = (*eps->orthog)(eps,eps->nconv,y,PETSC_NULL,&norm);CHKERRQ(ierr);
      }

      /* theta = w^* y */
      ierr = VecDot(y,w,&theta);CHKERRQ(ierr);
    }
    else {
      /* v = y/||y||_2 */
      ierr = VecCopy(y,v);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
      alpha = 1.0/norm;
      ierr = VecScale(&alpha,v);CHKERRQ(ierr);

      /* y = OP v */
      ierr = STApply(eps->OP,v,y);CHKERRQ(ierr);

      /* deflation of converged eigenvectors */
      if (eps->nconv>0) {
        ierr = (*eps->orthog)(eps,eps->nconv,y,PETSC_NULL,&norm);CHKERRQ(ierr);
      }

      /* theta = v^* y */
      ierr = VecDot(y,v,&theta);CHKERRQ(ierr);
    }

    /* if ||y-theta v||_2 / |theta| < tol, accept */
    ierr = VecCopy(y,e);CHKERRQ(ierr);
    alpha = -theta;
    ierr = VecAXPY(&alpha,v,e);CHKERRQ(ierr);
    ierr = VecNorm(e,NORM_2,&relerr);CHKERRQ(ierr);
    relerr = relerr / PetscAbsScalar(theta);
    eps->errest[eps->nconv] = relerr;
    eps->eigr[eps->nconv] = theta;

    if (relerr<tol) {
      if(isSinv) {
        ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
        alpha = 1.0/norm;
        ierr = VecScale(&alpha,v);CHKERRQ(ierr);
      }
      eps->nconv = eps->nconv + 1;
      if (eps->nconv==eps->nev) break;
      v = eps->V[eps->nconv];
    }

    EPSMonitorEstimates(eps,eps->its,eps->nconv,eps->errest,eps->nconv+1); 
    EPSMonitorValues(eps,eps->its,eps->nconv,eps->eigr,PETSC_NULL,eps->nconv+1); 
  }

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_POWER"
int EPSCreate_POWER(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->setup                = EPSSetUp_POWER;
  eps->ops->setdefaults          = EPSSetDefaults_POWER;
  eps->ops->solve                = EPSSolve_POWER;
  eps->ops->destroy              = EPSDefaultDestroy;
  eps->ops->view                 = 0;
  eps->ops->backtransform        = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
