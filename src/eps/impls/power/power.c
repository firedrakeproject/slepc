
/*                       
       This implements the power iteration for finding the eigenpair
       corresponding to the eigenvalue with largest magnitude.
*/
#include "src/eps/epsimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_POWER"
static int EPSSetUp_POWER(EPS eps)
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

  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_POWER"
static int  EPSSolve_POWER(EPS eps)
{
  int         ierr, i;
  Vec         v, y, e;
  PetscReal   relerr, norm;
  PetscScalar theta, alpha;

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

    /* v = y/||y||_B */
    ierr = VecCopy(y,v);CHKERRQ(ierr);
    ierr = STNorm(eps->OP,y,&norm);CHKERRQ(ierr);
    alpha = 1.0/norm;
    ierr = VecScale(&alpha,v);CHKERRQ(ierr);

    /* y = OP v */
    ierr = STApply(eps->OP,v,y);CHKERRQ(ierr);

    /* theta = v^* y */
    ierr = STInnerProduct(eps->OP,y,v,&theta);CHKERRQ(ierr);

    /* deflation of converged eigenvectors */
    ierr = EPSPurge(eps,y);

    /* if ||y-theta v||_2 / |theta| < tol, accept */
    ierr = VecCopy(y,e);CHKERRQ(ierr);
    alpha = -theta;
    ierr = VecAXPY(&alpha,v,e);CHKERRQ(ierr);
    ierr = VecNorm(e,NORM_2,&relerr);CHKERRQ(ierr);
    relerr = relerr / PetscAbsScalar(theta);
    eps->errest[eps->nconv] = relerr;
    eps->eigr[eps->nconv] = theta;

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_POWER"
int EPSCreate_POWER(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->setfromoptions       = 0;
  eps->ops->setup                = EPSSetUp_POWER;
  eps->ops->solve                = EPSSolve_POWER;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = 0;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->computevectors            = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
