
/*                       
       This implements the Arnoldi method with explicit restart and
       deflation.
*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARNOLDI"
static int EPSSetUp_ARNOLDI(EPS eps)
{
  int         ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+8);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
static int  EPSSolve_ARNOLDI(EPS eps)
{
  int         ierr, i, j, k, m, maxit=eps->max_it, ncv = eps->ncv;
  int         lwork, ilo, mout;
  Vec         w;
  PetscReal   norm, tol=eps->tol;
  PetscScalar alpha, *H, *Y, *S, *pV, *work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal   *rwork;
#endif

  PetscFunctionBegin;
  w  = eps->work[0];
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&H);CHKERRQ(ierr);
  ierr = PetscMemzero(H,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&S);CHKERRQ(ierr);

ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
ierr = VecRestoreArray(eps->V[0],&pV);CHKERRQ(ierr);
  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = VecNorm(eps->V[0],NORM_2,&norm);CHKERRQ(ierr);
  if (norm==0.0) SETERRQ( 1,"Null initial vector" );
  alpha = 1.0/norm;
  ierr = VecScale(&alpha,eps->V[0]);CHKERRQ(ierr);

  eps->its = 0;
  m = ncv-1; /* m is the number of Arnoldi vectors, one less than
                the available vectors because one is needed for v_{m+1} */
  k = 0;     /* k is the number of locked vectors */

  while (eps->its<maxit) {

    /* compute the projected matrix, H, with the basic Arnoldi method */
    for (j=k;j<m;j++) {

      /* w = OP v_j */
      ierr = STApply(eps->OP,eps->V[j],eps->V[j+1]);CHKERRQ(ierr);

      /* orthogonalize wrt previous vectors */
      ierr = (*eps->orthog)(eps,j+1,eps->V[j+1],&H[0+ncv*j],&norm);CHKERRQ(ierr);

      /* h_{j+1,j} = ||w||_2 */
      if (norm==0.0) SETERRQ( 1,"Breakdown in Arnoldi method" );
      H[j+1+ncv*j] = norm;
      alpha = 1.0/norm;
      ierr = VecScale(&alpha,eps->V[j+1]);CHKERRQ(ierr);

    }

    /* At this point, H has the following structure

              | *   * | *   *   *   * |
              |     * | *   *   *   * |
              | ------|-------------- |
          H = |       | *   *   *   * |
              |       | *   *   *   * |
              |       |     *   *   * |
              |       |         *   * |

       that is, a mxm upper Hessenberg matrix whose kxk principal submatrix
       is (quasi-)triangular.
     */

    /* reduce H to (real) Schur form, H = S \tilde{H} S'  */
    lwork = m;
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ilo = k+1;
#if !defined(PETSC_USE_COMPLEX)
    LAhseqr_("S","I",&m,&ilo,&m,H,&ncv,eps->eigr,eps->eigi,S,&ncv,work,&lwork,&ierr,1,1);
#else
    LAhseqr_("S","I",&m,&ilo,&m,H,&ncv,eps->eigr,S,&ncv,work,&lwork,&ierr,1,1);
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);
 
    /* compute eigenvectors y_i */
    ierr = PetscMemcpy(Y,S,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    lwork = 3*m;
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    LAtrevc_("R","B",PETSC_NULL,&m,H,&ncv,Y,&ncv,Y,&ncv,&ncv,&mout,work,&ierr,1,1);
#else
    ierr = PetscMalloc(2*m*sizeof(PetscScalar),&rwork);CHKERRQ(ierr);
    LAtrevc_("R","B",PETSC_NULL,&m,H,&ncv,Y,&ncv,Y,&ncv,&ncv,&mout,work,rwork,&ierr,1,1);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);

    /* compute error estimates */
    for (j=k;j<m;j++) {
      /* errest_j = h_{m+1,m} |e_m' y_j| */
      eps->errest[j] = PetscRealPart(H[m+ncv*(m-1)]) 
                     * PetscAbsScalar(Y[(m-1)+ncv*j]);
    }

    /* compute Ritz vectors */
    ierr = EPSReverseProjection(eps,k,m-k,S);CHKERRQ(ierr);

    /* lock converged Ritz pairs */
    for (j=k;j<m;j++) {
      if (eps->errest[j]<tol) {
        if (j>k) {
          ierr = EPSSwapEigenpairs(eps,k,j);CHKERRQ(ierr);
        }
        ierr = (*eps->orthog)(eps,k,eps->V[k],PETSC_NULL,&norm);CHKERRQ(ierr);
        if (norm==0.0) SETERRQ( 1,"Breakdown in Arnoldi method" );
        alpha = 1.0/norm;
        ierr = VecScale(&alpha,eps->V[k]);CHKERRQ(ierr);
        /* h_{i,k} = v_i' OP v_k, i=1..k */
        for (i=0;i<=k;i++) {
          ierr = STApply(eps->OP,eps->V[k],w);CHKERRQ(ierr);
          ierr = VecDot(w,eps->V[i],H+i+ncv*k);CHKERRQ(ierr);
        }
        H[k+1+ncv*k] = 0.0;
        k = k + 1;
      }
    }
    eps->nconv = k;

    /* select next wanted eigenvector as restart vector */
    ierr = EPSSortEigenvalues(m-k,eps->eigr+k,eps->eigi+k,eps->which,1,&i);CHKERRQ(ierr);
    ierr = EPSSwapEigenpairs(eps,k,k+i);CHKERRQ(ierr);

    /* orthogonalize u_k wrt previous vectors */
    ierr = (*eps->orthog)(eps,k,eps->V[k],PETSC_NULL,&norm);CHKERRQ(ierr);

    /* normalize new initial vector */
    if (norm==0.0) SETERRQ( 1,"Breakdown in Arnoldi method" );
    alpha = 1.0/norm;
    ierr = VecScale(&alpha,eps->V[k]);CHKERRQ(ierr);

    EPSMonitorEstimates(eps,eps->its + 1,eps->nconv,eps->errest,m); 
    EPSMonitorValues(eps,eps->its + 1,eps->nconv,eps->eigr,eps->eigi,m); 
    eps->its = eps->its + 1;

    if (eps->nconv>=eps->nev) break;

  }

  ierr = PetscFree(H);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(S);CHKERRQ(ierr);

  if( eps->its==maxit ) eps->its = eps->its - 1;
  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
int EPSCreate_ARNOLDI(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

