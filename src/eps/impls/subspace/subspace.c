
/*                       
       This implements the subspace iteration method for finding the 
       eigenpairs associtated to the eigenvalues with largest magnitude.
*/
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/

typedef struct {
  int        inner;
} EPS_SUBSPACE;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_SUBSPACE"
static int EPSSetUp_SUBSPACE(EPS eps)
{
  int          ierr, N;
  EPS_SUBSPACE *subspace = (EPS_SUBSPACE *)eps->data;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+8);
  eps->ncv = PetscMin(eps->ncv,N);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!subspace->inner || subspace->inner == PETSC_DEFAULT) {
    if (eps->ishermitian) subspace->inner = 10;
    else                  subspace->inner = 4;
  }
  if (!eps->tol) eps->tol = 1.e-7;

  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_SUBSPACE"
static int  EPSSolve_SUBSPACE(EPS eps)
{
  int          ierr, i, j, k, maxit=eps->max_it, ncv = eps->ncv;
  Vec          w;
  PetscReal    relerr, tol=eps->tol;
  PetscScalar  alpha, *H, *S;
  EPS_SUBSPACE *subspace = (EPS_SUBSPACE *)eps->data;

  PetscFunctionBegin;
  w  = eps->work[0];
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&H);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&S);CHKERRQ(ierr);

  /* Build Z from initial vector */
  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  for (k=1; k<ncv; k++) {
    ierr = STApply(eps->OP,eps->V[k-1],eps->V[k]); CHKERRQ(ierr);
  }
  /* QR-Factorize V R = Z */
  ierr = EPSQRDecomposition(eps,0,ncv,PETSC_NULL,ncv);CHKERRQ(ierr);

  i = 0;
  eps->its = 0;

  while (eps->its<maxit) {

    eps->its = eps->its + 1;

    /* Y = OP^inner V */
    for (k=i;k<ncv;k++) {
      for (j=0;j<subspace->inner;j++) {
        ierr = STApply(eps->OP,eps->V[k],w);CHKERRQ(ierr);
        ierr = VecCopy(w,eps->V[k]);CHKERRQ(ierr);
      }
    }

    /* QR-Factorize V R = Y */
    ierr = EPSQRDecomposition(eps,i,ncv,PETSC_NULL,ncv);CHKERRQ(ierr);
  
    /* compute the projected matrix, H = V^* A V */
    for (j=i;j<ncv;j++) {
      ierr = STApply(eps->OP,eps->V[j],w);CHKERRQ(ierr);
      ierr = VecMDot(ncv-i,w,eps->V+i,H+(ncv-i)*(j-i));CHKERRQ(ierr);
    }

    /* solve the reduced problem, compute the 
       eigendecomposition H = S Theta S^{-1} */
    ierr = EPSDenseNHEP(ncv-i,H,eps->eigr+i,eps->eigi+i,S);CHKERRQ(ierr);

    /* update V = V S */
    ierr = EPSReverseProjection(eps,i,ncv-i,S);CHKERRQ(ierr);

    /* check eigenvalue convergence */
    for (j=i;j<ncv;j++) {
      ierr = STApply(eps->OP,eps->V[j],w);CHKERRQ(ierr);
      alpha = -eps->eigr[j];
      ierr = VecAXPY(&alpha,eps->V[j],w);CHKERRQ(ierr);
      ierr = VecNorm(w,NORM_2,&relerr);CHKERRQ(ierr);
      eps->errest[j] = relerr;
    }

    /* lock converged Ritz pairs */
    eps->nconv = i;
    for (j=i;j<ncv;j++) {
      if (eps->errest[j]<tol) {
        if (j>eps->nconv) {
          ierr = EPSSwapEigenpairs(eps,eps->nconv,j);CHKERRQ(ierr);
        }
        eps->nconv = eps->nconv + 1;
      }
    }
    i = eps->nconv;

    EPSMonitorEstimates(eps,eps->its,eps->nconv,eps->errest,ncv); 
    EPSMonitorValues(eps,eps->its,eps->nconv,eps->eigr,PETSC_NULL,ncv); 

    if (eps->nconv>=eps->nev) break;

  }

  ierr = PetscFree(H);CHKERRQ(ierr);
  ierr = PetscFree(S);CHKERRQ(ierr);

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_SUBSPACE"
static int EPSView_SUBSPACE(EPS eps,PetscViewer viewer)
{
  EPS_SUBSPACE *subspace = (EPS_SUBSPACE *) eps->data;
  int          ierr;
  PetscTruth   isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSSUBSPACE",((PetscObject)viewer)->type_name);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"number of inner iterations: %d\n",subspace->inner);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_SUBSPACE"
static int EPSSetFromOptions_SUBSPACE(EPS eps)
{
  EPS_SUBSPACE *subspace = (EPS_SUBSPACE *)eps->data;
  int          ierr,val;
  PetscTruth   flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SUBSPACE options");CHKERRQ(ierr);
    val = subspace->inner;
    if (val <= 0) {
      if (eps->ishermitian) val = 10;
      else                  val = 4;
    }
    ierr = PetscOptionsInt("-eps_subspace_inner","Number of inner iterations","EPSSubspaceSetInner",val,&val,&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSubspaceSetInner(eps,val);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSSubspaceSetInner_SUBSPACE"
int EPSSubspaceSetInner_SUBSPACE(EPS eps,int val)
{
  EPS_SUBSPACE   *subspace;

  PetscFunctionBegin;
  subspace        = (EPS_SUBSPACE *) eps->data;
  subspace->inner = val;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSSubspaceSetInner"
/*@
   EPSSubspaceSetInner - Sets the number of inner iterations performed by
   the Subspace Iteration method.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  val - number of iterations to perform

   Options Database Key:
.  -eps_subspace_inner - Sets the value of the inner iterations

   Notes:
   This value specifies how many matrix-vector products have to be carried 
   out in the Subspace Iteration method between the orthogonalization steps.

   The default value is 10 for Hermitian problems and 4 for non-Hermitian ones.
   
   Level: advanced

@*/
int EPSSubspaceSetInner(EPS eps,int val)
{
  int ierr, (*f)(EPS,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSSubspaceSetInner_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,val);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_SUBSPACE"
int EPSCreate_SUBSPACE(EPS eps)
{
  EPS_SUBSPACE *subspace;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_SUBSPACE,&subspace);CHKERRQ(ierr);
  PetscMemzero(subspace,sizeof(EPS_SUBSPACE));
  PetscLogObjectMemory(eps,sizeof(EPS_SUBSPACE));
  eps->data                      = (void *) subspace;
  eps->ops->setup                = EPSSetUp_SUBSPACE;
  eps->ops->solve                = EPSSolve_SUBSPACE;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->setfromoptions       = EPSSetFromOptions_SUBSPACE;
  eps->ops->view                 = EPSView_SUBSPACE;
  eps->ops->backtransform        = EPSBackTransform_Default;

  subspace->inner = 0;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSSubspaceSetInner_C","EPSSubspaceSetInner_SUBSPACE",EPSSubspaceSetInner_SUBSPACE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

