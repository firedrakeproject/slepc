/*
    Routines used for the orthogonalization, taken from PETSc's
    GMRES module.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/eps/epsimpl.h"

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSModifiedGramSchmidtOrthogonalization"
int EPSModifiedGramSchmidtOrthogonalization(EPS eps,int it,PetscScalar *H)
{
  int         ierr,j;
  PetscScalar alpha;

  PetscFunctionBegin;
  for (j=0; j<=it; j++) {
    /* alpha = ( v_{it+1}, v_j ) */
    ierr = VecDot(eps->V[it+1],eps->V[j],&alpha);CHKERRQ(ierr);
    /* store coefficients if requested */
    if (H) *H++ = alpha;
    /* v_{it+1} <- v_{it+1} - alpha v_j */
    alpha = -alpha;
    ierr = VecAXPY(&alpha,eps->V[j],eps->V[it+1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT always recommended, 
  but it can give MUCH better performance than the default modified form
  when running in a parallel environment.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSUnmodifiedGramSchmidtOrthogonalization"
int EPSUnmodifiedGramSchmidtOrthogonalization(EPS eps,int it,PetscScalar *H)
{
  int         ierr,j;
  PetscScalar shh[100];
  PetscTruth  alloc = PETSC_FALSE;

  PetscFunctionBegin;

  if (!H) {
    if (it<10) H = shh;   /* Don't allocate small arrays */
    else { 
      ierr = PetscMalloc((it+1)*(it+1)*sizeof(PetscScalar),&H);CHKERRQ(ierr);
      alloc = PETSC_TRUE;
    }
  }

  /* This is really a matrix-vector product, with the matrix stored
     as pointer to rows */
  ierr = VecMDot(it+1,eps->V[it+1],eps->V,H);CHKERRQ(ierr);

  /* This is really a matrix-vector product: 
     [h_0,h_1,...]*[ v_0; v_1; ...] subtracted from v_{it+1}.  */
  for (j=0;j<=it;j++) H[j] = -H[j];
  ierr = VecMAXPY(it+1,H,eps->V[it+1],eps->V);CHKERRQ(ierr);
  for (j=0;j<=it;j++) H[j] = -H[j];

  if (alloc) { ierr = PetscFree(H);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*
  This version uses 1 iteration of iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSIROrthogonalization"
int EPSIROrthogonalization(EPS eps,int it,PetscScalar *H)
{
  int         ierr,j,ncnt;
  PetscScalar shh[100],shh2[100],*lhh;
  PetscTruth  alloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (!H) {
    if (it<10) H = shh2;   /* Don't allocate small arrays */
    else { 
      ierr = PetscMalloc((it+1)*(it+1)*sizeof(PetscScalar),&H);CHKERRQ(ierr);
      alloc = PETSC_TRUE;
    }
  }
  /* Don't allocate small arrays */
  if (it<100) lhh = shh;
  else { ierr = PetscMalloc((it+1)*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  
  /* Clear H since we will accumulate values into it */
  for (j=0;j<=it;j++) H[j] = 0.0;

  ncnt = 0;
  do {
    /* This is really a matrix-vector product, with the matrix stored
       as pointer to rows */
    ierr = VecMDot(it+1,eps->V[it+1],eps->V,lhh);CHKERRQ(ierr); /* <v,vnew> */

    /* This is really a matrix vector product: 
       [h_0,h_1,...]*[ v_0; v_1; ...] subtracted from v_{it+1}.  */
    for (j=0;j<=it;j++) lhh[j] = -lhh[j];
    ierr = VecMAXPY(it+1,lhh,eps->V[it+1],eps->V);CHKERRQ(ierr);
    for (j=0;j<=it;j++) {
      H[j] -= lhh[j];     /* H += -<v,vnew> */
    }
  } while (ncnt++ < 2);

  if (it>=100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  if (alloc) { ierr = PetscFree(H);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

