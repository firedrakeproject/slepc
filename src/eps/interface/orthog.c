/*
    Routines related to orthogonalization.

    See the SLEPc users manual for a detailed explanation.
*/
#include "src/eps/epsimpl.h"    /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSQRDecomposition"
/*@
   EPSQRDecomposition - Compute the QR factorization of a set of vectors.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  V - set of vectors
.  m - starting column
.  n - ending column
-  ldr - leading dimension of R

   Output Parameter:
.  R  - triangular matrix of QR factorization

   Notes:
   Assumes that the first m columns (from 0 to m-1) are already orthonormal
   to working precision.

   Level: developer

@*/
PetscErrorCode EPSQRDecomposition(EPS eps,Vec *V,int m,int n,PetscScalar *R,int ldr)
{
  PetscErrorCode ierr;
  int            k;
  PetscScalar    alpha;
  PetscReal      norm;
  PetscTruth     breakdown;
  
  PetscFunctionBegin;

  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (k>0) {
      if (R) { ierr = EPSOrthogonalize(eps,k,V,V[k],&R[0+ldr*k],&norm,&breakdown);CHKERRQ(ierr); }
      else   { ierr = EPSOrthogonalize(eps,k,V,V[k],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr); }
    }
    else {
      ierr = STNorm(eps->OP,V[0],&norm);CHKERRQ(ierr);
    }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (norm==0.0 || breakdown) { 
      PetscLogInfo(eps,"EPSQRDecomposition: Linearly dependent vector found, generating a new random vector\n" );
      ierr = SlepcVecSetRandom(V[k]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[k],&norm);CHKERRQ(ierr);
    }
    alpha = 1.0/norm;
    ierr = VecScale(&alpha,V[k]);CHKERRQ(ierr);
    if (R) R[k+ldr*k] = norm;

  }

  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using classical Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSClassicalGramSchmidtOrthogonalization"
static PetscErrorCode EPSClassicalGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    shh[100],shh2[100],*lhh;
  PetscTruth     alloc = PETSC_FALSE;
  PetscReal      hnorm = 0,lnorm;

  PetscFunctionBegin;

  if (!H) {
    if (n<=100) H = shh2;   /* Don't allocate small arrays */
    else { 
      ierr = PetscMalloc(n*sizeof(PetscScalar),&H);CHKERRQ(ierr);
      alloc = PETSC_TRUE;
    }
  }
  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else { ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  if (!norm) { norm = &lnorm; }
  
  /*** First orthogonalization ***/

  /* h = V^* v */
  /* q = v - V h */
  for (j=0;j<n;j++) {
    ierr = STInnerProduct(eps->OP,v,V[j],&H[j]);
    lhh[j] = -H[j];    
  }  
  ierr = VecMAXPY(n,lhh,v,V);CHKERRQ(ierr);
  ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);

  /* compute hnorm */
  if (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED || breakdown) {
    hnorm = 0.0;
    for (j=0; j<n; j++) {
      hnorm  +=  PetscRealPart(H[j] * PetscConj(H[j]));
    }
    hnorm = sqrt(hnorm);
  }
  
  /*** Second orthogonalization if necessary ***/
  
  /* if ||q|| < eta ||h|| */
  if (eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS || 
     (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED && 
      *norm < eps->orthog_eta * hnorm)) {
    PetscLogInfo(eps,"EPSClassicalGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",*norm,hnorm);

    /* s = V^* q */
    /* q = q - V s  ;  h = h + s */
    for (j=0;j<n;j++) {
      ierr = STInnerProduct(eps->OP,v,V[j],&lhh[j]);
      H[j] += lhh[j];
      lhh[j] = -lhh[j];    
    }
    ierr = VecMAXPY(n,lhh,v,V);CHKERRQ(ierr);

    hnorm = *norm;
    ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);
  }
  
  /* check breakdown */
  if (breakdown) {
    if (*norm < eps->orthog_eta * hnorm) *breakdown = PETSC_TRUE;
    else *breakdown = PETSC_FALSE;
  }

  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  if (alloc) { ierr = PetscFree(H);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using modified Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSModifiedGramSchmidtOrthogonalization"
PetscErrorCode EPSModifiedGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    alpha;
  PetscTruth     allocated;
  PetscScalar    lh[100],*h;
  PetscReal      hnorm = 0,lnorm;
  
  PetscFunctionBegin;
  
  allocated = PETSC_FALSE;
  if (!H) {
    if (n<=100) h = lh;
    else {
      ierr = PetscMalloc(n*sizeof(PetscScalar),&h);CHKERRQ(ierr);
      allocated = PETSC_TRUE;
    }
  } else h = H;
  
  if (!norm) { norm = &lnorm; }

  for (j=0; j<n; j++) {
    /* alpha = ( v, v_j ) */
    ierr = STInnerProduct(eps->OP,v,V[j],&alpha);CHKERRQ(ierr);
    /* store coefficients if requested */
    h[j] = alpha;
    /* v <- v - alpha v_j */
    alpha = -alpha;
    ierr = VecAXPY(&alpha,V[j],v);CHKERRQ(ierr);
  }
  ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);

  if (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED || breakdown) {
    hnorm = 0.0;
    for (j=0; j<n; j++) {
      hnorm += PetscRealPart(h[j] * PetscConj(h[j]));
    }
    hnorm = sqrt(hnorm);
  }
  
  if (eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS || 
     (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED &&
      *norm < eps->orthog_eta * hnorm)) {
    PetscLogInfo(eps,"EPSModifiedGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",*norm,hnorm);
    for (j=0; j<n; j++) {
      /* alpha = ( v, v_j ) */
      ierr = STInnerProduct(eps->OP,v,V[j],&alpha);CHKERRQ(ierr);
      /* store coefficients if requested */
      h[j] += alpha;
      /* v <- v - alpha v_j */
      alpha = -alpha;
      ierr = VecAXPY(&alpha,V[j],v);CHKERRQ(ierr);
    }
    hnorm = *norm;
    ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);
  }

  /* check breakdown */
  if (breakdown) {
    if (*norm < eps->orthog_eta * hnorm) *breakdown = PETSC_TRUE;
    else *breakdown = PETSC_FALSE;
  }
  
  if (allocated) {
    ierr = PetscFree(h);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPurge"
PetscErrorCode EPSPurge(EPS eps,Vec v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = EPSOrthogonalize(eps,eps->nds+eps->nconv,eps->DSV,v,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSOrthogonalize"
PetscErrorCode EPSOrthogonalize(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
    if (breakdown) *breakdown = PETSC_FALSE;
  } else {
    ierr = PetscLogEventBegin(EPS_Orthogonalization,eps,0,0,0);CHKERRQ(ierr);
    switch (eps->orthog_type) {
      case EPS_CGS_ORTH:
        ierr = EPSClassicalGramSchmidtOrthogonalization(eps,n,V,v,H,norm,breakdown);CHKERRQ(ierr);
        break;
      case EPS_MGS_ORTH:
        ierr = EPSModifiedGramSchmidtOrthogonalization(eps,n,V,v,H,norm,breakdown);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
    ierr = PetscLogEventEnd(EPS_Orthogonalization,eps,0,0,0);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
