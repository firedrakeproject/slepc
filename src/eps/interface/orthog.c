/*
     EPS routines related to orthogonalization.
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
   This routine orthonormalizes the columns of V so that V'*V=I up to 
   working precision. It assumes that the first m columns (from 0 to m-1) 
   are already orthonormal. The coefficients of the orthogonalization are
   stored in R so that V*R is equal to the original V.

   In some cases, this routine makes V B-orthonormal, that is, V'*B*V=I.

   If one of the vectors is linearly dependent wrt the rest (at working
   precision) then it is replaced by a random vector.

   Level: developer

.seealso: EPSOrthogonalize(), STNorm(), STInnerProduct().
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
    if (R) { ierr = EPSOrthogonalize(eps,k,V,V[k],&R[0+ldr*k],&norm,&breakdown);CHKERRQ(ierr); }
    else   { ierr = EPSOrthogonalize(eps,k,V,V[k],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr); }

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
  PetscScalar    shh[100],shh2[100],*lhh,
                 zero = 0.0,minus = -1.0;
  PetscTruth     alloc = PETSC_FALSE;
  PetscReal      hnorm = 0,lnorm;
  Vec            w;

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
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  
  /*** First orthogonalization ***/

  /* h = V^* v */
  /* q = v - V h */
  ierr = STMInnerProduct(eps->OP,n,v,V,H);CHKERRQ(ierr);
  ierr = VecSet(&zero,w);CHKERRQ(ierr);
  ierr = VecMAXPY(n,H,w,V);CHKERRQ(ierr);
  ierr = VecAXPY(&minus,w,v);CHKERRQ(ierr);
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
    ierr = STMInnerProduct(eps->OP,n,v,V,lhh);CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      H[j] += lhh[j];
    }
    ierr = VecSet(&zero,w);CHKERRQ(ierr);
    ierr = VecMAXPY(n,lhh,w,V);CHKERRQ(ierr);
    ierr = VecAXPY(&minus,w,v);CHKERRQ(ierr);

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
  ierr = VecDestroy(w);CHKERRQ(ierr);
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
  PetscScalar    alpha,
                 zero = 0.0,minus = -1.0;
  PetscTruth     allocated;
  PetscScalar    lh[100],*h;
  PetscReal      hnorm = 0,lnorm;
  Vec            w;
  
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
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);

  ierr = VecSet(&zero,w);CHKERRQ(ierr);
  for (j=0; j<n; j++) {
    /* alpha = ( v, v_j ) */
    ierr = STInnerProduct(eps->OP,v,V[j],&alpha);CHKERRQ(ierr);
    /* store coefficients if requested */
    h[j] = alpha;
    /* v <- v - alpha v_j */
    ierr = VecAXPY(&alpha,V[j],w);CHKERRQ(ierr);
  }
  ierr = VecAXPY(&minus,w,v);CHKERRQ(ierr);
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
    ierr = VecSet(&zero,w);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      /* alpha = ( v, v_j ) */
      ierr = STInnerProduct(eps->OP,v,V[j],&alpha);CHKERRQ(ierr);
      /* store coefficients if requested */
      h[j] += alpha;
      /* v <- v - alpha v_j */
      ierr = VecAXPY(&alpha,V[j],w);CHKERRQ(ierr);
    }
    ierr = VecAXPY(&minus,w,v);CHKERRQ(ierr);
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
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPurge"
/*@
   EPSPurge - Purge a vector of all converged vectors.

   Collective on EPS

   Input Parameters:
.  eps - the eigenproblem solver context

   Input/Output Parameter:
.  v - vector to be purged

   Notes:
   On exit, v is orthogonal to all the basis vectors of the currently
   converged invariant subspace as well as all the deflation vectors
   provided by the user.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: EPSOrthogonalize(), EPSAttachDeflationSpace()
@*/
PetscErrorCode EPSPurge(EPS eps,Vec v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = EPSOrthogonalize(eps,eps->nds+eps->nconv,eps->DSV,v,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSOrthogonalize"
/*@
   EPSOrthogonalize - Orthogonalize a vector with respect to a set of vectors.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  n - number of columns of V
-  V - set of vectors

   Input/Output Parameter:
.  v - vector to be orthogonalized

   Output Parameter:
+  H  - coefficients computed during orthogonalization
.  norm - norm of the vector ofter being orthogonalized
-  breakdown - flag indicating that refinement did not improve the quality
   of orthogonalization

   Notes:
   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: EPSSetOrthogonalization()
@*/
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
