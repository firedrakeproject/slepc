/*
     EPS routines related to orthogonalization.
     See the SLEPc users manual for a detailed explanation.
*/
#include "src/eps/epsimpl.h"    /*I "slepceps.h" I*/

int countorthog = 0;
int countreorthog = 0;

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
  PetscTruth     lindep;
  
  PetscFunctionBegin;

  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (R) { ierr = EPSOrthogonalize(eps,k,V,V[k],&R[0+ldr*k],&norm,&lindep);CHKERRQ(ierr); }
    else   { ierr = EPSOrthogonalize(eps,k,V,V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (norm==0.0 || lindep) { 
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
static PetscErrorCode EPSClassicalGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    shh[100],*lhh,
                 zero = 0.0,minus = -1.0;
  Vec            w;

  PetscFunctionBegin;

  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else { ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  
  /*** First orthogonalization ***/

  /* h = V^* v */
  /* q = v - V h */
  ierr = STMInnerProduct(eps->OP,n,v,V,H);CHKERRQ(ierr);
  ierr = VecSet(&zero,w);CHKERRQ(ierr);
  ierr = VecMAXPY(n,H,w,V);CHKERRQ(ierr);
  ierr = VecAXPY(&minus,w,v);CHKERRQ(ierr);
  
  /* compute hnorm */
  if (hnorm) {
    *hnorm = 0.0;
    for (j=0; j<n; j++) {
      *hnorm += PetscRealPart(H[j] * PetscConj(H[j]));
    }
    *hnorm = sqrt(*hnorm);
  }
  
  /* compute norm of v for refinement or linear dependence checking */
  if (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED ||
      (eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS && hnorm) ) {
    ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);  
  }

  /*** Second orthogonalization if necessary ***/
  
  /* if ||q|| < eta ||h|| */
  if ((eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED && *norm < eps->orthog_eta * *hnorm) || 
      eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS) {
    PetscLogInfo(eps,"EPSClassicalGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",norm ? *norm : 0,hnorm ? *hnorm : 0);
    countreorthog++;

    /* s = V^* q */
    /* q = q - V s  ;  h = h + s */
    ierr = STMInnerProduct(eps->OP,n,v,V,lhh);CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      H[j] += lhh[j];
    }
    ierr = VecSet(&zero,w);CHKERRQ(ierr);
    ierr = VecMAXPY(n,lhh,w,V);CHKERRQ(ierr);
    ierr = VecAXPY(&minus,w,v);CHKERRQ(ierr);

    if (hnorm) *hnorm = *norm;
  }
  
  /* compute norm of v */
  if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
  
  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using modified Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSModifiedGramSchmidtOrthogonalization"
PetscErrorCode EPSModifiedGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    alpha;
  
  PetscFunctionBegin;
  
  /*** First orthogonalization ***/

  for (j=0; j<n; j++) {
    /* alpha = ( v, v_j ) */
    ierr = STInnerProduct(eps->OP,v,V[j],&alpha);CHKERRQ(ierr);
    /* store coefficients if requested */
    H[j] = alpha;
    /* v <- v - alpha v_j */
    alpha = -alpha;
    ierr = VecAXPY(&alpha,V[j],v);CHKERRQ(ierr);
  }
  
  /* compute hnorm */
  if (hnorm) {
    *hnorm = 0.0;
    for (j=0; j<n; j++) {
      *hnorm += PetscRealPart(H[j] * PetscConj(H[j]));
    }
    *hnorm = sqrt(*hnorm);
  }
    
  /* compute norm of v for refinement or linear dependence checking */
  if (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED ||
      (eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS && hnorm) ) {
    ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);  
  }

  /*** Second orthogonalization if necessary ***/

  /* if ||q|| < eta ||h|| */
  if ((eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED && *norm < eps->orthog_eta * *hnorm) || 
      eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS) {
    PetscLogInfo(eps,"EPSModifiedGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",norm ? *norm : 0,hnorm ? *hnorm : 0);
    countreorthog++;
    for (j=0; j<n; j++) {
      /* alpha = ( v, v_j ) */
      ierr = STInnerProduct(eps->OP,v,V[j],&alpha);CHKERRQ(ierr);
      /* store coefficients if requested */
      H[j] += alpha;
      /* v <- v - alpha v_j */
      alpha = -alpha;
      ierr = VecAXPY(&alpha,V[j],v);CHKERRQ(ierr);
    }
    if (hnorm) *hnorm = *norm;
  }

  /* compute norm of v */
  if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }

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
-  lindep - flag indicating that refinement did not improve the quality
   of orthogonalization

   Notes:
   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: EPSSetOrthogonalization()
@*/
PetscErrorCode EPSOrthogonalize(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h;
  PetscTruth     allocated = PETSC_FALSE;
  PetscReal      lhnrm,*hnrm,lnrm,*nrm;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
  } else {
    ierr = PetscLogEventBegin(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
    
    /* allocate H if needed */
    if (!H) {
      if (n<=100) h = lh;
      else {
        ierr = PetscMalloc(n*sizeof(PetscScalar),&h);CHKERRQ(ierr);
        allocated = PETSC_TRUE;
      }
    } else h = H;
    
    /* retrieve hnrm and nrm for linear dependence check or conditional refinement */
    if (lindep || eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED) {
      hnrm = &lhnrm;
      if (norm) nrm = norm;
      else nrm = &lnrm;
    } else {
      hnrm = PETSC_NULL;
      nrm = norm;
    }
    
    countorthog++;
    switch (eps->orthog_type) {
      case EPS_CGS_ORTH:
        ierr = EPSClassicalGramSchmidtOrthogonalization(eps,n,V,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      case EPS_MGS_ORTH:
        ierr = EPSModifiedGramSchmidtOrthogonalization(eps,n,V,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
    
    /* check linear dependence */
    if (lindep) {
      if (*nrm < eps->orthog_eta * *hnrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    
    if (allocated) { ierr = PetscFree(h);CHKERRQ(ierr); }
    
    ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
