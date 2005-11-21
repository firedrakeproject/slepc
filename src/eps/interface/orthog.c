/*
     EPS routines related to orthogonalization.
     See the SLEPc users manual for a detailed explanation.
*/
#include "src/eps/epsimpl.h"    /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

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
  PetscReal      norm;
  PetscTruth     lindep;
  
  PetscFunctionBegin;

  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (R) { ierr = EPSOrthogonalize(eps,k,V,V[k],&R[0+ldr*k],&norm,&lindep);CHKERRQ(ierr); }
    else   { ierr = EPSOrthogonalize(eps,k,V,V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (norm==0.0 || lindep) { 
      PetscVerboseInfo((eps,"EPSQRDecomposition: Linearly dependent vector found, generating a new random vector\n"));
      ierr = SlepcVecSetRandom(V[k]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[k],&norm);CHKERRQ(ierr);
    }
    ierr = VecScale(V[k],1.0/norm);CHKERRQ(ierr);
    if (R) R[k+ldr*k] = norm;

  }

  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using classical Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSClassicalGramSchmidtOrthogonalization"
static PetscErrorCode EPSClassicalGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    shh[100],*lhh;
  Vec            w;

  PetscFunctionBegin;

  if (!W) W=V;

  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else { ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  
  /*** First orthogonalization ***/

  /* h = W^* v */
  /* q = v - V h */
  ierr = STMInnerProduct(eps->OP,n,v,W,H);CHKERRQ(ierr);
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,n,H,V);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);
  
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
    PetscVerboseInfo((eps,"EPSClassicalGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",norm ? *norm : 0,hnorm ? *hnorm : 0));

    /* s = W^* q */
    /* q = q - V s  ;  h = h + s */
    ierr = STMInnerProduct(eps->OP,n,v,W,lhh);CHKERRQ(ierr);
    for (j=0;j<n;j++) {
      H[j] += lhh[j];
    }
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,n,lhh,V);CHKERRQ(ierr);
    ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);

    if (hnorm) *hnorm = *norm;
  }
  
  /* compute norm of v */
  if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
  
  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSNewClassicalGramSchmidtOrthogonalization"
static PetscErrorCode EPSNewClassicalGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    shh[100],*lhh;
  Vec            w;

  PetscFunctionBegin;

  if (!W) W=V;

  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else { ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  
  /*** First orthogonalization ***/

  /* h = W^* v */
  /* q = v - V h */
  ierr = STMInnerProduct(eps->OP,n,v,W,H);CHKERRQ(ierr);
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,n,H,V);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);
  
  /*** Second orthogonalization ***/
  
  /* s = W^* q */
  /* q = q - V s  ;  h = h + s */
  ierr = STMInnerProductBegin(eps->OP,n,v,W,lhh);CHKERRQ(ierr);
  if (norm) { ierr = STNormBegin(eps->OP,v,norm);CHKERRQ(ierr); }
  ierr = STMInnerProductEnd(eps->OP,n,v,W,lhh);CHKERRQ(ierr);
  if (norm) { ierr = STNormEnd(eps->OP,v,norm);CHKERRQ(ierr); }
  for (j=0;j<n;j++) {
    H[j] += lhh[j];
  }
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,n,lhh,V);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);

  if (hnorm) *hnorm = *norm;
  
  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using modified Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSModifiedGramSchmidtOrthogonalization"
PetscErrorCode EPSModifiedGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    alpha;
  
  PetscFunctionBegin;
  
  if (!W) W=V;

  /*** First orthogonalization ***/

  for (j=0; j<n; j++) {
    /* alpha = ( v, v_j ) */
    ierr = STInnerProduct(eps->OP,v,W[j],&alpha);CHKERRQ(ierr);
    /* store coefficients if requested */
    H[j] = alpha;
    /* v <- v - alpha v_j */
    ierr = VecAXPY(v,-alpha,V[j]);CHKERRQ(ierr);
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
    PetscVerboseInfo((eps,"EPSModifiedGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",norm ? *norm : 0,hnorm ? *hnorm : 0));
    for (j=0; j<n; j++) {
      /* alpha = ( v, v_j ) */
      ierr = STInnerProduct(eps->OP,v,W[j],&alpha);CHKERRQ(ierr);
      /* store coefficients if requested */
      H[j] += alpha;
      /* v <- v - alpha v_j */
      ierr = VecAXPY(v,-alpha,V[j]);CHKERRQ(ierr);
    }
    if (hnorm) *hnorm = *norm;
  }

  /* compute norm of v */
  if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }

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
.  norm - norm of the vector after being orthogonalized
-  lindep - flag indicating that refinement did not improve the quality
   of orthogonalization

   Notes:
   This function applies an orthogonal projector to project vector v onto the
   orthogonal complement of the span of the columns of V.

   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: EPSSetOrthogonalization(), EPSBiOrthogonalize()
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
    
    switch (eps->orthog_type) {
      case EPS_CGS_ORTH:
        ierr = EPSClassicalGramSchmidtOrthogonalization(eps,n,V,PETSC_NULL,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      case EPS_NCGS_ORTH:
        ierr = EPSNewClassicalGramSchmidtOrthogonalization(eps,n,V,PETSC_NULL,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      case EPS_MGS_ORTH:
        ierr = EPSModifiedGramSchmidtOrthogonalization(eps,n,V,PETSC_NULL,v,h,hnrm,nrm);CHKERRQ(ierr);
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

/*
    Biorthogonalization routine using classical Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSCGSBiOrthogonalization"
static PetscErrorCode EPSCGSBiOrthogonalization(EPS eps,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
#if defined(SLEPC_MISSING_LAPACK_GELQF) || defined(SLEPC_MISSING_LAPACK_ORMLQ)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"xGELQF - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  int            j,ione=1,lwork,info;
  PetscScalar    shh[100],*lhh,*vw,*tau,one=1.0,*work;
  Vec            w;

  PetscFunctionBegin;

  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else { ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = PetscMalloc(n*n*sizeof(PetscScalar),&vw);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  
  for (j=0;j<n;j++) {
    ierr = STMInnerProduct(eps->OP,n,V[j],W,vw+j*n);CHKERRQ(ierr);
  }
  lwork = n;
  ierr = PetscMalloc(n*sizeof(PetscScalar),&tau);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  LAPACKgelqf_(&n,&n,vw,&n,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGELQF %i",info);
  
  /*** First orthogonalization ***/

  /* h = W^* v */
  /* q = v - V h */
  ierr = STMInnerProduct(eps->OP,n,v,W,H);CHKERRQ(ierr);
  BLAStrsm_("L","L","N","N",&n,&ione,&one,vw,&n,H,&n,1,1,1,1);
  LAPACKormlq_("L","N",&n,&ione,&n,vw,&n,tau,H,&n,work,&lwork,&info,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORMLQ %i",info);
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,n,H,V);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);

  /* compute norm of v */
  if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
  
  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = PetscFree(vw);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBiOrthogonalize"
/*@
   EPSBiOrthogonalize - Bi-orthogonalize a vector with respect to a set of vectors.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  n - number of columns of V
.  V - set of vectors
-  W - set of vectors

   Input/Output Parameter:
.  v - vector to be orthogonalized

   Output Parameter:
+  H  - coefficients computed during orthogonalization
-  norm - norm of the vector after being orthogonalized

   Notes:
   This function applies an oblique projector to project vector v onto the
   span of the columns of V along the orthogonal complement of the column
   space of W. 

   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: EPSSetOrthogonalization(), EPSOrthogonalize()
@*/
PetscErrorCode EPSBiOrthogonalize(EPS eps,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h;
  PetscTruth     allocated = PETSC_FALSE;
  PetscReal      lhnrm,*hnrm,lnrm,*nrm;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
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
    if (eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED) {
      hnrm = &lhnrm;
      if (norm) nrm = norm;
      else nrm = &lnrm;
    } else {
      hnrm = PETSC_NULL;
      nrm = norm;
    }
    
    switch (eps->orthog_type) {
      case EPS_CGS_ORTH:
        ierr = EPSCGSBiOrthogonalization(eps,n,V,W,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      case EPS_MGS_ORTH:
        ierr = EPSModifiedGramSchmidtOrthogonalization(eps,n,V,W,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
    
    if (allocated) { ierr = PetscFree(h);CHKERRQ(ierr); }
    
    ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

