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
      PetscInfo(eps,"Linearly dependent vector found, generating a new random vector\n");
      ierr = SlepcVecSetRandom(V[k]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[k],&norm);CHKERRQ(ierr);
    }
    ierr = VecScale(V[k],1.0/norm);CHKERRQ(ierr);
    if (R) R[k+ldr*k] = norm;

  }

  PetscFunctionReturn(0);
}

/* 
   EPSOrthogonalizeGS - Compute |v'|, |v| and one step of CGS or MGS 
*/
#undef __FUNCT__  
#define __FUNCT__ "EPSOrthogonalizeGS"
static PetscErrorCode EPSOrthogonalizeGS(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm,Vec w)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    alpha;
  PetscReal      sum;
  
  PetscFunctionBegin;
  /* orthogonalize */
  switch (eps->orthog_type) {
  case EPS_CGS_ORTH:
    /* h = W^* v ; alpha = (v , v) */
    eps->count_orthog_dots+=n;
    ierr = STMInnerProductBegin(eps->OP,n,v,V,H);CHKERRQ(ierr);
    if (onorm || norm) { 
      eps->count_orthog_dots++;
      ierr = STInnerProductBegin(eps->OP,v,v,&alpha);CHKERRQ(ierr); 
    }
    ierr = STMInnerProductEnd(eps->OP,n,v,V,H);CHKERRQ(ierr);
    if (onorm || norm) { ierr = STInnerProductEnd(eps->OP,v,v,&alpha);CHKERRQ(ierr); }
    /* q = v - V h */
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,n,H,V);CHKERRQ(ierr);
    ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);
    /* compute |v| and |v'| */
    if (onorm) *onorm = sqrt(PetscRealPart(alpha));
    if (norm) {
      sum = 0.0;
      for (j=0; j<n; j++)
	sum += PetscRealPart(H[j] * PetscConj(H[j]));
      *norm = PetscRealPart(alpha)-sum;
      if (*norm < 0.0 || eps->compute_norm) {
	ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr);
      } else *norm = sqrt(*norm);
    }
    break;
  case EPS_MGS_ORTH:
    /* compute |v| */
    if (onorm) { ierr = STNorm(eps->OP,v,onorm);CHKERRQ(ierr); }
    for (j=0; j<n; j++) {
      /* h_j = ( v, v_j ) */
      ierr = STInnerProduct(eps->OP,v,V[j],&H[j]);CHKERRQ(ierr);
      /* v <- v - h_j v_j */
      ierr = VecAXPY(v,-H[j],V[j]);CHKERRQ(ierr);
    }
    /* compute |v'| */
    if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
    break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
  }
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
  Vec            w = PETSC_NULL;
  PetscScalar    lh[100],*h,lc[100],*c;
  PetscTruth     allocatedh = PETSC_FALSE,allocatedc = PETSC_FALSE;
  PetscReal      lonrm,*onrm,lnrm,*nrm;
  int            j,k;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = STNorm(eps->OP,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = PetscLogEventBegin(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);

  /* allocate H, c and w if needed */
  if (!H) {
    if (n<=100) h = lh;
    else {
      ierr = PetscMalloc(n*sizeof(PetscScalar),&h);CHKERRQ(ierr);
      allocatedh = PETSC_TRUE;
    }
  } else h = H;
  if (eps->orthog_ref != EPS_ORTH_REFINE_NEVER) {
    if (n<=100) c = lc;
    else {
      ierr = PetscMalloc(n*sizeof(PetscScalar),&c);CHKERRQ(ierr);
      allocatedc = PETSC_TRUE;
    }
  }
  if (eps->orthog_type != EPS_MGS_ORTH) {
    ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  }

  /* retrieve onrm and nrm for linear dependence check or conditional refinement */
  if (lindep || eps->orthog_ref == EPS_ORTH_REFINE_IFNEEDED) {
    onrm = &lonrm;
    if (norm) nrm = norm;
    else nrm = &lnrm;
  } else {
    onrm = PETSC_NULL;
    nrm = norm;
  }

  /* orthogonalize and compute onorm */
  eps->count_orthog++;
  switch (eps->orthog_ref) {
  case EPS_ORTH_REFINE_NEVER:
    ierr = EPSOrthogonalizeGS(eps,n,V,v,h,PETSC_NULL,PETSC_NULL,w);CHKERRQ(ierr);
    /* compute |v| */
    if (nrm) { ierr = STNorm(eps->OP,v,nrm);CHKERRQ(ierr); }
    break;
  case EPS_ORTH_REFINE_ALWAYS:
    ierr = EPSOrthogonalizeGS(eps,n,V,v,h,PETSC_NULL,PETSC_NULL,w);CHKERRQ(ierr); 
    eps->count_reorthog++;
    ierr = EPSOrthogonalizeGS(eps,n,V,v,c,onrm,nrm,w);CHKERRQ(ierr); 
    for (j=0;j<n;j++)
      h[j] += c[j];
    break;
  case EPS_ORTH_REFINE_IFNEEDED:
    ierr = EPSOrthogonalizeGS(eps,n,V,v,h,onrm,nrm,w);CHKERRQ(ierr); 
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && *nrm < eps->orthog_eta * *onrm) {
      k++;
      eps->count_reorthog++;
      switch (eps->orthog_type) {
      case EPS_CGS_ORTH:
        ierr = EPSOrthogonalizeGS(eps,n,V,v,c,onrm,nrm,w);CHKERRQ(ierr); 
        break;
      case EPS_MGS_ORTH:
        *onrm = *nrm;
        ierr = EPSOrthogonalizeGS(eps,n,V,v,c,PETSC_NULL,nrm,w);CHKERRQ(ierr); 
	break;
      default:
	SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
      }        
      for (j=0;j<n;j++)
	h[j] += c[j];
    }
    break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization refinement");
  }

  /* check linear dependence */
  if (lindep) {
    if (eps->orthog_ref != EPS_ORTH_REFINE_NEVER && *nrm < eps->orthog_eta * *onrm)
      *lindep = PETSC_TRUE;
    else 
      *lindep = PETSC_FALSE;
  }

  /* free work space */
  if (allocatedc) { ierr = PetscFree(c);CHKERRQ(ierr); }
  if (allocatedh) { ierr = PetscFree(h);CHKERRQ(ierr); }
  if (w) { ierr = VecDestroy(w);CHKERRQ(ierr); }
        
  ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
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
      default:
        SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
    
    if (allocated) { ierr = PetscFree(h);CHKERRQ(ierr); }
    
    ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

