/*
     Routines related to orthogonalization.
     See the SLEPc Technical Report STR-1 for a detailed explanation.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/ip/ipimpl.h"      /*I "slepcip.h" I*/
#include "slepcblaslapack.h"

/* 
   IPOrthogonalizeMGS - Compute one step of Modified Gram-Schmidt 
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeMGS"
static PetscErrorCode IPOrthogonalizeMGS(IP ip,int n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H)
{
  PetscErrorCode ierr;
  int            j;
  
  PetscFunctionBegin;
  for (j=0; j<n; j++)
    if (!which || which[j]) {
      /* h_j = ( v, v_j ) */
      ierr = IPInnerProduct(ip,v,V[j],&H[j]);CHKERRQ(ierr);
      /* v <- v - h_j v_j */
      ierr = VecAXPY(v,-H[j],V[j]);CHKERRQ(ierr);
    }
  PetscFunctionReturn(0);
}

/* 
   IPOrthogonalizeCGS - Compute |v'| (estimated), |v| and one step of CGS with only one global synchronization
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeCGS"
PetscErrorCode IPOrthogonalizeCGS(IP ip,int n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm,Vec work)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    alpha;
  PetscReal      sum;
  
  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v , v) */
  if (which) { 
  /* use select array */
    for (j=0; j<n; j++) 
      if (which[j]) { ierr = IPInnerProductBegin(ip,v,V[j],&H[j]);CHKERRQ(ierr); }
    if (onorm || norm) { ierr = IPInnerProductBegin(ip,v,v,&alpha);CHKERRQ(ierr); }
    for (j=0; j<n; j++) 
      if (which[j]) { ierr = IPInnerProductEnd(ip,v,V[j],&H[j]);CHKERRQ(ierr); }
    if (onorm || norm) { ierr = IPInnerProductEnd(ip,v,v,&alpha);CHKERRQ(ierr); }
  } else { /* merge comunications */
    if (onorm || norm) {
      ierr = IPMInnerProductBegin(ip,v,n,V,H);CHKERRQ(ierr);
      ierr = IPInnerProductBegin(ip,v,v,&alpha);CHKERRQ(ierr); 
      ierr = IPMInnerProductEnd(ip,v,n,V,H);CHKERRQ(ierr);
      ierr = IPInnerProductEnd(ip,v,v,&alpha);CHKERRQ(ierr);
    } else { /* use simpler function */ 
      ierr = IPMInnerProduct(ip,v,n,V,H);CHKERRQ(ierr);
    }
  }

  /* q = v - V h */
  ierr = VecSet(work,0.0);CHKERRQ(ierr);
  if (which) {
    for (j=0; j<n; j++) 
      if (which[j]) { ierr = VecAXPY(work,H[j],V[j]);CHKERRQ(ierr); }
  } else {
    ierr = VecMAXPY(work,n,H,V);CHKERRQ(ierr);  
  }
  ierr = VecAXPY(v,-1.0,work);CHKERRQ(ierr);
    
  /* compute |v| */
  if (onorm) *onorm = sqrt(PetscRealPart(alpha));

  /* compute |v'| */
  if (norm) {
    sum = 0.0;
    for (j=0; j<n; j++)
      if (!which || which[j])
        sum += PetscRealPart(H[j] * PetscConj(H[j]));
    *norm = PetscRealPart(alpha)-sum;
    if (*norm <= 0.0) {
      ierr = IPNorm(ip,v,norm);CHKERRQ(ierr);
    } else *norm = sqrt(*norm);
  }
  PetscFunctionReturn(0);
}

/* 
   IPOrthogonalizeGS - Compute |v'|, |v| and one step of CGS or MGS 
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeGS"
static PetscErrorCode IPOrthogonalizeGS(IP ip,int n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm,Vec work)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  switch (ip->orthog_type) {
  case IP_CGS_ORTH:
    ierr = IPOrthogonalizeCGS(ip,n,which,V,v,H,onorm,norm,work);CHKERRQ(ierr);
    break;
  case IP_MGS_ORTH:
    /* compute |v| */
    if (onorm) { ierr = IPNorm(ip,v,onorm);CHKERRQ(ierr); }
    ierr = IPOrthogonalizeMGS(ip,n,which,V,v,H);CHKERRQ(ierr);
    /* compute |v'| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalize"
/*@
   IPOrthogonalize - Orthogonalize a vector with respect to a set of vectors.

   Collective on IP

   Input Parameters:
+  ip    - the inner product (IP) context
.  n      - number of columns of V
.  which  - logical array indicating columns of V to be used
.  V      - set of vectors
-  work   - workspace vector 

   Input/Output Parameter:
.  v      - (input) vector to be orthogonalized and (output) result of 
            orthogonalization

   Output Parameter:
+  H      - coefficients computed during orthogonalization
.  norm   - norm of the vector after being orthogonalized
-  lindep - flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function applies an orthogonal projector to project vector v onto the
   orthogonal complement of the span of the columns of V.

   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: IPSetOrthogonalization(), IPBiOrthogonalize()
@*/
PetscErrorCode IPOrthogonalize(IP ip,int n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *lindep,Vec work)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h,lc[100],*c;
  PetscTruth     allocatedh = PETSC_FALSE,allocatedc = PETSC_FALSE,allocatedw = PETSC_FALSE;
  PetscReal      onrm,nrm;
  int            j,k;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);

  /* allocate H, c and work if needed */
  if (!H) {
    if (n<=100) h = lh;
    else {
      ierr = PetscMalloc(n*sizeof(PetscScalar),&h);CHKERRQ(ierr);
      allocatedh = PETSC_TRUE;
    }
  } else h = H;
  if (ip->orthog_ref != IP_ORTH_REFINE_NEVER) {
    if (n<=100) c = lc;
    else {
      ierr = PetscMalloc(n*sizeof(PetscScalar),&c);CHKERRQ(ierr);
      allocatedc = PETSC_TRUE;
    }
  }
  if (ip->orthog_type == IP_CGS_ORTH) {
    ierr = VecDuplicate(v,&work);CHKERRQ(ierr);
    allocatedw = PETSC_TRUE;
  }

  /* orthogonalize and compute onorm */
  switch (ip->orthog_ref) {
  
  case IP_ORTH_REFINE_NEVER:
    ierr = IPOrthogonalizeGS(ip,n,which,V,v,h,PETSC_NULL,PETSC_NULL,work);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTH_REFINE_ALWAYS:
    ierr = IPOrthogonalizeGS(ip,n,which,V,v,h,PETSC_NULL,PETSC_NULL,work);CHKERRQ(ierr); 
    if (lindep) {
      ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,&onrm,&nrm,work);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      if (nrm < ip->orthog_eta * onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    } else {
      ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,PETSC_NULL,norm,work);CHKERRQ(ierr);
    }
    for (j=0;j<n;j++) 
      if (!which || which[j]) h[j] += c[j];
    break;
  
  case IP_ORTH_REFINE_IFNEEDED:
    ierr = IPOrthogonalizeGS(ip,n,which,V,v,h,&onrm,&nrm,work);CHKERRQ(ierr); 
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && nrm < ip->orthog_eta * onrm) {
      k++;
      switch (ip->orthog_type) {
      case IP_CGS_ORTH:
        ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,&onrm,&nrm,work);CHKERRQ(ierr); 
        break;
      case IP_MGS_ORTH:
        onrm = nrm;
        ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,PETSC_NULL,&nrm,work);CHKERRQ(ierr); 
	break;
      default:
	SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
      }        
      for (j=0;j<n;j++) 
	if (!which || which[j]) h[j] += c[j];
    }
    if (norm) *norm = nrm;
    if (lindep) {
      if (nrm < ip->orthog_eta * onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
        
    break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization refinement");
  }

  /* free work space */
  if (allocatedc) { ierr = PetscFree(c);CHKERRQ(ierr); }
  if (allocatedh) { ierr = PetscFree(h);CHKERRQ(ierr); }
  if (allocatedw) { ierr = VecDestroy(work);CHKERRQ(ierr); }
        
  ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPQRDecomposition"
/*@
   IPQRDecomposition - Compute the QR factorization of a set of vectors.

   Collective on IP

   Input Parameters:
+  ip - the eigenproblem solver context
.  V - set of vectors
.  m - starting column
.  n - ending column
.  ldr - leading dimension of R
-  work - workspace vector 

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

.seealso: IPOrthogonalize(), IPNorm(), IPInnerProduct().
@*/
PetscErrorCode IPQRDecomposition(IP ip,Vec *V,int m,int n,PetscScalar *R,int ldr,Vec work)
{
  PetscErrorCode ierr;
  int            k;
  PetscReal      norm;
  PetscTruth     lindep;
  
  PetscFunctionBegin;

  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (R) { ierr = IPOrthogonalize(ip,k,PETSC_NULL,V,V[k],&R[0+ldr*k],&norm,&lindep,work);CHKERRQ(ierr); }
    else   { ierr = IPOrthogonalize(ip,k,PETSC_NULL,V,V[k],PETSC_NULL,&norm,&lindep,work);CHKERRQ(ierr); }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (norm==0.0 || lindep) { 
      PetscInfo(ip,"Linearly dependent vector found, generating a new random vector\n");
      ierr = SlepcVecSetRandom(V[k]);CHKERRQ(ierr);
      ierr = IPNorm(ip,V[k],&norm);CHKERRQ(ierr);
    }
    ierr = VecScale(V[k],1.0/norm);CHKERRQ(ierr);
    if (R) R[k+ldr*k] = norm;

  }

  PetscFunctionReturn(0);
}

/*
    Biorthogonalization routine using classical Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "IPCGSBiOrthogonalization"
static PetscErrorCode IPCGSBiOrthogonalization(IP ip,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
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
    ierr = IPMInnerProduct(ip,V[j],n,W,vw+j*n);CHKERRQ(ierr);
  }
  lwork = n;
  ierr = PetscMalloc(n*sizeof(PetscScalar),&tau);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  LAPACKgelqf_(&n,&n,vw,&n,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xGELQF %i",info);
  
  /*** First orthogonalization ***/

  /* h = W^* v */
  /* q = v - V h */
  ierr = IPMInnerProduct(ip,v,n,W,H);CHKERRQ(ierr);
  BLAStrsm_("L","L","N","N",&n,&ione,&one,vw,&n,H,&n,1,1,1,1);
  LAPACKormlq_("L","N",&n,&ione,&n,vw,&n,tau,H,&n,work,&lwork,&info,1,1);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORMLQ %i",info);
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,n,H,V);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);

  /* compute norm of v */
  if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
  
  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = PetscFree(vw);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "IPBiOrthogonalize"
/*@
   IPBiOrthogonalize - Bi-orthogonalize a vector with respect to a set of vectors.

   Collective on IP

   Input Parameters:
+  ip - the inner product context
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

.seealso: IPSetOrthogonalization(), IPOrthogonalize()
@*/
PetscErrorCode IPBiOrthogonalize(IP ip,int n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h;
  PetscTruth     allocated = PETSC_FALSE;
  PetscReal      lhnrm,*hnrm,lnrm,*nrm;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
  } else {
    ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
    
    /* allocate H if needed */
    if (!H) {
      if (n<=100) h = lh;
      else {
        ierr = PetscMalloc(n*sizeof(PetscScalar),&h);CHKERRQ(ierr);
        allocated = PETSC_TRUE;
      }
    } else h = H;
    
    /* retrieve hnrm and nrm for linear dependence check or conditional refinement */
    if (ip->orthog_ref == IP_ORTH_REFINE_IFNEEDED) {
      hnrm = &lhnrm;
      if (norm) nrm = norm;
      else nrm = &lnrm;
    } else {
      hnrm = PETSC_NULL;
      nrm = norm;
    }
    
    switch (ip->orthog_type) {
      case IP_CGS_ORTH:
        ierr = IPCGSBiOrthogonalization(ip,n,V,W,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
    
    if (allocated) { ierr = PetscFree(h);CHKERRQ(ierr); }
    
    ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

