/*
     Routines related to orthogonalization.
     See the SLEPc Technical Report STR-1 for a detailed explanation.
*/
#include "src/ip/ipimpl.h"      /*I "slepcip.h" I*/
#include "slepcblaslapack.h"

/* 
   IPOrthogonalizeGS - Compute |v'|, |v| and one step of CGS or MGS 
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeGS"
PetscErrorCode IPOrthogonalizeGS(IP ip,int n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm,Vec w)
{
  PetscErrorCode ierr;
  int            j;
  PetscScalar    alpha;
  PetscReal      sum;
  
  PetscFunctionBegin;
  switch (ip->orthog_type) {
  
  case IP_CGS_ORTH:
    /* h = W^* v ; alpha = (v , v) */
    if (which) { /* use select array */
      for (j=0; j<n; j++) 
        if (which[j]) { 
	  ierr = IPInnerProductBegin(ip,v,V[j],&H[j]);CHKERRQ(ierr); 
	}
      if (onorm || norm) {
	ierr = IPInnerProductBegin(ip,v,v,&alpha);CHKERRQ(ierr); 
      }
      for (j=0; j<n; j++) 
        if (which[j]) { ierr = IPInnerProductEnd(ip,v,V[j],&H[j]);CHKERRQ(ierr); }
      if (onorm || norm) { ierr = IPInnerProductEnd(ip,v,v,&alpha);CHKERRQ(ierr); }
    } else { /* merge comunications */
      if (onorm || norm) {
	ierr = IPMInnerProductBegin(ip,n,v,V,H);CHKERRQ(ierr);
	ierr = IPInnerProductBegin(ip,v,v,&alpha);CHKERRQ(ierr); 
	ierr = IPMInnerProductEnd(ip,n,v,V,H);CHKERRQ(ierr);
	ierr = IPInnerProductEnd(ip,v,v,&alpha);CHKERRQ(ierr);
      } else { /* use simpler function */ 
        ierr = IPMInnerProduct(ip,n,v,V,H);CHKERRQ(ierr);
      }
    }

    /* q = v - V h */
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    if (which) {
      for (j=0; j<n; j++) 
        if (which[j]) { ierr = VecAXPY(w,H[j],V[j]);CHKERRQ(ierr); }
    } else {
      ierr = VecMAXPY(w,n,H,V);CHKERRQ(ierr);  
    }
    ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);
    
    /* compute |v| and |v'| */
    if (onorm) *onorm = sqrt(PetscRealPart(alpha));
    if (norm) {
      sum = 0.0;
      for (j=0; j<n; j++)
        if (!which || which[j])
	  sum += PetscRealPart(H[j] * PetscConj(H[j]));
      *norm = PetscRealPart(alpha)-sum;
      if (*norm < 0.0) {
	ierr = IPNorm(ip,v,norm);CHKERRQ(ierr);
      } else *norm = sqrt(*norm);
    }
    break;
    
  case IP_MGS_ORTH:
    /* compute |v| */
    if (onorm) { ierr = IPNorm(ip,v,onorm);CHKERRQ(ierr); }
    for (j=0; j<n; j++)
      if (!which || which[j]) {
	/* h_j = ( v, v_j ) */
	ierr = IPInnerProduct(ip,v,V[j],&H[j]);CHKERRQ(ierr);
	/* v <- v - h_j v_j */
	ierr = VecAXPY(v,-H[j],V[j]);CHKERRQ(ierr);
      }
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
-  V      - set of vectors

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
PetscErrorCode IPOrthogonalize(IP ip,int n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *lindep)
{
  PetscErrorCode ierr;
  Vec            w = PETSC_NULL;
  PetscScalar    lh[100],*h,lc[100],*c;
  PetscTruth     allocatedh = PETSC_FALSE,allocatedc = PETSC_FALSE;
  PetscReal      onrm,nrm;
  int            j,k;
  PetscFunctionBegin;
  if (n==0) {
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);

  /* allocate H, c and w if needed */
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
  if (ip->orthog_type != IP_MGS_ORTH) {
    ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
  }

  /* orthogonalize and compute onorm */
  switch (ip->orthog_ref) {
  
  case IP_ORTH_REFINE_NEVER:
    ierr = IPOrthogonalizeGS(ip,n,which,V,v,h,PETSC_NULL,PETSC_NULL,w);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTH_REFINE_ALWAYS:
    ierr = IPOrthogonalizeGS(ip,n,which,V,v,h,PETSC_NULL,PETSC_NULL,w);CHKERRQ(ierr); 
    if (lindep) {
      ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,&onrm,&nrm,w);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      if (nrm < ip->orthog_eta * onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    } else {
      ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,PETSC_NULL,norm,w);CHKERRQ(ierr);
    }
    for (j=0;j<n;j++) 
      if (!which || which[j]) h[j] += c[j];
    break;
  
  case IP_ORTH_REFINE_IFNEEDED:
    ierr = IPOrthogonalizeGS(ip,n,which,V,v,h,&onrm,&nrm,w);CHKERRQ(ierr); 
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && nrm < ip->orthog_eta * onrm) {
      k++;
      switch (ip->orthog_type) {
      case IP_CGS_ORTH:
        ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,&onrm,&nrm,w);CHKERRQ(ierr); 
        break;
      case IP_MGS_ORTH:
        onrm = nrm;
        ierr = IPOrthogonalizeGS(ip,n,which,V,v,c,PETSC_NULL,&nrm,w);CHKERRQ(ierr); 
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
  if (w) { ierr = VecDestroy(w);CHKERRQ(ierr); }
        
  ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
