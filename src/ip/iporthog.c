/*
     Routines related to orthogonalization.
     See the SLEPc Technical Report STR-1 for a detailed explanation.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/ipimpl.h"      /*I "slepcip.h" I*/
#include "slepcblaslapack.h"

/* 
   IPOrthogonalizeMGS1 - Compute one step of Modified Gram-Schmidt 
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeMGS1"
static PetscErrorCode IPOrthogonalizeMGS1(IP ip,PetscInt n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H)
{
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    dot;
  
  PetscFunctionBegin;
  for (j=0; j<n; j++)
    if (!which || which[j]) {
      /* h_j = ( v, v_j ) */
      ierr = IPInnerProduct(ip,v,V[j],&dot);CHKERRQ(ierr);
      /* v <- v - h_j v_j */
      ierr = VecAXPY(v,-dot,V[j]);CHKERRQ(ierr);
      if (H) H[j] += dot;
    }
  PetscFunctionReturn(0);
}

/* 
   IPOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with only one global synchronization
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeCGS1"
PetscErrorCode IPOrthogonalizeCGS1(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    alpha;
  PetscReal      sum;

  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v , v) */
  if (nds==0 && !which && !onorm && !norm) { 
    /* use simpler function */ 
    ierr = IPMInnerProduct(ip,v,n,V,H);CHKERRQ(ierr);
  } else {  
    /* merge comunications */
    ierr = IPMInnerProductBegin(ip,v,nds,DS,H);CHKERRQ(ierr); 
    if (which) { /* use select array */
      for (j=0; j<n; j++) 
        if (which[j]) { ierr = IPInnerProductBegin(ip,v,V[j],H+nds+j);CHKERRQ(ierr); }
    } else {
      ierr = IPMInnerProductBegin(ip,v,n,V,H+nds);CHKERRQ(ierr);
    }
    if (onorm || (norm && !ip->matrix)) { 
      ierr = IPInnerProductBegin(ip,v,v,&alpha);CHKERRQ(ierr); 
    }

    ierr = IPMInnerProductEnd(ip,v,nds,DS,H);CHKERRQ(ierr); 
    if (which) { /* use select array */
      for (j=0; j<n; j++) 
        if (which[j]) { ierr = IPInnerProductEnd(ip,v,V[j],H+nds+j);CHKERRQ(ierr); }
    } else {
      ierr = IPMInnerProductEnd(ip,v,n,V,H+nds);CHKERRQ(ierr);
    }
    if (onorm || (norm && !ip->matrix)) {
      ierr = IPInnerProductEnd(ip,v,v,&alpha);CHKERRQ(ierr);
    }
  }

  /* q = v - V h */
  ierr = SlepcVecMAXPBY(v,1.0,-1.0,nds,H,DS);CHKERRQ(ierr);
  if (which) {
    for (j=0; j<n; j++) 
      if (which[j]) { ierr = VecAXPBY(v,-H[nds+j],1.0,V[j]);CHKERRQ(ierr); }
  } else {
    ierr = SlepcVecMAXPBY(v,1.0,-1.0,n,H+nds,V);CHKERRQ(ierr);
  }
    
  /* compute |v| */
  if (onorm) *onorm = sqrt(PetscRealPart(alpha));

  if (norm) {
    if (!ip->matrix) {
      /* estimate |v'| from |v| */
      sum = 0.0;
      for (j=0; j<nds; j++)
        sum += PetscRealPart(H[j] * PetscConj(H[j]));
      for (j=0; j<n; j++)
        if (!which || which[j])
          sum += PetscRealPart(H[nds+j] * PetscConj(H[nds+j]));
      *norm = PetscRealPart(alpha)-sum;
      if (*norm <= 0.0) {
        ierr = IPNorm(ip,v,norm);CHKERRQ(ierr);
      } else *norm = sqrt(*norm);
    } else {
      /* compute |v'| */
      ierr = IPNorm(ip,v,norm);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* 
  IPOrthogonalizeMGS - Orthogonalize with modified Gram-Schmidt
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeMGS"
static PetscErrorCode IPOrthogonalizeMGS(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *lindep)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  PetscReal      onrm,nrm;

  PetscFunctionBegin;
  
  if (H) { 
    for (i=0;i<n;i++) 
      H[i] = 0; 
  }
  
  switch (ip->orthog_ref) {
  
  case IP_ORTH_REFINE_NEVER:
    ierr = IPOrthogonalizeMGS1(ip,nds,PETSC_NULL,DS,v,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPOrthogonalizeMGS1(ip,n,which,V,v,H);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTH_REFINE_ALWAYS:
    /* first step */
    ierr = IPOrthogonalizeMGS1(ip,nds,PETSC_NULL,DS,v,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPOrthogonalizeMGS1(ip,n,which,V,v,H);CHKERRQ(ierr);
    if (lindep) { ierr = IPNorm(ip,v,&onrm);CHKERRQ(ierr); }
    /* second step */
    ierr = IPOrthogonalizeMGS1(ip,nds,PETSC_NULL,DS,v,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPOrthogonalizeMGS1(ip,n,which,V,v,H);CHKERRQ(ierr);
    if (lindep || norm) { ierr = IPNorm(ip,v,&nrm);CHKERRQ(ierr); }
    if (lindep) {
      if (nrm < ip->orthog_eta * onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    if (norm) *norm = nrm;
    break;
  
  case IP_ORTH_REFINE_IFNEEDED:
    /* first step */
    ierr = IPNorm(ip,v,&onrm);CHKERRQ(ierr);
    ierr = IPOrthogonalizeMGS1(ip,nds,PETSC_NULL,DS,v,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPOrthogonalizeMGS1(ip,n,which,V,v,H);CHKERRQ(ierr);
    ierr = IPNorm(ip,v,&nrm);CHKERRQ(ierr);
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && nrm < ip->orthog_eta * onrm) {
      k++;
      onrm = nrm;
      ierr = IPOrthogonalizeMGS1(ip,nds,PETSC_NULL,DS,v,PETSC_NULL);CHKERRQ(ierr);
      ierr = IPOrthogonalizeMGS1(ip,n,which,V,v,H);CHKERRQ(ierr);
      ierr = IPNorm(ip,v,&nrm);CHKERRQ(ierr);
    }
    if (lindep) {
      if (nrm < ip->orthog_eta * onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    if (norm) *norm = nrm;
    break;
    
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization refinement");
  }

  PetscFunctionReturn(0);
}

/*
  IPOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeCGS"
static PetscErrorCode IPOrthogonalizeCGS(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h,lc[100],*c;
  PetscTruth     allocatedh = PETSC_FALSE,allocatedc = PETSC_FALSE;
  PetscReal      onrm,nrm;
  PetscInt       j,k;

  PetscFunctionBegin;
  /* allocate h and c if needed */
  if (!H || nds>0) {
    if (nds+n<=100) h = lh;
    else {
      ierr = PetscMalloc((nds+n)*sizeof(PetscScalar),&h);CHKERRQ(ierr);
      allocatedh = PETSC_TRUE;
    }
  } else h = H;
  if (ip->orthog_ref != IP_ORTH_REFINE_NEVER) {
    if (nds+n<=100) c = lc;
    else {
      ierr = PetscMalloc((nds+n)*sizeof(PetscScalar),&c);CHKERRQ(ierr);
      allocatedc = PETSC_TRUE;
    }
  }

  /* orthogonalize and compute onorm */
  switch (ip->orthog_ref) {
  
  case IP_ORTH_REFINE_NEVER:
    ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,h,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTH_REFINE_ALWAYS:
    ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,h,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); 
    if (lindep) {
      ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,c,&onrm,&nrm);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      if (nrm < ip->orthog_eta * onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    } else {
      ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,c,PETSC_NULL,norm);CHKERRQ(ierr);
    }
    for (j=0;j<n;j++) 
      if (!which || which[j]) h[nds+j] += c[nds+j];
    break;
  
  case IP_ORTH_REFINE_IFNEEDED:
    ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,h,&onrm,&nrm);CHKERRQ(ierr); 
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && nrm < ip->orthog_eta * onrm) {
      k++;
      if (!ip->matrix) {
        ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,c,&onrm,&nrm);CHKERRQ(ierr); 
      } else {
        onrm = nrm;
        ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,c,PETSC_NULL,&nrm);CHKERRQ(ierr); 
      }
      for (j=0;j<n;j++) 
	if (!which || which[j]) h[nds+j] += c[nds+j];
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

  /* recover H from workspace */
  if (H && nds>0) {
    for (j=0;j<n;j++) 
      if (!which || which[j]) H[j] = h[nds+j];
  }

  /* free work space */
  if (allocatedc) { ierr = PetscFree(c);CHKERRQ(ierr); }
  if (allocatedh) { ierr = PetscFree(h);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}        

#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalize"
/*@
   IPOrthogonalize - Orthogonalize a vector with respect to two set of vectors.

   Collective on IP

   Input Parameters:
+  ip     - the inner product (IP) context
.  nds    - number of columns of DS
.  DS     - first set of vectors
.  n      - number of columns of V
.  which  - logical array indicating columns of V to be used
-  V      - second set of vectors

   Input/Output Parameter:
.  v      - (input) vector to be orthogonalized and (output) result of 
            orthogonalization

   Output Parameter:
+  H      - coefficients computed during orthogonalization with V
.  norm   - norm of the vector after being orthogonalized
-  lindep - flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function applies an orthogonal projector to project vector v onto the
   orthogonal complement of the span of the columns of DS and V.

   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: IPSetOrthogonalization(), IPBiOrthogonalize()
@*/
PetscErrorCode IPOrthogonalize(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscTruth *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscTruth *lindep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  
  if (nds==0 && n==0) {
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
  } else {
    switch (ip->orthog_type) {
    case IP_ORTH_CGS:
      ierr = IPOrthogonalizeCGS(ip,nds,DS,n,which,V,v,H,norm,lindep);CHKERRQ(ierr); 
      break;
    case IP_ORTH_MGS:
      ierr = IPOrthogonalizeMGS(ip,nds,DS,n,which,V,v,H,norm,lindep);CHKERRQ(ierr); 
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
  }
  
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

.seealso: IPOrthogonalize(), IPNorm(), IPInnerProduct().
@*/
PetscErrorCode IPQRDecomposition(IP ip,Vec *V,PetscInt m,PetscInt n,PetscScalar *R,PetscInt ldr)
{
  PetscErrorCode ierr;
  PetscInt       k;
  PetscReal      norm;
  PetscTruth     lindep;
  
  PetscFunctionBegin;

  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (R) { ierr = IPOrthogonalize(ip,0,PETSC_NULL,k,PETSC_NULL,V,V[k],&R[0+ldr*k],&norm,&lindep);CHKERRQ(ierr); }
    else   { ierr = IPOrthogonalize(ip,0,PETSC_NULL,k,PETSC_NULL,V,V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (norm==0.0 || lindep) { 
      PetscInfo(ip,"Linearly dependent vector found, generating a new random vector\n");
      ierr = SlepcVecSetRandom(V[k],PETSC_NULL);CHKERRQ(ierr);
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
static PetscErrorCode IPCGSBiOrthogonalization(IP ip,PetscInt n_,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
#if defined(SLEPC_MISSING_LAPACK_GELQF) || defined(SLEPC_MISSING_LAPACK_ORMLQ)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"xGELQF - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   j,ione=1,lwork,info,n=n_;
  PetscScalar    shh[100],*lhh,*vw,*tau,one=1.0,*work;

  PetscFunctionBegin;

  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else { ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = PetscMalloc(n*n*sizeof(PetscScalar),&vw);CHKERRQ(ierr);
  
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
  BLAStrsm_("L","L","N","N",&n,&ione,&one,vw,&n,H,&n);
  LAPACKormlq_("L","N",&n,&ione,&n,vw,&n,tau,H,&n,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xORMLQ %i",info);
  ierr = SlepcVecMAXPBY(v,1.0,-1.0,n,H,V);CHKERRQ(ierr);

  /* compute norm of v */
  if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
  
  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = PetscFree(vw);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
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
PetscErrorCode IPBiOrthogonalize(IP ip,PetscInt n,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *norm)
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
      case IP_ORTH_CGS:
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

