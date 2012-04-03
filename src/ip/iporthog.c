/*
     Routines related to orthogonalization.
     See the SLEPc Technical Report STR-1 for a detailed explanation.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/ipimpl.h>      /*I "slepcip.h" I*/
#include <slepcblaslapack.h>

/* 
   IPOrthogonalizeMGS1 - Compute one step of Modified Gram-Schmidt 
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeMGS1"
static PetscErrorCode IPOrthogonalizeMGS1(IP ip,PetscInt n,PetscBool *which,Vec *V,Vec v,PetscScalar *H)
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
PetscErrorCode IPOrthogonalizeCGS1(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscBool *which,Vec *V,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
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
  if (onorm) *onorm = PetscSqrtReal(PetscRealPart(alpha));

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
      } else *norm = PetscSqrtReal(*norm);
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
static PetscErrorCode IPOrthogonalizeMGS(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscBool *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
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
  
  case IP_ORTHOG_REFINE_NEVER:
    ierr = IPOrthogonalizeMGS1(ip,nds,PETSC_NULL,DS,v,PETSC_NULL);CHKERRQ(ierr);
    ierr = IPOrthogonalizeMGS1(ip,n,which,V,v,H);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTHOG_REFINE_ALWAYS:
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
  
  case IP_ORTHOG_REFINE_IFNEEDED:
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
    SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unknown orthogonalization refinement");
  }
  PetscFunctionReturn(0);
}

/*
  IPOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt
*/
#undef __FUNCT__  
#define __FUNCT__ "IPOrthogonalizeCGS"
static PetscErrorCode IPOrthogonalizeCGS(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscBool *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h,lc[100],*c;
  PetscBool      allocatedh = PETSC_FALSE,allocatedc = PETSC_FALSE;
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
  if (ip->orthog_ref != IP_ORTHOG_REFINE_NEVER) {
    if (nds+n<=100) c = lc;
    else {
      ierr = PetscMalloc((nds+n)*sizeof(PetscScalar),&c);CHKERRQ(ierr);
      allocatedc = PETSC_TRUE;
    }
  }

  /* orthogonalize and compute onorm */
  switch (ip->orthog_ref) {
  
  case IP_ORTHOG_REFINE_NEVER:
    ierr = IPOrthogonalizeCGS1(ip,nds,DS,n,which,V,v,h,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTHOG_REFINE_ALWAYS:
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
  
  case IP_ORTHOG_REFINE_IFNEEDED:
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
    SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unknown orthogonalization refinement");
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
   IPOrthogonalize - Orthogonalize a vector with respect to a set of vectors.

   Collective on IP and Vec

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
   orthogonal complement of the span of the columns of DS and V. The columns
   of DS and V are assumed to be mutually orthonormal.

   On exit, v = v0 - V*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: IPSetOrthogonalization(), IPBiOrthogonalize()
@*/
PetscErrorCode IPOrthogonalize(IP ip,PetscInt nds,Vec *DS,PetscInt n,PetscBool *which,Vec *V,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ip,nds,2);
  PetscValidLogicalCollectiveInt(ip,n,4);
  ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  if (nds==0 && n==0) {
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
  } else {
    switch (ip->orthog_type) {
    case IP_ORTHOG_CGS:
      ierr = IPOrthogonalizeCGS(ip,nds,DS,n,which,V,v,H,norm,lindep);CHKERRQ(ierr); 
      break;
    case IP_ORTHOG_MGS:
      ierr = IPOrthogonalizeMGS(ip,nds,DS,n,which,V,v,H,norm,lindep);CHKERRQ(ierr); 
      break;
    default:
      SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
  }
  ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPQRDecomposition"
/*@
   IPQRDecomposition - Compute the QR factorization of a set of vectors.

   Collective on IP and Vec

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
  PetscBool      lindep;
  PetscRandom    rctx=PETSC_NULL;
  
  PetscFunctionBegin;
  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (R) { ierr = IPOrthogonalize(ip,0,PETSC_NULL,k,PETSC_NULL,V,V[k],&R[0+ldr*k],&norm,&lindep);CHKERRQ(ierr); }
    else   { ierr = IPOrthogonalize(ip,0,PETSC_NULL,k,PETSC_NULL,V,V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (norm==0.0 || lindep) { 
      ierr = PetscInfo(ip,"Linearly dependent vector found, generating a new random vector\n");CHKERRQ(ierr);
      if (!rctx) {
        ierr = PetscRandomCreate(((PetscObject)ip)->comm,&rctx);CHKERRQ(ierr);
        ierr = PetscRandomSetSeed(rctx,0x12345678);CHKERRQ(ierr);
        ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
      }
      ierr = SlepcVecSetRandom(V[k],rctx);CHKERRQ(ierr);
      ierr = IPNorm(ip,V[k],&norm);CHKERRQ(ierr);
    }
    ierr = VecScale(V[k],1.0/norm);CHKERRQ(ierr);
    if (R) R[k+ldr*k] = norm;

  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

