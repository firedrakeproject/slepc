/*
     Routines related to orthogonalization.
     See the SLEPc Technical Report STR-1 for a detailed explanation.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include "slepc-private/ipimpl.h"      /*I "slepcip.h" I*/
#include "slepcblaslapack.h"
#include "../src/eps/impls/davidson/common/davidson.h"

#define MyPetscSqrtReal(alpha) (PetscSign(PetscRealPart(alpha))*PetscSqrtReal(PetscAbs(PetscRealPart(alpha))))

/* 
   IPOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with only one global synchronization
*/
#undef __FUNCT__  
#define __FUNCT__ "IPBOrthogonalizeCGS1"
PetscErrorCode IPBOrthogonalizeCGS1(IP ip,PetscInt nds,Vec *defl,Vec *BDS,PetscReal *BDSnorms,PetscInt n,PetscBool *which,Vec *V,Vec *BV,PetscReal *BVnorms,Vec v,Vec Bv,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    alpha;

  PetscFunctionBegin;
  /* h = [defl V]^* Bv ; alpha = (Bv , v) */
  ierr = VecsMultIa(H,0,nds,defl,0,nds,&Bv,0,1);CHKERRQ(ierr); j = nds;
  if (!which) {
    ierr = VecsMultIa(H+j,0,n,V,0,n,&Bv,0,1);CHKERRQ(ierr); j+= n;
  } else {
    for (i=0; i<n; i++) {
      if (which[i]) {
        ierr = VecsMultIa(H+j,0,1,V+i,0,1,&Bv,0,1);CHKERRQ(ierr); j++;
      }
    }
  }
  if (onorm || norm) {
    ierr = VecsMultIa(H+j,0,1,&v,0,1,&Bv,0,1);CHKERRQ(ierr); j++;
  }
  ierr = VecsMultIb(H,0,j,j,1,PETSC_NULL,v);CHKERRQ(ierr);
  if (onorm || norm) alpha = H[j-1]; 

  /* h = J * h */
  if (BDSnorms && defl) for (i=0; i<nds; i++) H[i]*= BDSnorms[i];
  if (BVnorms && V) {
    if (!which) {
      for (i=0; i<n; i++) H[i+nds]*= BVnorms[i];
    } else {
      for (i=j=0; i<n; i++) {
        if (which[i]) H[j++]*= BVnorms[i];
      }
    }
  }

  /* v = v - V h */
  ierr = SlepcVecMAXPBY(v,1.0,-1.0,nds,H,defl);CHKERRQ(ierr);
  if (which) {
    for (j=0; j<n; j++) 
      if (which[j]) { ierr = VecAXPBY(v,-H[nds+j],1.0,V[j]);CHKERRQ(ierr); }
  } else {
    ierr = SlepcVecMAXPBY(v,1.0,-1.0,n,H+nds,V);CHKERRQ(ierr);
  }

  /* Bv = Bv - BV h */
  ierr = SlepcVecMAXPBY(Bv,1.0,-1.0,nds,H,BDS);CHKERRQ(ierr);
  if (which) {
    for (j=0; j<n; j++) 
      if (which[j]) { ierr = VecAXPBY(Bv,-H[nds+j],1.0,BV[j]);CHKERRQ(ierr); }
  } else {
    ierr = SlepcVecMAXPBY(Bv,1.0,-1.0,n,H+nds,BV);CHKERRQ(ierr);
  }

  /* compute |v| */
  if (onorm) *onorm = MyPetscSqrtReal(alpha);

  /* compute |v'| */
  if (norm) {
    ierr = VecDot(Bv, v, &alpha); CHKERRQ(ierr);
    *norm = MyPetscSqrtReal(alpha);
  }
  PetscFunctionReturn(0);
}

/*
  IPOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt
*/
#undef __FUNCT__  
#define __FUNCT__ "IPBOrthogonalizeCGS"
static PetscErrorCode IPBOrthogonalizeCGS(IP ip,PetscInt nds,Vec *defl,Vec *BDS,PetscReal *BDSnorms,PetscInt n,PetscBool *which,Vec *V,Vec *BV,PetscReal *BVnorms,Vec v,Vec Bv,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    lh[100],*h,lc[100],*c,alpha;
  PetscBool      allocatedh = PETSC_FALSE,allocatedc = PETSC_FALSE;
  PetscReal      onrm,nrm;
  PetscInt       j,k;

  PetscFunctionBegin;
  /* allocate h and c if needed */
  if (!H || nds>0) {
    if (nds+n+1<=100) h = lh;
    else {
      ierr = PetscMalloc((nds+n+1)*sizeof(PetscScalar),&h);CHKERRQ(ierr);
      allocatedh = PETSC_TRUE;
    }
  } else h = H;
  if (ip->orthog_ref != IP_ORTHOG_REFINE_NEVER) {
    if (nds+n+1<=100) c = lc;
    else {
      ierr = PetscMalloc((nds+n+1)*sizeof(PetscScalar),&c);CHKERRQ(ierr);
      allocatedc = PETSC_TRUE;
    }
  }

  /* orthogonalize and compute onorm */
  switch (ip->orthog_ref) {
  
  case IP_ORTHOG_REFINE_NEVER:
    ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,h,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) {
      ierr = VecDot(Bv,v,&alpha); CHKERRQ(ierr);
      *norm = MyPetscSqrtReal(alpha);
    }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;
    
  case IP_ORTHOG_REFINE_ALWAYS:
    ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,h,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    if (lindep) {
      ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,c,&onrm,&nrm);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      if (PetscAbs(nrm) < ip->orthog_eta * PetscAbs(onrm)) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    } else {
      ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,c,PETSC_NULL,norm);CHKERRQ(ierr);
    }
    for (j=0;j<n;j++) 
      if (!which || which[j]) h[nds+j] += c[nds+j];
    break;
  
  case IP_ORTHOG_REFINE_IFNEEDED:
    ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,h,&onrm,&nrm);CHKERRQ(ierr); 
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && PetscAbs(nrm) < ip->orthog_eta * PetscAbs(onrm)) {
      k++;
      if (!ip->matrix) {
        ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,c,&onrm,&nrm);CHKERRQ(ierr); 
      } else {
        onrm = nrm;
        ierr = IPBOrthogonalizeCGS1(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,c,PETSC_NULL,&nrm);CHKERRQ(ierr); 
      }
      for (j=0;j<n;j++) 
        if (!which || which[j]) h[nds+j] += c[nds+j];
    }
    if (norm) *norm = nrm;
    if (lindep) {
      if (PetscAbs(nrm) < ip->orthog_eta * PetscAbs(onrm)) *lindep = PETSC_TRUE;
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
#define __FUNCT__ "IPBOrthogonalize"
/*@
   IPBOrthogonalize - B-Orthogonalize a vector with respect to two set of vectors.

   Collective on IP

   Input Parameters:
+  ip     - the inner product (IP) context
.  nds    - number of columns of defl
.  defl   - first set of vectors
.  BDS    - B * defl
.  BDSnorms - DS_i' * B * DS_i
.  n      - number of columns of V
.  which  - logical array indicating columns of V to be used
.  V      - second set of vectors
.  BV     - B * V
-  BVnorms - V_i' * B * V_i

   Input/Output Parameter:
+  v      - (input) vector to be orthogonalized and (output) result of 
            orthogonalization
-  Bv     - (input/output) B * v

   Output Parameter:
+  H      - coefficients computed during orthogonalization with V, of size nds+n
            if norm == PETSC_NULL, and nds+n+1 otherwise.
.  norm   - norm of the vector after being orthogonalized
-  lindep - flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function applies an orthogonal projector to project vector v onto the
   orthogonal complement of the span of the columns of defl and V.

   On exit, v0 = [V v]*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector.

   Level: developer

.seealso: IPSetOrthogonalization(), IPBiOrthogonalize()
@*/
PetscErrorCode IPBOrthogonalize(IP ip,PetscInt nds,Vec *defl, Vec *BDS,PetscReal *BDSnorms,PetscInt n,PetscBool *which,Vec *V,Vec *BV,PetscReal *BVnorms,Vec v,Vec Bv,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    alpha;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ip,nds,2);
  PetscValidLogicalCollectiveInt(ip,n,4);
  ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
 
  /* Bv <- B * v */
  ierr = PetscLogEventBegin(IP_ApplyMatrix,ip,0,0,0);CHKERRQ(ierr);
  ierr = MatMult(ip->matrix, v, Bv); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IP_ApplyMatrix,ip,0,0,0);CHKERRQ(ierr);
   
  if (nds==0 && n==0) {
    if (norm) {
      ierr = VecDot(Bv, v, &alpha); CHKERRQ(ierr);
      *norm = MyPetscSqrtReal(alpha);
    }
    if (lindep) *lindep = PETSC_FALSE;
  } else {
    switch (ip->orthog_type) {
    case IP_ORTHOG_CGS:
      ierr = IPBOrthogonalizeCGS(ip,nds,defl,BDS,BDSnorms,n,which,V,BV,BVnorms,v,Bv,H,norm,lindep);CHKERRQ(ierr); 
      break;
    case IP_ORTHOG_MGS:
      SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unsupported orthogonalization type");
      break;
    default:
      SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
  }
  ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}


