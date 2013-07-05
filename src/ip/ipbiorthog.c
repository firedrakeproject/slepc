/*
     Routines related to bi-orthogonalization.
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

#include <slepc-private/ipimpl.h>      /*I "slepcip.h" I*/
#include <slepcblaslapack.h>

/*
    Biorthogonalization routine using classical Gram-Schmidt with refinement.
 */
#undef __FUNCT__
#define __FUNCT__ "IPCGSBiOrthogonalization"
static PetscErrorCode IPCGSBiOrthogonalization(IP ip,PetscInt n_,Vec *V,Vec *W,Vec v,PetscScalar *H,PetscReal *hnorm,PetscReal *norm)
{
#if defined(SLEPC_MISSING_LAPACK_GELQF) || defined(SLEPC_MISSING_LAPACK_ORMLQ)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GELQF/ORMLQ - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   j,ione=1,lwork,info,n=n_;
  PetscScalar    shh[100],*lhh,*vw,*tau,one=1.0,*work;

  PetscFunctionBegin;
  /* Don't allocate small arrays */
  if (n<=100) lhh = shh;
  else {
    ierr = PetscMalloc(n*sizeof(PetscScalar),&lhh);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(n*n*sizeof(PetscScalar),&vw);CHKERRQ(ierr);

  for (j=0;j<n;j++) {
    ierr = IPMInnerProduct(ip,V[j],n,W,vw+j*n);CHKERRQ(ierr);
  }
  lwork = n;
  ierr = PetscMalloc(n*sizeof(PetscScalar),&tau);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgelqf",LAPACKgelqf_(&n,&n,vw,&n,tau,work,&lwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PetscObjectComm((PetscObject)ip),PETSC_ERR_LIB,"Error in Lapack xGELQF %d",info);

  /*** First orthogonalization ***/

  /* h = W^* v */
  /* q = v - V h */
  ierr = IPMInnerProduct(ip,v,n,W,H);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("L","L","N","N",&n,&ione,&one,vw,&n,H,&n));
  PetscStackCallBLAS("LAPACKormlq",LAPACKormlq_("L","N",&n,&ione,&n,vw,&n,tau,H,&n,work,&lwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PetscObjectComm((PetscObject)ip),PETSC_ERR_LIB,"Error in Lapack xORMLQ %d",info);
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

   Collective on IP and Vec

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
  PetscBool      allocated = PETSC_FALSE;
  PetscReal      lhnrm,*hnrm,lnrm,*nrm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ip,n,2);
  if (!n) {
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
    if (ip->orthog_ref == IP_ORTHOG_REFINE_IFNEEDED) {
      hnrm = &lhnrm;
      if (norm) nrm = norm;
      else nrm = &lnrm;
    } else {
      hnrm = NULL;
      nrm = norm;
    }

    switch (ip->orthog_type) {
      case IP_ORTHOG_CGS:
        ierr = IPCGSBiOrthogonalization(ip,n,V,W,v,h,hnrm,nrm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)ip),PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }

    if (allocated) { ierr = PetscFree(h);CHKERRQ(ierr); }
    ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   IPPseudoOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with only one global synchronization (indefinite)
*/
#undef __FUNCT__
#define __FUNCT__ "IPPseudoOrthogonalizeCGS1"
PetscErrorCode IPPseudoOrthogonalizeCGS1(IP ip,PetscInt n,Vec *V,PetscReal* omega,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    alpha;
  PetscReal      sum;

  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v , v) */
  if (!onorm && !norm) {
    /* use simpler function */
    ierr = IPMInnerProduct(ip,v,n,V,H);CHKERRQ(ierr);
  } else {
    /* merge comunications */
    ierr = IPMInnerProductBegin(ip,v,n,V,H);CHKERRQ(ierr);
    if (onorm || (norm && !ip->matrix)) {
      ierr = IPInnerProductBegin(ip,v,v,&alpha);CHKERRQ(ierr);
    }

    ierr = IPMInnerProductEnd(ip,v,n,V,H);CHKERRQ(ierr);
    if (onorm || (norm && !ip->matrix)) {
      ierr = IPInnerProductEnd(ip,v,v,&alpha);CHKERRQ(ierr);
    }
  }

  /* q = v - V h */
  for (j=0;j<n;j++) H[j] /= omega[j];  /* apply inverse of signature */
  ierr = SlepcVecMAXPBY(v,1.0,-1.0,n,H,V);CHKERRQ(ierr);
  for (j=0;j<n;j++) H[j] *= omega[j];  /* revert signature */

  /* compute |v| */
  if (onorm) {
    if (PetscRealPart(alpha)>0.0) *onorm = PetscSqrtReal(PetscRealPart(alpha));
    else *onorm = -PetscSqrtReal(-PetscRealPart(alpha));
  }

  if (norm) {
    if (!ip->matrix) {
      /* estimate |v'| from |v| */
      sum = 0.0;
      for (j=0; j<n; j++)
        sum += PetscRealPart(H[j] * PetscConj(H[j]));
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
  IPPseudoOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt (indefinite)
*/
#undef __FUNCT__
#define __FUNCT__ "IPPseudoOrthogonalizeCGS"
static PetscErrorCode IPPseudoOrthogonalizeCGS(IP ip,PetscInt n,Vec *V,PetscReal *omega,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    *h,*c;
  PetscReal      onrm,nrm;
  PetscInt       sz=0,sz1,j,k;

  PetscFunctionBegin;
  /* allocate h and c if needed */
  if (!H) sz = n;
  sz1 = sz;
  if (ip->orthog_ref != IP_ORTHOG_REFINE_NEVER) sz += n;
  if (sz>ip->lwork) {
    ierr = PetscFree(ip->work);CHKERRQ(ierr);
    ierr = PetscMalloc(sz*sizeof(PetscScalar),&ip->work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ip,(sz-ip->lwork)*sizeof(PetscScalar));CHKERRQ(ierr);
    ip->lwork = sz;
  }
  if (!H) h = ip->work;
  else h = H;
  if (ip->orthog_ref != IP_ORTHOG_REFINE_NEVER) c = ip->work + sz1;

  /* orthogonalize and compute onorm */
  switch (ip->orthog_ref) {

  case IP_ORTHOG_REFINE_NEVER:
    ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,h,NULL,NULL);CHKERRQ(ierr);
    /* compute |v| */
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    /* linear dependence check does not work without refinement */
    if (lindep) *lindep = PETSC_FALSE;
    break;

  case IP_ORTHOG_REFINE_ALWAYS:
    ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,h,NULL,NULL);CHKERRQ(ierr);
    if (lindep) {
      ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,c,&onrm,&nrm);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      if (PetscAbs(nrm) < ip->orthog_eta * PetscAbs(onrm)) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    } else {
      ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,c,NULL,norm);CHKERRQ(ierr);
    }
    for (j=0;j<n;j++)
      h[j] += c[j];
    break;

  case IP_ORTHOG_REFINE_IFNEEDED:
    ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,h,&onrm,&nrm);CHKERRQ(ierr);
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && PetscAbs(nrm) < ip->orthog_eta * PetscAbs(onrm)) {
      k++;
      if (!ip->matrix) {
        ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,c,&onrm,&nrm);CHKERRQ(ierr);
      } else {
        onrm = nrm;
        ierr = IPPseudoOrthogonalizeCGS1(ip,n,V,omega,v,c,NULL,&nrm);CHKERRQ(ierr);
      }
      for (j=0;j<n;j++)
        h[j] += c[j];
    }
    if (norm) *norm = nrm;
    if (lindep) {
      if (PetscAbs(nrm) < ip->orthog_eta * PetscAbs(onrm)) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    break;

  default:
    SETERRQ(PetscObjectComm((PetscObject)ip),PETSC_ERR_ARG_WRONG,"Unknown orthogonalization refinement");
  }

  /* recover H from workspace */
  if (H) {
    for (j=0;j<n;j++)
      H[j] = h[j];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPPseudoOrthogonalize"
/*@
   IPPseudoOrthogonalize - Orthogonalize a vector with respect to two set of vectors
   in the sense of a pseudo-inner product.

   Collective on IP and Vec

   Input Parameters:
+  ip     - the inner product (IP) context
.  n      - number of columns of V
.  V      - set of vectors
-  omega  - set of signs that define a signature matrix

   Input/Output Parameter:
.  v      - (input) vector to be orthogonalized and (output) result of
            orthogonalization

   Output Parameter:
+  H      - coefficients computed during orthogonalization
.  norm   - norm of the vector after being orthogonalized
-  lindep - flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function is the analogue of IPOrthogonalize, but for the indefinite
   case. When using an indefinite IP the norm is not well defined, so we
   take the convention of having negative norms in such cases. The
   orthogonalization is then defined by a set of vectors V satisfying
   V'*B*V=Omega, where Omega is a signature matrix diag([+/-1,...,+/-1]).

   On exit, v = v0 - V*Omega*H, where v0 is the original vector v.

   This routine does not normalize the resulting vector. The output
   argument 'norm' may be negative.

   Level: developer

.seealso: IPSetOrthogonalization(), IPOrthogonalize()
@*/
PetscErrorCode IPPseudoOrthogonalize(IP ip,PetscInt n,Vec *V,PetscReal *omega,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ip,n,2);
  ierr = PetscLogEventBegin(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  if (n==0) {
    if (norm) { ierr = IPNorm(ip,v,norm);CHKERRQ(ierr); }
    if (lindep) *lindep = PETSC_FALSE;
  } else {
    switch (ip->orthog_type) {
    case IP_ORTHOG_CGS:
      ierr = IPPseudoOrthogonalizeCGS(ip,n,V,omega,v,H,norm,lindep);CHKERRQ(ierr);
      break;
    case IP_ORTHOG_MGS:
      SETERRQ(PetscObjectComm((PetscObject)ip),PETSC_ERR_SUP,"Modified Gram-Schmidt not implemented for indefinite case");
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ip),PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
    }
  }
  ierr = PetscLogEventEnd(IP_Orthogonalize,ip,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


