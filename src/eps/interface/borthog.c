/*
    Routines related to orthogonalization.

    See the SLEPc users manual for a detailed explanation.
*/
#include "src/eps/epsimpl.h"    /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSQRDecomposition"
/*@
   EPSQRDecomposition - Compute the QR factorization of the basis vectors.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  V - basis vectors
.  m - starting column
.  n - ending column
-  ldr - leading dimension of R

   Output Parameter:
.  R  - triangular matrix of QR factorization

   Notes:
   Assumes that the first m columns (from 0 to m-1) are already orthonormal
   to working precision.

   Level: developer

@*/
int EPSQRDecomposition(EPS eps,Vec *V,int m,int n,PetscScalar *R,int ldr)
{
  int         ierr,k;
  PetscScalar alpha;
  PetscReal   norm;
  
  PetscFunctionBegin;

  for (k=m; k<n; k++) {

    /* orthogonalize v_k with respect to v_0, ..., v_{k-1} */
    if (k>0) {
      ierr = PetscLogEventBegin(EPS_Orthogonalization,eps,0,0,0);CHKERRQ(ierr);
      if (R) { ierr = (*eps->orthog)(eps,k,V,V[k],&R[0+ldr*k],&norm);CHKERRQ(ierr); }
      else   { ierr = (*eps->orthog)(eps,k,V,V[k],PETSC_NULL,&norm);CHKERRQ(ierr); }
      ierr = PetscLogEventEnd(EPS_Orthogonalization,eps,0,0,0);CHKERRQ(ierr);
    }
    else {
      ierr = VecNorm(V[0],NORM_2,&norm);CHKERRQ(ierr);
    }

    /* normalize v_k: r_{k,k} = ||v_k||_2; v_k = v_k/r_{k,k} */
    if (R) { R[k+ldr*k] = norm; }
    if (norm==0.0) SETERRQ( 1,"Zero vector in QR decomposition" );
    alpha = 1.0/norm;
    ierr = VecScale(&alpha,V[k]);CHKERRQ(ierr);

  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetOrthogonalization"
/*@
   EPSSetOrthogonalization - Specifies the type of orthogonalization technique
   to be used inside the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps        - the eigensolver context 
.  type       - a known type of orthogonalization
.  refinement - type of refinement
-  eta        - parameter for dynamic refinement

   Options Database Keys:
+  -eps_orthog_type <type> -  Where <type> is cgs for Classical Gram-Schmidt
                              orthogonalization (default)
                              or mgs for Modified Gram-Schmidt orthogonalization
.  -eps_orthog_refinement <type> -  Where <type> is one of never, ifneeded
                              (default) or always 
-  -eps_orthog_eta <eta> -  For setting the value of eta (or PETSC_DEFAULT)
    
   Notes:  
   The value of eta is used only when refinement type is "ifneeded". 

   The default orthogonalization technique 
   works well for most problems. MGS is numerically more robust than CGS,
   but CGS may give better scalability.

   Level: intermediate

.seealso: EPSGetOrthogonalization()
@*/
int EPSSetOrthogonalization(EPS eps,EPSOrthogonalizationType type, EPSOrthogonalizationRefinementType refinement, PetscReal eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  switch (type) {
    case EPS_CGS_ORTH:
      eps->orthog = EPSClassicalGramSchmidtOrthogonalization;
      break;
    case EPS_MGS_ORTH:
      eps->orthog = EPSModifiedGramSchmidtOrthogonalization;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
  }
  eps->orth_type = type;
  switch (refinement) {
  case EPS_ORTH_REFINE_NEVER:
    eps->orth_eta = 0;
    break;
  case EPS_ORTH_REFINE_IFNEEDED:
    if (eta == PETSC_DEFAULT) eps->orth_eta = 0.7071;
    else eps->orth_eta = eta;
    break;
  case EPS_ORTH_REFINE_ALWAYS:
    eps->orth_eta = -1;
    break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown refinement type");
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetOrthogonalization"
/*@
   EPSGetOrthogonalization - Gets the orthogonalization type from the 
   EPS object.

   Not Collective

   Input Parameter:
.  eps - Eigensolver context 

   Output Parameter:
+  type       - type of orthogonalization technique
.  refinement - type of refinement
-  eta        - parameter for dynamic refinement

   Level: intermediate

.seealso: EPSSetOrthogonalization()
@*/
int EPSGetOrthogonalization(EPS eps,EPSOrthogonalizationType *type,EPSOrthogonalizationRefinementType *refinement, PetscReal *eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (type) *type = eps->orth_type;
  if (eps->orth_eta == 0) {
    if (refinement) *refinement = EPS_ORTH_REFINE_NEVER;
  } else if (eps->orth_eta < 0) {
    if (refinement) *refinement = EPS_ORTH_REFINE_ALWAYS;
  } else {
    if (refinement) *refinement = EPS_ORTH_REFINE_IFNEEDED;
    if (eta) *eta = eps->orth_eta;
  }
  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using classical Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSClassicalGramSchmidtOrthogonalization"
int EPSClassicalGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *norm)
{
  int         ierr,j;
  PetscScalar shh[100],shh2[100],*lhh;
  PetscTruth  alloc = PETSC_FALSE;
  PetscReal   hnorm;

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
  
  /*** First orthogonalization ***/

  /* h = V^* v */
  ierr = VecMDot(n,v,V,lhh);CHKERRQ(ierr);

  /* q = v - V h */
  for (j=0;j<n;j++) {
    H[j] = lhh[j];
    lhh[j] = -lhh[j];    
  }
  ierr = VecMAXPY(n,lhh,v,V);CHKERRQ(ierr);
  
  ierr = VecNorm(v, NORM_2, norm);CHKERRQ(ierr);

  /*** Second orthogonalization if necessary ***/
  if (eps->orth_eta != 0.0) {
    hnorm = 0.0;
    for (j=0; j<n; j++) {
      hnorm  +=  PetscRealPart(H[j] * PetscConj(H[j]));
    }
    hnorm = sqrt(hnorm);

    /* if ||q|| < eta ||h|| */
    if (eps->orth_eta < 0 || *norm < eps->orth_eta * hnorm) {
      PetscLogInfo(eps,"EPSClassicalGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",*norm,hnorm);
      
      /* s = V^* q */
      ierr = VecMDot(n,v,V,lhh);CHKERRQ(ierr); 

      /* q = q - V s  ;  h = h + s */
      for (j=0;j<n;j++) {
        H[j] += lhh[j];
        lhh[j] = -lhh[j];    
      }
      ierr = VecMAXPY(n,lhh,v,V);CHKERRQ(ierr);

      ierr = VecNorm(v, NORM_2, norm);CHKERRQ(ierr);
    }
  }

  if (n>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  if (alloc) { ierr = PetscFree(H);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*
    Orthogonalization routine using modified Gram-Schmidt with refinement.
 */
#undef __FUNCT__  
#define __FUNCT__ "EPSModifiedGramSchmidtOrthogonalization"
int EPSModifiedGramSchmidtOrthogonalization(EPS eps,int n,Vec *V,Vec v,PetscScalar *H,PetscReal *norm)
{
  int         ierr,j;
  PetscScalar alpha;
  PetscTruth  refinement,allocated;
  PetscScalar lh[100],*h;
  PetscReal   hnorm;
  
  PetscFunctionBegin;
  
  allocated = PETSC_FALSE;
  if (!H) {
    if (n<=100) h = lh;
    else {
      ierr = PetscMalloc(n*sizeof(PetscScalar),&h);CHKERRQ(ierr);
      allocated = PETSC_TRUE;
    }
  } else h = H;
  
  for (j=0; j<n; j++) {
    h[j] = 0;
  }
  refinement = PETSC_FALSE;
  do {
    for (j=0; j<n; j++) {
      /* alpha = ( v, v_j ) */
      ierr = VecDot(v,V[j],&alpha);CHKERRQ(ierr);
      /* store coefficients if requested */
      h[j] += alpha;
      /* v <- v - alpha v_j */
      alpha = -alpha;
      ierr = VecAXPY(&alpha,V[j],v);CHKERRQ(ierr);
    }
    ierr = VecNorm(v, NORM_2, norm);CHKERRQ(ierr);
    if (refinement) refinement = PETSC_FALSE;
    else if (eps->orth_eta != 0.0) {
      hnorm = 0.0;
      for (j=0; j<n; j++) {
        hnorm += PetscRealPart(h[j] * PetscConj(h[j]));
      }
      hnorm = sqrt(hnorm);
      if (eps->orth_eta < 0.0 || *norm < eps->orth_eta * hnorm) {
        PetscLogInfo(eps,"EPSModifiedGramSchmidtOrthogonalization:Performing iterative refinement wnorm %g hnorm %g\n",*norm,hnorm);
        refinement = PETSC_TRUE;
      }
    }
  } while (refinement);
  
  if (allocated) {
    ierr = PetscFree(h);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSAttachDeflationSpace"
/*@
   EPSAttachDeflationSpace - add vectors to the deflation space

   Not Collective

   Input Parameter:
+  eps   - Eigensolver context 
.  n     - size of deflation space
.  ds    - components of deflation space
-  ortho - PETSC_TRUE if components of deflation space are orthonormal

   Level: intermediate

@*/
int EPSAttachDeflationSpace(EPS eps,int n,Vec *ds,PetscTruth ortho)
{
  int ierr, i;
  Vec *tvec;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  tvec = eps->DS;
  if (n+eps->nds > 0) {
     ierr = PetscMalloc((n+eps->nds)*sizeof(Vec), &eps->DS);CHKERRQ(ierr);
  }
  if (eps->nds > 0) {
    for (i=0; i<eps->nds; i++) eps->DS[i] = tvec[i];
    ierr = PetscFree(tvec);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    ierr = VecDuplicate(ds[i],&eps->DS[i + eps->nds]);CHKERRQ(ierr);
    ierr = VecCopy(ds[i],eps->DS[i + eps->nds]);CHKERRQ(ierr);
  }
  eps->nds += n;
  if (!ortho) eps->ds_ortho = PETSC_FALSE;
  eps->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRemoveDeflationSpace"
/*@
   EPSRemoveDeflationSpace - removes the deflation space

   Not Collective

   Input Parameter:
+  eps   - Eigensolver context 

   Level: intermediate

@*/
int EPSRemoveDeflationSpace(EPS eps)
{
  int ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->nds > 0) {
    ierr = VecDestroyVecs(eps->DS, eps->nds);CHKERRQ(ierr);
  }
  eps->ds_ortho = PETSC_TRUE;
  eps->setupcalled = 0;
  PetscFunctionReturn(0);
}
