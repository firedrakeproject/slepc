/*
      EPS routines related to problem setup.
*/
#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp"
/*@
   EPSSetUp - Sets up all the internal data structures necessary for the
   execution of the eigensolver. Then calls STSetUp() for any set-up
   operations associated to the ST object.

   Collective on EPS

   Input Parameter:
.  eps   - eigenproblem solver context

   Level: advanced

   Notes:
   This function need not be called explicitly in most cases, since EPSSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   This function sets a random initial vector if none has been provided.

.seealso: EPSCreate(), EPSSolve(), EPSDestroy(), STSetUp()
@*/
PetscErrorCode EPSSetUp(EPS eps)
{
  PetscErrorCode ierr;
  int            i;   
  Mat            A,B; 
  PetscInt       N;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if (eps->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!eps->type_name) {
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  }
  
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  /* Set default problem type */
  if (!eps->problem_type) {
    if (B==PETSC_NULL) {
      ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
    }
    else {
      ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
    }
  } else if ((B && !eps->isgeneralized) || (!B && eps->isgeneralized)) {
    SETERRQ(0,"Warning: Inconsistent EPS state"); 
  }
  
  /* Create random initial vectors if none are available */
  /* right */
  if (eps->niv == 0) {
    eps->niv = 1;
    ierr = PetscMalloc(sizeof(Vec),&eps->IV);CHKERRQ(ierr);    
    ierr = MatGetVecs(A,&eps->IV[0],PETSC_NULL);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(eps->IV[0]);CHKERRQ(ierr);
  }
  /* left */
  if (eps->nliv == 0) {
    eps->nliv = 1;
    ierr = PetscMalloc(sizeof(Vec),&eps->LIV);CHKERRQ(ierr);    
    ierr = MatGetVecs(A,PETSC_NULL,&eps->LIV[0]);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(eps->LIV[0]);CHKERRQ(ierr);
  }

  ierr = VecGetSize(eps->IV[0],&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv > N) eps->ncv = N;

  ierr = (*eps->ops->setup)(eps);CHKERRQ(ierr);
  ierr = STSetUp(eps->OP); CHKERRQ(ierr); 

  /* DSV is equal to the columns of DS followed by the ones in V */
  ierr = PetscFree(eps->DSV);CHKERRQ(ierr);
  ierr = PetscMalloc((eps->ncv+eps->nds)*sizeof(Vec),&eps->DSV);CHKERRQ(ierr);    
  for (i = 0; i < eps->nds; i++) eps->DSV[i] = eps->DS[i];
  for (i = 0; i < eps->ncv; i++) eps->DSV[i+eps->nds] = eps->V[i];
  
  if (eps->nds>0) {
    if (!eps->ds_ortho) {
      /* orthonormalize vectors in DS if necessary */
      ierr = IPQRDecomposition(eps->ip,eps->DS,0,eps->nds,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = IPOrthogonalize(eps->ip,eps->nds,PETSC_NULL,eps->DS,eps->IV[0],PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); 
  }

  ierr = STCheckNullSpace(eps->OP,eps->nds,eps->DS);CHKERRQ(ierr);
    
  ierr = PetscLogEventEnd(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);
  eps->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialVector"
/*@
   EPSSetInitialVector - Provides an initial vector.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

   Notes:
   An initial vector is a vector that is used by the eigensolver 
   to start the iteration.

   If no initial vector is provided, SLEPc creates a random initial
   vector. This default behaviour is enough for most applications.

   Sometimes, a 'good' initial vector can improve the convergence of
   the solver. For this, the vector should have rich components in 
   the direction of the wanted eigenvectors.

   This function can be called several times. This is useful for cases
   when multiple initial vectors are required. Not all solvers can
   exploit this feature. The vector is appended to the internal
   list of initial vectors. For reseting this list, use 
   EPSClearInitialVectors(). 

.seealso: EPSGetInitialVector(), EPSSetLeftInitialVector(), EPSClearInitialVectors()

@*/
PetscErrorCode EPSSetInitialVector(EPS eps,Vec vec)
{
  int            i;
  Vec            *tmp;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (eps->niv > 0 && eps->useriv==PETSC_FALSE) {
    ierr = EPSClearInitialVectors(eps);CHKERRQ(ierr);
  }
  eps->useriv = PETSC_TRUE;
  tmp = eps->IV;
  ierr = PetscMalloc((eps->niv+1)*sizeof(Vec),&eps->IV);CHKERRQ(ierr);    
  if (eps->niv > 0) {
    for (i=0; i<eps->niv; i++) { eps->IV[i] = tmp[i]; }
    ierr = PetscFree(tmp);CHKERRQ(ierr);    
  }
  eps->IV[eps->niv] = vec;
  eps->niv++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetNumberInitialVectors"
/*@
   EPSGetNumberInitialVectors - Gets the number of initial vectors.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameters:
+  niv  - number of initial vectors
-  nliv - number of left initial vectors

   Notes:
   An initial vector is a vector that is used by the eigensolver 
   to start the iteration. Left initial vectors are needed by 
   two-sided eigensolvers only.

   Use PETSC_NULL for output arguments whose value is not of interest.

   Level: intermediate

.seealso: EPSSetInitialVector(), EPSGetInitialVector(), EPSClearInitialVectors()
@*/
PetscErrorCode EPSGetNumberInitialVectors(EPS eps,int *niv,int *nliv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (niv)  *niv = eps->niv;
  if (nliv) *nliv = eps->nliv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetInitialVector"
/*@
   EPSGetInitialVector - Gets the i-th initial vector associated with the 
   eigensolver.

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameters:
+  eps - the eigensolver context
-  i   - the index of the initial vector

   Output Parameter:
.  vec - the vector

   Level: intermediate

   Notes:
   An initial vector is a vector that is used by the eigensolver 
   to start the iteration.

   The index i should be a value between 0 and niv-1 (see 
   EPSGetNumberInitialVectors()).

   The initial vectors are either user-provided (see EPSSetInitialVector())
   or randomly generated at EPSSetUp().

.seealso: EPSGetNumberInitialVectors(), EPSSetInitialVector(), EPSGetLeftInitialVector()
@*/
PetscErrorCode EPSGetInitialVector(EPS eps,int i,Vec *vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(vec,3);
  if (i<0 || i>=eps->niv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  ierr = PetscObjectReference((PetscObject)eps->IV[i]);CHKERRQ(ierr);
  *vec = eps->IV[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetLeftInitialVector"
/*@
   EPSSetLeftInitialVector - Provides a left initial vector.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

   Notes:
   An initial vector is a vector that is used by the eigensolver 
   to start the iteration. Left initial vectors start the left
   recurrence in two-sided eigensolvers.

   If no initial vector is provided, SLEPc creates a random initial
   vector. This default behaviour is enough for most applications.

   Sometimes, a 'good' initial vector can improve the convergence of
   the solver. For this, the vector should have rich components in 
   the direction of the wanted eigenvectors.

   This function can be called several times. This is useful for cases
   when multiple initial vectors are required. Not all solvers can
   exploit this feature. The vector is appended to the internal
   list of initial vectors. For reseting this list, use 
   EPSClearInitialVectors(). 

.seealso: EPSGetLeftInitialVector(), EPSSetInitialVector(), EPSClearInitialVectors()
@*/
PetscErrorCode EPSSetLeftInitialVector(EPS eps,Vec vec)
{
  int            i;
  Vec            *tmp;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (eps->nliv > 0 && eps->useriv==PETSC_FALSE) {
    ierr = EPSClearInitialVectors(eps);CHKERRQ(ierr);
  }
  eps->userliv = PETSC_TRUE;
  tmp = eps->LIV;
  ierr = PetscMalloc((eps->nliv+1)*sizeof(Vec),&eps->LIV);CHKERRQ(ierr);    
  if (eps->nliv > 0) {
    for (i=0; i<eps->nliv; i++) { eps->LIV[i] = tmp[i]; }
    ierr = PetscFree(tmp);CHKERRQ(ierr);    
  }
  eps->LIV[eps->nliv] = vec;
  eps->nliv++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetLeftInitialVector"
/*@
   EPSGetLeftInitialVector - Gets the i-th left initial vector associated 
   with the eigensolver.

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameters:
+  eps - the eigensolver context
-  i   - the index of the initial vector

   Output Parameter:
.  vec - the vector

   Level: intermediate

   Notes:
   An initial vector is a vector that is used by the eigensolver 
   to start the iteration. Left initial vectors start the left
   recurrence in two-sided eigensolvers.

   The index i should be a value between 0 and nliv-1 (see 
   EPSGetNumberInitialVectors()).

   The initial vectors are either user-provided (see EPSSetLeftInitialVector())
   or randomly generated at EPSSetUp().

.seealso: EPSGetNumberInitialVectors(), EPSSetLeftInitialVector(), EPSGetInitialVector()
@*/
PetscErrorCode EPSGetLeftInitialVector(EPS eps,int i,Vec *vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(vec,3);
  if (i<0 || i>=eps->nliv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  ierr = PetscObjectReference((PetscObject)eps->LIV[i]);CHKERRQ(ierr);
  *vec = eps->LIV[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSClearInitialVectors"
/*@
   EPSClearInitialVectors - Resets the list of initial vectors associated 
   with the eigensolver.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context

   Level: intermediate

   Notes:
   An initial vector is a vector that is used by the eigensolver 
   to start the iteration. Left initial vectors start the left
   recurrence in two-sided eigensolvers.

   This function resets both right and left initial vectors.

   If the initial vectors were randomly generated by EPSSetUp(), this
   function destroys them so the next time they are freshly created.

.seealso: EPSGetNumberInitialVectors(), EPSSetInitialVector(), EPSGetInitialVector(),
          EPSSetLeftInitialVector(), EPSGetLeftInitialVector()
@*/
PetscErrorCode EPSClearInitialVectors(EPS eps)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->niv > 0) {
    ierr = VecDestroyVecs(eps->IV,eps->niv);CHKERRQ(ierr);
    eps->niv = 0;
  }
  if (eps->nliv > 0) {
    ierr = VecDestroyVecs(eps->LIV,eps->nliv);CHKERRQ(ierr);
    eps->nliv = 0;
  }
  eps->useriv = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetOperators"
/*@
   EPSSetOperators - Sets the matrices associated with the eigenvalue problem.

   Collective on EPS and Mat

   Input Parameters:
+  eps - the eigenproblem solver context
.  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes: 
   To specify a standard eigenproblem, use PETSC_NULL for parameter B.

   Level: beginner

.seealso: EPSSolve(), EPSGetST(), STGetOperators()
@*/
PetscErrorCode EPSSetOperators(EPS eps,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  PetscCheckSameComm(eps,1,A,2);
  if (B) PetscCheckSameComm(eps,1,B,3);

  /* Check for square matrices */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"A is a non-square matrix"); }
  if (B) { 
    ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr);
    if (m!=n) { SETERRQ(1,"B is a non-square matrix"); }
  }

  ierr = STSetOperators(eps->OP,A,B);CHKERRQ(ierr);
  eps->setupcalled = 0;  /* so that next solve call will call setup */

  /* Destroy randomly generated initial vectors */
  if (eps->useriv==PETSC_FALSE) {
    ierr = EPSClearInitialVectors(eps);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSAttachDeflationSpace"
/*@
   EPSAttachDeflationSpace - Add vectors to the basis of the deflation space.

   Not Collective

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors to add
.  ds    - set of basis vectors of the deflation space
-  ortho - PETSC_TRUE if basis vectors of deflation space are orthonormal

   Notes:
   When a deflation space is given, the eigensolver seeks the eigensolution
   in the restriction of the problem to the orthogonal complement of this
   space. This can be used for instance in the case that an invariant 
   subspace is known beforehand (such as the nullspace of the matrix).

   The basis vectors can be provided all at once or incrementally with
   several calls to EPSAttachDeflationSpace().

   Use a value of PETSC_TRUE for parameter ortho if all the vectors passed
   in are known to be mutually orthonormal.

   Level: intermediate

.seealso: EPSRemoveDeflationSpace()
@*/
PetscErrorCode EPSAttachDeflationSpace(EPS eps,int n,Vec *ds,PetscTruth ortho)
{
  PetscErrorCode ierr;
  int            i;
  Vec            *tvec;
  
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
   EPSRemoveDeflationSpace - Removes the deflation space.

   Not Collective

   Input Parameter:
.  eps   - the eigenproblem solver context

   Level: intermediate

.seealso: EPSAttachDeflationSpace()
@*/
PetscErrorCode EPSRemoveDeflationSpace(EPS eps)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->nds > 0) {
    ierr = VecDestroyVecs(eps->DS, eps->nds);CHKERRQ(ierr);
  }
  eps->ds_ortho = PETSC_TRUE;
  eps->setupcalled = 0;
  PetscFunctionReturn(0);
}
