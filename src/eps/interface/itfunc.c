/*
      Interface EPS routines that the user calls.
*/

#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy"
/*@
   EPSDestroy - Destroys the EPS context.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSSolve()
@*/
PetscErrorCode EPSDestroy(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (--eps->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(eps);CHKERRQ(ierr);

  ierr = STDestroy(eps->OP);CHKERRQ(ierr);

  if (eps->ops->destroy) {
    ierr = (*eps->ops->destroy)(eps); CHKERRQ(ierr);
  }
  
  if (eps->perm) {
    ierr = PetscFree(eps->perm);CHKERRQ(ierr);
  }

  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial);CHKERRQ(ierr);
  }

  if (eps->nds > 0) {
    ierr = VecDestroyVecs(eps->DS, eps->nds);
  }
  
  if (eps->DSV) {
    ierr = PetscFree(eps->DSV);CHKERRQ(ierr);
  }

  PetscLogObjectDestroy(eps);
  PetscHeaderDestroy(eps);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetST"
/*@
   EPSSetST - Associates a spectral transformation object to the
   eigensolver. 

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  st   - the spectral transformation object

   Note:
   Use EPSGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: EPSGetST()
@*/
PetscErrorCode EPSSetST(EPS eps,ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(st,ST_COOKIE,2);
  PetscCheckSameComm(eps,1,st,2);
  ierr = STDestroy(eps->OP); CHKERRQ(ierr);
  eps->OP = st;
  PetscObjectReference((PetscObject)eps->OP);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetST"
/*@
   EPSGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: beginner

.seealso: EPSSetST()
@*/
PetscErrorCode EPSGetST(EPS eps, ST *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *st = eps->OP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetMonitor"
/*@C
   EPSSetMonitor - Sets an ADDITIONAL function to be called at every 
   iteration to monitor the error estimates for each requested eigenpair.
      
   Collective on EPS

   Input Parameters:
+  eps     - eigensolver context obtained from EPSCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring
-  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)

   Calling Sequence of monitor:
$     monitor (EPS eps, int its, int nconv, PetscReal* errest, int nest, void *mctx)

+  eps    - eigensolver context obtained from EPSCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  errest - error estimates for each eigenpair
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by EPSSetMonitor()

   Options Database Keys:
+    -eps_monitor        - print error estimates at each iteration
-    -eps_cancelmonitors - cancels all monitors that have been hardwired into
      a code by calls to EPSetMonitor(), but does not cancel those set via
      the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   EPSSetMonitor() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: EPSDefaultEstimatesMonitor(), EPSClearMonitor()
@*/
PetscErrorCode EPSSetMonitor(EPS eps, int (*monitor)(EPS,int,int,PetscScalar*,PetscScalar*,PetscReal*,int,void*), void *mctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->numbermonitors >= MAXEPSMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many EPS monitors set");
  }
  eps->monitor[eps->numbermonitors]           = monitor;
  eps->monitorcontext[eps->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSClearMonitor"
/*@C
   EPSClearMonitor - Clears all monitors for an EPS object.

   Collective on EPS

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Options Database Key:
.    -eps_cancelmonitors - Cancels all monitors that have been hardwired 
      into a code by calls to EPSSetMonitor() or EPSSetValuesMonitor(), 
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: EPSSetMonitor(), EPSSetValuesMonitor()
@*/
PetscErrorCode EPSClearMonitor(EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetMonitorContext"
/*@C
   EPSGetMonitorContext - Gets the estimates monitor context, as set by 
   EPSSetMonitor() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: EPSSetMonitor(), EPSDefaultEstimatesMonitor()
@*/
PetscErrorCode EPSGetMonitorContext(EPS eps, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *ctx =      (eps->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeExplicitOperator"
/*@
    EPSComputeExplicitOperator - Computes the explicit operator associated
    to the eigenvalue problem with the specified spectral transformation.  

    Collective on EPS

    Input Parameter:
.   eps - the eigenvalue solver context

    Output Parameter:
.   mat - the explicit operator

    Notes:
    This routine builds a matrix containing the explicit operator. For 
    example, in generalized problems with shift-and-invert spectral
    transformation the result would be matrix (A - s B)^-1 B.
    
    This computation is done by applying the operator to columns of the 
    identity matrix.

    Currently, this routine uses a dense matrix format when 1 processor
    is used and a sparse format otherwise.  This routine is costly in general,
    and is recommended for use only with relatively small systems.

    Level: advanced

.seealso: STApply()   
@*/
PetscErrorCode EPSComputeExplicitOperator(EPS eps,Mat *mat)
{
  PetscErrorCode ierr;
  Vec            in,out;
  int            i,M,m,size,*rows,start,end;
  MPI_Comm       comm;
  PetscScalar    *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(mat,2);
  comm = eps->comm;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = VecDuplicate(eps->vec_initial,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&out);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  if (size == 1) {
    ierr = MatCreateSeqDense(comm,M,M,PETSC_NULL,mat);CHKERRQ(ierr);
  } else {
    ierr = MatCreateMPIAIJ(comm,m,m,M,M,0,0,0,0,mat);CHKERRQ(ierr);
  }
  
  for (i=0; i<M; i++) {

    ierr = VecSet(&zero,in);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = STApply(eps->OP,in,out); CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSIsGeneralized"
/*@
   EPSIsGeneralized - Ask if the EPS object corresponds to a generalized 
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

@*/
PetscErrorCode EPSIsGeneralized(EPS eps,PetscTruth* is)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = STGetOperators(eps->OP,PETSC_NULL,&B);CHKERRQ(ierr);
  if( B ) *is = PETSC_TRUE;
  else *is = PETSC_FALSE;
  if( eps->setupcalled ) {
    if( eps->isgeneralized != *is ) { 
      SETERRQ(0,"Warning: Inconsistent EPS state");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSIsHermitian"
/*@
   EPSIsHermitian - Ask if the EPS object corresponds to a Hermitian 
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

@*/
PetscErrorCode EPSIsHermitian(EPS eps,PetscTruth* is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if( eps->ishermitian ) *is = PETSC_TRUE;
  else *is = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReverseProjection"
/*@
   EPSReverseProjection - Compute the operation V=V*S, where the columns of
   V are m of the basis vectors of the EPS object and S is an mxm dense
   matrix.

   Collective on EPS

   Input Parameter:
+  eps - the eigenproblem solver context
.  V - basis vectors
.  S - pointer to the values of matrix S
.  k - starting column
.  m - dimension of matrix S
-  work - workarea of m vectors for intermediate results

   Level: developer

@*/
PetscErrorCode EPSReverseProjection(EPS eps,Vec* V,PetscScalar *S,int k,int m,Vec* work)
{
  PetscErrorCode ierr;
  int            i;
  PetscScalar    zero = 0.0;
  
  PetscFunctionBegin;
  for (i=k;i<m;i++) {
    ierr = VecSet(&zero,work[i]);CHKERRQ(ierr);
    ierr = VecMAXPY(m,S+m*i,work[i],V);CHKERRQ(ierr);
  }    
  for (i=k;i<m;i++) {
    ierr = VecCopy(work[i],V[i]);CHKERRQ(ierr);
  }    
  PetscFunctionReturn(0);
}
