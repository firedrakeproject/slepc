/*                       

   SLEPc singular value solver: "eigensolver"

   Method: Uses an Hermitian eigensolver for A^T*A, A*A^T or H(A)

   Last update: Nov 2006

*/
#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "slepceps.h"

typedef struct {
  SVDEigensolverMode mode;
  EPS eps;
  Mat mat;
  Vec x1,x2,y1,y2;
} SVD_EIGENSOLVER;

#undef __FUNCT__  
#define __FUNCT__ "ShellMatMult_EIGENSOLVER"
PetscErrorCode ShellMatMult_EIGENSOLVER(Mat B,Vec x, Vec y)
{
  PetscErrorCode  ierr;
  SVD             svd;
  SVD_EIGENSOLVER *eigen;
  PetscScalar     *px,*py;
  PetscInt        m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  eigen = (SVD_EIGENSOLVER *)svd->data;
  switch (eigen->mode) {
    case SVDEIGENSOLVER_CROSS:
      ierr = SVDMatMult(svd,PETSC_FALSE,x,eigen->x1);CHKERRQ(ierr);
      ierr = SVDMatMult(svd,PETSC_TRUE,eigen->x1,y);CHKERRQ(ierr);
      break;
    case SVDEIGENSOLVER_CYCLIC:
      ierr = SVDMatGetLocalSize(svd,&m,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecGetArray(x,&px);CHKERRQ(ierr);
      ierr = VecGetArray(y,&py);CHKERRQ(ierr);
      ierr = VecPlaceArray(eigen->x1,px);CHKERRQ(ierr);
      ierr = VecPlaceArray(eigen->x2,px+m);CHKERRQ(ierr);
      ierr = VecPlaceArray(eigen->y1,py);CHKERRQ(ierr);
      ierr = VecPlaceArray(eigen->y2,py+m);CHKERRQ(ierr);
      
      ierr = SVDMatMult(svd,PETSC_FALSE,eigen->x2,eigen->y1);CHKERRQ(ierr);
      ierr = SVDMatMult(svd,PETSC_TRUE,eigen->x1,eigen->y2);CHKERRQ(ierr);
            
      ierr = VecResetArray(eigen->x1);CHKERRQ(ierr);
      ierr = VecResetArray(eigen->x2);CHKERRQ(ierr);
      ierr = VecResetArray(eigen->y1);CHKERRQ(ierr);
      ierr = VecResetArray(eigen->y2);CHKERRQ(ierr);
      ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
      ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
      break;     
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ShellMatGetDiagonal_EIGENSOLVER"
PetscErrorCode ShellMatGetDiagonal_EIGENSOLVER(Mat B,Vec diag)
{
  PetscErrorCode    ierr;
  SVD               svd;
  SVD_EIGENSOLVER   *eigen = (SVD_EIGENSOLVER *)svd->data;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  eigen = (SVD_EIGENSOLVER *)svd->data;
  if (eigen->mode == SVDEIGENSOLVER_CROSS) {
    ierr = VecCopy(eigen->y1,diag);CHKERRQ(ierr);
  } else {
    ierr = VecSet(diag,0.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_EIGENSOLVER"
PetscErrorCode SVDSetUp_EIGENSOLVER(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_EIGENSOLVER   *eigen = (SVD_EIGENSOLVER *)svd->data;
  PetscInt          M,m,n,i,j,start,end,ncols,*pos;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  
  if (eigen->mat) { ierr = MatDestroy(eigen->mat);CHKERRQ(ierr); }
  if (eigen->x1) { ierr = VecDestroy(eigen->x1);CHKERRQ(ierr); } 
  if (eigen->x2) { ierr = VecDestroy(eigen->x2);CHKERRQ(ierr); } 
  if (eigen->y1) { ierr = VecDestroy(eigen->y1);CHKERRQ(ierr); } 
  if (eigen->y2) { ierr = VecDestroy(eigen->y2);CHKERRQ(ierr); } 
  eigen->x1 = eigen->x2 = eigen->y1 = eigen->y2 = PETSC_NULL;

  ierr = SVDMatGetLocalSize(svd,&m,&n);CHKERRQ(ierr);
  switch (eigen->mode) {
    case SVDEIGENSOLVER_CROSS:
      ierr = SVDMatGetVecs(svd,&eigen->y1,&eigen->x1);CHKERRQ(ierr);
      ierr = MatCreateShell(svd->comm,n,n,PETSC_DETERMINE,PETSC_DETERMINE,svd,&eigen->mat);CHKERRQ(ierr);
      ierr = MatShellSetOperation(eigen->mat,MATOP_MULT,(void(*)(void))ShellMatMult_EIGENSOLVER);CHKERRQ(ierr);  
      ierr = MatShellSetOperation(eigen->mat,MATOP_GET_DIAGONAL,(void(*)(void))ShellMatGetDiagonal_EIGENSOLVER);CHKERRQ(ierr);  
      /* compute diagonal from rows and store in eigen->y1 */
      ierr = PetscMalloc(sizeof(PetscScalar)*n,&work1);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(PetscScalar)*n,&work2);CHKERRQ(ierr);
      for (i=0;i<n;i++) work1[i] = work2[i] = 0.0;
      if (svd->AT) {
        ierr = MatGetOwnershipRange(svd->AT,&start,&end);CHKERRQ(ierr);
        for (i=start;i<end;i++) {
          ierr = MatGetRow(svd->AT,i,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
          for (j=0;j<ncols;j++)
            work1[i] += vals[j]*vals[j];
          ierr = MatRestoreRow(svd->AT,i,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
        }
      } else {
        ierr = MatGetOwnershipRange(svd->A,&start,&end);CHKERRQ(ierr);
        for (i=start;i<end;i++) {
          ierr = MatGetRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
          for (j=0;j<ncols;j++)
            work1[cols[j]] += vals[j]*vals[j];
          ierr = MatRestoreRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
        }
      }
      ierr = MPI_Allreduce(work1,work2,n,MPIU_SCALAR,MPI_SUM,svd->comm);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(eigen->y1,&start,&end);CHKERRQ(ierr);
      ierr = VecGetArray(eigen->y1,&diag);CHKERRQ(ierr);
      for (i=start;i<end;i++)
        diag[i-start] = work2[i];
      ierr = VecRestoreArray(eigen->y1,&diag);CHKERRQ(ierr);
      ierr = PetscFree(work1);CHKERRQ(ierr);
      ierr = PetscFree(work2);CHKERRQ(ierr);
      break;
    case SVDEIGENSOLVER_CYCLIC:
      ierr = VecCreateMPIWithArray(svd->comm,m,PETSC_DECIDE,PETSC_NULL,&eigen->x1);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(svd->comm,n,PETSC_DECIDE,PETSC_NULL,&eigen->x2);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(svd->comm,m,PETSC_DECIDE,PETSC_NULL,&eigen->y1);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(svd->comm,n,PETSC_DECIDE,PETSC_NULL,&eigen->y2);CHKERRQ(ierr);
      ierr = MatCreateShell(svd->comm,m+n,m+n,PETSC_DETERMINE,PETSC_DETERMINE,svd,&eigen->mat);CHKERRQ(ierr);
      ierr = MatShellSetOperation(eigen->mat,MATOP_MULT,(void(*)(void))ShellMatMult_EIGENSOLVER);CHKERRQ(ierr);  
      ierr = MatShellSetOperation(eigen->mat,MATOP_GET_DIAGONAL,(void(*)(void))ShellMatGetDiagonal_EIGENSOLVER);CHKERRQ(ierr);  
      break;
    case SVDEIGENSOLVER_CYCLIC_EXPLICIT:
      ierr = VecCreateMPIWithArray(svd->comm,m,PETSC_DECIDE,PETSC_NULL,&eigen->x1);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(svd->comm,n,PETSC_DECIDE,PETSC_NULL,&eigen->x2);CHKERRQ(ierr);
      ierr = MatCreate(svd->comm,&eigen->mat);CHKERRQ(ierr);
      ierr = MatSetSizes(eigen->mat,m+n,m+n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = MatSetFromOptions(eigen->mat);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(PetscInt)*n,&pos);CHKERRQ(ierr);
      ierr = SVDMatGetSize(svd,&M,PETSC_NULL);CHKERRQ(ierr);
      if (svd->AT) {
        ierr = MatGetOwnershipRange(svd->AT,&start,&end);CHKERRQ(ierr);
        for (i=start;i<end;i++) {
          ierr = MatGetRow(svd->AT,i,&ncols,&cols,&vals);CHKERRQ(ierr);
          j = i + M;
          ierr = MatSetValues(eigen->mat,1,&j,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(eigen->mat,ncols,cols,1,&j,vals,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        ierr = MatGetOwnershipRange(svd->A,&start,&end);CHKERRQ(ierr);
        for (i=start;i<end;i++) {
          ierr = MatGetRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) 
            pos[j] = cols[j] + M;
          ierr = MatSetValues(eigen->mat,1,&i,ncols,pos,vals,INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(eigen->mat,ncols,pos,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree(pos);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(eigen->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(eigen->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
  }

  ierr = EPSSetOperators(eigen->eps,eigen->mat,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eigen->eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetDimensions(eigen->eps,svd->nsv,svd->ncv);CHKERRQ(ierr);
  ierr = EPSSetTolerances(eigen->eps,svd->tol,svd->max_it);CHKERRQ(ierr);
  ierr = EPSSetUp(eigen->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eigen->eps,PETSC_NULL,&svd->ncv);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eigen->eps,&svd->tol,&svd->max_it);CHKERRQ(ierr);

  if (svd->U) {  
    for (i=0;i<svd->n;i++) { ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr); }
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
  }
  if (eigen->mode != SVDEIGENSOLVER_CROSS) {  
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) { ierr = SVDMatGetVecs(svd,PETSC_NULL,svd->U+i);CHKERRQ(ierr); }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_EIGENSOLVER"
PetscErrorCode SVDSolve_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  int             i,j;
  PetscInt        m;
  PetscScalar     sigma,*px;
  Vec             x;
  
  PetscFunctionBegin;
  ierr = EPSSetWhichEigenpairs(eigen->eps,svd->which == SVD_LARGEST ? EPS_LARGEST_REAL : EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);
  if (eigen->mode == SVDEIGENSOLVER_CROSS && svd->A) { ierr = EPSSetInitialVector(eigen->eps,svd->vec_initial);CHKERRQ(ierr); }
  ierr = EPSSolve(eigen->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(eigen->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eigen->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(eigen->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);
  ierr = EPSGetOperationCounters(eigen->eps,&svd->matvecs,&svd->dots,PETSC_NULL);CHKERRQ(ierr);
  switch (eigen->mode) {
    case SVDEIGENSOLVER_CROSS:
      for (i=0;i<svd->nconv;i++) {
	ierr = EPSGetEigenpair(eigen->eps,i,&sigma,PETSC_NULL,svd->V[i],PETSC_NULL);CHKERRQ(ierr);
	svd->sigma[i] = sqrt(PetscRealPart(sigma));
      }
      break;
    case SVDEIGENSOLVER_CYCLIC:
    case SVDEIGENSOLVER_CYCLIC_EXPLICIT:
      ierr = MatGetVecs(eigen->mat,&x,PETSC_NULL);CHKERRQ(ierr);
      ierr = SVDMatGetLocalSize(svd,&m,PETSC_NULL);CHKERRQ(ierr);
      for (i=0,j=0;i<svd->nconv;i++) {
	ierr = EPSGetEigenpair(eigen->eps,i,&sigma,PETSC_NULL,x,PETSC_NULL);CHKERRQ(ierr);
	if (PetscRealPart(sigma) > 0.0) {
	  svd->sigma[j] = PetscRealPart(sigma);
	  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
	  ierr = VecPlaceArray(eigen->x1,px);CHKERRQ(ierr);
	  ierr = VecPlaceArray(eigen->x2,px+m);CHKERRQ(ierr);
	  
	  ierr = VecCopy(eigen->x1,svd->U[j]);CHKERRQ(ierr);
	  ierr = VecScale(svd->U[j],1.0/sqrt(2.0));CHKERRQ(ierr);

	  ierr = VecCopy(eigen->x2,svd->V[j]);CHKERRQ(ierr);
	  ierr = VecScale(svd->V[j],1.0/sqrt(2.0));CHKERRQ(ierr);	  
	  
	  ierr = VecResetArray(eigen->x1);CHKERRQ(ierr);
	  ierr = VecResetArray(eigen->x2);CHKERRQ(ierr);
	  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
	  j++;
	}
      }
      svd->nconv = j;
      ierr = VecDestroy(x);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitor_EIGENSOLVER"
PetscErrorCode SVDMonitor_EIGENSOLVER(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int nest,void *ctx)
{
  int             i,j;
  SVD             svd = (SVD)ctx;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;

  if (eigen->mode != SVDEIGENSOLVER_CROSS) {
    nconv = 0;
    for (i=0,j=0;i<nest;i++) {
      if (PetscRealPart(eigr[i]) > 0.0) {
	svd->sigma[j] = PetscRealPart(eigr[i]);
	svd->errest[j] = errest[i];
	if (errest[i] < svd->tol) nconv++;
	j++;
      }
    }
    nest = j;
  } else {
    for (i=0,j=0;i<nest;i++) {
      svd->sigma[i] = sqrt(PetscRealPart(eigr[i]));
      svd->errest[i] = errest[i];
    }
  }

  SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_EIGENSOLVER"
PetscErrorCode SVDSetFromOptions_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  PetscTruth      flg;
  const char      *mode_list[3] = { "cross" , "cyclic" , "explicit" };
  PetscInt        mode;
  ST              st;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"EIGENSOLVER Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-svd_eigensolver_mode","Eigensolver SVD mode","SVDEigensolverSetMode",mode_list,3,mode_list[eigen->mode],&mode,&flg);CHKERRQ(ierr);
  if (flg) { eigen->mode = (SVDEigensolverMode)mode; }
  if (eigen->mode == SVDEIGENSOLVER_CYCLIC_EXPLICIT) {
    /* don't build the transpose */
    if (svd->transmode == PETSC_DECIDE)
      svd->transmode = SVD_TRANSPOSE_IMPLICIT;
  } else {
    /* use as default an ST with shell matrix and Jacobi */ 
    ierr = EPSGetST(eigen->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,STMATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eigen->eps);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDEigensolverSetMode_EIGENSOLVER"
PetscErrorCode SVDEigensolverSetMode_EIGENSOLVER(SVD svd,SVDEigensolverMode mode)
{
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  switch (eigen->mode) {
    case PETSC_DEFAULT:
      mode = SVDEIGENSOLVER_CROSS;
    case SVDEIGENSOLVER_CROSS:
    case SVDEIGENSOLVER_CYCLIC:
    case SVDEIGENSOLVER_CYCLIC_EXPLICIT:
      eigen->mode = mode;
      svd->setupcalled = 0;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
  }
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "SVDEigensolverSetMode"
/*@
   SVDEigensolverSetMode - Indicates which is the related eigenvalue 
   problem that has to be solved in order to compute the SVD.

   Collective on SVD

   Input Parameters:
+  svd  - singular value solver
-  mode - the mode flag, one of SVDEIGENSOLVER_CROSS , SVDEIGENSOLVER_CYCLIC or
          SVDEIGENSOLVER_CYCLIC_EXPLICIT

   Options Database Key:
.  -svd_eigensolver_mode <mode> - Indicates the mode flag, where <mode> 
    is one of 'cross', 'cyclic' or 'explicit' (see explanation below).

   Note:
   This parameter selects the eigensystem used to compute the SVD:
   A^T*A (SVDEIGENSOLVER_CROSS) or H(A) = [ 0  A ; A^T 0 ] (SVDEIGENSOLVER_CYCLIC).
   SVDEIGENSOLVER_CYCLIC_EXPLICIT does the same as SVDEIGENSOLVER_CYCLIC but it builds
   internally the H(A) matrix.

   Level: beginner

.seealso: SVDEigensolverGetMode()
@*/
PetscErrorCode SVDEigensolverSetMode(SVD svd,SVDEigensolverMode mode)
{
  PetscErrorCode ierr, (*f)(SVD,SVDEigensolverMode);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDEigensolverSetMode_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,mode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDEigensolverGetMode_EIGENSOLVER"
PetscErrorCode SVDEigensolverGetMode_EIGENSOLVER(SVD svd,SVDEigensolverMode *mode)
{
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(mode,2);
  *mode = eigen->mode;
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "SVDEigensolverGetMode"
/*@C
   SVDEigensolverGetMode - Returns the mode flag used to compute the SVD
   via a related eigenproblem. 

   Not collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  mode - the mode flag

   Level: beginner

.seealso: SVDEigensolverSetMode()
@*/
PetscErrorCode SVDEigensolverGetMode(SVD svd,SVDEigensolverMode *mode)
{
  PetscErrorCode ierr, (*f)(SVD,SVDEigensolverMode*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDEigensolverGetMode_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,mode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDEigensolverSetEPS_EIGENSOLVER"
PetscErrorCode SVDEigensolverSetEPS_EIGENSOLVER(SVD svd,EPS eps)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(eigen->eps);CHKERRQ(ierr);  
  eigen->eps = eps;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDEigensolverSetEPS"
/*@
   SVDEigensolverSetEPS - Associates an eigensolver object (EPS) to the
   singular value solver. 

   Collective on SVD

   Input Parameters:
+  svd - singular value solver
-  eps - the eigensolver object

   Level: advanced

.seealso: SVDEigensolverGetEPS()
@*/
PetscErrorCode SVDEigensolverSetEPS(SVD svd,EPS eps)
{
  PetscErrorCode ierr, (*f)(SVD,EPS eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDEigensolverSetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDEigensolverGetEPS_EIGENSOLVER"
PetscErrorCode SVDEigensolverGetEPS_EIGENSOLVER(SVD svd,EPS *eps)
{
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(eps,2);
  *eps = eigen->eps;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDEigensolverGetEPS"
/*@C
   SVDEigensolverGetEPS - Retrieve the eigensolver object (EPS) associated
   to the singular value solver.

   Not Collective

   Input Parameters:
.  svd - singular value solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: SVDEigensolverSetEPS()
@*/
PetscErrorCode SVDEigensolverGetEPS(SVD svd,EPS *eps)
{
  PetscErrorCode ierr, (*f)(SVD,EPS *eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDEigensolverGetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_EIGENSOLVER"
PetscErrorCode SVDView_EIGENSOLVER(SVD svd,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  const char      *mode_list[3] = { "implicit cross product" , "implicit cyclic matrix" , "explicit cyclic matrix" };

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"eigensolver SVD mode: %s\n",mode_list[eigen->mode]);CHKERRQ(ierr);
  ierr = EPSView(eigen->eps,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_EIGENSOLVER"
PetscErrorCode SVDDestroy_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(eigen->eps);CHKERRQ(ierr);
  if (eigen->mat) { ierr = MatDestroy(eigen->mat);CHKERRQ(ierr); }
  if (eigen->x1) { ierr = VecDestroy(eigen->x1);CHKERRQ(ierr); } 
  if (eigen->x2) { ierr = VecDestroy(eigen->x2);CHKERRQ(ierr); } 
  if (eigen->y1) { ierr = VecDestroy(eigen->y1);CHKERRQ(ierr); } 
  if (eigen->y2) { ierr = VecDestroy(eigen->y2);CHKERRQ(ierr); } 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_EIGENSOLVER"
PetscErrorCode SVDCreate_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen;
  
  PetscFunctionBegin;
  ierr = PetscNew(SVD_EIGENSOLVER,&eigen);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_EIGENSOLVER));
  svd->data                      = (void *)eigen;
  svd->ops->solve                = SVDSolve_EIGENSOLVER;
  svd->ops->setup                = SVDSetUp_EIGENSOLVER;
  svd->ops->setfromoptions       = SVDSetFromOptions_EIGENSOLVER;
  svd->ops->destroy              = SVDDestroy_EIGENSOLVER;
  svd->ops->view                 = SVDView_EIGENSOLVER;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDEigensolverSetEPS_C","SVDEigensolverSetEPS_EIGENSOLVER",SVDEigensolverSetEPS_EIGENSOLVER);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDEigensolverGetEPS_C","SVDEigensolverGetEPS_EIGENSOLVER",SVDEigensolverGetEPS_EIGENSOLVER);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDEigensolverSetMode_C","SVDEigensolverSetMode_EIGENSOLVER",SVDEigensolverSetMode_EIGENSOLVER);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDEigensolverGetMode_C","SVDEigensolverGetMode_EIGENSOLVER",SVDEigensolverGetMode_EIGENSOLVER);CHKERRQ(ierr);

  ierr = EPSCreate(svd->comm,&eigen->eps);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eigen->eps,svd->prefix);CHKERRQ(ierr);
  ierr = EPSAppendOptionsPrefix(eigen->eps,"svd_");CHKERRQ(ierr);
  PetscLogObjectParent(svd,eigen->eps);
  ierr = EPSSetWhichEigenpairs(eigen->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetMonitor(eigen->eps,SVDMonitor_EIGENSOLVER,svd,PETSC_NULL);CHKERRQ(ierr);
  eigen->mode = SVDEIGENSOLVER_CROSS;
  eigen->mat = PETSC_NULL;
  eigen->x1 = PETSC_NULL;
  eigen->x2 = PETSC_NULL;
  eigen->y1 = PETSC_NULL;
  eigen->y2 = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END

