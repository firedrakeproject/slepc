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
  Vec w;
} SVD_EIGENSOLVER;

#undef __FUNCT__  
#define __FUNCT__ "ShellMatMult_EIGENSOLVER"
PetscErrorCode ShellMatMult_EIGENSOLVER(Mat B,Vec x, Vec y)
{
  PetscErrorCode  ierr;
  SVD             svd;
  SVD_EIGENSOLVER *eigen;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  eigen = (SVD_EIGENSOLVER *)svd->data;
  switch (eigen->mode) {
    case SVDEIGENSOLVER_DIRECT:
      ierr = MatMult(svd->A,x,eigen->w);CHKERRQ(ierr);
      if (svd->AT) {
        ierr = MatMult(svd->AT,eigen->w,y);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(svd->A,eigen->w,y);CHKERRQ(ierr);
      }
      break;
    case SVDEIGENSOLVER_TRANSPOSE:
      if (svd->AT) {
        ierr = MatMult(svd->AT,x,eigen->w);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(svd->A,x,eigen->w);CHKERRQ(ierr);
      }
      ierr = MatMult(svd->A,eigen->w,y);CHKERRQ(ierr);
      break;
    case SVDEIGENSOLVER_CYCLIC:
      SETERRQ(1,"Not implemented :-)");
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_EIGENSOLVER"
PetscErrorCode SVDSetUp_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  PetscInt        m,n,M,N;

  PetscFunctionBegin;
  
  if (eigen->w) { ierr = VecDestroy(eigen->w);CHKERRQ(ierr); } 
  if (eigen->mat) { ierr = MatDestroy(eigen->mat);CHKERRQ(ierr); }

  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(svd->A,&m,&n);CHKERRQ(ierr);
  switch (eigen->mode) {
    case SVDEIGENSOLVER_DIRECT:
      ierr = MatGetVecs(svd->A,PETSC_NULL,&eigen->w);CHKERRQ(ierr);
      ierr = MatCreateShell(svd->comm,n,n,N,N,svd,&eigen->mat);CHKERRQ(ierr);
      break;
    case SVDEIGENSOLVER_TRANSPOSE:
      ierr = MatGetVecs(svd->A,&eigen->w,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatCreateShell(svd->comm,m,m,M,M,svd,&eigen->mat);CHKERRQ(ierr);
      break;
    case SVDEIGENSOLVER_CYCLIC:
      SETERRQ(1,"Not implemented :-)");
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
  }
  ierr = MatShellSetOperation(eigen->mat,MATOP_MULT,(void(*)(void))ShellMatMult_EIGENSOLVER);CHKERRQ(ierr);  

  ierr = EPSSetOperators(eigen->eps,eigen->mat,PETSC_NULL);CHKERRQ(ierr);
  /* EPSSetProblemType deberia estar en SVDSetFromOptions
     PENDIENTE: ARREGLAR EL PROBLEMA EN EPS */
  ierr = EPSSetProblemType(eigen->eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetUp(eigen->eps);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_EIGENSOLVER"
PetscErrorCode SVDSolve_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  int             i;
  PetscScalar     sigma;
  
  PetscFunctionBegin;
  ierr = EPSSolve(eigen->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(eigen->eps,&svd->nconv);CHKERRQ(ierr);

  if (svd->sigma) { ierr = PetscFree(svd->sigma);CHKERRQ(ierr); }
  if (svd->U) {
    for (i=0;i<svd->nconv;i++) {
      ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr);
    }
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
  }
  if (svd->V) {
    for (i=0;i<svd->nconv;i++) {
      ierr = VecDestroy(svd->V[i]);CHKERRQ(ierr); 
    }
    ierr = PetscFree(svd->V);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc(svd->nconv*sizeof(PetscReal),&svd->sigma);CHKERRQ(ierr);
  ierr = PetscMalloc(svd->nconv*sizeof(Vec),&svd->U);CHKERRQ(ierr);
  ierr = PetscMalloc(svd->nconv*sizeof(Vec),&svd->V);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = MatGetVecs(svd->A,svd->V+i,svd->U+i);CHKERRQ(ierr);
    switch (eigen->mode) {
      case SVDEIGENSOLVER_DIRECT:
        ierr = EPSGetEigenpair(eigen->eps,i,&sigma,PETSC_NULL,svd->V[i],PETSC_NULL);CHKERRQ(ierr);
        svd->sigma[i] = sqrt(PetscRealPart(sigma));
	ierr = MatMult(svd->A,svd->V[i],svd->U[i]);CHKERRQ(ierr);
	ierr = VecScale(svd->U[i],1.0/svd->sigma[i]);CHKERRQ(ierr);
	break;
      case SVDEIGENSOLVER_TRANSPOSE:
        ierr = EPSGetEigenpair(eigen->eps,i,&sigma,PETSC_NULL,svd->U[i],PETSC_NULL);CHKERRQ(ierr);
        svd->sigma[i] = sqrt(PetscRealPart(sigma));
	if (svd->AT) {
  	  ierr = MatMult(svd->AT,svd->U[i],svd->V[i]);CHKERRQ(ierr);
	} else {
  	  ierr = MatMultTranspose(svd->A,svd->U[i],svd->V[i]);CHKERRQ(ierr);
	}
	ierr = VecScale(svd->V[i],1.0/svd->sigma[i]);CHKERRQ(ierr);
	break;
      case SVDEIGENSOLVER_CYCLIC:
	SETERRQ(1,"Not implemented :-)");
      default:
	SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid SVD type"); 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_EIGENSOLVER"
PetscErrorCode SVDSetFromOptions_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  PetscTruth      flg;
  const char      *mode_list[3] = { "direct" , "transpose", "cyclic" };
  PetscInt        mode;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EIGEN options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-svd_eigensolver_mode","Eigensolver SVD mode","SVDEigenSolverSetMode",mode_list,3,mode_list[eigen->mode],&mode,&flg);CHKERRQ(ierr);
  if (flg) { eigen->mode = (SVDEigensolverMode)mode; }
  ierr = EPSSetFromOptions(eigen->eps);
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
    case SVDEIGENSOLVER_DIRECT:
    case SVDEIGENSOLVER_TRANSPOSE:
    case SVDEIGENSOLVER_CYCLIC:
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
   SVDEigensolverSetMode - Sets the transformation used in the eigensolver. 

   Collective on SVD

   Input Parameters:
+  svd  - singular value solver context obtained from SVDCreate()
-  mode - the mode flag, one of SVDEIGENSOLVER_DIRECT, 
          SVDEIGENSOLVER_TRANSPOSE or SVDEIGENSOLVER_CYCLIC

   Options Database Key:
.  -svd_eigensolver_mode <mode> - Indicates the mode flag, where <mode> 
    is one of 'direct', 'transpose' or 'cyclic' (see explanation below).

   Notes:
   This parameter selects the eigensystem used to compute the SVD:
   A^T*A (SVDEIGENSOLVER_DIRECT), A*A^T (SVDEIGENSOLVER_TRANSPOSE) 
   or H(A) = [ 0  A ; A^T 0 ] (SVDEIGENSOLVER_CYCLIC).

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
   SVDEigensolverGetMode - Gets the transformation used by the eigensolver. 

   Not collective

   Input Parameters:
+  svd  - singular value solver context obtained from SVDCreate()
   Output Parameters:
-  mode - the mode flag

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
   SVDEigensolverSetEPS - Associates a eigensolver object to the
   singular value solver. 

   Collective on SVD

   Input Parameters:
+  svd - singular value solver context obtained from SVDCreate()
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
   SVDEigensolverGetEPS - Obtain the eigensolver (EPS) object associated
   to the singular value solver object.

   Not Collective

   Input Parameters:
.  svd - singular value solver context obtained from SVDCreate()

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

  PetscFunctionBegin;
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
  if (eigen->w) { ierr = VecDestroy(eigen->w);CHKERRQ(ierr); }
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
  eigen->mode = SVDEIGENSOLVER_DIRECT;
  eigen->mat = PETSC_NULL;
  eigen->w = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END

