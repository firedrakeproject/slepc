/*
     SVD routines for setting up the solver.
*/
#include "src/svd/svdimpl.h"      /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSetOperator"
/*@C
   SVDSetOperator - Set the matrix associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd - the singular value solver context
-  A  - the matrix associated with the singular value problem

   Level: beginner

.seealso: SVDSolve(), SVDGetOperators()
@*/
PetscErrorCode SVDSetOperator(SVD svd,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidHeaderSpecific(mat,MAT_COOKIE,2);
  PetscCheckSameComm(svd,1,mat,2);
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  if (svd->A) {
    ierr = MatDestroy(svd->A);CHKERRQ(ierr);
  }
  svd->A = mat;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetTransposeMode"
/*@C
   SVDSetTransposeMode - Sets how to handle the transpose of the matrix 
   associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd  - the singular value solver context
.  mode - how to compute the transpose, one of SVD_TRANSPOSE_EXPLICIT, 
          SVD_TRANSPOSE_MATMULT or SVD_TRANSPOSE_USER (see notes below)
-  mat  - the transpose of the matrix associated with the singular value problem 
          when mode is SVD_TRANSPOSE_USER

   Options Database Key:
.  -svd_transpose_mode <mode> - Indicates the mode flag, where <mode> 
    is one of 'explicit', 'matmult' or 'user'.

   Notes:
   In the SVD_TRANSPOSE_EXPLICIT mode, the transpose of the matrix is
   explicitly built.

   The option SVD_TRANSPOSE_MATMULT does not build the transpose, but
   handles it implicitly via MatMultTranspose() operations. This is 
   likely to be more inefficient than SVD_TRANSPOSE_EXPLICIT, both in
   sequential and in parallel, but requires less storage.

   The option SVD_TRANSPOSE_USER is reserved for the case that the
   explicit transpose is already available and provided by the user.

   The default is SVD_TRANSPOSE_EXPLICIT if the matrix has defined the
   MatTranspose operation, and SVD_TRANSPOSE_MATMULT otherwise.
   
   Level: advanced
   
   .seealso: SVDSolve(), SVDSetOperator(), SVDGetOperators()
@*/
EXTERN PetscErrorCode SVDSetTransposeMode(SVD svd,SVDTransposeMode mode,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  switch (mode) {
    case SVD_TRANSPOSE_USER:
      PetscValidHeaderSpecific(mat,MAT_COOKIE,3);
      PetscCheckSameComm(svd,1,mat,2);
      ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      svd->AT = mat;
    case SVD_TRANSPOSE_EXPLICIT:
    case SVD_TRANSPOSE_MATMULT:
    case PETSC_DEFAULT:
      svd->transmode = mode;
      svd->setupcalled = 0;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetOperators"
/*@C
   SVDGetOperators - Get the matrices associated with the singular value problem.

   Not collective, though parallel Mats are returned if the SVD is parallel

   Input Parameter:
.  svd - the singular value solver context

   Output Parameters:
+  A    - the matrix associated with the singular value problem
.  mode - how to compute the transpose
-  AT   - the transpose of this matrix (PETSC_NULL in default mode)

   Level: advanced
   
   Notes:
   Any output parameter can be PETSC_NULL on input if it is not needed.

.seealso: SVDSetOperator(), SVDSetTransposeMode()
@*/
PetscErrorCode SVDGetOperators(SVD svd,Mat *A,SVDTransposeMode *mode,Mat *AT)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (A) *A = svd->A;
  if (mode) *mode = svd->transmode;
  if (AT) *AT = svd->AT;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions"
/*@
   SVDSetFromOptions - Sets SVD options from the options database.
   This routine must be called before SVDSetUp() if the user is to be 
   allowed to set the solver type. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner

.seealso: 
@*/
PetscErrorCode SVDSetFromOptions(SVD svd)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscTruth     flg;
  const char      *mode_list[3] = { "explicit", "matmult", "user" };
  PetscInt        mode;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  svd->setupcalled = 0;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"Singular Value Solver (SVD) Options","SVD");CHKERRQ(ierr);

  ierr = PetscOptionsList("-svd_type","Singular Value Solver method","SVDSetType",SVDList,(char*)(svd->type_name?svd->type_name:SVDEIGENSOLVER),type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDSetType(svd,type);CHKERRQ(ierr);
  } else if (!svd->type_name) {
    ierr = SVDSetType(svd,SVDEIGENSOLVER);CHKERRQ(ierr);
  }

  ierr = PetscOptionsName("-svd_view","Print detailed information on solver used","SVDiew",0);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-svd_transpose_mode","Transpose SVD mode","SVDSetTransposeMode",mode_list,3,svd->transmode == -1 ? mode_list[0] : mode_list[svd->transmode],&mode,&flg);CHKERRQ(ierr);
  if (flg) {
    svd->transmode = (SVDTransposeMode)mode;
  }   

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (svd->ops->setfromoptions) {
    ierr = (*svd->ops->setfromoptions)(svd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp"
/*@
   SVDSetUp - Sets up all the internal data structures necessary for the
   execution of the singular value solver.

   Collective on SVD

   Input Parameter:
.  SVD   - singular value solver context

   Level: advanced

   Notes:
   This function need not be called explicitly in most cases, since SVDSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

.seealso: SVDCreate(), SVDSolve(), SVDDestroy()
@*/
PetscErrorCode SVDSetUp(SVD svd)
{
  PetscErrorCode ierr;
  int            i;
  PetscTruth     flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (svd->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!svd->type_name) {
    ierr = SVDSetType(svd,SVDEIGENSOLVER);CHKERRQ(ierr);
  }

  /* check matrix */
  if (!svd->A) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSetOperator must be called first");     
  }
  
  /* determine how to build the transpose */
  if (svd->transmode == -1) {
    ierr = MatHasOperation(svd->A,MATOP_TRANSPOSE,&flg);CHKERRQ(ierr);    
    if (flg) svd->transmode = SVD_TRANSPOSE_EXPLICIT;
    else svd->transmode = SVD_TRANSPOSE_MATMULT;
  }
  
  /* build transpose matrix */
  switch (svd->transmode) {
    case SVD_TRANSPOSE_USER:
      if (!svd->AT) {
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSetTransposeOperator must be called first");     
      }
      break;
    case SVD_TRANSPOSE_EXPLICIT:
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      ierr = MatTranspose(svd->A,&svd->AT);CHKERRQ(ierr);
      break;
    case SVD_TRANSPOSE_MATMULT:
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      svd->AT = PETSC_NULL;    
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }

  /* free memory for previous solution  */
  if (svd->n) { 
    ierr = PetscFree(svd->sigma);CHKERRQ(ierr);
    for (i=0;i<svd->n;i++) {
      ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr);
    }
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
    for (i=0;i<svd->n;i++) {
      ierr = VecDestroy(svd->V[i]);CHKERRQ(ierr); 
    }
    ierr = PetscFree(svd->V);CHKERRQ(ierr);
  }

  /* call specific solver setup */
  ierr = (*svd->ops->setup)(svd);CHKERRQ(ierr);

  /* allocate memory for solution */
  ierr = PetscMalloc(svd->n*sizeof(PetscReal),&svd->sigma);CHKERRQ(ierr);
  ierr = PetscMalloc(svd->n*sizeof(Vec),&svd->U);CHKERRQ(ierr);
  ierr = PetscMalloc(svd->n*sizeof(Vec),&svd->V);CHKERRQ(ierr);
  for (i=0;i<svd->n;i++) {
    ierr = MatGetVecs(svd->A,svd->V+i,svd->U+i);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);
  svd->setupcalled = 1;
  PetscFunctionReturn(0);
}
