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
  if (svd->AT) {
    ierr = MatDestroy(svd->AT);CHKERRQ(ierr);
    svd->AT = PETSC_NULL;
  }
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetTransposeMode"
/*@C
   SVDSetTransposeMode - Sets how to compute the transpose of the matrix 
   associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd  - the singular value solver context
.  mode - how to compute the transpose, one of SVD_TRANSPOSE_DEFAULT, 
          SVD_TRANSPOSE_EXPLICIT or SVD_TRANSPOSE_USERDEFINED (see notes below)
-  mat  - the transpose of the matrix associated with the singular value problem 
          when mode is SVD_TRANSPOSE_USERDEFINED

   Options Database Key:
.  -svd_transpose_mode <mode> - Indicates the mode flag, where <mode> 
    is one of 'default', 'explicit' or 'user'.

   Notes:    
   The default behaviour is to compute explicitly the transpose matrix in order 
   to improve parallel perfomance (SVD_TRANSPOSE_EXPLICIT).
   It reverts to SVD_TRANSPOSE_DEFAULT in sequential runs or when the matrix
   is a shell one, in this case the PETSc MatMultTranspose() function with the
   original matrix is used.
   The user can provide its own transpose with SVD_TRANSPOSE_USERDEFINED, in 
   this case the PETSc MatMult() function with the specified matrix is used.

   Level: advanced
   
   .seealso: SVDSolve(), SVDSetOperator(), SVDGetOperators()
@*/
EXTERN PetscErrorCode SVDSetTransposeMode(SVD svd,SVDTransposeMode mode,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  switch (svd->transmode) {
    case SVD_TRANSPOSE_USERDEFINED:
      PetscValidHeaderSpecific(mat,MAT_COOKIE,3);
      PetscCheckSameComm(svd,1,mat,2);
      ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      svd->AT = mat;
      break;
    case SVD_TRANSPOSE_EXPLICIT:
    case SVD_TRANSPOSE_DEFAULT:
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      svd->AT = PETSC_NULL;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }
  svd->setupcalled = 0;
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
  const char      *mode_list[3] = { "default" , "explicit", "user" };
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

  ierr = PetscOptionsEList("-svd_transpose_mode","Transpose SVD mode","SVDSetTransposeMode",mode_list,3,mode_list[svd->transmode],&mode,&flg);CHKERRQ(ierr);
  if (flg) {
    svd->transmode = (SVDTransposeMode)mode;
  }   

  if (svd->ops->setfromoptions) {
    ierr = (*svd->ops->setfromoptions)(svd);CHKERRQ(ierr);
  }
  
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  
  /* build transpose matrix */
  switch (svd->transmode) {
    case SVD_TRANSPOSE_USERDEFINED:
      if (!svd->AT) {
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSetOperator must be called first");     
      }
      break;
    case SVD_TRANSPOSE_EXPLICIT:
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      ierr = MatTranspose(svd->A,&svd->AT);CHKERRQ(ierr);
      break;
    case SVD_TRANSPOSE_DEFAULT:
      if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
      svd->AT = PETSC_NULL;    
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }

  /* call specific solver setup */
  ierr = (*svd->ops->setup)(svd);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);
  svd->setupcalled = 1;
  PetscFunctionReturn(0);
}
