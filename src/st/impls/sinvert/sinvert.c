/*
      Implements the shift-and-invert technique for eigenvalue problems.
*/
#include "src/st/stimpl.h"          /*I "slepcst.h" I*/
#include "sinvert.h"             

typedef struct {
  STSinvertMatMode    shift_matrix; /* shift matrix rather than use shell mat */
  MatStructure str;          /* whether matrices have the same pattern or not */
  Mat          mat;
  Vec          w;
} ST_SINV;

#undef __FUNCT__  
#define __FUNCT__ "STApply_Sinvert"
static int STApply_Sinvert(ST st,Vec x,Vec y)
{
  int       ierr;
  ST_SINV   *ctx = (ST_SINV *) st->data;

  PetscFunctionBegin;
  if (st->B) {
    /* generalized eigenproblem: y = (A - sB)^-1 B x */
    ierr = MatMult(st->B,x,ctx->w);CHKERRQ(ierr);
    ierr = STAssociatedKSPSolve(st,ctx->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A - sI)^-1 x */
    ierr = STAssociatedKSPSolve(st,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyNoB_Sinvert"
static int STApplyNoB_Sinvert(ST st,Vec x,Vec y)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = STAssociatedKSPSolve(st,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Sinvert"
int STBackTransform_Sinvert(ST st,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscFunctionBegin;
  /* Note that this is not correct in the case of the RQI solver */
  if (eigr) *eigr = 1.0 / *eigr + st->sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPost_Sinvert"
int STPost_Sinvert(ST st)
{
  ST_SINV      *ctx = (ST_SINV *) st->data;
  PetscScalar  alpha;
  int          ierr;

  PetscFunctionBegin;
  if (ctx->shift_matrix == STSINVERT_MATMODE_INPLACE) {
    alpha = st->sigma;
    if( st->B ) { ierr = MatAXPY(&alpha,st->B,st->A,ctx->str);CHKERRQ(ierr); }
    else { ierr = MatShift( &alpha, st->A ); CHKERRQ(ierr); }
    st->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Sinvert"
static int STSetUp_Sinvert(ST st)
{
  int          ierr;
  ST_SINV      *ctx = (ST_SINV *) st->data;
  PetscScalar  alpha;

  PetscFunctionBegin;

  switch (ctx->shift_matrix) {
  case STSINVERT_MATMODE_INPLACE:
    alpha = -st->sigma;
    if (st->B) { ierr = MatAXPY(&alpha,st->B,st->A,ctx->str);CHKERRQ(ierr); }
    else { ierr = MatShift(&alpha,st->A);CHKERRQ(ierr); }
    /* In the following line, the SAME_NONZERO_PATTERN flag has been used to
     * improve performance when solving a number of related eigenproblems */
    ierr = KSPSetOperators(st->ksp,st->A,st->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case STSINVERT_MATMODE_SHELL:
    ierr = MatCreateMatSinvert(st,&ctx->mat);CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,ctx->mat,ctx->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&ctx->mat);CHKERRQ(ierr);
    alpha = -st->sigma;
    if (st->B) { ierr = MatAXPY(&alpha,st->B,ctx->mat,ctx->str);CHKERRQ(ierr); }
    else { ierr = MatShift(&alpha,ctx->mat);CHKERRQ(ierr); }
    /* In the following line, the SAME_NONZERO_PATTERN flag has been used to
     * improve performance when solving a number of related eigenproblems */
    ierr = KSPSetOperators(st->ksp,ctx->mat,ctx->mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (st->B && !ctx->w) { ierr = VecDuplicate(st->vec,&ctx->w);CHKERRQ(ierr); } 
  ierr = KSPSetRhs(st->ksp,st->vec);CHKERRQ(ierr);
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetShift_Sinvert"
static int STSetShift_Sinvert(ST st,PetscScalar newshift)
{
  int          ierr;
  ST_SINV      *stctx = (ST_SINV *) st->data;
  PetscScalar  alpha;
  CTX_SINV     *ctx;

  PetscFunctionBegin;

  /* Nothing to be done if STSetUp has not been called yet */
  if (!st->setupcalled) PetscFunctionReturn(0);

  switch (stctx->shift_matrix) {
  case STSINVERT_MATMODE_INPLACE:
    /* Undo previous operations */
    alpha = st->sigma;
    if (st->B) { ierr = MatAXPY(&alpha,st->B,st->A,stctx->str);CHKERRQ(ierr); }
    else { ierr = MatShift(&alpha,st->A);CHKERRQ(ierr); }
    /* Apply new shift */
    alpha = -newshift;
    if (st->B) { ierr = MatAXPY(&alpha,st->B,st->A,stctx->str);CHKERRQ(ierr); }
    else { ierr = MatShift(&alpha,st->A);CHKERRQ(ierr); }
    ierr = KSPSetOperators(st->ksp,st->A,st->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case STSINVERT_MATMODE_SHELL:
    ierr = MatShellGetContext(stctx->mat,(void**)&ctx);CHKERRQ(ierr);
    ctx->sigma = newshift;
    ierr = KSPSetOperators(st->ksp,stctx->mat,stctx->mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    ierr = MatCopy(st->A, stctx->mat, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    alpha = -st->sigma;
    if (st->B) { ierr = MatAXPY(&alpha,st->B,stctx->mat,stctx->str);CHKERRQ(ierr); }
    else { ierr = MatShift(&alpha,stctx->mat);CHKERRQ(ierr); }
    /* In the following line, the SAME_NONZERO_PATTERN flag has been used to
     * improve performance when solving a number of related eigenproblems */
    ierr = KSPSetOperators(st->ksp,stctx->mat,stctx->mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);    
  }
  ierr = KSPSetRhs(st->ksp,st->vec);CHKERRQ(ierr);
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STDestroy_Sinvert"
static int STDestroy_Sinvert(ST st)
{
  ST_SINV  *ctx = (ST_SINV *) st->data;
  int      ierr;

  PetscFunctionBegin;
  if (ctx->shift_matrix != STSINVERT_MATMODE_INPLACE && ctx->mat) { 
    ierr = MatDestroy(ctx->mat);CHKERRQ(ierr); 
  }
  if (st->B) { ierr = VecDestroy(ctx->w);CHKERRQ(ierr); }
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView_Sinvert"
static int STView_Sinvert(ST st,PetscViewer viewer)
{
  ST_SINV    *ctx = (ST_SINV *) st->data;
  int        ierr;
  PetscTruth isascii;
  char       *str;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for STSINV",((PetscObject)viewer)->type_name);
  }
  switch (ctx->shift_matrix) {
  case STSINVERT_MATMODE_INPLACE:
    ierr = PetscViewerASCIIPrintf(viewer,"Shifting the matrix and unshifting at exit\n");CHKERRQ(ierr);
    break;
  case STSINVERT_MATMODE_SHELL:
    ierr = PetscViewerASCIIPrintf(viewer,"Using a shell matrix\n");CHKERRQ(ierr);
    break;
  }
  if (st->B && ctx->shift_matrix != STSINVERT_MATMODE_SHELL) { 
    switch (ctx->str) {
      case SAME_NONZERO_PATTERN:      str = "same nonzero pattern";break;
      case DIFFERENT_NONZERO_PATTERN: str = "different nonzero pattern";break;
      case SUBSET_NONZERO_PATTERN:    str = "subset nonzero pattern";break;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"Matrices A and B have %s\n",str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetFromOptions_Sinvert"
static int STSetFromOptions_Sinvert(ST st)
{
  int        i,ierr;
  PetscTruth flg;
  PC         pc;
  const char *mode_list[3] = { "copy", "inplace", "shell" };

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ST Shift-and-invert Options");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-st_sinvert_matmode", "Shift matrix mode","STSinvertSetMatMode",mode_list,3,mode_list[0],&i,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = STSinvertSetMatMode(st, (STSinvertMatMode)i);CHKERRQ(ierr);
    }
    
    ierr = PetscOptionsLogicalGroupBegin("-st_sinvert_same_pattern","same nonzero pattern","STSinvertSetMatStructure",&flg);CHKERRQ(ierr);
    if (flg) {ierr = STSinvertSetMatStructure(st,SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-st_sinvert_different_pattern","different nonzero pattern","STSinvertSetMatStructure",&flg);CHKERRQ(ierr);
    if (flg) {ierr = STSinvertSetMatStructure(st,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-st_sinvert_subset_pattern","subset nonzero pattern","STSinvertSetMatStructure",&flg);CHKERRQ(ierr);
    if (flg) {ierr = STSinvertSetMatStructure(st,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STSinvertSetMatMode_Sinvert"
int STSinvertSetMatMode_Sinvert(ST st,STSinvertMatMode mode)
{
  int        ierr;
  PC         pc;
  ST_SINV    *ctx = (ST_SINV *) st->data;

  PetscFunctionBegin;
  ctx->shift_matrix = mode;
  if (mode == STSINVERT_MATMODE_SHELL) {
    /* if shift_mat is set then the default preconditioner is ILU,
       otherwise set Jacobi as the default */
    ierr = KSPGetPC(st->ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STSinvertSetMatMode"
/*@
   STSinvertSetMatMode - Sets a flag to indicate how the matrix is
   being shifted.

   Collective on ST

   Input Parameters:
+  st - the spectral transformation context
-  mode - the mode flags, one of STSINVERT_MATMODE_COPY, 
          STSINVERT_MATMODE_INPLACE or STSINVERT_MATMODE_SHELL

   Options Database Key:
.  -st_sinvert_matmode <mode> - Activates STSinvertSetMatMode()

   Note:
   By default (STSINVERT_MATMODE_COPY), a copy of the matrix is shifted explicitly. 
   Instead with STSINVERT_MATMODE_INPLACE, the matrix is shifted being shifted at 
   STSetUp() and unshifted at the end of the computations.
   With STSINVERT_MATMODE_SHELL, the solver
   works with an implicit shell matrix that represents the shifted matrix, 
   in which case only the Jacobi preconditioning is available for the linear
   solves performed in each iteration of the eigensolver.
   
   Level: intermediate

.seealso: STSetOperators()
@*/
int STSinvertSetMatMode(ST st, STSinvertMatMode mode)
{
  int ierr, (*f)(ST,STSinvertMatMode);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STSinvertSetMatMode_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,mode);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STSinvertSetMatStructure_Sinvert"
int STSinvertSetMatStructure_Sinvert(ST st,MatStructure str)
{
  ST_SINV    *ctx = (ST_SINV *) st->data;

  PetscFunctionBegin;
  ctx->str = str;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STSinvertSetMatStructure"
/*@
   STSinvertSetMatStructure - Sets an internal MatStructure attribute to 
   indicate which is the relation of the sparsity pattern of the two matrices
   A and B constituting the generalized eigenvalue problem. This function
   has no effect in the case of standard eigenproblems.

   Collective on ST

   Input Parameters:
+  st  - the spectral transformation context
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or
         SUBSET_NONZERO_PATTERN

   Options Database Key:
+  -st_sinvert_same_pattern - Indicates A and B have the same nonzero pattern
.  -st_sinvert_different_pattern - Indicates A and B have different nonzero pattern
-  -st_sinvert_subset_pattern - Indicates B's nonzero pattern is a subset of B's

   Note:
   By default, the sparsity patterns are assumed to be different. If the
   patterns are equal or a subset then it is recommended to set this attribute
   for efficiency reasons (in particular, for internal MatAXPY operations).
   
   Level: advanced

.seealso: STSetOperators()
@*/
int STSinvertSetMatStructure(ST st,MatStructure str)
{
  int ierr, (*f)(ST,MatStructure);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STSinvertSetMatStructure_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Sinvert"
int STCreate_Sinvert(ST st)
{
  int       ierr;
  char      *prefix;
  ST_SINV   *ctx;

  PetscFunctionBegin;
  ierr = PetscNew(ST_SINV,&ctx); CHKERRQ(ierr);
  PetscMemzero(ctx,sizeof(ST_SINV));
  PetscLogObjectMemory(st,sizeof(ST_SINV));
  st->numberofshifts      = 1;
  st->data                = (void *) ctx;

  st->ops->apply          = STApply_Sinvert;
  st->ops->applynoB       = STApplyNoB_Sinvert;
  st->ops->postsolve      = STPost_Sinvert;
  st->ops->backtr         = STBackTransform_Sinvert;
  st->ops->setup          = STSetUp_Sinvert;
  st->ops->setshift       = STSetShift_Sinvert;
  st->ops->destroy        = STDestroy_Sinvert;
  st->ops->setfromoptions = STSetFromOptions_Sinvert;
  st->ops->view           = STView_Sinvert;

  ierr = KSPCreate(st->comm,&st->ksp);CHKERRQ(ierr);
  ierr = STGetOptionsPrefix(st,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(st->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
  ctx->shift_matrix = STSINVERT_MATMODE_COPY;
  ctx->str          = DIFFERENT_NONZERO_PATTERN;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STSinvertSetShiftMat_C","STSinvertSetShiftMat_Sinvert",
                    STSinvertSetShiftMat_Sinvert);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STSinvertSetMatStructure_C","STSinvertSetMatStructure_Sinvert",
                    STSinvertSetMatStructure_Sinvert);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

