
/*                       
       This file implements a wrapper to the BLZPACK package
*/
#include "src/eps/impls/blzpack/blzpackp.h"   /*I "slepceps.h" I*/

const char* blzpack_error[33] = {
  "",
  "illegal data, LFLAG ",
  "illegal data, dimension of (U), (V), (X) ",
  "illegal data, leading dimension of (U), (V), (X) ",
  "illegal data, leading dimension of (EIG) ",
  "illegal data, number of required eigenpairs ",
  "illegal data, Lanczos algorithm block size ",
  "illegal data, maximum number of steps ",
  "illegal data, number of starting vectors ",
  "illegal data, number of eigenpairs provided ",
  "illegal data, problem type flag ",
  "illegal data, spectrum slicing flag ",
  "illegal data, eigenvectors purification flag ",
  "illegal data, level of output ",
  "illegal data, output file unit ",
  "illegal data, LCOMM (MPI or PVM) ",
  "illegal data, dimension of ISTOR ",
  "illegal data, convergence threshold ",
  "illegal data, dimension of RSTOR ",
  "illegal data on at least one PE ",
  "ISTOR(3:14) must be equal on all PEs ",
  "RSTOR(1:3) must be equal on all PEs ",
  "not enough space in ISTOR to start eigensolution ",
  "not enough space in RSTOR to start eigensolution ",
  "illegal data, number of negative eigenvalues ",
  "illegal data, entries of V ",
  "illegal data, entries of X ",
  "failure in computational subinterval ",
  "file I/O error, blzpack.__.BQ ",
  "file I/O error, blzpack.__.BX ",
  "file I/O error, blzpack.__.Q ",
  "file I/O error, blzpack.__.X ",
  "parallel interface error "
};

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_BLZPACK"
PetscErrorCode EPSSetUp_BLZPACK(EPS eps)
{
  PetscErrorCode ierr;
  int            listor, lrstor, ncuv, N, n, k1, k2, k3, k4;
  EPS_BLZPACK    *blz = (EPS_BLZPACK *)eps->data;
  PetscTruth     flg;
  KSP            ksp;
  PC             pc;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(eps->vec_initial,&n);CHKERRQ(ierr);
  if (eps->ncv) {
    if( eps->ncv < PetscMin(eps->nev+10,eps->nev*2) )
      SETERRQ(0,"Warning: BLZpack recommends that ncv be larger than min(nev+10,nev*2)");
  }
  else eps->ncv = PetscMin(eps->nev+10,eps->nev*2);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;

  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  if (blz->slice) {
    ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&flg);CHKERRQ(ierr);
    if (!flg)
      SETERRQ(PETSC_ERR_SUP,"Shift-and-invert ST is needed for spectrum slicing");
    ierr = STGetKSP(eps->OP,&ksp);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)ksp,KSPPREONLY,&flg);CHKERRQ(ierr);
    if (!flg)
      SETERRQ(PETSC_ERR_SUP,"Preonly KSP is needed for spectrum slicing");
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)pc,PCCHOLESKY,&flg);CHKERRQ(ierr);
    if (!flg)
      SETERRQ(PETSC_ERR_SUP,"Cholesky PC is needed for spectrum slicing");
  }
  if (eps->which!=EPS_SMALLEST_REAL)
    SETERRQ(1,"Wrong value of eps->which");

  k1 = PetscMin(N,180);
  k2 = blz->block_size;
  k4 = PetscMin(eps->ncv,N);
  k3 = 484+k1*(13+k1*2+k2+PetscMax(18,k2+2))+k2*k2*3+k4*2;

  listor = 123+k1*12;
  if (blz->istor) { ierr = PetscFree(blz->istor);CHKERRQ(ierr); }
  ierr = PetscMalloc((17+listor)*sizeof(int),&blz->istor);CHKERRQ(ierr);
  blz->istor[14] = listor;

  if (blz->slice) lrstor = n*(k2*4+k1*2+k4)+k3;
  else lrstor = n*(k2*4+k1)+k3;
  if (blz->rstor) { ierr = PetscFree(blz->rstor);CHKERRQ(ierr); }
  ierr = PetscMalloc((4+lrstor)*sizeof(PetscReal),&blz->rstor);CHKERRQ(ierr);
  blz->rstor[3] = lrstor;

  ncuv = PetscMax(3,blz->block_size);
  if (blz->u)     { ierr = PetscFree(blz->u);CHKERRQ(ierr); }
  ierr = PetscMalloc(ncuv*n*sizeof(PetscScalar),&blz->u);CHKERRQ(ierr);
  if (blz->v)     { ierr = PetscFree(blz->v);CHKERRQ(ierr); }
  ierr = PetscMalloc(ncuv*n*sizeof(PetscScalar),&blz->v);CHKERRQ(ierr);

  if (blz->eig)   { ierr = PetscFree(blz->eig);CHKERRQ(ierr); }
  ierr = PetscMalloc(2*eps->ncv*sizeof(PetscReal),&blz->eig);CHKERRQ(ierr);

  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_BLZPACK"
PetscErrorCode EPSSolve_BLZPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLZPACK    *blz = (EPS_BLZPACK *)eps->data; 
  int            i, n, nneig, lflag, nvopu;      
  Vec            x, y;                           
  PetscScalar    sigma,*pV;                      
  Mat            A;                              
  KSP            ksp;                            
  PC             pc;                             
  
  PetscFunctionBegin;

  ierr = VecGetLocalSize(eps->vec_initial,&n); CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(eps->comm,n,PETSC_DECIDE,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(eps->comm,n,PETSC_DECIDE,PETSC_NULL,&y);CHKERRQ(ierr);
  ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
  
  if (blz->slice) { ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr); }
  else sigma = 0.0;              /* shift of origin */
  nneig = 0;                     /* no. of eigs less than sigma */

  blz->istor[0]  = n;            /* number of rows of U, V, X*/
  blz->istor[1]  = n;            /* leading dimension of U, V, X */
  blz->istor[2]  = eps->nev;     /* number of required eigenpairs */
  blz->istor[3]  = eps->ncv;     /* number of working eigenpairs */
  blz->istor[4]  = blz->block_size;    /* number of vectors in a block */
  blz->istor[5]  = blz->nsteps;  /* maximun number of steps per run */
  blz->istor[6]  = 1;            /* number of starting vectors as input */
  blz->istor[7]  = 0;            /* number of eigenpairs given as input */
  blz->istor[8]  = blz->slice;   /* problem type */
  blz->istor[9]  = blz->slice;   /* spectrum slicing */
  blz->istor[10] = blz->slice;   /* solutions refinement (purify) */
  blz->istor[11] = 0;            /* level of printing */
  blz->istor[12] = 6;            /* file unit for output */
  blz->istor[13] = MPI_Comm_c2f(eps->comm);    /* communicator */

  blz->rstor[0]  = blz->initial; /* lower limit of eigenvalue interval */
  blz->rstor[1]  = blz->final;   /* upper limit of eigenvalue interval */
  blz->rstor[2]  = eps->tol;     /* threshold for convergence */

  lflag = 0;           /* reverse communication interface flag */
  eps->its  = 0;

  do {

    BLZpack_( blz->istor, blz->rstor, &sigma, &nneig, blz->u, blz->v, 
              &lflag, &nvopu, blz->eig, pV );

    switch (lflag) {
    case 1:
      /* compute v = OP u */
      for (i=0;i<nvopu;i++) {
        ierr = VecPlaceArray( x, blz->u+i*n );CHKERRQ(ierr);
        ierr = VecPlaceArray( y, blz->v+i*n );CHKERRQ(ierr);
        if (blz->slice) { 
          ierr = STApplyNoB( eps->OP, x, y ); CHKERRQ(ierr);
	} else {
          ierr = STApply( eps->OP, x, y ); CHKERRQ(ierr);
	}
        ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,y,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      }
      /* monitor */
      eps->nconv  = BLZistorr_(blz->istor,"NTEIG",5);
      EPSMonitor(eps,eps->its,eps->nconv,
        blz->rstor+BLZistorr_(blz->istor,"IRITZ",5),
        eps->eigi,
        blz->rstor+BLZistorr_(blz->istor,"IRITZ",5)+BLZistorr_(blz->istor,"JT",2),
        BLZistorr_(blz->istor,"NRITZ",5));
      eps->its = eps->its + 1;
      if (eps->its >= eps->max_it || eps->nconv >= eps->nev) lflag = 5;
      break;
    case 2:  
      /* compute v = B u */
      for (i=0;i<nvopu;i++) {
        ierr = VecPlaceArray( x, blz->u+i*n );CHKERRQ(ierr);
        ierr = VecPlaceArray( y, blz->v+i*n );CHKERRQ(ierr);
        ierr = STApplyB( eps->OP, x, y ); CHKERRQ(ierr);
      }
      break;
    case 3:  
      /* update shift */
      ierr = STSetShift(eps->OP,sigma);CHKERRQ(ierr);
      ierr = STGetKSP(eps->OP,&ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCGetFactoredMatrix(pc,&A);CHKERRQ(ierr);
      ierr = MatGetInertia(A,&nneig,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      break;
    case 4:  
      /* copy the initial vector */
      ierr = VecPlaceArray(x,blz->v);CHKERRQ(ierr);
      ierr = VecCopy(eps->vec_initial,x);CHKERRQ(ierr); 
      break;
    }
    
  } while (lflag > 0);

  ierr = VecRestoreArray( eps->V[0], &pV ); CHKERRQ(ierr);

  eps->nconv  = BLZistorr_(blz->istor,"NTEIG",5);
  eps->reason = EPS_CONVERGED_TOL;

  for (i=0;i<eps->nconv;i++) {
    eps->eigr[i]=blz->eig[i];
    eps->eigi[i]=0.0;
  }

  if (lflag!=0) { 
    char msg[2048] = "";
    for (i = 0; i < 33; i++) {
      if (blz->istor[15] & (1 << i)) PetscStrcat(msg, blzpack_error[i]);
    }
    SETERRQ2(PETSC_ERR_LIB,"Error in BLZPACK (code=%d): '%s'",blz->istor[15], msg); 
  }
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_BLZPACK"
PetscErrorCode EPSBackTransform_BLZPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLZPACK    *blz = (EPS_BLZPACK *)eps->data;

  PetscFunctionBegin;
  if (!blz->slice) {
    ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_BLZPACK"
PetscErrorCode EPSDestroy_BLZPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLZPACK    *blz = (EPS_BLZPACK *)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (blz->istor) { ierr = PetscFree(blz->istor);CHKERRQ(ierr); }
  if (blz->rstor) { ierr = PetscFree(blz->rstor);CHKERRQ(ierr); }
  if (blz->u)     { ierr = PetscFree(blz->u);CHKERRQ(ierr); }
  if (blz->v)     { ierr = PetscFree(blz->v);CHKERRQ(ierr); }
  if (blz->eig)   { ierr = PetscFree(blz->eig);CHKERRQ(ierr); }
  if (eps->data) {ierr = PetscFree(eps->data);CHKERRQ(ierr);}
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_BLZPACK"
PetscErrorCode EPSView_BLZPACK(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_BLZPACK    *blz = (EPS_BLZPACK *) eps->data;
  PetscTruth     isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSBLZPACK",((PetscObject)viewer)->type_name);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"block size of the block-Lanczos algorithm: %d\n",blz->block_size);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computational interval: [%f,%f]\n",blz->initial,blz->final);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_BLZPACK"
PetscErrorCode EPSSetFromOptions_BLZPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLZPACK    *blz = (EPS_BLZPACK *)eps->data;
  int            bs,n;
  PetscReal      interval[2];
  PetscTruth     flg;
  KSP            ksp;
  PC             pc;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BLZPACK options");CHKERRQ(ierr);

  bs = blz->block_size;
  ierr = PetscOptionsInt("-eps_blzpack_block_size","Block size","EPSBlzpackSetBlockSize",bs,&bs,&flg);CHKERRQ(ierr);
  if (flg) {ierr = EPSBlzpackSetBlockSize(eps,bs);CHKERRQ(ierr);}

  n = blz->nsteps;
  ierr = PetscOptionsInt("-eps_blzpack_nsteps","Number of steps","EPSBlzpackSetNSteps",n,&n,&flg);CHKERRQ(ierr);
  if (flg) {ierr = EPSBlzpackSetNSteps(eps,n);CHKERRQ(ierr);}

  interval[0] = blz->initial;
  interval[1] = blz->final;
  n = 2;
  ierr = PetscOptionsRealArray("-eps_blzpack_interval","Computational interval","EPSBlzpackSetInterval",interval,&n,&flg);CHKERRQ(ierr);
  if (flg) {
    if (n==1) interval[1]=interval[0];
    ierr = EPSBlzpackSetInterval(eps,interval[0],interval[1]);CHKERRQ(ierr);
  }

  if (blz->slice) {
    ierr = STSetType(eps->OP,STSINV);CHKERRQ(ierr);
    ierr = STGetKSP(eps->OP,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSBlzpackSetBlockSize_BLZPACK"
PetscErrorCode EPSBlzpackSetBlockSize_BLZPACK(EPS eps,int bs)
{
  EPS_BLZPACK *blz = (EPS_BLZPACK *) eps->data;;

  PetscFunctionBegin;
  if (bs == PETSC_DEFAULT) blz->block_size = 3;
  else if (bs <= 0) { 
    SETERRQ(1, "Incorrect block size"); 
  } else blz->block_size = bs;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSBlzpackSetBlockSize"
/*@
   EPSBlzpackSetBlockSize - Sets the block size for the BLZPACK package.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  bs - block size

   Options Database Key:
.  -eps_blzpack_block_size - Sets the value of the block size

   Level: advanced

.seealso: EPSBlzpackSetInterval()
@*/
PetscErrorCode EPSBlzpackSetBlockSize(EPS eps,int bs)
{
  PetscErrorCode ierr, (*f)(EPS,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSBlzpackSetBlockSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSBlzpackSetInterval_BLZPACK"
PetscErrorCode EPSBlzpackSetInterval_BLZPACK(EPS eps,PetscReal initial,PetscReal final)
{
  EPS_BLZPACK *blz = (EPS_BLZPACK *) eps->data;;

  PetscFunctionBegin;
  blz->initial    = initial;
  blz->final      = final;
  blz->slice      = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSBlzpackSetInterval"
/*@
   EPSBlzpackSetInterval - Sets the computational interval for the BLZPACK
   package.

   Collective on EPS

   Input Parameters:
+  eps     - the eigenproblem solver context
.  initial - lower bound of the interval
-  final   - upper bound of the interval

   Options Database Key:
.  -eps_blzpack_interval - Sets the bounds of the interval (two values
   separated by commas)

   Note:
   The following possibilities are accepted (see Blzpack user's guide for
   details).
     initial>final: start seeking for eigenpairs in the upper bound
     initial<final: start in the lower bound
     initial=final: run around a single value (no interval)
   
   Level: advanced

.seealso: EPSBlzpackSetBlockSize()
@*/
PetscErrorCode EPSBlzpackSetInterval(EPS eps,PetscReal initial,PetscReal final)
{
  PetscErrorCode ierr, (*f)(EPS,PetscReal,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSBlzpackSetInterval_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,initial,final);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSBlzpackSetNSteps_BLZPACK"
PetscErrorCode EPSBlzpackSetNSteps_BLZPACK(EPS eps,int nsteps)
{
  EPS_BLZPACK *blz = (EPS_BLZPACK *) eps->data;

  PetscFunctionBegin;
  blz->nsteps = nsteps == PETSC_DEFAULT ? 0 : nsteps;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSBlzpackSetNSteps"
/*@
   EPSBlzpackSetNSteps - Sets the maximum number of steps per run for the BLZPACK
   package.

   Collective on EPS

   Input Parameters:
+  eps     - the eigenproblem solver context
-  nsteps  - maximum number of steps

   Options Database Key:
.  -eps_blzpack_nsteps - Sets the maximum number of steps per run

   Level: advanced

@*/
PetscErrorCode EPSBlzpackSetNSteps(EPS eps,int nsteps)
{
  PetscErrorCode ierr, (*f)(EPS,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSBlzpackSetNSteps_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,nsteps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_BLZPACK"
PetscErrorCode EPSCreate_BLZPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLZPACK    *blzpack;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_BLZPACK,&blzpack);CHKERRQ(ierr);
  PetscMemzero(blzpack,sizeof(EPS_BLZPACK));
  PetscLogObjectMemory(eps,sizeof(EPS_BLZPACK));
  eps->data                      = (void *) blzpack;
  eps->ops->solve                = EPSSolve_BLZPACK;
  eps->ops->setup                = EPSSetUp_BLZPACK;
  eps->ops->setfromoptions       = EPSSetFromOptions_BLZPACK;
  eps->ops->destroy              = EPSDestroy_BLZPACK;
  eps->ops->view                 = EPSView_BLZPACK;
  eps->ops->backtransform        = EPSBackTransform_BLZPACK;
  eps->ops->computevectors       = EPSComputeVectors_Default;

  blzpack->block_size = 3;
  blzpack->initial = 0.0;
  blzpack->final = 0.0;
  blzpack->slice = 0;
  blzpack->nsteps = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSBlzpackSetBlockSize_C","EPSBlzpackSetBlockSize_BLZPACK",EPSBlzpackSetBlockSize_BLZPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSBlzpackSetInterval_C","EPSBlzpackSetInterval_BLZPACK",EPSBlzpackSetInterval_BLZPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSBlzpackSetNSteps_C","EPSBlzpackSetNSteps_BLZPACK",EPSBlzpackSetNSteps_BLZPACK);CHKERRQ(ierr);

  eps->which = EPS_SMALLEST_REAL;

  PetscFunctionReturn(0);
}
EXTERN_C_END
