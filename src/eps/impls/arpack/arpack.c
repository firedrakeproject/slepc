
/*                       
       This file implements a wrapper to the ARPACK package
*/
#include "src/eps/impls/arpack/arpackp.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARPACK"
static int EPSSetUp_ARPACK(EPS eps)
{
  int         ierr, N, n, ncv;
  EPS_ARPACK *ar = (EPS_ARPACK *)eps->data;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev+2) SETERRQ(1,"The value of ncv must be at least nev+2"); 
    if (eps->ncv>N) SETERRQ(1,"The value of ncv cannot be larger than N"); 
  }
  else /* set default value of ncv */
    eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),N);

  if (!eps->max_it) eps->max_it = PetscMax(300,(int)ceil(2*N/eps->ncv));
  if (!eps->tol) eps->tol = 1.e-7;

  ncv = eps->ncv;
#if defined(PETSC_USE_COMPLEX)
  if (ar->rwork)  { ierr = PetscFree(ar->rwork);CHKERRQ(ierr); }
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ar->rwork);CHKERRQ(ierr);
  ar->lworkl = 3*ncv*ncv+5*ncv;
  if (ar->workev) { ierr = PetscFree(ar->workev);CHKERRQ(ierr); }
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&ar->workev);CHKERRQ(ierr);
#else
  if( eps->ishermitian ) {
    ar->lworkl = ncv*(ncv+8);
  }
  else {
    ar->lworkl = 3*ncv*ncv+6*ncv;
    if (ar->workev) { ierr = PetscFree(ar->workev);CHKERRQ(ierr); }
    ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&ar->workev);CHKERRQ(ierr);
  }
#endif
  if (ar->workl)  { ierr = PetscFree(ar->workl);CHKERRQ(ierr); }
  ierr = PetscMalloc(ar->lworkl*sizeof(PetscScalar),&ar->workl);CHKERRQ(ierr);
  if (ar->select) { ierr = PetscFree(ar->select);CHKERRQ(ierr); }
  ierr = PetscMalloc(ncv*sizeof(PetscTruth),&ar->select);CHKERRQ(ierr);
  ierr = VecGetLocalSize(eps->vec_initial,&n); CHKERRQ(ierr);
  if (ar->workd)  { ierr = PetscFree(ar->workd);CHKERRQ(ierr); }
  ierr = PetscMalloc(3*n*sizeof(PetscScalar),&ar->workd);CHKERRQ(ierr);

  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARPACK"
static int  EPSSolve_ARPACK(EPS eps)
{
  EPS_ARPACK *ar = (EPS_ARPACK *)eps->data;
  char        bmat[1], *which, howmny[] = "A";
  int         i, n, iparam[11], ipntr[14], ido, info, ierr;
  PetscScalar sigmar, sigmai, *pV, *resid;
  Vec         x, y, w;
  Mat         A;
  PetscTruth  isSinv,rvec;
  MPI_Fint    fcomm;
  
  PetscFunctionBegin;

  fcomm = MPI_Comm_c2f(eps->comm);
  ierr = VecGetLocalSize(eps->vec_initial,&n); CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(eps->comm,n,PETSC_DECIDE,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(eps->comm,n,PETSC_DECIDE,PETSC_NULL,&y);CHKERRQ(ierr);
  ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
  ierr = VecGetArray(eps->vec_initial,&resid);CHKERRQ(ierr);
  
  ido  = 0;            /* first call to reverse communication interface */
  info = 1;            /* indicates a initial vector is provided */
  iparam[0] = 1;       /* use exact shifts */
  iparam[2] = eps->max_it;  /* maximum number of Arnoldi update iterations */
  iparam[3] = 1;       /* blocksize */
  iparam[4] = 0;       /* number of converged Ritz values */
 
  /*
     Computational modes ([]=not supported):
            symmetric    non-symmetric    complex
        1     1  'I'        1  'I'         1  'I'
        2     3  'I'        3  'I'         3  'I'
        3     2  'G'        2  'G'         2  'G'
        4     3  'G'        3  'G'         3  'G'
        5   [ 4  'G' ]    [ 3  'G' ]
        6   [ 5  'G' ]    [ 4  'G' ]
   */
  bmat[0] = eps->isgeneralized? 'G': 'I';
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&isSinv);CHKERRQ(ierr);
  if(isSinv) {
    iparam[6] = 3;
    ierr = STGetShift(eps->OP,&sigmar);CHKERRQ(ierr);
    sigmai = 0.0;
  }
  else iparam[6] = eps->isgeneralized? 2: 1;
 
#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      switch(eps->which) {
        case EPS_LARGEST_MAGNITUDE:  which = "LM"; break;
        case EPS_SMALLEST_MAGNITUDE: which = "SM"; break;
        case EPS_LARGEST_REAL:       which = "LA"; break;
        case EPS_SMALLEST_REAL:      which = "SA"; break;
        default: SETERRQ(1,"Wrong value of eps->which");
      }
    } else {
#endif
      switch(eps->which) {
        case EPS_LARGEST_MAGNITUDE:  which = "LM"; break;
        case EPS_SMALLEST_MAGNITUDE: which = "SM"; break;
        case EPS_LARGEST_REAL:       which = "LR"; break;
        case EPS_SMALLEST_REAL:      which = "SR"; break;
        case EPS_LARGEST_IMAGINARY:  which = "LI"; break;
        case EPS_SMALLEST_IMAGINARY: which = "SI"; break;
        default: SETERRQ(1,"Wrong value of eps->which");
      }
#if !defined(PETSC_USE_COMPLEX)
    }
#endif

  for(;;) {

#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      ARsaupd_( &fcomm, &ido, bmat, &n, which, &eps->nev, &eps->tol,
                resid, &eps->ncv, pV, &n, iparam, ipntr, ar->workd, 
                ar->workl, &ar->lworkl, &info, 1, 2 );
      EPSMonitorEstimates(eps,iparam[2],iparam[4],&ar->workl[ipntr[6]-1],eps->ncv); 
      EPSMonitorValues(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],PETSC_NULL,eps->ncv); 
    }
    else {
      ARnaupd_( &fcomm, &ido, bmat, &n, which, &eps->nev, &eps->tol,
                resid, &eps->ncv, pV, &n, iparam, ipntr, ar->workd, 
                ar->workl, &ar->lworkl, &info, 1, 2 );
      EPSMonitorEstimates(eps,iparam[2],iparam[4],&ar->workl[ipntr[7]-1],eps->ncv); 
      EPSMonitorValues(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],&ar->workl[ipntr[6]-1],eps->ncv); 
    }
#else
    ARnaupd_( &fcomm, &ido, bmat, &n, which, &eps->nev, &eps->tol,
              resid, &eps->ncv, pV, &n, iparam, ipntr, ar->workd, 
              ar->workl, &ar->lworkl, ar->rwork, &info, 1, 2 );
    EPSMonitorEstimates(eps,iparam[2],iparam[4],(PetscReal*)&ar->workl[ipntr[7]-1],eps->ncv); 
    EPSMonitorValues(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],PETSC_NULL,eps->ncv); 
#endif

    if( ido == -1 ) {
      ierr = VecPlaceArray( x, &ar->workd[ipntr[0]-1] );CHKERRQ(ierr);
      ierr = VecPlaceArray( y, &ar->workd[ipntr[1]-1] );CHKERRQ(ierr);
      ierr = STApply( eps->OP, x, y ); CHKERRQ(ierr);
      if (iparam[6]==2) { /* regular inverse mode */
        w = eps->work[0];
        ierr = STGetOperators( eps->OP, &A, PETSC_NULL ); CHKERRQ(ierr);
        ierr = MatMult(A,x,w);CHKERRQ(ierr); 
        ierr = VecCopy( w, x );CHKERRQ(ierr);
      }
    }
    else if( ido == 1 ) {
      ierr = VecPlaceArray( y, &ar->workd[ipntr[1]-1] );CHKERRQ(ierr);
      if (isSinv && eps->isgeneralized) { 
        ierr = VecPlaceArray( x, &ar->workd[ipntr[2]-1] );CHKERRQ(ierr);
        ierr = STApplyNoB( eps->OP, x, y ); CHKERRQ(ierr);
      }
      else {
        ierr = VecPlaceArray( x, &ar->workd[ipntr[0]-1] );CHKERRQ(ierr);
        ierr = STApply( eps->OP, x, y ); CHKERRQ(ierr);
      }
      if (iparam[6]==2) { /* regular inverse mode */
        w = eps->work[0];
        ierr = STGetOperators( eps->OP, &A, PETSC_NULL ); CHKERRQ(ierr);
        ierr = MatMult(A,x,w);CHKERRQ(ierr); 
        ierr = VecCopy( w, x );CHKERRQ(ierr);
      }
    }
    else if( ido == 2 ) {
      ierr = VecPlaceArray( x, &ar->workd[ipntr[0]-1] );CHKERRQ(ierr);
      ierr = VecPlaceArray( y, &ar->workd[ipntr[1]-1] );CHKERRQ(ierr);
      ierr = STApplyB( eps->OP, x, y ); CHKERRQ(ierr);
    }
    else break;
  }

  eps->nconv = iparam[4];
  eps->its   = iparam[2];
  
  if (info==1) { SETERRQ(0,"Exceeded maximum number of iterations in xxAUPD"); }
  else if(info==3) { SETERRQ(1,"No shift could be applied in xxAUPD.\n"
                               "Try increasing the size of NCV relative to NEV."); }
  else if (info!=0) { SETERRQ1(PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxAUPD (%d)",info);}

  if (eps->dropvectors) rvec = PETSC_FALSE;
  else rvec = PETSC_TRUE;

#if !defined(PETSC_USE_COMPLEX)
  if (eps->ishermitian) {
    ARseupd_ ( &fcomm, &rvec, howmny, ar->select, eps->eigr,  
               pV, &n, &sigmar, 
               bmat, &n, which, &eps->nev, &eps->tol,
               resid, &eps->ncv, pV, &n, iparam, ipntr, ar->workd, 
               ar->workl, &ar->lworkl, &info, 1, 1, 2 );
    for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
  }
  else {
    ARneupd_ ( &fcomm, &rvec, howmny, ar->select, eps->eigr, eps->eigi, 
               pV, &n, &sigmar, &sigmai, ar->workev, 
               bmat, &n, which, &eps->nev, &eps->tol,
               resid, &eps->ncv, pV, &n, iparam, ipntr, ar->workd, 
               ar->workl, &ar->lworkl, &info, 1, 1, 2 );
  }
#else
  ARneupd_ ( &fcomm, &rvec, howmny, ar->select, eps->eigr,
             pV, &n, &sigmar, ar->workev, 
             bmat, &n, which, &eps->nev, &eps->tol,
             resid, &eps->ncv, pV, &n, iparam, ipntr, ar->workd, 
             ar->workl, &ar->lworkl, ar->rwork, &info, 1, 1, 2 );
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  if (info!=0) { SETERRQ1(PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxEUPD (%d)",info); }

  ierr = VecRestoreArray( eps->V[0], &pV ); CHKERRQ(ierr);
  ierr = VecRestoreArray( eps->vec_initial, &resid ); CHKERRQ(ierr);
  eps->reason = EPS_CONVERGED_TOL;

  if (eps->ishermitian) {
    ierr = PetscMemcpy(eps->errest,&ar->workl[ipntr[8]-1],eps->nconv);CHKERRQ(ierr);
  } else {
    ierr = PetscMemcpy(eps->errest,&ar->workl[ipntr[10]-1],eps->nconv);CHKERRQ(ierr);
  }

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_ARPACK"
int EPSBackTransform_ARPACK(EPS eps)
{
  ST          st;
  int         ierr,i;
  PetscTruth  isSinv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)st,STSINV,&isSinv);CHKERRQ(ierr);
  if (!isSinv)
    for (i=0;i<eps->nconv;i++) {
      ierr = STBackTransform(st,&eps->eigr[i],&eps->eigi[i]);CHKERRQ(ierr);
    }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_ARPACK"
int EPSDestroy_ARPACK(EPS eps)
{
  EPS_ARPACK *ar = (EPS_ARPACK *)eps->data;
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (ar->workev) { ierr = PetscFree(ar->workev);CHKERRQ(ierr); }
  if (ar->workl)  { ierr = PetscFree(ar->workl);CHKERRQ(ierr); }
  if (ar->select) { ierr = PetscFree(ar->select);CHKERRQ(ierr); }
  if (ar->workd)  { ierr = PetscFree(ar->workd);CHKERRQ(ierr); }
#if defined(PETSC_USE_COMPLEX)
  if (ar->rwork)  { ierr = PetscFree(ar->rwork);CHKERRQ(ierr); }
#endif
  if (eps->data) {ierr = PetscFree(eps->data);CHKERRQ(ierr);}
  ierr = EPSDefaultFreeWork(eps);CHKERRQ(ierr);
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARPACK"
int EPSCreate_ARPACK(EPS eps)
{
  EPS_ARPACK *arpack;
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_ARPACK,&arpack);CHKERRQ(ierr);
  PetscMemzero(arpack,sizeof(EPS_ARPACK));
  PetscLogObjectMemory(eps,sizeof(EPS_ARPACK));
  eps->data                      = (void *) arpack;
  eps->ops->setup                = EPSSetUp_ARPACK;
  eps->ops->solve                = EPSSolve_ARPACK;
  eps->ops->destroy              = EPSDestroy_ARPACK;
  eps->ops->view                 = 0;
  eps->ops->backtransform        = EPSBackTransform_ARPACK;
  PetscFunctionReturn(0);
}
EXTERN_C_END
