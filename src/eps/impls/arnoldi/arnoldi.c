/*                       

   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi

   Algorithm:

       Arnoldi method with explicit restart and deflation.

   References:

       [1] "Arnoldi Methods in SLEPc", SLEPc Technical Report STR-4, 
           available at http://www.grycap.upv.es/slepc.

   Last update: Feb 2006

*/
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  PetscTruth delayed;
} EPS_ARNOLDI;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARNOLDI"
PetscErrorCode EPSSetUp_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv) {
    if (eps->ncv > N) eps->ncv = N;
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY))
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    ierr = PetscFree(eps->Tl);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->Tl);CHKERRQ(ierr);
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  }
  else { ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi"
/*
   EPSBasicArnoldi - Computes an m-step Arnoldi factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * H = f * e_m^T

   where the columns of V are the Arnoldi vectors (which are B-orthonormal),
   H is an upper Hessenberg matrix, f is the residual vector and e_m is
   the m-th vector of the canonical basis. The vector f is B-orthogonal to
   the columns of V. On exit, beta contains the B-norm of f and the next 
   Arnoldi vector can be computed as v_{m+1} = f / beta. 
*/
PetscErrorCode EPSBasicArnoldi(EPS eps,PetscTruth trans,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            j,m = *M;
  PetscReal      norm;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    if (trans) { ierr = STApplyTranspose(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    else { ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,PETSC_NULL,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,PETSC_NULL,V,V[j+1],H+m*j,&norm,breakdown);CHKERRQ(ierr);
    H[(m+1)*j+1] = norm;
    if (*breakdown) {
      *M = j+1;
      *beta = norm;
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  eps->its++;
  ierr = EPSOrthogonalize(eps,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSOrthogonalize(eps,m,PETSC_NULL,V,f,H+m*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDelayedArnoldi"
/*
   EPSDelayedArnoldi - This function is equivalent to EPSBasicArnoldi but
   performs the computation in a different way. The main idea is that
   reorthogonalization is delayed to the next Arnoldi step. This version is
   more scalable but in some case may be less robust numerically.
*/
PetscErrorCode EPSDelayedArnoldi(EPS eps,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            i,j,m=*M;
  Vec            w,u,t;
  PetscScalar    shh[100],*lhh,dot,dot2;
  PetscReal      norm1=0.0,norm2;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&t);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    eps->count_orthog++;
    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      eps->count_reorthog++;
      ierr = STMInnerProductBegin(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      ierr = STInnerProductBegin(eps->OP,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    if (j>k+1) {
      ierr = STNormBegin(eps->OP,u,&norm2);CHKERRQ(ierr); 
      ierr = VecDotBegin(u,V[j-2],&dot2);CHKERRQ(ierr);
    }
    
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STMInnerProductEnd(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      ierr = STInnerProductEnd(eps->OP,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    if (j>k+1) {
      ierr = STNormEnd(eps->OP,u,&norm2);CHKERRQ(ierr);
      ierr = VecDotEnd(u,V[j-2],&dot2);CHKERRQ(ierr);
      if (PetscAbsScalar(dot2/norm2) > PETSC_MACHINE_EPSILON) {
        *breakdown = PETSC_TRUE;
	*M = j-1;
	*beta = norm2;

	if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
	ierr = VecDestroy(w);CHKERRQ(ierr);
	ierr = VecDestroy(u);CHKERRQ(ierr);
	ierr = VecDestroy(t);CHKERRQ(ierr);
	PetscFunctionReturn(0);
      }
    }
    
    if (j>k) {      
      norm1 = sqrt(PetscRealPart(dot));
      for (i=0;i<j;i++)
	H[m*j+i] = H[m*j+i]/norm1;
      H[m*j+j] = H[m*j+j]/dot;
      
      ierr = VecCopy(V[j],t);CHKERRQ(ierr);
      ierr = VecScale(V[j],1.0/norm1);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm1);CHKERRQ(ierr);
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j>k) {
      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j,lhh,V);CHKERRQ(ierr);
      ierr = VecAXPY(t,-1.0,w);CHKERRQ(ierr);
      for (i=0;i<j;i++)
        H[m*(j-1)+i] += lhh[i];
    }

    if (j>k+1) {
      ierr = VecCopy(u,V[j-1]);CHKERRQ(ierr);
      ierr = VecScale(V[j-1],1.0/norm2);CHKERRQ(ierr);
      H[m*(j-2)+j-1] = norm2;
    }

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
      ierr = VecCopy(t,u);CHKERRQ(ierr);
    }
  }

  ierr = STNorm(eps->OP,t,&norm2);CHKERRQ(ierr);
  ierr = VecScale(t,1.0/norm2);CHKERRQ(ierr);
  ierr = VecCopy(t,V[m-1]);CHKERRQ(ierr);
  H[m*(m-2)+m-1] = norm2;

  eps->count_reorthog++;
  ierr = STMInnerProduct(eps->OP,m,f,V,lhh);CHKERRQ(ierr);
  
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,m,lhh,V);CHKERRQ(ierr);
  ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
  for (i=0;i<m;i++)
    H[m*(m-1)+i] += lhh[i];

  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;
  
  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(t);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDelayedArnoldi1"
PetscErrorCode EPSDelayedArnoldi1(EPS eps,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            i,j,m=*M;
  Vec            w;
  PetscScalar    dot;
  PetscReal      norm=0.0;

  PetscFunctionBegin;
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,PETSC_NULL,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    eps->count_orthog++;
    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STInnerProductBegin(eps->OP,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STInnerProductEnd(eps->OP,V[j],V[j],&dot);CHKERRQ(ierr); 
    }
    
    if (j>k) {      
      norm = sqrt(PetscRealPart(dot));
      ierr = VecScale(V[j],1.0/norm);CHKERRQ(ierr);
      H[m*(j-1)+j] = norm;

      for (i=0;i<j;i++)
	H[m*j+i] = H[m*j+i]/norm;
      H[m*j+j] = H[m*j+j]/dot;      
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }

  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;
  
  ierr = VecDestroy(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ArnoldiResiduals"
/*
   EPSArnoldiResiduals - Computes the 2-norm of the residual vectors from
   the information provided by the m-step Arnoldi factorization,

                    OP * V - V * H = f * e_m^T

   For the approximate eigenpair (k_i,V*y_i), the residual norm is computed as
   |beta*y(end,i)| where beta is the norm of f and y is the corresponding 
   eigenvector of H.
*/
PetscErrorCode ArnoldiResiduals(PetscScalar *H,int ldh,PetscScalar *U,PetscReal beta,int nconv,int ncv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscScalar *work)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  int            i,mout,info;
  PetscScalar    *Y=work+4*ncv;
  PetscReal      w;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork=(PetscReal*)(work+3*ncv);
#endif

  PetscFunctionBegin;

  /* Compute eigenvectors Y of H */
  ierr = PetscMemcpy(Y,U,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","B",PETSC_NULL,&ncv,H,&ldh,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,&info,1,1);
#else
  LAPACKtrevc_("R","B",PETSC_NULL,&ncv,H,&ldh,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,rwork,&info,1,1);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

  /* Compute residual norm estimates as beta*abs(Y(m,:)) */
  for (i=nconv;i<ncv;i++) { 
#if !defined(PETSC_USE_COMPLEX)
    if (eigi[i] != 0 && i<ncv-1) {
      errest[i] = beta*SlepcAbsEigenvalue(Y[i*ncv+ncv-1],Y[(i+1)*ncv+ncv-1]);
      w = SlepcAbsEigenvalue(eigr[i],eigi[i]);
      if (w > errest[i]) 
	errest[i] = errest[i] / w;
      errest[i+1] = errest[i];
      i++;
    } else
#endif
    {
      errest[i] = beta*PetscAbsScalar(Y[i*ncv+ncv-1]);
      w = PetscAbsScalar(eigr[i]);
      if (w > errest[i]) 
	errest[i] = errest[i] / w;
    }
  }  
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
PetscErrorCode EPSSolve_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k;
  Vec            f=eps->work[0];
  PetscScalar    *H=eps->T,*U,*work;
  PetscReal      beta;
  PetscTruth     breakdown;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  ierr = PetscMemzero(eps->T,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);
  ierr = PetscMalloc((eps->ncv+4)*eps->ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  
  eps->nconv = 0;
  eps->its = 0;
  for (i=0;i<eps->ncv;i++) eps->eigr[i]=eps->eigi[i]=eps->errest[i]=0.0;
  EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {

    /* Compute an nv-step Arnoldi factorization */
    eps->nv = eps->ncv;
    if (!arnoldi->delayed) {
      ierr = EPSBasicArnoldi(eps,PETSC_FALSE,H,eps->V,eps->nconv,&eps->nv,f,&beta,&breakdown);CHKERRQ(ierr);
    } else if (eps->orthog_ref == EPS_ORTH_REFINE_NEVER) {
      ierr = EPSDelayedArnoldi1(eps,H,eps->V,eps->nconv,&eps->nv,f,&beta,&breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSDelayedArnoldi(eps,H,eps->V,eps->nconv,&eps->nv,f,&beta,&breakdown);CHKERRQ(ierr);
    }

    /* Reduce H to (quasi-)triangular form, H <- U H U' */
    ierr = PetscMemzero(U,eps->nv*eps->nv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<eps->nv;i++) { U[i*(eps->nv+1)] = 1.0; }
    ierr = EPSDenseSchur(eps->nv,eps->nconv,H,eps->ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* Sort the remaining columns of the Schur form */
    ierr = EPSSortDenseSchur(eps->nv,eps->nconv,H,eps->ncv,U,eps->eigr,eps->eigi,eps->which);CHKERRQ(ierr);

    /* Compute residual norm estimates */
    ierr = ArnoldiResiduals(H,eps->ncv,U,beta,eps->nconv,eps->nv,eps->eigr,eps->eigi,eps->errest,work);CHKERRQ(ierr);
    
    /* Lock converged eigenpairs and update the corresponding vectors,
       including the restart vector: V(:,idx) = V*U(:,idx) */
    k = eps->nconv;
    while (k<eps->nv && eps->errest[k]<eps->tol) k++;
    for (i=eps->nconv;i<=k && i<eps->nv;i++) {
      ierr = VecSet(eps->AV[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(eps->AV[i],eps->nv,U+eps->nv*i,eps->V);CHKERRQ(ierr);
    }
    for (i=eps->nconv;i<=k && i<eps->nv;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);
    if (breakdown) {
      eps->count_breakdown++;
      PetscInfo2(eps,"Breakdown in Arnoldi method (it=%i norm=%g)\n",eps->its,beta);
      ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
      if (breakdown) {
        eps->reason = EPS_DIVERGED_BREAKDOWN;
	PetscInfo(eps,"Unable to generate more start vectors\n");
      }
    }
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }
  
  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_ARNOLDI"
PetscErrorCode EPSSetFromOptions_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ARNOLDI options");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-eps_arnoldi_delayed","Arnoldi with delayed reorthogonalization","EPSArnoldiSetDelayed",PETSC_FALSE,&arnoldi->delayed,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiSetDelayed_ARNOLDI"
PetscErrorCode EPSArnoldiSetDelayed_ARNOLDI(EPS eps,PetscTruth delayed)
{
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  arnoldi->delayed = delayed;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiSetDelayed"
/*@
   EPSArnoldiSetDelayed - Activates or deactivates delayed reorthogonalization 
   in the Arnoldi iteration. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  delayed - boolean flag for toggling delayed reorthogonalization

   Options Database Key:
.  -eps_arnoldi_delayed - Activates delayed reorthogonalization in Arnoldi
   
   Note:
   Delayed reorthogonalization is an aggressive optimization for the Arnoldi
   eigensolver than may provide better scalability, but it is sometimes less 
   robust than the default algorithm.

   Level: advanced

.seealso: EPSArnoldiGetDelayed()
@*/
PetscErrorCode EPSArnoldiSetDelayed(EPS eps,PetscTruth delayed)
{
  PetscErrorCode ierr, (*f)(EPS,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSArnoldiSetDelayed_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,delayed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiGetDelayed_ARNOLDI"
PetscErrorCode EPSArnoldiGetDelayed_ARNOLDI(EPS eps,PetscTruth *delayed)
{
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  *delayed = arnoldi->delayed;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSArnoldiGetDelayed"
/*@C
   EPSArnoldiGetDelayed - Gets the type of reorthogonalization used during the Arnoldi
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  delayed - boolean flag indicating if delayed reorthogonalization has been enabled

   Level: advanced

.seealso: EPSArnoldiSetDelayed()
@*/
PetscErrorCode EPSArnoldiGetDelayed(EPS eps,PetscTruth *delayed)
{
  PetscErrorCode ierr, (*f)(EPS,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSArnoldiGetDelayed_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,delayed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_ARNOLDI"
PetscErrorCode EPSView_ARNOLDI(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     isascii;
  EPS_ARNOLDI    *arnoldi = (EPS_ARNOLDI *)eps->data;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSARNOLDI",((PetscObject)viewer)->type_name);
  }
  if (arnoldi->delayed) {
    ierr = PetscViewerASCIIPrintf(viewer,"using delayed reorthogonalization\n");CHKERRQ(ierr);
  }  
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode EPSSolve_TS_ARNOLDI(EPS);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
PetscErrorCode EPSCreate_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARNOLDI    *arnoldi;
  
  PetscFunctionBegin;
  ierr = PetscNew(EPS_ARNOLDI,&arnoldi);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_ARNOLDI));
  eps->data                      = (void *)arnoldi;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->solvets              = EPSSolve_TS_ARNOLDI;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->setfromoptions       = EPSSetFromOptions_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_ARNOLDI;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  arnoldi->delayed               = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSArnoldiSetDelayed_C","EPSArnoldiSetDelayed_ARNOLDI",EPSArnoldiSetDelayed_ARNOLDI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSArnoldiGetDelayed_C","EPSArnoldiGetDelayed_ARNOLDI",EPSArnoldiGetDelayed_ARNOLDI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

