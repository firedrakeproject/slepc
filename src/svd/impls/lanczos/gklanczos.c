/*                       

   SLEPc singular value solver: "lanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization

   Last update: Jun 2007

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <private/ipimpl.h>                 /*I "slepcip.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscBool oneside;
} SVD_LANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_LANCZOS"
PetscErrorCode SVDSetUp_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;
  PetscInt       i,N,nloc;
  PetscScalar    *pU;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
  if (svd->ncv) { /* ncv set */
    if (svd->ncv<svd->nsv) SETERRQ(((PetscObject)svd)->comm,1,"The value of ncv must be at least nsv"); 
  }
  else if (svd->mpd) { /* mpd set */
    svd->ncv = PetscMin(N,svd->nsv+svd->mpd);
  }
  else { /* neither set: defaults depend on nsv being small or large */
    if (svd->nsv<500) svd->ncv = PetscMin(N,PetscMax(2*svd->nsv,10));
    else { svd->mpd = 500; svd->ncv = PetscMin(N,svd->nsv+svd->mpd); }
  }
  if (!svd->mpd) svd->mpd = svd->ncv;
  if (svd->ncv>svd->nsv+svd->mpd) SETERRQ(((PetscObject)svd)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!svd->max_it)
    svd->max_it = PetscMax(N/svd->ncv,100);
  if (svd->U) {
    ierr = VecGetArray(svd->U[0],&pU);CHKERRQ(ierr);
    for (i=0;i<svd->n;i++) { ierr = VecDestroy(&svd->U[i]); CHKERRQ(ierr); }
    ierr = PetscFree(pU);CHKERRQ(ierr);
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
  }
  if (!lanczos->oneside) {
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    ierr = SVDMatGetLocalSize(svd,&nloc,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(svd->ncv*nloc*sizeof(PetscScalar),&pU);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) {
      ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,nloc,PETSC_DECIDE,pU+i*nloc,&svd->U[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDTwoSideLanczos"
PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec *U,PetscInt k,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,k,PETSC_NULL,U,U[k],work,alpha,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecScale(U[k],1.0/alpha[0]);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i-1],V[i]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,i,PETSC_NULL,V,V[i],work,beta+i-k-1,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecScale(V[i],1.0/beta[i-k-1]);CHKERRQ(ierr);

    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,i,PETSC_NULL,U,U[i],work,alpha+i-k,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecScale(U[i],1.0/alpha[i-k]);CHKERRQ(ierr);
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,U[n-1],v);CHKERRQ(ierr);
  ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,n,PETSC_NULL,V,v,work,beta+n-k-1,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDOneSideLanczos"
static PetscErrorCode SVDOneSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec u,Vec u_1,PetscInt k,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      a,b;
  Vec            temp;
  
  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],u);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,u,V[i]);CHKERRQ(ierr);
    ierr = IPNormBegin(svd->ip,u,&a);CHKERRQ(ierr);
    ierr = IPMInnerProductBegin(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    ierr = IPNormEnd(svd->ip,u,&a);CHKERRQ(ierr);
    ierr = IPMInnerProductEnd(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    
    ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
    ierr = SlepcVecMAXPBY(V[i],1.0/a,-1.0/a,i,work,V);CHKERRQ(ierr);

    ierr = IPOrthogonalizeCGS1(svd->ip,0,PETSC_NULL,i,PETSC_NULL,V,V[i],work,PETSC_NULL,&b);CHKERRQ(ierr);
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);
  
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],u_1);CHKERRQ(ierr);
    ierr = VecAXPY(u_1,-b,u);CHKERRQ(ierr);

    alpha[i-k-1] = a;
    beta[i-k-1] = b;
    temp = u;
    u = u_1;
    u_1 = temp;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,u,v);CHKERRQ(ierr);
  ierr = IPNormBegin(svd->ip,u,&a);CHKERRQ(ierr);
  ierr = IPMInnerProductBegin(svd->ip,v,n,V,work);CHKERRQ(ierr);
  ierr = IPNormEnd(svd->ip,u,&a);CHKERRQ(ierr);
  ierr = IPMInnerProductEnd(svd->ip,v,n,V,work);CHKERRQ(ierr);
    
  ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
  ierr = SlepcVecMAXPBY(v,1.0/a,-1.0/a,n,work,V);CHKERRQ(ierr);

  ierr = IPOrthogonalizeCGS1(svd->ip,0,PETSC_NULL,n,PETSC_NULL,V,v,work,PETSC_NULL,&b);CHKERRQ(ierr);
  
  alpha[n-k-1] = a;
  beta[n-k-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LANCZOS"
PetscErrorCode SVDSolve_LANCZOS(SVD svd)
{
#if defined(SLEPC_MISSING_LAPACK_BDSDC)
  PetscFunctionBegin;
  SETERRQ(((PetscObject)svd)->comm,PETSC_ERR_SUP,"BDSDC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm,*work,*Q,*PT;
  PetscScalar    *swork;
  PetscBLASInt   n,info,*iwork;
  PetscInt       i,j,k,m,nv;
  Vec            v,u=0,u_1=0;
  PetscBool      conv;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(3*svd->n+4)*svd->n,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscBLASInt)*8*svd->n,&iwork);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (svd->which == SVD_SMALLEST) { 
#endif
    ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&swork);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&swork);CHKERRQ(ierr);
  }
#endif

  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = SVDMatGetVecs(svd,PETSC_NULL,&u);CHKERRQ(ierr);
    ierr = SVDMatGetVecs(svd,PETSC_NULL,&u_1);CHKERRQ(ierr);
  }
  
  /* normalize start vector */
  if (svd->nini==0) {
    ierr = SlepcVecSetRandom(svd->V[0],svd->rand);CHKERRQ(ierr);
  }
  ierr = VecNormalize(svd->V[0],&norm);CHKERRQ(ierr);
  
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->n);
    if (lanczos->oneside) {
      ierr = SVDOneSideLanczos(svd,alpha,beta,svd->V,v,u,u_1,svd->nconv,nv,swork);CHKERRQ(ierr);
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,v,svd->U,svd->nconv,nv,swork);CHKERRQ(ierr);
    }

    /* compute SVD of bidiagonal matrix */
    n = nv - svd->nconv;
    ierr = PetscMemzero(PT,sizeof(PetscReal)*n*n);CHKERRQ(ierr);
    ierr = PetscMemzero(Q,sizeof(PetscReal)*n*n);CHKERRQ(ierr);
    for (i=0;i<n;i++)
      PT[i*n+i] = Q[i*n+i] = 1.0;
    ierr = PetscLogEventBegin(SVD_Dense,0,0,0,0);CHKERRQ(ierr);
    LAPACKbdsdc_("U","I",&n,alpha,beta,Q,&n,PT,&n,PETSC_NULL,PETSC_NULL,work,iwork,&info);
    ierr = PetscLogEventEnd(SVD_Dense,0,0,0,0);CHKERRQ(ierr);

    /* compute error estimates */
    k = 0;
    conv = PETSC_TRUE;
    for (i=svd->nconv;i<nv;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i+svd->nconv-1;
      else j = i-svd->nconv;
      svd->sigma[i] = alpha[j];
      svd->errest[i] = PetscAbsScalar(Q[j*n+n-1])*beta[n-1];
      if (alpha[j] > svd->tol) svd->errest[i] /= alpha[j];
      if (conv) {
        if (svd->errest[i] < svd->tol) k++;
        else conv = PETSC_FALSE;
      }
    }
    
    /* check convergence */
    if (svd->its >= svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (svd->nconv+k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    
    /* compute restart vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (svd->which == SVD_SMALLEST) j = n-k-1;
      else j = k;
      for (m=0;m<n;m++) swork[m] = PT[m*n+j];
      ierr = SlepcVecMAXPBY(v,0.0,1.0,n,swork,svd->V+svd->nconv);CHKERRQ(ierr);
    }
    
    /* compute converged singular vectors */
#if !defined(PETSC_USE_COMPLEX)
    if (svd->which == SVD_SMALLEST) {
#endif
    for (i=0;i<k;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i-1;
      else j = i;
      for (m=0;m<n;m++) swork[i*n+m] = PT[m*n+j];
    }
    ierr = SlepcUpdateVectors(n,svd->V+svd->nconv,0,k,swork,n,PETSC_FALSE);CHKERRQ(ierr);
    if (!lanczos->oneside) {
      for (i=0;i<k;i++) {
        if (svd->which == SVD_SMALLEST) j = n-i-1;
        else j = i;
        for (m=0;m<n;m++) swork[i*n+m] = Q[j*n+m];
      }
      ierr = SlepcUpdateVectors(n,svd->U+svd->nconv,0,k,swork,n,PETSC_FALSE);CHKERRQ(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    } else {
      ierr = SlepcUpdateVectors(n,svd->V+svd->nconv,0,k,PT,n,PETSC_TRUE);CHKERRQ(ierr);
      if (!lanczos->oneside) {
        ierr = SlepcUpdateVectors(n,svd->U+svd->nconv,0,k,Q,n,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
#endif
        
    /* copy restart vector from temporary space */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      ierr = VecCopy(v,svd->V[svd->nconv+k]);CHKERRQ(ierr);
    }
        
    svd->nconv += k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }
  
  /* free working space */
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&u_1);CHKERRQ(ierr);
  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(swork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_LANCZOS"
PetscErrorCode SVDSetFromOptions_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)svd)->comm,((PetscObject)svd)->prefix,"LANCZOS Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-svd_lanczos_oneside","Lanczos one-side reorthogonalization","SVDLanczosSetOneSide",lanczos->oneside,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = SVDLanczosSetOneSide(svd,val);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDLanczosSetOneSide_LANCZOS"
PetscErrorCode SVDLanczosSetOneSide_LANCZOS(SVD svd,PetscBool oneside)
{
  SVD_LANCZOS *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  if (lanczos->oneside != oneside) {
    lanczos->oneside = oneside;
    svd->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosSetOneSide"
/*@
   SVDLanczosSetOneSide - Indicate if the variant of the Lanczos method 
   to be used is one-sided or two-sided.

   Collective on SVD

   Input Parameters:
+  svd     - singular value solver
-  oneside - boolean flag indicating if the method is one-sided or not

   Options Database Key:
.  -svd_lanczos_oneside <boolean> - Indicates the boolean flag

   Note:
   By default, a two-sided variant is selected, which is sometimes slightly
   more robust. However, the one-sided variant is faster because it avoids 
   the orthogonalization associated to left singular vectors. It also saves
   the memory required for storing such vectors.

   Level: advanced

.seealso: SVDTRLanczosSetOneSide()
@*/
PetscErrorCode SVDLanczosSetOneSide(SVD svd,PetscBool oneside)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  ierr = PetscTryMethod(svd,"SVDLanczosSetOneSide_C",(SVD,PetscBool),(svd,oneside));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosGetOneSide"
/*@
   SVDLanczosGetOneSide - Gets if the variant of the Lanczos method 
   to be used is one-sided or two-sided.

   Collective on SVD

   Input Parameters:
.  svd     - singular value solver

   Output Parameters:
.  oneside - boolean flag indicating if the method is one-sided or not

   Level: advanced

.seealso: SVDLanczosSetOneSide()
@*/
PetscErrorCode SVDLanczosGetOneSide(SVD svd,PetscBool *oneside)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  ierr = PetscTryMethod(svd,"SVDLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDLanczosGetOneSide_LANCZOS"
PetscErrorCode SVDLanczosGetOneSide_LANCZOS(SVD svd,PetscBool *oneside)
{
  SVD_LANCZOS *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(oneside,2);
  *oneside = lanczos->oneside;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_LANCZOS"
PetscErrorCode SVDDestroy_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  ierr = SVDDestroy_Default(svd);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDLanczosSetOneSide_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDLanczosGetOneSide_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_LANCZOS"
PetscErrorCode SVDView_LANCZOS(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Lanczos reorthogonalization: %s\n",lanczos->oneside ? "one-side" : "two-side");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_LANCZOS"
PetscErrorCode SVDCreate_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(SVD_LANCZOS,&lanczos);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_LANCZOS));
  svd->data                = (void *)lanczos;
  svd->ops->setup          = SVDSetUp_LANCZOS;
  svd->ops->solve          = SVDSolve_LANCZOS;
  svd->ops->destroy        = SVDDestroy_LANCZOS;
  svd->ops->setfromoptions = SVDSetFromOptions_LANCZOS;
  svd->ops->view           = SVDView_LANCZOS;
  lanczos->oneside         = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDLanczosSetOneSide_C","SVDLanczosSetOneSide_LANCZOS",SVDLanczosSetOneSide_LANCZOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDLanczosGetOneSide_C","SVDLanczosGetOneSide_LANCZOS",SVDLanczosGetOneSide_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
