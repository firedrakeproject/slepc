/*                       

   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi

   Algorithm:

       Arnoldi method with explicit restart and deflation.

   References:

       [1] "Arnoldi Methods in SLEPc", SLEPc Technical Report STR-4, 
           available at http://www.grycap.upv.es/slepc.

   Last update: June 2005

*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

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
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
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
PetscErrorCode EPSBasicArnoldi(EPS eps,PetscTruth trans,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            j,m = *M;
  PetscReal      norm;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    if (trans) { ierr = STApplyTranspose(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    else { ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,V,V[j+1],H+m*j,&norm,&breakdown);CHKERRQ(ierr);
    H[(m+1)*j+1] = norm;
    if (breakdown) {
      eps->count_breakdown++;
      PetscInfo1(eps,"Breakdown in Arnoldi method (norm=%g)\n",norm);
      *M = j+1;
      *beta = norm;
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  eps->its++;
  ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSOrthogonalize(eps,m,V,f,H+m*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi2"
static PetscErrorCode EPSBasicArnoldi2(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j;
  Vec            w;
  PetscScalar    shh[100],*lhh;

  PetscFunctionBegin;

  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    eps->its++;
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = STMInnerProductBegin(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
    }
    
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = STMInnerProductEnd(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      for (i=0;i<j;i++) {
	H[m*(j-1)+i] += lhh[i];
      }
      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j,lhh,V);CHKERRQ(ierr);
      ierr = VecAXPY(V[j],-1.0,w);CHKERRQ(ierr);
    }
    
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
    ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr); 
    ierr = VecScale(f, 1 / *beta);CHKERRQ(ierr);
    if (j < m-1) {
      H[m*j+j+1] = *beta;
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }

  if (j>k) {
    ierr = STMInnerProduct(eps->OP,m,f,V,lhh);CHKERRQ(ierr);
    for (i=0;i<m;i++) {
      H[m*(m-1)+i] += lhh[i];
    }
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j,lhh,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
  }
  
  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi3"
static PetscErrorCode EPSBasicArnoldi3(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j;
  Vec            w;
  PetscScalar    norm,shh[100],*lhh;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STInnerProductBegin(eps->OP,V[j],V[j],&norm);CHKERRQ(ierr); 
    }
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = STInnerProductEnd(eps->OP,V[j],V[j],&norm);CHKERRQ(ierr);
      
      H[m*j+j] = H[m*j+j]/norm;
      norm = PetscSqrtScalar(norm);
      for (i=0;i<j;i++)
	H[m*j+i] = H[m*j+i]/norm;

      H[m*(j-1)+j] = norm;
     
      ierr = VecScale(V[j],1.0/norm);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);
    }
  }

  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi4"
static PetscErrorCode EPSBasicArnoldi4(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j;
  Vec            w;
  PetscScalar    norm,shh[100],*lhh;

  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);

  PetscFunctionBegin;
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STMInnerProductBegin(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      ierr = STInnerProductBegin(eps->OP,V[j],V[j],&norm);CHKERRQ(ierr); 
    }
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = STMInnerProductEnd(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      ierr = STInnerProductEnd(eps->OP,V[j],V[j],&norm);CHKERRQ(ierr);
      
      H[m*j+j] = H[m*j+j]/norm;
      norm = PetscSqrtScalar(norm);
      for (i=0;i<j;i++)
	H[m*j+i] = H[m*j+i]/norm;

      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j,lhh,V);CHKERRQ(ierr);
      ierr = VecAXPY(V[j],-1.0,w);CHKERRQ(ierr);
      for (i=0;i<j;i++)
        H[m*(j-1)+i] += lhh[i];
      H[m*(j-1)+j] = norm;
     
      ierr = VecScale(V[j],1.0/norm);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);
    }
  }

  ierr = STMInnerProductBegin(eps->OP,m,f,V,lhh);CHKERRQ(ierr);
  ierr = STNormBegin(eps->OP,f,beta);CHKERRQ(ierr);
  ierr = STMInnerProductEnd(eps->OP,m,f,V,lhh);CHKERRQ(ierr);
  ierr = STNormEnd(eps->OP,f,beta);CHKERRQ(ierr);
  
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,m,lhh,V);CHKERRQ(ierr);
  ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
  for (i=0;i<m;i++)
    H[m*(m-1)+i] += lhh[i];

  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi5"
static PetscErrorCode EPSBasicArnoldi5(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j;
  Vec            w,u;
  PetscScalar    shh[100],*lhh;
  PetscReal      norm,hnorm;
  PetscTruth     refinement = PETSC_FALSE;

  PetscFunctionBegin;

  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    eps->its++;
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (refinement) {
      ierr = STNormBegin(eps->OP,u,&norm);CHKERRQ(ierr);
    }
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (refinement) {
      ierr = STNormEnd(eps->OP,u,&norm);CHKERRQ(ierr);
    }
        
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (refinement) {
      H[(j-1)*m+j] = norm;
      ierr = VecScale(u,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(u,V[j]);CHKERRQ(ierr);
    }
    
    switch (eps->orthog_ref) {
    case EPS_ORTH_REFINE_IFNEEDED:
      hnorm = 0.0;
      for (i=0; i<=j; i++)
	hnorm += PetscRealPart(H[m*j+i] * PetscConj(H[m*j+i]));
      hnorm = sqrt(hnorm);
      ierr = STNorm(eps->OP,f,&norm);CHKERRQ(ierr);
      if (norm < eps->orthog_eta * hnorm) {
        refinement = PETSC_TRUE;
        ierr = STMInnerProduct(eps->OP,j+1,f,V,lhh);CHKERRQ(ierr);
      } else refinement = PETSC_FALSE;
      break;
      
    case EPS_ORTH_REFINE_ALWAYS:
      ierr = STMInnerProductBegin(eps->OP,j+1,f,V,lhh);CHKERRQ(ierr);
      ierr = STNormBegin(eps->OP,f,&norm);CHKERRQ(ierr);
      ierr = STMInnerProductEnd(eps->OP,j+1,f,V,lhh);CHKERRQ(ierr);
      ierr = STNormEnd(eps->OP,f,&norm);CHKERRQ(ierr);
      refinement = PETSC_TRUE;
      break;
      
    case EPS_ORTH_REFINE_NEVER:
      refinement = PETSC_FALSE;
      break;
    }

    if (refinement) {
      eps->count_reorthog++;
      for (i=0;i<=j;i++) 
	H[m*j+i] += lhh[i];

      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j+1,lhh,V);CHKERRQ(ierr);
      ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
    }
  
    if (j < m-1) {
      if (refinement) {
        ierr = VecCopy(f,u);CHKERRQ(ierr);
      } else H[m*j+j+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }

  if (refinement) {
    ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  } else *beta = norm;
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi6"
static PetscErrorCode EPSBasicArnoldi6(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j;
  Vec            w,u,t;
  PetscScalar    shh[100],*lhh,norm1;
  PetscReal      norm2;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&t);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STMInnerProductBegin(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      ierr = STInnerProductBegin(eps->OP,V[j],V[j],&norm1);CHKERRQ(ierr); 
    }
    if (j>k+1) {
      ierr = STNormBegin(eps->OP,u,&norm2);CHKERRQ(ierr); 
    }
    
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (j>k) { 
      ierr = STMInnerProductEnd(eps->OP,j,V[j],V,lhh);CHKERRQ(ierr);
      ierr = STInnerProductEnd(eps->OP,V[j],V[j],&norm1);CHKERRQ(ierr); 
    }
    if (j>k+1) {
      ierr = STNormEnd(eps->OP,u,&norm2);CHKERRQ(ierr); 
    }
    
    if (j>k) { 
      H[m*j+j] = H[m*j+j]/norm1;
      norm1 = PetscSqrtScalar(norm1);
      for (i=0;i<j;i++)
	H[m*j+i] = H[m*j+i]/norm1;

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

  ierr = STMInnerProduct(eps->OP,m,f,V,lhh);CHKERRQ(ierr);
  
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,m,lhh,V);CHKERRQ(ierr);
  ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
  for (i=0;i<m;i++)
    H[m*(m-1)+i] += lhh[i];

  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(t);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi7"
static PetscErrorCode EPSBasicArnoldi7(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j;
  Vec            w,u,t;
  PetscScalar    shh[100],*lhh,norm1,hnorm;
  PetscReal      norm2,norm3;
  PetscTruth     refinement = PETSC_FALSE;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&lhh);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&t);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    ierr = STNormBegin(eps->OP,f,&norm3);CHKERRQ(ierr); 
    if (j>k) { 
      ierr = STInnerProductBegin(eps->OP,V[j],V[j],&norm1);CHKERRQ(ierr); 
    }
    if (j>k+1 && refinement) {
      ierr = STNormBegin(eps->OP,u,&norm2);CHKERRQ(ierr); 
    }
    
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    ierr = STNormEnd(eps->OP,f,&norm3);CHKERRQ(ierr); 
    if (j>k) { 
      ierr = STInnerProductEnd(eps->OP,V[j],V[j],&norm1);CHKERRQ(ierr); 
    }
    if (j>k+1 && refinement) {
      ierr = STNormEnd(eps->OP,u,&norm2);CHKERRQ(ierr); 
    }
    
    if (j>k) { 
      H[m*j+j] = H[m*j+j]/norm1;
      norm1 = PetscSqrtScalar(norm1);
      for (i=0;i<j;i++)
	H[m*j+i] = H[m*j+i]/norm1;

      ierr = VecCopy(V[j],t);CHKERRQ(ierr);
      ierr = VecScale(V[j],1.0/norm1);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm1);CHKERRQ(ierr);
//      norm3 = norm3 / norm1;
    }

    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j>k+1 && refinement) {
      ierr = VecCopy(u,V[j-1]);CHKERRQ(ierr);
      ierr = VecScale(V[j-1],1.0/norm2);CHKERRQ(ierr);
      H[m*(j-2)+j-1] = norm2;
    }

    if (j>k) {
/*      hnorm = 0.0;
      for (i=0; i<=j; i++)
	hnorm += PetscRealPart(H[m*(j-1)+i] * PetscConj(H[m*(j-1)+i]));
      hnorm = sqrt(hnorm); */
      if (eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS || norm1 < eps->orthog_eta * hnorm) {
        printf("%e %e\n",norm1,hnorm);
        ierr = STMInnerProduct(eps->OP,j,t,V,lhh);CHKERRQ(ierr);
	ierr = VecSet(w,0.0);CHKERRQ(ierr);
	ierr = VecMAXPY(w,j,lhh,V);CHKERRQ(ierr);
	ierr = VecAXPY(t,-1.0,w);CHKERRQ(ierr);
	for (i=0;i<j;i++)
          H[m*(j-1)+i] += lhh[i];
	refinement = PETSC_TRUE;
      } else {
        H[m*(j-1)+j] = norm1;    
        refinement = PETSC_FALSE;
      }
    }

    if (j<m-1) {
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
      ierr = VecCopy(t,u);CHKERRQ(ierr);
      hnorm = norm3;
    }
  }

  if (refinement) {
    ierr = STNorm(eps->OP,t,&norm2);CHKERRQ(ierr);
    ierr = VecScale(t,1.0/norm2);CHKERRQ(ierr);
    ierr = VecCopy(t,V[m-1]);CHKERRQ(ierr);
    H[m*(m-2)+m-1] = norm2;
  }

  if (eps->orthog_ref == EPS_ORTH_REFINE_ALWAYS || norm1 < eps->orthog_eta * norm3) {
    ierr = STMInnerProduct(eps->OP,m,f,V,lhh);CHKERRQ(ierr);
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,m,lhh,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
    for (i=0;i<m;i++)
      H[m*(m-1)+i] += lhh[i];
  }

  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  ierr = VecScale(f,1.0 / *beta);CHKERRQ(ierr);

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(t);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi8"
static PetscErrorCode EPSBasicArnoldi8(EPS eps,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j,m = *M;
  Vec            w,u;
  PetscScalar    sc[100],*c,alpha;
  PetscReal      norm,onorm,sum;
  PetscTruth     refinement;

  PetscFunctionBegin;

  if (m<=100) c = sc;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&c);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    eps->its++;
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* h = (V^T , f) ; alpha = (f , f) ; norm = |u| */
    ierr = STMInnerProductBegin(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (eps->orthog_ref != EPS_ORTH_REFINE_ALWAYS) {
      ierr = STInnerProductBegin(eps->OP,f,f,&alpha);CHKERRQ(ierr);
    }
    if (j>k) { 
      ierr = STNormBegin(eps->OP,u,&norm);CHKERRQ(ierr); 
    }
    ierr = STMInnerProductEnd(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    if (eps->orthog_ref != EPS_ORTH_REFINE_ALWAYS) {
      ierr = STInnerProductEnd(eps->OP,f,f,&alpha);CHKERRQ(ierr);
    }
    if (j>k) {
      ierr = STNormEnd(eps->OP,u,&norm);CHKERRQ(ierr); 
    }
    
    /* f = f - V h */
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);

    if (j>k) {
      /* v_j = u / |u| */
      H[(j-1)*m+j] = norm;
      ierr = VecScale(u,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(u,V[j]);CHKERRQ(ierr);
    }
    
    /* reorthogonalize f */
    switch (eps->orthog_ref) {
    case EPS_ORTH_REFINE_IFNEEDED:
      /* compute |f| and |f'| */     
      sum = 0.0;
      for (i=0; i<=j; i++)
	sum += PetscRealPart(H[m*j+i] * PetscConj(H[m*j+i])); 
      norm = PetscRealPart(alpha)-sum;
      if (norm >= 0.0) norm = sqrt(norm);
      else { ierr = STNorm(eps->OP,f,&norm);CHKERRQ(ierr); }
      onorm = sqrt(PetscRealPart(alpha));
      /* DGKS condition */
      if (norm < eps->orthog_eta * onorm) {
	/* c = (V^T , f) ; alpha = (f , f) */
	ierr = STMInnerProductBegin(eps->OP,j+1,f,V,c);CHKERRQ(ierr);
	ierr = STInnerProductBegin(eps->OP,f,f,&alpha);CHKERRQ(ierr);
	ierr = STMInnerProductEnd(eps->OP,j+1,f,V,c);CHKERRQ(ierr);
	ierr = STInnerProductEnd(eps->OP,f,f,&alpha);CHKERRQ(ierr);
        refinement = PETSC_TRUE;
      } else refinement = PETSC_FALSE;
      break;
      
    case EPS_ORTH_REFINE_ALWAYS:
      ierr = STMInnerProductBegin(eps->OP,j+1,f,V,c);CHKERRQ(ierr);
      ierr = STInnerProductBegin(eps->OP,f,f,&alpha);CHKERRQ(ierr);
      ierr = STMInnerProductEnd(eps->OP,j+1,f,V,c);CHKERRQ(ierr);
      ierr = STInnerProductEnd(eps->OP,f,f,&alpha);CHKERRQ(ierr);
      refinement = PETSC_TRUE;
      break;
      
    case EPS_ORTH_REFINE_NEVER:
      /* compute |f| and |f'| */     
      onorm = sqrt(PetscRealPart(alpha));
      sum = 0.0;
      for (i=0; i<=j; i++)
	sum += PetscRealPart(H[m*j+i] * PetscConj(H[m*j+i])); 
      norm = PetscRealPart(alpha)-sum;
      if (norm >= 0.0) norm = sqrt(norm);
      else { ierr = STNorm(eps->OP,f,&norm);CHKERRQ(ierr); }
      refinement = PETSC_FALSE;
      break;
    }

    if (refinement) {
      eps->count_reorthog++;
      /* f = f - V c */
      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j+1,c,V);CHKERRQ(ierr);
      ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
      /* h = h + c ; compute |f| and |f'| */  
      sum = 0.0;
      for (i=0;i<=j;i++) {
	H[m*j+i] += c[i];
	sum += PetscRealPart(c[i] * PetscConj(c[i]));
      }
      norm = PetscRealPart(alpha)-sum;
      if (norm >= 0.0) norm = sqrt(norm);
      else { ierr = STNorm(eps->OP,f,&norm);CHKERRQ(ierr); }
      onorm = sqrt(PetscRealPart(alpha));
    }

    /* check breakdown */
    if (norm < eps->orthog_eta * onorm) {
      eps->count_breakdown++;
      PetscInfo2(eps,"Breakdown in Arnoldi method (it=%i norm=%g)\n",eps->its,norm);
      *M = j+1;
      ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
      if (m>100) { ierr = PetscFree(c);CHKERRQ(ierr); }
      ierr = VecDestroy(w);CHKERRQ(ierr);
      ierr = VecDestroy(u);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }

    /* normalize f */
    if (j<m-1) {
      ierr = VecCopy(f,u);CHKERRQ(ierr);
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }    
  }

  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  if (m>100) { ierr = PetscFree(c);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi9"
static PetscErrorCode EPSBasicArnoldi9(EPS eps,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j,m = *M;
  Vec            w,t,u;
  PetscScalar    sc[100],*c,fdot,tdot;
  PetscReal      fnorm,tnorm,norm,sum;

  PetscFunctionBegin;

  if (m<=100) c = sc;
  else { ierr = PetscMalloc(m*sizeof(PetscScalar),&c);CHKERRQ(ierr); }
  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(f,&t);CHKERRQ(ierr);

  for (j=k;j<m;j++) {
    eps->its++;
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* h = (V^T , f) ; fdot = (f , f) */
    /* c = (V^T , t) ; tdot = (t , t) */
    /* norm = |u| */
    ierr = STMInnerProduct(eps->OP,j+1,f,V,H+m*j);CHKERRQ(ierr);
    ierr = STInnerProduct(eps->OP,f,f,&fdot);CHKERRQ(ierr);
    if (j>k) {
      ierr = STMInnerProduct(eps->OP,j,t,V,c);CHKERRQ(ierr);
      ierr = STInnerProduct(eps->OP,t,t,&tdot);CHKERRQ(ierr);
    }
    if (j>k+1) {
      ierr = STNorm(eps->OP,u,&norm);CHKERRQ(ierr);
      /* check breakdown */
      if (norm < eps->orthog_eta * tnorm) {
	eps->count_breakdown++;
	PetscInfo2(eps,"Breakdown in Arnoldi method (it=%i norm=%g)\n",eps->its,norm);
	*M = j-1;
	ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
	if (m>100) { ierr = PetscFree(c);CHKERRQ(ierr); }
	ierr = VecDestroy(w);CHKERRQ(ierr);
	ierr = VecDestroy(u);CHKERRQ(ierr);
	PetscFunctionReturn(0);
      }      
    }
      
    /* f = f - V h */
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,j+1,H+m*j,V);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
    /* fnorm = |f| */
    for (sum=0.0,i=0; i<=j; i++) sum += PetscRealPart(H[m*j+i] * PetscConj(H[m*j+i])); 
    fnorm = PetscRealPart(fdot)-sum;
    if (fnorm >= 0.0) fnorm = sqrt(fnorm);
    else { ierr = STNorm(eps->OP,f,&fnorm);CHKERRQ(ierr); }
    
    if (j>k) {
      /* t = t - V c ; h = h + c*/
      ierr = VecSet(w,0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(w,j,c,V);CHKERRQ(ierr);
      ierr = VecAXPY(t,-1.0,w);CHKERRQ(ierr);
      for (i=0;i<j;i++)	H[m*(j-1)+i] += c[i];
      /* tnorm = |t| */
      for (sum=0.0,i=0; i<j; i++) sum += PetscRealPart(c[i] * PetscConj(c[i])); 
      tnorm = PetscRealPart(tdot)-sum;
      if (tnorm >= 0.0) tnorm = sqrt(tnorm);
      else { ierr = STNorm(eps->OP,t,&tnorm);CHKERRQ(ierr); }
      /* v_j = t / |t|  */
      ierr = VecCopy(t,V[j]);CHKERRQ(ierr);
      ierr = VecScale(V[j],1.0/tnorm);CHKERRQ(ierr);      
    }
    
    if (j>k+1) {
      /* v_j-1 = u / |u|  */
      ierr = VecCopy(u,V[j-1]);CHKERRQ(ierr);
      ierr = VecScale(V[j-1],1.0/norm);CHKERRQ(ierr);
      H[m*(j-2)+j-1] = norm;
    }
    
    if (j<m-1) {
      /* v_j+1 = f / |f| */
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
      ierr = VecScale(V[j+1],1.0/fnorm);CHKERRQ(ierr);
      /* u = t ; t = f */
      ierr = VecCopy(t,u);CHKERRQ(ierr);
      ierr = VecCopy(f,t);CHKERRQ(ierr);
    }
  }  
  
  ierr = VecCopy(t,V[m-1]);CHKERRQ(ierr);
  ierr = STNorm(eps->OP,V[m-1],&norm);CHKERRQ(ierr);
  ierr = VecScale(V[m-1],1.0/norm);CHKERRQ(ierr);  
  H[m*(m-1)-1] = norm;

  ierr = STMInnerProduct(eps->OP,m,f,V,c);CHKERRQ(ierr);
  ierr = VecSet(w,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(w,m,c,V);CHKERRQ(ierr);
  ierr = VecAXPY(f,-1.0,w);CHKERRQ(ierr);
  for (i=0;i<m;i++) H[m*(m-1)+i] += c[i];
  ierr = STNorm(eps->OP,f,beta);CHKERRQ(ierr);
  
  if (m>100) { ierr = PetscFree(c);CHKERRQ(ierr); }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(t);CHKERRQ(ierr);
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
        errest[i] = beta*SlepcAbsEigenvalue(Y[i*ncv+ncv-1],Y[(i+1)*ncv+ncv-1]) /
                	 SlepcAbsEigenvalue(eigr[i],eigi[i]);
        errest[i+1] = errest[i];
        i++;
    } else
#endif
    errest[i] = beta*PetscAbsScalar(Y[i*ncv+ncv-1]) / PetscAbsScalar(eigr[i]);
  }  
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
PetscErrorCode EPSSolve_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k,type=1;
  Vec            f=eps->work[0];
  PetscScalar    *H=eps->T,*U,*work;
  PetscReal      beta,lev;
  const char     *pre;
  PetscTruth     orthog,breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(eps->T,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);
  ierr = PetscMalloc((eps->ncv+4)*eps->ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  
  ierr = PetscObjectGetOptionsPrefix((PetscObject)eps,&pre);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(pre,"-arnoldi",&type,PETSC_NULL);CHKERRQ(ierr);

  eps->nconv = 0;
  eps->its = 0;
  EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nv);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsHasName(PETSC_NULL,"-orthog",&orthog);CHKERRQ(ierr);
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {

    eps->nv = eps->ncv;
    /* Compute an ncv-step Arnoldi factorization */
    switch (type) {
    case 1:
      ierr = EPSBasicArnoldi(eps,PETSC_FALSE,H,eps->V,eps->nconv,&eps->nv,f,&beta);CHKERRQ(ierr);
      break;    
    case 2:
      ierr = EPSBasicArnoldi2(eps,H,eps->V,eps->nconv,eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 3:
      ierr = EPSBasicArnoldi3(eps,H,eps->V,eps->nconv,eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 4:
      ierr = EPSBasicArnoldi4(eps,H,eps->V,eps->nconv,eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 5:
      ierr = EPSBasicArnoldi5(eps,H,eps->V,eps->nconv,eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 6:
      ierr = EPSBasicArnoldi6(eps,H,eps->V,eps->nconv,eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 7:
      ierr = EPSBasicArnoldi7(eps,H,eps->V,eps->nconv,eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 8:
      ierr = EPSBasicArnoldi8(eps,H,eps->V,eps->nconv,&eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    case 9:
      ierr = EPSBasicArnoldi9(eps,H,eps->V,eps->nconv,&eps->nv,f,&beta);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(1,"Unknown Arnoldi method");
    }    
     
    if (orthog) {
      ierr = SlepcCheckOrthogonality(eps->V,eps->nv,eps->V,eps->nv,PETSC_NULL,&lev);CHKERRQ(ierr);
      if (lev > eps->level_orthog) eps->level_orthog = lev;
    }
    
    /* Reduce H to (quasi-)triangular form, H <- U H U' */
    ierr = PetscMemzero(U,eps->nv*eps->nv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<eps->nv;i++) { U[i*(eps->nv+1)] = 1.0; }
    ierr = EPSDenseSchur(eps->nv,eps->nconv,H,eps->ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* Sort the remaining columns of the Schur form */
    ierr = EPSSortDenseSchur(eps->nv,eps->nconv,H,eps->ncv,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

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
    if (eps->nv < eps->ncv) {
      ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
      if (breakdown) {
        eps->reason = EPS_DIVERGED_BREAKDOWN;
	PetscInfo(eps,"Unable to generate more start vectors\n");
      }
    }
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }
  
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode EPSSolve_TS_ARNOLDI(EPS);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
PetscErrorCode EPSCreate_ARNOLDI(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->solvets              = EPSSolve_TS_ARNOLDI;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

