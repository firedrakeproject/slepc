
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

/* #define DEBUG */

typedef PetscTruth logical;
typedef PetscBLASInt integer;
typedef PetscScalar doublereal;
typedef PetscBLASInt ftnlen;

extern int dlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *info);
        
extern doublereal dcond_(integer *m, doublereal *h__, integer *ldh, integer *ipvt, 
	doublereal *mult, integer *info);

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_SRRIT"
static int EPSSetUp_SRRIT(EPS eps)
{
  int       ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+15);
  eps->ncv = PetscMin(eps->ncv,N);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSdgroup"
static int EPSdgroup(int l,int m,PetscScalar *wr,PetscScalar *wi,PetscScalar *rsd,PetscScalar grptol,
  int *ngrp,PetscScalar *ctr,PetscScalar *ae,PetscScalar *arsd)
{
  int       i;
  PetscReal rmod,rmod1;

  PetscFunctionBegin;
  *ngrp = 0;
  *ctr = 0;
      
/* rmod = norm([ wr(l) wi(l) ]) */;
   rmod = LAlapy2_(wr+l,wi+l);

/* while 1
     l1 = l + ngrp;
     if l1>m, break, end
     rmod1 = norm([ wr(l1) wi(l1) ]);   % dlapy2
     if abs(rmod-rmod1)>grptol*(rmod+rmod1), break, end
     ctr = (rmod+rmod1)/2.0;
     if wi(l1)~=0
       ngrp = ngrp + 2;
     else
       ngrp = ngrp + 1;
     end
   end */
  for (i=l;i<m;) {
    rmod1 = LAlapy2_(wr+i,wi+i);
    if (PetscAbsReal(rmod-rmod1) > grptol*(rmod+rmod1)) break;
    *ctr = (rmod+rmod1)/2.0;
    if (wi[i] != 0.0) {
      (*ngrp)+=2;
      i+=2;
    } else {
      (*ngrp)++;
      i++;
    }
  }

  *ae = 0;
  *arsd = 0;

/* if ngrp~=0
     l1 = l + ngrp - 1;
     for j = l:l1
       ae = ae + wr(j);
       arsd = arsd + rsd(j)*rsd(j);
     end
     ae = ae/ngrp;
     arsd = sqrt(arsd/ngrp);
   end */
  if (*ngrp) {
    for (i=l;i<l+*ngrp;i++) {
      (*ae) += wr[i];
      (*arsd) += rsd[i]*rsd[i];
    }
    *ae = *ae / *ngrp;
    *arsd = PetscSqrtScalar(*arsd / *ngrp);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSchurResidualNorms"
static int EPSSchurResidualNorms(EPS eps,Vec *V,Vec *AV,PetscScalar *T,int l,int m,int ldt,PetscScalar *rsd)
{
  int         ierr,i;
  PetscScalar zero = 0.0,minus = -1.0;
    
/* R = AV(:,l+1:m) - V*T(:,l+1:m)
   rsd = sum(R.^2,1) */
  PetscFunctionBegin;
  for (i=l;i<m;i++) {
    ierr = VecSet(&zero,eps->work[0]);CHKERRQ(ierr);
    ierr = VecMAXPY(m,T+ldt*i,eps->work[0],V);CHKERRQ(ierr);
    ierr = VecWAXPY(&minus,eps->work[0],AV[i],eps->work[1]);CHKERRQ(ierr);
    ierr = VecDot(eps->work[1],eps->work[1],rsd+i);CHKERRQ(ierr);
  }

/* jnext = l+1 
   k = 0 
   for j=l+1:m 
     k = k + 1 
     if j<jnext, continue, end 
     if j==m 
       rsd(k) = sqrt(rsd(k)) 
     else 
       if T(j+1,j)==0.0 
         rsd(k) = sqrt(rsd(k)) 
         jnext = jnext + 1 
       else
         rsd(k) = sqrt((rsd(k)+rsd(k+1))/2)
         rsd(k+1) = rsd(k) 
         jnext = jnext + 2 
       end 
     end 
   end */
  for (i=l;i<m;i++) {
    if (i == m-1) {
      rsd[i] = PetscSqrtScalar(rsd[i]);  
    } else if (T[i+1+(ldt*i)]==0.0) {
      rsd[i] = PetscSqrtScalar(rsd[i]);
    } else {
      rsd[i] = PetscSqrtScalar(rsd[i]+rsd[i+1])/2.0;
      rsd[i+1] = rsd[i];
      i++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_SRRIT"
static int EPSSolve_SRRIT(EPS eps)
{
  int         ierr,i,j,N,info,ilo,lwork,ngrp,nogrp,*itrsd,*itrsdold,
              nxtsrr,idsrr,*iwork,idort,nxtort,ncv = eps->ncv,one = 1;
  PetscTruth  true = PETSC_TRUE;
  PetscScalar *T,*U,*tau,*work,zero = 0.0,
              ctr,ae,arsd,octr,oae,oarsd,tcond;
  PetscReal   *rsdold,norm;
  /* Parameters */
  int         init = 5;
  PetscScalar stpfac = 1.5,
              alpha = 1.0,
              beta = 1.1,
              grptol = 1e-8,
              cnvtol = 1e-6;
  int         orttol = 2;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  eps->its = 0;
  eps->nconv = 0;
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv*ncv,&T);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv*ncv,&U);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*ncv,&rsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ncv,&tau);CHKERRQ(ierr);
  lwork = ncv*ncv;
  ierr = PetscMalloc(sizeof(PetscScalar)*lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&itrsd);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&itrsdold);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*ncv,&iwork);CHKERRQ(ierr);
  for (i=0;i<ncv;i++) {
    ierr = SlepcVecSetRandom(eps->V[i]);CHKERRQ(ierr);
    eps->eigr[i] = 0;
    eps->eigi[i] = 0;
    eps->errest[i] = 0;
    itrsd[i] = -1;
  }
  ierr = EPSQRDecomposition(eps,eps->V,0,ncv,PETSC_NULL,0);CHKERRQ(ierr);
  
  while (eps->its<eps->max_it) {

    /* [ nogrp, octr, oae, oarsd ] = dgroup( nconv+1, ncv, wr, wi, rsd, grptol ) */
    EPSdgroup(eps->nconv,ncv,eps->eigr,eps->eigi,eps->errest,grptol,&nogrp,&octr,&oae,&oarsd);

    /* AV(:,idx) = stapply(st,V(:,idx)) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
    }

    /* T(:,idx) = V'*AV(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecMDot(ncv,eps->AV[i],eps->V,T+i*ncv);CHKERRQ(ierr);
    }

    /* [U,H] = hess(T) */
    ilo = eps->nconv + 1;
    LAgehrd_(&ncv,&ilo,&ncv,T,&ncv,tau,work,&lwork,&info);
    for (j=0;j<ncv-1;j++) {
      for (i=j+2;i<ncv;i++) {
        U[i+j*ncv] = T[i+j*ncv];
        T[i+j*ncv] = 0.0;
      }      
    }
    LAorghr_(&ncv,&ilo,&ncv,U,&ncv,tau,work,&lwork,&info);

    /* [T,wr,wi,U] = laqr3(H,U) */
    dlaqr3_(&true,&true,&ncv,&ilo,&ncv,T,&ncv,eps->eigr,eps->eigi,&one,&ncv,U,&ncv,work,&info);
    
    /* AV(:,idx) = AV*U(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecSet(&zero,eps->work[i]);CHKERRQ(ierr);
      ierr = VecMAXPY(ncv,U+ncv*i,eps->work[i],eps->AV);CHKERRQ(ierr);
    }    
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->work[i],eps->AV[i]);CHKERRQ(ierr);
    }    
    
    /* V(:,idx) = V*U(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecSet(&zero,eps->work[i]);CHKERRQ(ierr);
      ierr = VecMAXPY(ncv,U+ncv*i,eps->work[i],eps->V);CHKERRQ(ierr);
    }    
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->work[i],eps->V[i]);CHKERRQ(ierr);
    }    
    
    /* rsdold = rsd */
    for (i=0;i<ncv;i++) { rsdold[i] = eps->errest[i]; }

    /* rsd(idx) = SchurResidualNorms(V,AV,T,nconv,ncv) */
    EPSSchurResidualNorms(eps,eps->V,eps->AV,T,eps->nconv,ncv,ncv,eps->errest);

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv); 
  
#ifdef DEBUG
    printf("[%3d] eigr:  ",eps->its);
    for (i=0;i<5;i++) printf(" %12e",eps->eigr[i]);
    printf("\n");
    printf("[%3d] eigi:  ",eps->its);
    for (i=0;i<5;i++) printf(" %12e",eps->eigi[i]);
    printf("\n");
    printf("[%3d] error: ",eps->its);
    for (i=0;i<5;i++) printf(" %12e",eps->errest[i]);
    printf("\n");
#endif

    /* itrsdold = itrsd;
       for j=idx, itrsd(j)=its; end */
    for (i=0;i<ncv;i++) { itrsdold[i] = itrsd[i]; }
    for (i=eps->nconv;i<ncv;i++) { itrsd[i] = eps->its; }
    
    for (;;) {
      /* [ ngrp, ctr, ae, arsd ] = dgroup( nconv+1, ncv, wr, wi, rsd, grptol ) */
      EPSdgroup(eps->nconv,ncv,eps->eigr,eps->eigi,eps->errest,grptol,&ngrp,&ctr,&ae,&arsd);
#ifdef DEBUG
      printf("[%3d] ngrp=%d ctr=%e ae=%e arsd=%e\n",eps->its,ngrp,ctr,ae,arsd);
#endif

      if (ngrp!=nogrp) break;
      if (ngrp==0) break;
      if (PetscAbsScalar(ae-oae)>ctr*cnvtol*(itrsd[eps->nconv]-itrsdold[eps->nconv])) break;
      if (arsd>ctr*eps->tol) break;
      eps->nconv = eps->nconv + ngrp;
      if (eps->nconv>=ncv) break;
    }
    
    if (eps->nconv>=eps->nev) break;
    
    /* nxtsrr = min([maxit max([fix(stpfac*its) init])]) */
    nxtsrr = PetscMin(eps->max_it,PetscMax(floor(stpfac*eps->its), init));
    
    if (ngrp!=nogrp || ngrp==0 || arsd>=oarsd) {
      idsrr = nxtsrr - eps->its;
    } else {
      /* idsrr = max([1 alpha+beta*(itrsdold(nconv+1)-itrsd(nconv+1))*log(arsd/tol)/log(arsd/oarsd)]) */
      idsrr = floor(alpha+beta*(itrsdold[eps->nconv]-itrsd[eps->nconv])*log(arsd/eps->tol)/log(arsd/oarsd));
      idsrr = PetscMax(1,idsrr);
    }
    nxtsrr = PetscMin(nxtsrr,eps->its+idsrr);

    /* tcond = cond(T,inf) */
    ierr = PetscMemcpy(U,T,sizeof(PetscScalar)*ncv);CHKERRQ(ierr);
    tcond = dcond_(&ncv,U,&ncv,iwork,work,&info);
    
    /* idort = max([1 fix(orttol/max([1 log10(tcond)]))]) */
    idort = PetscMax(1,floor(orttol/PetscMax(1,log10(tcond))));    
    nxtort = PetscMin(eps->its+idort, nxtsrr);
#ifdef DEBUG
    printf("[%3d] nxtsrr=%d idort=%d\n",eps->its,nxtsrr,idort);
#endif

    /* V(:,idx) = AV(:,idx) */
    for (i=eps->nconv;i<ncv;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }
    eps->its++;

    do {
      while (eps->its<nxtort) {
      
        /* AV(:,idx) = stapply(st,V(:,idx)) */
        for (i=eps->nconv;i<ncv;i++) {
          ierr = STApply(eps->OP,eps->V[i],eps->AV[i]);CHKERRQ(ierr);
        }
        
        /* V(:,idx) = AV(:,idx)/norm(AV(:,idx),inf) */
        for (i=eps->nconv;i<ncv;i++) {
          ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
          ierr = VecNorm(eps->V[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          norm = 1 / norm;
          ierr = VecScale(&norm,eps->V[i]);CHKERRQ(ierr);
        }
      
        eps->its++;
      }
      for (i=eps->nconv;i<ncv;i++) {
        /* v = repgs(V(:,1:j-1),V(:,j))
           V(:,j) = v/norm(v) */
        ierr = (*eps->orthog)(eps,i,eps->V,eps->V[i],PETSC_NULL,&norm);CHKERRQ(ierr);
        if (norm < 1e-8) { SETERRQ(1,"Norm is zero"); }
        norm = 1 / norm;
        ierr = VecScale(&norm,eps->V[i]);CHKERRQ(ierr);
      }
      nxtort = PetscMin(eps->its+idort,nxtsrr);
    } while (eps->its<nxtsrr);
  }

  ierr = PetscFree(T);CHKERRQ(ierr);
  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(rsdold);CHKERRQ(ierr);
  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(itrsd);CHKERRQ(ierr);
  ierr = PetscFree(itrsdold);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);

  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_SRRIT"
int EPSCreate_SRRIT(EPS eps)
{
  PetscFunctionBegin;
  eps->ops->setup                = EPSSetUp_SRRIT;
  eps->ops->solve                = EPSSolve_SRRIT;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

