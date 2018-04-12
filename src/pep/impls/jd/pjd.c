/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc polynomial eigensolver: "jd"

   Method: Jacobi-Davidson

   Algorithm:

       Jacobi-Davidson for polynomial eigenvalue problems.
       Based on code contributed by the authors of [2] below.

   References:

       [1] G.L.G. Sleijpen et al., "Jacobi-Davidson type methods for
           generalized eigenproblems and polynomial eigenproblems", BIT
           36(3):595-633, 1996.

       [2] Feng-Nan Hwang, Zih-Hao Wei, Tsung-Ming Huang, Weichung Wang,
           "A Parallel Additive Schwarz Preconditioned Jacobi-Davidson
           Algorithm for Polynomial Eigenvalue Problems in Quantum Dot
           Simulation", J. Comput. Phys. 229(8):2932-2947, 2010.
*/

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscReal   keep;          /* restart parameter */
  PetscReal   fix;           /* fix parameter */
  BV          V;             /* work basis vectors to store the search space */
  BV          W;             /* work basis vectors to store the test space */
  BV          *TV;           /* work basis vectors to store T*V (each TV[i] is the coefficient for \lambda^i of T*V for the extended T) */
  BV          *AX;           /* work basis vectors to store A_i*X for locked eigenvectors */
  BV          N[2];          /* auxiliary work BVs */
  BV          X;             /* locked eigenvectors */
  PetscScalar *T;            /* matrix of the invariant pair */
  PetscScalar *Tj;           /* matrix containing the powers of the invariant pair matrix */
  PetscScalar *XpX;          /* X^H*X */
  PetscInt    ld;            /* leading dimension for Tj and XpX */
  PC          pcshell;       /* preconditioner including basic precond+projector */
  Mat         Pshell;        /* auxiliary shell matrix */
  PetscInt    nlock;         /* number of locked vectors in the invariant pair */
} PEP_JD;

typedef struct {
  PC          pc;            /* basic preconditioner */
  Vec         Bp;            /* preconditioned residual of derivative polynomial, B\p */
  Vec         u;             /* Ritz vector */
  PetscScalar gamma;         /* precomputed scalar u'*B\p */
  PetscScalar *M;
  PetscScalar *ps;
  PetscInt    ld;
  Vec         *work;
  BV          X;
  PetscInt    n;
} PEP_JD_PCSHELL;

typedef struct {
  Mat         P;             /*  */
  PEP         pep;
  Vec         *work;
  PetscScalar theta;
} PEP_JD_MATSHELL;

/*
   Duplicate and resize auxiliary basis
*/
static PetscErrorCode PEPJDDuplicateBasis(PEP pep,BV *basis)
{
  PetscErrorCode     ierr;
  PEP_JD             *pjd = (PEP_JD*)pep->data;
  PetscInt           nloc,m;
  PetscMPIInt        rank,nproc;
  BVType             type;
  BVOrthogType       otype;
  BVOrthogRefineType oref;
  PetscReal          oeta;
  BVOrthogBlockType  oblock;

  PetscFunctionBegin;
  if (pjd->ld>1) {
    ierr = BVCreate(PetscObjectComm((PetscObject)pep),basis);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&nproc);CHKERRQ(ierr);
    ierr = BVGetSizes(pep->V,&nloc,NULL,&m);CHKERRQ(ierr);
    if (rank==nproc-1) nloc += pjd->ld-1;
    ierr = BVSetSizes(*basis,nloc,PETSC_DECIDE,m);CHKERRQ(ierr);
    ierr = BVGetType(pep->V,&type);CHKERRQ(ierr);
    ierr = BVSetType(*basis,type);CHKERRQ(ierr);
    ierr = BVGetOrthogonalization(pep->V,&otype,&oref,&oeta,&oblock);CHKERRQ(ierr);
    ierr = BVSetOrthogonalization(*basis,otype,oref,oeta,oblock);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)*basis);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicate(pep->V,basis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetUp_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      isprecond,flg;
  PetscInt       i;

  PetscFunctionBegin;
  pep->lineariz = PETSC_FALSE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
  if (!pep->which) pep->which = PEP_TARGET_MAGNITUDE;
  if (pep->which != PEP_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"PEPJD only supports which=target_magnitude");;

  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STPRECOND,&isprecond);CHKERRQ(ierr);
  if (!isprecond) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"JD only works with PRECOND spectral transformation");

  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag unset, see STSetTransform()");

  if (!pjd->keep) pjd->keep = 0.5;
  ierr = PEPBasisCoefficients(pep,pep->pbc);CHKERRQ(ierr);
  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,5);CHKERRQ(ierr);
  pjd->ld = pep->nev;
#if !defined (PETSC_USE_COMPLEX)
  pjd->ld++;
#endif
  ierr = PetscMalloc2(pep->nmat,&pjd->TV,pep->nmat,&pjd->AX);CHKERRQ(ierr);
  for (i=0;i<pep->nmat;i++) {
    ierr = PEPJDDuplicateBasis(pep,pjd->TV+i);CHKERRQ(ierr);
  }
  ierr = PEPJDDuplicateBasis(pep,&pjd->W);CHKERRQ(ierr);
  if (pjd->ld>1) {
    ierr = PEPJDDuplicateBasis(pep,&pjd->V);CHKERRQ(ierr);
    ierr = BVSetFromOptions(pjd->V);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = BVDuplicateResize(pep->V,pjd->ld-1,pjd->AX+i);CHKERRQ(ierr);
    }
    ierr = BVDuplicateResize(pep->V,pjd->ld-1,pjd->N);CHKERRQ(ierr);
    ierr = BVDuplicateResize(pep->V,pjd->ld-1,pjd->N+1);CHKERRQ(ierr);
    pjd->X = pep->V;
    ierr = PetscCalloc3((pjd->ld)*(pjd->ld),&pjd->XpX,pep->ncv*pep->ncv,&pjd->T,pjd->ld*pjd->ld*pep->nmat,&pjd->Tj);CHKERRQ(ierr);
  } else pjd->V = pep->V;
  ierr = DSSetType(pep->ds,DSPEP);CHKERRQ(ierr);
  ierr = DSPEPSetDegree(pep->ds,pep->nmat-1);CHKERRQ(ierr);
  if (pep->basis!=PEP_BASIS_MONOMIAL) {
    ierr = DSPEPSetCoefficients(pep->ds,pep->pbc);CHKERRQ(ierr);
  }
  ierr = DSAllocate(pep->ds,pep->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Updates columns (low to (high-1)) of TV[i]
*/
static PetscErrorCode PEPJDUpdateTV(PEP pep,PetscInt low,PetscInt high,Vec *w)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscInt       pp,col,i,nloc,nconv;
  Vec            v1,v2,t1,t2;
  PetscScalar    *array1,*array2,*x2,*xx,*N,*Np,*y2=NULL,zero=0.0,sone=1.0,*pT,fact,*psc;
  PetscReal      *cg,*ca,*cb;
  PetscMPIInt    rk,np,count;
  PetscBLASInt   n_,ld_,one=1;
  Mat            T;
  BV             pbv;

  PetscFunctionBegin;
  ca = pep->pbc; cb = ca+pep->nmat; cg = cb + pep->nmat;
  nconv = pjd->nlock;
  ierr = PetscMalloc5(nconv,&x2,nconv,&xx,nconv*nconv,&pT,nconv*nconv,&N,nconv*nconv,&Np);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
  ierr = BVGetSizes(pep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
  t1 = w[0];
  t2 = w[1];
  ierr = PetscBLASIntCast(pjd->nlock,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(pjd->ld,&ld_);CHKERRQ(ierr);
  if (nconv){
    for (i=0;i<nconv;i++) {
      ierr = PetscMemcpy(pT+i*nconv,pjd->T+i*pep->ncv,nconv*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,nconv,nconv,pT,&T);CHKERRQ(ierr);
  }
  for (col=low;col<high;col++) {
    ierr = BVGetColumn(pjd->V,col,&v1);CHKERRQ(ierr);
    ierr = VecGetArray(v1,&array1);CHKERRQ(ierr);
    if (nconv>0) {
      if (rk==np-1) { for (i=0;i<nconv;i++) x2[i] = array1[nloc+i]; }
      ierr = PetscMPIIntCast(nconv,&count);CHKERRQ(ierr);
      ierr = MPI_Bcast(x2,nconv,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pep));CHKERRQ(ierr);
    }
    ierr = VecPlaceArray(t1,array1);CHKERRQ(ierr);
    if (nconv) {
      ierr = BVSetActiveColumns(pjd->N[0],0,nconv);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(pjd->N[1],0,nconv);CHKERRQ(ierr);
      ierr = BVDotVec(pjd->X,t1,xx);CHKERRQ(ierr);
    }
    for (pp=pep->nmat-1;pp>=0;pp--) {
      ierr = BVGetColumn(pjd->TV[pp],col,&v2);CHKERRQ(ierr);
      ierr = VecGetArray(v2,&array2);CHKERRQ(ierr);
      ierr = VecPlaceArray(t2,array2);CHKERRQ(ierr);
      ierr = MatMult(pep->A[pp],t1,t2);CHKERRQ(ierr);
      if (nconv) {
        if (rk==np-1 && pp<pep->nmat-1) {
          y2 = array2+nloc;
          ierr = PetscMemcpy(y2,xx,nconv*sizeof(PetscScalar));CHKERRQ(ierr);
          PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n_,pjd->Tj+pjd->ld*pjd->ld*pp,&ld_,y2,&one));
        }
        if (pp<pep->nmat-3) {
          ierr = BVMult(pjd->N[0],1.0,-cg[pp+2],pjd->AX[pp+1],NULL);CHKERRQ(ierr);
          ierr = MatShift(T,-cb[pp+1]);CHKERRQ(ierr);
          ierr = BVMult(pjd->N[0],1.0/ca[pp],1.0/ca[pp],pjd->N[1],T);CHKERRQ(ierr);
          pbv = pjd->N[0]; pjd->N[0] = pjd->N[1]; pjd->N[1] = pbv;
          ierr = BVMultVec(pjd->N[1],1.0,1.0,t2,x2);CHKERRQ(ierr);
          if (rk==np-1) {
            fact = -cg[pp+2];
            PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,pjd->Tj+(pp+1)*pjd->ld*pjd->ld,&ld_,pjd->XpX,&ld_,&fact,Np,&n_));
            fact = 1/ca[pp];
            PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&fact,N,&n_,pT,&n_,&fact,Np,&n_));
            psc = Np; Np = N; N = psc;
            PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,N,&n_,x2,&one,&sone,y2,&one));
          }
          ierr = MatShift(T,cb[pp+1]);CHKERRQ(ierr);
        } else if (pp==pep->nmat-3) {
          ierr = BVCopy(pjd->AX[pp+2],pjd->N[0]);CHKERRQ(ierr);
          ierr = BVScale(pjd->N[0],1/ca[pp+1]);CHKERRQ(ierr);
          ierr = BVCopy(pjd->AX[pp+1],pjd->N[1]);CHKERRQ(ierr);
          ierr = MatShift(T,-cb[pp+1]);CHKERRQ(ierr);
          ierr = BVMult(pjd->N[1],1.0/ca[pp],1.0/ca[pp],pjd->N[0],T);CHKERRQ(ierr);
          ierr = BVMultVec(pjd->N[1],1.0,1.0,t2,x2);CHKERRQ(ierr);
          if (rk==np-1) {
            fact = 1/ca[pp];
            PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&fact,pjd->Tj+(pp+1)*pjd->ld*pjd->ld,&ld_,pjd->XpX,&ld_,&zero,N,&n_));
            PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,N,&n_,x2,&one,&sone,y2,&one));
          }
          ierr = MatShift(T,cb[pp+1]);CHKERRQ(ierr);
        } else if (pp==pep->nmat-2) {
          ierr = BVMultVec(pjd->AX[pp+1],1.0/ca[pp],1.0,t2,x2);CHKERRQ(ierr);
          if (rk==np-1) {
            ierr = PetscMemzero(Np,nconv*nconv*sizeof(PetscScalar));CHKERRQ(ierr);
          }
        }
      }
      ierr = VecResetArray(t2);CHKERRQ(ierr);
      ierr = VecRestoreArray(v2,&array2);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->TV[pp],col,&v2);CHKERRQ(ierr);
    }
    ierr = VecResetArray(t1);CHKERRQ(ierr);
    ierr = VecRestoreArray(v1,&array1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,col,&v1);CHKERRQ(ierr);
  }
  if (nconv) {ierr = MatDestroy(&T);CHKERRQ(ierr);}
  ierr = PetscFree5(x2,xx,pT,N,Np);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   RRQR of X. Xin*P=Xou*R. Rank of R is rk
*/
static PetscErrorCode PEPJDOrthogonalize(PetscInt row,PetscInt col,PetscScalar *X,PetscInt ldx,PetscInt *rk,PetscInt *P,PetscScalar *R,PetscInt ldr)
{
#if defined(SLEPC_MISSING_LAPACK_GEQP3) || defined(PETSC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQP3/QRGQR - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,n,r;
  PetscBLASInt   row_,col_,ldx_,*p,lwork,info,n_;
  PetscScalar    *tau,*work;
  PetscReal      tol,*rwork;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(row,&row_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(col,&col_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldx,&ldx_);CHKERRQ(ierr);
  n = PetscMin(row,col);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  lwork = 3*col_+1;
  ierr = PetscMalloc4(col,&p,n,&tau,lwork,&work,2*col,&rwork);CHKERRQ(ierr);
  for (i=1;i<col;i++) p[i] = 0;
  p[0] = 1;

  /* rank revealing QR */
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgeqp3",LAPACKgeqp3_(&row_,&col_,X,&ldx_,p,tau,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgeqp3",LAPACKgeqp3_(&row_,&col_,X,&ldx_,p,tau,work,&lwork,&info));
#endif
  SlepcCheckLapackInfo("geqp3",info);
  if (P) for (i=0;i<col;i++) P[i] = p[i];

  /* rank computation */
  tol = PetscMax(row,col)*PETSC_MACHINE_EPSILON*PetscAbsScalar(X[0]);
  r = 1;
  for (i=1;i<n;i++) {
    if (PetscAbsScalar(X[i+ldx*i])>tol) r++;
    else break;
  }
  if (rk) *rk=r;

  /* copy upper triangular matrix if requested */
  if (R) {
     for (i=0;i<r;i++) {
       ierr = PetscMemzero(R+i*ldr,r*sizeof(PetscScalar));CHKERRQ(ierr);
       for (j=0;j<=i;j++) R[i*ldr+j] = X[i*ldx+j];
     }
  }
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&row_,&n_,&n_,X,&ldx_,tau,work,&lwork,&info));
  SlepcCheckLapackInfo("orgqr",info);
  ierr = PetscFree4(p,tau,work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

/*
   Application of extended preconditioner
*/
static PetscErrorCode PEPJDExtendedPCApply(PC pc,Vec x,Vec y)
{
  PetscInt          i,j,nloc,n,ld;
  PetscMPIInt       rk,np,count;
  Vec               tx,ty;
  PEP_JD_PCSHELL    *ctx;
  PetscErrorCode    ierr;
  const PetscScalar *array1;
  PetscScalar       *x2=NULL,*t=NULL,*ps,*array2;
  PetscBLASInt      one=1.0,ld_,n_;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);
  n  = ctx->n;
  ps = ctx->ps;
  ld = ctx->ld;
  if (n) {
    ierr = PetscMalloc2(n,&x2,n,&t);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&np);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = VecGetLocalSize(ctx->work[0],&nloc);CHKERRQ(ierr);
      ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
      for (i=0;i<n;i++) x2[i] = array1[nloc+i];
      ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(n,&count);CHKERRQ(ierr);
    ierr = MPI_Bcast(x2,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  }

  /* y = B\x apply PC */
  tx = ctx->work[0];
  ty = ctx->work[1];
  ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tx,array1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(ty,array2);CHKERRQ(ierr);
  ierr = PCApply(ctx->pc,tx,ty);CHKERRQ(ierr);
  if (n) {
    for (j=0;j<n;j++) {
      t[j] = 0.0;
      for (i=0;i<n;i++) t[j] += ctx->M[i+j*ld]*x2[i];
    }
    if (rk==np-1) for (i=0;i<n;i++) array2[nloc+i] = t[i];
    ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
    PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n_,ps,&ld_,t,&one));
    ierr = BVMultVec(ctx->X,-1.0,1.0,ty,t);CHKERRQ(ierr);
    ierr = PetscFree2(x2,t);CHKERRQ(ierr);
  }
  ierr = VecResetArray(tx);CHKERRQ(ierr);
  ierr = VecResetArray(ty);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&array2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Application of shell preconditioner:
      y = B\x - eta*B\p,  with eta = (u'*B\x)/(u'*B\p)
*/
static PetscErrorCode PCShellApply_PEPJD(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscScalar    eta;
  PEP_JD_PCSHELL *ctx;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);

  /* y = B\x apply extended PC */
  ierr = PEPJDExtendedPCApply(pc,x,y);CHKERRQ(ierr);

  /* Compute eta = u'*y / u'*Bp */
  ierr = VecDot(y,ctx->u,&eta);CHKERRQ(ierr);
  eta /= ctx->gamma;

  /* y = y - eta*Bp */
  ierr = VecAXPY(y,-eta,ctx->Bp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDCopyToExtendedVec(PEP pep,Vec v,PetscScalar *a,PetscInt na,PetscInt off,Vec vex,PetscBool back)
{
  PetscErrorCode ierr;
  PetscMPIInt    np,rk,count;
  PetscScalar    *array1,*array2;
  PetscInt       nloc;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
  ierr = BVGetSizes(pep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
  if (v) {
    ierr = VecGetArray(v,&array1);CHKERRQ(ierr);
    ierr = VecGetArray(vex,&array2);CHKERRQ(ierr);
    if (back) {
      ierr = PetscMemcpy(array1,array2,nloc*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemcpy(array2,array1,nloc*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(v,&array1);CHKERRQ(ierr);
    ierr = VecRestoreArray(vex,&array2);CHKERRQ(ierr);
  }
  if (a) {
    if (rk==np-1) {
      ierr = VecGetArray(vex,&array2);CHKERRQ(ierr);
      if (back) {
        ierr = PetscMemcpy(a,array2+nloc+off,na*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(array2+nloc+off,a,na*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(vex,&array2);CHKERRQ(ierr);
    }
    if (back) {
      ierr = PetscMPIIntCast(na,&count);CHKERRQ(ierr);
      ierr = MPI_Bcast(a,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pep));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Computes Phi^hat(lambda) times a vector or its derivative (depends on beval)
     if no vector is provided returns a matrix
 */
static PetscErrorCode PEPJDEvaluateHatBasis(PEP pep,PetscInt n,PetscScalar *H,PetscInt ldh,PetscScalar *beval,PetscScalar *t,PetscInt idx,PetscScalar *qpp,PetscScalar *qp,PetscScalar *q)
{
  PetscErrorCode ierr;
  PetscInt       j,i;
  PetscBLASInt   n_,ldh_,one=1;
  PetscReal      *a,*b,*g;
  PetscScalar    sone=1.0;

  PetscFunctionBegin;
  a = pep->pbc; b=a+pep->nmat; g=b+pep->nmat;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  if (idx<1) {
    ierr = PetscMemzero(q,(t?n:n*n)*sizeof(PetscScalar));CHKERRQ(ierr);
  } else if (idx==1) {
    if (t) {for (j=0;j<n;j++) q[j] = t[j]*beval[idx-1]/a[0];}
    else {
      ierr = PetscMemzero(q,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=0;j<n;j++) q[(j+1)*n] = beval[idx-1]/a[0];
    }
  } else {
    if (t) {
      ierr = PetscMemcpy(q,qp,n*sizeof(PetscScalar));CHKERRQ(ierr);
      PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n_,H,&ldh_,q,&one));
      for (j=0;j<n;j++) {
        q[j] += beval[idx-1]*t[j]-b[idx-1]*qp[j]-g[idx-1]*qpp[j];
        q[j] /= a[idx-1]; 
      }
    } else {
      ierr = PetscMemcpy(q,qp,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
      PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&n_,&n_,&sone,H,&ldh_,q,&n_));
      for (j=0;j<n;j++) {
        q[j+n*j] += beval[idx-1];
        for (i=0;i<n;i++) {
          q[i+n*j] += -b[idx-1]*qp[j*n+i]-g[idx-1]*qpp[j*n+i];
          q[i+n*j] /= a[idx-1]; 
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDComputeResidual(PEP pep,PetscBool derivative,Vec u,PetscScalar theta,Vec p,Vec *work)
{
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscErrorCode ierr;
  PetscMPIInt    rk,np,count;
  Vec            tu,tp,w;
  PetscScalar    *dval,*array1,*array2,*x2=NULL,*y2,*qj=NULL,*tt=NULL,*xx=NULL,sone=1.0;
  PetscInt       i,j,nconv,nloc,deg=pep->nmat-1;
  PetscBLASInt   n,ld,one=1;

  PetscFunctionBegin;
  nconv = pjd->nlock;
  if (!nconv) {
    ierr = PetscMalloc1(pep->nmat,&dval);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc5(pep->nmat,&dval,nconv,&xx,nconv,&tt,nconv,&x2,nconv*pep->nmat,&qj);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = BVGetSizes(pep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
      ierr = VecGetArray(u,&array1);CHKERRQ(ierr);
      for (i=0;i<nconv;i++) x2[i] = array1[nloc+i];
      ierr = VecRestoreArray(u,&array1);CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(nconv,&count);CHKERRQ(ierr);
    ierr = MPI_Bcast(x2,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pep));CHKERRQ(ierr);
  }
  tu = work[0];
  tp = work[1];
  w  = work[2];
  ierr = VecGetArray(u,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tu,array1);CHKERRQ(ierr);
  ierr = VecGetArray(p,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(tp,array2);CHKERRQ(ierr);
  ierr = VecSet(tp,0.0);CHKERRQ(ierr);
  if (derivative) {
    ierr = PEPEvaluateBasisDerivative(pep,theta,0.0,dval,NULL);CHKERRQ(ierr);
  } else {
    ierr = PEPEvaluateBasis(pep,theta,0.0,dval,NULL);CHKERRQ(ierr);
  }
  for (i=derivative?1:0;i<pep->nmat;i++) {
    ierr = MatMult(pep->A[i],tu,w);CHKERRQ(ierr);
    ierr = VecAXPY(tp,dval[i],w);CHKERRQ(ierr);
  }
  if (nconv) {
    for (i=0;i<pep->nmat;i++) {
      ierr = PEPJDEvaluateHatBasis(pep,nconv,pjd->T,pep->ncv,dval,x2,i,i>1?qj+(i-2)*nconv:NULL,i>0?qj+(i-1)*nconv:NULL,qj+i*nconv);CHKERRQ(ierr);
    }
    for (i=derivative?2:1;i<pep->nmat;i++) {
      ierr = BVMultVec(pjd->AX[i],1.0,1.0,tp,qj+i*nconv);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(pjd->X,0,nconv);CHKERRQ(ierr);
    ierr = BVDotVec(pjd->X,tu,xx);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = PetscBLASIntCast(nconv,&n);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(pjd->ld,&ld);CHKERRQ(ierr);
      y2 = array2+nloc;
      ierr = PetscMemzero(y2,nconv*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=derivative?1:0;j<deg;j++) {
        ierr = PetscMemcpy(tt,xx,nconv*sizeof(PetscScalar));CHKERRQ(ierr);
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,pjd->XpX,&ld,qj+j*nconv,&one,&dval[j],tt,&one));
        PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&sone,pjd->Tj+j*ld*ld,&ld,tt,&one,&sone,y2,&one)); 
      }
    }
    ierr = PetscFree5(dval,xx,tt,x2,qj);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(dval);CHKERRQ(ierr);
  }
  ierr = VecResetArray(tu);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&array1);CHKERRQ(ierr);
  ierr = VecResetArray(tp);CHKERRQ(ierr);
  ierr = VecRestoreArray(p,&array2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDProcessInitialSpace(PEP pep,Vec *w)
{
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscErrorCode ierr;
  PetscScalar    *tt;
  Vec            vg,wg;
  PetscInt       i;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = PetscMalloc1(pjd->ld-1,&tt);CHKERRQ(ierr);
  if (pep->nini==0) {
    ierr = BVSetRandomColumn(pjd->V,0);CHKERRQ(ierr);
    for (i=0;i<pjd->ld-1;i++) tt[i] = 0.0;
    ierr = BVGetColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = PEPJDCopyToExtendedVec(pep,NULL,tt,pjd->ld-1,0,vg,PETSC_FALSE);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVNormColumn(pjd->V,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->V,0,1.0/norm);CHKERRQ(ierr);
    ierr = BVGetColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVGetColumn(pjd->W,0,&wg);CHKERRQ(ierr);
    ierr = VecSet(wg,0.0);CHKERRQ(ierr);
    /* W = P(target)*V */
    ierr = PEPJDComputeResidual(pep,PETSC_FALSE,vg,pep->target,wg,w);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->W,0,&wg);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVNormColumn(pjd->W,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->W,0,1.0/norm);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Support for initial vectors not implemented yet");
  ierr = PetscFree(tt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDShellMatMult(Mat P,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_JD_MATSHELL   *matctx;
  PEP_JD            *pjd;
  PetscMPIInt       rk,np,count;
  PetscInt          i,j,nconv,nloc,nmat,ldt,deg,ncv;
  Vec               tx,ty;
  PetscScalar       *array2,*x2=NULL,*y2,*tt=NULL,*xx=NULL,theta,sone=1.0,*qj,*val;
  PetscBLASInt      n,ld,one=1;
  const PetscScalar *array1;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(P,(void**)&matctx);CHKERRQ(ierr);
  pjd   = (PEP_JD*)(matctx->pep->data);
  nconv = pjd->nlock;
  theta = matctx->theta;
  nmat  = matctx->pep->nmat;
  ncv   = matctx->pep->ncv;
  deg   = nmat-1;
  ldt   = pjd->ld;
  if (nconv>0) {
    ierr = PetscMalloc5(nconv,&tt,nconv,&x2,nconv*nmat,&qj,nconv,&xx,nmat,&val);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)P),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)P),&np);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = BVGetSizes(matctx->pep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
      ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
      for (i=0;i<nconv;i++) x2[i] = array1[nloc+i];
      ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(nconv,&count);CHKERRQ(ierr);
    ierr = MPI_Bcast(x2,nconv,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)P));CHKERRQ(ierr);
  }
  tx = matctx->work[0];
  ty = matctx->work[1];
  ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tx,array1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(ty,array2);CHKERRQ(ierr);
  ierr = VecSet(ty,0.0);CHKERRQ(ierr);
  ierr = MatMult(matctx->P,tx,ty);CHKERRQ(ierr);
  if (nconv) {
    ierr = PEPEvaluateBasis(matctx->pep,theta,0.0,val,NULL);CHKERRQ(ierr);
    for (i=0;i<nmat;i++) {
      ierr = PEPJDEvaluateHatBasis(matctx->pep,nconv,pjd->T,ncv,val,x2,i,i>1?qj+(i-2)*nconv:NULL,i>0?qj+(i-1)*nconv:NULL,qj+i*nconv);CHKERRQ(ierr);
    }
    for (i=1;i<nmat;i++) {
      ierr = BVMultVec(pjd->AX[i],1.0,1.0,ty,qj+i*nconv);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(pjd->X,0,nconv);CHKERRQ(ierr);
    ierr = BVDotVec(pjd->X,tx,xx);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = PetscBLASIntCast(pjd->nlock,&n);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(ldt,&ld);CHKERRQ(ierr);
      y2 = array2+nloc;
      ierr = PetscMemzero(y2,nconv*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=0;j<deg;j++) {
        ierr = PetscMemcpy(tt,xx,nconv*sizeof(PetscScalar));CHKERRQ(ierr);
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,pjd->XpX,&ld,qj+j*nconv,&one,&val[j],tt,&one));
        PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,pjd->Tj+ld*ld*j,&ld,tt,&one));
        for (i=0;i<nconv;i++) y2[i] += tt[i];
      }
    }
    ierr = PetscFree5(tt,x2,qj,xx,val);CHKERRQ(ierr);
  }
  ierr = VecResetArray(tx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecResetArray(ty);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&array2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDMatSetUp(PEP pep,PetscScalar theta)
{
  PetscErrorCode  ierr;
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PEP_JD_MATSHELL *matctx;
  MatStructure    str;
  PetscScalar     *vals;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(pjd->Pshell,(void**)&matctx);CHKERRQ(ierr);
  if (matctx->P && matctx->theta==theta)
    PetscFunctionReturn(0);
  ierr = PetscMalloc1(pep->nmat,&vals);CHKERRQ(ierr);
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  if (!matctx->P) {
    ierr = MatDuplicate(pep->A[0],MAT_COPY_VALUES,&matctx->P);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(pep->A[0],matctx->P,str);CHKERRQ(ierr);
  }
  ierr = PEPEvaluateBasis(pep,theta,0.0,vals,NULL);CHKERRQ(ierr);
  for (i=1;i<pep->nmat;i++) {
    ierr = MatAXPY(matctx->P,vals[i],pep->A[i],str);CHKERRQ(ierr);
  }
  matctx->theta = theta;
  ierr = PetscFree(vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDCreateShellPC(PEP pep,Vec *ww)
{
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PEP_JD_PCSHELL  *pcctx;
  PEP_JD_MATSHELL *matctx;
  KSP             ksp;
  PetscInt        nloc,mloc;
  PetscMPIInt     np,rk;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PCCreate(PetscObjectComm((PetscObject)pep),&pjd->pcshell);CHKERRQ(ierr);
  ierr = PCSetType(pjd->pcshell,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetName(pjd->pcshell,"PCPEPJD");CHKERRQ(ierr);
  ierr = PCShellSetApply(pjd->pcshell,PCShellApply_PEPJD);CHKERRQ(ierr);
  ierr = PetscNew(&pcctx);CHKERRQ(ierr);
  ierr = PCShellSetContext(pjd->pcshell,pcctx);CHKERRQ(ierr);
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
  ierr = BVCreateVec(pjd->V,&pcctx->Bp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pcctx->pc);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)pcctx->pc);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pep->A[0],&mloc,&nloc);CHKERRQ(ierr);
  if (pjd->ld>1) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
    if (rk==np-1) { nloc += pjd->ld-1; mloc += pjd->ld-1; }
  }
  ierr = PetscNew(&matctx);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)pep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,&pjd->Pshell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(pjd->Pshell,MATOP_MULT,(void(*)())PEPJDShellMatMult);CHKERRQ(ierr);
  matctx->pep = pep;
  ierr = PEPJDMatSetUp(pep,pep->target);CHKERRQ(ierr);
  ierr = PCSetOperators(pcctx->pc,matctx->P,matctx->P);CHKERRQ(ierr);
  ierr = PCSetErrorIfFailure(pcctx->pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PCSetReusePreconditioner(pcctx->pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pjd->pcshell);CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,pjd->Pshell,pjd->Pshell);CHKERRQ(ierr);
  if (pjd->ld>1) {
    ierr = PetscMalloc2(pjd->ld*pjd->ld,&pcctx->M,pjd->ld*pjd->ld,&pcctx->ps);CHKERRQ(ierr);
    pcctx->X  = pjd->X;
    pcctx->ld = pjd->ld;
  }
  matctx->work = ww;
  pcctx->work  = ww;
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPJDUpdateExtendedPC(PEP pep,PetscScalar theta)
{
#if defined(PETSC_MISSING_LAPACK_GESVD) || defined(PETSC_MISSING_LAPACK_GETRI) || defined(PETSC_MISSING_LAPACK_GETRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD/GETRI/GETRF - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PEP_JD_PCSHELL *pcctx;
  PetscInt       i,j,k,n=pjd->nlock,ld=pjd->ld,deg=pep->nmat-1;
  PetscScalar    *M,*ps,*work,*U,*V,*S,*Sp,*Spp,snone=-1.0,sone=1.0,zero=0.0,*val;
  PetscReal      tol,maxeig=0.0,*sg,*rwork;
  PetscBLASInt   n_,info,ld_,*p,lw_,rk=0;

  PetscFunctionBegin;
  if (n) {
    ierr = PCShellGetContext(pjd->pcshell,(void**)&pcctx);CHKERRQ(ierr);
    pcctx->n = n;
    M  = pcctx->M;
    ps = pcctx->ps;
    ierr = PetscCalloc7(2*n*n,&U,3*n*n,&S,n,&sg,10*n,&work,5*n,&rwork,n,&p,deg+1,&val);CHKERRQ(ierr);
    V = U+n*n;
    /* pseudo-inverse */
    for (j=0;j<n;j++) {
      for (i=0;i<j;i++) S[n*j+i] = -pjd->T[pep->ncv*j+i];
      S[n*j+j] = theta-pjd->T[pep->ncv*j+j];
    }
    ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
    lw_ = 10*n_;
#if !defined (PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&n_,&n_,S,&n_,sg,U,&n_,V,&n_,work,&lw_,&info));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&n_,&n_,S,&n_,sg,U,&n_,V,&n_,work,&lw_,rwork,&info));
#endif
    SlepcCheckLapackInfo("gesvd",info);
    for (i=0;i<n;i++) maxeig = PetscMax(maxeig,sg[i]);
    tol = 10*PETSC_MACHINE_EPSILON*n*maxeig;
    for (j=0;j<n;j++) {
      if (sg[j]>tol) {
        for (i=0;i<n;i++) U[j*n+i] /= sg[j];
        rk++;
      } else break;
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&rk,&sone,U,&n_,V,&n_,&zero,ps,&ld_));

    /* compute M */
    ierr = PEPEvaluateBasis(pep,theta,0.0,val,NULL);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&snone,pjd->XpX,&ld_,ps,&ld_,&zero,M,&ld_));
    ierr = PetscMemzero(S,2*n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    Sp = S+n*n;
    for (j=0;j<n;j++) S[j*(n+1)] = 1.0; 
    for (k=1;k<deg;k++) {
      for (j=0;j<n;j++) for (i=0;i<n;i++) V[j*n+i] = S[j*n+i] - ps[j*ld+i]*val[k];
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,pjd->XpX,&ld_,V,&n_,&zero,U,&n_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,pjd->Tj+k*ld*ld,&ld_,U,&n_,&sone,M,&ld_));
      Spp = Sp; Sp = S;
      ierr = PEPJDEvaluateHatBasis(pep,n,pjd->T,pep->ncv,val,NULL,k+1,Spp,Sp,S);CHKERRQ(ierr);
    }
    /* inverse */
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n_,&n_,M,&ld_,p,&info));
    SlepcCheckLapackInfo("getrf",info);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n_,M,&ld_,p,work,&n_,&info));
    SlepcCheckLapackInfo("getri",info);
    ierr = PetscFree7(U,S,sg,work,rwork,p,val);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#endif
}

static PetscErrorCode PEPJDEigenvectors(PEP pep)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBLASInt   ld,nconv,info,nc;
  PetscScalar    *Z,*w;
  PetscReal      *wr,norm;
  PetscInt       i;
  Mat            U;

  PetscFunctionBegin;
  ierr = PetscMalloc3(pep->nconv*pep->nconv,&Z,3*pep->ncv,&wr,2*pep->ncv,&w);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(pep->ncv,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(pep->nconv,&nconv);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R","A",NULL,&nconv,pjd->T,&ld,NULL,&nconv,Z,&nconv,&nconv,&nc,wr,&info));
#else
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R","A",NULL,&nconv,pjd->T,&ld,NULL,&nconv,Z,&nconv,&nconv,&nc,w,wr,&info));
#endif
  SlepcCheckLapackInfo("trevc",info);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nconv,nconv,Z,&U);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pjd->X,0,pep->nconv);CHKERRQ(ierr);
  ierr = BVMultInPlace(pjd->X,U,0,pep->nconv);CHKERRQ(ierr);
  for (i=0;i<pep->nconv;i++) {
    ierr = BVNormColumn(pjd->X,i,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->X,i,1.0/norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  ierr = PetscFree3(Z,wr,w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

static PetscErrorCode PEPJDLockConverged(PEP pep,PetscInt *nv,PetscInt sz)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscInt       j,i,ldds,rk=0,nvv=*nv;
  Vec            v,x,w;
  PetscScalar    *R,*pX;
  Mat            X;

  PetscFunctionBegin;
  /* update AX and XpX */
  for (i=sz;i>0;i--) {
    ierr = BVGetColumn(pjd->X,pjd->nlock-i,&x);CHKERRQ(ierr);
    for (j=0;j<pep->nmat;j++) {
      ierr = BVGetColumn(pjd->AX[j],pjd->nlock-i,&v);CHKERRQ(ierr);
      ierr = MatMult(pep->A[j],x,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->AX[j],pjd->nlock-i,&v);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(pjd->AX[j],0,pjd->nlock-i+1);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(pjd->X,pjd->nlock-i,&x);CHKERRQ(ierr);
    ierr = BVDotColumn(pjd->X,(pjd->nlock-i),pjd->XpX+(pjd->nlock-i)*(pep->nev));CHKERRQ(ierr);
    pjd->XpX[(pjd->nlock-i)*(1+pep->nev)] = 1.0;
    for (j=0;j<pjd->nlock-i;j++) pjd->XpX[j*(pep->nev)+pjd->nlock-i] = PetscConj(pjd->XpX[(pjd->nlock-i)*(pep->nev)+j]);
  }

  /* evaluate the polynomial basis in T */
  ierr = PetscMemzero(pjd->Tj,pep->nev*pep->nev*pep->nmat*sizeof(PetscScalar));CHKERRQ(ierr);
  for (j=0;j<pep->nmat;j++) {
    ierr = PEPEvaluateBasisMat(pep,pjd->nlock,pjd->T,pep->ncv,j,(j>1)?pjd->Tj+(j-2)*pep->nev*pep->nev:NULL,pep->nev,j?pjd->Tj+(j-1)*pep->nev*pep->nev:NULL,pep->nev,pjd->Tj+j*pep->nev*pep->nev,pep->nev);CHKERRQ(ierr);
  }

  /* Extend search space */
  ierr = PetscCalloc1(nvv*nvv,&R);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);
  ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  ierr = PEPJDOrthogonalize(nvv,nvv,pX+ldds,ldds,&rk,NULL,NULL,0);CHKERRQ(ierr);
  rk -= sz;
  for (j=0;j<rk;j++) {
    ierr = PetscMemcpy(R+j*nvv,pX+(j+sz)*ldds,nvv*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nvv,rk,R,&X);CHKERRQ(ierr);
  ierr = BVMultInPlace(pjd->V,X,0,rk);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pjd->V,0,rk);CHKERRQ(ierr);
  for (j=0;j<rk;j++) {
    /* W = P(target)*V */
    ierr = BVGetColumn(pjd->W,j,&w);CHKERRQ(ierr);
    ierr = BVGetColumn(pjd->V,j,&v);CHKERRQ(ierr);
    ierr = PEPJDComputeResidual(pep,PETSC_FALSE,v,pep->target,w,pep->work);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,j,&v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->W,j,&w);CHKERRQ(ierr);
  }
  ierr = BVSetActiveColumns(pjd->W,0,rk);CHKERRQ(ierr);
  ierr = BVOrthogonalize(pjd->W,NULL);CHKERRQ(ierr);
  *nv = rk;
  ierr = PetscFree(R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSolve_JD(PEP pep)
{
  PetscErrorCode  ierr;
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PetscInt        k,nv,ld,minv,dim,bupdated=0,sz=1,idx;
  PetscScalar     theta=0.0,*pX,*eig,ritz;
  PetscReal       norm,*res;
  PetscBool       lindep;
  Vec             t,u,p,r,*ww=pep->work,v;
  Mat             G,X,Y;
  KSP             ksp;
  PEP_JD_PCSHELL  *pcctx;
  PEP_JD_MATSHELL *matctx;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc2(pep->ncv,&eig,pep->ncv,&res);CHKERRQ(ierr);
  ierr = BVCreateVec(pjd->V,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&p);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);
  pjd->nlock = 0;
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
  ierr = PEPJDProcessInitialSpace(pep,ww);CHKERRQ(ierr);
  nv = (pep->nini)?pep->nini:1;
  ierr = BVCopyVec(pjd->V,0,u);CHKERRQ(ierr);
  
  /* Replace preconditioner with one containing projectors */
  ierr = PEPJDCreateShellPC(pep,ww);CHKERRQ(ierr);
  ierr = PCShellGetContext(pjd->pcshell,(void**)&pcctx);CHKERRQ(ierr);

  /* Restart loop */
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;
    ierr = DSSetDimensions(pep->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->V,bupdated,nv);CHKERRQ(ierr);
    ierr = PEPJDUpdateTV(pep,bupdated,nv,ww);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->W,bupdated,nv);CHKERRQ(ierr);
    for (k=0;k<pep->nmat;k++) {
      ierr = BVSetActiveColumns(pjd->TV[k],bupdated,nv);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
      ierr = BVMatProject(pjd->TV[k],NULL,pjd->W,G);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(pjd->V,0,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->W,0,nv);CHKERRQ(ierr);

    /* Solve projected problem */
    ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSSynchronize(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
    idx = 0;
    do {
      ritz = pep->eigr[idx];
#if !defined(PETSC_USE_COMPLEX)
      ritzi = pep->eigi[idx];
      if (PetscAbsScalar(ritzi!=0.0)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PJD solver not implemented for complex Ritz values in real arithmetic");
#endif
      /* Compute Ritz vector u=V*X(:,1) */
      ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(pjd->V,0,nv);CHKERRQ(ierr);
      ierr = BVMultVec(pjd->V,1.0,0.0,u,pX+idx*ld);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = PEPJDComputeResidual(pep,PETSC_FALSE,u,ritz,r,ww);CHKERRQ(ierr);
      /* Check convergence */
      ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
      ierr = (*pep->converged)(pep,ritz,0,norm,&pep->errest[pep->nconv],pep->convergedctx);CHKERRQ(ierr);
      ierr = (*pep->stopping)(pep,pep->its,pep->max_it,(pep->errest[pep->nconv]<pep->tol)?pep->nconv+1:pep->nconv,pep->nev,&pep->reason,pep->stoppingctx);CHKERRQ(ierr);
      if (pep->errest[pep->nconv]<pep->tol) {
        /* Ritz pair converged */
        minv = PetscMin(nv,(PetscInt)(pjd->keep*pep->ncv));
        if (pjd->ld>1) {
          ierr = BVGetColumn(pjd->X,pep->nconv,&v);CHKERRQ(ierr);
          ierr = PEPJDCopyToExtendedVec(pep,v,pjd->T+pep->ncv*pep->nconv,pjd->ld-1,0,u,PETSC_TRUE);CHKERRQ(ierr);
          ierr = BVRestoreColumn(pjd->X,pep->nconv,&v);CHKERRQ(ierr);
          ierr = BVSetActiveColumns(pjd->X,0,pep->nconv+1);CHKERRQ(ierr);
          ierr = BVNormColumn(pjd->X,pep->nconv,NORM_2,&norm);CHKERRQ(ierr);
          ierr = BVScaleColumn(pjd->X,pep->nconv,1.0/norm);CHKERRQ(ierr);
          for (k=0;k<pep->nconv;k++) pjd->T[pep->ncv*pep->nconv+k] /= norm;
          pjd->T[(pep->ncv+1)*pep->nconv] = ritz;
          eig[pep->nconv] = ritz;
          idx++;
        } else {
          ierr = BVInsertVec(pep->V,pep->nconv,u);CHKERRQ(ierr);
        }
        pep->nconv++;
      }
    } while (pep->errest[pep->nconv]<pep->tol && pep->nconv<nv);  

    if (pep->reason==PEP_CONVERGED_ITERATING) {
      if (idx) {
        pjd->nlock +=idx;
        ierr = PEPJDLockConverged(pep,&nv,idx);CHKERRQ(ierr);
        ierr = PEPJDUpdateExtendedPC(pep,pep->target);CHKERRQ(ierr);
      }
      if (nv+sz>=pep->ncv-1) {
        /* Basis full, force restart */
        minv = PetscMin(nv,(PetscInt)(pjd->keep*pep->ncv));
        ierr = DSGetDimensions(pep->ds,&dim,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
        ierr = PEPJDOrthogonalize(dim,minv,pX,ld,&minv,NULL,NULL,ld);CHKERRQ(ierr);
        ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
        ierr = DSGetArray(pep->ds,DS_MAT_Y,&pX);CHKERRQ(ierr);
        ierr = PEPJDOrthogonalize(dim,minv,pX,ld,&minv,NULL,NULL,ld);CHKERRQ(ierr);
        ierr = DSRestoreArray(pep->ds,DS_MAT_Y,&pX);CHKERRQ(ierr);
        ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
        ierr = BVMultInPlace(pjd->V,X,0,minv);CHKERRQ(ierr);
        ierr = MatDestroy(&X);CHKERRQ(ierr);
        ierr = DSGetMat(pep->ds,DS_MAT_Y,&Y);CHKERRQ(ierr);
        ierr = BVMultInPlace(pjd->W,Y,0,minv);CHKERRQ(ierr);
        ierr = MatDestroy(&Y);CHKERRQ(ierr);
        nv = minv;
        bupdated = 0;
      } else {
        theta = pep->errest[pep->nconv]<pjd->fix?ritz:pep->target;
        /* Update system mat */
        ierr = PEPJDMatSetUp(pep,theta);CHKERRQ(ierr);
        /* Compute r' */
        ierr = PEPJDComputeResidual(pep,PETSC_TRUE,u,theta,p,ww);CHKERRQ(ierr);
        pcctx->u = u;
        /* Solve correction equation to expand basis */  
        ierr = PEPJDExtendedPCApply(pjd->pcshell,p,pcctx->Bp);CHKERRQ(ierr);
        ierr = VecDot(pcctx->Bp,u,&pcctx->gamma);CHKERRQ(ierr);
        ierr = BVGetColumn(pjd->V,nv,&t);CHKERRQ(ierr);
        ierr = KSPSolve(ksp,r,t);CHKERRQ(ierr);
        ierr = BVRestoreColumn(pjd->V,nv,&t);CHKERRQ(ierr);
        ierr = BVOrthogonalizeColumn(pjd->V,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
        if (lindep || norm==0.0) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
        ierr = BVScaleColumn(pjd->V,nv,1.0/norm);CHKERRQ(ierr);
        /* W = P(target)*V */
        ierr = BVGetColumn(pjd->V,nv,&t);CHKERRQ(ierr);
        ierr = BVGetColumn(pjd->W,nv,&v);CHKERRQ(ierr);
        ierr = PEPJDComputeResidual(pep,PETSC_FALSE,t,pep->target,v,ww);CHKERRQ(ierr);
        ierr = BVRestoreColumn(pjd->W,nv,&v);CHKERRQ(ierr);
        ierr = BVRestoreColumn(pjd->V,nv,&t);CHKERRQ(ierr);
        ierr = BVOrthogonalizeColumn(pjd->W,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
        if (lindep) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
        ierr = BVScaleColumn(pjd->W,nv,1.0/norm);CHKERRQ(ierr);
        bupdated = idx?0:nv;
        nv++;
      } 
    }
    for (k=pep->nconv;k<nv;k++) {
      eig[k] = pep->eigr[idx+k-pep->nconv];
#if !defined(PETSC_USE_COMPLEX)
      pep->eigi[k-pep->nconv] = 0.0;
#endif
    }
    ierr = PEPMonitor(pep,pep->its,pep->nconv,eig,pep->eigi,pep->errest,pep->nconv+1);CHKERRQ(ierr);
  }
  if (pjd->ld>1) {
    if (pep->nconv>0) { ierr = PEPJDEigenvectors(pep);CHKERRQ(ierr); }
    for (k=0;k<pep->nconv;k++) {
      pep->eigr[k] = eig[k];
      pep->eigi[k] = 0.0;
    }
    ierr = PetscFree2(pcctx->M,pcctx->ps);CHKERRQ(ierr);
  }
  ierr = KSPSetPC(ksp,pcctx->pc);CHKERRQ(ierr);
  ierr = MatShellGetContext(pjd->Pshell,(void**)&matctx);CHKERRQ(ierr);
  ierr = MatDestroy(&matctx->P);CHKERRQ(ierr);
  ierr = VecDestroy(&pcctx->Bp);CHKERRQ(ierr);
  ierr = MatDestroy(&pjd->Pshell);CHKERRQ(ierr);
  ierr = PCDestroy(&pcctx->pc);CHKERRQ(ierr);
  ierr = PetscFree(pcctx);CHKERRQ(ierr);
  ierr = PetscFree(matctx);CHKERRQ(ierr);
  ierr = PCDestroy(&pjd->pcshell);CHKERRQ(ierr);
  ierr = PetscFree2(eig,res);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPJDSetRestart_JD(PEP pep,PetscReal keep)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) pjd->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    pjd->keep = keep;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPJDSetRestart - Sets the restart parameter for the Jacobi-Davidson
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -pep_jd_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: PEPJDGetRestart()
@*/
PetscErrorCode PEPJDSetRestart(PEP pep,PetscReal keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  ierr = PetscTryMethod(pep,"PEPJDSetRestart_C",(PEP,PetscReal),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPJDGetRestart_JD(PEP pep,PetscReal *keep)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  *keep = pjd->keep;
  PetscFunctionReturn(0);
}

/*@
   PEPJDGetRestart - Gets the restart parameter used in the Jacobi-Davidson method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: PEPJDSetRestart()
@*/
PetscErrorCode PEPJDGetRestart(PEP pep,PetscReal *keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscUseMethod(pep,"PEPJDGetRestart_C",(PEP,PetscReal*),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPJDSetFix_JD(PEP pep,PetscReal fix)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  if (fix == PETSC_DEFAULT || fix == PETSC_DECIDE) pjd->fix = 0.01;
  else {
    if (fix < 0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid fix value");
    pjd->fix = fix;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPJDSetFix - Sets the threshold for changing the target in the correction
   equation.

   Logically Collective on PEP

   Input Parameters:
+  pep - the eigenproblem solver context
-  fix - threshold for changing the target

   Options Database Key:
.  -pep_jd_fix - the fix value

   Note:
   The target in the correction equation is fixed at the first iterations.
   When the norm of the residual vector is lower than the fix value,
   the target is set to the corresponding eigenvalue.

   Level: advanced

.seealso: PEPJDGetFix()
@*/
PetscErrorCode PEPJDSetFix(PEP pep,PetscReal fix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,fix,2);
  ierr = PetscTryMethod(pep,"PEPJDSetFix_C",(PEP,PetscReal),(pep,fix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPJDGetFix_JD(PEP pep,PetscReal *fix)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  *fix = pjd->fix;
  PetscFunctionReturn(0);
}

/*@
   PEPJDGetFix - Returns the threshold for changing the target in the correction
   equation.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  fix - threshold for changing the target

   Note:
   The target in the correction equation is fixed at the first iterations.
   When the norm of the residual vector is lower than the fix value,
   the target is set to the corresponding eigenvalue.

   Level: advanced

.seealso: PEPJDSetFix()
@*/
PetscErrorCode PEPJDGetFix(PEP pep,PetscReal *fix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(fix,2);
  ierr = PetscUseMethod(pep,"PEPJDGetFix_C",(PEP,PetscReal*),(pep,fix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetFromOptions_JD(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscReal      r1;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP JD Options");CHKERRQ(ierr);

    ierr = PetscOptionsReal("-pep_jd_restart","Proportion of vectors kept after restart","PEPJDSetRestart",0.5,&r1,&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPJDSetRestart(pep,r1);CHKERRQ(ierr); }

    ierr = PetscOptionsReal("-pep_jd_fix","Tolerance for changing the target in the correction equation","PEPJDSetFix",0.01,&r1,&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPJDSetFix(pep,r1);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPView_JD(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  %d%% of basis vectors kept after restart\n",(int)(100*pjd->keep));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  threshold for changing the target in the correction equation (fix): %g\n",(double)pjd->fix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetDefaultST_JD(PEP pep)
{
  PetscErrorCode ierr;
  KSP            ksp;

  PetscFunctionBegin;
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STPRECOND);CHKERRQ(ierr);
    ierr = STPrecondSetKSPHasMat(pep->st,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = STSetTransform(pep->st,PETSC_FALSE);CHKERRQ(ierr);
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPBCGSL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1e-5,PETSC_DEFAULT,PETSC_DEFAULT,100);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPReset_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<pep->nmat;i++) {
    ierr = BVDestroy(pjd->TV+i);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&pjd->W);CHKERRQ(ierr);
  if (pjd->ld>1) {
    ierr = BVDestroy(&pjd->V);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = BVDestroy(pjd->AX+i);CHKERRQ(ierr);
    }
    ierr = BVDestroy(&pjd->N[0]);CHKERRQ(ierr);
    ierr = BVDestroy(&pjd->N[1]);CHKERRQ(ierr);
    ierr = PetscFree3(pjd->XpX,pjd->T,pjd->Tj);CHKERRQ(ierr);
  }
  ierr = PetscFree2(pjd->TV,pjd->AX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PEPDestroy_JD(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetFix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetFix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PEPCreate_JD(PEP pep)
{
  PEP_JD         *pjd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&pjd);CHKERRQ(ierr);
  pep->data = (void*)pjd;

  pjd->fix = 0.01;

  pep->ops->solve          = PEPSolve_JD;
  pep->ops->setup          = PEPSetUp_JD;
  pep->ops->setfromoptions = PEPSetFromOptions_JD;
  pep->ops->destroy        = PEPDestroy_JD;
  pep->ops->reset          = PEPReset_JD;
  pep->ops->view           = PEPView_JD;
  pep->ops->setdefaultst   = PEPSetDefaultST_JD;

  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",PEPJDSetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",PEPJDGetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetFix_C",PEPJDSetFix_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetFix_C",PEPJDGetFix_JD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

