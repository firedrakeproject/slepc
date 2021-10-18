/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV developer functions needed in contour integral methods
*/

#include <slepc/private/bvimpl.h>            /*I "slepcbv.h" I*/
#include <slepcblaslapack.h>

#define p_id(i) (i*subcomm->n + subcomm->color)

/*@
   BVScatter - Scatters the columns of a BV to another BV created in a
   subcommunicator.

   Collective on Vin

   Input Parameters:
+  Vin  - input basis vectors (defined on the whole communicator)
.  scat - VecScatter object that contains the info for the communication
-  xdup - an auxiliary vector

   Output Parameter:
.  Vout - output basis vectors (defined on the subcommunicator)

   Notes:
   Currently implemented as a loop for each the active column, where each
   column is scattered independently. The vector xdup is defined on the
   contiguous parent communicator and have enough space to store one
   duplicate of the original vector per each subcommunicator.

   Level: developer

.seealso: BVGetColumn()
@*/
PetscErrorCode BVScatter(BV Vin,BV Vout,VecScatter scat,Vec xdup)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  Vec               v;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vin,BV_CLASSID,1);
  PetscValidHeaderSpecific(Vout,BV_CLASSID,2);
  PetscValidHeaderSpecific(scat,PETSCSF_CLASSID,3);
  PetscValidHeaderSpecific(xdup,VEC_CLASSID,4);
  for (i=Vin->l;i<Vin->k;i++) {
    ierr = BVGetColumn(Vin,i,&v);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat,v,xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat,v,xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = BVRestoreColumn(Vin,i,&v);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xdup,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(Vout->t,array);CHKERRQ(ierr);
    ierr = BVInsertVec(Vout,i,Vout->t);CHKERRQ(ierr);
    ierr = VecResetArray(Vout->t);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xdup,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   BVSumQuadrature - Computes the sum of terms required in the quadrature
   rule to approximate the contour integral.

   Collective on S

   Input Parameters:
+  Y       - input basis vectors
.  M       - number of moments
.  L       - block size
.  L_max   - maximum block size
.  w       - quadrature weights
.  zn      - normalized quadrature points
.  scat    - (optional) VecScatter object to communicate between subcommunicators
.  subcomm - subcommunicator layout
.  npoints - number of points to process by the subcommunicator
-  useconj - whether conjugate points can be used or not

   Output Parameter:
.  S       - output basis vectors

   Notes:
   This is a generalization of BVMult(). The resulting matrix S consists of M
   panels of L columns, and the following formula is computed for each panel:
   S_k = sum_j w_j*zn_j^k*Y_j, where Y_j is the j-th panel of Y containing
   the result of solving T(z_j)^{-1}*X for each integration point j. L_max is
   the width of the panels in Y.

   When using subcommunicators, Y is stored in the subcommunicators for a subset
   of integration points. In that case, the computation is done in the subcomm
   and then scattered to the whole communicator in S using the VecScatter scat.
   The value npoints is the number of points to be processed in this subcomm
   and the flag useconj indicates whether symmetric points can be reused.

   Level: developer

.seealso: BVMult(), BVScatter(), BVDotQuadrature(), RGComputeQuadrature(), RGCanUseConjugates()
@*/
PetscErrorCode BVSumQuadrature(BV S,BV Y,PetscInt M,PetscInt L,PetscInt L_max,PetscScalar *w,PetscScalar *zn,VecScatter scat,PetscSubcomm subcomm,PetscInt npoints,PetscBool useconj)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,nloc;
  Vec            v,sj;
  PetscScalar    *ppk,*pv,one=1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,BV_CLASSID,1);
  PetscValidHeaderSpecific(Y,BV_CLASSID,2);
  if (scat) PetscValidHeaderSpecific(scat,PETSCSF_CLASSID,8);

  ierr = BVGetSizes(Y,&nloc,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints,&ppk);CHKERRQ(ierr);
  for (i=0;i<npoints;i++) ppk[i] = 1.0;
  ierr = BVCreateVec(Y,&v);CHKERRQ(ierr);
  for (k=0;k<M;k++) {
    for (j=0;j<L;j++) {
      ierr = VecSet(v,0.0);CHKERRQ(ierr);
      for (i=0;i<npoints;i++) {
        ierr = BVSetActiveColumns(Y,i*L_max+j,i*L_max+j+1);CHKERRQ(ierr);
        ierr = BVMultVec(Y,ppk[i]*w[p_id(i)],1.0,v,&one);CHKERRQ(ierr);
      }
      if (useconj) {
        ierr = VecGetArray(v,&pv);CHKERRQ(ierr);
        for (i=0;i<nloc;i++) pv[i] = 2.0*PetscRealPart(pv[i]);
        ierr = VecRestoreArray(v,&pv);CHKERRQ(ierr);
      }
      ierr = BVGetColumn(S,k*L+j,&sj);CHKERRQ(ierr);
      if (scat) {
        ierr = VecScatterBegin(scat,v,sj,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(scat,v,sj,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(v,sj);CHKERRQ(ierr);
      }
      ierr = BVRestoreColumn(S,k*L+j,&sj);CHKERRQ(ierr);
    }
    for (i=0;i<npoints;i++) ppk[i] *= zn[p_id(i)];
  }
  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVDotQuadrature - Computes the projection terms required in the quadrature
   rule to approximate the contour integral.

   Collective on Y

   Input Parameters:
+  Y       - first basis vectors
.  V       - second basis vectors
.  M       - number of moments
.  L       - block size
.  L_max   - maximum block size
.  w       - quadrature weights
.  zn      - normalized quadrature points
.  subcomm - subcommunicator layout
.  npoints - number of points to process by the subcommunicator
-  useconj - whether conjugate points can be used or not

   Output Parameter:
.  Mu      - computed result

   Notes:
   This is a generalization of BVDot(). The resulting matrix Mu consists of M
   blocks of size LxL (placed horizontally), each of them computed as:
   Mu_k = sum_j w_j*zn_j^k*V'*Y_j, where Y_j is the j-th panel of Y containing
   the result of solving T(z_j)^{-1}*X for each integration point j. L_max is
   the width of the panels in Y.

   When using subcommunicators, Y is stored in the subcommunicators for a subset
   of integration points. In that case, the computation is done in the subcomm
   and then the final result is combined via reduction.
   The value npoints is the number of points to be processed in this subcomm
   and the flag useconj indicates whether symmetric points can be reused.

   Level: developer

.seealso: BVDot(), BVScatter(), BVSumQuadrature(), RGComputeQuadrature(), RGCanUseConjugates()
@*/
PetscErrorCode BVDotQuadrature(BV Y,BV V,PetscScalar *Mu,PetscInt M,PetscInt L,PetscInt L_max,PetscScalar *w,PetscScalar *zn,PetscSubcomm subcomm,PetscInt npoints,PetscBool useconj)
{
  PetscErrorCode ierr;
  PetscMPIInt       sub_size,count;
  PetscInt          i,j,k,s;
  PetscScalar       *temp,*temp2,*ppk,alp;
  Mat               H;
  const PetscScalar *pH;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);

  ierr = MPI_Comm_size(PetscSubcommChild(subcomm),&sub_size);CHKERRMPI(ierr);
  ierr = PetscMalloc3(npoints*L*(L+1),&temp,2*M*L*L,&temp2,npoints,&ppk);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,L,L_max*npoints,NULL,&H);CHKERRQ(ierr);
  ierr = PetscArrayzero(temp2,2*M*L*L);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(Y,0,L_max*npoints);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(V,0,L);CHKERRQ(ierr);
  ierr = BVDot(Y,V,H);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(H,&pH);CHKERRQ(ierr);
  for (i=0;i<npoints;i++) {
    for (j=0;j<L;j++) {
      for (k=0;k<L;k++) {
        temp[k+j*L+i*L*L] = pH[k+j*L+i*L*L_max];
      }
    }
  }
  ierr = MatDenseRestoreArrayRead(H,&pH);CHKERRQ(ierr);
  for (i=0;i<npoints;i++) ppk[i] = 1;
  for (k=0;k<2*M;k++) {
    for (j=0;j<L;j++) {
      for (i=0;i<npoints;i++) {
        alp = ppk[i]*w[p_id(i)];
        for (s=0;s<L;s++) {
          if (useconj) temp2[s+(j+k*L)*L] += 2.0*PetscRealPart(alp*temp[s+(j+i*L)*L]);
          else temp2[s+(j+k*L)*L] += alp*temp[s+(j+i*L)*L];
        }
      }
    }
    for (i=0;i<npoints;i++) ppk[i] *= zn[p_id(i)];
  }
  for (i=0;i<2*M*L*L;i++) temp2[i] /= sub_size;
  ierr = PetscMPIIntCast(2*M*L*L,&count);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(temp2,Mu,count,MPIU_SCALAR,MPIU_SUM,PetscSubcommParent(subcomm));CHKERRMPI(ierr);
  ierr = PetscFree3(temp,temp2,ppk);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVTraceQuadrature - Computes an estimate of the number of eigenvalues
   inside a region via quantities computed in the quadrature rule of
   contour integral methods.

   Collective on Y

   Input Parameters:
+  Y       - first basis vectors
.  V       - second basis vectors
.  L       - block size
.  L_max   - maximum block size
.  w       - quadrature weights
.  scat    - (optional) VecScatter object to communicate between subcommunicators
.  subcomm - subcommunicator layout
.  npoints - number of points to process by the subcommunicator
-  useconj - whether conjugate points can be used or not

   Output Parameter:
.  est_eig - estimated eigenvalue count

   Notes:
   This function returns an estimation of the number of eigenvalues in the
   region, computed as trace(V'*S_0), where S_0 is the first panel of S
   computed by BVSumQuadrature().

   When using subcommunicators, Y is stored in the subcommunicators for a subset
   of integration points. In that case, the computation is done in the subcomm
   and then scattered to the whole communicator in S using the VecScatter scat.
   The value npoints is the number of points to be processed in this subcomm
   and the flag useconj indicates whether symmetric points can be reused.

   Level: developer

.seealso: BVScatter(), BVDotQuadrature(), BVSumQuadrature(), RGComputeQuadrature(), RGCanUseConjugates()
@*/
PetscErrorCode BVTraceQuadrature(BV Y,BV V,PetscInt L,PetscInt L_max,PetscScalar *w,VecScatter scat,PetscSubcomm subcomm,PetscInt npoints,PetscBool useconj,PetscReal *est_eig)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  Vec            y,yall,vj;
  PetscScalar    dot,sum=0.0,one=1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);
  if (scat) PetscValidHeaderSpecific(scat,PETSCSF_CLASSID,6);

  ierr = BVCreateVec(Y,&y);CHKERRQ(ierr);
  ierr = BVCreateVec(V,&yall);CHKERRQ(ierr);
  for (j=0;j<L;j++) {
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    for (i=0;i<npoints;i++) {
      ierr = BVSetActiveColumns(Y,i*L_max+j,i*L_max+j+1);CHKERRQ(ierr);
      ierr = BVMultVec(Y,w[p_id(i)],1.0,y,&one);CHKERRQ(ierr);
    }
    ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
    if (scat) {
      ierr = VecScatterBegin(scat,y,yall,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(scat,y,yall,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecDot(vj,yall,&dot);CHKERRQ(ierr);
    } else {
      ierr = VecDot(vj,y,&dot);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
    if (useconj) sum += 2.0*PetscRealPart(dot);
    else sum += dot;
  }
  *est_eig = PetscAbsScalar(sum)/(PetscReal)L;
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&yall);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVSVDAndRank_Refine(BV S,PetscReal delta,PetscScalar *pA,PetscReal *sigma,PetscInt *rank)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,ml=S->k;
  PetscMPIInt    len;
  PetscScalar    *work,*B,*tempB,*sarray,*Q1,*Q2,*temp2,alpha=1.0,beta=0.0;
  PetscBLASInt   l,m,n,lda,ldu,ldvt,lwork,info,ldb,ldc;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
  ierr = BVGetArray(S,&sarray);CHKERRQ(ierr);
  ierr = PetscMalloc6(ml*ml,&temp2,S->n*ml,&Q1,S->n*ml,&Q2,ml*ml,&B,ml*ml,&tempB,5*ml,&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(5*ml,&rwork);CHKERRQ(ierr);
#endif
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);

  ierr = PetscArrayzero(B,ml*ml);CHKERRQ(ierr);
  for (i=0;i<ml;i++) B[i*ml+i]=1;

  for (k=0;k<2;k++) {
    ierr = PetscBLASIntCast(S->n,&m);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ml,&l);CHKERRQ(ierr);
    n = l; lda = m; ldb = m; ldc = l;
    if (!k) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,sarray,&lda,sarray,&ldb,&beta,pA,&ldc));
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,Q1,&lda,Q1,&ldb,&beta,pA,&ldc));
    }
    ierr = PetscArrayzero(temp2,ml*ml);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(ml*ml,&len);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(pA,temp2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)S));CHKERRMPI(ierr);

    ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
    n = m; lda = m; lwork = 5*m, ldu = 1; ldvt = 1;
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&m,&n,temp2,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&m,&n,temp2,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
    SlepcCheckLapackInfo("gesvd",info);

    ierr = PetscBLASIntCast(S->n,&l);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ml,&n);CHKERRQ(ierr);
    m = n; lda = l; ldb = m; ldc = l;
    if (!k) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,sarray,&lda,temp2,&ldb,&beta,Q1,&ldc));
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,Q1,&lda,temp2,&ldb,&beta,Q2,&ldc));
    }

    ierr = PetscBLASIntCast(ml,&l);CHKERRQ(ierr);
    m = l; n = l; lda = l; ldb = m; ldc = l;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,B,&lda,temp2,&ldb,&beta,tempB,&ldc));
    for (i=0;i<ml;i++) {
      sigma[i] = PetscSqrtReal(sigma[i]);
      for (j=0;j<S->n;j++) {
        if (k%2) Q2[j+i*S->n] /= sigma[i];
        else Q1[j+i*S->n] /= sigma[i];
      }
      for (j=0;j<ml;j++) B[j+i*ml] = tempB[j+i*ml]*sigma[i];
    }
  }

  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  n = m; lda = m; ldu=1; ldvt=1;
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&m,&n,B,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&m,&n,B,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);

  ierr = PetscBLASIntCast(S->n,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&n);CHKERRQ(ierr);
  m = n; lda = l; ldb = m; ldc = l;
  if (k%2) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&l,&n,&m,&alpha,Q1,&lda,B,&ldb,&beta,sarray,&ldc));
  } else {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&l,&n,&m,&alpha,Q2,&lda,B,&ldb,&beta,sarray,&ldc));
  }

  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = BVRestoreArray(S,&sarray);CHKERRQ(ierr);

  if (rank) {
    (*rank) = 0;
    for (i=0;i<ml;i++) {
      if (sigma[i]/PetscMax(sigma[0],1.0)>delta) (*rank)++;
    }
  }
  ierr = PetscFree6(temp2,Q1,Q2,B,tempB,work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode BVSVDAndRank_QR(BV S,PetscReal delta,PetscScalar *pA,PetscReal *sigma,PetscInt *rank)
{
  PetscErrorCode ierr;
  PetscInt       i,n,ml=S->k;
  PetscBLASInt   m,lda,lwork,info;
  PetscScalar    *work;
  PetscReal      *rwork;
  Mat            A;
  Vec            v;

  PetscFunctionBegin;
  /* Compute QR factorizaton of S */
  ierr = BVGetSizes(S,NULL,&n,NULL);CHKERRQ(ierr);
  n    = PetscMin(n,ml);
  ierr = BVSetActiveColumns(S,0,n);CHKERRQ(ierr);
  ierr = PetscArrayzero(pA,ml*n);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_SELF,n,n,PETSC_DECIDE,PETSC_DECIDE,pA,&A);CHKERRQ(ierr);
  ierr = BVOrthogonalize(S,A);CHKERRQ(ierr);
  if (n<ml) {
    /* the rest of the factorization */
    for (i=n;i<ml;i++) {
      ierr = BVGetColumn(S,i,&v);CHKERRQ(ierr);
      ierr = BVOrthogonalizeVec(S,v,pA+i*n,NULL,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(S,i,&v);CHKERRQ(ierr);
    }
  }
  ierr = PetscBLASIntCast(n,&lda);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  ierr = PetscMalloc2(5*ml,&work,5*ml,&rwork);CHKERRQ(ierr);
  lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&lda,&m,pA,&lda,sigma,NULL,&lda,NULL,&lda,work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&lda,&m,pA,&lda,sigma,NULL,&lda,NULL,&lda,work,&lwork,rwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  *rank = 0;
  for (i=0;i<n;i++) {
    if (sigma[i]/PetscMax(sigma[0],1)>delta) (*rank)++;
  }
  /* n first columns of A have the left singular vectors */
  ierr = BVMultInPlace(S,A,0,*rank);CHKERRQ(ierr);
  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BVSVDAndRank_QR_CAA(BV S,PetscInt M,PetscInt L,PetscReal delta,PetscScalar *pA,PetscReal *sigma,PetscInt *rank)
{
  PetscErrorCode ierr;
  PetscInt       i,j,n,ml=S->k;
  PetscBLASInt   m,k_,lda,lwork,info;
  PetscScalar    *work,*T,*U,*R,sone=1.0,zero=0.0;
  PetscReal      *rwork;
  Mat            A;

  PetscFunctionBegin;
  /* Compute QR factorizaton of S */
  ierr = BVGetSizes(S,NULL,&n,NULL);CHKERRQ(ierr);
  if (n<ml) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"The QR_CAA method does not support problem size n < m*L");
  ierr = BVSetActiveColumns(S,0,ml);CHKERRQ(ierr);
  ierr = PetscArrayzero(pA,ml*ml);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_SELF,ml,ml,PETSC_DECIDE,PETSC_DECIDE,pA,&A);CHKERRQ(ierr);
  ierr = BVOrthogonalize(S,A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /* SVD of first (M-1)*L diagonal block */
  ierr = PetscBLASIntCast((M-1)*L,&m);CHKERRQ(ierr);
  ierr = PetscMalloc5(m*m,&T,m*m,&R,m*m,&U,5*ml,&work,5*ml,&rwork);CHKERRQ(ierr);
  for (j=0;j<m;j++) {
    ierr = PetscArraycpy(R+j*m,pA+j*ml,m);CHKERRQ(ierr);
  }
  lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&m,&m,R,&m,sigma,U,&m,NULL,&m,work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&m,&m,R,&m,sigma,U,&m,NULL,&m,work,&lwork,rwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  *rank = 0;
  for (i=0;i<m;i++) {
    if (sigma[i]/PetscMax(sigma[0],1)>delta) (*rank)++;
  }
  ierr = MatCreateDense(PETSC_COMM_SELF,m,m,PETSC_DECIDE,PETSC_DECIDE,U,&A);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(S,0,m);CHKERRQ(ierr);
  ierr = BVMultInPlace(S,A,0,*rank);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  /* Projected linear system */
  /* m first columns of A have the right singular vectors */
  ierr = PetscBLASIntCast(*rank,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&lda);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&m,&k_,&m,&sone,pA+L*lda,&lda,R,&m,&zero,T,&m));
  ierr = PetscArrayzero(pA,ml*ml);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&m,&sone,U,&m,T,&m,&zero,pA,&k_));
  for (j=0;j<k_;j++) for (i=0;i<k_;i++) pA[j*k_+i] /= sigma[j];
  ierr = PetscFree5(T,R,U,work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVSVDAndRank - Compute the SVD (left singular vectors only, and singular
   values) and determine the numerical rank according to a tolerance.

   Collective on S

   Input Parameters:
+  S     - the basis vectors
.  m     - the moment degree
.  l     - the block size
.  delta - the tolerance used to determine the rank
-  meth  - the method to be used

   Output Parameters:
+  A     - workspace, on output contains relevant values in the CAA method
.  sigma - computed singular values
-  rank  - estimated rank (optional)

   Notes:
   This function computes [U,Sigma,V] = svd(S) and replaces S with U.
   The current implementation computes this via S'*S, and it may include
   some kind of iterative refinement to improve accuracy in some cases.

   The parameters m and l refer to the moment and block size of contour
   integral methods. All columns up to m*l are modified, and the active
   columns are set to 0..m*l.

   The method is one of BV_SVD_METHOD_REFINE, BV_SVD_METHOD_QR, BV_SVD_METHOD_QR_CAA.

   The A workspace should be m*l*m*l in size.

   Once the decomposition is computed, the numerical rank is estimated
   by counting the number of singular values that are larger than the
   tolerance delta, relative to the first singular value.

   Level: developer

.seealso: BVSetActiveColumns()
@*/
PetscErrorCode BVSVDAndRank(BV S,PetscInt m,PetscInt l,PetscReal delta,BVSVDMethod meth,PetscScalar *A,PetscReal *sigma,PetscInt *rank)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(S,m,2);
  PetscValidLogicalCollectiveInt(S,l,3);
  PetscValidLogicalCollectiveReal(S,delta,4);
  PetscValidLogicalCollectiveEnum(S,meth,5);
  PetscValidPointer(A,6);
  PetscValidPointer(sigma,7);
  PetscValidPointer(rank,8);

  ierr = PetscLogEventBegin(BV_SVDAndRank,S,0,0,0);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(S,0,m*l);CHKERRQ(ierr);
  switch (meth) {
    case BV_SVD_METHOD_REFINE:
      ierr = BVSVDAndRank_Refine(S,delta,A,sigma,rank);CHKERRQ(ierr);
      break;
    case BV_SVD_METHOD_QR:
      ierr = BVSVDAndRank_QR(S,delta,A,sigma,rank);CHKERRQ(ierr);
      break;
    case BV_SVD_METHOD_QR_CAA:
      ierr = BVSVDAndRank_QR_CAA(S,m,l,delta,A,sigma,rank);CHKERRQ(ierr);
      break;
  }
  ierr = PetscLogEventEnd(BV_SVDAndRank,S,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

