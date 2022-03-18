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
  PetscInt          i;
  Vec               v;
  const PetscScalar *array;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vin,BV_CLASSID,1);
  PetscValidHeaderSpecific(Vout,BV_CLASSID,2);
  PetscValidHeaderSpecific(scat,PETSCSF_CLASSID,3);
  PetscValidHeaderSpecific(xdup,VEC_CLASSID,4);
  for (i=Vin->l;i<Vin->k;i++) {
    CHKERRQ(BVGetColumn(Vin,i,&v));
    CHKERRQ(VecScatterBegin(scat,v,xdup,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat,v,xdup,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(BVRestoreColumn(Vin,i,&v));
    CHKERRQ(VecGetArrayRead(xdup,&array));
    CHKERRQ(VecPlaceArray(Vout->t,array));
    CHKERRQ(BVInsertVec(Vout,i,Vout->t));
    CHKERRQ(VecResetArray(Vout->t));
    CHKERRQ(VecRestoreArrayRead(xdup,&array));
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
   panels of L columns, and the following formula is computed for each panel
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
  PetscInt       i,j,k,nloc;
  Vec            v,sj;
  PetscScalar    *ppk,*pv,one=1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,BV_CLASSID,1);
  PetscValidHeaderSpecific(Y,BV_CLASSID,2);
  if (scat) PetscValidHeaderSpecific(scat,PETSCSF_CLASSID,8);

  CHKERRQ(BVGetSizes(Y,&nloc,NULL,NULL));
  CHKERRQ(PetscMalloc1(npoints,&ppk));
  for (i=0;i<npoints;i++) ppk[i] = 1.0;
  CHKERRQ(BVCreateVec(Y,&v));
  for (k=0;k<M;k++) {
    for (j=0;j<L;j++) {
      CHKERRQ(VecSet(v,0.0));
      for (i=0;i<npoints;i++) {
        CHKERRQ(BVSetActiveColumns(Y,i*L_max+j,i*L_max+j+1));
        CHKERRQ(BVMultVec(Y,ppk[i]*w[p_id(i)],1.0,v,&one));
      }
      if (PetscUnlikely(useconj)) {
        CHKERRQ(VecGetArray(v,&pv));
        for (i=0;i<nloc;i++) pv[i] = 2.0*PetscRealPart(pv[i]);
        CHKERRQ(VecRestoreArray(v,&pv));
      }
      CHKERRQ(BVGetColumn(S,k*L+j,&sj));
      if (PetscUnlikely(scat)) {
        CHKERRQ(VecScatterBegin(scat,v,sj,ADD_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(scat,v,sj,ADD_VALUES,SCATTER_REVERSE));
      } else {
        CHKERRQ(VecCopy(v,sj));
      }
      CHKERRQ(BVRestoreColumn(S,k*L+j,&sj));
    }
    for (i=0;i<npoints;i++) ppk[i] *= zn[p_id(i)];
  }
  CHKERRQ(PetscFree(ppk));
  CHKERRQ(VecDestroy(&v));
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
   blocks of size LxL (placed horizontally), each of them computed as
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
  PetscMPIInt       sub_size,count;
  PetscInt          i,j,k,s;
  PetscScalar       *temp,*temp2,*ppk,alp;
  Mat               H;
  const PetscScalar *pH;
  MPI_Comm          child,parent;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);

  CHKERRQ(PetscSubcommGetChild(subcomm,&child));
  CHKERRMPI(MPI_Comm_size(child,&sub_size));
  CHKERRQ(PetscMalloc3(npoints*L*(L+1),&temp,2*M*L*L,&temp2,npoints,&ppk));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,L,L_max*npoints,NULL,&H));
  CHKERRQ(PetscArrayzero(temp2,2*M*L*L));
  CHKERRQ(BVSetActiveColumns(Y,0,L_max*npoints));
  CHKERRQ(BVSetActiveColumns(V,0,L));
  CHKERRQ(BVDot(Y,V,H));
  CHKERRQ(MatDenseGetArrayRead(H,&pH));
  for (i=0;i<npoints;i++) {
    for (j=0;j<L;j++) {
      for (k=0;k<L;k++) {
        temp[k+j*L+i*L*L] = pH[k+j*L+i*L*L_max];
      }
    }
  }
  CHKERRQ(MatDenseRestoreArrayRead(H,&pH));
  for (i=0;i<npoints;i++) ppk[i] = 1;
  for (k=0;k<2*M;k++) {
    for (j=0;j<L;j++) {
      for (i=0;i<npoints;i++) {
        alp = ppk[i]*w[p_id(i)];
        for (s=0;s<L;s++) {
          if (!useconj) temp2[s+(j+k*L)*L] += alp*temp[s+(j+i*L)*L];
          else temp2[s+(j+k*L)*L] += 2.0*PetscRealPart(alp*temp[s+(j+i*L)*L]);
        }
      }
    }
    for (i=0;i<npoints;i++) ppk[i] *= zn[p_id(i)];
  }
  for (i=0;i<2*M*L*L;i++) temp2[i] /= sub_size;
  CHKERRQ(PetscMPIIntCast(2*M*L*L,&count));
  CHKERRQ(PetscSubcommGetParent(subcomm,&parent));
  CHKERRMPI(MPIU_Allreduce(temp2,Mu,count,MPIU_SCALAR,MPIU_SUM,parent));
  CHKERRQ(PetscFree3(temp,temp2,ppk));
  CHKERRQ(MatDestroy(&H));
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
  PetscInt       i,j;
  Vec            y,yall,vj;
  PetscScalar    dot,sum=0.0,one=1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);
  if (scat) PetscValidHeaderSpecific(scat,PETSCSF_CLASSID,6);

  CHKERRQ(BVCreateVec(Y,&y));
  CHKERRQ(BVCreateVec(V,&yall));
  for (j=0;j<L;j++) {
    CHKERRQ(VecSet(y,0.0));
    for (i=0;i<npoints;i++) {
      CHKERRQ(BVSetActiveColumns(Y,i*L_max+j,i*L_max+j+1));
      CHKERRQ(BVMultVec(Y,w[p_id(i)],1.0,y,&one));
    }
    CHKERRQ(BVGetColumn(V,j,&vj));
    if (scat) {
      CHKERRQ(VecScatterBegin(scat,y,yall,ADD_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(scat,y,yall,ADD_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecDot(vj,yall,&dot));
    } else {
      CHKERRQ(VecDot(vj,y,&dot));
    }
    CHKERRQ(BVRestoreColumn(V,j,&vj));
    if (useconj) sum += 2.0*PetscRealPart(dot);
    else sum += dot;
  }
  *est_eig = PetscAbsScalar(sum)/(PetscReal)L;
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&yall));
  PetscFunctionReturn(0);
}

PetscErrorCode BVSVDAndRank_Refine(BV S,PetscReal delta,PetscScalar *pA,PetscReal *sigma,PetscInt *rank)
{
  PetscInt       i,j,k,ml=S->k;
  PetscMPIInt    len;
  PetscScalar    *work,*B,*tempB,*sarray,*Q1,*Q2,*temp2,alpha=1.0,beta=0.0;
  PetscBLASInt   l,m,n,lda,ldu,ldvt,lwork,info,ldb,ldc;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
  CHKERRQ(BVGetArray(S,&sarray));
  CHKERRQ(PetscMalloc6(ml*ml,&temp2,S->n*ml,&Q1,S->n*ml,&Q2,ml*ml,&B,ml*ml,&tempB,5*ml,&work));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscMalloc1(5*ml,&rwork));
#endif
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));

  CHKERRQ(PetscArrayzero(B,ml*ml));
  for (i=0;i<ml;i++) B[i*ml+i]=1;

  for (k=0;k<2;k++) {
    CHKERRQ(PetscBLASIntCast(S->n,&m));
    CHKERRQ(PetscBLASIntCast(ml,&l));
    n = l; lda = m; ldb = m; ldc = l;
    if (!k) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,sarray,&lda,sarray,&ldb,&beta,pA,&ldc));
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,Q1,&lda,Q1,&ldb,&beta,pA,&ldc));
    }
    CHKERRQ(PetscArrayzero(temp2,ml*ml));
    CHKERRQ(PetscMPIIntCast(ml*ml,&len));
    CHKERRMPI(MPIU_Allreduce(pA,temp2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)S)));

    CHKERRQ(PetscBLASIntCast(ml,&m));
    n = m; lda = m; lwork = 5*m, ldu = 1; ldvt = 1;
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&m,&n,temp2,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&m,&n,temp2,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
    SlepcCheckLapackInfo("gesvd",info);

    CHKERRQ(PetscBLASIntCast(S->n,&l));
    CHKERRQ(PetscBLASIntCast(ml,&n));
    m = n; lda = l; ldb = m; ldc = l;
    if (!k) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,sarray,&lda,temp2,&ldb,&beta,Q1,&ldc));
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,Q1,&lda,temp2,&ldb,&beta,Q2,&ldc));
    }

    CHKERRQ(PetscBLASIntCast(ml,&l));
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

  CHKERRQ(PetscBLASIntCast(ml,&m));
  n = m; lda = m; ldu=1; ldvt=1;
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&m,&n,B,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&m,&n,B,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);

  CHKERRQ(PetscBLASIntCast(S->n,&l));
  CHKERRQ(PetscBLASIntCast(ml,&n));
  m = n; lda = l; ldb = m; ldc = l;
  if (k%2) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&l,&n,&m,&alpha,Q1,&lda,B,&ldb,&beta,sarray,&ldc));
  } else {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&l,&n,&m,&alpha,Q2,&lda,B,&ldb,&beta,sarray,&ldc));
  }

  CHKERRQ(PetscFPTrapPop());
  CHKERRQ(BVRestoreArray(S,&sarray));

  if (rank) {
    (*rank) = 0;
    for (i=0;i<ml;i++) {
      if (sigma[i]/PetscMax(sigma[0],1.0)>delta) (*rank)++;
    }
  }
  CHKERRQ(PetscFree6(temp2,Q1,Q2,B,tempB,work));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscFree(rwork));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode BVSVDAndRank_QR(BV S,PetscReal delta,PetscScalar *pA,PetscReal *sigma,PetscInt *rank)
{
  PetscInt       i,n,ml=S->k;
  PetscBLASInt   m,lda,lwork,info;
  PetscScalar    *work;
  PetscReal      *rwork;
  Mat            A;
  Vec            v;

  PetscFunctionBegin;
  /* Compute QR factorizaton of S */
  CHKERRQ(BVGetSizes(S,NULL,&n,NULL));
  n    = PetscMin(n,ml);
  CHKERRQ(BVSetActiveColumns(S,0,n));
  CHKERRQ(PetscArrayzero(pA,ml*n));
  CHKERRQ(MatCreateDense(PETSC_COMM_SELF,n,n,PETSC_DECIDE,PETSC_DECIDE,pA,&A));
  CHKERRQ(BVOrthogonalize(S,A));
  if (n<ml) {
    /* the rest of the factorization */
    for (i=n;i<ml;i++) {
      CHKERRQ(BVGetColumn(S,i,&v));
      CHKERRQ(BVOrthogonalizeVec(S,v,pA+i*n,NULL,NULL));
      CHKERRQ(BVRestoreColumn(S,i,&v));
    }
  }
  CHKERRQ(PetscBLASIntCast(n,&lda));
  CHKERRQ(PetscBLASIntCast(ml,&m));
  CHKERRQ(PetscMalloc2(5*ml,&work,5*ml,&rwork));
  lwork = 5*m;
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&lda,&m,pA,&lda,sigma,NULL,&lda,NULL,&lda,work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&lda,&m,pA,&lda,sigma,NULL,&lda,NULL,&lda,work,&lwork,rwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);
  CHKERRQ(PetscFPTrapPop());
  *rank = 0;
  for (i=0;i<n;i++) {
    if (sigma[i]/PetscMax(sigma[0],1)>delta) (*rank)++;
  }
  /* n first columns of A have the left singular vectors */
  CHKERRQ(BVMultInPlace(S,A,0,*rank));
  CHKERRQ(PetscFree2(work,rwork));
  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

PetscErrorCode BVSVDAndRank_QR_CAA(BV S,PetscInt M,PetscInt L,PetscReal delta,PetscScalar *pA,PetscReal *sigma,PetscInt *rank)
{
  PetscInt       i,j,n,ml=S->k;
  PetscBLASInt   m,k_,lda,lwork,info;
  PetscScalar    *work,*T,*U,*R,sone=1.0,zero=0.0;
  PetscReal      *rwork;
  Mat            A;

  PetscFunctionBegin;
  /* Compute QR factorizaton of S */
  CHKERRQ(BVGetSizes(S,NULL,&n,NULL));
  PetscCheck(n>=ml,PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"The QR_CAA method does not support problem size n < m*L");
  CHKERRQ(BVSetActiveColumns(S,0,ml));
  CHKERRQ(PetscArrayzero(pA,ml*ml));
  CHKERRQ(MatCreateDense(PETSC_COMM_SELF,ml,ml,PETSC_DECIDE,PETSC_DECIDE,pA,&A));
  CHKERRQ(BVOrthogonalize(S,A));
  CHKERRQ(MatDestroy(&A));

  /* SVD of first (M-1)*L diagonal block */
  CHKERRQ(PetscBLASIntCast((M-1)*L,&m));
  CHKERRQ(PetscMalloc5(m*m,&T,m*m,&R,m*m,&U,5*ml,&work,5*ml,&rwork));
  for (j=0;j<m;j++) {
    CHKERRQ(PetscArraycpy(R+j*m,pA+j*ml,m));
  }
  lwork = 5*m;
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&m,&m,R,&m,sigma,U,&m,NULL,&m,work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&m,&m,R,&m,sigma,U,&m,NULL,&m,work,&lwork,rwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);
  CHKERRQ(PetscFPTrapPop());
  *rank = 0;
  for (i=0;i<m;i++) {
    if (sigma[i]/PetscMax(sigma[0],1)>delta) (*rank)++;
  }
  CHKERRQ(MatCreateDense(PETSC_COMM_SELF,m,m,PETSC_DECIDE,PETSC_DECIDE,U,&A));
  CHKERRQ(BVSetActiveColumns(S,0,m));
  CHKERRQ(BVMultInPlace(S,A,0,*rank));
  CHKERRQ(MatDestroy(&A));
  /* Projected linear system */
  /* m first columns of A have the right singular vectors */
  CHKERRQ(PetscBLASIntCast(*rank,&k_));
  CHKERRQ(PetscBLASIntCast(ml,&lda));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&m,&k_,&m,&sone,pA+L*lda,&lda,R,&m,&zero,T,&m));
  CHKERRQ(PetscArrayzero(pA,ml*ml));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&m,&sone,U,&m,T,&m,&zero,pA,&k_));
  for (j=0;j<k_;j++) for (i=0;i<k_;i++) pA[j*k_+i] /= sigma[j];
  CHKERRQ(PetscFree5(T,R,U,work,rwork));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(S,m,2);
  PetscValidLogicalCollectiveInt(S,l,3);
  PetscValidLogicalCollectiveReal(S,delta,4);
  PetscValidLogicalCollectiveEnum(S,meth,5);
  PetscValidScalarPointer(A,6);
  PetscValidRealPointer(sigma,7);
  PetscValidIntPointer(rank,8);

  CHKERRQ(PetscLogEventBegin(BV_SVDAndRank,S,0,0,0));
  CHKERRQ(BVSetActiveColumns(S,0,m*l));
  switch (meth) {
    case BV_SVD_METHOD_REFINE:
      CHKERRQ(BVSVDAndRank_Refine(S,delta,A,sigma,rank));
      break;
    case BV_SVD_METHOD_QR:
      CHKERRQ(BVSVDAndRank_QR(S,delta,A,sigma,rank));
      break;
    case BV_SVD_METHOD_QR_CAA:
      CHKERRQ(BVSVDAndRank_QR_CAA(S,m,l,delta,A,sigma,rank));
      break;
  }
  CHKERRQ(PetscLogEventEnd(BV_SVDAndRank,S,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   BVCISSResizeBases - Resize the bases involved in CISS solvers when the L grows.

   Collective on S

   Input Parameters:
+  S      - basis of L*M columns
.  V      - basis of L columns (may be associated to subcommunicators)
.  Y      - basis of npoints*L columns
.  Lold   - old value of L
.  Lnew   - new value of L
.  M      - the moment size
-  npoints - number of integration points

   Level: developer

.seealso: BVResize()
@*/
PetscErrorCode BVCISSResizeBases(BV S,BV V,BV Y,PetscInt Lold,PetscInt Lnew,PetscInt M,PetscInt npoints)
{
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,BV_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);
  PetscValidHeaderSpecific(Y,BV_CLASSID,3);
  PetscValidLogicalCollectiveInt(S,Lold,4);
  PetscValidLogicalCollectiveInt(S,Lnew,5);
  PetscValidLogicalCollectiveInt(S,M,6);
  PetscValidLogicalCollectiveInt(S,npoints,7);

  CHKERRQ(BVResize(S,Lnew*M,PETSC_FALSE));
  CHKERRQ(BVResize(V,Lnew,PETSC_TRUE));
  CHKERRQ(BVResize(Y,Lnew*npoints,PETSC_TRUE));
  /* columns of Y are interleaved */
  for (i=npoints-1;i>=0;i--) {
    for (j=Lold-1;j>=0;j--) {
      CHKERRQ(BVCopyColumn(Y,i*Lold+j,i*Lnew+j));
    }
  }
  PetscFunctionReturn(0);
}
