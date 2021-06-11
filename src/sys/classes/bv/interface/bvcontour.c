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
   of intergration points. In that case, the computation is done in the subcomm
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
      ierr = VecSet(v,0);CHKERRQ(ierr);
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
   BVDotQuadrature - Computes the sum of terms required in the quadrature
   rule to approximate the contour integral.

   Collective on S

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
   of intergration points. In that case, the computation is done in the subcomm
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

