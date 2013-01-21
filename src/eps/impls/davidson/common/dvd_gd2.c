/*
  SLEPc eigensolver: "davidson"

  Step: improve the eigenvectors X with GD2

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include "davidson.h"
#include <slepcvec.h>
#include <slepcblaslapack.h>

PetscErrorCode dvd_improvex_gd2_d(dvdDashboard *d);
PetscErrorCode dvd_improvex_gd2_gen(dvdDashboard *d, Vec *D,
                                   PetscInt max_size_D, PetscInt r_s,
                                   PetscInt r_e, PetscInt *size_D);
PetscErrorCode dvd_improvex_get_eigenvectors(dvdDashboard *d, PetscScalar *pX,
  PetscScalar *pY, PetscInt ld_,
  PetscScalar *auxS, PetscInt size_auxS);

#define size_Z (64*4)

/**** GD2 update step K*[A*X B*X]  ****/

typedef struct {
  PetscInt size_X;
  void
    *old_improveX_data;   /* old improveX_data */
  improveX_type
    old_improveX;         /* old improveX */
} dvdImprovex_gd2;

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_gd2"
PetscErrorCode dvd_improvex_gd2(dvdDashboard *d,dvdBlackboard *b,KSP ksp,PetscInt max_bs)
{
  PetscErrorCode  ierr;
  dvdImprovex_gd2 *data;
  PetscBool       her_probl,std_probl;
  PC              pc;
  PetscInt        s=1;

  PetscFunctionBegin;
  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

  /* Setting configuration constrains */
  /* If the arithmetic is real and the problem is not Hermitian, then
     the block size is incremented in one */
#if !defined(PETSC_USE_COMPLEX)
  if (!her_probl) {
    max_bs++;
    b->max_size_P = PetscMax(b->max_size_P, 2);
    s = 2;
  } else
#endif
    b->max_size_P = PetscMax(b->max_size_P, 1);
  b->max_size_X = PetscMax(b->max_size_X, max_bs);
  b->max_size_auxV = PetscMax(b->max_size_auxV,
     s +
     ((her_probl || !d->eps->trueres)?1:PetscMax(s*2,b->max_size_cX_proj+b->max_size_X))); /* testConv */
 
  b->max_size_auxS = PetscMax(b->max_size_auxS,
      (her_probl || !d->eps->trueres)?0:b->max_nev*b->max_nev+PetscMax(b->max_nev*6,(b->max_nev+b->max_size_proj)*s+b->max_nev*(b->max_size_X+b->max_size_cX_proj)*(std_probl?2:4)+64)); /* preTestConv */

  /* Setup the preconditioner */
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = dvd_static_precond_PC(d,b,pc);CHKERRQ(ierr);
  } else {
    ierr = dvd_static_precond_PC(d,b,0);CHKERRQ(ierr);
  }

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdImprovex_gd2),&data);CHKERRQ(ierr);
    data->old_improveX_data = d->improveX_data;
    d->improveX_data = data;
    data->old_improveX = d->improveX;
    data->size_X = b->max_size_X;
    d->improveX = dvd_improvex_gd2_gen;

    DVD_FL_ADD(d->destroyList,dvd_improvex_gd2_d);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_gd2_d"
PetscErrorCode dvd_improvex_gd2_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdImprovex_gd2 *data = (dvdImprovex_gd2*)d->improveX_data;

  PetscFunctionBegin;
  /* Restore changes in dvdDashboard */
  d->improveX_data = data->old_improveX_data;

  /* Free local data and objects */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DVD_COMPLEX_RAYLEIGH_QUOTIENT(ur,ui,Axr,Axi,Bxr,Bxi,eigr,eigi,b,ierr)\
{ \
  ierr = VecDot((Axr), (ur), &(b)[0]);CHKERRQ(ierr); /* r*A*r */ \
  ierr = VecDot((Axr), (ui), &(b)[1]);CHKERRQ(ierr); /* i*A*r */ \
  ierr = VecDot((Axi), (ur), &(b)[2]);CHKERRQ(ierr); /* r*A*i */ \
  ierr = VecDot((Axi), (ui), &(b)[3]);CHKERRQ(ierr); /* i*A*i */ \
  ierr = VecDot((Bxr), (ur), &(b)[4]);CHKERRQ(ierr); /* r*B*r */ \
  ierr = VecDot((Bxr), (ui), &(b)[5]);CHKERRQ(ierr); /* i*B*r */ \
  ierr = VecDot((Bxi), (ur), &(b)[6]);CHKERRQ(ierr); /* r*B*i */ \
  ierr = VecDot((Bxi), (ui), &(b)[7]);CHKERRQ(ierr); /* i*B*i */ \
  (b)[0]  = (b)[0]+(b)[3]; /* rAr+iAi */ \
  (b)[2] =  (b)[2]-(b)[1]; /* rAi-iAr */ \
  (b)[4] = (b)[4]+(b)[7]; /* rBr+iBi */ \
  (b)[6] = (b)[6]-(b)[5]; /* rBi-iBr */ \
  (b)[7] = (b)[4]*(b)[4] + (b)[6]*(b)[6]; /* k */ \
  *(eigr) = ((b)[0]*(b)[4] + (b)[2]*(b)[6]) / (b)[7]; /* eig_r */ \
  *(eigi) = ((b)[2]*(b)[4] - (b)[0]*(b)[6]) / (b)[7]; /* eig_i */ \
}

#if !defined(PETSC_USE_COMPLEX)
#define DVD_COMPUTE_N_RR(eps,i,i_s,n,eigr,eigi,u,Ax,Bx,b,ierr) \
  for ((i)=0; (i)<(n); (i)++) { \
    if ((eigi)[(i_s)+(i)] != 0.0) { \
      /* eig_r = [(rAr+iAi)*(rBr+iBi) + (rAi-iAr)*(rBi-iBr)]/k \
         eig_i = [(rAi-iAr)*(rBr+iBi) - (rAr+iAi)*(rBi-iBr)]/k \
         k     =  (rBr+iBi)*(rBr+iBi) + (rBi-iBr)*(rBi-iBr)    */ \
      DVD_COMPLEX_RAYLEIGH_QUOTIENT((u)[(i)], (u)[(i)+1], (Ax)[(i)], \
        (Ax)[(i)+1], (Bx)[(i)], (Bx)[(i)+1], &(b)[8], &(b)[9], (b), (ierr)); \
      if (PetscAbsScalar((eigr)[(i_s)+(i)] - (b)[8])/ \
            PetscAbsScalar((eigr)[(i_s)+(i)]) > 1e-10    || \
          PetscAbsScalar((eigi)[(i_s)+(i)] - (b)[9])/ \
            PetscAbsScalar((eigi)[(i_s)+(i)]) > 1e-10) { \
        (ierr) = PetscInfo4((eps), "The eigenvalue %G+%G is far from its "\
                            "Rayleigh quotient value %G+%G\n", \
                            (eigr)[(i_s)+(i)], \
                            (eigi)[(i_s)+1], (b)[8], (b)[9]);CHKERRQ(ierr); \
      } \
      (i)++; \
    } else { \
      (ierr) = VecDot((Ax)[(i)], (u)[(i)], &(b)[0]);CHKERRQ(ierr); \
      (ierr) = VecDot((Bx)[(i)], (u)[(i)], &(b)[1]);CHKERRQ(ierr); \
      (b)[0] = (b)[0]/(b)[1]; \
      if (PetscAbsScalar((eigr)[(i_s)+(i)] - (b)[0])/ \
            PetscAbsScalar((eigr)[(i_s)+(i)]) > 1e-10) { \
        (ierr) = PetscInfo3((eps), "The eigenvalue %G is far from its " \
               "Rayleigh quotient value %G. (y'*B*x = %G)\n", \
               (eigr)[(i_s)+(i)], \
               (b)[0], (b)[1]);CHKERRQ(ierr); \
      } \
    } \
  }
#else
#define DVD_COMPUTE_N_RR(eps,i,i_s,n,eigr,eigi,u,Ax,Bx,b,ierr) \
  for ((i)=0; (i)<(n); (i)++) { \
      (ierr) = VecDot((Ax)[(i)], (u)[(i)], &(b)[0]);CHKERRQ(ierr); \
      (ierr) = VecDot((Bx)[(i)], (u)[(i)], &(b)[1]);CHKERRQ(ierr); \
      (b)[0] = (b)[0]/(b)[1]; \
      if (PetscAbsScalar((eigr)[(i_s)+(i)] - (b)[0])/ \
            PetscAbsScalar((eigr)[(i_s)+(i)]) > 1e-10) { \
        (ierr) = PetscInfo4((eps), "The eigenvalue %G+%G is far from its " \
               "Rayleigh quotient value %G+%G\n", \
               PetscRealPart((eigr)[(i_s)+(i)]), \
               PetscImaginaryPart((eigr)[(i_s)+(i)]), PetscRealPart((b)[0]), \
               PetscImaginaryPart((b)[0])); \
      } \
    }
#endif

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_gd2_gen"
PetscErrorCode dvd_improvex_gd2_gen(dvdDashboard *d,Vec *D,PetscInt max_size_D,PetscInt r_s,PetscInt r_e,PetscInt *size_D)
{
  dvdImprovex_gd2 *data = (dvdImprovex_gd2*)d->improveX_data;
  PetscErrorCode  ierr;
  PetscInt        i,j,n,s,ld,k;
  PetscScalar     *pX,*pY,b[10],Z[size_Z];
  Vec             *Ax,*Bx,X[4];

  PetscFunctionBegin;
  /* Compute the number of pairs to improve */
  n = PetscMin(PetscMin(PetscMin(data->size_X*2,max_size_D),(r_e-r_s)*2),d->max_size_proj-d->size_H)/2;
#if !defined(PETSC_USE_COMPLEX)
  /* If the last eigenvalue is a complex conjugate pair, n is increased by one */
  for (i=0; i<n; i++) {
    if (d->eigi[i] != 0.0) i++;
  }
  if (i > n) {
    n = PetscMin(PetscMin(PetscMin(data->size_X*2,max_size_D),(n+1)*2),d->max_size_proj-d->size_H)/2;
    if (i > n) n--;
  }
#endif

  /* Quick exit */
  if (max_size_D == 0 || r_e-r_s <= 0 || n == 0) {
   *size_D = 0;
   /* Callback old improveX */
    if (data->old_improveX) {
      d->improveX_data = data->old_improveX_data;
      ierr = data->old_improveX(d,PETSC_NULL,0,0,0,PETSC_NULL);CHKERRQ(ierr);
      d->improveX_data = data;
    }
    PetscFunctionReturn(0);
  }

  /* Compute the eigenvectors of the selected pairs */
  for (i=0;i<n;) {
    k = r_s+i+d->cX_in_H;
    ierr = DSVectors(d->ps,DS_MAT_X,&k,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSNormalize(d->ps,DS_MAT_X,r_s+i+d->cX_in_H);CHKERRQ(ierr);
    k = r_s+i+d->cX_in_H;
    ierr = DSVectors(d->ps,DS_MAT_Y,&k,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSNormalize(d->ps,DS_MAT_Y,r_s+i+d->cX_in_H);CHKERRQ(ierr);
    /* Jump complex conjugate pairs */
    i = k+1;
  }
  ierr = DSGetArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_Y,&pY);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);

  /* Bx <- B*X(i) */
  Bx = D+n;
  if (d->BV) {
    /* Compute the norms of the eigenvectors */
    if (d->correctXnorm) {
      ierr = dvd_improvex_compute_X(d,r_s,r_s+n,Bx,pX,ld);CHKERRQ(ierr);
    } else {
      for (i=0; i<n; i++) d->nX[r_s+i] = 1.0;
    }
    ierr = SlepcUpdateVectorsZ(Bx,0.0,1.0,d->BV-d->cX_in_H,d->size_BV+d->cX_in_H,&pX[ld*r_s],ld,d->size_H,n);CHKERRQ(ierr);
  } else if (d->B) {
    for (i=0;i<n;i++) {
      /* auxV(0) <- X(i) */
      ierr = dvd_improvex_compute_X(d,r_s+i,r_s+i+1,d->auxV,pX,ld);CHKERRQ(ierr);
      /* Bx(i) <- B*auxV(0) */
      ierr = MatMult(d->B,d->auxV[0],Bx[i]);CHKERRQ(ierr);
    }
  } else {
    /* Bx <- X */
    ierr = dvd_improvex_compute_X(d,r_s,r_s+n,Bx,pX,ld);CHKERRQ(ierr);
  }

  /* Ax <- A*X(i) */
  Ax = D;
  ierr = SlepcUpdateVectorsZ(Ax,0.0,1.0,d->AV-d->cX_in_H,d->size_AV+d->cX_in_H,&pX[ld*r_s],ld,d->size_H,n);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  s = d->eigi[r_s] == 0.0 ? 1 : 2;
  /* If the available vectors allow the computation of the eigenvalue */
  if (s <= n) {
#else
  s = 1;
#endif
  /* v <- Y(i) */
  ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,(d->W?d->W:d->V)-d->cX_in_H,d->size_V+d->cX_in_H,&pY[ld*r_s],ld,d->size_H,s);CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps,i,r_s,1,d->eigr,d->eigi,d->auxV,Ax,Bx,b,ierr);
#if !defined(PETSC_USE_COMPLEX)
  }
#endif

  ierr = DSRestoreArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->ps,DS_MAT_Y,&pY);CHKERRQ(ierr);

  for (i=0,s=0;i<n;i+=s) {
#if !defined(PETSC_USE_COMPLEX)
    if (d->eigi[r_s+i] != 0.0) {
       /* [Ax_i Ax_i+1 Bx_i Bx_i+1]*= [   1        0 
                                          0        1
                                       -eigr_i -eigi_i
                                        eigi_i -eigr_i] */
      b[0] = b[5] = 1.0/d->nX[r_s+i];
      b[2] = b[7] = -d->eigr[r_s+i]/d->nX[r_s+i];
      b[6] = -(b[3] = d->eigi[r_s+i]/d->nX[r_s+i]);
      b[1] = b[4] = 0.0;
      X[0] = Ax[i]; X[1] = Ax[i+1]; X[2] = Bx[i]; X[3] = Bx[i+1];
      ierr = SlepcUpdateVectorsD(X,4,1.0,b,4,4,2,Z,size_Z);CHKERRQ(ierr);
      s = 2;
    } else
#endif
    {
      /* [Ax_i Bx_i]*= [ 1/nX_i    conj(eig_i/nX_i)
                       -eig_i/nX_i     1/nX_i       ] */
      b[0] = 1.0/d->nX[r_s+i];
      b[1] = -d->eigr[r_s+i]/d->nX[r_s+i];
      b[2] = PetscConj(d->eigr[r_s+i]/d->nX[r_s+i]);
      b[3] = 1.0/d->nX[r_s+i];
      X[0] = Ax[i]; X[1] = Bx[i];
      ierr = SlepcUpdateVectorsD(X,2,1.0,b,2,2,2,Z,size_Z);CHKERRQ(ierr);
      s = 1;
    }
    for (j=0; j<s; j++) d->nX[r_s+i+j] = 1.0;

    /* Ax = R <- P*(Ax - eig_i*Bx) */
    ierr = d->calcpairs_proj_res(d,r_s+i,r_s+i+s,&Ax[i]);CHKERRQ(ierr);

    /* Check if the first eigenpairs are converged */
    if (i == 0) {
      ierr = d->preTestConv(d,0,s,s,Ax,PETSC_NULL,&d->npreconv);CHKERRQ(ierr);
      if (d->npreconv > 0) break;
    }
  }
 
  /* D <- K*[Ax Bx] */
  if (d->npreconv == 0) {
    ierr = VecCopy(D[0],d->auxV[0]);CHKERRQ(ierr);
    for (i=0;i<2*n-1;i++) {
      ierr = d->improvex_precond(d,r_s+(i+1)%n,D[i+1],D[i]);CHKERRQ(ierr);
    }
    ierr = d->improvex_precond(d,r_s,d->auxV[0],D[2*n-1]);CHKERRQ(ierr);
    *size_D = 2*n;
#if !defined(PETSC_USE_COMPLEX)
    if (d->eigi[r_s] != 0.0) {
      s = 4;
    } else
#endif
    s = 2;
    /* Prevent that short vectors are discarded in the orthogonalization */
    if (d->eps->errest[d->nconv+r_s] > PETSC_MACHINE_EPSILON && d->eps->errest[d->nconv+r_s] < PETSC_MAX_REAL) {
      for (i=0; i<s && i<*size_D; i++) {
        ierr = VecScale(D[i],1.0/d->eps->errest[d->nconv+r_s]);CHKERRQ(ierr);
      }
    }
  } else {
    *size_D = 0;
  }
 
  /* Callback old improveX */
  if (data->old_improveX) {
    d->improveX_data = data->old_improveX_data;
    ierr = data->old_improveX(d,PETSC_NULL,0,0,0,PETSC_NULL);CHKERRQ(ierr);
    d->improveX_data = data;
  }
  PetscFunctionReturn(0);
}
