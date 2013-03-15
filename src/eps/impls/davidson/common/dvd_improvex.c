/*
  SLEPc eigensolver: "davidson"

  Step: improve the eigenvectors X

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
#include <slepc-private/vecimplslepc.h>         /*I "slepcvec.h" I*/

PetscErrorCode dvd_improvex_PfuncV(dvdDashboard *d,void *funcV,Vec *D,PetscInt max_size_D,PetscInt r_s,PetscInt r_e,Vec *auxV,PetscScalar *auxS);
PetscErrorCode dvd_pcapplyba(PC pc,PCSide side,Vec in,Vec out,Vec w);
PetscErrorCode dvd_pcapply(PC pc,Vec in,Vec out);
PetscErrorCode dvd_pcapplytrans(PC pc,Vec in,Vec out);
PetscErrorCode dvd_matmult_jd(Mat A,Vec in,Vec out);
PetscErrorCode dvd_matmulttrans_jd(Mat A,Vec in,Vec out);
PetscErrorCode dvd_matgetvecs_jd(Mat A,Vec *right,Vec *left);
PetscErrorCode dvd_improvex_jd_d(dvdDashboard *d);
PetscErrorCode dvd_improvex_jd_start(dvdDashboard *d);
PetscErrorCode dvd_improvex_jd_end(dvdDashboard *d);
PetscErrorCode dvd_improvex_jd_gen(dvdDashboard *d,Vec *D,PetscInt max_size_D,PetscInt r_s,PetscInt r_e,PetscInt *size_D);
PetscErrorCode dvd_improvex_jd_proj_cuv(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec **u,Vec **v,Vec *kr,Vec **auxV,PetscScalar **auxS,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld);
PetscErrorCode dvd_improvex_jd_proj_uv_KXX(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec *u,Vec *v,Vec *kr,Vec *auxV,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld);
PetscErrorCode dvd_improvex_jd_proj_uv_KZX(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec *u,Vec *v,Vec *kr,Vec *auxV,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld);
PetscErrorCode dvd_improvex_jd_lit_const_0(dvdDashboard *d,PetscInt i,PetscScalar* theta,PetscScalar* thetai,PetscInt *maxits,PetscReal *tol);
PetscErrorCode dvd_improvex_apply_proj(dvdDashboard *d,Vec *V,PetscInt cV,PetscScalar *auxS);
PetscErrorCode dvd_improvex_applytrans_proj(dvdDashboard *d,Vec *V,PetscInt cV,PetscScalar *auxS);

#define size_Z (64*4)

/**** JD update step (I - Kfg'/(g'Kf)) K(A - sB) (I - Kfg'/(g'Kf)) t = (I - Kfg'/(g'Kf))r  ****/

typedef struct {
  PetscInt size_X;
  void
    *old_improveX_data;   /* old improveX_data */
  improveX_type
    old_improveX;         /* old improveX */
  KSP ksp;                /* correction equation solver */
  Vec
    friends,              /* reference vector for composite vectors */
    *auxV;                /* auxiliar vectors */
  PetscScalar *auxS,      /* auxiliar scalars */
    *theta,
    *thetai;              /* the shifts used in the correction eq. */
  PetscInt maxits,        /* maximum number of iterations */
    r_s, r_e,             /* the selected eigenpairs to improve */
    ksp_max_size;         /* the ksp maximum subvectors size */
  PetscReal tol,          /* the maximum solution tolerance */
    lastTol,              /* last tol for dynamic stopping criterion */
    fix;                  /* tolerance for using the approx. eigenvalue */
  PetscBool
    dynamic;              /* if the dynamic stopping criterion is applied */
  dvdDashboard
    *d;                   /* the currect dvdDashboard reference */
  PC old_pc;              /* old pc in ksp */
  Vec
    *u,                   /* new X vectors */
    *real_KZ,             /* original KZ */
    *KZ;                  /* KZ vecs for the projector KZ*inv(X'*KZ)*X' */
  PetscScalar
   *XKZ,                  /* X'*KZ */
   *iXKZ;                 /* inverse of XKZ */
  PetscInt
    ldXKZ,                /* leading dimension of XKZ */
    size_iXKZ,            /* size of iXKZ */
    ldiXKZ,               /* leading dimension of iXKZ */
    size_KZ,              /* size of converged KZ */
    size_real_KZ,         /* original size of KZ */
    size_cX,              /* last value of d->size_cX */
    old_size_X;           /* last number of improved vectors */
  PetscBLASInt
    *iXKZPivots;          /* array of pivots */
} dvdImprovex_jd;

PETSC_STATIC_INLINE PetscErrorCode dvd_aux_matmult(dvdImprovex_jd *data,const Vec *x,const Vec *y,const Vec *auxV);
PETSC_STATIC_INLINE PetscErrorCode dvd_aux_matmulttrans(dvdImprovex_jd *data,const Vec *x,const Vec *y,const Vec *auxV);

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd"
PetscErrorCode dvd_improvex_jd(dvdDashboard *d,dvdBlackboard *b,KSP ksp,PetscInt max_bs,PetscInt cX_impr,PetscBool dynamic)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscBool       useGD,her_probl,std_probl;
  PC              pc;
  PetscInt        size_P,s=1;

  PetscFunctionBegin;
  std_probl = DVD_IS(d->sEP,DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP,DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

  /* Setting configuration constrains */
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&useGD);CHKERRQ(ierr);

  /* If the arithmetic is real and the problem is not Hermitian, then
     the block size is incremented in one */
#if !defined(PETSC_USE_COMPLEX)
  if (!her_probl) {
    max_bs++;
    b->max_size_P = PetscMax(b->max_size_P,2);
    s = 2;
  } else
#endif
    b->max_size_P = PetscMax(b->max_size_P,1);
  b->max_size_X = PetscMax(b->max_size_X,max_bs);
  size_P = b->max_size_P+cX_impr;
  b->max_size_auxV = PetscMax(b->max_size_auxV,
     b->max_size_X*3+(useGD?0:2)+ /* u, kr, auxV(max_size_X+2?) */
     ((her_probl || !d->eps->trueres)?1:PetscMax(s*2,b->max_size_cX_proj+b->max_size_X))); /* testConv */

  b->own_scalars+= size_P*size_P; /* XKZ */
  b->max_size_auxS = PetscMax(b->max_size_auxS,
    b->max_size_X*3 + /* theta, thetai */
    size_P*size_P + /* iXKZ */
    FromIntToScalar(size_P) + /* iXkZPivots */
    PetscMax(PetscMax(
      3*b->max_size_proj*b->max_size_X, /* dvd_improvex_apply_proj */
      8*cX_impr*b->max_size_X), /* dvd_improvex_jd_proj_cuv_KZX */
      (her_probl || !d->eps->trueres)?0:b->max_nev*b->max_nev+PetscMax(b->max_nev*6,(b->max_nev+b->max_size_proj)*s+b->max_nev*(b->max_size_X+b->max_size_cX_proj)*(std_probl?2:4)+64))); /* preTestConv */
  b->own_vecs+= size_P; /* KZ */

  /* Setup the preconditioner */
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = dvd_static_precond_PC(d,b,pc);CHKERRQ(ierr);
  } else {
    ierr = dvd_static_precond_PC(d,b,0);CHKERRQ(ierr);
  }

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdImprovex_jd),&data);CHKERRQ(ierr);
    data->dynamic = dynamic;
    data->size_real_KZ = size_P;
    data->real_KZ = b->free_vecs; b->free_vecs+= data->size_real_KZ;
    d->max_cX_in_impr = cX_impr;
    data->XKZ = b->free_scalars; b->free_scalars+= size_P*size_P;
    data->ldXKZ = size_P;
    data->size_X = b->max_size_X;
    data->old_improveX_data = d->improveX_data;
    d->improveX_data = data;
    data->old_improveX = d->improveX;
    data->ksp = useGD?NULL:ksp;
    data->d = d;
    d->improveX = dvd_improvex_jd_gen;
    data->ksp_max_size = max_bs;

    DVD_FL_ADD(d->startList,dvd_improvex_jd_start);
    DVD_FL_ADD(d->endList,dvd_improvex_jd_end);
    DVD_FL_ADD(d->destroyList,dvd_improvex_jd_d);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_start"
PetscErrorCode dvd_improvex_jd_start(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;
  PetscInt        rA, cA, rlA, clA;
  Mat             A;
  PetscBool       t;
  PC              pc;

  PetscFunctionBegin;
  data->KZ = data->real_KZ;
  data->size_KZ = data->size_cX = data->old_size_X = 0;
  data->lastTol = data->dynamic?0.5:0.0;

  /* Setup the ksp */
  if (data->ksp) {
    /* Create the reference vector */
    ierr = VecCreateCompWithVecs(d->V, data->ksp_max_size, NULL,
                                 &data->friends);CHKERRQ(ierr);

    /* Save the current pc and set a PCNONE */
    ierr = KSPGetPC(data->ksp, &data->old_pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)data->old_pc,PCNONE,&t);CHKERRQ(ierr);
    data->lastTol = 0.5;
    if (t) {
      data->old_pc = 0;
    } else {
      ierr = PetscObjectReference((PetscObject)data->old_pc);CHKERRQ(ierr);
      ierr = PCCreate(PetscObjectComm((PetscObject)d->eps),&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
      ierr = PCSetOperators(pc, d->A, d->A, SAME_PRECONDITIONER);CHKERRQ(ierr);
      ierr = PCShellSetApply(pc,dvd_pcapply);CHKERRQ(ierr);
      ierr = PCShellSetApplyBA(pc,dvd_pcapplyba);CHKERRQ(ierr);
      ierr = PCShellSetApplyTranspose(pc,dvd_pcapplytrans);CHKERRQ(ierr);
      ierr = KSPSetPC(data->ksp,pc);CHKERRQ(ierr);
      ierr = PCDestroy(&pc);CHKERRQ(ierr);
    }

    /* Create the (I-v*u')*K*(A-s*B) matrix */
    ierr = MatGetSize(d->A, &rA, &cA);CHKERRQ(ierr);
    ierr = MatGetLocalSize(d->A, &rlA, &clA);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)d->A), rlA*data->ksp_max_size,
                          clA*data->ksp_max_size, rA*data->ksp_max_size,
                          cA*data->ksp_max_size, data, &A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_MULT,
                                (void(*)(void))dvd_matmult_jd);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_MULT_TRANSPOSE,
                                (void(*)(void))dvd_matmulttrans_jd);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_GET_VECS,
                                (void(*)(void))dvd_matgetvecs_jd);CHKERRQ(ierr);

    /* Try to avoid KSPReset */
    ierr = KSPGetOperatorsSet(data->ksp,&t,NULL);CHKERRQ(ierr);
    if (t) {
      Mat M;
      PetscInt rM;
      ierr = KSPGetOperators(data->ksp,&M,NULL,NULL);CHKERRQ(ierr);
      ierr = MatGetSize(M,&rM,NULL);CHKERRQ(ierr);
      if (rM != rA*data->ksp_max_size) { ierr = KSPReset(data->ksp);CHKERRQ(ierr); }
    }
    ierr = KSPSetOperators(data->ksp, A, A, SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetUp(data->ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  } else {
    data->old_pc = 0;
    data->friends = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_end"
PetscErrorCode dvd_improvex_jd_end(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;

  PetscFunctionBegin;
  if (data->friends) { ierr = VecDestroy(&data->friends);CHKERRQ(ierr); }

  /* Restore the pc of ksp */
  if (data->old_pc) {
    ierr = KSPSetPC(data->ksp, data->old_pc);CHKERRQ(ierr);
    ierr = PCDestroy(&data->old_pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_d"
PetscErrorCode dvd_improvex_jd_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;

  PetscFunctionBegin;
  /* Restore changes in dvdDashboard */
  d->improveX_data = data->old_improveX_data;

  /* Free local data and objects */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_gen"
PetscErrorCode dvd_improvex_jd_gen(dvdDashboard *d,Vec *D,PetscInt max_size_D,PetscInt r_s,PetscInt r_e,PetscInt *size_D)
{
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;
  PetscErrorCode  ierr;
  PetscInt        i,j,n,maxits,maxits0,lits,s,ld,k;
  PetscScalar     *pX,*pY,*auxS = d->auxS,*auxS0;
  PetscReal       tol,tol0;
  Vec             *u,*v,*kr,kr_comp,D_comp;
  PetscBool       odd_situation = PETSC_FALSE;

  PetscFunctionBegin;
  /* Quick exit */
  if ((max_size_D == 0) || r_e-r_s <= 0) {
   *size_D = 0;
   /* Callback old improveX */
    if (data->old_improveX) {
      d->improveX_data = data->old_improveX_data;
      data->old_improveX(d, NULL, 0, 0, 0, NULL);
      d->improveX_data = data;
    }
    PetscFunctionReturn(0);
  }

  n = PetscMin(PetscMin(data->size_X, max_size_D), r_e-r_s);
  if (n == 0) SETERRQ(PETSC_COMM_SELF,1, "n == 0");
  if (data->size_X < r_e-r_s) SETERRQ(PETSC_COMM_SELF,1, "size_X < r_e-r_s");

  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);

  /* Restart lastTol if a new pair converged */
  if (data->dynamic && data->size_cX < d->size_cX)
    data->lastTol = 0.5;

  for (i=0,s=0,auxS0=auxS;i<n;i+=s) {
    /* If the selected eigenvalue is complex, but the arithmetic is real... */
#if !defined(PETSC_USE_COMPLEX)
    if (d->eigi[i] != 0.0) {
      if (i+2 <= max_size_D) s=2;
      else break;
    } else
#endif
      s=1;

    data->auxV = d->auxV;
    data->r_s = r_s+i; data->r_e = r_s+i+s;
    auxS = auxS0;
    data->theta = auxS; auxS+= 2*s;
    data->thetai = auxS; auxS+= s;
    kr = data->auxV; data->auxV+= s;

    /* Compute theta, maximum iterations and tolerance */
    maxits = 0; tol = 1;
    for (j=0;j<s;j++) {
      ierr = d->improvex_jd_lit(d, r_s+i+j, &data->theta[2*j],
                                &data->thetai[j], &maxits0, &tol0);CHKERRQ(ierr);
      maxits+= maxits0; tol*= tol0;
    }
    maxits/= s; tol = data->dynamic?data->lastTol:exp(log(tol)/s);

    /* Compute u, v and kr */
    k = r_s+i+d->cX_in_H;
    ierr = DSVectors(d->ps,DS_MAT_X,&k,NULL);CHKERRQ(ierr);
    ierr = DSNormalize(d->ps,DS_MAT_X,r_s+i+d->cX_in_H);CHKERRQ(ierr);
    k = r_s+i+d->cX_in_H;
    ierr = DSVectors(d->ps,DS_MAT_Y,&k,NULL);CHKERRQ(ierr);
    ierr = DSNormalize(d->ps,DS_MAT_Y,r_s+i+d->cX_in_H);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_Y,&pY);CHKERRQ(ierr);
    ierr = dvd_improvex_jd_proj_cuv(d,r_s+i,r_s+i+s,&u,&v,kr,&data->auxV,&auxS,data->theta,data->thetai,pX,pY,ld);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_Y,&pY);CHKERRQ(ierr);
    data->u = u;

    /* Check if the first eigenpairs are converged */
    if (i == 0) {
      PetscInt n_auxV = data->auxV-d->auxV+s, n_auxS = auxS - d->auxS;
      d->auxV+= n_auxV; d->size_auxV-= n_auxV;
      d->auxS+= n_auxS; d->size_auxS-= n_auxS;
      ierr = d->preTestConv(d,0,s,s,d->auxV-s,NULL,&d->npreconv);CHKERRQ(ierr);
      d->auxV-= n_auxV; d->size_auxV+= n_auxV;
      d->auxS-= n_auxS; d->size_auxS+= n_auxS;
      if (d->npreconv > 0) break;
    }

    /* Test the odd situation of solving Ax=b with A=I */
#if !defined(PETSC_USE_COMPLEX)
    odd_situation = (data->ksp && data->theta[0] == 1. && data->theta[1] == 0. && data->thetai[0] == 0. && d->B == NULL)?PETSC_TRUE:PETSC_FALSE;
#else
    odd_situation = (data->ksp && data->theta[0] == 1. && data->theta[1] == 0. && d->B == NULL)?PETSC_TRUE:PETSC_FALSE;
#endif
    /* If JD */
    if (data->ksp && !odd_situation) {
      data->auxS = auxS;

      /* kr <- -kr */
      for (j=0;j<s;j++) {
        ierr = VecScale(kr[j],-1.0);CHKERRQ(ierr);
      }

      /* Compouse kr and D */
      ierr = VecCreateCompWithVecs(kr, data->ksp_max_size, data->friends,
                                   &kr_comp);CHKERRQ(ierr);
      ierr = VecCreateCompWithVecs(&D[i], data->ksp_max_size, data->friends,
                                   &D_comp);CHKERRQ(ierr);
      ierr = VecCompSetSubVecs(data->friends,s,NULL);CHKERRQ(ierr);

      /* Solve the correction equation */
      ierr = KSPSetTolerances(data->ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT,
                              maxits);CHKERRQ(ierr);
      ierr = KSPSolve(data->ksp, kr_comp, D_comp);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(data->ksp, &lits);CHKERRQ(ierr);
      d->eps->st->lineariterations+= lits;

      /* Destroy the composed ks and D */
      ierr = VecDestroy(&kr_comp);CHKERRQ(ierr);
      ierr = VecDestroy(&D_comp);CHKERRQ(ierr);

    /* If GD */
    } else {
      for (j=0;j<s;j++) {
        ierr = d->improvex_precond(d, r_s+i+j, kr[j], D[i+j]);CHKERRQ(ierr);
      }
      ierr = dvd_improvex_apply_proj(d, &D[i], s, auxS);CHKERRQ(ierr);
    }
    /* Prevent that short vectors are discarded in the orthogonalization */
    if (i == 0 && d->eps->errest[d->nconv+r_s] > PETSC_MACHINE_EPSILON && d->eps->errest[d->nconv+r_s] < PETSC_MAX_REAL) {
      for (j=0;j<s;j++) {
        ierr = VecScale(D[j],1.0/d->eps->errest[d->nconv+r_s]);CHKERRQ(ierr);
      }
    }
  }
  *size_D = i;
  if (data->dynamic) data->lastTol = PetscMax(data->lastTol/2.0,PETSC_MACHINE_EPSILON*10.0);

  /* Callback old improveX */
  if (data->old_improveX) {
    d->improveX_data = data->old_improveX_data;
    data->old_improveX(d, NULL, 0, 0, 0, NULL);
    d->improveX_data = data;
  }
  PetscFunctionReturn(0);
}

/* y <- theta[1]A*x - theta[0]*B*x
   auxV, two auxiliary vectors */
#undef __FUNCT__
#define __FUNCT__ "dvd_aux_matmult"
PETSC_STATIC_INLINE PetscErrorCode dvd_aux_matmult(dvdImprovex_jd *data,const Vec *x,const Vec *y,const Vec *auxV)
{
  PetscErrorCode  ierr;
  PetscInt        n,i;
  const Vec       *Bx;

  PetscFunctionBegin;
  n = data->r_e - data->r_s;
  for (i=0;i<n;i++) {
    ierr = MatMult(data->d->A,x[i],y[i]);CHKERRQ(ierr);
  }

  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (data->d->eigi[data->r_s+i] != 0.0) {
      if (data->d->B) {
        ierr = MatMult(data->d->B,x[i],auxV[0]);CHKERRQ(ierr);
        ierr = MatMult(data->d->B,x[i+1],auxV[1]);CHKERRQ(ierr);
        Bx = auxV;
      } else Bx = &x[i];

      /* y_i   <- [ t_2i+1*A*x_i   - t_2i*Bx_i + ti_i*Bx_i+1;
         y_i+1      t_2i+1*A*x_i+1 - ti_i*Bx_i - t_2i*Bx_i+1  ] */
      ierr = VecAXPBYPCZ(y[i],-data->theta[2*i],data->thetai[i],data->theta[2*i+1],Bx[0],Bx[1]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(y[i+1],-data->thetai[i],-data->theta[2*i],data->theta[2*i+1],Bx[0],Bx[1]);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      if (data->d->B) {
        ierr = MatMult(data->d->B,x[i],auxV[0]);CHKERRQ(ierr);
        Bx = auxV;
      } else Bx = &x[i];
      ierr = VecAXPBY(y[i],-data->theta[i*2],data->theta[i*2+1],Bx[0]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* y <- theta[1]'*A'*x - theta[0]'*B'*x
   auxV, two auxiliary vectors */
#undef __FUNCT__
#define __FUNCT__ "dvd_aux_matmulttrans"
PETSC_STATIC_INLINE PetscErrorCode dvd_aux_matmulttrans(dvdImprovex_jd *data,const Vec *x,const Vec *y,const Vec *auxV)
{
  PetscErrorCode  ierr;
  PetscInt        n,i;
  const Vec       *Bx;

  PetscFunctionBegin;
  n = data->r_e - data->r_s;
  for (i=0;i<n;i++) {
    ierr = MatMultTranspose(data->d->A,x[i],y[i]);CHKERRQ(ierr);
  }

  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (data->d->eigi[data->r_s+i] != 0.0) {
      if (data->d->B) {
        ierr = MatMultTranspose(data->d->B,x[i],auxV[0]);CHKERRQ(ierr);
        ierr = MatMultTranspose(data->d->B,x[i+1],auxV[1]);CHKERRQ(ierr);
        Bx = auxV;
      } else Bx = &x[i];

      /* y_i   <- [ t_2i+1*A*x_i   - t_2i*Bx_i - ti_i*Bx_i+1;
         y_i+1      t_2i+1*A*x_i+1 + ti_i*Bx_i - t_2i*Bx_i+1  ] */
      ierr = VecAXPBYPCZ(y[i],-data->theta[2*i],-data->thetai[i],data->theta[2*i+1],Bx[0],Bx[1]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(y[i+1],data->thetai[i],-data->theta[2*i],data->theta[2*i+1],Bx[0],Bx[1]);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      if (data->d->B) {
        ierr = MatMultTranspose(data->d->B,x[i],auxV[0]);CHKERRQ(ierr);
        Bx = auxV;
      } else Bx = &x[i];
      ierr = VecAXPBY(y[i],PetscConj(-data->theta[i*2]),PetscConj(data->theta[i*2+1]),Bx[0]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_pcapplyba"
PetscErrorCode dvd_pcapplyba(PC pc,PCSide side,Vec in,Vec out,Vec w)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscInt        n,i;
  const Vec       *inx,*outx,*wx;
  Mat             A;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,&A,NULL,NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(A,(void**)&data);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(in,NULL,&inx);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(out,NULL,&outx);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(w,NULL,&wx);CHKERRQ(ierr);
  n = data->r_e - data->r_s;

  /* Check auxiliary vectors */
  if (&data->auxV[n] > data->d->auxV+data->d->size_auxV) SETERRQ(PETSC_COMM_SELF,1, "Insufficient auxiliary vectors");

  switch (side) {
  case PC_LEFT:
    /* aux <- theta[1]A*in - theta[0]*B*in */
    ierr = dvd_aux_matmult(data,inx,data->auxV,outx);CHKERRQ(ierr);

    /* out <- K * aux */
    for (i=0;i<n;i++) {
      ierr = data->d->improvex_precond(data->d,data->r_s+i,data->auxV[i],outx[i]);CHKERRQ(ierr);
    }
    break;
  case PC_RIGHT:
    /* aux <- K * in */
    for (i=0;i<n;i++) {
      ierr = data->d->improvex_precond(data->d,data->r_s+i,inx[i],data->auxV[i]);CHKERRQ(ierr);
    }

    /* out <- theta[1]A*auxV - theta[0]*B*auxV */
    ierr = dvd_aux_matmult(data,data->auxV,outx,wx);CHKERRQ(ierr);
    break;
  case PC_SYMMETRIC:
    /* aux <- K^{1/2} * in */
    for (i=0;i<n;i++) {
      ierr = PCApplySymmetricRight(data->old_pc,inx[i],data->auxV[i]);CHKERRQ(ierr);
    }

    /* out <- theta[1]A*auxV - theta[0]*B*auxV */
    ierr = dvd_aux_matmult(data,data->auxV,wx,outx);CHKERRQ(ierr);

    /* aux <- K^{1/2} * in */
    for (i=0;i<n;i++) {
      ierr = PCApplySymmetricLeft(data->old_pc,wx[i],outx[i]);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,1, "Unsupported KSP side");
  }

  /* out <- out - v*(u'*out) */
  ierr = dvd_improvex_apply_proj(data->d,(Vec*)outx,n,data->auxS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_pcapply"
PetscErrorCode dvd_pcapply(PC pc,Vec in,Vec out)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscInt        n,i;
  const Vec       *inx, *outx;
  Mat             A;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,&A,NULL,NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(A,(void**)&data);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(in,NULL,&inx);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(out,NULL,&outx);CHKERRQ(ierr);
  n = data->r_e - data->r_s;

  /* out <- K * in */
  for (i=0;i<n;i++) {
    ierr = data->d->improvex_precond(data->d,data->r_s+i,inx[i],outx[i]);CHKERRQ(ierr);
  }

  /* out <- out - v*(u'*out) */
  ierr = dvd_improvex_apply_proj(data->d,(Vec*)outx,n,data->auxS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_pcapplytrans"
PetscErrorCode dvd_pcapplytrans(PC pc,Vec in,Vec out)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscInt        n,i;
  const Vec       *inx, *outx;
  Mat             A;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,&A,NULL,NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(A,(void**)&data);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(in,NULL,&inx);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(out,NULL,&outx);CHKERRQ(ierr);
  n = data->r_e - data->r_s;

  /* Check auxiliary vectors */
  if (&data->auxV[n] > data->d->auxV+data->d->size_auxV) SETERRQ(PETSC_COMM_SELF,1, "Insufficient auxiliary vectors");

  /* auxV <- in */
  for (i=0;i<n;i++) {
    ierr = VecCopy(inx[i],data->auxV[i]);CHKERRQ(ierr);
  }

  /* auxV <- auxV - u*(v'*auxV) */
  ierr = dvd_improvex_applytrans_proj(data->d,data->auxV,n,data->auxS);CHKERRQ(ierr);

  /* out <- K' * aux */
  for (i=0;i<n;i++) {
    ierr = PCApplyTranspose(data->old_pc,data->auxV[i],outx[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_matmult_jd"
PetscErrorCode dvd_matmult_jd(Mat A,Vec in,Vec out)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscInt        n;
  const Vec       *inx, *outx;
  PCSide          side;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&data);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(in,NULL,&inx);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(out,NULL,&outx);CHKERRQ(ierr);
  n = data->r_e - data->r_s;

  /* Check auxiliary vectors */
  if (&data->auxV[2] > data->d->auxV+data->d->size_auxV) SETERRQ(PETSC_COMM_SELF,1, "Insufficient auxiliary vectors");

  /* out <- theta[1]A*in - theta[0]*B*in */
  ierr = dvd_aux_matmult(data,inx,outx,data->auxV);CHKERRQ(ierr);

  ierr = KSPGetPCSide(data->ksp,&side);CHKERRQ(ierr);
  if (side == PC_RIGHT) {
    /* out <- out - v*(u'*out) */
    ierr = dvd_improvex_apply_proj(data->d,(Vec*)outx,n,data->auxS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_matmulttrans_jd"
PetscErrorCode dvd_matmulttrans_jd(Mat A,Vec in,Vec out)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscInt        n,i;
  const Vec       *inx,*outx,*r,*auxV;
  PCSide          side;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&data);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(in,NULL,&inx);CHKERRQ(ierr);
  ierr = VecCompGetSubVecs(out,NULL,&outx);CHKERRQ(ierr);
  n = data->r_e - data->r_s;

  /* Check auxiliary vectors */
  if (&data->auxV[n+2] > data->d->auxV+data->d->size_auxV) SETERRQ(PETSC_COMM_SELF,1, "Insufficient auxiliary vectors");

  ierr = KSPGetPCSide(data->ksp,&side);CHKERRQ(ierr);
  if (side == PC_RIGHT) {
    /* auxV <- in */
    for (i=0;i<n;i++) {
      ierr = VecCopy(inx[i],data->auxV[i]);CHKERRQ(ierr);
    }

    /* auxV <- auxV - v*(u'*auxV) */
    ierr = dvd_improvex_applytrans_proj(data->d,data->auxV,n,data->auxS);CHKERRQ(ierr);
    r = data->auxV;
    auxV = data->auxV+n;
  } else {
    r = inx;
    auxV = data->auxV;
  }

  /* out <- theta[1]A*r - theta[0]*B*r */
  ierr = dvd_aux_matmulttrans(data,r,outx,auxV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_matgetvecs_jd"
PetscErrorCode dvd_matgetvecs_jd(Mat A,Vec *right,Vec *left)
{
  PetscErrorCode  ierr;
  Vec             *r, *l;
  dvdImprovex_jd  *data;
  PetscInt        n, i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void**)&data);CHKERRQ(ierr);
  n = data->ksp_max_size;
  if (right) {
    ierr = PetscMalloc(sizeof(Vec)*n, &r);CHKERRQ(ierr);
  }
  if (left) {
    ierr = PetscMalloc(sizeof(Vec)*n, &l);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    ierr = MatGetVecs(data->d->A, right?&r[i]:NULL,left?&l[i]:NULL);CHKERRQ(ierr);
  }
  if (right) {
    ierr = VecCreateCompWithVecs(r, n, data->friends, right);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = VecDestroy(&r[i]);CHKERRQ(ierr);
    }
  }
  if (left) {
    ierr = VecCreateCompWithVecs(l, n, data->friends, left);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = VecDestroy(&l[i]);CHKERRQ(ierr);
    }
  }

  if (right) {
    ierr = PetscFree(r);CHKERRQ(ierr);
  }
  if (left) {
    ierr = PetscFree(l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_proj_uv"
PetscErrorCode dvd_improvex_jd_proj_uv(dvdDashboard *d,dvdBlackboard *b,ProjType_t p)
{
  PetscFunctionBegin;
  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    switch (p) {
    case DVD_PROJ_KXX:
      d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KXX; break;
    case DVD_PROJ_KZX:
      d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KZX; break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_proj_cuv"
/*
  Compute: u <- X, v <- K*(theta[0]*A+theta[1]*B)*X,
  kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1], Y <- W*pY[i_s..i_e-1]
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
  auxS: max(8*bs*max_cX_in_proj,size_V*size_V)
*/
PetscErrorCode dvd_improvex_jd_proj_cuv(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec **u,Vec **v,Vec *kr,Vec **auxV,PetscScalar **auxS,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld)
{
#if defined(PETSC_MISSING_LAPACK_GETRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF - Lapack routine is unavailable");
#else
  PetscErrorCode    ierr;
  PetscInt          n = i_e - i_s, size_KZ, V_new, rm, i, size_in;
  dvdImprovex_jd    *data = (dvdImprovex_jd*)d->improveX_data;
  PetscBLASInt      s, ldXKZ, info;
  DvdReduction      r;
  DvdReductionChunk ops[2];
  DvdMult_copy_func sr[2];

  PetscFunctionBegin;
  /* Check consistency */
  V_new = d->size_cX - data->size_cX;
  if (V_new > data->old_size_X) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  data->old_size_X = n;

  /* KZ <- KZ(rm:rm+max_cX-1) */
  rm = PetscMax(V_new+data->size_KZ-d->max_cX_in_impr, 0);
  for (i=0; i<d->max_cX_in_impr; i++) {
    ierr = VecCopy(data->KZ[i+rm], data->KZ[i]);CHKERRQ(ierr);
  }
  data->size_cX = d->size_cX;

  /* XKZ <- XKZ(rm:rm+max_cX-1,rm:rm+max_cX-1) */
  for (i=0; i<d->max_cX_in_impr; i++) {
    ierr = SlepcDenseCopy(&data->XKZ[i*data->ldXKZ+i], data->ldXKZ, &data->XKZ[(i+rm)*data->ldXKZ+i+rm], data->ldXKZ, data->size_KZ, 1);CHKERRQ(ierr);
  }
  data->size_KZ = PetscMin(d->max_cX_in_impr, data->size_KZ+V_new);

  /* Compute X, KZ and KR */
  *u = *auxV; *auxV+= n;
  *v = &data->KZ[data->size_KZ];
  ierr = d->improvex_jd_proj_uv(d, i_s, i_e, *u, *v, kr, *auxV, theta, thetai,
                                pX, pY, ld);CHKERRQ(ierr);

  /* XKZ <- X'*KZ */
  size_KZ = data->size_KZ+n;
  size_in = 2*n*data->size_KZ+n*n;
  ierr = SlepcAllReduceSumBegin(ops,2,*auxS,*auxS+size_in,size_in,&r,PetscObjectComm((PetscObject)d->V[0]));CHKERRQ(ierr);
  ierr = VecsMultS(data->XKZ,0,data->ldXKZ,d->V-data->size_KZ,0,data->size_KZ,data->KZ,data->size_KZ,size_KZ,&r,&sr[0]);CHKERRQ(ierr);
  ierr = VecsMultS(&data->XKZ[data->size_KZ],0,data->ldXKZ,*u,0,n,data->KZ,0,size_KZ,&r,&sr[1]);CHKERRQ(ierr);
  ierr = SlepcAllReduceSumEnd(&r);CHKERRQ(ierr);

  /* iXKZ <- inv(XKZ) */
  ierr = PetscBLASIntCast(size_KZ,&s);CHKERRQ(ierr);
  data->ldiXKZ = data->size_iXKZ = size_KZ;
  data->iXKZ = *auxS; *auxS+= size_KZ*size_KZ;
  data->iXKZPivots = (PetscBLASInt*)*auxS;
  *auxS += FromIntToScalar(size_KZ);
  ierr = SlepcDenseCopy(data->iXKZ,data->ldiXKZ,data->XKZ,data->ldXKZ,size_KZ,size_KZ);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(data->ldiXKZ,&ldXKZ);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCall("LAPACKgetrf",LAPACKgetrf_(&s, &s, data->iXKZ, &ldXKZ, data->iXKZPivots, &info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Error in Lapack XGETRF %d", info);
  PetscFunctionReturn(0);
#endif
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
                            (eigi)[(i_s)+(i)], (b)[8], (b)[9]); \
      } \
      (i)++; \
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
#define __FUNCT__ "dvd_improvex_jd_proj_uv_KZX"
/*
  Compute: u <- X, v <- K*(theta[0]*A+theta[1]*B)*X,
  kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1], Y <- W*pY[i_s..i_e-1]
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
*/
PetscErrorCode dvd_improvex_jd_proj_uv_KZX(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec *u,Vec *v,Vec *kr,Vec *auxV,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     b[16];
  Vec             *Ax, *Bx, *r=auxV, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;
  /* u <- X(i) */
  ierr = dvd_improvex_compute_X(d,i_s,i_e,u,pX,ld);CHKERRQ(ierr);

  /* v <- theta[0]A*u + theta[1]*B*u */

  /* Bx <- B*X(i) */
  Bx = kr;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV-d->cX_in_H, d->size_BV+d->cX_in_H, &pX[ld*i_s], ld, d->size_H, n);CHKERRQ(ierr);
  } else {
    for (i=0;i<n;i++) {
      if (d->B) {
        ierr = MatMult(d->B, u[i], Bx[i]);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(u[i], Bx[i]);CHKERRQ(ierr);
      }
    }
  }

  /* Ax <- A*X(i) */
  Ax = r;
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV-d->cX_in_H, d->size_AV+d->cX_in_H, &pX[ld*i_s], ld, d->size_H, n);CHKERRQ(ierr);

  /* v <- Y(i) */
  ierr = SlepcUpdateVectorsZ(v, 0.0, 1.0, (d->W?d->W:d->V)-d->cX_in_H, d->size_V+d->cX_in_H, &pY[ld*i_s], ld, d->size_H, n);CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, v, Ax, Bx, b, ierr);

  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (d->eigi[i_s+i] != 0.0) {
      /* [r_i r_i+1 kr_i kr_i+1]*= [ theta_2i'    0            1        0
                                       0         theta_2i'     0        1
                                     theta_2i+1 -thetai_i   -eigr_i -eigi_i
                                     thetai_i    theta_2i+1  eigi_i -eigr_i ] */
      b[0] = b[5] = PetscConj(theta[2*i]);
      b[2] = b[7] = -theta[2*i+1];
      b[6] = -(b[3] = thetai[i]);
      b[1] = b[4] = 0.0;
      b[8] = b[13] = 1.0/d->nX[i_s+i];
      b[10] = b[15] = -d->eigr[i_s+i]/d->nX[i_s+i];
      b[14] = -(b[11] = d->eigi[i_s+i]/d->nX[i_s+i]);
      b[9] = b[12] = 0.0;
      X[0] = Ax[i]; X[1] = Ax[i+1]; X[2] = Bx[i]; X[3] = Bx[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 4, Z, size_Z);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      /* [Ax_i Bx_i]*= [ theta_2i'    1/nX_i
                        theta_2i+1  -eig_i/nX_i ] */
      b[0] = PetscConj(theta[i*2]);
      b[1] = theta[i*2+1];
      b[2] = 1.0/d->nX[i_s+i];
      b[3] = -d->eigr[i_s+i]/d->nX[i_s+i];
      X[0] = Ax[i]; X[1] = Bx[i];
      ierr = SlepcUpdateVectorsD(X, 2, 1.0, b, 2, 2, 2, Z, size_Z);CHKERRQ(ierr);
    }
  }
  for (i=0; i<n; i++) d->nX[i_s+i] = 1.0;

  /* v <- K^{-1} r = K^{-1}(theta_2i'*Ax + theta_2i+1*Bx) */
  for (i=0;i<n;i++) {
    ierr = d->improvex_precond(d, i_s+i, r[i], v[i]);CHKERRQ(ierr);
  }

  /* kr <- P*(Ax - eig_i*Bx) */
  ierr = d->calcpairs_proj_res(d, i_s, i_e, kr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_proj_uv_KXX"
/*
  Compute: u <- K^{-1}*X, v <- X,
  kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1]
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
*/
PetscErrorCode dvd_improvex_jd_proj_uv_KXX(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec *u,Vec *v,Vec *kr,Vec *auxV,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     b[16];
  Vec             *Ax, *Bx, *r = auxV, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;
  /* [v u] <- X(i) Y(i) */
  ierr = dvd_improvex_compute_X(d,i_s,i_e,v,pX,ld);CHKERRQ(ierr);
  ierr = SlepcUpdateVectorsZ(u, 0.0, 1.0, (d->W?d->W:d->V)-d->cX_in_H, d->size_V+d->cX_in_H, &pY[ld*i_s], ld, d->size_H, n);CHKERRQ(ierr);

  /* Bx <- B*X(i) */
  Bx = r;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV-d->cX_in_H, d->size_BV+d->cX_in_H, &pX[ld*i_s], ld, d->size_H, n);CHKERRQ(ierr);
  } else {
    if (d->B) {
      for (i=0;i<n;i++) {
        ierr = MatMult(d->B, v[i], Bx[i]);CHKERRQ(ierr);
      }
    } else Bx = v;
  }

  /* Ax <- A*X(i) */
  Ax = kr;
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV-d->cX_in_H, d->size_AV+d->cX_in_H, &pX[ld*i_s], ld, d->size_H, n);CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, u, Ax, Bx, b, ierr);

  for (i=0;i<n;i++) {
    if (d->eigi[i_s+i] == 0.0) {
      /* kr <- Ax -eig*Bx */
      ierr = VecAXPBY(kr[i], -d->eigr[i_s+i]/d->nX[i_s+i], 1.0/d->nX[i_s+i], Bx[i]);CHKERRQ(ierr);
    } else {
      /* [kr_i kr_i+1 r_i r_i+1]*= [   1        0
                                       0        1
                                    -eigr_i -eigi_i
                                     eigi_i -eigr_i] */
      b[0] = b[5] = 1.0/d->nX[i_s+i];
      b[2] = b[7] = -d->eigr[i_s+i]/d->nX[i_s+i];
      b[6] = -(b[3] = d->eigi[i_s+i]/d->nX[i_s+i]);
      b[1] = b[4] = 0.0;
      X[0] = kr[i]; X[1] = kr[i+1]; X[2] = r[i]; X[3] = r[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 2, Z, size_Z);CHKERRQ(ierr);
      i++;
    }
  }
  for (i=0; i<n; i++) d->nX[i_s+i] = 1.0;

  /* kr <- P*kr */
  ierr = d->calcpairs_proj_res(d, i_s, i_e, r);CHKERRQ(ierr);

  /* u <- K^{-1} X(i) */
  for (i=0;i<n;i++) {
    ierr = d->improvex_precond(d, i_s+i, v[i], u[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_lit_const"
PetscErrorCode dvd_improvex_jd_lit_const(dvdDashboard *d,dvdBlackboard *b,PetscInt maxits,PetscReal tol,PetscReal fix)
{
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;

  PetscFunctionBegin;
  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    data->maxits = maxits;
    data->tol = tol;
    data->fix = fix;
    d->improvex_jd_lit = dvd_improvex_jd_lit_const_0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_jd_lit_const_0"
PetscErrorCode dvd_improvex_jd_lit_const_0(dvdDashboard *d,PetscInt i,PetscScalar* theta,PetscScalar* thetai,PetscInt *maxits,PetscReal *tol)
{
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;
  PetscReal       a;

  PetscFunctionBegin;
  a = SlepcAbsEigenvalue(d->eigr[i],d->eigi[i]);

  if (d->nR[i]/a < data->fix) {
    theta[0] = d->eigr[i];
    theta[1] = 1.0;
#if !defined(PETSC_USE_COMPLEX)
    *thetai = d->eigi[i];
#endif
  } else {
    theta[0] = d->target[0];
    theta[1] = d->target[1];
#if !defined(PETSC_USE_COMPLEX)
    *thetai = 0.0;
#endif
}

#if defined(PETSC_USE_COMPLEX)
  if (thetai) *thetai = 0.0;
#endif
  *maxits = data->maxits;
  *tol = data->tol;
  PetscFunctionReturn(0);
}

/**** Patterns implementation *************************************************/

typedef PetscInt (*funcV0_t)(dvdDashboard*,PetscInt,PetscInt,Vec*);
typedef PetscInt (*funcV1_t)(dvdDashboard*,PetscInt,PetscInt,Vec*,PetscScalar*,Vec);

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_PfuncV"
/* Compute D <- K^{-1} * funcV[r_s..r_e] */
PetscErrorCode dvd_improvex_PfuncV(dvdDashboard *d,void *funcV,Vec *D,PetscInt max_size_D,PetscInt r_s,PetscInt r_e,Vec *auxV,PetscScalar *auxS)
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  if (max_size_D >= r_e-r_s+1) {
    /* The optimized version needs one vector extra of D */
    /* D(1:r.size) = R(r_s:r_e-1) */
    if (auxS) ((funcV1_t)funcV)(d, r_s, r_e, D+1, auxS, auxV[0]);
    else      ((funcV0_t)funcV)(d, r_s, r_e, D+1);

    /* D = K^{-1} * R */
    for (i=0; i<r_e-r_s; i++) {
      ierr = d->improvex_precond(d, i+r_s, D[i+1], D[i]);CHKERRQ(ierr);
    }
  } else if (max_size_D == r_e-r_s) {
    /* Non-optimized version */
    /* auxV <- R[r_e-1] */
    if (auxS) ((funcV1_t)funcV)(d, r_e-1, r_e, auxV, auxS, auxV[1]);
    else      ((funcV0_t)funcV)(d, r_e-1, r_e, auxV);

    /* D(1:r.size-1) = R(r_s:r_e-2) */
    if (auxS) ((funcV1_t)funcV)(d, r_s, r_e-1, D+1, auxS, auxV[1]);
    else      ((funcV0_t)funcV)(d, r_s, r_e-1, D+1);

    /* D = K^{-1} * R */
    for (i=0; i<r_e-r_s-1; i++) {
      ierr = d->improvex_precond(d, i+r_s, D[i+1], D[i]);CHKERRQ(ierr);
    }
    ierr = d->improvex_precond(d, r_e-1, auxV[0], D[r_e-r_s-1]);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,1, "Problem: r_e-r_s > max_size_D");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_apply_proj"
/* Compute (I - KZ*iXKZ*X')*V where,
   V, the vectors to apply the projector,
   cV, the number of vectors in V,
   auxS, auxiliar vector of size length 3*size_iXKZ*cV
*/
PetscErrorCode dvd_improvex_apply_proj(dvdDashboard *d,Vec *V,PetscInt cV,PetscScalar *auxS)
{
#if defined(PETSC_MISSING_LAPACK_GETRS)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routines are unavailable");
#else
  PetscErrorCode    ierr;
  dvdImprovex_jd    *data = (dvdImprovex_jd*)d->improveX_data;
  PetscInt          size_in = data->size_iXKZ*cV, i, ldh;
  PetscScalar       *h, *in, *out;
  PetscBLASInt      cV_, n, info, ld;
  DvdReduction      r;
  DvdReductionChunk ops[4];
  DvdMult_copy_func sr[4];

  PetscFunctionBegin;
  if (cV > 2) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* h <- X'*V */
  h = auxS; in = h+size_in; out = in+size_in; ldh = data->size_iXKZ;
  ierr = SlepcAllReduceSumBegin(ops, 4, in, out, size_in, &r,
                                PetscObjectComm((PetscObject)d->V[0]));CHKERRQ(ierr);
  for (i=0; i<cV; i++) {
    ierr = VecsMultS(&h[i*ldh],0,ldh,d->V-data->size_KZ,0,data->size_KZ,V+i,0,1,&r,&sr[i*2]);CHKERRQ(ierr);
    ierr = VecsMultS(&h[i*ldh+data->size_KZ],0,ldh,data->u,0,data->size_iXKZ-data->size_KZ,V+i,0,1,&r,&sr[i*2+1]);CHKERRQ(ierr);
  }
  ierr = SlepcAllReduceSumEnd(&r);CHKERRQ(ierr);

  /* h <- iXKZ\h */
  ierr = PetscBLASIntCast(cV,&cV_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(data->size_iXKZ,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(data->ldiXKZ,&ld);CHKERRQ(ierr);
  PetscValidScalarPointer(data->iXKZ,0);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCall("LAPACKgetrs",LAPACKgetrs_("N", &n, &cV_, data->iXKZ, &ld, data->iXKZPivots, h, &n, &info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Error in Lapack XGETRS %d", info);

  /* V <- V - KZ*h */
  for (i=0; i<cV; i++) {
    ierr = SlepcUpdateVectorsZ(V+i,1.0,-1.0,data->KZ,data->size_iXKZ,&h[ldh*i],ldh,data->size_iXKZ,1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_applytrans_proj"
/* Compute (I - X*iXKZ*KZ')*V where,
   V, the vectors to apply the projector,
   cV, the number of vectors in V,
   auxS, auxiliar vector of size length 3*size_iXKZ*cV
*/
PetscErrorCode dvd_improvex_applytrans_proj(dvdDashboard *d,Vec *V,PetscInt cV,PetscScalar *auxS)
{
#if defined(PETSC_MISSING_LAPACK_GETRS)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routines are unavailable");
#else
  PetscErrorCode    ierr;
  dvdImprovex_jd    *data = (dvdImprovex_jd*)d->improveX_data;
  PetscInt          size_in = data->size_iXKZ*cV, i, ldh;
  PetscScalar       *h, *in, *out;
  PetscBLASInt      cV_, n, info, ld;
  DvdReduction      r;
  DvdReductionChunk ops[2];
  DvdMult_copy_func sr[2];

  PetscFunctionBegin;
  if (cV > 2) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* h <- KZ'*V */
  h = auxS; in = h+size_in; out = in+size_in; ldh = data->size_iXKZ;
  ierr = SlepcAllReduceSumBegin(ops, 2, in, out, size_in, &r,
                                PetscObjectComm((PetscObject)d->V[0]));CHKERRQ(ierr);
  for (i=0; i<cV; i++) {
    ierr = VecsMultS(&h[i*ldh],0,ldh,data->KZ,0,data->size_KZ,V+i,0,1,&r,&sr[i]);CHKERRQ(ierr);
  }
  ierr = SlepcAllReduceSumEnd(&r);CHKERRQ(ierr);

  /* h <- iXKZ\h */
  ierr = PetscBLASIntCast(cV,&cV_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(data->size_iXKZ,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(data->ldiXKZ,&ld);CHKERRQ(ierr);
  PetscValidScalarPointer(data->iXKZ,0);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCall("LAPACKgetrs",LAPACKgetrs_("C", &n, &cV_, data->iXKZ, &ld, data->iXKZPivots, h, &n, &info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Error in Lapack XGETRS %d", info);

  /* V <- V - X*h */
  for (i=0; i<cV; i++) {
    ierr = SlepcUpdateVectorsZ(V+i,1.0,-1.0,d->V-data->size_KZ,data->size_KZ,&h[ldh*i],ldh,data->size_KZ,1);CHKERRQ(ierr);
    ierr = SlepcUpdateVectorsZ(V+i,1.0,-1.0,data->u,data->size_iXKZ-data->size_KZ,&h[ldh*i+data->size_KZ],ldh,data->size_iXKZ-data->size_KZ,1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#endif
}
