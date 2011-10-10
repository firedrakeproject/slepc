/*
  SLEPc eigensolver: "davidson"

  Step: improve the eigenvectors X

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

#include "davidson.h"
#include <slepcvec.h>

PetscErrorCode dvd_improvex_PfuncV(dvdDashboard *d, void *funcV, Vec *D,
  PetscInt max_size_D, PetscInt r_s, PetscInt r_e, Vec *auxV,
  PetscScalar *auxS);
PetscErrorCode dvd_matmult_jd(Mat A, Vec in, Vec out);
PetscErrorCode dvd_matgetvecs_jd(Mat A, Vec *right, Vec *left);
PetscErrorCode dvd_improvex_jd_d(dvdDashboard *d);
PetscErrorCode dvd_improvex_jd_start(dvdDashboard *d);
PetscErrorCode dvd_improvex_jd_end(dvdDashboard *d);
PetscErrorCode dvd_improvex_jd_gen(dvdDashboard *d, Vec *D,
                                   PetscInt max_size_D, PetscInt r_s,
                                   PetscInt r_e, PetscInt *size_D);
PetscErrorCode dvd_improvex_jd_proj_uv_KBXX(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld);
PetscErrorCode dvd_improvex_jd_proj_uv_KBXY(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld);
PetscErrorCode dvd_improvex_jd_proj_uv_KBXZ(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld);
PetscErrorCode dvd_improvex_jd_proj_uv_KBXZY(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld);
PetscErrorCode dvd_improvex_jd_proj_uv_KZX(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld);
PetscErrorCode dvd_improvex_jd_lit_const_0(dvdDashboard *d, PetscInt i,
  PetscScalar* theta, PetscScalar* thetai,
  PetscInt *maxits, PetscReal *tol);
PetscErrorCode dvd_improvex_get_eigenvectors(dvdDashboard *d, PetscScalar *pX,
  PetscScalar *pY, PetscInt ld_,
  PetscScalar *auxS, PetscInt size_auxS);

#define size_Z (64*4)

/**** JD update step (I - Kfg'/(g'Kf)) K(A - sB) (I - Kfg'/(g'Kf)) t = (I - Kfg'/(g'Kf))r  ****/

typedef struct {
  PetscInt size_X;
  void
    *old_improveX_data;   /* old improveX_data */
  improveX_type
    old_improveX;         /* old improveX */
  KSP ksp;                /* correction equation solver */
  Vec *u,*v,              /* the projector is (I-v*u') */
    friends,              /* reference vector for composite vectors */
    *auxV;                /* auxiliar vectors */
  PetscScalar *theta,
    *thetai;              /* the shifts used in the correction eq. */
  PetscInt maxits,        /* maximum number of iterations */
    r_s, r_e,             /* the selected eigenpairs to improve */
    n_uv,                 /* number of vectors u, v and kr */
    ksp_max_size;         /* the ksp maximum subvectors size */
  PetscReal tol,          /* the maximum solution tolerance */
    fix;                  /* tolerance for using the approx. eigenvalue */
  dvdDashboard
    *d;                   /* the currect dvdDashboard reference */
  PC old_pc;              /* old pc in ksp */
} dvdImprovex_jd;

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd"
PetscErrorCode dvd_improvex_jd(dvdDashboard *d, dvdBlackboard *b, KSP ksp,
                               PetscInt max_bs)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscBool       t;
  PC              pc;

  PetscFunctionBegin;

  /* Setting configuration constrains */
  /* If the arithmetic is real and the problem is not Hermitian, then
     the block size is incremented in one */
#if !defined(PETSC_USE_COMPLEX)
  if (DVD_ISNOT(d->sEP, DVD_EP_HERMITIAN)) {
    max_bs++;
    b->max_size_X = PetscMax(b->max_size_X, max_bs);
  }
#endif
  b->max_size_auxV = PetscMax(b->max_size_auxV, b->max_size_X*4); /* u,v,kr,
                                                                     auxV */
  b->max_size_auxS = PetscMax(b->max_size_auxS,
                              b->max_size_X*3 + /* theta, thetai */
                              (DVD_IS(d->sEP, DVD_EP_HERMITIAN)?0:1)*(
                              2*b->max_size_V*b->max_size_V + /* pX, pY */
                              11*b->max_size_V+4*b->max_size_V*b->max_size_V)
                                           /* dvd_improvex_get_eigenvectors */
                             );

  /* Setup the preconditioner */
  if (ksp) {
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = dvd_static_precond_PC(d, b, pc); CHKERRQ(ierr);
  } else {
    ierr = dvd_static_precond_PC(d, b, 0); CHKERRQ(ierr);
  }

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdImprovex_jd), &data); CHKERRQ(ierr);

    data->size_X = b->max_size_X;
    data->old_improveX_data = d->improveX_data;
    d->improveX_data = data;
    data->old_improveX = d->improveX;
    ierr = PetscTypeCompare((PetscObject)ksp, KSPPREONLY, &t); CHKERRQ(ierr);
    data->ksp = t?0:ksp;
    data->d = d;
    d->improveX = dvd_improvex_jd_gen;
    data->ksp_max_size = max_bs;

    /* Create the reference vector */
    ierr = VecCreateCompWithVecs(d->V, max_bs, PETSC_NULL, &data->friends);
    CHKERRQ(ierr);

    DVD_FL_ADD(d->startList, dvd_improvex_jd_start);
    DVD_FL_ADD(d->endList, dvd_improvex_jd_end);
    DVD_FL_ADD(d->destroyList, dvd_improvex_jd_d);
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

  /* Setup the ksp */
  if(data->ksp) {
    /* Save the current pc and set a PCNONE */
    ierr = KSPGetPC(data->ksp, &data->old_pc); CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)data->old_pc, PCNONE, &t);
    CHKERRQ(ierr);
    if (t) {
      data->old_pc = 0;
    } else {
      ierr = PetscObjectReference((PetscObject)data->old_pc); CHKERRQ(ierr);
      ierr = PCCreate(((PetscObject)d->eps)->comm, &pc); CHKERRQ(ierr);
      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
      ierr = PCSetOperators(pc, d->A, d->A, SAME_PRECONDITIONER); CHKERRQ(ierr);
      ierr = KSPSetPC(data->ksp, pc); CHKERRQ(ierr);
      ierr = PCDestroy(&pc); CHKERRQ(ierr);
    }

    /* Create the (I-v*u')*K*(A-s*B) matrix */
    ierr = MatGetSize(d->A, &rA, &cA); CHKERRQ(ierr);
    ierr = MatGetLocalSize(d->A, &rlA, &clA); CHKERRQ(ierr);
    ierr = MatCreateShell(((PetscObject)d->A)->comm, rlA*data->ksp_max_size,
                          clA*data->ksp_max_size, rA*data->ksp_max_size,
                          cA*data->ksp_max_size, data, &A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_MULT,
                                (void(*)(void))dvd_matmult_jd); CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_GET_VECS,
                                (void(*)(void))dvd_matgetvecs_jd);
    CHKERRQ(ierr);

    /* Try to avoid KSPReset */
    ierr = KSPGetOperatorsSet(data->ksp,&t,PETSC_NULL);CHKERRQ(ierr);
    if (t) {
      Mat M;
      PetscInt rM;
      ierr = KSPGetOperators(data->ksp,&M,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetSize(M,&rM,PETSC_NULL);CHKERRQ(ierr);
      if (rM != rA*data->ksp_max_size) { ierr = KSPReset(data->ksp);CHKERRQ(ierr); }
    }
    ierr = KSPSetOperators(data->ksp, A, A, SAME_PRECONDITIONER);
    CHKERRQ(ierr);
    ierr = KSPSetUp(data->ksp); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
  } else
    data->old_pc = 0;
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_end"
PetscErrorCode dvd_improvex_jd_end(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;

  PetscFunctionBegin;

  /* Restore the pc of ksp */
  if (data->old_pc) {
    ierr = KSPSetPC(data->ksp, data->old_pc); CHKERRQ(ierr);
    ierr = PCDestroy(&data->old_pc); CHKERRQ(ierr);
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
  ierr = VecDestroy(&data->friends); CHKERRQ(ierr);
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_gen"
PetscErrorCode dvd_improvex_jd_gen(dvdDashboard *d, Vec *D,
                             PetscInt max_size_D, PetscInt r_s,
                             PetscInt r_e, PetscInt *size_D)
{
  dvdImprovex_jd  *data = (dvdImprovex_jd*)d->improveX_data;
  PetscErrorCode  ierr;
  PetscInt        i, j, n, maxits, maxits0, lits, s;
  PetscScalar     a, *pX, *pY, *auxS = d->auxS;
  PetscReal       tol, tol0;
  Vec             *u, *v, *kr, kr_comp, D_comp;

  PetscFunctionBegin;

  /* Quick exit */
  if ((max_size_D == 0) || r_e-r_s <= 0) {
   /* Callback old improveX */
    if (data->old_improveX) {
      d->improveX_data = data->old_improveX_data;
      data->old_improveX(d, PETSC_NULL, 0, 0, 0, PETSC_NULL);
      d->improveX_data = data;
    }
    PetscFunctionReturn(0);
  }
 
  n = PetscMin(PetscMin(data->size_X, max_size_D), r_e-r_s);
  if (n == 0) SETERRQ(PETSC_COMM_SELF,1, "n == 0!\n");
  if (data->size_X < r_e-r_s) SETERRQ(PETSC_COMM_SELF,1, "size_X < r_e-r_s!\n");

  /* Compute the eigenvectors of the selected pairs */
  if (DVD_IS(d->sEP, DVD_EP_HERMITIAN)) {
    pX = pY = d->pX;
  } else {
    pX = auxS; auxS+= d->size_H*d->size_H;
    pY = auxS; auxS+= d->size_H*d->size_H;
    ierr = dvd_improvex_get_eigenvectors(d, pX, pY, d->size_H, auxS,
                                         d->size_auxS-(auxS-d->auxS));
    CHKERRQ(ierr);
  }

  for(i=0, s=0; i<n; i+=s) {
    /* If the selected eigenvalue is complex, but the arithmetic is real... */
#if !defined(PETSC_USE_COMPLEX)
    if (PetscAbsScalar(d->eigi[i] != 0.0)) { 
      if (i+2 <= max_size_D) s=2; else break;
    } else
#endif
      s=1;

    data->auxV = d->auxV;
    data->r_s = r_s+i; data->r_e = r_s+i+s;
    data->n_uv = s;
    data->theta = auxS; data->thetai = auxS+2*s;

    /* Compute theta, maximum iterations and tolerance */
    maxits = 0; tol = 1;
    for(j=0; j<s; j++) {
      ierr = d->improvex_jd_lit(d, r_s+i+j, &data->theta[2*j],
                                &data->thetai[j], &maxits0, &tol0);
      CHKERRQ(ierr);
      maxits+= maxits0; tol*= tol0;
    }
    maxits/= s; tol = exp(log(tol)/s);

    /* Compute u, v and kr */
    ierr = d->improvex_jd_proj_uv(d, r_s+i, r_s+i+s, &u, &v, &kr,
                                  &data->auxV, data->theta, data->thetai,
                                  &pX[d->size_H*(r_s+i)],
                                  &pY[d->size_H*(r_s+i)], d->size_H);
    CHKERRQ(ierr);
    data->u = u; data->v = v;

    /* Check if the first eigenpairs are converged */
    if (i == 0) {
      ierr = d->preTestConv(d, 0, s, s, PETSC_NULL, PETSC_NULL, &d->npreconv);
      CHKERRQ(ierr);
      if (d->npreconv > 0) break;
    }

    /* Compute kr <- kr - v*(u'*kr) */
    for(j=0; j<s; j++) {
      ierr = VecDot(kr[j], u[j], &a); CHKERRQ(ierr);
      ierr = VecAXPY(kr[j], -a, v[j]); CHKERRQ(ierr);
      ierr = VecScale(kr[j], -1.0); CHKERRQ(ierr);
    }

    /* If KSP==0, D <- kr */
    if (!data->ksp) {
      for(j=0; j<s; j++) {
        ierr = VecCopy(kr[j], D[j+i]); CHKERRQ(ierr);
      }
    } else {
      /* Compouse kr and D */
      ierr = VecCreateCompWithVecs(kr, data->ksp_max_size, data->friends,
                                   &kr_comp); CHKERRQ(ierr);
      ierr = VecCreateCompWithVecs(&D[i], data->ksp_max_size, data->friends,
                                   &D_comp); CHKERRQ(ierr);
      ierr = VecCompSetVecs(data->friends, PETSC_NULL, s); CHKERRQ(ierr);
  
      /* Solve the correction equation */
      ierr = KSPSetTolerances(data->ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT,
                              maxits); CHKERRQ(ierr);
      ierr = KSPSolve(data->ksp, kr_comp, D_comp); CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(data->ksp, &lits); CHKERRQ(ierr);
      d->eps->OP->lineariterations+= lits;
  
      /* Destroy the composed ks and D */
      ierr = VecDestroy(&kr_comp); CHKERRQ(ierr);
      ierr = VecDestroy(&D_comp); CHKERRQ(ierr);
    }
  }
  *size_D = i;
 
  /* Callback old improveX */
  if (data->old_improveX) {
    d->improveX_data = data->old_improveX_data;
    data->old_improveX(d, PETSC_NULL, 0, 0, 0, PETSC_NULL);
    d->improveX_data = data;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_matmult_jd"
PetscErrorCode dvd_matmult_jd(Mat A, Vec in, Vec out)
{
  PetscErrorCode  ierr;
  dvdImprovex_jd  *data;
  PetscScalar     a;
  PetscInt        n, i;
  const Vec       *inx, *outx, *Bx;

  PetscFunctionBegin;

  ierr = MatShellGetContext(A, (void**)&data); CHKERRQ(ierr);
  ierr = VecCompGetVecs(in, &inx, PETSC_NULL); CHKERRQ(ierr);
  ierr = VecCompGetVecs(out, &outx, PETSC_NULL); CHKERRQ(ierr);
  n = data->r_e - data->r_s;

  /* aux <- theta[1]A*in - theta[0]*B*in */
  for(i=0; i<n; i++) {
    ierr = MatMult(data->d->A, inx[i], data->auxV[i]); CHKERRQ(ierr);
  }
  if (data->d->B) {
    for(i=0; i<n; i++) {
      ierr = MatMult(data->d->B, inx[i], outx[i]); CHKERRQ(ierr);
    }
    Bx = outx;
  } else
    Bx = inx;

  for(i=0; i<n; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if(data->d->eigi[data->r_s+i] != 0.0) {
      /* aux_i   <- [ t_2i+1*A*inx_i   - t_2i*Bx_i + ti_i*Bx_i+1;
         aux_i+1      t_2i+1*A*inx_i+1 - ti_i*Bx_i - t_2i*Bx_i+1  ] */
      ierr = VecAXPBYPCZ(data->auxV[i], -data->theta[2*i], data->thetai[i],
                         data->theta[2*i+1], Bx[i], Bx[i+1]);
      CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(data->auxV[i+1], -data->thetai[i],
                         -data->theta[2*i], data->theta[2*i+1], Bx[i],
                         Bx[i+1]); CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      ierr = VecAXPBY(data->auxV[i], -data->theta[i*2], data->theta[i*2+1],
                      Bx[i]); CHKERRQ(ierr);
    }
  }

  /* out <- K * aux */
  for(i=0; i<n; i++) {
    ierr = data->d->improvex_precond(data->d, data->r_s+i, data->auxV[i],
                                     outx[i]); CHKERRQ(ierr);
  }

  /* out <- out - v*(u'*out) */
  for(i=0; i<n; i++) {
    ierr = VecDot(outx[i], data->u[i], &a); CHKERRQ(ierr);
    ierr = VecAXPY(outx[i], -a, data->v[i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_matgetvecs_jd"
PetscErrorCode dvd_matgetvecs_jd(Mat A, Vec *right, Vec *left)
{
  PetscErrorCode  ierr;
  Vec             *r, *l;
  dvdImprovex_jd  *data;
  PetscInt        n, i;

  PetscFunctionBegin;

  ierr = MatShellGetContext(A, (void**)&data); CHKERRQ(ierr);
  n = data->ksp_max_size;
  if (right) {
    ierr = PetscMalloc(sizeof(Vec)*n, &r); CHKERRQ(ierr);
  }
  if (left) {
    ierr = PetscMalloc(sizeof(Vec)*n, &l); CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    ierr = MatGetVecs(data->d->A, right?&r[i]:PETSC_NULL,
                      left?&l[i]:PETSC_NULL); CHKERRQ(ierr);
  }
  if(right) {
    ierr = VecCreateCompWithVecs(r, n, data->friends, right); CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = VecDestroy(&r[i]); CHKERRQ(ierr);
    }
  }
  if(left) {
    ierr = VecCreateCompWithVecs(l, n, data->friends, left); CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = VecDestroy(&l[i]); CHKERRQ(ierr);
    }
  }

  if (right) {
    ierr = PetscFree(r); CHKERRQ(ierr);
  }
  if (left) {
    ierr = PetscFree(l); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_proj_uv"
PetscErrorCode dvd_improvex_jd_proj_uv(dvdDashboard *d, dvdBlackboard *b,
                                       ProjType_t p)
{
  PetscFunctionBegin;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    /* switch(p) {
    case DVD_PROJ_KBXX:
      d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KBXX; break;
    case DVD_PROJ_KBXY:
      d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KBXY; break;
    case DVD_PROJ_KBXZ:
      d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KBXZ; break;
    case DVD_PROJ_KBXZY:
      d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KBXZY; break;
    } */
    //TODO: Temporally broken other projectors!!
    d->improvex_jd_proj_uv = dvd_improvex_jd_proj_uv_KZX;
  }

  PetscFunctionReturn(0);
}

#define DVD_COMPLEX_RAYLEIGH_QUOTIENT(ur,ui,Axr,Axi,Bxr,Bxi,eigr,eigi,b,ierr)\
{ \
  ierr = VecDot((Axr), (ur), &(b)[0]); CHKERRQ(ierr); /* r*A*r */ \
  ierr = VecDot((Axr), (ui), &(b)[1]); CHKERRQ(ierr); /* i*A*r */ \
  ierr = VecDot((Axi), (ur), &(b)[2]); CHKERRQ(ierr); /* r*A*i */ \
  ierr = VecDot((Axi), (ui), &(b)[3]); CHKERRQ(ierr); /* i*A*i */ \
  ierr = VecDot((Bxr), (ur), &(b)[4]); CHKERRQ(ierr); /* r*B*r */ \
  ierr = VecDot((Bxr), (ui), &(b)[5]); CHKERRQ(ierr); /* i*B*r */ \
  ierr = VecDot((Bxi), (ur), &(b)[6]); CHKERRQ(ierr); /* r*B*i */ \
  ierr = VecDot((Bxi), (ui), &(b)[7]); CHKERRQ(ierr); /* i*B*i */ \
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
  for((i)=0; (i)<(n); (i)++) { \
    if ((eigi)[(i_s)+(i)] != 0.0) { \
      /* eig_r = [(rAr+iAi)*(rBr+iBi) + (rAi-iAr)*(rBi-iBr)]/k \
         eig_i = [(rAi-iAr)*(rBr+iBi) - (rAr+iAi)*(rBi-iBr)]/k \
         k     =  (rBr+iBi)*(rBr+iBi) + (rBi-iBr)*(rBi-iBr)    */ \
      DVD_COMPLEX_RAYLEIGH_QUOTIENT((u)[(i)], (u)[(i)+1], (Ax)[(i)], \
        (Ax)[(i)+1], (Bx)[(i)], (Bx)[(i)+1], &(b)[8], &(b)[9], (b), (ierr)); \
      if (PetscAbsScalar((eigr)[(i_s)+(i)] - (b)[8])/ \
            PetscAbsScalar((eigr)[(i_s)+(i)]) > 1e-10    || \
          PetscAbsScalar((eigi)[(i_s)+(i)] - (b)[9])/ \
            PetscAbsScalar((eigi)[(i_s)+(i)]) > 1e-10         ) { \
        (ierr) = PetscInfo4((eps), "The eigenvalue %G+%G is far from its "\
                            "Rayleigh quotient value %G+%G\n", \
                            (eigr)[(i_s)+(i)], \
                            (eigi)[(i_s)+1], (b)[8], (b)[9]); \
      } \
      (i)++; \
    } \
  }
#else
#define DVD_COMPUTE_N_RR(eps,i,i_s,n,eigr,eigi,u,Ax,Bx,b,ierr) \
  for((i)=0; (i)<(n); (i)++) { \
      (ierr) = VecDot((Ax)[(i)], (u)[(i)], &(b)[0]); CHKERRQ(ierr); \
      (ierr) = VecDot((Bx)[(i)], (u)[(i)], &(b)[1]); CHKERRQ(ierr); \
      (b)[0] = (b)[0]/(b)[1]; \
      if (PetscAbsScalar((eigr)[(i_s)+(i)] - (b)[0])/ \
            PetscAbsScalar((eigr)[(i_s)+(i)]) > 1e-10     ) { \
        (ierr) = PetscInfo4((eps), "The eigenvalue %G+%G is far from its " \
               "Rayleigh quotient value %G+%G\n", \
               PetscRealPart((eigr)[(i_s)+(i)]), \
               PetscImaginaryPart((eigr)[(i_s)+(i)]), PetscRealPart((b)[0]), \
               PetscImaginaryPart((b)[0])); \
      } \
    }
#endif

#if !defined(PETSC_USE_COMPLEX)
#define DVD_NORM_FOR_UV(x) PetscAbsScalar(x)
#else
#define DVD_NORM_FOR_UV(x) (x)
#endif

#define DVD_NORMALIZE_UV(u,v,ierr,a) { \
    (ierr) = VecDot((u), (v), &(a)); CHKERRQ(ierr); \
    if ((a) == 0.0) { \
      SETERRQ(((PetscObject)u)->comm,1, "Inappropriate approximate eigenvector norm"); \
    } \
    if ((u) == (v)) { \
      ierr = VecScale((u), 1.0/PetscSqrtScalar(DVD_NORM_FOR_UV(a))); \
      CHKERRQ(ierr); \
    } else { \
      ierr = VecScale((u), 1.0/(a)); CHKERRQ(ierr); \
    } \
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_proj_uv_KBXZ"
/* 
  Compute: u <- K^{-1}*B*X, v <- (theta[0]*A+theta[1]*B)*X,
           kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1],
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
*/
PetscErrorCode dvd_improvex_jd_proj_uv_KBXZ(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     a, b[16];
  Vec             *Ax, *Bx, *r, *auxV = *auxV_, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;

  /* Book space for u, v and kr */
  *u = auxV; auxV+= n;
  *kr = auxV; auxV+= n;
  *v = auxV; auxV+= n;

  /* Ax = v, Bx = r = auxV */
  Ax = *v;
  r = Bx = auxV; auxV+= n;

  /* Ax <- A*X(i) */
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV, d->size_AV, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Bx <- B*X(i) */
  for(i=i_s; i<i_e; i++) d->nX[i] = 1.0;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV, d->size_BV, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
  } else {
    ierr = SlepcUpdateVectorsZ(d->B?*u:Bx, 0.0, 1.0, d->V, d->size_V, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
    if (d->B) {
      for(i=0; i<n; i++) {
        ierr = MatMult(d->B, (*u)[i], Bx[i]); CHKERRQ(ierr);
      }
    }
  }

  /* Recompute the eigenvalue */
  ierr = SlepcUpdateVectorsZ(*u, 0.0, 1.0, d->W?d->W:d->V, d->size_V, pY, ld,
                             d->size_H, n); CHKERRQ(ierr);
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, *u, Ax, Bx, b, ierr);

  /* u <- K^{-1} Bx */
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, Bx[i], (*u)[i]); CHKERRQ(ierr);
  }

  for(i=0; i<n; i++) {
    if (d->eigi[i_s+i] == 0.0) {
      /* [v r] <- [Ax Bx][theta_2i'     1 ]
                         [theta_2i+1  -eig] */
      b[0] = PetscConj(theta[i*2]); b[1] = theta[2*i+1];
      b[2] = 1.0; b[3] = -d->eigr[i_s+i];
      ierr = SlepcUpdateVectorsS(&(*v)[i], n, 0.0, 1.0, &(*v)[i], 2*n, n,
                                 b, 2, 2, 2); CHKERRQ(ierr);
    } else {
      /* [v_i v_i+1 r_i r_i+1]*= [tau_0' 0      1         0 
                                  0      tau_0' 0         1
                                  tau_1  0      -eigr_i -eigi_i
                                  0      tau_1  eigi_i  -eigr_i  ] */
      b[0] = b[5] = PetscConj(theta[2*i]);
      b[2] = b[7] = theta[2*i+1];
      b[8] = b[13] = 1.0;
      b[10] = b[15] = -d->eigr[i_s+i];
      b[14] = -(b[11] = d->eigi[i_s+i]);
      b[1] = b[3] = b[4] = b[6] = b[9] = b[12] = 0.0;
      X[0] = (*v)[i]; X[1] = (*v)[i+1]; X[2] = r[i]; X[3] = r[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 4, Z, size_Z);
      CHKERRQ(ierr);
      i++;
    }
  }

  /* kr <- K^{-1}*r */
  d->calcpairs_proj_res(d, i_s, i_e, r);
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, r[i], (*kr)[i]); CHKERRQ(ierr);
  }

  /* Free r */
  auxV-= n;

  /* Normalize the projector */
  for(i=0; i<n; i++) DVD_NORMALIZE_UV((*u)[i],(*v)[i],ierr,a);

  /* Return the next free vector */
  *auxV_ = auxV;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_proj_uv_KBXZY"
/* 
  Compute: u <- K^{-1}*B*X, v <- (theta[0]*A+theta[1]*B)*Y,
  kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1], Y <- W*pY[i_s..i_e-1]
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
*/
PetscErrorCode dvd_improvex_jd_proj_uv_KBXZY(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     a, b[16];
  Vec             *Ax, *Bx, *r, *auxV = *auxV_, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;

  /* Book space for u, v and kr */
  *u = auxV; auxV+= n;
  *kr = auxV; auxV+= n;
  *v = auxV; auxV+= n;
  r = auxV; auxV+= n;

  /* u <- Y(i) */
  ierr = SlepcUpdateVectorsZ(*u, 0.0, 1.0, d->W?d->W:d->V, d->size_V, pY, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* v <- theta[0]A*u + theta[1]*B*u */
  for(i=0; i<n; i++) {
    ierr = MatMult(d->A, (*u)[i], (*v)[i]); CHKERRQ(ierr);
  }
  if (d->B) {
    for(i=0; i<n; i++) {
      ierr = MatMult(d->B, (*u)[i], (*kr)[i]); CHKERRQ(ierr);
    }
    Bx = *kr;
  } else
    Bx = *u;

  for(i=0; i<n; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if(d->eigi[i_s+i] != 0.0) {
      /* [v_i v_i+1 Bx_i Bx_i+1]*= [ theta_2i'    0
                                       0         theta_2i'
                                     theta_2i+1 -thetai_i 
                                     thetai_i    theta_2i+1 ] */
      b[0] = b[5] = PetscConj(theta[2*i]);
      b[2] = b[7] = -theta[2*i+1];
      b[6] = -(b[3] = thetai[i]);
      b[1] = b[4] = 0.0;
      X[0] = (*v)[i]; X[1] = (*v)[i+1]; X[2] = Bx[i]; X[3] = Bx[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 2, Z, size_Z);
      CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      /* v_i <- v_i*theta_2i' + Bx_i*theta_2i+1 */
      ierr = VecAXPBY((*v)[i], theta[i*2+1], PetscConj(theta[i*2]), Bx[i]);
      CHKERRQ(ierr);
    }
  }

  /* Bx <- B*X(i) */
  Bx = *kr;
  for(i=i_s; i<i_e; i++) d->nX[i] = 1.0;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV, d->size_BV, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
  } else {
    ierr = SlepcUpdateVectorsZ(d->B?r:Bx, 0.0, 1.0, d->V, d->size_V, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
    if (d->B) {
      for(i=0; i<n; i++) {
        ierr = MatMult(d->B, r[i], Bx[i]); CHKERRQ(ierr);
      }
    }
  }

  /* Ax <- A*X(i) */
  Ax = r;
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV, d->size_AV, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, *u, Ax, Bx, b, ierr);

  /* u <- K^{-1} Bx */
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, Bx[i], (*u)[i]); CHKERRQ(ierr);
  }

  for(i=0; i<n; i++) {
    if (d->eigi[i_s+i] == 0.0) {
      /* r <- Ax -eig*Bx */
      ierr = VecAXPBY(r[i], -d->eigr[i_s+i], 1.0, Bx[i]); CHKERRQ(ierr);
    } else {
      /* [r_i r_i+1 kr_i kr_i+1]*= [   1        0 
                                       0        1
                                    -eigr_i -eigi_i
                                     eigi_i -eigr_i] */
      b[0] = b[5] = 1.0;
      b[2] = b[7] = -d->eigr[i_s+i];
      b[6] = -(b[3] = d->eigi[i_s+i]);
      b[1] = b[4] = 0.0;
      X[0] = r[i]; X[1] = r[i+1]; X[2] = (*kr)[i]; X[3] = (*kr)[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 2, Z, size_Z);
      CHKERRQ(ierr);
      i++;
    }
  }

  /* kr <- K^{-1}*r */
  d->calcpairs_proj_res(d, i_s, i_e, r);
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, r[i], (*kr)[i]); CHKERRQ(ierr);
  }

  /* Free r */
  auxV-= n;

  /* Normalize the projector */
  for(i=0; i<n; i++) DVD_NORMALIZE_UV((*u)[i],(*v)[i],ierr,a);

  /* Return the next free vector */
  *auxV_ = auxV;

  PetscFunctionReturn(0);
}


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
PetscErrorCode dvd_improvex_jd_proj_uv_KZX(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     a, b[16];
  Vec             *Ax, *Bx, *r, *auxV = *auxV_, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;

  /* Book space for u, v and kr */
  *u = auxV; auxV+= n;
  *kr = auxV; auxV+= n;
  *v = auxV; auxV+= n;
  r = auxV; auxV+= n;

  /* u <- X(i) */
  ierr = SlepcUpdateVectorsZ(*u, 0.0, 1.0, d->V, d->size_V, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);
  /* nX(i) <- ||X(i)|| */
  if (d->correctXnorm) {
    for (i=0; i<n; i++) {
      ierr = VecNormBegin((*u)[i], NORM_2, &d->nX[i_s+i]);CHKERRQ(ierr);
    }
    for (i=0; i<n; i++) {
      ierr = VecNormEnd((*u)[i], NORM_2, &d->nX[i_s+i]);CHKERRQ(ierr);
    }
  } else {
    for (i=0; i<n; i++) d->nX[i_s+i] = 1.0;
  }

  /* v <- theta[0]A*u + theta[1]*B*u */

  /* Bx <- B*X(i) */
  Bx = *kr;
  for(i=i_s; i<i_e; i++) d->nX[i] = 1.0;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV, d->size_BV, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
  } else {
    for(i=0; i<n; i++) {
      if (d->B) {
        ierr = MatMult(d->B, (*u)[i], Bx[i]); CHKERRQ(ierr);
      } else {
        ierr = VecCopy((*u)[i], Bx[i]); CHKERRQ(ierr);
      }
    }
  }

  /* Ax <- A*X(i) */
  Ax = r;
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV, d->size_AV, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* v <- Y(i) */
  ierr = SlepcUpdateVectorsZ(*v, 0.0, 1.0, d->W?d->W:d->V, d->size_V, pY, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, *v, Ax, Bx, b, ierr);

  for(i=0; i<n; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if(d->eigi[i_s+i] != 0.0) {
      /* [r_i r_i+1 kr_i kr_i+1]*= [ theta_2i'    0            1        0   
                                       0         theta_2i'     0        1   
                                     theta_2i+1 -thetai_i   -eigr_i -eigi_i 
                                     thetai_i    theta_2i+1  eigi_i -eigr_i ] */
      b[0] = b[5] = PetscConj(theta[2*i]);
      b[2] = b[7] = -theta[2*i+1];
      b[6] = -(b[3] = thetai[i]);
      b[1] = b[4] = 0.0;
      b[8] = b[13] = 1.0;
      b[10] = b[15] = -d->eigr[i_s+i];
      b[14] = -(b[11] = d->eigi[i_s+i]);
      b[9] = b[12] = 0.0;
      X[0] = Ax[i]; X[1] = Ax[i+1]; X[2] = Bx[i]; X[3] = Bx[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 4, Z, size_Z);
      CHKERRQ(ierr);
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
      ierr = SlepcUpdateVectorsD(X, 2, 1.0, b, 2, 2, 2, Z, size_Z);
      CHKERRQ(ierr);
    }
  }

  /* v <- K^{-1} r = K^{-1}(theta_2i'*Ax + theta_2i+1*Bx) */
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, r[i], (*v)[i]); CHKERRQ(ierr);
  }

  /* kr <- K^{-1}*kr = K^{-1}*(Ax - eig_i*Bx) */
  d->calcpairs_proj_res(d, i_s, i_e, Bx);
  for(i=0; i<n; i++) {
    ierr = VecCopy(Bx[i], r[i]); CHKERRQ(ierr);
    ierr = d->improvex_precond(d, i_s+i, r[i], (*kr)[i]); CHKERRQ(ierr);
  }

  /* Free r */
  auxV-= n;

  /* Normalize the projector */
  for(i=0; i<n; i++) DVD_NORMALIZE_UV((*v)[i],(*u)[i],ierr,a);

  /* Return the next free vector */
  *auxV_ = auxV;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_proj_uv_KBXX"
/* 
  Compute: u <- K^{-1}*B*X, v <- X,
  kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1]
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
*/
PetscErrorCode dvd_improvex_jd_proj_uv_KBXX(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     a, b[16];
  Vec             *Ax, *Bx, *r, *auxV = *auxV_, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;

  /* Book space for u, v and kr */
  *u = auxV; auxV+= n;
  *kr = auxV; auxV+= n;
  *v = auxV; auxV+= n;
  r = auxV; auxV+= n;


  /* [v u] <- X(i) Y(i) */
  ierr = SlepcUpdateVectorsZ(*v, 0.0, 1.0, d->V, d->size_V, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);
  ierr = SlepcUpdateVectorsZ(*u, 0.0, 1.0, d->W?d->W:d->V, d->size_V, pY, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Bx <- B*X(i) */
  Bx = *kr;
  for(i=i_s; i<i_e; i++) d->nX[i] = 1.0;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV, d->size_BV, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
  } else {
    if (d->B) {
      for(i=0; i<n; i++) {
        ierr = MatMult(d->B, (*v)[i], Bx[i]); CHKERRQ(ierr);
      }
    } else
      Bx = *v;
  }

  /* Ax <- A*X(i) */
  Ax = r;
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV, d->size_AV, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, *u, Ax, Bx, b, ierr);

  /* u <- K^{-1} Bx */
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, Bx[i], (*u)[i]); CHKERRQ(ierr);
  }

  for(i=0; i<n; i++) {
    if (d->eigi[i_s+i] == 0.0) {
      /* r <- Ax -eig*Bx */
      ierr = VecAXPBY(r[i], -d->eigr[i_s+i], 1.0, Bx[i]); CHKERRQ(ierr);
    } else {
      /* [r_i r_i+1 kr_i kr_i+1]*= [   1        0 
                                       0        1
                                    -eigr_i -eigi_i
                                     eigi_i -eigr_i] */
      b[0] = b[5] = 1.0;
      b[2] = b[7] = -d->eigr[i_s+i];
      b[6] = -(b[3] = d->eigi[i_s+i]);
      b[1] = b[4] = 0.0;
      X[0] = r[i]; X[1] = r[i+1]; X[2] = (*kr)[i]; X[3] = (*kr)[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 2, Z, size_Z);
      CHKERRQ(ierr);
      i++;
    }
  }

  /* kr <- K^{-1}*r */
  d->calcpairs_proj_res(d, i_s, i_e, r);
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, r[i], (*kr)[i]); CHKERRQ(ierr);
  }

  /* Free r */
  auxV-= n;

  /* Normalize the projector */
  for(i=0; i<n; i++) DVD_NORMALIZE_UV((*u)[i],(*v)[i],ierr,a);

  /* Return the next free vector */
  *auxV_ = auxV;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_proj_uv_KBXY"
/* 
  Compute: u <- K^{-1}*B*X, v <- Y,
  kr <- K^{-1}*(A-eig*B)*X, being X <- V*pX[i_s..i_e-1], Y <- V*pY[i_s..i_e-1]
  where
  auxV, 4*(i_e-i_s) auxiliar global vectors
  pX,pY, the right and left eigenvectors of the projected system
  ld, the leading dimension of pX and pY
*/
PetscErrorCode dvd_improvex_jd_proj_uv_KBXY(dvdDashboard *d, PetscInt i_s,
  PetscInt i_e, Vec **u, Vec **v, Vec **kr, Vec **auxV_, PetscScalar *theta,
  PetscScalar *thetai, PetscScalar *pX, PetscScalar *pY, PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s, i;
  PetscScalar     a, b[16];
  Vec             *Ax, *Bx, *r, *auxV = *auxV_, X[4];
  /* The memory manager doen't allow to call a subroutines */
  PetscScalar     Z[size_Z];

  PetscFunctionBegin;

  /* Book space for u, v and kr */
  *u = auxV; auxV+= n;
  *kr = auxV; auxV+= n;
  *v = auxV; auxV+= n;
  r = auxV; auxV+= n;


  /* v <- Y(i) */
  ierr = SlepcUpdateVectorsZ(*v, 0.0, 1.0, d->W?d->W:d->V, d->size_V, pY, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Bx <- B*X(i) */
  Bx = *kr;
  for(i=i_s; i<i_e; i++) d->nX[i] = 1.0;
  if (d->BV) {
    ierr = SlepcUpdateVectorsZ(Bx, 0.0, 1.0, d->BV, d->size_BV, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
  } else {
    ierr = SlepcUpdateVectorsZ(d->B?*u:Bx, 0.0, 1.0, d->V, d->size_V, pX, ld,
                               d->size_H, n); CHKERRQ(ierr);
    if (d->B) {
      for(i=0; i<n; i++) {
        ierr = MatMult(d->B, (*u)[i], Bx[i]); CHKERRQ(ierr);
      }
    }
  }

  /* Ax <- A*X(i) */
  Ax = r;
  ierr = SlepcUpdateVectorsZ(Ax, 0.0, 1.0, d->AV, d->size_AV, pX, ld,
                             d->size_H, n); CHKERRQ(ierr);

  /* Recompute the eigenvalue */
  DVD_COMPUTE_N_RR(d->eps, i, i_s, n, d->eigr, d->eigi, *v, Ax, Bx, b, ierr);

  /* u <- K^{-1} Bx */
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, Bx[i], (*u)[i]); CHKERRQ(ierr);
  }

  for(i=0; i<n; i++) {
    if (d->eigi[i_s+i] == 0.0) {
      /* r <- Ax -eig*Bx */
      ierr = VecAXPBY(r[i], -d->eigr[i_s+i], 1.0, Bx[i]); CHKERRQ(ierr);
    } else {
      /* [r_i r_i+1 kr_i kr_i+1]*= [   1        0 
                                       0        1
                                    -eigr_i -eigi_i
                                     eigi_i -eigr_i] */
      b[0] = b[5] = 1.0;
      b[2] = b[7] = -d->eigr[i_s+i];
      b[6] = -(b[3] = d->eigi[i_s+i]);
      b[1] = b[4] = 0.0;
      X[0] = r[i]; X[1] = r[i+1]; X[2] = (*kr)[i]; X[3] = (*kr)[i+1];
      ierr = SlepcUpdateVectorsD(X, 4, 1.0, b, 4, 4, 2, Z, size_Z);
      CHKERRQ(ierr);
      i++;
    }
  }

  /* kr <- K^{-1}*r */
  d->calcpairs_proj_res(d, i_s, i_e, r);
  for(i=0; i<n; i++) {
    ierr = d->improvex_precond(d, i_s+i, r[i], (*kr)[i]); CHKERRQ(ierr);
  }

  /* Free r */
  auxV-= n;

  /* Normalize the projector */
  for(i=0; i<n; i++) DVD_NORMALIZE_UV((*u)[i],(*v)[i],ierr,a);

  /* Return the next free vector */
  *auxV_ = auxV;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_jd_lit_const"
PetscErrorCode dvd_improvex_jd_lit_const(dvdDashboard *d, dvdBlackboard *b,
                                         PetscInt maxits, PetscReal tol,
                                         PetscReal fix)
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
PetscErrorCode dvd_improvex_jd_lit_const_0(dvdDashboard *d, PetscInt i,
  PetscScalar* theta, PetscScalar* thetai, PetscInt *maxits, PetscReal *tol)
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
  if(thetai) *thetai = 0.0;
#endif
  *maxits = data->maxits;
  *tol = data->tol;

  PetscFunctionReturn(0);
}


/**** Patterns implementation *************************************************/

typedef PetscInt (*funcV0_t)(dvdDashboard*, PetscInt, PetscInt, Vec*);
typedef PetscInt (*funcV1_t)(dvdDashboard*, PetscInt, PetscInt, Vec*,
                             PetscScalar*, Vec);

#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_PfuncV"
/* Compute D <- K^{-1} * funcV[r_s..r_e] */
PetscErrorCode dvd_improvex_PfuncV(dvdDashboard *d, void *funcV, Vec *D,
  PetscInt max_size_D, PetscInt r_s, PetscInt r_e, Vec *auxV,
  PetscScalar *auxS)
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
      ierr = d->improvex_precond(d, i+r_s, D[i+1], D[i]); CHKERRQ(ierr);
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
      ierr = d->improvex_precond(d, i+r_s, D[i+1], D[i]); CHKERRQ(ierr);
    }
    ierr = d->improvex_precond(d, r_e-1, auxV[0], D[r_e-r_s-1]); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,1, "Problem: r_e-r_s > max_size_D!");
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_improvex_get_eigenvectors"
/* Compute the left and right projected eigenvectors where,
   pX, the returned right eigenvectors
   pY, the returned left eigenvectors,
   ld_, the leading dimension of pX and pY,
   auxS, auxiliar vector of size length 6*d->size_H
*/
PetscErrorCode dvd_improvex_get_eigenvectors(dvdDashboard *d, PetscScalar *pX,
  PetscScalar *pY, PetscInt ld, PetscScalar *auxS, PetscInt size_auxS)
{
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;

  ierr = SlepcDenseCopy(pY, ld, d->T?d->pY:d->pX, d->ldpX, d->size_H,
                        d->size_H); CHKERRQ(ierr);
  ierr = SlepcDenseCopy(pX, ld, d->pX, d->ldpX, d->size_H, d->size_H);
  CHKERRQ(ierr);
  
  /* [qX, qY] <- eig(S, T); pX <- d->pX * qX; pY <- d->pY * qY */
  ierr = dvd_compute_eigenvectors(d->size_H, d->S, d->ldS, d->T, d->ldT, pX,
                                  ld, pY, ld, auxS, size_auxS, PETSC_TRUE);
  CHKERRQ(ierr); 

  /* 2-Normalize the columns of pX an pY */
  ierr = SlepcDenseNorm(pX, ld, d->size_H, d->size_H, d->eigi); CHKERRQ(ierr);
  ierr = SlepcDenseNorm(pY, ld, d->size_H, d->size_H, d->eigi); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
