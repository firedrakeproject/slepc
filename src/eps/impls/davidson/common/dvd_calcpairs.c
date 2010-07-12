/*
  SLEPc eigensolver: "davidson"

  Step: calc the best eigenpairs in the subspace V.

  For that, performs these steps:
    1) Update W <- A * V
    2) Update H <- V' * W
    3) Obtain eigenpairs of H
    4) Select some eigenpairs
    5) Compute the Ritz pairs of the selected ones
*/

#include "slepc.h"
#include "private/epsimpl.h"
#include "davidson.h"
#include "slepcblaslapack.h"

PetscInt dvd_calcpairs_proj_qz(dvdDashboard *d);
PetscInt dvd_calcpairs_proj_qz_harm(dvdDashboard *d);
PetscInt dvd_calcpairs_updateV(dvdDashboard *d);
PetscInt dvd_calcpairs_updateW(dvdDashboard *d);
PetscInt dvd_calcpairs_updateAV(dvdDashboard *d);
PetscInt dvd_calcpairs_updateBV(dvdDashboard *d);
PetscInt dvd_calcpairs_VtAV_gen(dvdDashboard *d, DvdReduction *r,
                                DvdMult_copy_func *sr);
PetscInt dvd_calcpairs_VtBV_gen(dvdDashboard *d, DvdReduction *r,
                                DvdMult_copy_func *sr);
PetscInt dvd_calcpairs_projeig_qz_std(dvdDashboard *d);
PetscInt dvd_calcpairs_projeig_qz_gen(dvdDashboard *d);
PetscInt dvd_calcpairs_selectPairs_qz(dvdDashboard *d, PetscInt n);
PetscInt dvd_calcpairs_X(dvdDashboard *d, PetscInt r_s, PetscInt r_e, Vec *X);
PetscInt dvd_calcpairs_Y(dvdDashboard *d, PetscInt r_s, PetscInt r_e, Vec *Y);
PetscInt dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                             Vec *R, PetscScalar *auxS, Vec auxV);
PetscInt dvd_calcpairs_proj_res(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                                Vec *R);
PetscInt dvd_calcpairs_updateMatV(Mat A, Vec **AV, PetscInt *size_AV,
                                  dvdDashboard *d);
PetscInt dvd_calcpairs_WtMatV_gen(PetscScalar **H, MatType_t sH, PetscInt ldH,
                                  PetscInt *size_H, PetscScalar *MTY,
                                  PetscInt ldMTY, PetscScalar *MTX,
                                  PetscInt ldMTX, PetscInt rMT, PetscInt cMT, 
                                  Vec *W, Vec *V, PetscInt size_V,
                                  PetscScalar *auxS, DvdReduction *r,
                                  DvdMult_copy_func *sr, dvdDashboard *d);

/**** Control routines ********************************************************/
#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz"
PetscInt dvd_calcpairs_qz(dvdDashboard *d, dvdBlackboard *b, IP ipI)
{
  PetscTruth      std_probl, her_probl;
  PetscInt        i;

  PetscFunctionBegin;

  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

  /* Setting configuration constrains */
#ifndef PETSC_USE_COMPLEX
  /* if the last converged eigenvalue is complex its conjugate pair is also
     converged */
  b->max_nev = PetscMax(b->max_nev, d->nev+1);
#else
  b->max_nev = PetscMax(b->max_nev, d->nev);
#endif
  b->own_vecs+= b->max_size_V*(d->B?2:1) /* AV, BV? */;
  b->own_scalars+= b->max_size_V*b->max_size_V*2*(std_probl?1:2);
                                              /* H, G?, S, T? */
  b->own_scalars+= b->max_size_V*b->max_size_V*(std_probl?1:2);
                                              /* pX, pY? */
  b->own_scalars+= b->max_nev*b->max_nev*(std_probl?1:2); /* cS, cT? */
  b->max_size_auxS = PetscMax(PetscMax(
                              b->max_size_auxS,
                              b->max_size_V*b->max_size_V*4
                                                      /* SlepcReduction */ ),
                              std_probl?0:(b->max_size_V*11+16) /* projeig */);

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    d->size_AV = 0;
    d->real_AV = d->AV = b->free_vecs; b->free_vecs+= b->max_size_V;
    d->size_H = 0;
    d->H = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    d->real_H = d->H;
    d->ldH = b->max_size_V;
    d->pX = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    d->S = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    d->cS = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
    for (i=0; i<b->max_nev*b->max_nev; i++) d->cS[i] = 0.0;
    d->ldcS = b->max_nev;
    d->ipV = ipI;
    d->ipW = ipI;
    d->size_cX = d->size_cY = 0;
    d->cY = PETSC_NULL;
    d->pY = PETSC_NULL;
    d->T = PETSC_NULL;
    d->ldcT = PETSC_NULL;
    d->cT = 0;
    if (d->B) {
      d->size_BV = 0;
      d->real_BV = d->BV = b->free_vecs; b->free_vecs+= b->max_size_V;
    }
    if (!std_probl) {
      d->size_G = 0;
      d->G = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      d->real_G = d->G;
      d->T = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      d->cT = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      for (i=0; i<b->max_nev*b->max_nev; i++) d->cT[i] = 0.0;
      d->ldcT = b->max_nev;
      d->pY = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      /* If the problem is GHEP without B-orthonormalization, active BcX */
      if(her_probl) d->BcX = d->AV;

      /* Else, active the left and right converged invariant subspaces */
      else d->cY = d->AV;
    }

    d->calcPairs = d->W?dvd_calcpairs_proj_qz_harm:dvd_calcpairs_proj_qz;
    d->calcpairs_residual = dvd_calcpairs_res_0;
    d->calcpairs_proj_res = dvd_calcpairs_proj_res;
    d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_qz;
    d->calcpairs_X = dvd_calcpairs_X;
    d->calcpairs_Y = dvd_calcpairs_Y;
    d->ipI = ipI;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj_qz"
PetscInt dvd_calcpairs_proj_qz(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  DvdReduction    r;
  DvdReductionChunk
                  ops[2];
  DvdMult_copy_func
                  sr[2];
  PetscInt        size_in = 2*d->size_V*d->size_V;
  PetscScalar     *in = d->auxS, *out = in+size_in;

  PetscFunctionBegin;

  /* Prepare reductions */
  ierr = SlepcAllReduceSumBegin(ops, 2, in, out, size_in, &r,
                                ((PetscObject)d->V[0])->comm); CHKERRQ(ierr);

  /* Update AV, BV and the projected matrices */
  dvd_calcpairs_updateV(d);
  dvd_calcpairs_updateAV(d);
  dvd_calcpairs_VtAV_gen(d, &r, &sr[0]);
  if (d->BV) dvd_calcpairs_updateBV(d);
  if (DVD_ISNOT(d->sEP, DVD_EP_STD)) dvd_calcpairs_VtBV_gen(d, &r, &sr[1]);

  /* Do reductions */
  ierr = SlepcAllReduceSumEnd(&r); CHKERRQ(ierr);

  if (d->MT_type != DVD_MT_IDENTITY) {
    d->MT_type = DVD_MT_IDENTITY;
//    d->pX_type|= DVD_MAT_IDENTITY;
    d->V_tra_s = d->V_tra_e = 0;
  }
  d->pX_type = 0;
//  if(d->V_new_e - d->V_new_s > 0) {
    if (DVD_IS(d->sEP, DVD_EP_STD))     dvd_calcpairs_projeig_qz_std(d);
    else                                dvd_calcpairs_projeig_qz_gen(d);
//  }
  d->V_new_s = d->V_new_e;

  /* Check consistency */
  if ((d->size_V != d->V_new_e) || (d->size_V != d->size_H) ||
      (d->size_V != d->size_AV) || (DVD_ISNOT(d->sEP, DVD_EP_STD) && (
      (d->size_V != d->size_G) || (d->size_V != d->size_BV) ))) {
    SETERRQ(1, "Consistency broken!");
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj_qz_harm"
PetscInt dvd_calcpairs_proj_qz_harm(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  DvdReduction    r;
  DvdReductionChunk
                  ops[2];
  DvdMult_copy_func
                  sr[2];
  PetscInt        size_in = 2*d->size_V*d->size_V;
  PetscScalar     *in = d->auxS, *out = in+size_in;

  PetscFunctionBegin;

  /* Prepare reductions */
  ierr = SlepcAllReduceSumBegin(ops, 2, in, out, size_in, &r,
                                ((PetscObject)d->V[0])->comm); CHKERRQ(ierr);

  /* Update AV, BV and the projected matrices */
  dvd_calcpairs_updateV(d);
  dvd_calcpairs_updateAV(d);
  if (d->BV) dvd_calcpairs_updateBV(d);
  dvd_calcpairs_updateW(d);
  dvd_calcpairs_VtAV_gen(d, &r, &sr[0]);
  if (DVD_ISNOT(d->sEP, DVD_EP_STD)) dvd_calcpairs_VtBV_gen(d, &r, &sr[1]);

  /* Do reductions */
  ierr = SlepcAllReduceSumEnd(&r); CHKERRQ(ierr);

  /* Perform the transformation on the projected problem */
  d->calcpairs_proj_trans(d);

  if (d->MT_type != DVD_MT_IDENTITY) {
    d->MT_type = DVD_MT_IDENTITY;
//    d->pX_type|= DVD_MAT_IDENTITY;
    d->V_tra_s = d->V_tra_e = 0;
  }
  d->pX_type = 0;
//  if(d->V_new_e - d->V_new_s > 0) {
    if (DVD_IS(d->sEP, DVD_EP_STD))     dvd_calcpairs_projeig_qz_std(d);
    else                                dvd_calcpairs_projeig_qz_gen(d);
//  }
  d->V_new_s = d->V_new_e;

  PetscFunctionReturn(0);
}

/**** Basic routines **********************************************************/

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateV"
PetscInt dvd_calcpairs_updateV(dvdDashboard *d)
{
  PetscInt        r;
  Vec             *cX = d->BcX? d->BcX : ( (d->cY && !d->W)? d->cY : d->cX );

  PetscFunctionBegin;

  /* V <- gs([cX f.V(0:f.V_new_s-1)], f.V(V_new_s:V_new_e-1)) */
  r = dvd_orthV(d->ipV, d->eps->DS, d->eps->nds, cX, d->size_cX, d->V,
                d->V_new_s, d->V_new_e, d->auxS, d->auxV[0], d->rand);

  PetscFunctionReturn(r);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateW"
PetscInt dvd_calcpairs_updateW(dvdDashboard *d)
{
  PetscInt        r;

  PetscFunctionBegin;

  /* Update W */
  r = d->calcpairs_W(d);

  /* W <- gs([cY f.W(0:f.V_new_s-1)], f.W(V_new_s:V_new_e-1)) */
  r = dvd_orthV(d->ipW, PETSC_NULL, 0, d->cY, d->size_cY, d->W, d->V_new_s,
                d->V_new_e, d->auxS, d->auxV[0], d->rand);

  PetscFunctionReturn(r);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateAV"
PetscInt dvd_calcpairs_updateAV(dvdDashboard *d)
{
  PetscInt        r;

  PetscFunctionBegin;

  /* f.AV(f.V_tra) = f.AV * f.MT; f.AV(f.V_new) = A*f.V(f.V_new) */
  r = dvd_calcpairs_updateMatV(d->A, &d->AV, &d->size_AV, d);

  PetscFunctionReturn(r);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV"
PetscInt dvd_calcpairs_updateBV(dvdDashboard *d)
{
  PetscInt        r;

  PetscFunctionBegin;

  /* f.BV(f.V_tra) = f.BV * f.MT; f.BV(f.V_new) = B*f.V(f.V_new) */
  r = dvd_calcpairs_updateMatV(d->B, &d->BV, &d->size_BV, d);

  PetscFunctionReturn(r);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_VtAV_gen"
PetscInt dvd_calcpairs_VtAV_gen(dvdDashboard *d, DvdReduction *r,
                                DvdMult_copy_func *sr)
{
  PetscInt        i,
                  ldMTY = d->MTY?d->ldMTY:d->ldMTX;
  /* WARNING: auxS uses space assigned to r */
  PetscScalar     *auxS = r->out,
                  *MTY = d->MTY?d->MTY:d->MTX;
  Vec             *W = d->W?d->W:d->V;

  PetscFunctionBegin;

  /* f.H = [f.H(f.V_imm,f.V_imm)        f.V(f.V_imm)'*f.AV(f.V_new);
            f.V(f.V_new)'*f.AV(f.V_imm) f.V(f.V_new)'*f.AV(f.V_new) ] */
  if (DVD_IS(d->sA,DVD_MAT_HERMITIAN))
    d->sH = DVD_MAT_HERMITIAN | DVD_MAT_IMPLICIT | DVD_MAT_UTRIANG;
  if ((d->V_imm_e - d->V_imm_s == 0) && (d->V_tra_e - d->V_tra_s == 0))
    d->size_H = 0;
  i = dvd_calcpairs_WtMatV_gen(&d->H, d->sH, d->ldH, &d->size_H,
                                  &MTY[ldMTY*d->V_tra_s], ldMTY,
                               &d->MTX[d->ldMTX*d->V_tra_s], d->ldMTX,
                               d->size_MT, d->V_tra_e-d->V_tra_s,
                               W, d->AV, d->size_V,
                               auxS, r, sr, d);

  PetscFunctionReturn(i);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_VtBV_gen"
PetscInt dvd_calcpairs_VtBV_gen(dvdDashboard *d, DvdReduction *r,
                                DvdMult_copy_func *sr)
{
  PetscInt        i,
                  ldMTY = d->MTY?d->ldMTY:d->ldMTX;
  /* WARNING: auxS uses space assigned to r */
  PetscScalar     *auxS = r->out,
                  *MTY = d->MTY?d->MTY:d->MTX;
  Vec             *W = d->W?d->W:d->V;

  PetscFunctionBegin;

  /* f.G = [f.G(f.V_imm,f.V_imm)        f.V(f.V_imm)'*f.BV(f.V_new);
            f.V(f.V_new)'*f.BV(f.V_imm) f.V(f.V_new)'*f.BV(f.V_new) ] */
  if (DVD_IS(d->sB,DVD_MAT_HERMITIAN))
    d->sG = DVD_MAT_HERMITIAN | DVD_MAT_IMPLICIT | DVD_MAT_UTRIANG;
  if ((d->V_imm_e - d->V_imm_s == 0) && (d->V_tra_e - d->V_tra_s == 0))
    d->size_G = 0;
  i = dvd_calcpairs_WtMatV_gen(&d->G, d->sG, d->ldH, &d->size_G,
                                  &MTY[ldMTY*d->V_tra_s], ldMTY,
                               &d->MTX[d->ldMTX*d->V_tra_s], d->ldMTX,
                               d->size_MT, d->V_tra_e-d->V_tra_s,
                               W, d->BV?d->BV:d->V, d->size_V,
                               auxS, r, sr, d);

  PetscFunctionReturn(i);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_qz_std"
PetscInt dvd_calcpairs_projeig_qz_std(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* S <- H */
  d->ldS = d->ldpX = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, 0, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);

  /* S = pX' * H * pX */
  ierr = EPSDenseHessenberg(d->size_H, 0, d->S, d->ldS, d->pX);
  CHKERRQ(ierr);
  ierr = EPSDenseSchur(d->size_H, 0, d->S, d->ldS, d->pX, d->eigr, d->eigi);
  CHKERRQ(ierr);

  d->pX_type = (d->pX_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;

  PetscFunctionReturn(0);
}

/*
  auxS(dgges) = size_H (beta) + 8*size_H+16 (work)
  auxS(zgges) = size_H (beta) + 1+2*size_H (work) + 8*size_H (rwork)
*/
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_qz_gen"
PetscInt dvd_calcpairs_projeig_qz_gen(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscScalar     *beta = d->auxS;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar     *auxS = beta + d->size_H;
  PetscBLASInt    n_auxS = d->size_auxS - d->size_H;
#else
  PetscReal       *auxR = (PetscReal*)(beta + d->size_H);
  PetscScalar     *auxS = (PetscScalar*)(auxR+8*d->size_H);
  PetscBLASInt    n_auxS = d->size_auxS - 9*d->size_H;
#endif
  PetscInt        i;
  PetscBLASInt    info,n,a;

  PetscFunctionBegin;

  /* S <- H, T <- G */
  d->ldS = d->ldT = d->ldpX = d->ldpY = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, 0, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);
  ierr = SlepcDenseCopyTriang(d->T, 0, d->size_H, d->G, d->sG, d->ldH,
                              d->size_H, d->size_H);

  /* S = Z'*H*Q, T = Z'*G*Q */
  n = d->size_H;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgges_(d->pY?"V":"N", "V", "N", PETSC_NULL, &n, d->S, &n, d->T, &n,
              &a, d->eigr, d->eigi, beta, d->pY, &n, d->pX, &n,
              auxS, &n_auxS, PETSC_NULL, &info);
#else
  LAPACKgges_(d->pY?"V":"N", "V", "N", PETSC_NULL, &n, d->S, &n, d->T, &n,
              &a, d->eigr, beta, d->pY, &n, d->pX, &n,
              auxS, &n_auxS, auxR, PETSC_NULL, &info);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB, "Error in Lapack GGES %d", info);

  /* eigr[i] <- eigr[i] / beta[i] */
  for (i=0; i<d->size_H; i++)
    d->eigr[i] /= beta[i],
    d->eigi[i] /= beta[i];

  d->pX_type = (d->pX_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;
  d->pY_type = (d->pY_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_selectPairs_qz"
PetscInt dvd_calcpairs_selectPairs_qz(dvdDashboard *d, PetscInt n)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  if ((d->ldpX != d->size_H) ||
      ( d->T &&
        ((d->ldS != d->ldT) || (d->ldpX != d->ldpY) ||
         (d->ldpX != d->size_H)) ) ) {
     SETERRQ(1, "Error before ordering eigenpairs!");
  }

  if (d->T) {
    ierr = EPSSortDenseSchurGeneralized(d->eps, d->size_H, 0, d->S, d->T,
                                        d->ldS, d->pY, d->pX, d->eigr,
                                        d->eigi); CHKERRQ(ierr);
  } else {
    ierr = EPSSortDenseSchur(d->eps, d->size_H, 0, d->S, d->ldS, d->pX,
                             d->eigr, d->eigi); CHKERRQ(ierr);
  }

  if (d->calcpairs_eigs_trans) d->calcpairs_eigs_trans(d);

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_X"
PetscInt dvd_calcpairs_X(dvdDashboard *d, PetscInt r_s, PetscInt r_e, Vec *X)
{
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* X = V * U(0:n-1) */
  if (DVD_IS(d->pX_type,DVD_MAT_IDENTITY)) {
    if (d->V != X) for(i=r_s; i<r_e; i++) {
      ierr = VecCopy(d->V[i], X[i]); CHKERRQ(ierr);
    }
  } else {
    ierr = SlepcUpdateVectorsZ(X, 0.0, 1.0, d->V, d->size_H, &d->pX[d->ldpX*r_s],
                               d->ldpX, d->size_H, r_e-r_s); CHKERRQ(ierr);
  }

  /* nX[i] <- ||X[i]|| */
  if (d->correctXnorm == PETSC_TRUE) for(i=0; i<r_e-r_s; i++) {
    ierr = VecNorm(X[i], NORM_2, &d->nX[r_s+i]); CHKERRQ(ierr);
  } else for(i=0; i<r_e-r_s; i++) {
    d->nX[r_s+i] = 1.0;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_Y"
PetscInt dvd_calcpairs_Y(dvdDashboard *d, PetscInt r_s, PetscInt r_e, Vec *Y)
{
  PetscInt        i, ldpX = d->pY?d->ldpY:d->ldpX;
  PetscErrorCode  ierr;
  Vec             *V = d->W?d->W:d->V;
  PetscScalar     *pX = d->pY?d->pY:d->pX;

  PetscFunctionBegin;

  /* Y = V * pX(0:n-1) */
  if (DVD_IS(d->pX_type,DVD_MAT_IDENTITY)) {
    if (V != Y) for(i=r_s; i<r_e; i++) {
      ierr = VecCopy(V[i], Y[i]); CHKERRQ(ierr);
    }
  } else {
    ierr = SlepcUpdateVectorsZ(Y, 0.0, 1.0, V, d->size_H, &pX[ldpX*r_s], ldpX,
                               d->size_H, r_e-r_s); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* Compute the residual vectors R(i) <- (AV - BV*eigr(i))*pX(i), and also
   the norm, where
   i <- r_s..r_e,
   UL, auxiliar scalar matrix of size size_H*(r_e-r_s),
   auxV, auxiliar global vector.
*/
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_res_0"
PetscInt dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                             Vec *R, PetscScalar *UL, Vec auxV)
{
  PetscInt        i, j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* If the eigenproblem is not reduced to standard */
  if ((d->B == PETSC_NULL) || DVD_ISNOT(d->sEP, DVD_EP_STD)) {
    /* UL = f.U(0:n-1) * diag(f.pL(0:n-1)) */
    for(i=r_s; i<r_e; i++) for(j=0; j<d->size_H; j++)
      UL[d->size_H*(i-r_s)+j] = d->pX[d->ldpX*i+j]*d->eigr[i];

    if (d->B == PETSC_NULL) {
      /* R <- V * UL */
      ierr = SlepcUpdateVectorsZ(R, 0.0, 1.0, d->V, d->size_V, UL, d->size_H,
                                 d->size_H, r_e-r_s); CHKERRQ(ierr);
    } else {
      /* R <- BV * UL */
      ierr = SlepcUpdateVectorsZ(R, 0.0, 1.0, d->BV, d->size_BV, UL,
                                 d->size_H, d->size_H, r_e-r_s);
      CHKERRQ(ierr);
    }
    /* R <- AV*U - R */
    ierr = SlepcUpdateVectorsZ(R, -1.0, 1.0, d->AV, d->size_AV,
                               &d->pX[d->ldpX*r_s], d->ldpX, d->size_H, r_e-r_s);
    CHKERRQ(ierr);

  /* If the problem was reduced to standard, R[i] = B*X[i] */
  } else {
    /* R[i] <- R[i] * eigr[i] */
    for(i=r_s; i<r_e; i++) {
      ierr = VecScale(R[i-r_s], d->eigr[i]); CHKERRQ(ierr); 
    }
      
    /* R <- AV*U - R */
    ierr = SlepcUpdateVectorsZ(R, -1.0, 1.0, d->AV, d->size_AV,
                               &d->pX[d->ldpX*r_s], d->ldpX, d->size_H, r_e-r_s);
    CHKERRQ(ierr);
  }

  d->calcpairs_proj_res(d, r_s, r_e, R);

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj_res"
PetscInt dvd_calcpairs_proj_res(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                                Vec *R)
{
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscTruth      lindep;
  Vec             *cX;

  PetscFunctionBegin;

  /* If exists the BcX, R <- orth(BcX, R), nR[i] <- ||R[i]|| */
  if (d->BcX)
    cX = d->BcX;

  /* If exists left subspace, R <- orth(cY, R), nR[i] <- ||R[i]|| */
  else if (d->cY)
    cX = d->cY;

  /* If fany configurations, R <- orth(cX, R), nR[i] <- ||R[i]|| */
  else if (!(DVD_IS(d->sEP, DVD_EP_STD) && DVD_IS(d->sEP, DVD_EP_HERMITIAN)))
    cX = d->cX;

  /* Otherwise, nR[i] <- ||R[i]|| */
  else
    cX = PETSC_NULL;

  if (cX) for (i=0; i<r_e-r_s; i++) {
    ierr = IPOrthogonalize(d->ipI, 0, PETSC_NULL, d->size_cX, PETSC_NULL,
                           cX, R[i], PETSC_NULL, &d->nR[r_s+i], &lindep);
    CHKERRQ(ierr);
    if((lindep == PETSC_TRUE) || (d->nR[r_s+i] < PETSC_MACHINE_EPSILON)) {
        SETERRQ(1, "Error during the residual computation of the eigenvectors!");
    }

  } else for(i=0; i<r_e-r_s; i++) {
    ierr = VecNorm(R[i], NORM_2, &d->nR[r_s+i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/**** Patterns implementation *************************************************/
#undef __FUNCT__ 
#define __FUNCT__ "calcPairs_updateMatV"
PetscInt dvd_calcpairs_updateMatV(Mat A, Vec **AV, PetscInt *size_AV,
                                  dvdDashboard *d)
{
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* f.AV((0:f.V_tra.size)+f.imm.s) = f.AV * f.U(f.V_tra) */
  if (d->MT_type == DVD_MT_pX) {
    ierr = SlepcUpdateVectorsZ(*AV+d->V_imm_e, 0.0, 1.0, *AV, *size_AV,
                               &d->pX[d->ldpX*d->V_tra_s], d->ldpX,
                               *size_AV, d->V_tra_e-d->V_tra_s); CHKERRQ(ierr);
  } else if (d->MT_type == DVD_MT_ORTHO) {
    ierr = SlepcUpdateVectorsZ(*AV+d->V_imm_e, 0.0, 1.0, *AV, *size_AV,
                               &d->MTX[d->ldMTX*d->V_tra_s], d->ldMTX,
                               *size_AV, d->V_tra_e-d->V_tra_s); CHKERRQ(ierr);
  }
  *AV = *AV+d->V_imm_s;

  /* f.AV(f.V_new) = A*f.V(f.V_new) */
  if (d->V_imm_e-d->V_imm_s + d->V_tra_e-d->V_tra_s != d->V_new_s) {
    SETERRQ(1, "d->V_imm_e-d->V_imm_s + d->V_tra_e-d->V_tra_s != d->V_new_s !");
    PetscFunctionReturn(1);
  }

  for (i=d->V_new_s; i<d->V_new_e; i++) {
    ierr = MatMult(A, d->V[i], (*AV)[i]); CHKERRQ(ierr);
  }
  *size_AV = d->V_new_e;

  PetscFunctionReturn(0);
}

/*
  Compute f.H = [MTY'*H*MTX     W(tra)'*V(new);
                 W(new)'*V(tra) W(new)'*V(new) ]
  where
  tra = 0:cMT-1,
  new = cMT:size_V-1,
  ldH, the leading dimension of H,
  auxS, auxiliary scalar vector of size ldH*max(tra,size_V),
  */
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_WtMatV_gen"
PetscInt dvd_calcpairs_WtMatV_gen(PetscScalar **H, MatType_t sH, PetscInt ldH,
                                  PetscInt *size_H, PetscScalar *MTY,
                                  PetscInt ldMTY, PetscScalar *MTX,
                                  PetscInt ldMTX, PetscInt rMT, PetscInt cMT, 
                                  Vec *W, Vec *V, PetscInt size_V,
                                  PetscScalar *auxS, DvdReduction *r,
                                  DvdMult_copy_func *sr, dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* H <- MTY^T * (H * MTX) */
  if (cMT > 0) {
    ierr = SlepcDenseMatProdTriang(auxS, 0, ldH,
                                   *H, sH, ldH, *size_H, *size_H, PETSC_FALSE,
                                   MTX, 0, ldMTX, rMT, cMT, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = SlepcDenseMatProdTriang(*H, sH, ldH,
                                   MTY, 0, ldMTY, rMT, cMT, PETSC_TRUE,
                                   auxS, 0, ldH, *size_H, cMT, PETSC_FALSE);
    CHKERRQ(ierr);
    *size_H = cMT;
  }

  /* H = [H              W(tra)'*W(new);
          W(new)'*V(tra) W(new)'*V(new) ] */
  ierr = VecsMultS(*H, sH, ldH, W, *size_H, size_V, V, *size_H, size_V, r, sr);
  CHKERRQ(ierr);
  *size_H = size_V;

  PetscFunctionReturn(0);
}
