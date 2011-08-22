/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with spectrum slicing for symmetric eigenproblems

   References:

       [1] R.G. Grimes et al., "A shifted block Lanczos algorithm for solving
           sparse symmetric generalized eigenproblems", SIAM J. Matrix Analysis
           and App., 15(1), pp. 228–272, 1994.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

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

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

extern PetscErrorCode EPSProjectedKSSym(EPS,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*);

/**/
PetscInt   allKs,def,deg,db;

/* Type of data characterizing a shift (place from where an eps is applied) */
typedef struct _n_shift *shift;
struct _n_shift{
  PetscReal	value;
  PetscInt	inertia;
  PetscBool	comp[2]; //
  shift  	neighb[2];//
  PetscInt	index;    //
  PetscInt	neigs;    //
  PetscReal     ext[2];   //
  PetscInt      nsch[2];  //
  PetscReal     pert;     //
  PetscBool     deg;  //not used
};

/* Type of data  for storing the state of spectrum slicing*/
struct _n_SR{
  PetscReal       int0,int1; // extrems of the interval
  PetscInt        dir; // determines the order of values in eig (+1 incr, -1 decr)
  PetscBool       hasEnd; // tells whether the interval has an end
  PetscInt        inertia0,inertia1;
  Vec             *V;
  PetscScalar     *eig;
  PetscInt        *perm;// permutation for keeping the eigenvalues in order
  PetscInt        numEigs; // number of eigenvalues in the interval
  PetscInt        indexEig;
  shift           sPres; // present shift 
  shift           *pending;//pending shifts array
  PetscInt        nPend;// number of pending shifts
  PetscInt        maxPend;// size of "pending" array
  Vec             *VDef; // vector for deflation
  PetscInt        *idxDef;//for deflation
  PetscInt        nMAXCompl;
  PetscInt        iterCompl;
  PetscInt        itsKs;
  PetscInt        nShifts;//number of computed shifts
  shift           s0;// initial shift
  PetscReal       tolDeg;
  PetscInt        nDeg;//number of values coinciding with a shift
  PetscInt        defMin; //minimum amount of values for deflation
};
typedef struct _n_SR  *SR;

/* 
   Fills the fields of a shift structure

*/
#undef __FUNCT__
#define __FUNCT__ "EPSCreateShift"
static PetscErrorCode EPSCreateShift(EPS eps,PetscScalar val, shift neighb0,shift neighb1)
{
  PetscErrorCode   ierr;
  shift            s,*pending2;
  PetscInt         i;
  SR               sr;

  PetscFunctionBegin;
  sr = (SR)(eps->data);
  ierr = PetscMalloc(sizeof(struct _n_shift),&s);CHKERRQ(ierr);
  s->value = val;
  s->neighb[0] = neighb0;
  if(neighb0) neighb0->neighb[1] = s;
  s->neighb[1] = neighb1;
  if(neighb1) neighb1->neighb[0] = s;
  s->comp[0] = PETSC_FALSE;
  s->comp[1] = PETSC_FALSE;
  s->index = -1;
  s->neigs = 0;
  s->deg = PETSC_FALSE;
  s->nsch[0] = s->nsch[1]=0;
  // inserts in the stack of pending shifts
  // if needed, the array is resized
  if(sr->nPend >= sr->maxPend){
    if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD,"resizing pending shifts array\n");CHKERRQ(ierr);}
    sr->maxPend *= 2;
    ierr = PetscMalloc((sr->maxPend)*sizeof(shift),&pending2);CHKERRQ(ierr);
    for(i=0;i < sr->nPend; i++)pending2[i] = sr->pending[i];
    ierr = PetscFree(sr->pending);CHKERRQ(ierr);
    sr->pending = pending2;
  }
  sr->pending[sr->nPend++]=s;
  sr->nShifts++;
  PetscFunctionReturn(0);
}

/* Provides next shift to be computed */
#undef __FUNCT__
#define __FUNCT__ "EPSExtractShift"
static PetscErrorCode EPSExtractShift(EPS eps){
  PetscErrorCode   ierr;
  PetscInt         iner;
  Mat              F;
  PC               pc;
  KSP              ksp;
  SR               sr;

  PetscFunctionBegin;  
  sr = (SR)(eps->data);
  if(sr->nPend > 0){
    sr->sPres = sr->pending[--sr->nPend];
    ierr = STSetShift(eps->OP, sr->sPres->value);CHKERRQ(ierr);
    ierr = STGetKSP(eps->OP, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    ierr = MatGetInertia(F,&iner,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    sr->sPres->inertia = iner;
    eps->target = sr->sPres->value;
    eps->nconv = 0;
    eps->reason = EPS_CONVERGED_ITERATING;
    eps->its =0;
   }else sr->sPres = PETSC_NULL;
   PetscFunctionReturn(0);
}
  
/*
   Symmetric KrylovSchur adapted to spectrum slicing:
   allows searching an specific amount of eigenvalues in the subintervals left and right.
   returns whether the search has succeed
*/
#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchur_Slice"
static PetscErrorCode EPSKrylovSchur_Slice(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,conv,k,l,lds,lt,nv,m,*iwork,p,j;
  Vec            u=eps->work[0];
  PetscScalar    *Q;
  PetscReal      *a,*b,*work,beta,*Qreal,rtmp;//
  PetscBool      breakdown;
  PetscInt       count0,count1;//nconv_prev;
  PetscReal      theta,lambda;
  shift          sPres;
  PetscBool      complIterating;/* shows whether iterations are made for completion */
  PetscBool       sch0,sch1;//shows whether values are looked after on each side
  PetscInt        iterCompl,n0,n1;
  //PetscReal       res;

  SR             sr;

  PetscFunctionBegin;
  /* spectrum slicing data */
  sr = (SR)eps->data;
  sPres = sr->sPres;
  complIterating =PETSC_FALSE;
  sch1 = sch0 = PETSC_TRUE;
  lds = PetscMin(eps->mpd,eps->ncv);
  ierr = PetscMalloc(lds*lds*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(2*lds*sizeof(PetscInt),&iwork);CHKERRQ(ierr);
  lt = PetscMin(eps->nev+eps->mpd,eps->ncv);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&b);CHKERRQ(ierr);
  count0=0;count1=0; // found on both sides
  
  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    m = PetscMin(eps->nconv+eps->mpd,eps->ncv);

    ierr = EPSFullLanczos(eps,a+l,b+l,eps->V,eps->nconv+l,&m,u,&breakdown);CHKERRQ(ierr);
    nv = m - eps->nconv;
    beta = b[nv-1];

    /* Solve projected problem and compute residual norm estimates */
    ierr = EPSProjectedKSSym(eps,nv,l,a,b,eps->eigr+eps->nconv,Q,work,iwork);CHKERRQ(ierr);
    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,eps->nconv,nv,PETSC_NULL,nv,Q,eps->V+eps->nconv,nv,beta,1.0,&k,PETSC_NULL);CHKERRQ(ierr);
    if(allKs ==1){//option for accepting all converging values
      Qreal = (PetscReal*)Q;//
      conv=k=j=0;
      for(i=0;i<nv;i++)if(eps->errest[eps->nconv+i] < eps->tol)conv++;
      for(i=0;i<nv;i++){
        if(eps->errest[eps->nconv+i] < eps->tol){
          iwork[j++]=i;
        }else iwork[conv+k++]=i;
      }
      for(i=0;i<nv;i++)a[i]=eps->eigr[eps->nconv+i];
      for(i=0;i<nv;i++){
        eps->eigr[eps->nconv+i] = a[iwork[i]];
      }
      for( i=0;i<nv;i++){
        p=iwork[i];
        if(p!=i){
          j=i+1;
          while(iwork[j]!=i)j++;
          iwork[j]=p;iwork[i]=i;
          for(k=0;k<nv;k++){
            rtmp=Qreal[k+p*nv];Qreal[k+p*nv]=Qreal[k+i*nv];Qreal[k+i*nv]=rtmp;
            //rtmp=Q[k+p*nv];Q[k+p*nv]=Q[k+i*nv];Q[k+i*nv]=rtmp; 
          }
        } 
      }
      k=eps->nconv+conv;
    }
  /*checking proximity to an eigenvalue*/

  if(deg==1){
    for(i=0;i < k; i++){
      theta = PetscRealPart(eps->eigr[i]);
      if(PetscAbsReal(theta*sPres->value*eps->tol*10)>1){
        if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD,"DEGENERATED SHIFT\n");CHKERRQ(ierr);}
        sr->nDeg++;
        sPres->deg = PETSC_TRUE;
      }else break;
    }
  }
  /*
  if(deg == 1 && sr->nDeg > 0){
    eps->reason = EPS_CONVERGED_TOL;
  }else{
  */
    /* Checking values obtained for completing */
    count0=count1=0;
    for(i=0;i<k;i++){      
      theta = PetscRealPart(eps->eigr[i]);
      lambda = sPres->value + 1/theta;
      if( ((sr->dir)*theta < 0) && ((sr->dir)*(lambda - sPres->ext[0]) > 0))count0++;
      if( ((sr->dir)*theta > 0) && ((sr->dir)*(sPres->ext[1] - lambda) > 0))count1++;
    }
    
    /* checks completion */
    if( (!sch0||count0 >= sPres->nsch[0]) && (!sch1 ||count1 >= sPres->nsch[1]) ) {
      eps->reason = EPS_CONVERGED_TOL;
    }else {
      if(!complIterating && eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
      if(complIterating){
        if(--iterCompl <= 0) eps->reason = EPS_DIVERGED_ITS;
      }else if (k >= eps->nev) {//eps->reason = EPS_CONVERGED_TOL;
        n0 = sPres->nsch[0]-count0;
        n1 = sPres->nsch[1]-count1;
        if( sr->iterCompl>0 && ( (n0>0&& n0<= sr->nMAXCompl)||(n1>0&&n1<=sr->nMAXCompl) )){
          complIterating = PETSC_TRUE;
          if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD,"iterating for completion nMAXComp=%d\n",sr->nMAXCompl);CHKERRQ(ierr);}
          if(n0 >sr->nMAXCompl)sch0 = PETSC_FALSE;
          if(n1 >sr->nMAXCompl)sch1 = PETSC_FALSE;
          iterCompl = sr->iterCompl;
        }else eps->reason = EPS_CONVERGED_TOL;
      }      
    }
  
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (eps->nconv+nv-k)/2;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          PetscInfo(eps,"Unable to generate more start vectors\n");
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=0;i<l;i++) {
          a[i] = PetscRealPart(eps->eigr[i+k]);
          b[i] = PetscRealPart(Q[nv-1+(i+k-eps->nconv)*nv]*beta);
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv+eps->nconv);CHKERRQ(ierr);
    //nconv_prev = eps->nconv;//
    eps->nconv = k;
  }
  /* check for completion */
  sPres->comp[0] = (count0 >= sPres->nsch[0])?PETSC_TRUE:PETSC_FALSE;
  sPres->comp[1] = (count1 >= sPres->nsch[1])?PETSC_TRUE:PETSC_FALSE;
  if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD," found count0=%d(of %d) and count1=%d(of %d)\n",count0,sPres->nsch[0],count1,sPres->nsch[1]);CHKERRQ(ierr);}
  sr->itsKs += eps->its;  

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
  obtains value of subsequent shift
*/
#undef __FUNCT__
#define __FUNCT__ "EPSGetNewShiftValue"
static PetscErrorCode EPSGetNewShiftValue(EPS eps,PetscInt side,PetscReal *newS){
  PetscReal   lambda,d_prev;//pert,d,d_avg;
  PetscInt    i,idxP;
  SR          sr;
  shift       sPres;

  PetscFunctionBegin;  
  sr = (SR)eps->data;
  sPres = sr->sPres;
/*  
  pert = 0;
  if(sPres->neigs >0){
      idxP=0;//number of computed eigenvalues previous to sPres->value
      for(i=sPres->index;i< sPres->index + sPres->neigs;i++){
        lambda = PetscRealPart(sr->eig[i]);
        if((sr->dir)*(lambda - sPres->value) < 0)idxP++;
        else break;
      }
      //middle point between shift and previous/posterior value
      pert = PetscAbs(sr->eig[sPres->index+idxP]- sr->sPres->value)/2;
  }
*/
  if( sPres->neighb[side]){
  /* completing a previous interval */
    *newS=(sPres->value + sPres->neighb[side]->value)/2;
    
  }else{ //(only for side=1). creating a new interval.
    if(sPres->neigs==0){// no value has been accepted
      if(sPres->neighb[0]){
        // multiplying by 10 the previous distance
        *newS = sPres->value + 10*(sr->dir)*PetscAbsReal(sPres->value - sPres->neighb[0]->value);
      }else {//first shift
        if(eps->nconv != 0){
           //unaccepted values give information for next shift
           idxP=0;//number of values left from shift 
           for(i=0;i<eps->nconv;i++){
             lambda = PetscRealPart(eps->eigr[i]);
             if( (sr->dir)*(lambda - sPres->value) <0)idxP++;
             else break;
           }
           //avoiding substraction of eigenvalues (might be the same).
           if(idxP>0){
             d_prev = PetscAbsReal(sPres->value - PetscRealPart(eps->eigr[0]))/(idxP+0.3);
           }else {
             d_prev = PetscAbsReal(sPres->value - PetscRealPart(eps->eigr[eps->nconv-1]))/(eps->nconv+0.3);
           }
           *newS = sPres->value + ((sr->dir)*d_prev*eps->nev)/2;
        }else{//no values found, no information for next shift
          // changing the end of the interval
        }
      }
    }else{// accepted values found
      //average distance of values in previous subinterval
      shift s = sPres->neighb[0];
      while(s && PetscAbs(s->inertia - sPres->inertia)==0){
        s = s->neighb[0];//looking for previous shifts with eigenvalues within
      }
      if(s){
        d_prev = PetscAbsReal( (sPres->value - s->value)/(sPres->inertia - s->inertia));
      }else{//firts shift. average distance obtained with values in this shift        
        d_prev = PetscAbsReal( PetscRealPart(sr->eig[sPres->index+sPres->neigs-1]) - sPres->value)/(sPres->neigs+0.3); 
      }
      // average distance is used for next shift by adding it to value on the rigth or to shift
      if( (sr->dir)*(PetscRealPart(sr->eig[sPres->index + sPres->neigs -1]) - sPres->value) >0){
        *newS = PetscRealPart(sr->eig[sPres->index + sPres->neigs -1])+ ((sr->dir)*d_prev*(eps->nev))/2;   
      }else{//last accepted value is on the left of shift. adding to shift.
        *newS = sPres->value + ((sr->dir)*d_prev*(eps->nev))/2;
      }
    //}
    }
    //end of interval can not be surpassed
    if(sr->hasEnd && ((sr->dir)*(*newS - sr->int1) > 0))*newS=sr->int1;
  }//of neighb[side]==null
  PetscFunctionReturn(0);
}

/* 
  function for sorting an array of real values
*/
#undef __FUNCT__
#define __FUNCT__ "sortRealEigenvalues"
static PetscErrorCode sortRealEigenvalues(PetscScalar *r,PetscInt *perm,PetscInt nr,PetscBool prev,PetscInt dir)
{
  PetscReal      re;
  PetscInt       i,j,tmp;
  
  PetscFunctionBegin; 
  if(!prev) for (i=0; i < nr; i++) { perm[i] = i; }
  /* insertion sort */
  for (i=1; i < nr; i++) {
    re = PetscRealPart(r[perm[i]]);
    j = i-1;
    while ( j>=0 && dir*(re - PetscRealPart(r[perm[j]])) <= 0 ) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
    }
  }
  PetscFunctionReturn(0);
}

/* Stores the pairs obtained since the last shift in the global arrays */
#undef __FUNCT__
#define __FUNCT__ "EPSStoreEigenpairs"
PetscErrorCode EPSStoreEigenpairs(EPS eps)
{
  PetscErrorCode ierr;
  PetscReal      lambda,error;
  PetscInt       i,count;
  PetscBool      cond;
  SR             sr;
  shift          sPres;

  PetscFunctionBegin; 
  sr = (SR)(eps->data);
  sPres = sr->sPres;
  sPres->index = sr->indexEig;
  count = sr->indexEig;
  error=0;
  /* backtransform */
  for(i=0;i < eps->nconv; i++) eps->eigr[i] =  sPres->value + 1.0/(eps->eigr[i]);
  /* sort eigenvalues */
  ierr = sortRealEigenvalues(eps->eigr,eps->perm,eps->nconv,PETSC_FALSE,sr->dir);
  /* values stored in global array */
  // condition for avoiding comparing whith a non-existing end.
  cond = (!sPres->neighb[1] && !sr->hasEnd)?PETSC_TRUE:PETSC_FALSE; 
  for( i=0; i < eps->nconv ;i++ ){
    lambda = PetscRealPart(eps->eigr[eps->perm[i]]);
    if(db>1){ierr = EPSComputeRelativeError(eps,eps->perm[i],&error);CHKERRQ(ierr);}
    if( ( ((sr->dir)*(lambda - sPres->ext[0]) > 0) && ( cond || ((sr->dir)*(lambda - sPres->ext[1]) < 0)) ) ){
      if(count>=sr->numEigs){//Error found
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of reserved values exceeded  lambda=%.14g\n",lambda);
        break;
      }   
      sr->eig[count] = lambda;
      ierr = VecCopy(eps->V[eps->perm[i]], sr->V[count]);CHKERRQ(ierr);
      if(db>1){ierr = PetscPrintf(PETSC_COMM_WORLD,"i=%d perm[i]=%d lambda=%.14g error=%g indexEig=%d\n",i,eps->perm[i],lambda,error,count);CHKERRQ(ierr);}
      count++;
    }else if(db>1){ierr = PetscPrintf(PETSC_COMM_WORLD,"i=%d perm[i]=%d lambda=%.14g NOT VALID\n",i,eps->perm[i],lambda);CHKERRQ(ierr);}
  }
  sPres->neigs = count - sr->indexEig;
  if(db>=1){PetscPrintf(PETSC_COMM_WORLD," stored between %d and %d\n",sr->indexEig,count);CHKERRQ(ierr);}
  sr->indexEig = count;
  
  ierr = sortRealEigenvalues(sr->eig,sr->perm,count,PETSC_TRUE,sr->dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLookForDeflation"
PetscErrorCode EPSLookForDeflation(EPS eps)
{
  PetscErrorCode  ierr;
  PetscReal       val,lambda,lambda2;
  PetscInt        i,count0=0,count1=0;
  shift           sPres;
  PetscInt        ini,fin,defMin,k,idx0,idx1;
  PetscBool       consec;
  SR              sr;

  PetscFunctionBegin; 
  sr = (SR)(eps->data);
  sPres = sr->sPres;
  defMin = sr->defMin;

  if(sPres->neighb[0]) ini = (sr->dir)*(sPres->neighb[0]->inertia - sr->inertia0);
  else ini = 0;
  fin = sr->indexEig;
  // selection of ends for searching new values
  // later modified with version def=0
  if(!sPres->neighb[0]) sPres->ext[0] = sr->int0;//first shift
  else sPres->ext[0] = sPres->neighb[0]->value;
  if(!sPres->neighb[1]) {
    if(sr->hasEnd) sPres->ext[1] = sr->int1;
    else sPres->ext[1] = (sr->dir > 0)?PETSC_MAX_REAL:PETSC_MIN_REAL;
  }else sPres->ext[1] = sPres->neighb[1]->value;
  if(db>1){ierr = PetscPrintf(PETSC_COMM_WORLD,"ext0=%g ext1=%g\n",sPres->ext[0],sPres->ext[1]);CHKERRQ(ierr);}
  //selection of values between right and left ends
  for(i=ini;i<fin;i++){
    val=PetscRealPart(sr->eig[sr->perm[i]]);
    //values to the right of left shift
    if( (sr->dir)*(val - sPres->ext[1]) < 0 ){
      if((sr->dir)*(val - sPres->value) < 0)count0++;
      else count1++;
    }else break;
  }
  // the number of values on each side are found
  if(sPres->neighb[0])
     sPres->nsch[0] = (sr->dir)*(sPres->inertia - sPres->neighb[0]->inertia)-count0;
  else sPres->nsch[0] = 0;

  if(sPres->neighb[1])
    sPres->nsch[1] = (sr->dir)*(sPres->neighb[1]->inertia - sPres->inertia) - count1;
  else sPres->nsch[1] = (sr->dir)*(sr->inertia1 - sPres->inertia);
  
  //completing vector of indexes for deflation
  if(def==0 && !sPres->neighb[1]){//new interval && no deflation
    if(db>1){ierr = PetscPrintf(PETSC_COMM_WORLD,"def=0 y neig1=null\n");CHKERRQ(ierr);}
    k=0;
    for(i=fin-1;i>ini;i--){
      k++;
      lambda = PetscRealPart(sr->eig[sr->perm[i]]);
      lambda2 = PetscRealPart(sr->eig[sr->perm[i-1]]);
      if( PetscAbsReal((lambda - lambda2)/lambda) > sr->tolDeg){//relative tolerance 
        break;
      }
    }
    // if i!=ini values for i and i-1 more than toldeg apart
    if(db>1){ierr = PetscPrintf(PETSC_COMM_WORLD,"lookDef i=%d ini=%d\n",i,ini);CHKERRQ(ierr);}
    if(i<=ini){
      sPres->ext[0] = sPres->value;
    }else{//middle point
       sPres->ext[0] = (PetscRealPart(sr->eig[sr->perm[i]])+PetscRealPart(sr->eig[sr->perm[i-1]]))/2;         
    }
    idx0=ini+count0-k;
    idx1=ini+count0;
    if(db>1){ierr = PetscPrintf(PETSC_COMM_WORLD,"ext0=%g ext1=%g idx0=%d idx1=%d count0=%d k=%d\n",sPres->ext[0],sPres->ext[1],idx0,idx1,count0,k);CHKERRQ(ierr);}
  }else{  //completing a subinterval or without deflation
    k = PetscMax(0,defMin-count0);
    idx0 = PetscMax(0,ini-k);
    if(def==0 && sPres->nsch[0]==0){//no deflation towards 0
      idx0 = ini + count0;
      sPres->ext[0] = sPres->value;
    }
    k = PetscMax(0,defMin-count1);
    idx1 = PetscMin(sr->indexEig,ini+count0+count1+k);
    if(def==0 && sPres->nsch[1]==0){//no deflation towards 1
      idx1 = ini + count0;
      sPres->ext[1] = sPres->value;
    }
  }
  k=0;
  for(i=idx0;i<idx1;i++)sr->idxDef[k++]=sr->perm[i];
   ///// info
  if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD," deflated %d in [0] and %d in [1]",count0,count1);CHKERRQ(ierr);}
    /////  

  // check for consecutives
  consec=PETSC_TRUE;
  for(i=1;i<k;i++)if(sr->idxDef[i]!=sr->idxDef[i-1]+1){consec = PETSC_FALSE; break;}
  // if not consecutives, copied in array
//if(consec){
//  V o which 
//else{
  for(i=0;i<k;i++)sr->VDef[i]=sr->V[sr->idxDef[i]];
  eps->DS = sr->VDef;
//}
  eps->nds = k;
  //////info
  if(db>=1){
    if(consec){ ierr = PetscPrintf(PETSC_COMM_WORLD," (%d consecutive values)\n",k);CHKERRQ(ierr);}
    else{ ierr = PetscPrintf(PETSC_COMM_WORLD," (%d non consecutive values)\n",k);CHKERRQ(ierr);}
  }
 
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Slice"
PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt 	 imax,jmax;
  PetscReal      newS;
  KSP            ksp;
  PC             pc;
  Mat            B,F;  
  PetscScalar    *eigi;
  Vec            t,w;
  SR             sr;
  PetscReal      orthMax;
  PetscScalar    inerd;
  double         t1,t2;
 
  PetscFunctionBegin;
  eps->trackall = PETSC_TRUE;
  allKs = 0;
  def = 1;
  deg=0;
  db = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-db",&db,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-deg",&deg,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-def",&def,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-allKs",&allKs,PETSC_NULL);CHKERRQ(ierr);
  if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD,"Options: allKs=%d, def=%d, deg=%d \n",allKs,def,deg);CHKERRQ(ierr);}
  ierr = PetscMalloc(sizeof(struct _n_SR),&sr);CHKERRQ(ierr);
  eps->data = sr;
  sr->tolDeg = sqrt(eps->tol);//default
  ierr = PetscOptionsGetReal(PETSC_NULL,"-toldeg",&sr->tolDeg,PETSC_NULL);CHKERRQ(ierr);
  if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD,"toldeg=%g\n",sr->tolDeg);CHKERRQ(ierr);}
  sr->defMin = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-defMin",&sr->defMin,PETSC_NULL);CHKERRQ(ierr);
  if(def==0)sr->defMin =0;
  //checking presence of ends and finding direction
  if( eps->inta > PETSC_MIN_REAL){
    sr->int0 = eps->inta;
    sr->int1 = eps->intb;
    sr->dir = 1;
    if(eps->intb >= PETSC_MAX_REAL){ /* right-open interval */
      sr->hasEnd = PETSC_FALSE;
      sr->inertia1 = eps->n;
    }else sr->hasEnd = PETSC_TRUE;
  }else{ /* left-open interval */
    sr->int0 = eps->intb;
    sr->int1 = eps->inta;
    sr->dir = -1;
    sr->hasEnd = PETSC_FALSE;
    sr->inertia1 = 0;
  }
  if(db>=1){ierr = PetscPrintf(PETSC_COMM_WORLD,"dir=%d int0=%g\n",sr->dir,sr->int0);CHKERRQ(ierr);}
  sr->nMAXCompl = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-comp",&sr->nMAXCompl,PETSC_NULL);
  sr->iterCompl = sr->nMAXCompl+5;//=======
  //i = PetscMin(eps->mpd,eps->ncv);//=======
  //ierr = PetscMalloc(i*sizeof(PetscReal),&sr->aprox);CHKERRQ(ierr);//======
  // array of pending shifts
  sr->maxPend = 100;//initial size;
  ierr = PetscMalloc((sr->maxPend)*sizeof(shift),&sr->pending);CHKERRQ(ierr);
  if(sr->hasEnd){
    ierr = STGetKSP(eps->OP, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    /* not looking for values in b (just inertia).*/
    ierr = MatGetInertia(F,&sr->inertia1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  sr->nShifts = 0;
  sr->nPend = 0;
  ierr = EPSCreateShift(eps,sr->int0,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSExtractShift(eps);
  sr->s0 = sr->sPres;
  sr->inertia0 = sr->s0->inertia;
  sr->numEigs = (sr->dir)*(sr->inertia1 - sr->inertia0);
  sr->indexEig = 0;
  sr->itsKs = 0;
  sr->nDeg = 0;
  if(db>=1){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dir=%d inertia in 0(=%g) %d and in 1(=%g) %d\n",sr->dir,sr->int0,sr->inertia0,sr->int1,sr->inertia1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"numEigs=%d\n\n",sr->numEigs);
  }
  /* only whith eigenvalues present in the interval ...*/
  if(sr->numEigs==0){ 
    eps->reason = EPS_CONVERGED_TOL;
    PetscFunctionReturn(0);
  }

  /* memory reservation for eig, V and perm */
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscScalar),&sr->eig);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscScalar),&eigi);CHKERRQ(ierr);
  for(i=0;i<sr->numEigs;i++)eigi[i]=0;
  ierr = VecCreateMPI(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,&t);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(t,sr->numEigs,&sr->V);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  // vector for maintaining order of eigenvalues
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscInt),&sr->perm);CHKERRQ(ierr);
  for(i=0;i< sr->numEigs;i++)sr->perm[i]=i;
  // vectors for deflation
  ierr = PetscMalloc((sr->numEigs+sr->defMin)*sizeof(PetscInt),&sr->idxDef);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs)*sizeof(Vec),&sr->VDef);CHKERRQ(ierr);
  sr->indexEig = 0;

  t1 = MPI_Wtime();
  while(sr->sPres){

    //////////info
    if(db>=1){
      if(sr->sPres->neighb[1]){ierr = PetscPrintf(PETSC_COMM_WORLD,"Completing ");CHKERRQ(ierr);}
      else {ierr = PetscPrintf(PETSC_COMM_WORLD,"New ");CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_WORLD,"shift: %.14g (s0=",sr->sPres->value);CHKERRQ(ierr);
      if (sr->sPres->neighb[0]){ierr = PetscPrintf(PETSC_COMM_WORLD,"%g",sr->sPres->neighb[0]->value);CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_WORLD," s1=");CHKERRQ(ierr);
      if (sr->sPres->neighb[1]){ierr = PetscPrintf(PETSC_COMM_WORLD,"%g",sr->sPres->neighb[1]->value);CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_WORLD,")\n");CHKERRQ(ierr);
    }
    ///////////

    /* search for deflation */
    ierr = EPSLookForDeflation(eps);CHKERRQ(ierr);
    /* krylovSchur */
    ierr = EPSKrylovSchur_Slice(eps);CHKERRQ(ierr);
    ierr = EPSStoreEigenpairs(eps);CHKERRQ(ierr);
    /* select new shift */
    if(!sr->sPres->comp[1]){
      ierr = EPSGetNewShiftValue(eps,1,&newS);CHKERRQ(ierr);
      ierr = EPSCreateShift(eps,newS,sr->sPres,sr->sPres->neighb[1]);
    }
    if(!sr->sPres->comp[0]){
      // completing earlier interval
      ierr = EPSGetNewShiftValue(eps,0,&newS);CHKERRQ(ierr);
      ierr = EPSCreateShift(eps,newS,sr->sPres->neighb[0],sr->sPres);
    }
    /* preparing for a new search of values */
    ierr = EPSExtractShift(eps);CHKERRQ(ierr);
  }
  t2 = MPI_Wtime();
  /* checking orthogonality */
  if(db>=1){
    ierr = STGetOperators(eps->OP,PETSC_NULL,&B);CHKERRQ(ierr);
    orthMax=0;
    imax=jmax=-1;
    ierr = VecDuplicate(sr->V[0],&w);CHKERRQ(ierr);
    for(i=0;i < sr->indexEig; i++){
      ierr = MatMult(B,sr->V[i],w);CHKERRQ(ierr);
      for(j=0;j < sr->indexEig;j++){
        if(i != j) {
          ierr = VecDot(w,sr->V[j],&inerd);CHKERRQ(ierr);
          if(PetscRealPart(inerd)>orthMax){orthMax = PetscRealPart(inerd); imax = i; jmax = j;}
        }
      }
    }
    ierr = VecDestroy(&w);CHKERRQ(ierr);    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nStored indexEig=%d (of %d)\n (orthog max %g in i=%d j=%d)\n\n",sr->indexEig,sr->numEigs,orthMax,imax,jmax);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," time %g\n",t2-t1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," number of shifts: %d\n",sr->nShifts);CHKERRQ(ierr);
  }
  /* updating eps values prior to exit */
  //ierr = EPSFreeSolution(eps);
  ierr = VecDestroyVecs(eps->allocated_ncv,&eps->V);CHKERRQ(ierr);
  eps->V = sr->V;
  ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
  ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
  eps->eigr = sr->eig;
  eps->eigi = eigi;
  eps->its = sr->itsKs;
  eps->ncv = eps->allocated_ncv = sr->numEigs;
  ierr = PetscFree(eps->errest);CHKERRQ(ierr);
  ierr = PetscFree(eps->errest_left);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest_left);CHKERRQ(ierr);
  ierr = PetscFree(eps->perm);CHKERRQ(ierr);
  eps->perm = sr->perm;
  eps->nconv = sr->indexEig;
  eps->reason = EPS_CONVERGED_TOL;
  eps->nds = 0;
  eps->DS = PETSC_NULL;
  ierr = PetscFree(sr->VDef);CHKERRQ(ierr);
  ierr = PetscFree(sr->idxDef);CHKERRQ(ierr);
  ierr = PetscFree(sr->pending);CHKERRQ(ierr);
  // reviewing list of shifts to free memmory
  shift s = sr->s0;
  if(s){
    while(s->neighb[1]){
      s = s->neighb[1];
      ierr = PetscFree(s->neighb[0]);CHKERRQ(ierr);
    }
    ierr = PetscFree(s);CHKERRQ(ierr);
  }
  ierr = PetscFree(sr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

