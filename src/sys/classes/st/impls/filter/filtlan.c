/*
   This file is an adaptation of several subroutines from FILTLAN, the
   Filtered Lanczos Package, authored by Haw-ren Fang and Yousef Saad.

   More information at:
   http://www-users.cs.umn.edu/~saad/software/filtlan

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/stimpl.h>
#include "./filter.h"

static PetscErrorCode FILTLAN_FilteredConjugateResidualPolynomial(Mat,Mat,PetscReal*,PetscReal*,PetscInt);
static PetscReal FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(Mat,PetscReal*,PetscInt,PetscReal);
static PetscErrorCode FILTLAN_ExpandNewtonPolynomialInChebyshevBasis(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

/* ////////////////////////////////////////////////////////////////////////////
   //    Newton - Hermite Polynomial Interpolation
   //////////////////////////////////////////////////////////////////////////// */

/*
   FILTLAN function NewtonPolynomial

   build P(z) by Newton's divided differences in the form
       P(z) = a(1) + a(2)*(z-x(1)) + a(3)*(z-x(1))*(z-x(2)) + ... + a(n)*(z-x(1))*...*(z-x(n-1)),
   such that P(x(i)) = y(i) for i=1,...,n, where
       x,y are input vectors of length n, and a is the output vector of length n
   if x(i)==x(j) for some i!=j, then it is assumed that the derivative of P(z) is to be zero at x(i),
       and the Hermite polynomial interpolation is applied
   in general, if there are k x(i)'s with the same value x0, then
       the j-th order derivative of P(z) is zero at z=x0 for j=1,...,k-1
*/
static PetscErrorCode FILTLAN_NewtonPolynomial(PetscInt n,PetscReal *x,PetscReal *y,PetscReal *sa)
{
  PetscErrorCode ierr;
  PetscReal      d,*sx=x,*sy=y,*sf;
  PetscInt       j,k;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&sf);CHKERRQ(ierr);
  ierr = PetscMemcpy(sf,sy,n*sizeof(PetscReal));CHKERRQ(ierr);

  /* apply Newton's finite difference method */
  sa[0] = sf[0];
  for (j=1;j<n;j++) {
    for (k=n-1;k>=j;k--) {
      d = sx[k]-sx[k-j];
      if (d == 0.0) sf[k] = 0.0;  /* assume that the derivative is 0.0 and apply the Hermite interpolation */
      else sf[k] = (sf[k]-sf[k-1]) / d;
    }
    sa[j] = sf[j];
  }

  ierr = PetscFree(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
/*
   FILTLAN function NewtonPolynomialEvaluation

   return the evaluated P(z0), i.e. the value of P(z) at z=z0, where P(z) is a Newton polynomial defined by
      P(z) = a(1) + a(2)*(z-x(1)) + a(3)*(z-x(1))*(z-x(2)) + ... + a(n)*(z-x(1))*...*(z-x(n-1))
   this routine also works for evaluating the function value of a Hermite interpolating polynomial,
      which is in the same form as a Newton polynomial
*/
static PetscReal FILTLAN_NewtonPolynomialEvaluation(PetscInt n,PetscReal *a,PetscReal *x,const PetscReal z0)
{
  PetscReal fval,*sa,*sx;
  PetscInt  jj;

  PetscFunctionBegin;
  sa = a+n-1;
  sx = x+n-2;
  jj = n-1;
  fval = *sa--;
  while (jj--) {
    fval *= (z0-(*sx--));
    fval += (*sa--);
  }
  PetscFunctionReturn(fval);
}
#endif

/*
   FILTLAN function HermiteBaseFilterInChebyshevBasis

   compute a base filter P(z) which is a continuous, piecewise polynomial P(z) expanded
   in a basis of `translated' (i.e. scale-and-shift) Chebyshev polynomials in each interval

   The base filter P(z) equals P_j(z) for z in the j-th interval [intv(j), intv(j+1)), where
   P_j(z) a Hermite interpolating polynomial

   input:
   intv is a vector which defines the intervals; the j-th interval is [intv(j), intv(j+1))
   HiLowFlags determines the shape of the base filter P(z)
   Consider the j-th interval [intv(j), intv(j+1)]
   HighLowFlag[j-1]==1,  P(z)==1 for z in [intv(j), intv(j+1)]
                   ==0,  P(z)==0 for z in [intv(j), intv(j+1)]
                   ==-1, [intv(j), intv(j+1)] is a transition interval;
                         P(intv(j)) and P(intv(j+1)) are defined such that P(z) is continuous
   baseDeg is the degree of smoothness of the Hermite (piecewise polynomial) interpolation
   to be precise, the i-th derivative of P(z) is zero, i.e. d^{i}P(z)/dz^i==0, at all interval
   end points z=intv(j) for i=1,...,baseDeg

   output:
   P(z) expanded in a basis of `translated' (scale-and-shift) Chebyshev polynomials
   to be precise, for z in the j-th interval [intv(j),intv(j+1)), P(z) equals
       P_j(z) = pp(1,j)*S_0(z) + pp(2,j)*S_1(z) + ... + pp(n,j)*S_{n-1}(z),
   where S_i(z) is the `translated' Chebyshev polynomial in that interval,
       S_i((z-c)/h) = T_i(z),  c = (intv(j)+intv(j+1))) / 2,  h = (intv(j+1)-intv(j)) / 2,
   with T_i(z) the Chebyshev polynomial of the first kind,
       T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2
   the return matrix is the matrix of Chebyshev coefficients pp just described

   note that the degree of P(z) in each interval is (at most) 2*baseDeg+1, with 2*baseDeg+2 coefficients
   let n be the length of intv; then there are n-1 intervals
   therefore the return matrix pp is of size (2*baseDeg+2)-by-(n-1)
*/
static PetscErrorCode FILTLAN_HermiteBaseFilterInChebyshevBasis(Mat baseFilter,PetscReal *intv,PetscInt npoints,const PetscInt *HighLowFlags,PetscInt baseDeg)
{
  PetscErrorCode ierr;
  PetscInt       n,m,ii,jj;
  PetscReal      flag,flag0,flag2,aa,bb,*px,*py,*sx,*sy,*pp,*qq,*sq,*currentPoint = intv;
  PetscScalar    *bf,*sbf;
  const PetscInt *hilo = HighLowFlags;

  PetscFunctionBegin;
  ierr = MatGetSize(baseFilter,&m,&n);CHKERRQ(ierr);
  if (m!=2*baseDeg+2 || n!=npoints-1) SETERRQ(PetscObjectComm((PetscObject)baseFilter),1,"Wrong dimensions");
  jj = npoints-1;  /* jj is initialized as the number of intervals */
  ierr = PetscMalloc4(m,&px,m,&py,m,&pp,m,&qq);CHKERRQ(ierr);
  ierr = MatDenseGetArray(baseFilter,&bf);CHKERRQ(ierr);
  sbf = bf;

  while (jj--) {  /* the main loop to compute the Chebyshev coefficients */

    flag = (PetscReal)(*hilo++);  /* get the flag of the current interval */
    if (flag == -1.0) {  /* flag == -1 means that the current interval is a transition polynomial */

      flag2 = (PetscReal)(*hilo);  /* get flag2, the flag of the next interval */
      flag0 = 1.0-flag2;       /* the flag of the previous interval is 1-flag2 */

      /* two pointers for traversing x[] and y[] */
      sx = px;
      sy = py;

      /* find the current interval [aa,bb] */
      aa = *currentPoint++;
      bb = *currentPoint;

      /* now left-hand side */
      ii = baseDeg+1;
      while (ii--) {
        *sy++ = flag0;
        *sx++ = aa;
      }

      /* now right-hand side */
      ii = baseDeg+1;
      while (ii--) {
        *sy++ = flag2;
        *sx++ = bb;
      }

      /* build a Newton polynomial (indeed, the generalized Hermite interpolating polynomial) with x[] and y[] */
      ierr = FILTLAN_NewtonPolynomial(m,px,py,pp);CHKERRQ(ierr);

      /* pp contains coefficients of the Newton polynomial P(z) in the current interval [aa,bb], where
         P(z) = pp(1) + pp(2)*(z-px(1)) + pp(3)*(z-px(1))*(z-px(2)) + ... + pp(n)*(z-px(1))*...*(z-px(n-1)) */

      /* translate the Newton coefficients to the Chebyshev coefficients */
      ierr = FILTLAN_ExpandNewtonPolynomialInChebyshevBasis(m,aa,bb,pp,px,qq);CHKERRQ(ierr);
      /* qq contains coefficients of the polynomial in [aa,bb] in the `translated' Chebyshev basis */

      /* copy the Chebyshev coefficients to baseFilter
         OCTAVE/MATLAB: B(:,j) = qq, where j = (npoints-1)-jj and B is the return matrix */
      sq = qq;
      ii = 2*baseDeg+2;
      while (ii--) *sbf++ = *sq++;

    } else {

      /* a constant polynomial P(z)=flag, where either flag==0 or flag==1
       OCTAVE/MATLAB: B(1,j) = flag, where j = (npoints-1)-jj and B is the return matrix */
      *sbf++ = flag;

      /* the other coefficients are all zero, since P(z) is a constant
       OCTAVE/MATLAB: B(1,j) = 0, where j = (npoints-1)-jj and B is the return matrix */
      ii = 2*baseDeg+1;
      while (ii--) *sbf++ = 0.0;

      /* for the next point */
      currentPoint++;
    }
  }
  ierr = MatDenseRestoreArray(baseFilter,&bf);CHKERRQ(ierr);
  ierr = PetscFree4(px,py,pp,qq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ////////////////////////////////////////////////////////////////////////////
   //    Base Filter
   //////////////////////////////////////////////////////////////////////////// */

/*
   FILTLAN function GetIntervals

   this routine determines the intervals (including the transition one(s)) by an iterative process

   frame is a vector consisting of 4 ordered elements:
       [frame(1),frame(4)] is the interval which (tightly) contains all eigenvalues, and
       [frame(2),frame(3)] is the interval in which the eigenvalues are sought
   baseDeg is the left-and-right degree of the base filter for each interval
   polyDeg is the (maximum possible) degree of s(z), with z*s(z) the polynomial filter
   intv is the output; the j-th interval is [intv(j),intv(j+1))
   opts is a collection of interval options

   the base filter P(z) is a piecewise polynomial from Hermite interpolation with degree baseDeg
   at each end point of intervals

   the polynomial filter Q(z) is in the form z*s(z), i.e. Q(0)==0, such that ||(1-P(z))-z*s(z)||_w is
   minimized with s(z) a polynomial of degree up to polyDeg

   the resulting polynomial filter Q(z) satisfies Q(x)>=Q(y) for x in [frame[1],frame[2]], and
   y in [frame[0],frame[3]] but not in [frame[1],frame[2]]

   the routine fills a PolynomialFilterInfo struct which gives some information of the polynomial filter
*/
static PetscErrorCode FILTLAN_GetIntervals(PetscReal *intervals,PetscReal *frame,PetscInt polyDeg,PetscInt baseDeg,FILTLAN_IOP opts,FILTLAN_PFI filterInfo)
{
  PetscErrorCode  ierr;
  Mat             baseFilter,polyFilter;
  PetscReal       intv[6],x,y,z1,z2,c,c1,c2,fc,fc2,halfPlateau,leftDelta,rightDelta,gridSize;
  PetscReal       yLimit,ySummit,yLeftLimit,yRightLimit,bottom,qIndex;
  PetscReal       yLimitGap=0.0,yLeftSummit=0.0,yLeftBottom=0.0,yRightSummit=0.0,yRightBottom=0.0;
  PetscInt        i,ii,npoints,numIter,numLeftSteps=1,numRightSteps=1,numMoreLooked=0;
  PetscBool       leftBoundaryMet=PETSC_FALSE,rightBoundaryMet=PETSC_FALSE,stepLeft,stepRight;
  const PetscReal a=frame[0],a1=frame[1],b1=frame[2],b=frame[3];
  const PetscInt  HighLowFlags[5] = { 1, -1, 0, -1, 1 };  /* if filterType is 1, only first 3 elements will be used */
  const PetscInt  numLookMore = 2*(PetscInt)(0.5+(PetscLogReal(2.0)/PetscLogReal(opts->shiftStepExpanRate)));

  PetscFunctionBegin;
  if (a>a1 || a1>b1 || b1>b) SETERRQ(PETSC_COMM_SELF,1,"Values in the frame vector should be non-decreasing");
  if (a1 == b1) SETERRQ(PETSC_COMM_SELF,1,"The range of wanted eigenvalues cannot be of size zero");
  filterInfo->filterType = 2;      /* mid-pass filter, for interior eigenvalues */
  if (b == b1) {
    if (a == a1) SETERRQ(PETSC_COMM_SELF,1,"A polynomial filter should not cover all eigenvalues");
    filterInfo->filterType = 1;    /* high-pass filter, for largest eigenvalues */
  } else if (a == a1) SETERRQ(PETSC_COMM_SELF,1,"filterType==3 for smallest eigenvalues should be pre-converted to filterType==1 for largest eigenvalues");

  /* the following recipe follows Yousef Saad (2005, 2006) with a few minor adaptations / enhancements */
  halfPlateau = 0.5*(b1-a1)*opts->initialPlateau;    /* half length of the "plateau" of the (dual) base filter */
  leftDelta = (b1-a1)*opts->initialShiftStep;        /* initial left shift */
  rightDelta = leftDelta;                            /* initial right shift */
  opts->numGridPoints = PetscMax(opts->numGridPoints,(PetscInt)(2.0*(b-a)/halfPlateau));
  gridSize = (b-a) / (PetscReal)(opts->numGridPoints);

  if (filterInfo->filterType == 2) {  /* for interior eigenvalues */
    npoints = 6;
    intv[0] = a;
    intv[5] = b;
    /* intv[1], intv[2], intv[3], and intv[4] to be determined */
  } else { /* filterType == 1 (or 3 with conversion), for extreme eigenvalues */
    npoints = 4;
    intv[0] = a;
    intv[3] = b;
    /* intv[1], and intv[2] to be determined */
  }
  z1 = a1 - leftDelta;
  z2 = b1 + rightDelta;
  filterInfo->filterOK = 0;  /* not yet found any OK filter */

  /* allocate matrices */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,2*baseDeg+2,npoints-1,NULL,&baseFilter);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,polyDeg+2,npoints-1,NULL,&polyFilter);CHKERRQ(ierr);

  /* initialize the intervals, mainly for the case opts->maxOuterIter == 0 */
  intervals[0] = intv[0];
  intervals[3] = intv[3];
  intervals[5] = intv[5];
  intervals[1] = z1;
  if (filterInfo->filterType == 2) {  /* a mid-pass filter for interior eigenvalues */
    intervals[4] = z2;
    c = (a1+b1) / 2.0;
    intervals[2] = c - halfPlateau;
    intervals[3] = c + halfPlateau;
  } else {  /* filterType == 1 (or 3 with conversion) for extreme eigenvalues */
    intervals[2] = z1 + (b1-z1)*opts->transIntervalRatio;
  }

  /* the main loop */
  for (numIter=1;numIter<=opts->maxOuterIter;numIter++) {
    if (z1 <= a) {  /* outer loop updates z1 and z2 */
      z1 = a;
      leftBoundaryMet = PETSC_TRUE;
    }
    if (filterInfo->filterType == 2) {  /* a <= z1 < (a1) */
      if (z2 >= b) {  /* a mid-pass filter for interior eigenvalues */
        z2 = b;
        rightBoundaryMet = PETSC_TRUE;
      }
      c = (z1+z2) / 2.0;
      /* a <= z1 < c-h < c+h < z2 <= b, where h is halfPlateau */
      /* [z1, c-h] and [c+h, z2] are transition interval */
      intv[1] = z1;
      intv[4] = z2;
      c1 = z1 + halfPlateau;
      intv[2] = z1;           /* i.e. c1 - halfPlateau */
      intv[3] = c1 + halfPlateau;
      ierr = FILTLAN_HermiteBaseFilterInChebyshevBasis(baseFilter,intv,6,HighLowFlags,baseDeg);CHKERRQ(ierr);
      ierr = FILTLAN_FilteredConjugateResidualPolynomial(polyFilter,baseFilter,intv,opts->intervalWeights,polyDeg);CHKERRQ(ierr);
      /* fc1 = FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,b1) - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,a1); */
      c2 = z2 - halfPlateau;
      intv[2] = c2 - halfPlateau;
      intv[3] = z2;           /* i.e. c2 + halfPlateau */
      ierr = FILTLAN_HermiteBaseFilterInChebyshevBasis(baseFilter,intv,6,HighLowFlags,baseDeg);CHKERRQ(ierr);
      ierr = FILTLAN_FilteredConjugateResidualPolynomial(polyFilter,baseFilter,intv,opts->intervalWeights,polyDeg);CHKERRQ(ierr);
      fc2 = FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,b1) - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,a1);
      yLimitGap = PETSC_MAX_REAL;
      ii = opts->maxInnerIter;
      while (ii-- && !(yLimitGap <= opts->yLimitTol)) {
        /* recursive bisection to get c such that p(a1) are p(b1) approximately the same */
        c = (c1+c2) / 2.0;
        intv[2] = c - halfPlateau;
        intv[3] = c + halfPlateau;
        ierr = FILTLAN_HermiteBaseFilterInChebyshevBasis(baseFilter,intv,6,HighLowFlags,baseDeg);CHKERRQ(ierr);
        ierr = FILTLAN_FilteredConjugateResidualPolynomial(polyFilter,baseFilter,intv,opts->intervalWeights,polyDeg);CHKERRQ(ierr);
        fc = FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,b1) - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,a1);
        if (fc*fc2 < 0.0) {
          c1 = c;
          /* fc1 = fc; */
        } else {
          c2 = c;
          fc2 = fc;
        }
        yLimitGap = PetscAbsReal(fc);
      }
    } else {  /* filterType == 1 (or 3 with conversion) for extreme eigenvalues */
      intv[1] = z1;
      intv[2] = z1 + (b1-z1)*opts->transIntervalRatio;
      ierr = FILTLAN_HermiteBaseFilterInChebyshevBasis(baseFilter,intv,4,HighLowFlags,baseDeg);CHKERRQ(ierr);
      ierr = FILTLAN_FilteredConjugateResidualPolynomial(polyFilter,baseFilter,intv,opts->intervalWeights,polyDeg);CHKERRQ(ierr);
    }
    /* polyFilter contains the coefficients of the polynomial filter which approximates phi(x)
       expanded in the `translated' Chebyshev basis */
    /* psi(x) = 1.0 - phi(x) is the dual base filter approximated by a polynomial in the form x*p(x) */
    yLeftLimit  = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,a1);
    yRightLimit = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,b1);
    yLimit  = (yLeftLimit < yRightLimit) ? yLeftLimit : yRightLimit;
    ySummit = (yLeftLimit > yRightLimit) ? yLeftLimit : yRightLimit;
    x = a1;
    while ((x+=gridSize) < b1) {
      y = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,x);
      if (y < yLimit)  yLimit  = y;
      if (y > ySummit) ySummit = y;
    }
    /* now yLimit is the minimum of x*p(x) for x in [a1, b1] */
    stepLeft  = PETSC_FALSE;
    stepRight = PETSC_FALSE;
    if ((yLimit < yLeftLimit && yLimit < yRightLimit) || yLimit < opts->yBottomLine) {
      /* very bad, step to see what will happen */
      stepLeft = PETSC_TRUE;
      if (filterInfo->filterType == 2) stepRight = PETSC_TRUE;
    } else if (filterInfo->filterType == 2) {
      if (yLeftLimit < yRightLimit) {
        if (yRightLimit-yLeftLimit > opts->yLimitTol) stepLeft = PETSC_TRUE;
      } else if (yLeftLimit-yRightLimit > opts->yLimitTol) stepRight = PETSC_TRUE;
    }
    if (!stepLeft) {
      yLeftBottom = yLeftLimit;
      x = a1;
      while ((x-=gridSize) >= a) {
        y = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,x);
        if (y < yLeftBottom) yLeftBottom = y;
        else if (y > yLeftBottom) break;
      }
      yLeftSummit = yLeftBottom;
      while ((x-=gridSize) >= a) {
        y = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,x);
        if (y > yLeftSummit) {
          yLeftSummit = y;
          if (yLeftSummit > yLimit*opts->yRippleLimit) {
            stepLeft = PETSC_TRUE;
            break;
          }
        }
        if (y < yLeftBottom) yLeftBottom = y;
      }
    }
    if (filterInfo->filterType == 2 && !stepRight) {
      yRightBottom = yRightLimit;
      x = b1;
      while ((x+=gridSize) <= b) {
        y = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,x);
        if (y < yRightBottom) yRightBottom = y;
        else if (y > yRightBottom) break;
      }
      yRightSummit = yRightBottom;
      while ((x+=gridSize) <= b) {
        y = 1.0 - FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(polyFilter,intv,npoints,x);
        if (y > yRightSummit) {
          yRightSummit = y;
          if (yRightSummit > yLimit*opts->yRippleLimit) {
            stepRight = PETSC_TRUE;
            break;
          }
        }
        if (y < yRightBottom) yRightBottom = y;
      }
    }
    if (!stepLeft && !stepRight) {
      if (filterInfo->filterType == 2) bottom = PetscMin(yLeftBottom,yRightBottom);
      else bottom = yLeftBottom;
      qIndex = 1.0 - (yLimit-bottom) / (ySummit-bottom);
      if (filterInfo->filterOK == 0 || filterInfo->filterQualityIndex < qIndex) {
        /* found the first OK filter or a better filter */
        for (i=0;i<6;i++) intervals[i] = intv[i];
        filterInfo->filterOK           = 1;
        filterInfo->filterQualityIndex = qIndex;
        filterInfo->numIter            = numIter;
        filterInfo->yLimit             = yLimit;
        filterInfo->ySummit            = ySummit;
        filterInfo->numLeftSteps       = numLeftSteps;
        filterInfo->yLeftSummit        = yLeftSummit;
        filterInfo->yLeftBottom        = yLeftBottom;
        if (filterInfo->filterType == 2) {
          filterInfo->yLimitGap        = yLimitGap;
          filterInfo->numRightSteps    = numRightSteps;
          filterInfo->yRightSummit     = yRightSummit;
          filterInfo->yRightBottom     = yRightBottom;
        }
        numMoreLooked = 0;
      } else if (++numMoreLooked == numLookMore) {
        /* filter has been optimal */
        filterInfo->filterOK = 2;
        break;
      }
      /* try stepping further to see whether it can improve */
      stepLeft = PETSC_TRUE;
      if (filterInfo->filterType == 2) stepRight = PETSC_TRUE;
    }
    /* check whether we can really "step" */
    if (leftBoundaryMet) {
      if (filterInfo->filterType == 1 || rightBoundaryMet) break;  /* cannot step further, so break the loop */
      if (stepLeft) {
        /* cannot step left, so try stepping right */
        stepLeft  = PETSC_FALSE;
        stepRight = PETSC_TRUE;
      }
    }
    if (rightBoundaryMet && stepRight) {
      /* cannot step right, so try stepping left */
      stepRight = PETSC_FALSE;
      stepLeft  = PETSC_TRUE;
    }
    /* now "step" */
    if (stepLeft) {
      numLeftSteps++;
      if (filterInfo->filterType == 2) leftDelta *= opts->shiftStepExpanRate; /* expand the step for faster convergence */
      z1 -= leftDelta;
    }
    if (stepRight) {
      numRightSteps++;
      rightDelta *= opts->shiftStepExpanRate;  /* expand the step for faster convergence */
      z2 += rightDelta;
    }
    if (filterInfo->filterType == 2) {
      /* shrink the "plateau" of the (dual) base filter */
      if (stepLeft && stepRight) halfPlateau /= opts->plateauShrinkRate;
      else halfPlateau /= PetscSqrtReal(opts->plateauShrinkRate);
    }
  }
  filterInfo->totalNumIter = numIter;
  ierr = MatDestroy(&baseFilter);CHKERRQ(ierr);
  ierr = MatDestroy(&polyFilter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ////////////////////////////////////////////////////////////////////////////
   //    Chebyshev Polynomials
   //////////////////////////////////////////////////////////////////////////// */

/*
   FILTLAN function ExpandNewtonPolynomialInChebyshevBasis

   translate the coefficients of a Newton polynomial to the coefficients
   in a basis of the `translated' (scale-and-shift) Chebyshev polynomials

   input:
   a Newton polynomial defined by vectors a and x:
       P(z) = a(1) + a(2)*(z-x(1)) + a(3)*(z-x(1))*(z-x(2)) + ... + a(n)*(z-x(1))*...*(z-x(n-1))
   the interval [aa,bb] defines the `translated' Chebyshev polynomials S_i(z) = T_i((z-c)/h),
       where c=(aa+bb)/2 and h=(bb-aa)/2, and T_i is the Chebyshev polynomial of the first kind
   note that T_i is defined by T_0(z)=1, T_1(z)=z, and T_i(z)=2*z*T_{i-1}(z)+T_{i-2}(z) for i>=2

   output:
   a vector q containing the Chebyshev coefficients:
       P(z) = q(1)*S_0(z) + q(2)*S_1(z) + ... + q(n)*S_{n-1}(z)
*/
static PetscErrorCode FILTLAN_ExpandNewtonPolynomialInChebyshevBasis(PetscInt n,PetscReal aa,PetscReal bb,PetscReal *a,PetscReal *x,PetscReal *q)
{
  PetscErrorCode ierr;
  PetscInt       m,mm;
  PetscReal      *sa,*sx,*sq,*sq2,*q2,c,c2,h,h2;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&q2);CHKERRQ(ierr);
  sa = a+n;    /* pointers for traversing a and x */
  sx = x+n-1;
  *q = *--sa;  /* set q[0] = a(n) */

  c = (aa+bb)/2.0;
  h = (bb-aa)/2.0;
  h2 = h/2.0;

  for (m=1;m<=n-1;m++) {  /* the main loop for translation */

    /* compute q2[0:m-1] = (c-x[n-m-1])*q[0:m-1] */
    mm = m;
    sq = q;
    sq2 = q2;
    c2 = c-(*--sx);
    while (mm--) *(sq2++) = c2*(*sq++);
    *sq2 = 0.0;         /* q2[m] = 0.0 */
    *(q2+1) += h*(*q);  /* q2[1] = q2[1] + h*q[0] */

    /* compute q2[0:m-2] = q2[0:m-2] + h2*q[1:m-1] */
    mm = m-1;
    sq2 = q2;
    sq = q+1;
    while (mm--) *(sq2++) += h2*(*sq++);

    /* compute q2[2:m] = q2[2:m] + h2*q[1:m-1] */
    mm = m-1;
    sq2 = q2+2;
    sq = q+1;
    while (mm--) *(sq2++) += h2*(*sq++);

    /* compute q[0:m] = q2[0:m] */
    mm = m+1;
    sq2 = q2;
    sq = q;
    while (mm--) *sq++ = *sq2++;
    *q += (*--sa);      /* q[0] = q[0] + p[n-m-1] */

  }
  ierr = PetscFree(q2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FILTLAN function PolynomialEvaluationInChebyshevBasis

   evaluate P(z) at z=z0, where P(z) is a polynomial expanded in a basis of
   the `translated' (i.e. scale-and-shift) Chebyshev polynomials

   input:
   c is a vector of Chebyshev coefficients which defines the polynomial
       P(z) = c(1)*S_0(z) + c(2)*S_1(z) + ... + c(n)*S_{n-1}(z),
   where S_i is the `translated' Chebyshev polynomial S_i((z-c)/h) = T_i(z), with
       c = (intv(j)+intv(j+1)) / 2,  h = (intv(j+1)-intv(j)) / 2
   note that T_i(z) is the Chebyshev polynomial of the first kind,
       T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2

   output:
   the evaluated value of P(z) at z=z0
*/
static PetscReal FILTLAN_PolynomialEvaluationInChebyshevBasis(Mat pp,PetscInt idx,PetscReal z0,PetscReal aa,PetscReal bb)
{
  PetscErrorCode ierr;
  PetscInt       m,n,ii,deg1;
  PetscReal      y,zz,t0,t1,t2,*sp,*sc;

  PetscFunctionBegin;
  ierr = MatGetSize(pp,&m,&n);CHKERRQ(ierr);
  deg1 = m;
  if (aa==-1.0 && bb==1.0) zz = z0;  /* treat it as a special case to reduce rounding errors */
  else zz = (aa==bb)? 0.0 : -1.0+2.0*(z0-aa)/(bb-aa);

  /* compute y = P(z0), where we utilize the Chebyshev recursion */
  ierr = MatDenseGetArray(pp,&sp);CHKERRQ(ierr);
  sc = sp+(idx-1)*m;   /* sc points to column idx of pp */
  y = *sc++;
  t1 = 1.0; t2 = zz;
  ii = deg1-1;
  while (ii--) {
    /* Chebyshev recursion: T_0(zz)=1, T_1(zz)=zz, and T_{i+1}(zz) = 2*zz*T_i(zz) + T_{i-1}(zz) for i>=2
       the values of T_{i+1}(zz), T_i(zz), T_{i-1}(zz) are stored in t0, t1, t2, respectively */
    t0 = 2*zz*t1 - t2;
    /* it also works for the base case / the first iteration, where t0 equals 2*zz*1-zz == zz which is T_1(zz) */
    t2 = t1;
    t1 = t0;
    y += (*sc++)*t0;
  }
  ierr = MatDenseRestoreArray(pp,&sp);CHKERRQ(ierr);
  PetscFunctionReturn(y);
}

#define basisTranslated PETSC_TRUE
/*
   FILTLAN function PiecewisePolynomialEvaluationInChebyshevBasis

   evaluate P(z) at z=z0, where P(z) is a piecewise polynomial expanded
   in a basis of the (optionally translated, i.e. scale-and-shift) Chebyshev polynomials for each interval

   input:
   intv is a vector which defines the intervals; the j-th interval is [intv(j), intv(j+1))
   pp is a matrix of Chebyshev coefficients which defines a piecewise polynomial P(z)
   in a basis of the `translated' Chebyshev polynomials in each interval
   the polynomial P_j(z) in the j-th interval, i.e. when z is in [intv(j), intv(j+1)), is defined by the j-th column of pp:
       if basisTranslated == false, then
           P_j(z) = pp(1,j)*T_0(z) + pp(2,j)*T_1(z) + ... + pp(n,j)*T_{n-1}(z),
       where T_i(z) is the Chebyshev polynomial of the first kind,
           T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2
       if basisTranslated == true, then
           P_j(z) = pp(1,j)*S_0(z) + pp(2,j)*S_1(z) + ... + pp(n,j)*S_{n-1}(z),
       where S_i is the `translated' Chebyshev polynomial S_i((z-c)/h) = T_i(z), with
           c = (intv(j)+intv(j+1)) / 2,  h = (intv(j+1)-intv(j)) / 2

   output:
   the evaluated value of P(z) at z=z0

   note that if z0 falls below the first interval, then the polynomial in the first interval will be used to evaluate P(z0)
             if z0 flies over  the last  interval, then the polynomial in the last  interval will be used to evaluate P(z0)
*/
static PetscReal FILTLAN_PiecewisePolynomialEvaluationInChebyshevBasis(Mat pp,PetscReal *intv,PetscInt npoints,PetscReal z0)
{
  PetscReal *sintv,aa,bb,resul;
  PetscInt  idx;

  PetscFunctionBegin;
  /* determine the interval which contains z0 */
  sintv = &intv[1];
  idx = 1;
  if (npoints>2 && z0 >= *sintv) {
    sintv++;
    while (++idx < npoints-1) {
      if (z0 < *sintv) break;
      sintv++;
    }
  }
  /* idx==1 if npoints<=2; otherwise idx satisfies:
         intv(idx) <= z0 < intv(idx+1),  if 2 <= idx <= npoints-2
         z0 < intv(idx+1),               if idx == 1
         intv(idx) <= z0,                if idx == npoints-1
     in addition, sintv points to &intv(idx+1) */
  if (basisTranslated) {
    /* the basis consists of `translated' Chebyshev polynomials */
    /* find the interval of concern, [aa,bb] */
    aa = *(sintv-1);
    bb = *sintv;
    resul = FILTLAN_PolynomialEvaluationInChebyshevBasis(pp,idx,z0,aa,bb);
  } else {
    /* the basis consists of standard Chebyshev polynomials, with interval [-1.0,1.0] for integration */
    resul = FILTLAN_PolynomialEvaluationInChebyshevBasis(pp,idx,z0,-1.0,1.0);
  }
  PetscFunctionReturn(resul);
}

/*
   FILTLAN function PiecewisePolynomialInnerProductInChebyshevBasis

   compute the weighted inner product of two piecewise polynomials expanded
   in a basis of `translated' (i.e. scale-and-shift) Chebyshev polynomials for each interval

   pp and qq are two matrices of Chebyshev coefficients which define the piecewise polynomials P(z) and Q(z), respectively
   for z in the j-th interval, P(z) equals
       P_j(z) = pp(1,j)*S_0(z) + pp(2,j)*S_1(z) + ... + pp(n,j)*S_{n-1}(z),
   and Q(z) equals
       Q_j(z) = qq(1,j)*S_0(z) + qq(2,j)*S_1(z) + ... + qq(n,j)*S_{n-1}(z),
   where S_i(z) is the `translated' Chebyshev polynomial in that interval,
       S_i((z-c)/h) = T_i(z),  c = (aa+bb)) / 2,  h = (bb-aa) / 2,
   with T_i(z) the Chebyshev polynomial of the first kind
       T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2

   the (scaled) j-th interval inner product is defined by
       <P_j,Q_j> = (Pi/2)*( pp(1,j)*qq(1,j) + sum_{k} pp(k,j)*qq(k,j) ),
   which comes from the property
       <T_0,T_0>=pi, <T_i,T_i>=pi/2 for i>=1, and <T_i,T_j>=0 for i!=j

   the weighted inner product is <P,Q> = sum_{j} intervalWeights(j)*<P_j,Q_j>,
   which is the return value

   note that for unit weights, pass an empty vector of intervalWeights (i.e. of length 0)
*/
static PetscErrorCode FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(Mat pp,Mat qq,const PetscReal *intervalWeights,PetscScalar *resul)
{
  PetscErrorCode  ierr;
  PetscInt        prows,pcols,qrows,qcols,numIntv,deg1,dp,dq,ii,kk;
  PetscReal       ans=0.0,ans2;
  const PetscReal *sw;
  PetscScalar     *sq,*sp;

  PetscFunctionBegin;
  ierr = MatGetSize(pp,&prows,&pcols);CHKERRQ(ierr);
  ierr = MatGetSize(qq,&qrows,&qcols);CHKERRQ(ierr);
  deg1 = PetscMin(prows,qrows);  /* number of effective coefficients, one more than the effective polynomial degree */
  if (!deg1) {  /* a special case, zero polynomial */
    *resul = 0.0;
    PetscFunctionReturn(0);
  }
  numIntv = PetscMin(pcols,qcols);  /* number of intervals */
  dp = prows - deg1;  /* the extra amount to skip, if any */
  dq = qrows - deg1;

  ierr = MatDenseGetArray(pp,&sp);CHKERRQ(ierr);
  ierr = MatDenseGetArray(qq,&sq);CHKERRQ(ierr);

  /* scaled by intervalWeights(i) in the i-th interval (we assume intervalWeights[] are always provided).
     compute ans = sum_{i=1,...,numIntv} intervalWeights(i)*[ pp(1,i)*qq(1,i) + sum_{k=1,...,deg} pp(k,i)*qq(k,i) ] */
  ii = numIntv;
  sw = intervalWeights;
  while (ii--) {
    /* compute ans2 = pp(1,i)*qq(1,i) + sum_{k=1,...,deg} pp(k,i)*qq(k,i), where i = numIntv-ii */
    ans2 = (*sp) * (*sq);  /* the first term pp(1,i)*qq(1,i) is being added twice */
    kk = deg1;
    while (kk--) ans2 += (*sp++)*(*sq++);  /* add pp(k,i)*qq(k,i), where k = deg1-kk */
    ans += ans2*(*sw++);  /* compute ans += ans2*intervalWeights(i) */
    sp += dp;  /* skip the extra */
    sq += dq;
  }
  *resul = ans*PETSC_PI/2.0; /* return the inner product */

  ierr = MatDenseRestoreArray(pp,&sp);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(qq,&sq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FILTLAN function PiecewisePolynomialInChebyshevBasisMultiplyX

   compute Q(z) = z*P(z), where P(z) and Q(z) are piecewise polynomials expanded
   in a basis of `translated' (i.e. scale-and-shift) Chebyshev polynomials for each interval

   P(z) and Q(z) are stored as matrices of Chebyshev coefficients pp and qq, respectively

   For z in the j-th interval, P(z) equals
       P_j(z) = pp(1,j)*S_0(z) + pp(2,j)*S_1(z) + ... + pp(n,j)*S_{n-1}(z),
   and Q(z) equals
       Q_j(z) = qq(1,j)*S_0(z) + qq(2,j)*S_1(z) + ... + qq(n,j)*S_{n-1}(z),
   where S_i(z) is the `translated' Chebyshev polynomial in that interval,
       S_i((z-c)/h) = T_i(z),  c = (intv(j)+intv(j+1))) / 2,  h = (intv(j+1)-intv(j)) / 2,
   with T_i(z) the Chebyshev polynomial of the first kind
       T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2

   the returned matrix is qq which represents Q(z) = z*P(z)
*/
static PetscErrorCode FILTLAN_PiecewisePolynomialInChebyshevBasisMultiplyX(Mat pp,PetscReal *intv,Mat qq)
{
  PetscErrorCode ierr;
  PetscInt       nintv,deg1,m,n,i,jj;
  PetscReal      c,h,h2,tmp,*sq,*sq2,*sp,*sp2,*sintv;

  PetscFunctionBegin;
  ierr = MatGetSize(pp,&deg1,&nintv);CHKERRQ(ierr);
  ierr = MatGetSize(qq,&m,&n);CHKERRQ(ierr);
  if (m!=deg1+1 || n!=nintv) SETERRQ(PetscObjectComm((PetscObject)pp),1,"Wrong dimensions");
  ierr = MatDenseGetArray(pp,&sp);CHKERRQ(ierr);
  ierr = MatDenseGetArray(qq,&sq);CHKERRQ(ierr);
  sp2 = sp;
  sintv = intv;
  sq2 = sq+1;

  jj = nintv;
  while (jj--) {  /* consider interval between intv(j) and intv(j+1), where j == nintv-jj */
    c = 0.5*(*sintv + *(sintv+1));    /* compute c = (intv(j) + intv(j+1))/2 */
    h = 0.5*(*(sintv+1) - (*sintv));  /* compute h = (intv(j+1) - intv(j))/2  and  h2 = h/2 */
    h2 = 0.5*h;
    i = deg1;
    while (i--) *sq++ = c*(*sp++);    /* compute q(1:deg1,j) = c*p(1:deg1,j) */
    *sq++ = 0.0;                      /* set q(deg1+1,j) = 0.0 */
    *(sq2++) += h*(*sp2++);           /* compute q(2,j) = q(2,j) + h*p(1,j) */
    i = deg1-1;
    while (i--) {       /* compute q(3:deg1+1,j) += h2*p(2:deg1,j) and then q(1:deg1-1,j) += h2*p(2:deg1,j) */
      tmp = h2*(*sp2++);
      *(sq2-2) += tmp;
      *(sq2++) += tmp;
    }
    sq2++;   /* for pointing to q(2,j+1) */
    sintv++; /* for the next interval */
  }
  ierr = MatDenseRestoreArray(pp,&sp);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(qq,&sq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ////////////////////////////////////////////////////////////////////////////
   //    Conjugate Residual Method in the Polynomial Space
   //////////////////////////////////////////////////////////////////////////// */

/*
   Auxiliary function to extend a Mat with one additional row, optionally copying the values
*/
static PetscErrorCode MatDenseExtend(Mat *A,MatDuplicateOption cpvalues)
{
  PetscErrorCode ierr;
  PetscScalar    *pA,*pB;
  PetscBool      isdense;
  PetscInt       j,m,n,lda;
  Mat            B=*A;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  if (!isdense) SETERRQ(PetscObjectComm((PetscObject)B),1,"Matrix must be of type seqdense");
  ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr);
  lda = m+1;
  ierr = MatCreateSeqDense(PetscObjectComm((PetscObject)B),lda,n,NULL,A);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES) {
    ierr = MatDenseGetArray(*A,&pA);CHKERRQ(ierr);
    ierr = MatDenseGetArray(B,&pB);CHKERRQ(ierr);
    for (j=0;j<n;j++) { ierr = PetscMemcpy(pA+j*lda,pB+j*m,m*sizeof(PetscScalar));CHKERRQ(ierr); }
    ierr = MatDenseRestoreArray(*A,&pA);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(B,&pB);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FILTLAN function FilteredConjugateResidualPolynomial

   ** Conjugate Residual Method in the Polynomial Space

   this routine employs a conjugate-residual-type algorithm in polynomial space to minimize ||P(z)-Q(z)||_w,
   where P(z), the base filter, is the input piecewise polynomial, and
         Q(z) is the output polynomial satisfying Q(0)==1, i.e. the constant term of Q(z) is 1
   niter is the number of conjugate-residual iterations; therefore, the degree of Q(z) is up to niter+1
   both P(z) and Q(z) are expanded in the `translated' (scale-and-shift) Chebyshev basis for each interval,
   and presented as matrices of Chebyshev coefficients, denoted by pp and qq, respectively

   input:
   intv is a vector which defines the intervals; the j-th interval is [intv(j),intv(j+1))
   w is a vector of Chebyshev weights; the weight of j-th interval is w(j)
       the interval weights define the inner product of two continuous functions and then
       the derived w-norm ||P(z)-Q(z)||_w
   pp is a matrix of Chebyshev coefficients which defines the piecewise polynomial P(z)
   to be specific, for z in [intv(j), intv(j+1)), P(z) equals
       P_j(z) = pp(1,j)*S_0(z) + pp(2,j)*S_1(z) + ... + pp(niter+2,j)*S_{niter+1}(z),
   where S_i(z) is the `translated' Chebyshev polynomial in that interval,
       S_i((z-c)/h) = T_i(z),  c = (intv(j)+intv(j+1))) / 2,  h = (intv(j+1)-intv(j)) / 2,
   with T_i(z) the Chebyshev polynomial of the first kind,
       T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2

   output:
   the return matrix, denoted by qq, represents a polynomial Q(z) with degree up to 1+niter
   and satisfying Q(0)==1, such that ||P(z))-Q(z)||_w is minimized
   this polynomial Q(z) is expanded in the `translated' Chebyshev basis for each interval
   to be precise, considering z in [intv(j), intv(j+1)), Q(z) equals
       Q_j(z) = qq(1,j)*S_0(z) + qq(2,j)*S_1(z) + ... + qq(niter+2,j)*S_{niter+1}(z)

   note:
   1. since Q(0)==1, P(0)==1 is expected; if P(0)!=1, one can translate P(z)
      for example, if P(0)==0, one can use 1-P(z) as input instead of P(z)
   2. typically, the base filter, defined by pp and intv, is from Hermite interpolation
      in intervals [intv(j),intv(j+1)) for j=1,...,nintv, with nintv the number of intervals
*/
static PetscErrorCode FILTLAN_FilteredConjugateResidualPolynomial(Mat polyFilter,Mat baseFilter,PetscReal *intv,PetscReal *intervalWeights,PetscInt niter)
{
  PetscErrorCode ierr;
  PetscInt       nintv,i,jj;
  PetscScalar    *pp,*sp,rho,rho0,rho1,den,bet,alp,alp0;
  Mat            cpol,ppol,rpol,appol,arpol;

  PetscFunctionBegin;
  ierr = MatGetSize(baseFilter,NULL,&nintv);CHKERRQ(ierr);
  /* initialize polynomial ppol to be 1 (i.e. multiplicative identity) in all intervals */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,2,nintv,NULL,&ppol);CHKERRQ(ierr);
  ierr = MatDenseGetArray(ppol,&pp);CHKERRQ(ierr);
  /* set the polynomial be 1 for all intervals */
  sp = pp;
  jj = nintv;
  while (jj--) {
    *sp++ = 1.0;
    *sp++ = 0.0;
  }
  ierr = MatDenseRestoreArray(ppol,&pp);CHKERRQ(ierr);
  /* ppol is the initial p-polynomial (corresponding to the A-conjugate vector p in CG)
     rpol is the r-polynomial (corresponding to the residual vector r in CG)
     cpol is the "corrected" residual polynomial (result of this function) */
  ierr = MatDuplicate(ppol,MAT_COPY_VALUES,&rpol);CHKERRQ(ierr);
  ierr = MatDuplicate(ppol,MAT_COPY_VALUES,&cpol);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,nintv,NULL,&appol);CHKERRQ(ierr);
  ierr = FILTLAN_PiecewisePolynomialInChebyshevBasisMultiplyX(ppol,intv,appol);CHKERRQ(ierr);
  ierr = MatDuplicate(appol,MAT_COPY_VALUES,&arpol);CHKERRQ(ierr);
  ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(rpol,arpol,intervalWeights,&rho);CHKERRQ(ierr);
  for (i=0;i<niter;i++) {
    ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(appol,appol,intervalWeights,&den);CHKERRQ(ierr);
    alp0 = rho/den;
    ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(baseFilter,appol,intervalWeights,&rho1);CHKERRQ(ierr);
    alp  = (rho-rho1)/den;
    ierr = MatDenseExtend(&rpol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatDenseExtend(&cpol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatAXPY(rpol,-alp0,appol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(cpol,-alp,appol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    if (i+1 == niter) break;
    ierr = MatDenseExtend(&arpol,MAT_DO_NOT_COPY_VALUES);CHKERRQ(ierr);
    ierr = FILTLAN_PiecewisePolynomialInChebyshevBasisMultiplyX(rpol,intv,arpol);CHKERRQ(ierr);
    rho0 = rho;
    ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(rpol,arpol,intervalWeights,&rho);CHKERRQ(ierr);
    bet  = rho / rho0;
    ierr = MatDenseExtend(&ppol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatDenseExtend(&appol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatAYPX(ppol,bet,rpol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAYPX(appol,bet,arpol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatCopy(cpol,polyFilter,SAME_NONZERO_PATTERN);CHKERRQ(ierr);  /* copy the result */
  ierr = MatDestroy(&ppol);CHKERRQ(ierr);
  ierr = MatDestroy(&rpol);CHKERRQ(ierr);
  ierr = MatDestroy(&cpol);CHKERRQ(ierr);
  ierr = MatDestroy(&appol);CHKERRQ(ierr);
  ierr = MatDestroy(&arpol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FILTLAN function FilteredConjugateResidualMatrixPolynomialVectorProduct

   this routine employs a conjugate-residual-type algorithm in polynomial space to compute
   x = x0 + s(A)*r0 with r0 = b - A*x0, such that ||(1-P(z))-z*s(z)||_w is minimized, where
   P(z) is a given piecewise polynomial, called the base filter,
   s(z) is a polynomial of degree up to niter, the number of conjugate-residual iterations,
   and b and x0 are given vectors

   note that P(z) is expanded in the `translated' (scale-and-shift) Chebyshev basis for each interval,
   and presented as a matrix of Chebyshev coefficients pp

   input:
   A is a sparse matrix
   x0, b are vectors
   niter is the number of conjugate-residual iterations
   intv is a vector which defines the intervals; the j-th interval is [intv(j),intv(j+1))
   w is a vector of Chebyshev weights; the weight of j-th interval is w(j)
       the interval weights define the inner product of two continuous functions and then
       the derived w-norm ||P(z)-Q(z)||_w
   pp is a matrix of Chebyshev coefficients which defines the piecewise polynomial P(z)
   to be specific, for z in [intv(j), intv(j+1)), P(z) equals
       P_j(z) = pp(1,j)*S_0(z) + pp(2,j)*S_1(z) + ... + pp(niter+2,j)*S_{niter+1}(z),
   where S_i(z) is the `translated' Chebyshev polynomial in that interval,
       S_i((z-c)/h) = T_i(z),  c = (intv(j)+intv(j+1))) / 2,  h = (intv(j+1)-intv(j)) / 2,
   with T_i(z) the Chebyshev polynomial of the first kind,
       T_0(z) = 1, T_1(z) = z, and T_i(z) = 2*z*T_{i-1}(z) - T_{i-2}(z) for i>=2
   tol is the tolerance; if the residual polynomial in z-norm is dropped by a factor lower
       than tol, then stop the conjugate-residual iteration

   output:
   the return vector is x = x0 + s(A)*r0 with r0 = b - A*x0, such that ||(1-P(z))-z*s(z)||_w is minimized,
   subject to that s(z) is a polynomial of degree up to niter, where P(z) is the base filter
   in short, z*s(z) approximates 1-P(z)

   note:
   1. since z*s(z) approximates 1-P(z), P(0)==1 is expected; if P(0)!=1, one can translate P(z)
      for example, if P(0)==0, one can use 1-P(z) as input instead of P(z)
   2. typically, the base filter, defined by pp and intv, is from Hermite interpolation
      in intervals [intv(j),intv(j+1)) for j=1,...,nintv, with nintv the number of intervals
   3. a common application is to compute R(A)*b, where R(z) approximates 1-P(z)
      in this case, one can set x0 = 0 and then the return vector is x = s(A)*b, where
      z*s(z) approximates 1-P(z); therefore, A*x is the wanted R(A)*b
*/
static PetscErrorCode FILTLAN_FilteredConjugateResidualMatrixPolynomialVectorProduct(Mat A,Vec b,Vec x,Mat baseFilter,PetscReal *intv,PetscReal *intervalWeights,PetscInt niter,Vec *work)
{
  PetscErrorCode ierr;
  PetscInt       nintv,i,jj;
  PetscReal      tol=0.0;
  PetscScalar    *pp,*sp,rho,rho0,rho00,rho1,den,bet,alp,alp0;
  Mat            cpol,ppol,rpol,appol,arpol;
  Vec            r=work[0],p=work[1],ap=work[2],w=work[3];

  PetscFunctionBegin;
  ierr = MatGetSize(baseFilter,NULL,&nintv);CHKERRQ(ierr);
  /* initialize polynomial ppol to be 1 (i.e. multiplicative identity) in all intervals */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,2,nintv,NULL,&ppol);CHKERRQ(ierr);
  ierr = MatDenseGetArray(ppol,&pp);CHKERRQ(ierr);
  /* set the polynomial be 1 for all intervals */
  sp = pp;
  jj = nintv;
  while (jj--) {
    *sp++ = 1.0;
    *sp++ = 0.0;
  }
  ierr = MatDenseRestoreArray(ppol,&pp);CHKERRQ(ierr);
  /* ppol is the initial p-polynomial (corresponding to the A-conjugate vector p in CG)
     rpol is the r-polynomial (corresponding to the residual vector r in CG)
     cpol is the "corrected" residual polynomial */
  ierr = MatDuplicate(ppol,MAT_COPY_VALUES,&rpol);CHKERRQ(ierr);
  ierr = MatDuplicate(ppol,MAT_COPY_VALUES,&cpol);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,nintv,NULL,&appol);CHKERRQ(ierr);
  ierr = FILTLAN_PiecewisePolynomialInChebyshevBasisMultiplyX(ppol,intv,appol);CHKERRQ(ierr);
  ierr = MatDuplicate(appol,MAT_COPY_VALUES,&arpol);CHKERRQ(ierr);
  ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(rpol,arpol,intervalWeights,&rho00);CHKERRQ(ierr);
  rho = rho00;

  /* corrected CR in vector space */
  /* we assume x0 is always 0 */
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecCopy(b,r);CHKERRQ(ierr);     /* initial residual r = b-A*x0 */
  ierr = VecCopy(r,p);CHKERRQ(ierr);     /* p = r */
  ierr = MatMult(A,p,ap);CHKERRQ(ierr);  /* ap = A*p */

  for (i=0;i<niter;i++) {
    /* iteration in the polynomial space */
    ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(appol,appol,intervalWeights,&den);CHKERRQ(ierr);
    alp0 = rho/den;
    ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(baseFilter,appol,intervalWeights,&rho1);CHKERRQ(ierr);
    alp  = (rho-rho1)/den;
    ierr = MatDenseExtend(&rpol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatDenseExtend(&cpol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatAXPY(rpol,-alp0,appol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(cpol,-alp,appol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDenseExtend(&arpol,MAT_DO_NOT_COPY_VALUES);CHKERRQ(ierr);
    ierr = FILTLAN_PiecewisePolynomialInChebyshevBasisMultiplyX(rpol,intv,arpol);CHKERRQ(ierr);
    rho0 = rho;
    ierr = FILTLAN_PiecewisePolynomialInnerProductInChebyshevBasis(rpol,arpol,intervalWeights,&rho);CHKERRQ(ierr);

    /* update x in the vector space */
    ierr = VecAXPY(x,alp,p);CHKERRQ(ierr);   /* x += alp*p */
    if (rho < tol*rho00) break;

    /* finish the iteration in the polynomial space */
    bet = rho / rho0;
    ierr = MatDenseExtend(&ppol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatDenseExtend(&appol,MAT_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatAYPX(ppol,bet,rpol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAYPX(appol,bet,arpol,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

    /* finish the iteration in the vector space */
    ierr = VecAXPY(r,-alp0,ap);CHKERRQ(ierr);   /* r -= alp0*ap */
    ierr = VecAYPX(p,bet,r);CHKERRQ(ierr);      /* p = r + bet*p */
    ierr = MatMult(A,r,w);CHKERRQ(ierr);        /* ap = A*r + bet*ap */
    ierr = VecAYPX(ap,bet,w);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ppol);CHKERRQ(ierr);
  ierr = MatDestroy(&cpol);CHKERRQ(ierr);
  ierr = MatDestroy(&rpol);CHKERRQ(ierr);
  ierr = MatDestroy(&arpol);CHKERRQ(ierr);
  ierr = MatDestroy(&appol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Gateway to FILTLAN for evaluating y=p(A)*x
*/
PetscErrorCode STFilter_FILTLAN_Apply(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST_FILTER      *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  ierr = FILTLAN_FilteredConjugateResidualMatrixPolynomialVectorProduct(st->T[0],x,y,ctx->baseFilter,ctx->intervals,ctx->opts->intervalWeights,ctx->polyDegree,st->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   FILTLAN function PolynomialFilterInterface::setFilter

   Creates the shifted (and scaled) matrix and the base filter P(z)
*/
PetscErrorCode STFilter_FILTLAN_setFilter(ST st)
{
  PetscErrorCode ierr;
  ST_FILTER      *ctx = (ST_FILTER*)st->data;
  PetscInt       i,npoints;
  PetscReal      frame2[4];
  PetscScalar    alpha;
  const PetscInt HighLowFlags[5] = { 1, -1, 0, -1, 1 };

  PetscFunctionBegin;
  ierr = MatDestroy(&st->T[0]);CHKERRQ(ierr);
  if (ctx->frame[0] == ctx->frame[1]) {  /* low pass filter, convert it to high pass filter */
    /* T = frame[3]*eye(n) - A */
    ierr = MatDuplicate(st->A[0],MAT_COPY_VALUES,&st->T[0]);CHKERRQ(ierr);
    ierr = MatScale(st->T[0],-1.0);CHKERRQ(ierr);
    alpha = ctx->frame[3];
    ierr = MatShift(st->T[0],alpha);CHKERRQ(ierr);
    for (i=0;i<4;i++) frame2[i] = ctx->frame[3] - ctx->frame[3-i];
    ierr = FILTLAN_GetIntervals(ctx->intervals,frame2,ctx->polyDegree,ctx->baseDegree,ctx->opts,ctx->filterInfo);CHKERRQ(ierr);
    /* translate the intervals back */
    for (i=0;i<4;i++) ctx->intervals2[i] = ctx->frame[3] - ctx->intervals[3-i];
  } else {  /* it can be a mid-pass filter or a high-pass filter */
      if (ctx->frame[0] == 0.0) {
      ierr = PetscObjectReference((PetscObject)st->A[0]);CHKERRQ(ierr);
      st->T[0] = st->A[0];
      ierr = FILTLAN_GetIntervals(ctx->intervals,ctx->frame,ctx->polyDegree,ctx->baseDegree,ctx->opts,ctx->filterInfo);CHKERRQ(ierr);
      for (i=0;i<6;i++) ctx->intervals2[i] = ctx->intervals[i];
    } else {
      /* T = A - frame[0]*eye(n) */
      ierr = MatDuplicate(st->A[0],MAT_COPY_VALUES,&st->T[0]);CHKERRQ(ierr);
      alpha = -ctx->frame[0];
      ierr = MatShift(st->T[0],alpha);CHKERRQ(ierr);
      for (i=0;i<4;i++) frame2[i] = ctx->frame[i] - ctx->frame[0];
      ierr = FILTLAN_GetIntervals(ctx->intervals,frame2,ctx->polyDegree,ctx->baseDegree,ctx->opts,ctx->filterInfo);CHKERRQ(ierr);
      /* translate the intervals back */
      for (i=0;i<6;i++) ctx->intervals2[i] = ctx->intervals[i] + ctx->frame[0];
    }
  }
  npoints = (ctx->filterInfo->filterType == 2)? 6: 4;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,2*ctx->baseDegree+2,npoints-1,NULL,&ctx->baseFilter);CHKERRQ(ierr);
  ierr = FILTLAN_HermiteBaseFilterInChebyshevBasis(ctx->baseFilter,ctx->intervals,npoints,HighLowFlags,ctx->baseDegree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

