classdef SlepcSVD < PetscObject
%
%   SlepcSVD - a SLEPc singular value solver object
%
%   Creation:
%     svd = SlepcSVD;
%     svd.SetType('cross');
%     svd.SetOperator(A);
%     svd.SetFromOptions;

%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%  SLEPc - Scalable Library for Eigenvalue Problem Computations
%  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
%
%  This file is part of SLEPc.
%
%  SLEPc is free software: you can redistribute it and/or modify it under  the
%  terms of version 3 of the GNU Lesser General Public License as published by
%  the Free Software Foundation.
%
%  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
%  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
%  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
%  more details.
%
%  You  should have received a copy of the GNU Lesser General  Public  License
%  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  properties (Constant)
    SVD_TRANSPOSE_EXPLICIT=0;
    SVD_TRANSPOSE_IMPLICIT=1;

    LARGEST=0;
    SMALLEST=1;
  end
  methods
    function obj = SlepcSVD(pid,flag)
      if (nargin > 1)
        %  SelpcSVD(pid,'pobj') uses an already existing SLEPc SVD object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'SVDCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'SVDSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'SVDSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'SVDSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj)
      err = calllib('libslepc', 'SVDSolve', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperator(obj,A)
      err = calllib('libslepc', 'SVDSetOperator', obj.pobj,A.pobj);PetscCHKERRQ(err);
    end
    function err = SetImplicitTranspose(obj,t)
      err = calllib('libslepc', 'SVDSetImplicitTranspose', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetWhichSingularTriplets(obj,t)
      err = calllib('libslepc', 'SVDSetWhichSingularTriplets', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetTolerances(obj,t,mx)
      if (nargin == 2) mx = 0; end
      err = calllib('libslepc', 'SVDSetTolerances', obj.pobj,t,mx);PetscCHKERRQ(err);
    end
    function err = SetDimensions(obj,nev,ncv,mpd)
      if (nargin < 3) ncv = 0; end
      if (nargin < 4) mpd = 0; end
      err = calllib('libslepc', 'SVDSetDimensions', obj.pobj,nev,ncv,mpd);PetscCHKERRQ(err);
    end
    function [nconv,err] = GetConverged(obj)
      nconv = 0;
      [err,nconv] = calllib('libslepc', 'SVDGetConverged', obj.pobj,nconv);PetscCHKERRQ(err);
    end
    function [sigma,u,v,err] = GetSingularTriplet(obj,i,uu,vv)
      sigma = 0.0;
      if (nargin < 3)
        x = 0;
      else
        x = uu.pobj;
      end
      if (nargin < 4)
        y = 0;
      else
        y = vv.pobj;
      end
      if (nargout > 1 && (x==0 || y==0))
        [err,pid] = calllib('libslepc', 'SVDGetOperator', obj.pobj,0);PetscCHKERRQ(err);
        A = PetscMat(pid,'pobj');
        [m,n] = A.GetSize();
      end
      freexr = 0;
      freexi = 0;
      if (nargout > 1 && x==0)
        uu = PetscVec();
        freexr = 1;
        uu.SetType('seq');
        uu.SetSizes(m,m);
        x = uu.pobj;
      end
      if (nargout > 2 && y==0)
        vv = PetscVec();
        freexi = 1;
        vv.SetType('seq');
        vv.SetSizes(n,n);
        y = vv.pobj;
      end
      [err,sigma] = calllib('libslepc', 'SVDGetSingularTriplet', obj.pobj,i-1,sigma,x,y);PetscCHKERRQ(err);
      if (nargout > 1)
        if (x ~= 0)
          u = uu(:);
        else
          u = 0;
        end
      end
      if (nargout > 2)
        if (y ~= 0)
          v = vv(:);
        else
          v = 0;
        end
      end
      if (freexr)
        uu.Destroy();
      end
      if (freexi)
        vv.Destroy();
      end
    end
    function [relerr,err] = ComputeRelativeError(obj,i)
      relerr = 0.0;
      [err,relerr] = calllib('libslepc', 'SVDComputeRelativeError', obj.pobj,i-1,relerr);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'SVDView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'SVDView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'SVDDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

