classdef SlepcPEP < PetscObject
%
%   SlepcPEP - a SLEPc polynomial eigenvalue solver object
%
%   Creation:
%     eps = SlepcPEP();
%     PEP.SetType('toar');
%     PEP.SetOperators({K,C,M});
%     PEP.SetFromOptions();

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
    GENERAL=1;
    HERMITIAN=2;
    GYROSCOPIC=3;

    LARGEST_MAGNITUDE=1;
    SMALLEST_MAGNITUDE=2;
    LARGEST_REAL=3;
    SMALLEST_REAL=4;
    LARGEST_IMAGINARY=5;
    SMALLEST_IMAGINARY=6;
    TARGET_MAGNITUDE=7;
    TARGET_REAL=8;
    TARGET_IMAGINARY=9;

    BASIS_MONOMIAL=0;
    BASIS_CHEBYSHEV1=1;
    BASIS_CHEBYSHEV2=2;
    BASIS_LEGENDRE=3;
    BASIS_LAGUERRE=4;
    BASIS_HERMITE=5;
  end
  methods
    function obj = SlepcPEP(pid,flag)
      if (nargin > 1)
        %  SelpcPEP(pid,'pobj') uses an already existing SLEPc PEP object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'PEPCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'PEPSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'PEPSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'PEPSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj)
      err = calllib('libslepc', 'PEPSolve', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,C)
      if ~iscell(C), error('Argument of PEP.SetOperators must be a cell array'), end
      n = length(C);
      M = [];
      for i=1:n, M = [M, C{i}.pobj]; end
      err = calllib('libslepc', 'PEPSetOperators', obj.pobj, n, M);PetscCHKERRQ(err);
    end
    function err = SetProblemType(obj,t)
      err = calllib('libslepc', 'PEPSetProblemType', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetWhichEigenpairs(obj,t)
      err = calllib('libslepc', 'PEPSetWhichEigenpairs', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetTarget(obj,t)
      err = calllib('libslepc', 'PEPSetTarget', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetTolerances(obj,t,mx)
      if (nargin == 2) mx = 0; end
      err = calllib('libslepc', 'PEPSetTolerances', obj.pobj,t,mx);PetscCHKERRQ(err);
    end
    function err = SetDimensions(obj,nev,ncv,mpd)
      if (nargin < 3) ncv = 0; end
      if (nargin < 4) mpd = 0; end
      err = calllib('libslepc', 'PEPSetDimensions', obj.pobj,nev,ncv,mpd);PetscCHKERRQ(err);
    end
    function [nconv,err] = GetConverged(obj)
      nconv = 0;
      [err,nconv] = calllib('libslepc', 'PEPGetConverged', obj.pobj,nconv);PetscCHKERRQ(err);
    end
    function [lambda,v,err] = GetEigenpair(obj,i,xr,xi)
      lambda = 0.0;
      img = 0.0;
      if (nargin < 3)
        x = 0;
      else
        x = xr.pobj;
      end
      if (nargin < 4)
        y = 0;
      else
        y = xi.pobj;
      end
      if (nargout > 1 && (x==0 || y==0))
        [err,pid] = calllib('libslepc', 'PEPGetOperators', obj.pobj,0,0);PetscCHKERRQ(err);
        A = PetscMat(pid,'pobj');
        n = A.GetSize();
      end
      freexr = 0;
      freexi = 0;
      if (nargout > 1 && x==0)
        xr = PetscVec();
        freexr = 1;
        xr.SetType('seq');
        xr.SetSizes(n,n);
        x = xr.pobj;
      end
      if (nargout > 1 && y==0)
        xi = PetscVec();
        freexi = 1;
        xi.SetType('seq');
        xi.SetSizes(n,n);
        y = xi.pobj;
      end
      [err,lambda,img] = calllib('libslepc', 'PEPGetEigenpair', obj.pobj,i-1,lambda,img,x,y);PetscCHKERRQ(err);
      if img~=0.0, lambda = lambda+j*img; end
      if (nargout > 1)
        if (x ~= 0)
          vr = xr(:);
        else
          vr = 0;
        end
        if (y ~= 0)
          vi = xi(:);
        else
          vi = 0;
        end
        v = vr+j*vi;
        if (freexr)
          xr.Destroy();
        end
        if (freexi)
          xi.Destroy();
        end
      end
    end
    function [relerr,err] = ComputeRelativeError(obj,i)
      relerr = 0.0;
      [err,relerr] = calllib('libslepc', 'PEPComputeRelativeError', obj.pobj,i-1,relerr);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'PEPView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'PEPView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'PEPDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

