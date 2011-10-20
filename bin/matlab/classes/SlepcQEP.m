classdef SlepcQEP < PetscObject
%
%   SlepcQEP - a SLEPc quadratic eigenvalue solver object
%
%   Creation:
%     eps = SlepcQEP();
%     QEP.SetType('linear');
%     QEP.SetOperators(M,C,K);
%     (optional) QEP.SetProblemType(...);
%     QEP.SetFromOptions();
 
%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%  SLEPc - Scalable Library for Eigenvalue Problem Computations
%  Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain
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
  end
  methods
    function obj = SlepcQEP(pid,flag)
      if (nargin > 1) 
        %  SelpcQEP(pid,'pobj') uses an already existing SLEPc QEP object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'QEPCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'QEPSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'QEPSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'QEPSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj)
      err = calllib('libslepc', 'QEPSolve', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,M,C,K)
      err = calllib('libslepc', 'QEPSetOperators', obj.pobj,M.pobj,C.pobj,K.pobj);PetscCHKERRQ(err);
    end
    function err = SetProblemType(obj,t)
      err = calllib('libslepc', 'QEPSetProblemType', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetWhichEigenpairs(obj,t)
      err = calllib('libslepc', 'QEPSetWhichEigenpairs', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetTolerances(obj,t,mx)
      if (nargin == 2) mx = 0; end
      err = calllib('libslepc', 'QEPSetTolerances', obj.pobj,t,mx);PetscCHKERRQ(err);
    end
    function err = SetDimensions(obj,nev,ncv,mpd)
      if (nargin < 3) ncv = 0; end
      if (nargin < 4) mpd = 0; end
      err = calllib('libslepc', 'QEPSetDimensions', obj.pobj,nev,ncv,mpd);PetscCHKERRQ(err);
    end
    function err = SetScaleFactor(obj,t)
      err = calllib('libslepc', 'QEPSetScaleFactor', obj.pobj,t);PetscCHKERRQ(err);
    end
    function [nconv,err] = GetConverged(obj)
      nconv = 0;
      [err,nconv] = calllib('libslepc', 'QEPGetConverged', obj.pobj,nconv);PetscCHKERRQ(err);
    end
    function [lambda,err] = GetEigenpair(obj,i)
      lambda = 0.0;
      img = 0.0;
      [err,lambda,img] = calllib('libslepc', 'QEPGetEigenpair', obj.pobj,i-1,lambda,img,0,0);PetscCHKERRQ(err);
      if img~=0.0, lambda = lambda+j*img; end
    end
    function [relerr,err] = ComputeRelativeError(obj,i)
      relerr = 0.0;
      [err,relerr] = calllib('libslepc', 'QEPComputeRelativeError', obj.pobj,i-1,relerr);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'QEPView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'QEPView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'QEPDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
