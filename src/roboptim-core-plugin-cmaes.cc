// Copyright (C) 2014 by Benjamin Chretien, CNRS-LIRMM.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#include <cstring>
#include <fstream>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <roboptim/core/function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-error.hh>

#include <roboptim/core/plugin/cmaes/cmaes.hh>

namespace roboptim
{
  namespace cmaes
  {

    CMAESSolver::CMAESSolver (const problem_t& problem) :
      parent_t (problem),
      n_ (problem.function ().inputSize ()),
      m_ (problem.function ().outputSize ()),
      x_ (n_),
      cost_ (problem.function ()),
      solverState_ (problem),
      tempInitials_ (),
      tempSignals_ ()
    {
      // Initialize this class parameters
      x_.setZero ();

      generateSignalsFile ();
    }

    CMAESSolver::~CMAESSolver () throw ()
    {
      // Remove temporary files
      if (boost::filesystem::exists (tempInitials_))
	boost::filesystem::remove(tempInitials_);

      if (boost::filesystem::exists (tempSignals_))
	boost::filesystem::remove(tempSignals_);
    }

    void CMAESSolver::generateInitialsFile()
    {
      tempInitials_ = boost::filesystem::unique_path ();

      std::stringstream ss;

      // Load optional starting point
      if (problem ().startingPoint ())
	{
	  x_ = *(problem ().startingPoint ());
	}

      // Problem dimension
      ss << "N " << n_ << std::endl;

      // Initial search point. Syntax: 1 == read one number
      ss << "initialX " << x_.size () << ":" << std::endl;

      for (Function::argument_t::Index i = 0; i < x_.size (); ++i)
	ss << x_[i] << " ";
      ss << std::endl;

      ss << "typicalX 1:" << std::endl
	 << "0." << std::endl;

      // Chose 1/4 of the search interval
      ss << "initialStandardDeviations " << problem ().function ().inputSize ()
         << ":" << std::endl;
      for (std::size_t i = 0;
           i < static_cast<std::size_t> (problem ().function ().inputSize ()); ++i)
	{
	  ss << (problem ().argumentBounds ()[i].second - problem ().argumentBounds ()[i].first)/4.
	     << " ";
	}

      std::ofstream temp_file;
      temp_file.open (tempInitials_.c_str ());
      temp_file << ss.str ();
      temp_file.close ();
    }

    void CMAESSolver::generateSignalsFile()
    {
      tempSignals_ = boost::filesystem::unique_path ();

      std::stringstream ss;

      // print every 200 seconds
      //ss << "print fewinfo     200" << std::endl;

      // clock: used processor time since start
      //ss << "print few+clock     2" << std::endl;

      std::ofstream temp_file;
      temp_file.open (tempSignals_.c_str ());
      temp_file << ss.str ();
      temp_file.close ();
    }

    void CMAESSolver::solve () throw ()
    {
      using namespace Eigen;

      generateInitialsFile ();

      // CMA-ES type struct or "object"
      cmaes_t evo;
      double* ar_objectives;
      double* const* ar_pop;
      double *ar_xfinal;

      // Initialize everything into the struct evo, 0 means default
      ar_objectives = cmaes_init (&evo, 0, NULL, NULL, 0, 0,
                                  tempInitials_.c_str ());
      //std::cout << cmaes_SayHello (&evo) << std::endl;

      Map<VectorXd> objectives (ar_objectives,
                                static_cast<VectorXd::Index> (evo.sp.lambda));

      // Write header and initial values
      cmaes_ReadSignals (&evo, "cmaes_signals.par");

      // Iterate until stop criterion holds
      while (!cmaes_TestForTermination (&evo))
	{
	  // Generate lambda new search points, sample population
	  // Do not change content of pop
	  ar_pop = cmaes_SamplePopulation (&evo);

	  // Here we may resample each solution point pop[i] until it
	  // becomes feasible. function is_feasible(...) needs to be
	  // user-defined.
	  // Assumptions: the feasible domain is convex, the optimum is
	  // not on (or very close to) the domain boundary, initialX is
	  // feasible and initialStandardDeviations are sufficiently small
	  // to prevent quasi-infinite looping.
	  /* for (i = 0; i < cmaes_Get(&evo, "popsize"); ++i)
	     while (!is_feasible(pop[i]))
	     cmaes_ReSampleSingle(&evo, i);
	  */

	  // Evaluate the new search points using fitfun
	  for (int i = 0; i < objectives.size (); ++i)
	    {
	      Map<VectorXd> pop_i (ar_pop[i],
				   static_cast<VectorXd::Index> (cmaes_Get(&evo, "dim")));
	      objectives[i] = cost_ (pop_i).sum ();
	    }

	  // Update the search distribution used for cmaes_SamplePopulation()
	  cmaes_UpdateDistribution (&evo, ar_objectives);

	  // Read instructions for printing output or changing termination conditions
	  cmaes_ReadSignals (&evo, tempSignals_.c_str ());

	  if (!callback_.empty ())
	    {
	      // TODO: find what to return when dealing with a population > 1.
	      solverState_.x() = Map<VectorXd> (ar_pop[0],
						static_cast<VectorXd::Index>
						(cmaes_Get(&evo, "dim")));
	      solverState_.cost () = objectives.minCoeff ();
	      callback_ (problem (), solverState_);
	    }
	}

      // Print termination reason
      //std::cout << "Stop:\n" << cmaes_TestForTermination (&evo) << std::endl;

      // Write final results
      cmaes_WriteToFile (&evo, "all", "allcmaes.dat");

      // Get best estimator for the optimum, xmean
      // "xbestever" might be used as well
      ar_xfinal = cmaes_GetNew (&evo, "xmean");

      Map<VectorXd> x_map (ar_xfinal,
                           static_cast<VectorXd::Index> (cmaes_Get(&evo, "dim")));

      Result result (n_, 1);
      result.x = x_map;
      result.value = problem ().function () (result.x);
      result_ = result;

      // Do something with final solution and finally release memory
      cmaes_exit (&evo);
      free (ar_xfinal);
    }

  } // namespace cmaes
} // end of namespace roboptim

extern "C"
{
  using namespace roboptim::cmaes;
  typedef CMAESSolver::parent_t solver_t;

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ();
  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ();
  ROBOPTIM_DLLEXPORT solver_t* create (const CMAESSolver::problem_t& pb);
  ROBOPTIM_DLLEXPORT void destroy (solver_t* p);

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ()
  {
    return sizeof (solver_t::problem_t);
  }

  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ()
  {
    return typeid (solver_t::problem_t::constraintsList_t).name ();
  }

  ROBOPTIM_DLLEXPORT solver_t* create (const CMAESSolver::problem_t& pb)
  {
    return new CMAESSolver (pb);
  }

  ROBOPTIM_DLLEXPORT void destroy (solver_t* p)
  {
    delete p;
  }
}