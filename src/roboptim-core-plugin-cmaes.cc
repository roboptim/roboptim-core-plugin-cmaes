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
#include <cmath>

#include <boost/assign/list_of.hpp>

#include <roboptim/core/function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-error.hh>

#include <roboptim/core/plugin/cmaes/cmaes.hh>


#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)	\
  do {							\
    this->parameters_[KEY].description = DESCRIPTION;	\
    this->parameters_[KEY].value = VALUE;		\
  } while (0)

namespace roboptim
{
  namespace cmaes
  {
    namespace detail
    {
      void remove_temp (const boost::filesystem::path& tmp_file)
      {
	if (boost::filesystem::exists (tmp_file))
          boost::filesystem::remove (tmp_file);
      }
    } // end of namespace detail

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

      DEFINE_PARAMETER ("max-iterations", "number of iterations", 3000);
      DEFINE_PARAMETER ("cmaes.lambda", "number of offspring (samplesize)",
                        4 + (int)(3 * std::log ((double)n_)));
      DEFINE_PARAMETER ("cmaes.output_file", "CMA-ES output file", "");
      DEFINE_PARAMETER ("cmaes.log_file", "CMA-ES log file", "");
    }

    CMAESSolver::~CMAESSolver () throw ()
    {
      // Remove temporary files
      detail::remove_temp (tempInitials_);
      detail::remove_temp (tempSignals_);
    }

    CMAESSolver::function_t::result_t CMAESSolver::costBounds
    (const Function::argument_t& x)
    {
      function_t::result_t res (problem ().function ().inputSize ());

      for (std::size_t i = 0;
	   i < static_cast<std::size_t> (res.size ());
	   ++i)
	{
	  CMAESSolver::vector_t::Index i_ =
	    static_cast<CMAESSolver::vector_t::Index> (i);

	  // Check min
	  if (x[i_] < problem ().argumentBounds ()[i].first)
	    res[i_] = std::fabs (x[i_] - problem ().argumentBounds ()[i].first);
	  // Check max
	  else if (x[i_] > problem ().argumentBounds ()[i].second)
	    res[i_] = std::fabs (x[i_] - problem ().argumentBounds ()[i].second);
	}
      return res;
    }

    void CMAESSolver::generateInitialsFile()
    {
      detail::remove_temp (tempInitials_);

      tempInitials_ = boost::filesystem::unique_path ("cmaes_initials_%%%%-%%%%.par");

      std::stringstream ss;

      // Load optional starting point
      if (problem ().startingPoint ())
	{
	  x_ = *(problem ().startingPoint ());
	}

      /// FIRST: mandatory parameters

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
      ss << std::endl;

      /// SECOND: optional parameters

      ss << "stopMaxIter " << parameters()["max-iterations"].value;

      std::ofstream temp_file;
      temp_file.open (tempInitials_.c_str ());
      temp_file << ss.str ();
      temp_file.close ();
    }

    void CMAESSolver::generateSignalsFile()
    {
      detail::remove_temp (tempSignals_);

      tempSignals_ = boost::filesystem::unique_path ("cmaes_signals_%%%%-%%%%.par");

      std::stringstream ss;

      // number > 0 switches checking on, Check_Eigen() is O(n^3)!
      ss << "checkeigen 0" << std::endl;

      std::ofstream temp_file;
      temp_file.open (tempSignals_.c_str ());
      temp_file << ss.str ();
      temp_file.close ();
    }

    void CMAESSolver::solve () throw ()
    {
      using namespace Eigen;

      generateInitialsFile ();
      generateSignalsFile ();

      // CMA-ES type struct or "object"
      cmaes_t evo;
      double* ar_objectives;
      double* const* ar_pop;
      double *ar_xfinal;

      // Initialize everything into the struct evo, 0 means default
      int lambda = boost::get<int> (parameters ()["cmaes.lambda"].value);
      ar_objectives = cmaes_init (&evo, 0, NULL, NULL, 0, lambda,
                                  tempInitials_.c_str ());

      // If using a log file
      if (parameters ().find ("cmaes.log_file") != parameters ().end ())
    {
          std::string filename = boost::get<std::string>
	    (parameters ()["cmaes.log_file"].value);
          if (filename.compare ("") != 0)
	    {
	      std::stringstream ss;

	      ss << cmaes_SayHello (&evo) << std::endl;

	      std::ofstream log_file;
          log_file.open (filename.c_str (),
                         std::fstream::out | std::fstream::app);
	      log_file << ss.str ();
	      log_file.close ();
	    }
	}

      Map<VectorXd> objectives (ar_objectives,
                                static_cast<VectorXd::Index> (evo.sp.lambda));

      // Write header and initial values
      cmaes_ReadSignals (&evo, tempSignals_.c_str ());

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
	      // Map the population to Eigen
	      Map<VectorXd> pop_i (ar_pop[i],
				   static_cast<VectorXd::Index> (cmaes_Get(&evo, "dim")));

	      // Compute the cost: actual cost + bound-related cost
	      objectives[i] = cost_ (pop_i).sum () + costBounds (pop_i).sum ();
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

      // Output file
      if (parameters ().find ("cmaes.output_file") != parameters ().end ())
	{
          // Display the traces of CMA-ES
          std::string filename = boost::get<std::string>
	    (parameters ()["cmaes.output_file"].value);
          if (filename.compare ("") != 0)
	    {
              // Write final results
              cmaes_WriteToFile (&evo, "all", filename.c_str ());
	    }
	}

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

  } // end of namespace cmaes
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
