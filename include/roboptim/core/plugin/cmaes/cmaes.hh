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

#ifndef ROBOPTIM_CORE_PLUGIN_CMAES_HH
# define ROBOPTIM_CORE_PLUGIN_CMAES_HH

# include <cstring>

# include <boost/mpl/vector.hpp>
# include <boost/filesystem.hpp>

# include <roboptim/core/solver.hh>
# include <roboptim/core/solver-state.hh>
# include <roboptim/core/differentiable-function.hh>

extern "C" {
#  include "cma/cmaes_interface.h"
}

namespace roboptim {
  namespace cmaes {
    /// \brief Solver implementing CMA-ES (Covariance Matrix Adaptation
    /// Evolution Strategy) algorithm.
    /// \see <a href="https://www.lri.fr/~hansen/cmaesintro.html">N. Hansen's webpage</a>.
    class CMAESSolver
      : public Solver<GenericFunction<EigenMatrixDense>, boost::mpl::vector<> >
    {
    public:
      /// \brief Parent type
      typedef Solver<GenericFunction<EigenMatrixDense>,
                     boost::mpl::vector<> > parent_t;
      /// \brief Cost function type
      typedef problem_t::function_t function_t;
      /// \brief type of result
      typedef parent_t::result_t result_t;
      /// \brief type of gradient
      typedef DifferentiableFunction::gradient_t gradient_t;
      /// \brief Size type
      typedef Function::size_type size_type;

      /// \brief Solver state
      typedef SolverState<parent_t::problem_t> solverState_t;

      /// \brief RobOptim callback
      typedef parent_t::callback_t callback_t;

      /// \brief Constructot by problem
      explicit CMAESSolver (const problem_t& problem);
      virtual ~CMAESSolver () throw ();
      /// \brief Solve the optimization problem
      virtual void solve () throw ();

      /// \brief Return the number of variables.
      size_type n () const
      {
	return n_;
      }

      /// \brief Return the number of functions.
      size_type m () const
      {
	return m_;
      }

      /// \brief Get the optimization parameters.
      Function::argument_t& parameter ()
      {
	return x_;
      }

      /// \brief Get the optimization parameters.
      const Function::argument_t& parameter () const
      {
	return x_;
      }

      /// \brief Get the cost function.
      const function_t& cost () const
      {
        return cost_;
      }

      /// \brief Set the callback called at each iteration.
      virtual void
      setIterationCallback (callback_t callback) throw (std::runtime_error)
      {
        callback_ = callback;
      }

      /// \brief Get the callback called at each iteration.
      const callback_t& callback () const throw ()
      {
        return callback_;
      }

    private:
      /// \brief Generate CMA-ES initialization file from problem parameters.
      virtual void generateInitialsFile ();

      /// \brief Generate CMA-ES signals file from problem parameters.
      virtual void generateSignalsFile ();

      /// \brief Compute the part of the cost function related to the
      /// argument bounds.
      function_t::result_t costBounds (const Function::argument_t& x);

    private:
      /// \brief Number of variables
      size_type n_;
      /// \brief Dimension of the cost function
      size_type m_;

      /// \brief Parameter of the function
      Function::argument_t x_;

      /// \brief Reference to cost function
      const function_t& cost_;

      /// \brief State of the solver at each iteration
      solverState_t solverState_;

      /// \brief Intermediate callback (called at each end of iteration).
      callback_t callback_;

      /// \brief Generated temporary initialization file.
      boost::filesystem::path tempInitials_;

      /// \brief Generated temporary signals file.
      boost::filesystem::path tempSignals_;
    }; // class CMAESSolver
  } // namespace cmaes
} // namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_CMAES_HH
