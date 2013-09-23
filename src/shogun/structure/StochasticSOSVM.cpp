/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <shogun/mathematics/Math.h>
#include <shogun/structure/StochasticSOSVM.h>

using namespace shogun;

CStochasticSOSVM::CStochasticSOSVM()
: CLinearStructuredOutputMachine()
{
	init();

	if (m_debug)
	{
		m_progress = new CSOSVMHelper();
		SG_REF(m_progress);
	}
}

CStochasticSOSVM::CStochasticSOSVM(
		CStructuredModel*  model,
		CStructuredLabels* labs,
		bool do_weighted_averaging,
		bool debug)
: CLinearStructuredOutputMachine(model, labs)
{
	init();
	m_do_weighted_averaging = do_weighted_averaging;
	m_debug = debug;

	if (m_debug)
	{
		m_progress = new CSOSVMHelper();
		SG_REF(m_progress);
	}
}

void CStochasticSOSVM::init()
{
	SG_ADD(&m_lambda, "lambda", "Regularization constant", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_iter, "num_iter", "Number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_debug, "debug", "Debug switch", MS_NOT_AVAILABLE);
	SG_ADD(&m_do_weighted_averaging, "do_weighted_averaging", "Do weighted averaging", MS_NOT_AVAILABLE);
	SG_ADD(&m_debug_multiplier, "debug_multiplier", "Debug multiplier", MS_NOT_AVAILABLE);
	SG_ADD(&m_rand_seed, "rand_seed", "Random seed", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_progress, "progress", "Training progress", MS_NOT_AVAILABLE);

	m_lambda = 1.0 / m_model->get_num_labels();
	m_num_iter = 50;
	m_debug = false;
	m_do_weighted_averaging = true;
	m_debug_multiplier = 0;
	m_rand_seed = 1;
	m_progress = NULL;
}

CStochasticSOSVM::~CStochasticSOSVM()
{
	SG_UNREF(m_progress);
}

EMachineType CStochasticSOSVM::get_classifier_type() const
{
	return CT_STOCHASTICSOSVM;
}

bool CStochasticSOSVM::train_machine(CFeatures* data)
{
	SG_DEBUG("Entering CStochasticSOSVM::train_machine.\n");
	if (data)
		set_features(data);

	// Initialize the model for training
	m_model->init_training();
	// Check that the scenary is correct to start with training
	m_model->check_training_setup();
	SG_DEBUG("The training setup is correct.\n");

	// Dimensionality of the joint feature space
	int32_t M = m_model->get_dim();
	// Number of training examples
	int32_t N = m_model->get_num_labels();

	SG_DEBUG("M=%d, N =%d.\n", M, N);

	// Initialize the weight vector
	m_w = SGVector<float64_t>(M);
	m_w.zero();

	SGVector<float64_t> w_avg;
	if (m_do_weighted_averaging)
		w_avg = m_w.clone();

	// logging 
	int32_t debug_iter = 1;
	if (m_debug_multiplier == 0)
	{
		debug_iter = N;
		m_debug_multiplier = 100;
	}

	CMath::init_random(m_rand_seed);

	// Main loop
	int32_t k = 0;
	for (int32_t pi = 0; pi < m_num_iter; ++pi)
	{
		for (int32_t si = 0; si < N; ++si)
		{
			// 1) Picking random example
			int32_t i = CMath::random(0, N-1);

			// 2) solve the loss-augmented inference for point i
			CResultSet* result = m_model->argmax(m_w, i);

			// 3) get the subgradient 
			// psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
			SGVector<float64_t> psi_i(M);
			SGVector<float64_t> w_s(M);

			SGVector<float64_t>::add(psi_i.vector, 
				1.0, result->psi_truth.vector, -1.0, result->psi_pred.vector, psi_i.vlen);

			w_s = psi_i.clone();
			w_s.scale(1.0 / (N*m_lambda));

			// 4) step-size gamma
			float64_t gamma = 1.0 / (k+1.0);

			// 5) finally update the weights
			SGVector<float64_t>::add(m_w.vector, 
				1.0-gamma, m_w.vector, gamma*N, w_s.vector, m_w.vlen);

			// 6) Optionally, update the weighted average
			if (m_do_weighted_averaging)
			{
				float64_t rho = 2.0 / (k+2.0);
				SGVector<float64_t>::add(w_avg.vector, 
					1.0-rho, w_avg.vector, rho, m_w.vector, w_avg.vlen);
			}
			
			k += 1;
			SG_UNREF(result);

			// Debug: compute objective and training error
			if (m_debug && k == debug_iter)
			{
				SGVector<float64_t> w_debug;
				if (m_do_weighted_averaging)
					w_debug = w_avg.clone();
				else
					w_debug = m_w.clone();

				float64_t primal = CSOSVMHelper::primal_objective(w_debug, m_model, m_lambda);
				float64_t train_error = CSOSVMHelper::average_loss(w_debug, m_model);

				SG_SPRINT("pass %d (iteration %d), SVM primal = %f, train_error = %f \n", 
					pi, k, primal, train_error);

				m_progress->add_debug_info(primal, (1.0*k) / N, train_error);

				debug_iter = CMath::min(debug_iter+N, debug_iter*(1+m_debug_multiplier/100));
			}
		}
	}
	
	if (m_do_weighted_averaging)
		m_w = w_avg.clone();

	if (m_debug)
		m_progress->terminate();

	SG_DEBUG("Leaving CStochasticSOSVM::train_machine.\n");
	return true;
}

float64_t CStochasticSOSVM::get_lambda() const
{
	return m_lambda;
}

void CStochasticSOSVM::set_lambda(float64_t lbda)
{
	m_lambda = lbda;
}

int32_t CStochasticSOSVM::get_num_iter() const
{
	return m_num_iter;
}

void CStochasticSOSVM::set_num_iter(int32_t num_iter)
{
	m_num_iter = num_iter;
}

int32_t CStochasticSOSVM::get_debug_multiplier() const
{
	return m_debug_multiplier;
}

void CStochasticSOSVM::set_debug_multiplier(int32_t multiplier)
{
	m_debug_multiplier = multiplier;
}

uint32_t CStochasticSOSVM::get_rand_seed() const
{
	return m_rand_seed;
}

void CStochasticSOSVM::set_rand_seed(uint32_t rand_seed)
{
	m_rand_seed = rand_seed;
}

CSOSVMHelper* CStochasticSOSVM::get_helper() const
{
	SG_REF(m_progress);
	return m_progress;
}
