/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#include <shogun/structure/MAPInference.h>
#include <shogun/structure/BeliefPropagation.h>

#include <string>

using namespace shogun;

CMAPInference::CMAPInference() : CSGObject()
{
	SG_UNSTABLE("CMAPInference::CMAPInference()", "\n");

	init();
}

CMAPInference::CMAPInference(CFactorGraph* fg, EMAPInferType inference_method)
	: CSGObject()
{
	init();
	m_fg = fg;

	REQUIRE(fg != NULL, "%s::CMAPInference(): fg cannot be NULL!\n", get_name());

	switch(inference_method)
	{
		case TREE_MAX_PROD:
			m_infer_impl = new CTreeMaxProduct(fg);
			break;
		case LOOPY_MAX_PROD:
			SG_ERROR("%s::CMAPInference(): LoopyMaxProduct has not been implemented!\n", 
				get_name());
			break;
		case LP_RELAXATION:
			SG_ERROR("%s::CMAPInference(): LPRelaxation has not been implemented!\n", 
				get_name());
			break;
		case TRWS_MAX_PROD:
			SG_ERROR("%s::CMAPInference(): TRW-S has not been implemented!\n", 
				get_name());
			break;
		case ITER_COND_MODE:
			SG_ERROR("%s::CMAPInference(): ICM has not been implemented!\n", 
				get_name());
			break;
		case NAIVE_MEAN_FIELD:
			SG_ERROR("%s::CMAPInference(): NaiveMeanField has not been implemented!\n", 
				get_name());
			break;
		case STRUCT_MEAN_FIELD:
			SG_ERROR("%s::CMAPInference(): StructMeanField has not been implemented!\n", 
				get_name());
			break;
		default:
			SG_ERROR("%s::CMAPInference(): unsupported inference method!\n", 
				get_name());
			break;
	}

	SG_REF(m_infer_impl);
	SG_REF(m_fg);
}

CMAPInference::~CMAPInference()
{
	SG_UNREF(m_infer_impl);
	SG_UNREF(m_outputs);
	SG_UNREF(m_fg);
}

void CMAPInference::init()
{
	SG_ADD((CSGObject**)&m_fg, "fg", "factor graph", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_outputs, "outputs", "Structured outputs", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_infer_impl, "infer_impl", "Inference implementation", MS_NOT_AVAILABLE);
	SG_ADD(&m_energy, "energy", "Minimized energy", MS_NOT_AVAILABLE);

	m_outputs = NULL;
	m_infer_impl = NULL;
	m_fg = NULL;
	m_energy = 0;
}

void CMAPInference::inference()
{
	SGVector<int32_t> assignment(m_fg->get_cardinalities().size());
	assignment.zero();
	m_energy = m_infer_impl->inference(assignment);

	SG_UNREF(m_outputs);
	m_outputs = new CFactorGraphObservation(assignment); // already ref() in constructor
	SG_REF(m_outputs);
}

CFactorGraphObservation* CMAPInference::get_structured_outputs() const
{
	SG_REF(m_outputs);
	return m_outputs;
}

float64_t CMAPInference::get_energy() const
{
	return m_energy;
}

//-----------------------------------------------------------------

CMAPInferImpl::CMAPInferImpl() : CSGObject()
{ 
	register_parameters();
}

CMAPInferImpl::CMAPInferImpl(CFactorGraph* fg) 
	: CSGObject()
{ 
	register_parameters();
	m_fg = fg;
}

CMAPInferImpl::~CMAPInferImpl() 
{ 
}

void CMAPInferImpl::register_parameters()
{
	SG_ADD((CSGObject**)&m_fg, "m_fg", 
		"Factor graph pointer", MS_NOT_AVAILABLE);

	m_fg = NULL;
}

