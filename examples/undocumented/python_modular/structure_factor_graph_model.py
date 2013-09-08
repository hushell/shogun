#!/usr/bin/env python

import numpy as np
import scipy

def structure_factor_graph_model (num_samples, show_data = False):
	from modshogun import Math
	from modshogun import FactorType, TableFactor, TableFactorType, FactorGraph
	from modshogun import FactorGraphObservation, FactorGraphLabels, FactorGraphFeatures
	from modshogun import FactorGraphModel, MAPInference, TREE_MAX_PROD
	from modshogun import DualLibQPBMSOSVM

	Math.init_random(17)

	cards = np.array([2,2], np.int32)
	w = np.array([0.3,0.5,1.0,0.2,0.05,0.6,-0.2,0.75])
	tid = int(0)
	ftype = TableFactorType(tid, cards, w)

	samples = FactorGraphFeatures(num_samples)
	labels = FactorGraphLabels(num_samples)

	import ipdb
	ipdb.set_trace()

	for i in xrange(num_samples):
		vc = np.array([2,2,2], np.int32)
		fg = FactorGraph(vc)

		data1 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)])
		vind1 = np.array([0,1], np.int32)
		fac1 = TableFactor(ftype, vind1, data1)
		fg.add_factor(fac1)

		data2 = np.array([2.0*Math.random(0.0,1.0)-1.0 for i in xrange(2)], np.float64)
		vind2 = np.array([1,2], np.int32)
		fac2 = TableFactor(ftype, vind2, data2)
		fg.add_factor(fac2)

		samples.add_sample(fg)
		fg.connect_components()
		fg.compute_energies()

		infer_met = MAPInference(fg, TREE_MAX_PROD)
		infer_met.inference()

		fg_obs = infer_met.get_structured_outputs()
		labels.add_label(fg_obs)

		if show_data:
			state = fg_obs.get_data()
			print state

	#model = HMSVMModel(features, labels, SMT_TWO_STATE, num_obs)

	#sosvm = DualLibQPBMSOSVM(model, labels, 5000.0)
	#sosvm.train()
	##print sosvm.get_w()

	#predicted = sosvm.apply(features)
	#evaluator = StructuredAccuracy()
	#acc = evaluator.evaluate(predicted, labels)
	##print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("Factor Graph Model")
	num_samples = 5
	structure_factor_graph_model(num_samples, True)
