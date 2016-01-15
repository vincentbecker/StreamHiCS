package streamhics_contrast;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import environment.Stopwatch;
import fullsystem.Contrast;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.SummarisationAdapter;
import subspace.Subspace;
import weka.core.DenseInstance;
import weka.core.Instance;

public class NumberDims {

	private static Contrast contrastEvaluator;
	private static SummarisationAdapter adapter;
	/**
	 * The number of instances used in the test.
	 */
	private static final int numInstances = 10000;
	/**
	 * The {@link Subspace} which should be tested.
	 */
	private static Subspace subspace;

	private static final int m = 100;
	private static double alpha;
	
	@Before
	public void setUp() throws Exception {
		alpha = 0.1;
		int horizon = 5000;
		double radius = 0.001;
		double learningRate = 0.1;

		adapter = new CentroidsAdapter(horizon, radius, learningRate, "radius");
		
		contrastEvaluator = new Contrast(m, alpha, adapter);
		subspace = new Subspace();
	}

	@Test
	public void dimensionalityTest() {
		Stopwatch stopwatch = new Stopwatch();
		for (int numberDims = 2; numberDims <= 100; numberDims++) {
			contrastEvaluator.clear();
			subspace.clear();
			stopwatch.reset();
			for(int i = 0; i < numberDims; i++){
				subspace.addDimension(i);
			}
			for (int i = 0; i < numInstances; i++) {
				double[] values = new double[numberDims];
				double x = Math.random();
				for (int j = 0; j < numberDims; j++) {
					values[j] = x;
				}
				addInstance(values);
			}
			//System.out.println(contrastEvaluator.getNumberOfElements());
			//System.out.print(numberDims + ",");
			stopwatch.start("Evaluation");
			double contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
			stopwatch.stop("Evaluation");
			System.out.println(numberDims + "," + contrast + "," + stopwatch.getTime("Evaluation"));
		}
		fail();
	}

	private void addInstance(double[] values) {
		Instance inst = new DenseInstance(values.length);
		for (int i = 0; i < values.length; i++) {
			inst.setValue(i, values[i]);
		}
		contrastEvaluator.add(inst);
	}
	
}
