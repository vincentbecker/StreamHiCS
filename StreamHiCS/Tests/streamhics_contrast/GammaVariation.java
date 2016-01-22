package streamhics_contrast;

import org.apache.commons.math3.util.MathArrays;
import org.junit.Before;
import org.junit.Test;

import clustree.ClusTree;
import environment.Stopwatch;
import fullsystem.Contrast;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import subspace.Subspace;
import weka.core.DenseInstance;
import weka.core.Instance;

public class GammaVariation {

	interface DoubleFunction {
		double function(double x);
	}

	/**
	 * The contrast measure.
	 */
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
	private static final double alpha = 0.1;

	Stopwatch stopwatch;

	@Before
	public void setUp() throws Exception {
		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(5000);
		mcs.resetLearning();
		adapter = new MicroclusteringAdapter(mcs);

		contrastEvaluator = new Contrast(m, alpha, adapter);
		subspace = new Subspace(0, 1);
		stopwatch = new Stopwatch();
	}

	@Test
	public void test() {
		double lowerBound = 0;
		double upperBound = 1;
		double gamma = 0;
		double step = (upperBound - lowerBound) / numInstances;
		for (int a = 1; a <= 100; a++) {
			stopwatch.reset();
			gamma = 0.01 * a;
			int[] indexes = new int[numInstances];
			for (int i = 0; i < numInstances; i++) {
				indexes[i] = i;
			}
			MathArrays.shuffle(indexes);
			double x = 0;
			double y = 0;
			for (int i = 0; i < numInstances; i++) {
				x = lowerBound + indexes[i] * step;
				y = (1 - gamma) * x + gamma * Math.random();
				addInstance(x, y);
			}

			stopwatch.start("Evaluation");
			double contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
			stopwatch.stop("Evaluation");
			System.out.println(gamma + "," + contrast + "," + stopwatch.getTime("Evaluation"));
		}
	}

	private void addInstance(double x, double y) {
		Instance inst = new DenseInstance(2);
		inst.setValue(0, x);
		inst.setValue(1, y);
		contrastEvaluator.add(inst);
	}

}
