import static org.junit.Assert.*;

import org.apache.commons.math3.util.MathArrays;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import centroids.DensityChecker;
import contrast.Callback;
import contrast.CentroidContrast;
import contrast.Contrast;
import weka.core.DenseInstance;
import weka.core.Instance;

public class DensityCheckTest {

	interface DoubleFunction {
		double function(double x);
	}

	/**
	 * The contrast measure.
	 */
	private static Contrast contrastEvaluator;
	/**
	 * The number of instances used in the test.
	 */
	private static final int numInstances = 10000;

	private static Callback callback;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		// contrastEvaluator = new SlidingWindowContrast(null, 2, numInstances +
		// 1, 20, 0.4)
		callback = new Callback() {

			@Override
			public void onAlarm() {
				System.out.println("Alarm.");
			}

		};
		contrastEvaluator = new CentroidContrast(callback, 2, 20, 0.4, 0.01, 0.2, numInstances, 0.1, 0.2,
				new DensityChecker(25, 1));
	}

	@Before
	public void setUp() throws Exception {
		contrastEvaluator.clear();
	}

	@Test
	public void test1() {
		DoubleFunction f = x -> x;
		addInstances(f, 0, 1);
		f = x -> -x;
		addInstances(f, 0, 1);

		fail("Not yet implemented");
	}

	@Test
	public void test2() {
		int count = 0;
		double x = 0;
		double y = 0;
		while (count <= numInstances) {
			x = 0.1 * Math.random() - 0.5;
			y = 0.1 * Math.random() - 0.5;
			addInstance(x, y);
			x = 0.1 * Math.random() + 0.5;
			y = 0.1 * Math.random() + 0.5;
			addInstance(x, y);
			count++;
		}

		fail("Not yet implemented");
	}

	private void addInstances(DoubleFunction f, double lowerBound, double upperBound) {
		double step = (upperBound - lowerBound) / numInstances;
		int[] indexes = new int[numInstances];
		for (int i = 0; i < numInstances; i++) {
			indexes[i] = i;
		}
		MathArrays.shuffle(indexes);
		double x = 0;
		double y = 0;
		for (int i = 0; i < numInstances; i++) {
			x = lowerBound + indexes[i] * step;
			y = f.function(x);
			addInstance(x, y);
		}
	}

	private void addInstance(double x, double y) {
		Instance inst = new DenseInstance(2);
		inst.setValue(0, x);
		inst.setValue(1, y);
		contrastEvaluator.add(inst);
	}
}
