package contrast;

import static org.junit.Assert.*;

import org.apache.commons.math3.util.MathArrays;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import fullsystem.Contrast;
import moa.clusterers.clustree.ClusTree;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.MicroclusterAdapter;
import streamdatastructures.SlidingWindow;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import subspace.Subspace;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * Testing the contrast measure in 2D space. It represents a static test, as
 * long as a {@link SlidingWindow} is used, since we fill it completely and then
 * carry out the test. BASED ON THE KOLMOGOROV-SMIRNOV-TEST. As max we expect
 * values of 0.75
 * 
 * @author Vincent
 *
 */
public class ContrastTest {

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

	private final static String method = "ClusTreeMC";

	private static final int m = 100;
	private static double alpha;
	/**
	 * The allowed error to the correct contrast value.
	 */
	private static final double epsilon = 0.1;
	private static double targetLowContrast;
	private static double targetMiddleContrast;
	private static double targetHighContrast;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			adapter = new SlidingWindowAdapter(2, numInstances);

			targetLowContrast = 0;
			targetMiddleContrast = 0.2;
			targetHighContrast = 0.5;

		} else if (method.equals("adaptiveCentroids")) {
			alpha = 0.1;
			double fadingLambda = 0.005;
			double radius = 0.2;
			double weightThreshold = 0.1;
			double learningRate = 0.1;
			
			adapter = new CentroidsAdapter(fadingLambda, radius, weightThreshold, learningRate);
			contrastEvaluator = new Contrast(m, alpha, adapter);

			targetLowContrast = 0.1;
			targetMiddleContrast = 0.3;
			targetHighContrast = 0.6;

		} else if (method.equals("DenStreamMC")) {
			alpha = 0.1;
			WithDBSCAN mcs = new WithDBSCAN();
			mcs.speedOption.setValue(100);
			mcs.epsilonOption.setValue(1);
			mcs.betaOption.setValue(0.2);
			mcs.muOption.setValue(10);
			mcs.lambdaOption.setValue(0.005);
			mcs.resetLearningImpl();
			adapter = new MicroclusterAdapter(mcs);

			targetLowContrast = 0.2;
			targetMiddleContrast = 0.4;
			targetHighContrast = 0.6;
			
		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.1;
			ClusTree mcs = new ClusTree();
			mcs.resetLearningImpl();
			adapter = new MicroclusterAdapter(mcs);

			targetLowContrast = 0.15;
			targetMiddleContrast = 0.35;
			targetHighContrast = 0.7;
			
		} else {
			adapter = null;
		}

		contrastEvaluator = new Contrast(m, alpha, adapter);
		subspace = new Subspace(0, 1);
	}

	@Before
	public void setUp() throws Exception {
		contrastEvaluator.clear();
	}

	@Test
	public void contrastTest1() {
		// Contrast should be high
		DoubleFunction f = x -> x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 1 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest2() {
		// Contrast should be high
		DoubleFunction f = x -> -x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 2 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest3() {
		// Contrast should be high
		DoubleFunction f = x -> 2 * x + 1;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 3 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest4() {
		// Contrast should be high
		DoubleFunction f = x -> 0.1 * x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 4 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest5() {
		// Contrast should be high
		DoubleFunction f = x -> 20 * x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 5 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest6() {
		// Quadratic relationship, contrast should be high
		DoubleFunction f = x -> x * x;
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 6 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest7() {
		// Contrast should be high
		DoubleFunction f = x -> x * x * x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 7 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest8() {
		// Contrast should be high
		DoubleFunction f = x -> Math.pow(Math.E, x);
		double contrast = carryOutTest(f, 0, 10);
		System.out.println("Test 8 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest9() {
		// Contrast should be high
		DoubleFunction f = x -> Math.pow(Math.E, -x);
		double contrast = carryOutTest(f, 0, 10);
		System.out.println("Test 9 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest10() {
		// Contrast should be high
		DoubleFunction f = x -> 1 - Math.pow(Math.E, -x);
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 10 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest11() {
		// Contrast should be 0
		DoubleFunction f = x -> 0;
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 11 : " + contrast);
		assertTrue(Math.abs(0 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest12() {
		// Contrast should be 0, since random
		DoubleFunction f = x -> Math.random();
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 12 : " + contrast);
		assertTrue(Math.abs(targetLowContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest13() {
		// Contrast should be > 0
		DoubleFunction f = x -> x + Math.random();
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 13 : " + contrast);
		assertTrue(Math.abs(targetMiddleContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest14() {
		// Contrast should be high
		DoubleFunction f = x -> Math.sin(x);
		double contrast = carryOutTest(f, 0, 2*Math.PI);
		System.out.println("Test 14 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest15() {
		// Contrast should be high
		DoubleFunction f = x -> x + Math.sin(x);
		double contrast = carryOutTest(f, 0, 2*Math.PI);
		System.out.println("Test 15 : " + contrast);
		assertTrue(Math.abs(targetHighContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest16() {
		contrastEvaluator.clear();
		// Contrast should be 0
		double x = 0;
		double y = 0;
		for (int i = 0; i < numInstances; i++) {
			x = Math.random();
			y = Math.random();
			addInstance(x, y);
		}
		double contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
		System.out.println("Test 16 : " + contrast);
		assertTrue(Math.abs(targetLowContrast - contrast) <= epsilon);
	}

	@Test
	public void contrastTest17() {
		// Contrast should be 0
		double x = 0;
		double y = 1;
		for (int i = 0; i < numInstances; i++) {
			addInstance(x, y);
		}
		double contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
		System.out.println("Test 17 : " + contrast);
		assertTrue(Math.abs(0 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest18() {
		// Contrast should be > 0
		double x = 0;
		double y = 0;
		for (int i = 0; i < numInstances; i++) {
			x = Math.random();
			y = x + Math.random();
			addInstance(x, y);
		}
		double contrast = contrastEvaluator.evaluateSubspaceContrast(subspace);
		System.out.println("Test 18 : " + contrast);
		assertTrue(Math.abs(targetMiddleContrast - contrast) <= epsilon);
	}

	private double carryOutTest(DoubleFunction f, double lowerBound, double upperBound) {
		contrastEvaluator.clear();
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
		return contrastEvaluator.evaluateSubspaceContrast(subspace);
	}

	private void addInstance(double x, double y) {
		Instance inst = new DenseInstance(2);
		inst.setValue(0, x);
		inst.setValue(1, y);
		contrastEvaluator.add(inst);
	}
}
