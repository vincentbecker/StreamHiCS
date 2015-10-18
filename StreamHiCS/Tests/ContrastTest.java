import static org.junit.Assert.*;

import org.apache.commons.math3.util.MathArrays;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import contrast.CentroidContrast;
import contrast.Contrast;
import streamDataStructures.SlidingWindow;
import streamDataStructures.Subspace;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * Testing the contrast measure in 2D space. It represents a static test, as
 * long as a {@link SlidingWindow} is used, since we fill it completely and then
 * carry out the test. BASED ON THE KOLMOGOROV-SMIRNOV-TEST. As max we expect values of 0.5
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
	/**
	 * The number of instances used in the test.
	 */
	private static final int numInstances = 10000;
	/**
	 * The allowed error to the correct contrast value.
	 */
	private static final double epsilon = 0.1;
	/**
	 * The {@link Subspace} which should be tested.
	 */
	private static Subspace subspace;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		//contrastEvaluator = new SlidingWindowContrast(null, 2, numInstances + 1, 20, 0.4);
		contrastEvaluator  = new CentroidContrast(null, 2, 20, 0.4, numInstances + 1);
		subspace = new Subspace(0, 1);
	}

	@Before
	public void setUp() throws Exception {
		contrastEvaluator.clear();
	}

	@Test
	public void contrastTest1() {
		// Contrast should be 0.5
		DoubleFunction f = x -> x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 1 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest2() {
		// Contrast should be 0.5
		DoubleFunction f = x -> -x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 2 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest3() {
		// Contrast should be 0.5
		DoubleFunction f = x -> 2 * x + 1;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 3 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest4() {
		// Contrast should be 0.5
		DoubleFunction f = x -> 0.1 * x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 4 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest5() {
		// Contrast should be 0.5
		DoubleFunction f = x -> 20 * x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 5 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest6() {
		// Quadratic relationship, contrast should be 0.71
		DoubleFunction f = x -> x * x;
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 6 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest7() {
		// Contrast should be 0.5
		DoubleFunction f = x -> x * x * x;
		double contrast = carryOutTest(f, -1, 1);
		System.out.println("Test 7 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest8() {
		// Contrast should be 0.5
		DoubleFunction f = x -> Math.pow(Math.E, x);
		double contrast = carryOutTest(f, 0, 10);
		System.out.println("Test 8 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest9() {
		// Contrast should be 0.5
		DoubleFunction f = x -> Math.pow(Math.E, -x);
		double contrast = carryOutTest(f, 0, 10);
		System.out.println("Test 9 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest10() {
		// Contrast should be 0.5
		DoubleFunction f = x -> 1 - Math.pow(Math.E, -x);
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 10 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
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
		assertTrue(Math.abs(0 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest13() {
		// Contrast should be > 0
		DoubleFunction f = x -> x + Math.random();
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 13 : " + contrast);
		assertTrue(Math.abs(0.3 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest14() {
		// Contrast should be 0.5
		DoubleFunction f = x -> Math.sin(x);
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 14 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest15() {
		// Contrast should be > 0
		DoubleFunction f = x -> x + Math.sin(x);
		double contrast = carryOutTest(f, 0, 1);
		System.out.println("Test 15 : " + contrast);
		assertTrue(Math.abs(0.5 - contrast) <= epsilon);
	}

	@Test
	public void contrastTest16() {
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
		assertTrue(Math.abs(0 - contrast) <= epsilon);
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
		assertTrue(Math.abs(0.3 - contrast) <= epsilon);
	}

	private double carryOutTest(DoubleFunction f, double lowerBound, double upperBound) {
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
