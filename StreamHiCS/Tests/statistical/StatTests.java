package statistical;
import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

import statisticaltests.KolmogorovSmirnov;
import statisticaltests.StatisticalTest;
import statisticaltests.WelchT;
import streamdatastructures.DataBundle;

public class StatTests {

	private static final double epsilon = 0.005;
	private static Random r = new Random();
	private StatisticalTest statTest;

	@Test
	public void test1() {
		statTest = new WelchT();
		int numSamples = 10;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomUniformDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 1");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test2() {
		statTest = new WelchT();
		int numSamples = 100;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomUniformDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 2");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test3() {
		statTest = new WelchT();
		int numSamples = 1000;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomUniformDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 3");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test4() {
		statTest = new WelchT();
		int numSamples = 10000;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomUniformDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 4");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test5() {
		statTest = new WelchT();
		int numSamples = 10;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 5");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test6() {
		statTest = new WelchT();
		int numSamples = 100;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 6");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test7() {
		statTest = new WelchT();
		int numSamples = 1000;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 7");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test8() {
		statTest = new WelchT();
		int numSamples = 10000;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 8");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test9() {
		statTest = new WelchT();
		int numSamples = 10;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 9");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test10() {
		statTest = new WelchT();
		int numSamples = 100;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 10");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test11() {
		statTest = new WelchT();
		int numSamples = 1000;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 11");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test12() {
		statTest = new WelchT();
		int numSamples = 10000;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 12");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test13() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 10;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomUniformDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 13");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test14() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 100;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomUniformDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 14");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test15() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 10;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 15");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test16() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 100;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 16");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test17() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 1000;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 17");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test18() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 10000;
		double[] data1 = createRandomGaussianDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 18");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test19() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 10;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 19");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test20() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 100;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 20");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test21() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 1000;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 21");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	@Test
	public void test22() {
		statTest = new KolmogorovSmirnov();
		int numSamples = 10000;
		double[] data1 = createRandomUniformDataSet(numSamples);
		double[] data2 = createRandomGaussianDataSet(numSamples);
		double[] weights1 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights1[i] = 1;
		}
		double[] weights2 = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			weights2[i] = 1;
		}
		System.out.println("Test 22");
		assertTrue(carryOutTest(data1, weights1, data2, weights2));
	}

	private double[] createRandomUniformDataSet(int numSamples) {
		double[] data = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			data[i] = r.nextDouble();
		}
		return data;
	}

	private double[] createRandomGaussianDataSet(int numSamples) {
		double[] data = new double[numSamples];
		for (int i = 0; i < numSamples; i++) {
			data[i] = r.nextGaussian();
		}
		return data;
	}

	private boolean carryOutTest(double[] data1, double[] weights1, double[] data2, double[] weights2) {
		DataBundle dataBundle1 = new DataBundle(data1, weights1);
		DataBundle dataBundle2 = new DataBundle(data2, weights2);

		double normal = statTest.calculateDeviation(data1, data2);
		double weighted = statTest.calculateWeightedDeviation(dataBundle1, dataBundle2);

		System.out.println("Normal deviation: " + normal + ". Weighted deviation: " + weighted);

		return Math.abs(normal - weighted) <= epsilon;
	}
}
