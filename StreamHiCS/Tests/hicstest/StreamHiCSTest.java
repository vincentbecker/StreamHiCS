package hicstest;

import static org.junit.Assert.*;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.CSVReader;
import environment.Evaluator;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;
import moa.clusterers.clustream.Clustream;
import moa.clusterers.clustree.ClusTree;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CoresetAdapter;
import streamdatastructures.MicroclusterAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;

public class StreamHiCSTest {

	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private Contrast contrastEvaluator;
	private SummarisationAdapter adapter;
	private final int numInstances = 10000;
	private final int m = 50;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private final String method = "ClusTreeMC";
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static double tpVSfpSum = 0;
	private static double amjsSum = 0;
	private static int testCounter = 0;
	private static Stopwatch stopwatch;

	@BeforeClass
	public static void setUpBeforeClass() {
		csvReader = new CSVReader();
		stopwatch = new Stopwatch();
	}

	@AfterClass
	public static void calculateAverageScores() {
		System.out.println("Average TPvsFP-score: " + tpVSfpSum / testCounter);
		System.out.println("Average AMJS-score: " + amjsSum / testCounter);
		System.out.println(stopwatch.toString());
	}

	@Test
	public void subspaceTest1() {
		String testName = "Test1";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		// No correlated subspaces should have been found, so the correctResult
		// is empty.
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest2() {
		String testName = "Test2";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest3() {
		String testName = "Test3";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest4() {
		String testName = "Test4";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest5() {
		String testName = "Test5";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest6() {
		String testName = "Test6";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest7() {
		String testName = "Test7";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest8() {
		String testName = "Test8";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest9() {
		String testName = "Test9";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest10() {
		String testName = "Test10";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest11() {
		String testName = "Test11";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest12() {
		String testName = "Test12";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(1, 3));
		correctResult.addSubspace(new Subspace(0, 2, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest13() {
		String testName = "Test13";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(1, 3));
		correctResult.addSubspace(new Subspace(0, 2, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest14() {
		String testName = "Test14";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest15() {
		String testName = "Test15";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 2));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest16() {
		String testName = "Test16";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 3));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest17() {
		String testName = "Test17";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest18() {
		String testName = "Test18";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		// correctResult.addSubspace(new Subspace(0, 1));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest19() {
		String testName = "Test19";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		// correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest20() {
		String testName = "Test20";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest21() {
		String testName = "Test21";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest22() {
		String testName = "Test22";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest23() {
		String testName = "Test23";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest24() {
		String testName = "Test24";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest25() {
		String testName = "Test25";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(3, 4, 5));
		correctResult.addSubspace(new Subspace(7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest26() {
		String testName = "Test26";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(1, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest27() {
		String testName = "Test27";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest28() {
		String testName = "Test28";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest29() {
		String testName = "Test29";
		double[][] covarianceMatrix = csvReader.read(path + testName + ".csv");
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	private boolean carryOutSubspaceTest(double[][] covarianceMatrix, SubspaceSet correctResult) {
		stream = new GaussianStream(null, covarianceMatrix);
		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 6;
			pruningDifference = 0.1;

			adapter = new SlidingWindowAdapter(covarianceMatrix.length, numInstances);
		} else if (method.equals("adaptiveCentroids")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.23;
			cutoff = 8;
			pruningDifference = 0.15;

			double fadingLambda = 0.005;
			double radius = 0.2;
			double weightThreshold = 0.1;
			double learningRate = 0.1;

			adapter = new CentroidsAdapter(covarianceMatrix.length, fadingLambda, radius, weightThreshold,
					learningRate);
		} else if (method.equals("DenStreamMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.6;
			cutoff = 8;
			pruningDifference = 0.15;

			WithDBSCAN mcs = new WithDBSCAN();
			mcs.speedOption.setValue(1000);
			mcs.epsilonOption.setValue(0.4);
			mcs.betaOption.setValue(0.2);
			mcs.muOption.setValue(10);
			mcs.lambdaOption.setValue(0.05);
			mcs.resetLearning();
			adapter = new MicroclusterAdapter(mcs);

		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.28;
			cutoff = 8;
			pruningDifference = 0.1;

			ClusTree mcs = new ClusTree();
			mcs.horizonOption.setValue(1000);
			mcs.resetLearningImpl();
			adapter = new MicroclusterAdapter(mcs);

		} else if (method.equals("ClustreamMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.28;
			cutoff = 8;
			pruningDifference = 0.1;

			Clustream mcs = new Clustream();
			mcs.kernelRadiFactorOption.setValue(2);
			mcs.maxNumKernelsOption.setValue(500);
			mcs.prepareForUse();
			adapter = new MicroclusterAdapter(mcs);
		} else if (method.equals("Coreset")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.32;
			cutoff = 8;
			pruningDifference = 0.1;
			adapter = new CoresetAdapter(10000, 1000);
		} else {
			adapter = null;
		}

		contrastEvaluator = new Contrast(m, alpha, adapter);

		System.out.println(method);

		SubspaceBuilder subspaceBuilder = new AprioriBuilder(covarianceMatrix.length, threshold, cutoff,
				pruningDifference, contrastEvaluator);

		// SubspaceBuilder subspaceBuilder = new
		// FastBuilder(covarianceMatrix.length, threshold, pruningDifference,
		// contrastEvaluator);

		/*
		 * SubspaceBuilder subspaceBuilder = new
		 * HierarchicalBuilder(covarianceMatrix.length, threshold,
		 * contrastEvaluator, true);
		 */

		ChangeChecker changeChecker = new TimeCountChecker(numInstances);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback, stopwatch);
		changeChecker.setCallback(streamHiCS);

		// System.out.println("StreamHiCSTest. m = " + m + ", alpha = " + alpha
		// + ", threshold = " + threshold
		// + ", cutoff = " + cutoff + ", pruningDifference = " +
		// pruningDifference);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		System.out.println("Number of elements: " + contrastEvaluator.getNumberOfElements());

		// Evaluation
		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		Evaluator.displayResult(result, correctResult);
		double tpVSfp = Evaluator.evaluateTPvsFP(result, correctResult);
		tpVSfpSum += tpVSfp;
		double amjs = Evaluator.evaluateJaccardIndex(result, correctResult);
		amjsSum += amjs;
		testCounter++;
		System.out.println();

		return tpVSfp >= 0.75;
	}
}