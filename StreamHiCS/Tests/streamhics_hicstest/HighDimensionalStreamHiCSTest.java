package streamhics_hicstest;

import static org.junit.Assert.*;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustream.Clustream;
import moa.clusterers.clustree.ClusTree;
import moa.clusterers.denstream.WithDBSCAN;
import streamdatastructures.MCAdapter;
import streamdatastructures.CoresetAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class HighDimensionalStreamHiCSTest {

	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private Contrast contrastEvaluator;
	private final int numInstances = 10000;
	private final int m = 50;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static Stopwatch stopwatch;
	private String method;
	private SummarisationAdapter adapter;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}

	@AfterClass
	public static void calculateAverageScores() {
		// System.out.println("Average TPvsFP-score: " + tpVSfpSum /
		// testCounter);
		// System.out.println("Average AMJS-score: " + amjsSum / testCounter);
		//System.out.println(stopwatch.toString());
	}

	@Test
	public void subspaceTest1() {
		String testName = "Test1";
		int numberOfDimensions = 10;
		int[] blockBeginnings = {0, 3};
		int[] blockSizes = {3, 3};
		int horizon = 1000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(3, 4, 5));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest2() {
		String testName = "Test2";
		int numberOfDimensions = 10;
		int[] blockBeginnings = {0};
		int[] blockSizes = {10};
		int horizon = 2000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest3() {
		String testName = "Test3";
		int numberOfDimensions = 20;
		int[] blockBeginnings = {0, 5};
		int[] blockSizes = {5, 5};
		int horizon = 2000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		correctResult.addSubspace(new Subspace(5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest4() {
		String testName = "Test4";
		int numberOfDimensions = 20;
		int[] blockBeginnings = {0};
		int[] blockSizes = {10};
		int horizon = 2000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest5() {
		String testName = "Test5";
		int numberOfDimensions = 50;
		int[] blockBeginnings = {0, 5};
		int[] blockSizes = {5, 5};
		int horizon = 4000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		correctResult.addSubspace(new Subspace(5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest6() {
		String testName = "Test6";
		int numberOfDimensions = 50;
		int[] blockBeginnings = {0, 10};
		int[] blockSizes = {10, 10};
		int horizon = 4000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		correctResult.addSubspace(new Subspace(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest7() {
		String testName = "Test7";
		int numberOfDimensions = 50;
		int[] blockBeginnings = {0, 20};
		int[] blockSizes = {20, 20};
		int horizon = 4000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		correctResult.addSubspace(new Subspace(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest8() {
		String testName = "Test8";
		int numberOfDimensions = 100;
		int[] blockBeginnings = {0, 5};
		int[] blockSizes = {5, 5};
		int horizon = 6000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		correctResult.addSubspace(new Subspace(5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest9() {
		String testName = "Test9";
		int numberOfDimensions = 100;
		int[] blockBeginnings = {0, 10};
		int[] blockSizes = {10, 10};
		int horizon = 6000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		correctResult.addSubspace(new Subspace(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	@Test
	public void subspaceTest10() {
		String testName = "Test10";
		int numberOfDimensions = 100;
		int[] blockBeginnings = {0, 20};
		int[] blockSizes = {20, 20};
		int horizon = 6000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		correctResult.addSubspace(new Subspace(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39));		
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockBeginnings, blockSizes, horizon, correctResult));
	}

	private boolean carryOutSubspaceTest(int numberOfDimensions, int[] blockBeginnings, int[] blockSizes, int horizon,
			SubspaceSet correctResult) {
		stopwatch.reset();
		double[][] covarianceMatrix = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, blockBeginnings,
				blockSizes, 0.9);
		stream = new GaussianStream(null, covarianceMatrix, 1);

		method = "ClusTreeMC";

		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 40;
			pruningDifference = 0.2;

			adapter = new SlidingWindowAdapter(covarianceMatrix.length, numInstances);
		} else if (method.equals("adaptiveCentroids")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.23;
			cutoff = 15;
			pruningDifference = 0.25;

			double radius = 2.5;
			// double radius = 0.7;
			double learningRate = 0.2;

			horizon = 1000;
			adapter = new MCAdapter(horizon, radius, learningRate, "adapting");
		} else if (method.equals("DenStreamMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.6;
			cutoff = 40;
			pruningDifference = 0.15;

			WithDBSCAN mcs = new WithDBSCAN();
			mcs.speedOption.setValue(1000);
			mcs.epsilonOption.setValue(0.8);
			mcs.betaOption.setValue(0.2);
			mcs.muOption.setValue(10);
			mcs.lambdaOption.setValue(0.005);
			mcs.resetLearning();
			adapter = new MicroclusteringAdapter(mcs);

		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.05;
			epsilon = 0;
			//threshold = 0.5;
			threshold = 0.25;
			cutoff = 20;
			pruningDifference = 0.2;

			ClusTree mcs = new ClusTree();
			mcs.horizonOption.setValue(horizon);
			mcs.resetLearning();
			adapter = new MicroclusteringAdapter(mcs);

			System.out.println("Horizon: " + horizon);

		} else if (method.equals("ClustreamMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.26;
			cutoff = 40;
			pruningDifference = 0.15;

			Clustream mcs = new Clustream();
			mcs.kernelRadiFactorOption.setValue(2);
			mcs.maxNumKernelsOption.setValue(500);
			mcs.prepareForUse();
			adapter = new MicroclusteringAdapter(mcs);
		} else if (method.equals("Coreset")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.35;
			cutoff = 40;
			pruningDifference = 0.15;
			adapter = new CoresetAdapter(10000, 300);
		} else {
			adapter = null;
		}

		/*
		 * alpha = 0.2; epsilon = 0; threshold = 0.4; cutoff = 40;
		 * pruningDifference = 0.2; ClusTree mcs = new ClusTree();
		 * mcs.horizonOption.setValue(horizon); mcs.resetLearningImpl();
		 * SummarisationAdapter adapter = new MicroclusterAdapter(mcs);
		 */
		contrastEvaluator = new Contrast(m, alpha, adapter);

		CorrelationSummary correlationSummary = new CorrelationSummary(covarianceMatrix.length, horizon);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(covarianceMatrix.length, threshold, cutoff,
				contrastEvaluator, correlationSummary);

		//SubspaceBuilder subspaceBuilder = new HierarchicalBuilderCutoff(covarianceMatrix.length, threshold, cutoff,
		//contrastEvaluator, correlationSummary, true);

		ChangeChecker changeChecker = new TimeCountChecker(numInstances);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			stopwatch.start("Total");
			stopwatch.stop("Total");
			streamHiCS.add(inst);
			numberSamples++;
		}

		System.out.println("Dimensionality: " + numberOfDimensions + ", block  size: " + blockSizes[0]);

		//System.out.println("Faded: " + ((CentroidsAdapter) adapter).getFadedCount());

		// Evaluation
		Evaluator.displayResult(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult);
		double tpVSfp = Evaluator.evaluateTPvsFP(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult);
		double amjs = Evaluator.evaluateJaccardIndex(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult);
		double amss = Evaluator.evaluateStructuralSimilarity(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult);
		int numElements = streamHiCS.getNumberOfElements();
		String measures = tpVSfp + ", " + amjs + ", " + amss + ", "
				+ numElements + ", " + stopwatch.getTime("Evaluation") + ", " + stopwatch.getTime("Adding") + ", " + stopwatch.getTime("Total");
		System.out.println(measures);
		
		return tpVSfp >= 0.75;
	}
}
