package hicstest;

import static org.junit.Assert.*;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
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

	/*
	@Test
	public void subspaceTest1() {
		String testName = "Test1";
		int numberOfDimensions = 10;
		int blockSize = 5;
		int horizon = 1000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}

	@Test
	public void subspaceTest2() {
		String testName = "Test2";
		int numberOfDimensions = 10;
		int blockSize = 10;
		int horizon = 2000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}
	
	@Test
	public void subspaceTest3() {
		String testName = "Test3";
		int numberOfDimensions = 20;
		int blockSize = 5;
		int horizon = 2000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}

	@Test
	public void subspaceTest4() {
		String testName = "Test4";
		int numberOfDimensions = 20;
		int blockSize = 10;
		int horizon = 2000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}
	
	@Test
	public void subspaceTest5() {
		String testName = "Test5";
		int numberOfDimensions = 50;
		int blockSize = 5;
		int horizon = 4000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}

	@Test
	public void subspaceTest6() {
		String testName = "Test6";
		int numberOfDimensions = 50;
		int blockSize = 10;
		int horizon = 4000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}

	@Test
	public void subspaceTest7() {
		String testName = "Test7";
		int numberOfDimensions = 50;
		int blockSize = 20;
		int horizon = 4000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}
	*/
	
	@Test
	public void subspaceTest8() {
		String testName = "Test8";
		int numberOfDimensions = 100;
		int blockSize = 5;
		int horizon = 6000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}

	@Test
	public void subspaceTest9() {
		String testName = "Test9";
		int numberOfDimensions = 100;
		int blockSize = 10;
		int horizon = 6000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}

	@Test
	public void subspaceTest10() {
		String testName = "Test10";
		int numberOfDimensions = 100;
		int blockSize = 20;
		int horizon = 6000;
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(numberOfDimensions, blockSize, horizon, correctResult));
	}
	
	private boolean carryOutSubspaceTest(int numberOfDimensions, int blockSize, int horizon,
			SubspaceSet correctResult) {
		double[][] covarianceMatrix = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, blockSize,
				0.9);
		stream = new GaussianStream(covarianceMatrix);

		alpha = 0.2;
		epsilon = 0;
		threshold = 0.4;
		cutoff = 40;
		pruningDifference = 0.2;

		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(horizon);
		mcs.resetLearningImpl();
		contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);

		SubspaceBuilder subspaceBuilder = new AprioriBuilder(covarianceMatrix.length, threshold, cutoff,
				pruningDifference, contrastEvaluator);

		ChangeChecker changeChecker = new TimeCountChecker(numInstances);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback);
		changeChecker.setCallback(streamHiCS);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		System.out.println("Dimensionality: " + numberOfDimensions + ", block  size: " + blockSize);
		System.out.println("Horizon: " + horizon);
		System.out.println("Number of elements: " + contrastEvaluator.getNumberOfElements());

		// Evaluation
		return Evaluator.evaluateTPvsFP(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult) >= 0.75;
	}
}
