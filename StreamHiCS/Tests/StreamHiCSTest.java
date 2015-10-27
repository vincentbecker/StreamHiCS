import static org.junit.Assert.*;
import org.junit.Test;

import changechecker.TimeCountChecker;
import contrast.CentroidContrast;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import contrast.SlidingWindowContrast;
import streamDataStructures.WithDBSCAN;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import weka.core.Instance;
import moa.clusterers.clustree.ClusTree;

public class StreamHiCSTest {

	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private Contrast contrastEvaluator;
	private final int numInstances = 10000;
	private final int m = 20;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private final String method = "ClusTreeMC";

	@Test
	public void subspaceTest1() {
		double[][] covarianceMatrix = { { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 },
				{ 0, 0, 0, 0, 1 } };

		// No correlated subspaces should have been found, so the correctResult
		// is empty.
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println("Test 1");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest2() {
		double[][] covarianceMatrix = { { 1, 0.9, 0, 0, 0 }, { 0.9, 1, 0, 0, 0 }, { 0, 0, 1, 0.9, 0.9 },
				{ 0, 0, 0.9, 1, 0.9 }, { 0, 0, 0.9, 0.9, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println("Test 2");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest3() {
		double[][] covarianceMatrix = { { 1, 0.9, 0.9, 0.9, 0.9 }, { 0.9, 1, 0.9, 0.9, 0.9 }, { 0.9, 0.9, 1, 0.9, 0.9 },
				{ 0.9, 0.9, 0.9, 1, 0.9 }, { 0.9, 0.9, 0.9, 0.9, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println("Test 3");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest4() {
		double[][] covarianceMatrix = { { 1, 0.8, 0.8, 0.4, 0.4 }, { 0.8, 1, 0.8, 0.4, 0.4 }, { 0.8, 0.8, 1, 0.8, 0.8 },
				{ 0.4, 0.4, 0.8, 1, 0.8 }, { 0.4, 0.4, 0.8, 0.8, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		System.out.println("Test 4");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest5() {
		double[][] covarianceMatrix = { { 1, 0.5, 0.5, 0, 0 }, { 0.5, 1, 0.5, 0, 0 }, { 0.5, 0.5, 1, 0.1, 0.1 },
				{ 0, 0, 0.1, 1, 0.1 }, { 0, 0, 0.1, 0.1, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		System.out.println("Test 5");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest6() {
		double[][] covarianceMatrix = { { 1, 0.6, 0.2, 0.2, 0.2 }, { 0.6, 1, 0.6, 0.2, 0.2 }, { 0.2, 0.6, 1, 0.7, 0.2 },
				{ 0.2, 0.2, 0.6, 1, 0.6 }, { 0.2, 0.2, 0.2, 0.6, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(3, 4));
		System.out.println("Test 6");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest7() {
		double[][] covarianceMatrix = { { 1, 0.6, 0.1 }, { 0.6, 1, 0.6 }, { 0.1, 0.6, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		System.out.println("Test 7");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest8() {
		double[][] covarianceMatrix = { { 1, 0.2, 0.2 }, { 0.2, 1, 0.2 }, { 0.2, 0.2, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println("Test 8");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest9() {
		double[][] covarianceMatrix = { { 1, 0.4, 0.1 }, { 0.4, 1, 0.4 }, { 0.1, 0.4, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		System.out.println("Test 9");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest10() {
		double[][] covarianceMatrix = { { 1, 0, 0.1 }, { 0, 1, 0 }, { 0.1, 0, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println("Test 10");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest11() {
		double[][] covarianceMatrix = { { 1, 0.7, 0.7 }, { 0.7, 1, 0.7 }, { 0.7, 0.7, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2));
		System.out.println("Test 11");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest12() {
		double[][] covarianceMatrix = { { 1, 0, 0.9, 0, 0.9 }, { 0, 1, 0, 0.9, 0 }, { 0.9, 0, 1, 0, 0.9 },
				{ 0, 0.9, 0, 1, 0 }, { 0.9, 0, 0.9, 0, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(1, 3));
		correctResult.addSubspace(new Subspace(0, 2, 4));
		System.out.println("Test 12");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest13() {
		double[][] covarianceMatrix = { { 1, 0, 0.6, 0, 0.6 }, { 0, 1, 0, 0.6, 0 }, { 0.6, 0, 1, 0, 0.6 },
				{ 0, 0.6, 0, 1, 0 }, { 0.6, 0, 0.6, 0, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(1, 3));
		correctResult.addSubspace(new Subspace(0, 2, 4));
		System.out.println("Test 13");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest14() {
		double[][] covarianceMatrix = { { 1, 0.1, 0.1, 0.1, 0.1 }, { 0.1, 1, 0.1, 0.1, 0.1 }, { 0.1, 0.1, 1, 0.1, 0.1 },
				{ 0.1, 0.1, 0.1, 1, 0.1 }, { 0.1, 0.1, 0.1, 0.1, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		System.out.println("Test 14");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest15() {
		double[][] covarianceMatrix = { { 1, 0, 0.7, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0.7, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 },
				{ 0, 0, 0, 0, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 2));
		System.out.println("Test 15");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest16() {
		double[][] covarianceMatrix = { { 1, 0.6, 0, 0.6, 0 }, { 0.6, 1, 0, 0.6, 0 }, { 0, 0, 1, 0, 0 },
				{ 0.6, 0.6, 0, 1, 0 }, { 0, 0, 0, 0, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 3));
		System.out.println("Test 16");
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	private boolean carryOutSubspaceTest(double[][] covarianceMatrix, SubspaceSet correctResult) {
		stream = new GaussianStream(covarianceMatrix);
		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 6;
			pruningDifference = 0.1;

			contrastEvaluator = new SlidingWindowContrast(null, covarianceMatrix.length, numInstances, m, alpha,
					numInstances);
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

			contrastEvaluator = new CentroidContrast(null, covarianceMatrix.length, m, alpha, fadingLambda, radius,
					weightThreshold, learningRate, new TimeCountChecker(numInstances));
		} else if (method.equals("DenStreamMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.3;
			cutoff = 8;
			pruningDifference = 0.15;

			WithDBSCAN mcs = new WithDBSCAN();
			mcs.speedOption.setValue(100);
			mcs.epsilonOption.setValue(0.5);
			mcs.betaOption.setValue(0.005);
			mcs.lambdaOption.setValue(0.005);
			mcs.resetLearningImpl();
			contrastEvaluator = new MicroclusterContrast(null, m, alpha, mcs, new TimeCountChecker(numInstances));

		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.25;
			cutoff = 8;
			pruningDifference = 0.15;

			ClusTree mcs = new ClusTree();
			contrastEvaluator = new MicroclusterContrast(null, m, alpha, mcs, new TimeCountChecker(numInstances));

		} else {
			contrastEvaluator = null;
		}

		streamHiCS = new StreamHiCS(covarianceMatrix.length, epsilon, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		contrastEvaluator.setCallback(streamHiCS);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		System.out.println("Number of elements: " + contrastEvaluator.getNumberOfElements());

		// Evaluation
		int l = correctResult.size();
		int tp = 0;
		for (Subspace s : correctResult.getSubspaces()) {
			if (streamHiCS.getCurrentlyCorrelatedSubspaces().contains(s)) {
				tp++;
			}
		}
		int fp = streamHiCS.getCurrentlyCorrelatedSubspaces().size() - tp;
		double recall = 1;
		double fpRatio = 0;
		if (l > 0) {
			recall = ((double) tp) / correctResult.size();
			fpRatio = ((double) fp) / correctResult.size();
		} else {
			if (fp > 0) {
				fpRatio = 1;
			}
		}
		System.out.println("True positives: " + tp + " out of " + correctResult.size() + "; False positives: " + fp);
		System.out.println();

		// return
		// streamHiCS.getCurrentlyCorrelatedSubspaces().equals(correctResult);
		return ((recall - fpRatio) >= 0.75);
	}

}
