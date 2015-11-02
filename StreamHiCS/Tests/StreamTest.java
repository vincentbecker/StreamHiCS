import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.TimeCountChecker;
import contrast.Callback;
import contrast.CentroidContrast;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import contrast.SlidingWindowContrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import streamDataStructures.WithDBSCAN;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class StreamTest {

	private static GaussianStream stream;
	private static StreamHiCS streamHiCS;
	private static Contrast contrastEvaluator;
	private static final int numInstances = 10000;
	private static final int m = 20;
	private static double alpha;
	private static double epsilon;
	private static double threshold;
	private static int cutoff;
	private static double pruningDifference;
	private static double[][][] covarianceMatrices = {
			{ { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } },
			{ { 1, 0.2, 0.1, 0, 0 }, { 0.2, 1, 0.1, 0, 0 }, { 0.1, 0.1, 1, 0.1, 0.1 }, { 0, 0, 0.1, 1, 0.1 },
					{ 0, 0, 0, 0.1, 1 } },
			{ { 1, 0.4, 0.2, 0, 0 }, { 0.4, 1, 0.2, 0, 0 }, { 0.2, 0.2, 1, 0.2, 0.2 }, { 0, 0, 0.2, 1, 0.2 },
					{ 0, 0, 0, 0.2, 1 } },
			{ { 1, 0.6, 0.4, 0, 0 }, { 0.6, 1, 0.4, 0, 0 }, { 0.4, 0.4, 1, 0.4, 0.4 }, { 0, 0, 0.4, 1, 0.4 },
					{ 0, 0, 0, 0.4, 1 } },
			{ { 1, 0.8, 0.6, 0, 0 }, { 0.8, 1, 0.6, 0, 0 }, { 0.6, 0.6, 1, 0.6, 0.6 }, { 0, 0, 0.6, 1, 0.6 },
					{ 0, 0, 0, 0.6, 1 } },
			{ { 1, 0.9, 0.8, 0.2, 0.2 }, { 0.9, 1, 0.8, 0.2, 0.2 }, { 0.8, 0.8, 1, 0.8, 0.8 },
					{ 0.2, 0.2, 0.8, 1, 0.8 }, { 0.2, 0.2, 0.2, 0.8, 1 } } };
	private static SubspaceSet correctResult;
	private static final String method = "ClusTreeMC";
	private static Callback callback = new Callback(){

		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm()");
			
		}
		
	};

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		stream = new GaussianStream(covarianceMatrices[0]);

		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 6;
			pruningDifference = 0.1;

			contrastEvaluator = new SlidingWindowContrast(null, covarianceMatrices[0].length, 1000, m, alpha, 2000,
					new TimeCountChecker(numInstances));
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

			contrastEvaluator = new CentroidContrast(null, covarianceMatrices[0].length, m, alpha, fadingLambda, radius,
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
			epsilon = 0.05;
			threshold = 0.25;
			cutoff = 8;
			pruningDifference = 0.1;

			ClusTree mcs = new ClusTree();
			mcs.resetLearningImpl();
			contrastEvaluator = new MicroclusterContrast(null, m, alpha, mcs, new TimeCountChecker(numInstances));

		} else {
			contrastEvaluator = null;
		}

		SubspaceBuilder subspaceBuilder = new AprioriBuilder(covarianceMatrices[0].length, threshold, cutoff,
				pruningDifference, contrastEvaluator);
		streamHiCS = new StreamHiCS(epsilon, threshold, contrastEvaluator, subspaceBuilder, callback);
		contrastEvaluator.setCallback(streamHiCS);

		correctResult = new SubspaceSet();
	}

	@Before
	public void setUp() {
		correctResult.clear();
	}

	@Test
	public void test1() {
		carryOutTest(0);
	}

	@Test
	public void test2() {
		correctResult.addSubspace(new Subspace(0, 1));
		carryOutTest(1);
	}

	@Test
	public void test3() {
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		carryOutTest(2);
	}

	@Test
	public void test4() {
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		carryOutTest(3);
	}

	@Test
	public void test5() {
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		carryOutTest(4);
	}

	@Test
	public void test6() {
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		carryOutTest(5);
	}

	private void carryOutTest(int iteration) {
		int numberSamples = 0;
		System.out.println("Iteration: " + (iteration + 1));
		stream.setCovarianceMatrix(covarianceMatrices[iteration]);
		numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		System.out.println("Number of elements: " + contrastEvaluator.getNumberOfElements());

		assertTrue(evaluate() >= 0.75);
	}

	private double evaluate() {
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
		}
		System.out.println("True positives: " + tp + " out of " + correctResult.size() + "; False positives: " + fp);
		System.out.println();
		return recall - fpRatio;
	}

}
