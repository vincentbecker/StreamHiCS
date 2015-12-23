package streamtests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.Evaluator;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.MicroclusterAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class StreamTest {

	private static GaussianStream stream;
	private static StreamHiCS streamHiCS;
	private static SummarisationAdapter adapter;
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
	private static Callback callback = new Callback() {

		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm()");
		}

	};
	private static Stopwatch stopwatch;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		stream = new GaussianStream(null, covarianceMatrices[0], 1);

		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 6;
			pruningDifference = 0.1;

			adapter = new SlidingWindowAdapter(covarianceMatrices[0].length, 2000);
		} else if (method.equals("adaptiveCentroids")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.23;
			cutoff = 8;
			pruningDifference = 0.15;

			int horizon = 1000;
			double radius = 0.2;
			double learningRate = 0.1;

			adapter = new CentroidsAdapter(horizon, radius, learningRate);
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
			adapter = new MicroclusterAdapter(mcs);

		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.1;
			epsilon = 0.05;
			threshold = 0.25;
			cutoff = 8;
			pruningDifference = 0.1;

			ClusTree mcs = new ClusTree();
			mcs.resetLearningImpl();
			adapter = new MicroclusterAdapter(mcs);

		} else {
			adapter = null;
		}

		contrastEvaluator = new Contrast(m, alpha, adapter);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(covarianceMatrices[0].length, threshold, cutoff,
				pruningDifference, contrastEvaluator);
		ChangeChecker changeChecker = new TimeCountChecker(numInstances);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback, stopwatch);
		changeChecker.setCallback(streamHiCS);

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

		assertTrue(Evaluator.evaluateTPvsFP(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult) >= 0.75);
	}
}
