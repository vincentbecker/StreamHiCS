package streamhics_streamtests;

import static org.junit.Assert.*;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.clusterers.denstream.WithDBSCAN;
import moa.streams.generators.RandomRBFGeneratorDrift;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class RBFDriftTest {

	private RandomRBFGeneratorDrift stream;
	private StreamHiCS streamHiCS;
	private int numberSamples = 0;
	private final int numInstances = 10000;
	private final String method = "ClusTreeMC";
	private final int numberOfDimensions = 10;
	private final int m = 20;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm()");

		}
	};
	private static Stopwatch stopwatch;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}
	
	@AfterClass
	public static void calculateAverageScores() {
		// System.out.println("Average TPvsFP-score: " + tpVSfpSum /
		// testCounter);
		// System.out.println("Average AMJS-score: " + amjsSum / testCounter);
		System.out.println(stopwatch.toString());
	}
	
	@Before
	public void setUp() throws Exception {
		stream = new RandomRBFGeneratorDrift();
		stream.prepareForUse();
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		SummarisationAdapter adapter;
		Contrast contrastEvaluator;
		double alpha = 0.05;
		double epsilon = 0.1;
		double threshold = 0.1;
		int cutoff = 6;
		double pruningDifference = 0.1;

		int horizon = 0;
		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 6;
			pruningDifference = 0.1;

			adapter = new SlidingWindowAdapter(numberOfDimensions, numInstances);
		} else if (method.equals("adaptiveCentroids")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.23;
			cutoff = 8;
			pruningDifference = 0.15;

			horizon = 1000;
			double radius = 0.2;
			double learningRate = 0.1;

			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
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
			adapter = new MicroclusteringAdapter(mcs);

		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.1;
			epsilon = 0.15;
			threshold = 0.2;
			cutoff = 8;
			pruningDifference = 0.15;
			horizon = 1000;
			
			ClusTree mcs = new ClusTree();
			mcs.horizonOption.setValue(horizon);
			mcs.prepareForUse();
			adapter = new MicroclusteringAdapter(mcs);
		} else {
			adapter = null;
		}

		contrastEvaluator = new Contrast(m, alpha, adapter);
		CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff,
				contrastEvaluator, correlationSummary);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);
	}

	@Test
	public void test() {
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		fail("Not yet implemented");
	}

}
