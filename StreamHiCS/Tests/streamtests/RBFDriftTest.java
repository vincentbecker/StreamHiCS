package streamtests;
import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.CentroidContrast;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import contrast.SlidingWindowContrast;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.streams.generators.RandomRBFGeneratorDrift;
import streamDataStructures.WithDBSCAN;
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
	private Callback callback = new Callback(){
		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm()");
			
		}
	};

	@Before
	public void setUp() throws Exception {
		stream = new RandomRBFGeneratorDrift();
		stream.prepareForUse();
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		Contrast contrastEvaluator;
		double alpha = 0.05;
		double epsilon = 0.1;
		double threshold = 0.1;
		int cutoff = 6;
		double pruningDifference = 0.1;

		if (method.equals("slidingWindow")) {
			alpha = 0.05;
			epsilon = 0;
			threshold = 0.1;
			cutoff = 6;
			pruningDifference = 0.1;

			contrastEvaluator = new SlidingWindowContrast(numberOfDimensions, m, alpha,
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

			contrastEvaluator = new CentroidContrast(numberOfDimensions, m, alpha, fadingLambda, radius,
					weightThreshold, learningRate);
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
			contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);

		} else if (method.equals("ClusTreeMC")) {
			alpha = 0.1;
			epsilon = 0.15;
			threshold = 0.2;
			cutoff = 8;
			pruningDifference = 0.15;

			ClusTree mcs = new ClusTree();
			mcs.resetLearningImpl();
			contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);

		} else {
			contrastEvaluator = null;
		}

		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker, callback);
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
