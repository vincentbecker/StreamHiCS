import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import centroids.TimeCountChecker;
import contrast.CentroidContrast;
import contrast.Contrast;
import contrast.SlidingWindowContrast;
import streamDataStructures.Subspace;
import streamDataStructures.SubspaceSet;
import streams.GaussianStream;
import weka.core.Instance;

public class StreamTest {

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
	private double[][][] covarianceMatrices = {
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
	private ArrayList<SubspaceSet> correctResults;
	private final String method = "adaptiveCentroids";

	@Before
	public void setUp() throws Exception {
		stream = new GaussianStream(covarianceMatrices[0]);
		
		if (method.equals("slidingWindow")) {
			alpha = 0.1;
			epsilon = 0;
			threshold = 0.2;
			cutoff = 6;
			pruningDifference = 0.1;

			contrastEvaluator = new SlidingWindowContrast(null, covarianceMatrices[0].length, numInstances, m, alpha,
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

			contrastEvaluator = new CentroidContrast(null, covarianceMatrices[0].length, m, alpha, fadingLambda, radius,
					numInstances, weightThreshold, learningRate, new TimeCountChecker());
		} else {
			contrastEvaluator = null;
		}
		streamHiCS = new StreamHiCS(covarianceMatrices[0].length, epsilon, threshold, cutoff, pruningDifference, contrastEvaluator);
		contrastEvaluator.setCallback(streamHiCS);
		
		correctResults = new ArrayList<SubspaceSet>();
	}

	@Test
	public void test() {
		SubspaceSet s1 = new SubspaceSet();
		correctResults.add(s1);
		SubspaceSet s2 = new SubspaceSet();
		s2.addSubspace(new Subspace(0, 1));
		correctResults.add(s2);
		SubspaceSet s3 = new SubspaceSet();
		s3.addSubspace(new Subspace(0, 1));
		s3.addSubspace(new Subspace(1, 2));
		s3.addSubspace(new Subspace(0, 1, 2));
		s3.addSubspace(new Subspace(2, 3, 4));
		correctResults.add(s3);
		SubspaceSet s4 = new SubspaceSet();
		s4.addSubspace(new Subspace(0, 1));
		s4.addSubspace(new Subspace(1, 2));
		s4.addSubspace(new Subspace(0, 1, 2));
		s4.addSubspace(new Subspace(2, 3, 4));
		correctResults.add(s4);
		SubspaceSet s5 = new SubspaceSet();
		s5.addSubspace(new Subspace(0, 1));
		s5.addSubspace(new Subspace(0, 1, 2));
		s5.addSubspace(new Subspace(0, 1, 2, 3, 4));
		correctResults.add(s5);
		SubspaceSet s6 = new SubspaceSet();
		s6.addSubspace(new Subspace(0, 1));
		s6.addSubspace(new Subspace(0, 1, 2));
		s6.addSubspace(new Subspace(0, 1, 2, 3, 4));
		correctResults.add(s6);

		carryOutTest();
	}

	private void carryOutTest() {
		double sum = 0;
		int numberSamples = 0;
		for (int i = 0; i < covarianceMatrices.length; i++) {
			System.out.println("Iteration: " + i);
			stream.setCovarianceMatrix(covarianceMatrices[i]);
			numberSamples = 0;
			while (stream.hasMoreInstances() && numberSamples < numInstances) {
				Instance inst = stream.nextInstance();
				streamHiCS.add(inst);
				numberSamples++;
			}
			sum += evaluate(i);
		}
		assertTrue(sum / covarianceMatrices.length >= 0.75);
	}

	private double evaluate(int i) {
		SubspaceSet correctResult = correctResults.get(i);
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
