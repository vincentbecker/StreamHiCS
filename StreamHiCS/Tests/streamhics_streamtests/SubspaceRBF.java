package streamhics_streamtests;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.Evaluator;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustream.Clustream;
import streamdatastructures.WithDBSCAN;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.SubspaceRandomRBFGeneratorDrift;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class SubspaceRBF {

	private SubspaceRandomRBFGeneratorDrift stream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private final int horizon = 2000;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon = 0;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private int numberOfDimensions;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static Random random;
	private static Stopwatch stopwatch;
	private Contrast contrastEvaluator;
	private static final int numberTestRuns = 10;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
		random = new Random();
	}

	@Test
	public void test() {
		// Output
		results = new LinkedList<String>();
		SummarisationAdapter adapter;
		SubspaceBuilder subspaceBuilder;
		for (StreamSummarisation summarisation : StreamSummarisation.values()) {
			summarisationDescription = null;
			for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
				builderDescription = null;
				if (summarisation == StreamSummarisation.RADIUSCENTROIDS) {
					for (numberOfDimensions = 3; numberOfDimensions <= 50; numberOfDimensions++) {
						stopwatch.reset();
						double sumTPvsFP = 0;
						double sumAMJS = 0;
						double sumAMSS = 0;
						int sumElements = 0;

						// Creating the stream
						stream = new SubspaceRandomRBFGeneratorDrift();
						stream.numAttsOption.setValue(numberOfDimensions);
						stream.avgSubspaceSizeOption.setValue(numberOfDimensions / 2);
						stream.numCentroidsOption.setValue(10);
						stream.numSubspaceCentroidsOption.setValue(5);
						stream.sameSubspaceOption.set();
						stream.randomSubspaceSizeOption.setValue(false);
						stream.prepareForUse();

						// Creating the StreamHiCS system
						adapter = createSummarisationAdapter(summarisation);
						contrastEvaluator = new Contrast(m, alpha, adapter);
						subspaceBuilder = createSubspaceBuilder(buildup);
						ChangeChecker changeChecker = new TimeCountChecker(numInstances);
						streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
								subspaceBuilder, changeChecker, callback, null, stopwatch);
						changeChecker.setCallback(streamHiCS);

						for (int i = 0; i < numberTestRuns; i++) {
							double[] performanceMeasures = testRun();
							sumTPvsFP += performanceMeasures[0];
							sumAMJS += performanceMeasures[1];
							sumAMSS += performanceMeasures[2];
							sumElements += performanceMeasures[3];
						}

						// Calculate results
						double sumEvaluationTime = stopwatch.getTime("Evaluation");
						double sumAddingTime = stopwatch.getTime("Adding");
						double sumTotalTime = stopwatch.getTime("Total");

						double avgTPvsFP = sumTPvsFP / numberTestRuns;
						double avgAMJS = sumAMJS / numberTestRuns;
						double avgAMSS = sumAMSS / numberTestRuns;
						double avgNumElements = sumElements / numberTestRuns;
						double avgEvalTime = sumEvaluationTime / numberTestRuns;
						double avgAddingTime = sumAddingTime / numberTestRuns;
						double avgTotalTime = sumTotalTime / numberTestRuns;

						System.out.println("Average TPvsFP: " + avgTPvsFP);
						System.out.println("Average AMJS: " + avgAMJS);
						System.out.println("Average AMSS: " + avgAMSS);
						System.out.println("Average number of elements: " + avgNumElements);
						System.out.println("Average evaluation time: " + avgEvalTime);
						System.out.println("Average adding time: " + avgAddingTime);
						System.out.println("Average total time: " + avgTotalTime);

						System.out.println(numberOfDimensions + "," + avgTPvsFP + ", " + avgAMJS + ", " + avgAMSS + ", "
								+ avgNumElements + ", " + avgEvalTime + ", " + avgAddingTime + ", " + avgTotalTime);
					}
				}
			}
		}

		// Write the results
		try {
			Files.write(
					Paths.get(
							"D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/SubspaceRandomRBF/same_noDrift.txt"),
					results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		fail("No condition");
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case SLIDINGWINDOW:
			threshold = 0.23;
			adapter = new SlidingWindowAdapter(numberOfDimensions, horizon);
			summarisationDescription = "Sliding window, window size: " + horizon;
			break;
		case CLUSTREAM:
			threshold = 0.23;
			Clustream cluStream = new Clustream();
			cluStream.kernelRadiFactorOption.setValue(2);
			int numberKernels = 750;
			cluStream.maxNumKernelsOption.setValue(numberKernels);
			cluStream.prepareForUse();
			adapter = new MicroclusteringAdapter(cluStream);
			summarisationDescription = "CluStream, maximum number kernels: " + numberKernels;
			break;
		case DENSTREAM:
			threshold = 0.23;
			WithDBSCAN denStream = new WithDBSCAN();
			int speed = 1;
			double epsilon = 2.5;
			double beta = 0.2;
			double mu = 10;
			denStream.speedOption.setValue(speed);
			denStream.epsilonOption.setValue(epsilon);
			denStream.betaOption.setValue(beta);
			denStream.muOption.setValue(mu);
			// lambda calculated from horizon
			double lambda = -Math.log(0.01) / Math.log(2) / (double) horizon;
			denStream.lambdaOption.setValue(0.001);
			denStream.prepareForUse();
			adapter = new MicroclusteringAdapter(denStream);
			summarisationDescription = "DenStream, speed: " + speed + ", epsilon: " + epsilon + ", beta" + beta + ", mu"
					+ mu + ", lambda" + lambda;
			break;
		case CLUSTREE_DEPTHFIRST:
			threshold = 0.23;
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			threshold = 0.23;
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			threshold = 0.2;
			// double radius = 7 * Math.log(numberOfDimensions) - 0.5;
			// double radius = 8.38 * Math.log(numberOfDimensions) - 3.09;
			double radius = 5 * Math.sqrt(numberOfDimensions) - 1;
			// double radius = 27;
			double learningRate = 0.1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			threshold = 0.15;
			radius = 5 * Math.sqrt(numberOfDimensions) - 1;
			adapter = new CentroidsAdapter(horizon, radius, 0.1, "readius");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(builderDescription);
		}
		results.add(summarisationDescription);
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb) {
		cutoff = 15;
		pruningDifference = 0.15;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, contrastEvaluator, null);
			builderDescription = "Apriori, threshold:" + threshold + "cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, threshold, cutoff, contrastEvaluator, null,
					true);
			builderDescription = "Hierarchical, threshold: " + threshold + ", cutoff: " + cutoff;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
		}
		return builder;
	}

	private double[] testRun() {
		stream.instanceRandomSeedOption.setValue(random.nextInt(1000000000));
		stream.prepareForUse();
		streamHiCS.clear();

		int numberSamples = 0;

		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			stopwatch.start("Total");
			streamHiCS.add(inst);
			stopwatch.stop("Total");
			numberSamples++;
		}

		// System.out.println("Number of elements: " +
		// streamHiCS.getNumberOfElements());

		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		SubspaceSet correctResult = stream.getSubspaces();
		for (Subspace s : correctResult.getSubspaces()) {
			s.sort();
		}
		correctResult.sort();
		double[] performanceMeasures = new double[4];
		// Evaluator.displayResult(result, correctResult);
		performanceMeasures[0] = Evaluator.evaluateTPvsFP(result, correctResult);
		performanceMeasures[1] = Evaluator.evaluateJaccardIndex(result, correctResult);
		performanceMeasures[2] = Evaluator.evaluateStructuralSimilarity(result, correctResult);
		performanceMeasures[3] = streamHiCS.getNumberOfElements();

		return performanceMeasures;
	}

}
