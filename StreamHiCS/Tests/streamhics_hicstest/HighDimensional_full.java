package streamhics_hicstest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedList;
import java.util.List;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustream.Clustream;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class HighDimensional_full {
	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private int horizon;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon = 0;
	private double aprioriThreshold;
	private double hierarchicalThreshold;
	private int cutoff;
	private double pruningDifference;
	private int numberOfDimensions;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static Stopwatch stopwatch;
	private Contrast contrastEvaluator;
	private static final int numberTestRuns = 10;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
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
				if (summarisation == StreamSummarisation.ADAPTINGCENTROIDS || summarisation == StreamSummarisation.RADIUSCENTROIDS) {
					for (int test = 1; test <= 10; test++) {
						stopwatch.reset();
						double sumTPvsFP = 0;
						double sumAMJS = 0;
						double sumAMSS = 0;
						int sumElements = 0;

						// Create the stream
						horizon = 0;
						SubspaceSet correctResult = new SubspaceSet();

						int[] blockBeginnings = null;
						int[] blockSizes = null;
						switch (test) {
						case 1:
							numberOfDimensions = 10;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 3;
							blockSizes = new int[2];
							blockSizes[0] = 3;
							blockSizes[1] = 3;
							horizon = 1000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(3, 4, 5));
							break;
						case 2:
							numberOfDimensions = 10;
							blockBeginnings = new int[1];
							blockBeginnings[0] = 0;
							blockSizes = new int[1];
							blockSizes[0] = 10;
							horizon = 2000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
							break;
						case 3:
							numberOfDimensions = 20;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 5;
							blockSizes = new int[2];
							blockSizes[0] = 5;
							blockSizes[1] = 5;
							horizon = 2000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
							correctResult.addSubspace(new Subspace(5, 6, 7, 8, 9));
							break;
						case 4:
							numberOfDimensions = 20;
							blockBeginnings = new int[1];
							blockBeginnings[0] = 0;
							blockSizes = new int[1];
							blockSizes[0] = 10;
							horizon = 2000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
							break;
						case 5:
							numberOfDimensions = 50;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 5;
							blockSizes = new int[2];
							blockSizes[0] = 5;
							blockSizes[1] = 5;
							horizon = 4000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
							correctResult.addSubspace(new Subspace(5, 6, 7, 8, 9));
							break;
						case 6:
							numberOfDimensions = 50;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 10;
							blockSizes = new int[2];
							blockSizes[0] = 10;
							blockSizes[1] = 10;
							horizon = 4000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
							correctResult.addSubspace(new Subspace(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
							break;
						case 7:
							numberOfDimensions = 50;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 20;
							blockSizes = new int[2];
							blockSizes[0] = 20;
							blockSizes[1] = 20;
							horizon = 4000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(
									new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
							correctResult.addSubspace(new Subspace(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
									33, 34, 35, 36, 37, 38, 39));
							break;
						case 8:
							numberOfDimensions = 100;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 5;
							blockSizes = new int[2];
							blockSizes[0] = 5;
							blockSizes[1] = 5;
							horizon = 6000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
							correctResult.addSubspace(new Subspace(5, 6, 7, 8, 9));
							break;
						case 9:
							numberOfDimensions = 100;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 10;
							blockSizes = new int[2];
							blockSizes[0] = 10;
							blockSizes[1] = 10;
							horizon = 6000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
							correctResult.addSubspace(new Subspace(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
							break;
						case 10:
							numberOfDimensions = 100;
							blockBeginnings = new int[2];
							blockBeginnings[0] = 0;
							blockBeginnings[1] = 20;
							blockSizes = new int[2];
							blockSizes[0] = 20;
							blockSizes[1] = 20;
							horizon = 6000;
							correctResult = new SubspaceSet();
							correctResult.addSubspace(
									new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
							correctResult.addSubspace(new Subspace(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
									33, 34, 35, 36, 37, 38, 39));
							break;
						}
						double[][] covarianceMatrix = CovarianceMatrixGenerator
								.generateCovarianceMatrix(numberOfDimensions, blockBeginnings, blockSizes, 0.9);
						stream = new GaussianStream(null, covarianceMatrix, 1);

						double threshold = 1;
						switch (buildup) {
						case APRIORI:
							threshold = aprioriThreshold;
							break;
						case HIERARCHICAL:
							threshold = hierarchicalThreshold;
							break;
						}

						// Creating the StreamHiCS system
						adapter = createSummarisationAdapter(summarisation);
						contrastEvaluator = new Contrast(m, alpha, adapter);
						CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
						subspaceBuilder = createSubspaceBuilder(buildup, correlationSummary);
						ChangeChecker changeChecker = new TimeCountChecker(numInstances);
						streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
								subspaceBuilder, changeChecker, callback, correlationSummary, stopwatch);
						changeChecker.setCallback(streamHiCS);

						for (int i = 0; i < numberTestRuns; i++) {
							double[] performanceMeasures = testRun(correctResult);
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

						String measures = numberOfDimensions + "," + avgTPvsFP + ", " + avgAMJS + ", " + avgAMSS + ", "
								+ avgNumElements + ", " + avgEvalTime + ", " + avgAddingTime + ", " + avgTotalTime;
						System.out.println(measures);
						results.add(measures);
					}
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/GaussianStreams/HighDimensional/results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case SLIDINGWINDOW:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.35;
			adapter = new SlidingWindowAdapter(numberOfDimensions, horizon);
			summarisationDescription = "Sliding window, window size: " + horizon;
			break;
		case CLUSTREAM:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.4;
			Clustream cluStream = new Clustream();
			cluStream.kernelRadiFactorOption.setValue(2);
			int numberKernels = 400;
			cluStream.maxNumKernelsOption.setValue(numberKernels);
			cluStream.prepareForUse();
			adapter = new MicroclusteringAdapter(cluStream);
			summarisationDescription = "CluStream, maximum number kernels: " + numberKernels;
			break;
		case DENSTREAM:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.35;
			WithDBSCAN denStream = new WithDBSCAN();
			int speed = 100;
			// double epsilon = 1 * Math.sqrt(numberOfDimensions) - 1; double
			epsilon = 1;
			double beta = 0.2;
			double mu = 10;
			denStream.speedOption.setValue(speed);
			denStream.epsilonOption.setValue(epsilon);
			denStream.betaOption.setValue(beta);
			denStream.muOption.setValue(mu);
			// lambda calculated from horizon double lambda = -Math.log(0.01) /
			// Math.log(2) / (double) horizon;
			double lambda = 0.001;
			denStream.lambdaOption.setValue(lambda);
			denStream.prepareForUse();
			adapter = new MicroclusteringAdapter(denStream);
			summarisationDescription = "DenStream, speed: " + speed + ", epsilon: " + epsilon + ", beta" + beta + ", mu"
					+ mu + ", lambda" + lambda;
			break;
		case CLUSTREE_DEPTHFIRST:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.4;
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree depthFirst, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.3;
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.breadthFirstSearchOption.set();
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree breadthFirst, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.4;
			// double radius = 5;
			double radius = 5 * Math.sqrt(numberOfDimensions) - 1;
			double learningRate = 0.1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.4;
			radius = 5 * Math.sqrt(numberOfDimensions) - 1;
			// radius = 5;
			adapter = new CentroidsAdapter(horizon, radius, 0.1, "readius");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
			addDescription = false;
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, CorrelationSummary correlationSummary) {
		pruningDifference = 0.15;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			cutoff = 20;
			builder = new AprioriBuilder(numberOfDimensions, aprioriThreshold, cutoff, contrastEvaluator,
					correlationSummary);
			builderDescription = "Apriori, threshold:" + aprioriThreshold + "cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			cutoff = 2;
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, hierarchicalThreshold, cutoff,
					contrastEvaluator, correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + hierarchicalThreshold + ", cutoff: " + cutoff;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
			addDescription = false;
		}
		return builder;
	}

	private double[] testRun(SubspaceSet correctResult) {
		streamHiCS.clear();

		int numberSamples = 0;

		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			stopwatch.start("Total");
			streamHiCS.add(inst);
			stopwatch.stop("Total");
			numberSamples++;
		}

		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
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
