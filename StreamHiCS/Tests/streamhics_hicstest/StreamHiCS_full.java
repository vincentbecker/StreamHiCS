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
import environment.CSVReader;
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

public class StreamHiCS_full {
	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private final int horizon = 1000;
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
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	private double[] resultSummary;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
		csvReader = new CSVReader();
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
				if (summarisation == StreamSummarisation.RADIUSCENTROIDS && buildup == SubspaceBuildup.HIERARCHICAL) {
					resultSummary = new double[7];
					for (int test = 1; test <= 29; test++) {
						stopwatch.reset();
						double sumTPvsFP = 0;
						double sumAMJS = 0;
						double sumAMSS = 0;
						int sumElements = 0;

						double threshold = 1;
						switch (buildup) {
						case APRIORI:
							threshold = aprioriThreshold;
							break;
						case HIERARCHICAL:
							threshold = hierarchicalThreshold;
							break;
						}

						// Create the stream
						double[][] covarianceMatrix = csvReader.read(path + "Test" + test + ".csv");
						numberOfDimensions = covarianceMatrix.length;
						stream = new GaussianStream(null, covarianceMatrix, 1);

						SubspaceSet correctResult = new SubspaceSet();

						switch (test) {
						case 1:
							break;
						case 2:
							correctResult.addSubspace(new Subspace(0, 1));
							correctResult.addSubspace(new Subspace(2, 3, 4));
							break;
						case 3:
							correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
							break;
						case 4:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(2, 3, 4));
							break;
						case 5:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							break;
						case 6:
							correctResult.addSubspace(new Subspace(0, 1));
							correctResult.addSubspace(new Subspace(1, 2));
							correctResult.addSubspace(new Subspace(2, 3));
							correctResult.addSubspace(new Subspace(3, 4));
							break;
						case 7:
							correctResult.addSubspace(new Subspace(0, 1));
							correctResult.addSubspace(new Subspace(1, 2));
							break;
						case 8:
							break;
						case 9:
							correctResult.addSubspace(new Subspace(0, 1));
							correctResult.addSubspace(new Subspace(1, 2));
							break;
						case 10:
							break;
						case 11:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							break;
						case 12:
							correctResult.addSubspace(new Subspace(1, 3));
							correctResult.addSubspace(new Subspace(0, 2, 4));
							break;
						case 13:
							correctResult.addSubspace(new Subspace(1, 3));
							correctResult.addSubspace(new Subspace(0, 2, 4));
							break;
						case 14:
							break;
						case 15:
							correctResult.addSubspace(new Subspace(0, 2));
							break;
						case 16:
							correctResult.addSubspace(new Subspace(0, 1, 3));
							break;
						case 17:
							correctResult.addSubspace(new Subspace(0, 1));
							break;
						case 18:
							break;
						case 19:
							correctResult.addSubspace(new Subspace(0, 1));
							break;
						case 20:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(2, 3, 4));
							break;
						case 21:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(2, 3, 4));
							break;
						case 22:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(2, 3, 4));
							break;
						case 23:
							correctResult.addSubspace(new Subspace(0, 1));
							break;
						case 24:
							break;
						case 25:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(3, 4, 5));
							correctResult.addSubspace(new Subspace(7, 8, 9));
							break;
						case 26:
							correctResult.addSubspace(new Subspace(0, 1));
							correctResult.addSubspace(new Subspace(2, 3));
							correctResult.addSubspace(new Subspace(1, 9));
							break;
						case 27:
							break;
						case 28:
							correctResult.addSubspace(new Subspace(0, 1, 2));
							correctResult.addSubspace(new Subspace(7, 8, 9));
							break;
						case 29:
							correctResult.addSubspace(new Subspace(0, 1, 2));
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

						String measures = test + "," + avgTPvsFP + ", " + avgAMJS + ", " + avgAMSS + ", "
								+ avgNumElements + ", " + avgEvalTime + ", " + avgAddingTime + ", " + avgTotalTime;
						System.out.println(measures);

						resultSummary[0] += avgTPvsFP;
						resultSummary[1] += avgAMJS;
						resultSummary[2] += avgAMSS;
						resultSummary[3] += avgNumElements;
						resultSummary[4] += avgEvalTime;
						resultSummary[5] += avgAddingTime;
						resultSummary[6] += avgTotalTime;

						results.add(measures);
					}
					System.out.println(resultSummary[0] / 29 + ", " + resultSummary[1] / 29 + ", "
							+ resultSummary[2] / 29 + ", " + resultSummary[3] / 29 + ", " + resultSummary[4] / 29 + ", "
							+ resultSummary[5] / 29 + ", " + resultSummary[6] / 29);
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/GaussianStreams/Standard/Results.txt";

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
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			adapter = new SlidingWindowAdapter(numberOfDimensions, horizon);
			summarisationDescription = "Sliding window, window size: " + horizon;
			break;
		case CLUSTREAM:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			Clustream cluStream = new Clustream();
			cluStream.kernelRadiFactorOption.setValue(2);
			int numberKernels = 400;
			cluStream.maxNumKernelsOption.setValue(numberKernels);
			cluStream.prepareForUse();
			adapter = new MicroclusteringAdapter(cluStream);
			summarisationDescription = "CluStream, maximum number kernels: " + numberKernels;
			break;
		case DENSTREAM:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			WithDBSCAN denStream = new WithDBSCAN();
			int speed = 100;
			double epsilon = 0.5;
			double beta = 0.2;
			double mu = 10;
			denStream.speedOption.setValue(speed);
			denStream.epsilonOption.setValue(epsilon);
			denStream.betaOption.setValue(beta);
			denStream.muOption.setValue(mu);
			// lambda calculated from horizon
			double lambda = -Math.log(0.01) / Math.log(2) / (double) horizon;
			denStream.lambdaOption.setValue(lambda);
			denStream.prepareForUse();
			adapter = new MicroclusteringAdapter(denStream);
			summarisationDescription = "DenStream, speed: " + speed + ", epsilon: " + epsilon + ", beta" + beta + ", mu"
					+ mu + ", lambda" + lambda;
			break;
		case CLUSTREE_DEPTHFIRST:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree depthFirst, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.breadthFirstSearchOption.set();
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree breadthFirst, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.35;
			double radius = 3.5;
			double learningRate = 0.1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.35;
			radius = 0.75;
			adapter = new CentroidsAdapter(horizon, radius, 0.1, "radius");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, CorrelationSummary correlationSummary) {
		cutoff = 8;
		pruningDifference = 0.15;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, aprioriThreshold, cutoff, contrastEvaluator,
					correlationSummary);
			builderDescription = "Apriori, threshold: " + aprioriThreshold + ", cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, hierarchicalThreshold, cutoff,
					contrastEvaluator, correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + hierarchicalThreshold + ", cutoff: " + cutoff;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
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
