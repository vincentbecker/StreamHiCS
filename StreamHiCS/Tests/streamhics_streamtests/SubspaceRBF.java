package streamhics_streamtests;

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
import streamdatastructures.MCAdapter;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.SubspaceRandomRBFGeneratorDrift;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.ComponentBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class SubspaceRBF {

	private static final boolean DRIFT = true;
	private static final boolean SAME_SUBSPACES = true;
	private int numberSubspaceCentroids;

	private SubspaceRandomRBFGeneratorDrift stream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private final int horizon = 2000;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon = 0;
	private double aprioriThreshold;
	private double hierarchicalThreshold;
	private double connectedComponentsThreshold;
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
		for (numberSubspaceCentroids = 5; numberSubspaceCentroids <= 5; numberSubspaceCentroids += 5) {
			results.add("Number subspace centroids: " + numberSubspaceCentroids);
			for (StreamSummarisation summarisation : StreamSummarisation.values()) {
				summarisationDescription = null;
				for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
					builderDescription = null;
					if (summarisation == StreamSummarisation.ADAPTINGCENTROIDS) {
						for (numberOfDimensions = 5; numberOfDimensions <= 50; numberOfDimensions += 5) {
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
							stream.numSubspaceCentroidsOption.setValue(numberSubspaceCentroids);
							stream.sameSubspaceOption.setValue(SAME_SUBSPACES);
							stream.randomSubspaceSizeOption.setValue(false);
							if (DRIFT) {
								stream.numDriftCentroidsOption.setValue(10);
								stream.speedChangeOption.setValue(0.1);
							}
							stream.prepareForUse();

							// Creating the StreamHiCS system
							adapter = createSummarisationAdapter(summarisation);
							contrastEvaluator = new Contrast(m, alpha, adapter);
							
							double threshold = 1;
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentsThreshold;
								break;
							default:
								System.out.println("Error!");
								break;
							}
							
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

							String measures = numberOfDimensions + "," + avgTPvsFP + ", " + avgAMJS + ", " + avgAMSS
									+ ", " + avgNumElements + ", " + avgEvalTime + ", " + avgAddingTime + ", "
									+ avgTotalTime;
							System.out.println(measures);
							results.add(measures);
						}
					}
				}
			}
		}

		// Write the results
		String filePath = null;
		if (DRIFT) {
			filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/SubspaceRandomRBF/same_drift.txt";
		} else {
			filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/SubspaceRandomRBF/same_noDrift.txt";
		}

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
		case CLUSTREE_DEPTHFIRST:
			aprioriThreshold = 0.4;
			hierarchicalThreshold = 0.25;
			connectedComponentsThreshold = 0.4;
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree depth-first, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			if (DRIFT) {
				if (numberSubspaceCentroids == 5) {
					aprioriThreshold = 0.3;
					hierarchicalThreshold = 0.3;
					connectedComponentsThreshold = 0.3;
				} else if (numberSubspaceCentroids == 10) {
					aprioriThreshold = 0.2;
					hierarchicalThreshold = 0.2;
					connectedComponentsThreshold = 0.25;
				}
			} else {
				if (numberSubspaceCentroids == 5) {
					aprioriThreshold = 0.3;
					hierarchicalThreshold = 0.3;
					connectedComponentsThreshold = 0.3;
				} else if (numberSubspaceCentroids == 10) {
					aprioriThreshold = 0.25;
					hierarchicalThreshold = 0.25;
					connectedComponentsThreshold = 0.25;
				}
			}

			// double radius = 7 * Math.log(numberOfDimensions) - 0.5;
			// double radius = 8.38 * Math.log(numberOfDimensions) - 3.09;
			double radius = 4 * Math.sqrt(numberOfDimensions) - 1;
			/*
			 * if (numberSubspaceCentroids == 10) {
			 * 
			 * } else if (numberSubspaceCentroids == 5) { radius = 7 *
			 * Math.sqrt(numberOfDimensions) - 1; }
			 */
			double learningRate = 1;
			adapter = new MCAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb) {
		pruningDifference = 0.15;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			cutoff = 20;
			builder = new AprioriBuilder(numberOfDimensions, aprioriThreshold, cutoff, contrastEvaluator, null);
			builderDescription = "Apriori, threshold:" + aprioriThreshold + "cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			cutoff = 2;
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, hierarchicalThreshold, cutoff,
					contrastEvaluator, null, true);
			builderDescription = "Hierarchical, threshold: " + hierarchicalThreshold + ", cutoff: " + cutoff;
			break;
		case CONNECTED_COMPONENTS:
			pruningDifference = -1;
			builder = new ComponentBuilder(numberOfDimensions, connectedComponentsThreshold, contrastEvaluator, null);
			builderDescription = "Connected Components, threshold: " + connectedComponentsThreshold;
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
		stream.instanceRandomSeedOption.setValue(random.nextInt());
		stream.modelRandomSeedOption.setValue(random.nextInt());
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

		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		for (Subspace s : result.getSubspaces()) {
			s.sort();
		}
		result.sort();
		SubspaceSet correctResult = stream.getSubspaces();
		for (Subspace s : correctResult.getSubspaces()) {
			s.sort();
		}
		correctResult.sort();
		double[] performanceMeasures = new double[4];
		//Evaluator.displayResult(result, correctResult);
		performanceMeasures[0] = Evaluator.evaluateTPvsFP(result, correctResult);
		performanceMeasures[1] = Evaluator.evaluateJaccardIndex(result, correctResult);
		performanceMeasures[2] = Evaluator.evaluateStructuralSimilarity(result, correctResult);
		performanceMeasures[3] = streamHiCS.getNumberOfElements();

		return performanceMeasures;
	}

}
