package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
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
import fullsystem.CorrelatedSubspacesChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.clusterers.clustream.Clustream;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import streams.GaussianStream;
import streams.SubspaceRandomRBFGeneratorDrift;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;
import weka.core.Utils;

public class SubspaceRBFDriftTests {
	private SubspaceRandomRBFGeneratorDrift stream;
	private StreamHiCS streamHiCS;
	private int numInstances;
	private final int horizon = 1000;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon = 0.15;
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
	private double[][] resultSummary;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;

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
				if (summarisation == StreamSummarisation.CLUSTREE_DEPTHFIRST && buildup == SubspaceBuildup.APRIORI) {
					resultSummary = new double[2][7];
					//int from = 1;
					//int to = 1;
					//int numberTests = from - to + 1;
					for (int test = 3; test <= 3; test++) {
						stopwatch.reset();
						double threshold = 1;

						ArrayList<Double> trueChanges = new ArrayList<Double>();
						results.add("" + test);
						switch (test) {
						case 1:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.5;
								break;
							case RADIUSCENTROIDS:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.65;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							}
							numberOfDimensions = 6;
							numInstances = 20000;
							stream = new SubspaceRandomRBFGeneratorDrift();
							stream.numAttsOption.setValue(numberOfDimensions);
							stream.avgSubspaceSizeOption.setValue(numberOfDimensions / 2);
							stream.numCentroidsOption.setValue(5);
							stream.numSubspaceCentroidsOption.setValue(5);
							stream.sameSubspaceOption.setValue(true);
							stream.randomSubspaceSizeOption.setValue(false);
							stream.numDriftCentroidsOption.setValue(5);
							stream.speedChangeOption.setValue(0.1);
							stream.prepareForUse();

							
							break;
						case 2:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.25;
								break;
							case RADIUSCENTROIDS:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.3;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							}
							numberOfDimensions = 5;
							numInstances = 20000;
							break;
						}

						// Creating the StreamHiCS system
						adapter = createSummarisationAdapter(summarisation);
						contrastEvaluator = new Contrast(m, alpha, adapter);
						CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
						subspaceBuilder = createSubspaceBuilder(buildup, correlationSummary);
						ChangeChecker changeChecker = new TimeCountChecker(1000);
						streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
								subspaceBuilder, changeChecker, callback, correlationSummary, stopwatch);
						changeChecker.setCallback(streamHiCS);

						cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions, streamHiCS);
						cscd.prepareForUse();
						streamHiCS.setCallback(cscd);

						refDetector = new FullSpaceChangeDetector();
						AbstractClassifier baseLearner = new HoeffdingTree();
						baseLearner.prepareForUse();
						refDetector.baseLearnerOption.setCurrentObject(baseLearner);
						refDetector.prepareForUse();
						
						double[] cscdSums = new double[8];
						double[] refSums = new double[8];

						stopwatch.reset();
						for (int i = 0; i < numberTestRuns; i++) {
							System.out.println("Run: " +  (i + 1));
							double[][] performanceMeasures = testRun(trueChanges);
							for (int j = 0; j < 5; j++) {
								cscdSums[j] += performanceMeasures[0][j];
								refSums[j] += performanceMeasures[1][j];
							}
						}

						// Calculate results
						cscdSums[5] = stopwatch.getTime("Evaluation");
						cscdSums[6] = stopwatch.getTime("Adding");
						cscdSums[7] = stopwatch.getTime("Total_CSCD");
						refSums[7] = stopwatch.getTime("Total_REF");

						for (int j = 0; j < 8; j++) {
							cscdSums[j] /= numberTestRuns;
							refSums[j] /= numberTestRuns;
						}

						System.out.println("Test, MTFA, MTD, MDR, MTR, Accuracy, Evaluation time, Adding time, Total time");
						String cscdMeasures = "CSCD," + test + "," + cscdSums[0] + ", " + cscdSums[1] + ", "
								+ cscdSums[2] + ", " + cscdSums[3] + ", " + cscdSums[4] + ", " + cscdSums[5] + ", " + cscdSums[6] + ", " + cscdSums[7];
						String refMeasures = "REF," + test + "," + refSums[0] + ", " + refSums[1] + ", " + refSums[2]
								+ ", " + refSums[3] + ", " + refSums[4] + ", " + refSums[7];
						System.out.println(cscdMeasures);
						System.out.println(refMeasures);
						results.add(cscdMeasures);
						results.add(refMeasures);
						
						for (int i = 0; i < 7; i++) {
							resultSummary[0][i] += cscdSums[i];
							resultSummary[1][i] += refSums[i];
						}

						
					}
					/*
					for (int i = 0; i < 7; i++) {
						resultSummary[0][i] /= numberTests;
						resultSummary[1][i] /= numberTests;
					}
					String avgCSCD = "CSCD,";
					String avgREF = "REF,";
					for (int i = 0; i < 6; i++) {
						avgCSCD += resultSummary[0][i] + ",";
						avgREF += resultSummary[1][i] + ",";
					}
					avgCSCD += resultSummary[0][6];
					avgREF += resultSummary[1][6];
					System.out.println("Overall:");
					System.out.println(avgCSCD);
					System.out.print(avgREF);
					*/
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/Gaussian/Tests.txt";

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
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.breadthFirstSearchOption.set();
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			double radius = 3.5;
			double learningRate = 0.1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			radius = 1;
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

	private double[][] testRun(ArrayList<Double> trueChanges) {
		stream.restart();
		cscd.resetLearning();
		refDetector.resetLearning();

		int numberSamples = 0;

		ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		int numberCorrectCSCD = 0;
		int numberCorrectREF = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			
			// For accuracy
			int trueClass = (int) inst.classValue();
			if(trueClass == cscd.getClassPrediction(inst)){
				numberCorrectCSCD++;
			}
			if(trueClass == Utils.maxIndex(refDetector.getVotesForInstance(inst))){
				numberCorrectREF++;
			}
			
			stopwatch.start("Total_CSCD");
			cscd.trainOnInstance(inst);
			stopwatch.stop("Total_CSCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

			if (cscd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				cscdDetectedChanges.add((double) numberSamples);
				System.out.println("cscd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				refDetectedChanges.add((double) numberSamples);
				//System.out.println("refDetector: CHANGE at " + numberSamples);
			}

			numberSamples++;
		}

		double[] cscdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, cscdDetectedChanges,
				numInstances);
		double[] refPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, refDetectedChanges,
				numInstances);
		double[][] performanceMeasures = new double[2][5];
		for(int i = 0; i < 4; i++){
			performanceMeasures[0][i] = cscdPerformanceMeasures[i];
			performanceMeasures[1][i] = refPerformanceMeasures[i];
		}
		performanceMeasures[0][4] = ((double) numberCorrectCSCD) / numInstances;
		performanceMeasures[1][4] = ((double) numberCorrectREF) / numInstances;
		return performanceMeasures;
	}
}

